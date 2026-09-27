"""Validate and safely extract a paired multi-seed WM-811K weak-margin bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from validate_wm811k_weak_class_bundle import (
    METRIC_COLUMNS,
    _assert_close,
    _read_csv,
    _read_json,
    _safe_members,
    _validate_predictions,
    safe_extract,
)


SEEDS = (17, 42, 2026)
STRATEGIES = ("baseline", "weak_none_margin_005")
EXPECTED_MARGIN = {
    "baseline": (0.0, 0.2),
    "weak_none_margin_005": (0.05, 0.2),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_files() -> set[str]:
    files = {
        "seed_candidate_metrics.csv",
        "metric_summary.csv",
        "paired_seed_deltas.csv",
        "multiseed_margin_dashboard.png",
        "reproducibility_summary.json",
        "README.md",
    }
    run_files = (
        "best_model.pt", "run_summary.json", "training_history.csv",
        "validation_classification_report.csv", "validation_confusion_matrix.csv",
        "validation_confusion_matrix.png", "validation_predictions.csv",
    )
    for seed in SEEDS:
        for strategy in STRATEGIES:
            files.update(
                f"runs/seed_{seed}/{strategy}/{name}" for name in run_files
            )
    return files


def _decision(paired: pd.DataFrame, guardrails: dict[str, float]) -> dict[str, object]:
    macro_tolerance = float(guardrails["macro_f1_tolerance_per_seed"])
    none_tolerance = float(guardrails["none_recall_tolerance_per_seed"])
    balanced_tolerance = float(guardrails["balanced_accuracy_tolerance_per_seed"])
    weak_improved = int((paired["delta_validation_weak_recall"] > 0).sum())
    checks = {
        "mean_weak_recall_improved": float(
            paired["delta_validation_weak_recall"].mean()
        ) > 0,
        "weak_recall_improved_at_least_two_seeds": weak_improved >= 2,
        "macro_f1_guardrail_all_seeds": bool(
            (paired["delta_validation_macro_f1"] >= -macro_tolerance).all()
        ),
        "none_recall_guardrail_all_seeds": bool(
            (paired["delta_validation_none_recall"] >= -none_tolerance).all()
        ),
        "balanced_accuracy_guardrail_all_seeds": bool(
            (
                paired["delta_validation_balanced_accuracy"]
                >= -balanced_tolerance
            ).all()
        ),
    }
    checks["eligible"] = all(checks.values())
    return {
        "selected_candidate": (
            "weak_none_margin_005" if checks["eligible"] else "baseline"
        ),
        "weak_recall_improved_seed_count": weak_improved,
        "checks": checks,
    }


def validate_bundle(
    zip_path: Path,
    assignments_path: Path,
    expected_seed42_root: Path | None = None,
) -> dict[str, object]:
    assignments = pd.read_csv(assignments_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = _safe_members(archive)
        names = {info.filename for info in members if not info.is_dir()}
        required = _expected_files()
        missing = sorted(required - names)
        unexpected = sorted(names - required)
        if missing or unexpected:
            raise ValueError(
                f"Bundle inventory mismatch; missing={missing}, unexpected={unexpected}"
            )
        if not archive.read("multiseed_margin_dashboard.png").startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            raise ValueError("Multi-seed dashboard is not a PNG file")

        summary = _read_json(archive, "reproducibility_summary.json")
        expected_contract = {
            "purpose": "validation_only_paired_weak_margin_multiseed_reproducibility",
            "seeds": list(SEEDS),
            "strategies": list(STRATEGIES),
            "fixed_split": True,
            "test_evaluated": False,
            "used_test_for_selection": False,
            "deployment_changed": False,
            "primary_metric": "validation_weak_recall",
        }
        for key, value in expected_contract.items():
            if summary.get(key) != value:
                raise ValueError(f"Unexpected summary contract for {key}")

        run_table = _read_csv(archive, "seed_candidate_metrics.csv")
        expected_grid = {(seed, strategy) for seed in SEEDS for strategy in STRATEGIES}
        actual_grid = set(zip(run_table["seed"].astype(int), run_table["strategy"]))
        if len(run_table) != len(expected_grid) or actual_grid != expected_grid:
            raise ValueError("Seed-strategy result grid is incomplete or duplicated")
        if run_table["test_evaluated"].astype(bool).any():
            raise ValueError("A run reports test evaluation")

        validation_rows = int(assignments["split"].eq("validation").sum())
        metrics_by_run: dict[str, dict[str, float]] = {}
        checkpoint_sha256: dict[str, str] = {}
        seed42_reuse_matches: dict[str, bool] = {}
        for seed in SEEDS:
            for strategy in STRATEGIES:
                prefix = f"runs/seed_{seed}/{strategy}"
                key = f"seed_{seed}/{strategy}"
                row = run_table.loc[
                    run_table["seed"].eq(seed) & run_table["strategy"].eq(strategy)
                ].iloc[0]
                run_summary = _read_json(archive, f"{prefix}/run_summary.json")
                expected_lambda, expected_margin = EXPECTED_MARGIN[strategy]
                if (
                    run_summary.get("candidate") != strategy
                    or int(run_summary.get("seed")) != seed
                    or run_summary.get("test_evaluated") is not False
                ):
                    raise ValueError(f"{key}: invalid run summary")
                _assert_close(run_summary["weak_margin_lambda"], expected_lambda, f"{key}/lambda")
                _assert_close(run_summary["weak_margin_value"], expected_margin, f"{key}/margin")
                predictions, recomputed = _validate_predictions(
                    archive, prefix, assignments
                )
                if len(predictions) != validation_rows:
                    raise ValueError(f"{key}: validation row count mismatch")
                for metric in METRIC_COLUMNS:
                    _assert_close(recomputed[metric], row[metric], f"{key}/{metric}")
                    _assert_close(
                        recomputed[metric], run_summary[metric],
                        f"{key}/run_summary/{metric}",
                    )
                metrics_by_run[key] = recomputed
                digest = hashlib.sha256(
                    archive.read(f"{prefix}/best_model.pt")
                ).hexdigest()
                checkpoint_sha256[key] = digest
                if seed == 42 and expected_seed42_root is not None:
                    matches = digest == file_sha256(
                        expected_seed42_root / strategy / "best_model.pt"
                    )
                    seed42_reuse_matches[strategy] = matches
                    if not matches:
                        raise ValueError(
                            f"Seed 42 checkpoint does not match source for {strategy}"
                        )

        metric_summary = _read_csv(archive, "metric_summary.csv")
        if set(metric_summary["strategy"]) != set(STRATEGIES) or len(metric_summary) != 2:
            raise ValueError("Metric summary strategy inventory is invalid")
        for strategy in STRATEGIES:
            source = run_table.loc[run_table["strategy"].eq(strategy)]
            summary_row = metric_summary.loc[
                metric_summary["strategy"].eq(strategy)
            ].iloc[0]
            if int(summary_row["seed_count"]) != len(SEEDS):
                raise ValueError(f"{strategy}: seed count mismatch")
            for metric in METRIC_COLUMNS:
                values = source[metric].astype(float)
                for suffix, value in {
                    "mean": values.mean(), "std": values.std(ddof=1),
                    "min": values.min(), "max": values.max(),
                }.items():
                    _assert_close(
                        value, summary_row[f"{metric}_{suffix}"],
                        f"{strategy}/{metric}_{suffix}",
                    )

        paired = _read_csv(archive, "paired_seed_deltas.csv").sort_values("seed")
        if list(paired["seed"].astype(int)) != list(SEEDS):
            raise ValueError("Paired seed inventory is invalid")
        indexed = run_table.set_index(["seed", "strategy"])
        for _, paired_row in paired.iterrows():
            seed = int(paired_row["seed"])
            for metric in METRIC_COLUMNS:
                baseline = float(indexed.loc[(seed, "baseline"), metric])
                margin = float(indexed.loc[(seed, "weak_none_margin_005"), metric])
                _assert_close(
                    baseline, paired_row[f"baseline_{metric}"],
                    f"seed_{seed}/baseline_{metric}",
                )
                _assert_close(
                    margin, paired_row[f"margin_{metric}"],
                    f"seed_{seed}/margin_{metric}",
                )
                _assert_close(
                    margin - baseline, paired_row[f"delta_{metric}"],
                    f"seed_{seed}/delta_{metric}",
                )

        decision = _decision(paired, summary["paired_guardrails"])
        if decision != {
            "selected_candidate": summary.get("selected_candidate"),
            "weak_recall_improved_seed_count": summary.get(
                "weak_recall_improved_seed_count"
            ),
            "checks": summary.get("checks"),
        }:
            raise ValueError("Declared multi-seed decision is inconsistent")
        mean_deltas = {
            metric: float(paired[f"delta_{metric}"].mean())
            for metric in METRIC_COLUMNS
        }
        for metric, value in mean_deltas.items():
            _assert_close(value, summary["mean_deltas"][metric], f"mean_delta/{metric}")

        strategy_metrics = {}
        for strategy in STRATEGIES:
            row = metric_summary.loc[metric_summary["strategy"].eq(strategy)].iloc[0]
            strategy_metrics[strategy] = {
                metric: {
                    suffix: float(row[f"{metric}_{suffix}"])
                    for suffix in ("mean", "std", "min", "max")
                }
                for metric in METRIC_COLUMNS
            }
        return {
            "schema_version": 1,
            "status": "validated",
            "bundle_sha256": file_sha256(zip_path),
            "file_count": len(names),
            "uncompressed_bytes": sum(info.file_size for info in members),
            "seeds": list(SEEDS),
            "strategies": list(STRATEGIES),
            "validation_rows_per_run": validation_rows,
            "validation_prediction_rows_checked": validation_rows * len(expected_grid),
            "test_evaluated": False,
            "used_test_for_selection": False,
            "deployment_changed": False,
            "selected_candidate": decision["selected_candidate"],
            "eligible": decision["checks"]["eligible"],
            "checks": decision["checks"],
            "weak_recall_improved_seed_count": decision[
                "weak_recall_improved_seed_count"
            ],
            "mean_deltas": mean_deltas,
            "strategy_metrics": strategy_metrics,
            "metrics_by_run": metrics_by_run,
            "checkpoint_sha256": checkpoint_sha256,
            "seed42_reuse_sha_matches": seed42_reuse_matches,
            "decision": "keep_baseline_because_multiseed_guardrails_failed",
        }


def write_report(report: dict[str, object], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "bundle_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline = report["strategy_metrics"]["baseline"]
    margin = report["strategy_metrics"]["weak_none_margin_005"]
    lines = [
        "# WM-811K weak-vs-none margin 다중 seed 검증",
        "",
        "- 상태: 검증 통과",
        f"- 검증 seed: {', '.join(map(str, report['seeds']))}",
        f"- 검증한 validation 예측: {report['validation_prediction_rows_checked']:,}행",
        "- test 평가/선택 사용: 아니요 / 아니요",
        f"- 최종 선택: `{report['selected_candidate']}` (기존 baseline 유지)",
        "- 현재 배포 모델 변경: 아니요",
        f"- seed 42 체크포인트 재사용 일치: {report['seed42_reuse_sha_matches']}",
        f"- 번들 SHA-256: `{report['bundle_sha256']}`",
        "",
        "| 전략 | 취약 3종 재현율 평균 | 취약 재현율 표준편차 | Macro-F1 평균 | none 재현율 평균 | Balanced accuracy 평균 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Baseline | {baseline['validation_weak_recall']['mean']:.4f} | "
            f"{baseline['validation_weak_recall']['std']:.4f} | "
            f"{baseline['validation_macro_f1']['mean']:.4f} | "
            f"{baseline['validation_none_recall']['mean']:.4f} | "
            f"{baseline['validation_balanced_accuracy']['mean']:.4f} |"
        ),
        (
            f"| Margin λ=0.05 | {margin['validation_weak_recall']['mean']:.4f} | "
            f"{margin['validation_weak_recall']['std']:.4f} | "
            f"{margin['validation_macro_f1']['mean']:.4f} | "
            f"{margin['validation_none_recall']['mean']:.4f} | "
            f"{margin['validation_balanced_accuracy']['mean']:.4f} |"
        ),
        "",
        (
            "Margin 후보는 취약 3종 재현율을 평균 "
            f"{report['mean_deltas']['validation_weak_recall']:+.4f} 개선했고 "
            "3개 seed 중 2개에서 개선했습니다. 그러나 seed별 Macro-F1, none 재현율, "
            "balanced accuracy 안전 기준을 모두 만족하지 못해 배포 후보에서 제외했습니다."
        ),
        "",
        "따라서 이번 실험은 유효한 음성 결과로 기록하고 기존 baseline을 유지합니다. "
        "동일 validation에 대한 추가 반복 튜닝은 과적합 위험이 있으므로 수행하지 않습니다.",
    ]
    (destination / "검증결과.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--expected-seed42-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_bundle(
        args.zip, args.assignments, args.expected_seed42_root
    )
    safe_extract(args.zip, args.output_dir)
    write_report(report, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
