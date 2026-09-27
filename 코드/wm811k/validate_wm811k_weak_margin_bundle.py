"""Validate and safely extract a validation-only WM-811K weak-margin bundle."""

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
    _sha256_bytes,
    _validate_predictions,
    safe_extract,
)


CANDIDATES = ("baseline", "weak_none_margin_005", "weak_none_margin_010")
EXPECTED_MARGIN = {
    "baseline": (0.0, 0.2),
    "weak_none_margin_005": (0.05, 0.2),
    "weak_none_margin_010": (0.10, 0.2),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(
    zip_path: Path,
    assignments_path: Path,
    expected_baseline_path: Path | None = None,
) -> dict[str, object]:
    assignments = pd.read_csv(assignments_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = _safe_members(archive)
        names = {info.filename for info in members if not info.is_dir()}
        required = {
            "experiment_summary.json", "validation_margin_candidate_comparison.csv",
            "validation_margin_dashboard.png", "README.md",
        }
        for candidate in CANDIDATES:
            required.update({
                f"{candidate}/best_model.pt", f"{candidate}/run_summary.json",
                f"{candidate}/training_history.csv", f"{candidate}/validation_predictions.csv",
                f"{candidate}/validation_classification_report.csv",
                f"{candidate}/validation_confusion_matrix.csv",
                f"{candidate}/validation_confusion_matrix.png",
            })
        missing = sorted(required - names)
        unexpected = sorted(names - required)
        if missing or unexpected:
            raise ValueError(f"Bundle inventory mismatch; missing={missing}, unexpected={unexpected}")
        if not archive.read("validation_margin_dashboard.png").startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Candidate dashboard is not a PNG file")

        summary = _read_json(archive, "experiment_summary.json")
        if summary.get("purpose") != "validation_only_weak_none_margin_candidate_comparison":
            raise ValueError("Unexpected experiment purpose")
        if summary.get("used_test_for_selection") is not False or summary.get("test_evaluated") is not False:
            raise ValueError("Test selection and evaluation must remain locked")
        comparison = _read_csv(archive, "validation_margin_candidate_comparison.csv")
        if set(comparison["candidate"]) != set(CANDIDATES) or len(comparison) != len(CANDIDATES):
            raise ValueError("Candidate inventory is invalid")
        if comparison["test_evaluated"].astype(bool).any():
            raise ValueError("A candidate reports test evaluation")
        summary_candidates = pd.DataFrame(summary.get("candidates", []))
        if set(summary_candidates.get("candidate", [])) != set(CANDIDATES):
            raise ValueError("Summary candidate inventory is invalid")

        validation_rows = int(assignments["split"].eq("validation").sum())
        metrics_by_candidate: dict[str, dict[str, float]] = {}
        checkpoint_sha256: dict[str, str] = {}
        for candidate in CANDIDATES:
            row = comparison.loc[comparison["candidate"].eq(candidate)].iloc[0]
            summary_row = summary_candidates.loc[summary_candidates["candidate"].eq(candidate)].iloc[0]
            run_summary = _read_json(archive, f"{candidate}/run_summary.json")
            expected_lambda, expected_margin = EXPECTED_MARGIN[candidate]
            if run_summary.get("candidate") != candidate or run_summary.get("test_evaluated") is not False:
                raise ValueError(f"{candidate}: invalid run summary")
            _assert_close(run_summary["weak_margin_lambda"], expected_lambda, f"{candidate}/lambda")
            _assert_close(run_summary["weak_margin_value"], expected_margin, f"{candidate}/margin")
            predictions, recomputed = _validate_predictions(archive, candidate, assignments)
            if len(predictions) != validation_rows:
                raise ValueError(f"{candidate}: validation row count mismatch")
            for metric in METRIC_COLUMNS:
                _assert_close(recomputed[metric], row[metric], f"{candidate}/{metric}")
                _assert_close(recomputed[metric], run_summary[metric], f"{candidate}/run_summary/{metric}")
                _assert_close(recomputed[metric], summary_row[metric], f"{candidate}/experiment_summary/{metric}")
            metrics_by_candidate[candidate] = recomputed
            checkpoint_sha256[candidate] = _sha256_bytes(archive.read(f"{candidate}/best_model.pt"))

        baseline = comparison.loc[comparison["candidate"].eq("baseline")].iloc[0]
        expected_macro = comparison["validation_macro_f1"] >= baseline["validation_macro_f1"] - float(summary["macro_f1_tolerance"])
        expected_none = comparison["validation_none_recall"] >= baseline["validation_none_recall"] - float(summary["none_recall_tolerance"])
        expected_eligible = expected_macro & expected_none
        if not np.array_equal(comparison["macro_f1_guardrail"].astype(bool), expected_macro):
            raise ValueError("macro-F1 guardrail flags are inconsistent")
        if not np.array_equal(comparison["none_recall_guardrail"].astype(bool), expected_none):
            raise ValueError("none recall guardrail flags are inconsistent")
        if not np.array_equal(comparison["eligible"].astype(bool), expected_eligible):
            raise ValueError("eligibility flags are inconsistent")
        selected = comparison.loc[expected_eligible].sort_values(
            ["validation_weak_recall", "validation_macro_f1", "candidate"],
            ascending=[False, False, True],
        ).iloc[0]["candidate"]
        if selected != summary.get("selected_candidate"):
            raise ValueError("Selected candidate does not match the declared selection rule")

        baseline_reuse_matches = None
        if expected_baseline_path is not None:
            baseline_reuse_matches = (
                checkpoint_sha256["baseline"] == file_sha256(expected_baseline_path)
            )
            if not baseline_reuse_matches:
                raise ValueError("Reused baseline checkpoint does not match the validated source")
        baseline_metrics = metrics_by_candidate["baseline"]
        selected_metrics = metrics_by_candidate[str(selected)]
        return {
            "schema_version": 1,
            "status": "validated",
            "bundle_sha256": file_sha256(zip_path),
            "file_count": len(names),
            "uncompressed_bytes": sum(info.file_size for info in members),
            "validation_rows_per_candidate": validation_rows,
            "test_evaluated": False,
            "used_test_for_selection": False,
            "selected_candidate": str(selected),
            "selected_margin_lambda": float(
                comparison.loc[comparison["candidate"].eq(selected), "weak_margin_lambda"].iloc[0]
            ),
            "deployment_changed": False,
            "baseline_reuse_sha_matches": baseline_reuse_matches,
            "metrics": metrics_by_candidate,
            "selected_deltas_vs_baseline": {
                metric: selected_metrics[metric] - baseline_metrics[metric]
                for metric in METRIC_COLUMNS
            },
            "checkpoint_sha256": checkpoint_sha256,
            "decision": "replicate_selected_candidate_across_multiple_seeds_before_deployment",
        }


def write_report(report: dict[str, object], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "bundle_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = report["metrics"]
    baseline = metrics["baseline"]
    labels = {
        "baseline": "Baseline",
        "weak_none_margin_005": "Margin λ=0.05",
        "weak_none_margin_010": "Margin λ=0.10",
    }
    lines = [
        "# WM-811K weak-vs-none margin 결과 검증",
        "",
        "- 상태: 검증 통과",
        f"- 선택 후보: `{report['selected_candidate']}`",
        f"- 후보별 validation 행: {report['validation_rows_per_candidate']:,}개",
        "- test 평가/선택 사용: 아니요 / 아니요",
        "- 현재 배포 모델 변경: 아니요",
        f"- 기존 baseline 체크포인트 일치: {'예' if report['baseline_reuse_sha_matches'] else '아니요'}",
        f"- 번들 SHA-256: `{report['bundle_sha256']}`",
        "",
        "| 후보 | Validation macro-F1 | 취약 3종 평균 재현율 | none 재현율 | baseline 대비 취약 재현율 |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        row = metrics[candidate]
        lines.append(
            f"| {labels[candidate]} | {row['validation_macro_f1']:.4f} | "
            f"{row['validation_weak_recall']:.4f} | {row['validation_none_recall']:.4f} | "
            f"{row['validation_weak_recall'] - baseline['validation_weak_recall']:+.4f} |"
        )
    lines += [
        "",
        "λ=0.05 후보가 세 안전 기준을 통과하면서 가장 높은 취약 클래스 평균 재현율을 기록했습니다. "
        "현재 결과는 seed 42 validation 전용이므로 모델을 배포하지 않았습니다. 다음 단계는 선택 후보의 다중 seed 재현입니다.",
    ]
    (destination / "검증결과.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--expected-baseline", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_bundle(args.zip, args.assignments, args.expected_baseline)
    safe_extract(args.zip, args.output_dir)
    write_report(report, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
