"""Validate a compact WM-811K SSL train-OOF calibration result bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd


EXPECTED_FILES = {
    "oof_fold_metrics.csv",
    "oof_bias_sweep.csv",
    "calibration_summary.json",
}
EXPECTED_FOLDS = (0, 1, 2)
TARGET_NONE_RECALL = 0.995
WEAK_RECALL_TOLERANCE = 0.01
EXPECTED_BIASES = np.round(np.arange(0.0, 0.500001, 0.025), 6)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "weak_recall",
    "none_recall",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> bool:
    member = PurePosixPath(name.replace("\\", "/"))
    return not member.is_absolute() and ".." not in member.parts


def _read_csv(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archive.read(name)))


def _close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-11):
        raise ValueError(f"{label}: expected {expected}, found {actual}")


def _validate_folds(folds: pd.DataFrame, train_rows: int, train_lots: int) -> None:
    required = {
        "fold",
        "fit_rows",
        "holdout_rows",
        "fit_lots",
        "holdout_lots",
        "fixed_epochs",
        "final_train_loss",
        "holdout_accuracy",
        "holdout_balanced_accuracy",
        "holdout_macro_f1",
    }
    if set(folds.columns) != required or len(folds) != len(EXPECTED_FOLDS):
        raise ValueError("OOF fold table shape or columns are invalid")
    folds = folds.sort_values("fold").reset_index(drop=True)
    if tuple(folds["fold"].astype(int)) != EXPECTED_FOLDS:
        raise ValueError("OOF fold identifiers are invalid")
    if not (folds["fixed_epochs"].astype(int) == 10).all():
        raise ValueError("OOF training did not use the frozen 10 epochs")
    if int(folds["holdout_rows"].sum()) != train_rows:
        raise ValueError("OOF holdout rows do not cover the frozen train split once")
    if int(folds["holdout_lots"].sum()) != train_lots:
        raise ValueError("OOF holdout lots do not cover the frozen train lots once")
    if not (
        folds["fit_rows"].astype(int) + folds["holdout_rows"].astype(int)
        == train_rows
    ).all():
        raise ValueError("OOF fit/holdout row totals are inconsistent")
    if not (
        folds["fit_lots"].astype(int) + folds["holdout_lots"].astype(int)
        == train_lots
    ).all():
        raise ValueError("OOF fit/holdout lot totals are inconsistent")
    finite_columns = [
        "final_train_loss",
        "holdout_accuracy",
        "holdout_balanced_accuracy",
        "holdout_macro_f1",
    ]
    if not np.isfinite(folds[finite_columns].to_numpy(dtype=float)).all():
        raise ValueError("OOF fold metrics contain non-finite values")
    for column in finite_columns[1:]:
        if not folds[column].between(0.0, 1.0).all():
            raise ValueError(f"OOF {column} is outside [0, 1]")
    if (folds["final_train_loss"] < 0).any():
        raise ValueError("OOF training loss is negative")


def _validate_sweep(sweep: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    expected_columns = {
        "none_logit_bias",
        *METRICS,
        "delta_macro_f1",
        "delta_balanced_accuracy",
        "delta_weak_recall",
        "delta_none_recall",
        "eligible",
    }
    if set(sweep.columns) != expected_columns or len(sweep) != len(EXPECTED_BIASES):
        raise ValueError("Bias sweep shape or columns are invalid")
    sweep = sweep.sort_values("none_logit_bias").reset_index(drop=True)
    if not np.allclose(
        sweep["none_logit_bias"].to_numpy(float), EXPECTED_BIASES, atol=1e-12
    ):
        raise ValueError("Bias sweep grid differs from the frozen protocol")
    if not np.isfinite(sweep[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Bias sweep contains non-finite metrics")
    if not ((sweep[list(METRICS)] >= 0) & (sweep[list(METRICS)] <= 1)).all().all():
        raise ValueError("Bias sweep metrics are outside [0, 1]")

    baseline = sweep.iloc[0]
    delta_sources = {
        "delta_macro_f1": "macro_f1",
        "delta_balanced_accuracy": "balanced_accuracy",
        "delta_weak_recall": "weak_recall",
        "delta_none_recall": "none_recall",
    }
    for delta_column, metric in delta_sources.items():
        expected = sweep[metric].to_numpy(float) - float(baseline[metric])
        if not np.allclose(sweep[delta_column].to_numpy(float), expected, atol=1e-11):
            raise ValueError(f"{delta_column} does not match recomputation")

    recomputed_eligible = (
        sweep["none_recall"] >= TARGET_NONE_RECALL
    ) & (
        sweep["weak_recall"]
        >= float(baseline["weak_recall"]) - WEAK_RECALL_TOLERANCE
    )
    declared = sweep["eligible"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if declared.isna().any() or not np.array_equal(
        declared.to_numpy(bool), recomputed_eligible.to_numpy(bool)
    ):
        raise ValueError("Declared bias eligibility differs from recomputation")
    max_none = sweep.sort_values(
        ["none_recall", "weak_recall", "macro_f1"], ascending=False
    ).iloc[0]
    best_macro = sweep.sort_values(
        ["macro_f1", "none_recall"], ascending=False
    ).iloc[0]
    return baseline, max_none, best_macro


def validate_bundle(zip_path: Path, assignments_path: Path) -> dict[str, object]:
    assignments = pd.read_csv(assignments_path)
    required_assignments = {"array_index", "label_id", "lot_name", "split"}
    if required_assignments.difference(assignments.columns):
        raise ValueError("Frozen split assignments are incomplete")
    train = assignments.loc[assignments["split"] == "train"]
    train_rows = len(train)
    train_lots = train["lot_name"].astype(str).nunique()

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not all(_safe_name(name) for name in names):
            raise ValueError("ZIP contains duplicate or unsafe paths")
        if set(names) != EXPECTED_FILES:
            raise ValueError(
                f"Result bundle mismatch: missing={sorted(EXPECTED_FILES-set(names))}, "
                f"extra={sorted(set(names)-EXPECTED_FILES)}"
            )
        summary = json.loads(
            archive.read("calibration_summary.json").decode("utf-8")
        )
        folds = _read_csv(archive, "oof_fold_metrics.csv")
        sweep = _read_csv(archive, "oof_bias_sweep.csv")

    _validate_folds(folds, train_rows, train_lots)
    baseline, max_none, best_macro = _validate_sweep(sweep)
    expected_summary = {
        "schema_version": 1,
        "experiment_type": "wm811k_ssl_train_oof_none_bias",
        "selection_source": "train_lot_oof_only",
        "selected_none_logit_bias": None,
        "oof_target_met": False,
        "test_evaluated": False,
        "test_used_for_selection": False,
        "validation_evaluated": False,
        "validation_used_for_bias_selection": False,
        "deployment_model_changed": False,
        "decision": "retain_existing_model",
    }
    if summary != expected_summary:
        raise ValueError("Calibration summary does not match the failed-safe contract")
    if sweep["eligible"].astype(bool).any():
        raise ValueError("Summary declares failure but an eligible bias exists")

    fold_metrics = []
    for row in folds.sort_values("fold").to_dict(orient="records"):
        fold_metrics.append(
            {
                key: int(value) if key in {
                    "fold", "fit_rows", "holdout_rows", "fit_lots",
                    "holdout_lots", "fixed_epochs"
                } else float(value)
                for key, value in row.items()
            }
        )
    def metric_row(row: pd.Series) -> dict[str, float]:
        return {
            key: float(row[key])
            for key in (
                "none_logit_bias", *METRICS, "delta_macro_f1",
                "delta_balanced_accuracy", "delta_weak_recall",
                "delta_none_recall",
            )
        }

    return {
        "schema_version": 1,
        "status": "validated",
        "bundle_sha256": file_sha256(zip_path),
        "bundle_file_count": len(EXPECTED_FILES),
        "split_assignments_sha256": file_sha256(assignments_path),
        "selection_source": "train_lot_oof_only",
        "oof_folds": len(EXPECTED_FOLDS),
        "oof_rows_checked": train_rows,
        "oof_lots_checked": train_lots,
        "fixed_epochs_per_fold": 10,
        "target_none_recall": TARGET_NONE_RECALL,
        "weak_recall_tolerance": WEAK_RECALL_TOLERANCE,
        "eligible_bias_count": 0,
        "selected_none_logit_bias": None,
        "baseline_oof": metric_row(baseline),
        "maximum_none_recall_candidate": metric_row(max_none),
        "best_macro_f1_candidate": metric_row(best_macro),
        "fold_metrics": fold_metrics,
        "oof_target_met": False,
        "validation_evaluated": False,
        "test_evaluated": False,
        "test_used_for_selection": False,
        "deployment_model_changed": False,
        "decision": "retain_existing_model",
        "conclusion": "no_train_oof_bias_met_predeclared_safety_targets",
    }


def write_report(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bundle_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline = report["baseline_oof"]
    maximum = report["maximum_none_recall_candidate"]
    lines = [
        "# WM-811K SSL train-OOF 보정 검증",
        "",
        "- 상태: 업로드 번들 검증 통과",
        f"- train OOF: {report['oof_rows_checked']:,}행 · {report['oof_lots_checked']:,} lot · 3-fold",
        "- bias 선택에 validation 사용: 아니요",
        "- 고정 test 사용: 아니요",
        "- 운영 모델 변경: 아니요",
        "- 결정: 기존 champion 유지",
        f"- 번들 SHA-256: {report['bundle_sha256']}",
        "",
        "| 지표 | bias 0.000 | none recall 최대 후보 | 변화 |",
        "|---|---:|---:|---:|",
        f"| none logit bias | 0.000 | {maximum['none_logit_bias']:.3f} | - |",
        f"| none recall | {baseline['none_recall']:.4f} | {maximum['none_recall']:.4f} | {maximum['delta_none_recall']:+.4f} |",
        f"| 약한 결함 recall | {baseline['weak_recall']:.4f} | {maximum['weak_recall']:.4f} | {maximum['delta_weak_recall']:+.4f} |",
        f"| Macro-F1 | {baseline['macro_f1']:.4f} | {maximum['macro_f1']:.4f} | {maximum['delta_macro_f1']:+.4f} |",
        "",
        "## 판정",
        "",
        (
            f"가장 높은 none recall은 {maximum['none_recall']:.4f}로 사전 목표 "
            f"{report['target_none_recall']:.4f}에 미달했습니다. 동시에 약한 결함 "
            f"recall은 {abs(maximum['delta_weak_recall']):.4f} 감소해 허용치 "
            f"{report['weak_recall_tolerance']:.4f}도 넘었습니다."
        ),
        "",
        (
            "따라서 기준을 사후 변경하지 않고 보정 후보를 기각했습니다. "
            "validation과 고정 test는 열지 않았으며 기존 champion을 유지합니다."
        ),
    ]
    (output_dir / "검증결과.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_bundle(args.bundle, args.assignments)
    write_report(report, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
