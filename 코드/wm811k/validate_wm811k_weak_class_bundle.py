"""Validate and safely extract a WM-811K validation-only candidate bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score


CANDIDATES = ("baseline", "weak_defect_thinning", "weak_thinning_hard_negative")
WEAK_CLASSES = ("Scratch", "Loc", "Edge-Loc")
METRIC_COLUMNS = (
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_macro_f1",
    "validation_weighted_f1",
    "validation_weak_recall",
    "validation_none_recall",
)
MAX_FILES = 100
MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-10) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if not members or len(members) > MAX_FILES:
        raise ValueError(f"Unexpected ZIP entry count: {len(members)}")
    if sum(info.file_size for info in members) > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("ZIP uncompressed size exceeds the validation limit")
    for info in members:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ValueError(f"Unsafe ZIP path: {info.filename}")
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_IFMT(mode) == stat.S_IFLNK:
            raise ValueError(f"Symbolic links are not allowed: {info.filename}")
        lowered = [part.lower() for part in path.parts]
        if any(part.startswith("test") for part in lowered):
            raise ValueError(f"Test artifact must not be in this validation-only bundle: {info.filename}")
    return members


def _read_json(archive: zipfile.ZipFile, name: str) -> dict:
    return json.loads(archive.read(name).decode("utf-8"))


def _read_csv(archive: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archive.read(name)), **kwargs)


def _recomputed_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    true = predictions["label_id"].to_numpy(dtype=int)
    predicted = predictions["predicted_label_id"].to_numpy(dtype=int)
    weak_labels = (
        predictions.loc[predictions["failure_type"].isin(WEAK_CLASSES), ["failure_type", "label_id"]]
        .drop_duplicates()
        .set_index("failure_type")["label_id"]
    )
    if set(weak_labels.index) != set(WEAK_CLASSES):
        raise ValueError("Weak-class label mapping is incomplete")
    none_labels = predictions.loc[predictions["failure_type"].eq("none"), "label_id"].unique()
    if len(none_labels) != 1:
        raise ValueError("Exactly one none label is required")
    return {
        "validation_accuracy": accuracy_score(true, predicted),
        "validation_balanced_accuracy": balanced_accuracy_score(true, predicted),
        "validation_macro_f1": f1_score(true, predicted, average="macro"),
        "validation_weighted_f1": f1_score(true, predicted, average="weighted"),
        "validation_weak_recall": recall_score(
            true, predicted, labels=[int(weak_labels[name]) for name in WEAK_CLASSES], average="macro"
        ),
        "validation_none_recall": recall_score(
            true, predicted, labels=[int(none_labels[0])], average="macro"
        ),
    }


def _validate_predictions(
    archive: zipfile.ZipFile,
    candidate: str,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    predictions = _read_csv(archive, f"{candidate}/validation_predictions.csv")
    if predictions.empty or set(predictions["split"].astype(str)) != {"validation"}:
        raise ValueError(f"{candidate}: predictions are not validation-only")
    if predictions["array_index"].duplicated().any():
        raise ValueError(f"{candidate}: duplicate array_index values")
    expected = assignments.loc[assignments["split"].eq("validation"), [
        "array_index", "label_id", "failure_type", "split"
    ]].sort_values("array_index").reset_index(drop=True)
    actual = predictions[["array_index", "label_id", "failure_type", "split"]].sort_values(
        "array_index"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    probability_columns = [column for column in predictions if column.startswith("probability_")]
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    if len(probability_columns) != 9 or not np.isfinite(probabilities).all():
        raise ValueError(f"{candidate}: invalid probability matrix")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{candidate}: class probabilities do not sum to one")
    if not np.array_equal(predictions["predicted_label_id"].to_numpy(dtype=int), probabilities.argmax(axis=1)):
        raise ValueError(f"{candidate}: predicted labels do not match probability argmax")
    if not np.array_equal(
        predictions["correct"].astype(bool).to_numpy(),
        predictions["label_id"].to_numpy(dtype=int) == predictions["predicted_label_id"].to_numpy(dtype=int),
    ):
        raise ValueError(f"{candidate}: incorrect correctness flags")
    return predictions, _recomputed_metrics(predictions)


def validate_bundle(zip_path: Path, assignments_path: Path) -> dict[str, object]:
    assignments = pd.read_csv(assignments_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = _safe_members(archive)
        names = {info.filename for info in members if not info.is_dir()}
        required = {
            "experiment_summary.json", "validation_candidate_comparison.csv",
            "validation_candidate_dashboard.png", "train_hard_negative_indices.csv", "README.md",
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

        summary = _read_json(archive, "experiment_summary.json")
        if summary.get("purpose") != "validation_only_weak_class_candidate_comparison":
            raise ValueError("Unexpected experiment purpose")
        if summary.get("used_test_for_selection") is not False or summary.get("test_evaluated") is not False:
            raise ValueError("The bundle must keep test selection and evaluation locked")
        comparison = _read_csv(archive, "validation_candidate_comparison.csv")
        if set(comparison["candidate"]) != set(CANDIDATES) or len(comparison) != len(CANDIDATES):
            raise ValueError("Candidate inventory is invalid")
        if comparison["test_evaluated"].astype(bool).any():
            raise ValueError("A candidate reports test evaluation")

        validation_rows = int(assignments["split"].eq("validation").sum())
        metric_checks: dict[str, dict[str, float]] = {}
        checkpoint_sha256: dict[str, str] = {}
        for candidate in CANDIDATES:
            row = comparison.loc[comparison["candidate"].eq(candidate)].iloc[0]
            run_summary = _read_json(archive, f"{candidate}/run_summary.json")
            if run_summary.get("candidate") != candidate or run_summary.get("test_evaluated") is not False:
                raise ValueError(f"{candidate}: invalid run summary")
            predictions, metrics = _validate_predictions(archive, candidate, assignments)
            if len(predictions) != validation_rows:
                raise ValueError(f"{candidate}: validation row count mismatch")
            for metric in METRIC_COLUMNS:
                _assert_close(metrics[metric], row[metric], f"{candidate}/{metric}")
                _assert_close(metrics[metric], run_summary[metric], f"{candidate}/run_summary/{metric}")
            metric_checks[candidate] = metrics
            checkpoint_sha256[candidate] = _sha256_bytes(archive.read(f"{candidate}/best_model.pt"))

        baseline = comparison.loc[comparison["candidate"].eq("baseline")].iloc[0]
        macro_tolerance = float(summary["macro_f1_tolerance"])
        none_tolerance = float(summary["none_recall_tolerance"])
        expected_macro = comparison["validation_macro_f1"] >= baseline["validation_macro_f1"] - macro_tolerance
        expected_none = comparison["validation_none_recall"] >= baseline["validation_none_recall"] - none_tolerance
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

        hard_negatives = _read_csv(archive, "train_hard_negative_indices.csv")
        hard_index = set(hard_negatives["array_index"].astype(int))
        train_none = set(assignments.loc[
            assignments["split"].eq("train") & assignments["failure_type"].eq("none"), "array_index"
        ].astype(int))
        expected_count = math.ceil(len(train_none) * float(summary["hard_negative_fraction"]))
        if len(hard_index) != expected_count or not hard_index.issubset(train_none):
            raise ValueError("Hard-negative indices are not a train-none-only subset of the expected size")

        return {
            "schema_version": 1,
            "status": "validated",
            "bundle_sha256": _sha256_bytes(zip_path.read_bytes()),
            "file_count": len(names),
            "uncompressed_bytes": sum(info.file_size for info in members),
            "validation_rows_per_candidate": validation_rows,
            "test_evaluated": False,
            "used_test_for_selection": False,
            "selected_candidate": selected,
            "deployment_changed": False,
            "hard_negative_count": len(hard_index),
            "metrics": metric_checks,
            "checkpoint_sha256": checkpoint_sha256,
        }


def safe_extract(zip_path: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        _safe_members(archive)
        archive.extractall(destination)


def write_report(report: dict[str, object], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "bundle_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = report["metrics"]
    baseline = metrics["baseline"]
    lines = [
        "# WM-811K 취약 클래스 후보 결과 검증",
        "",
        "- 상태: 검증 통과",
        f"- 선택 후보: `{report['selected_candidate']}`",
        f"- 후보별 validation 행: {report['validation_rows_per_candidate']:,}개",
        "- test 평가/선택 사용: 아니요 / 아니요",
        "- 현재 배포 모델 변경: 아니요",
        f"- 번들 SHA-256: `{report['bundle_sha256']}`",
        "",
        "| 후보 | Validation macro-F1 | 취약 3종 평균 재현율 | none 재현율 | baseline 대비 취약 재현율 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "baseline": "Baseline",
        "weak_defect_thinning": "약한 결함 thinning",
        "weak_thinning_hard_negative": "Thinning + hard negative",
    }
    for candidate in CANDIDATES:
        row = metrics[candidate]
        delta = row["validation_weak_recall"] - baseline["validation_weak_recall"]
        lines.append(
            f"| {labels[candidate]} | {row['validation_macro_f1']:.4f} | "
            f"{row['validation_weak_recall']:.4f} | {row['validation_none_recall']:.4f} | {delta:+.4f} |"
        )
    lines += [
        "",
        "두 증강 후보 모두 macro-F1 안전 기준을 통과하지 못해 baseline이 유지되었습니다. "
        "이 결과는 validation 전용이며 test를 열거나 배포 체크포인트를 교체하지 않았습니다.",
    ]
    (destination / "검증결과.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_bundle(args.zip, args.assignments)
    safe_extract(args.zip, args.output_dir)
    write_report(report, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
