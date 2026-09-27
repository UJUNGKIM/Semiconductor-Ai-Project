"""Reproduce the deployed WM-811K CNN test predictions from local arrays.

The deployed model's raw test prediction table lives only in the local Colab
strategy bundle, which is not tracked. This script rebuilds it from the
preprocessed arrays and the committed checkpoint, then refuses to write
anything unless the result matches independent committed evidence: the class
report, the confusion matrix, the review-policy counts, and the per-wafer
predictions and calibrated confidences that the Colab XAI run recorded for its
421 fixed test wafers. An earlier demo-sample manifest can be passed as an
extra reference; a manifest regenerated from this script's own output is not
independent and must not be used as the reference.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from build_uncertainty_policy import temperature_scale
from diagnose_wm811k import load_checkpoint
from hash_evidence import text_hashes
from train_wm811k_cnn import validate_inputs


PROJECT_DIR = Path(__file__).resolve().parents[2]
WM_DIR = PROJECT_DIR / "결과물" / "wm811k"
SELECTED_DIR = WM_DIR / "selected_model_results"
REPORT_TOLERANCE = 1e-9
CONFIDENCE_TOLERANCE = 1e-5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict_split(
    model: torch.nn.Module, maps: np.ndarray, indices: np.ndarray, batch_size: int
) -> np.ndarray:
    """Return uncalibrated softmax probabilities, as saved by training."""
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = np.asarray(maps[indices[start : start + batch_size]], dtype=np.uint8)
            channels = np.stack((batch > 0, batch == 2), axis=1).astype(np.float32)
            logits = model(torch.from_numpy(np.ascontiguousarray(channels)))
            probabilities.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(probabilities).astype(np.float64)


def build_prediction_table(
    assignments: pd.DataFrame, probabilities: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    """Mirror train_wm811k_cnn.save_evaluation_artifacts row predictions."""
    rows = assignments.loc[assignments["split"] == "test"].sort_values("array_index").copy()
    predicted = probabilities.argmax(axis=1)
    rows["predicted_label_id"] = predicted
    rows["predicted_failure_type"] = [class_names[label] for label in predicted]
    rows["correct"] = rows["label_id"].to_numpy() == predicted
    for label_id, class_name in enumerate(class_names):
        safe_name = class_name.lower().replace("-", "_").replace(" ", "_")
        rows[f"probability_{safe_name}"] = probabilities[:, label_id]
    return rows.reset_index(drop=True)


def load_xai_reference(bundle: Path) -> pd.DataFrame:
    """Per-wafer predictions and calibrated confidences from the Colab XAI run."""
    with zipfile.ZipFile(bundle) as archive:
        records = pd.read_csv(io.BytesIO(archive.read("xai_method_records.csv")))
    return records.drop_duplicates("array_index")[
        ["array_index", "predicted_class", "baseline_confidence"]
    ].rename(
        columns={
            "predicted_class": "predicted_class_name",
            "baseline_confidence": "calibrated_confidence",
        }
    )


def compare_samples(
    reference: pd.DataFrame,
    table: pd.DataFrame,
    confidence: np.ndarray,
    class_names: list[str],
) -> dict[str, float | int]:
    position = {int(index): row for row, index in enumerate(table["array_index"])}
    predicted = table["predicted_label_id"].to_numpy(dtype=int)
    mismatches = 0
    differences = []
    for item in reference.itertuples(index=False):
        row = position[int(item.array_index)]
        mismatches += int(class_names[predicted[row]] != item.predicted_class_name)
        differences.append(abs(float(confidence[row]) - float(item.calibrated_confidence)))
    return {
        "samples_compared": len(differences),
        "prediction_mismatches": mismatches,
        "max_calibrated_confidence_difference": max(differences),
    }


def verify_against_committed(
    table: pd.DataFrame,
    probabilities: np.ndarray,
    class_names: list[str],
    policy: dict,
    committed_report: pd.DataFrame,
    committed_matrix: np.ndarray,
    xai_reference: pd.DataFrame,
    legacy_demo_manifest: dict | None,
) -> dict[str, object]:
    true = table["label_id"].to_numpy(dtype=int)
    predicted = table["predicted_label_id"].to_numpy(dtype=int)
    report = pd.DataFrame(
        classification_report(
            true,
            predicted,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_difference = float(
        (
            report.loc[committed_report.index, committed_report.columns]
            - committed_report
        ).abs().to_numpy().max()
    )
    matrix = confusion_matrix(true, predicted, labels=np.arange(len(class_names)))
    calibrated = temperature_scale(probabilities, float(policy["temperature"]))
    confidence = calibrated.max(axis=1)
    review = confidence < float(policy["review_threshold"])
    errors = predicted != true
    policy_counts = {
        "rows": int(len(true)),
        "automatic_count": int((~review).sum()),
        "review_count": int(review.sum()),
        "total_errors": int(errors.sum()),
        "captured_errors": int((errors & review).sum()),
    }
    committed_counts = {
        key: int(policy["test_evaluation"][key]) for key in policy_counts
    }
    xai = compare_samples(xai_reference, table, confidence, class_names)
    checks = {
        "classification_report_matches": report_difference <= REPORT_TOLERANCE,
        "confusion_matrix_identical": bool(np.array_equal(matrix, committed_matrix)),
        "review_policy_counts_identical": policy_counts == committed_counts,
        "xai_bundle_predictions_identical": xai["prediction_mismatches"] == 0,
        "xai_bundle_confidences_within_tolerance": xai["max_calibrated_confidence_difference"]
        <= CONFIDENCE_TOLERANCE,
    }
    result: dict[str, object] = {
        "max_classification_report_difference": report_difference,
        "review_policy_counts": policy_counts,
        "committed_review_policy_counts": committed_counts,
        "xai_bundle_reference": xai,
        "rows_within_1e-5_of_review_threshold": int(
            (np.abs(confidence - float(policy["review_threshold"])) < 1e-5).sum()
        ),
    }
    if legacy_demo_manifest is not None:
        legacy = pd.DataFrame(legacy_demo_manifest["samples"])[
            ["array_index", "predicted_class", "calibrated_confidence"]
        ].rename(columns={"predicted_class": "predicted_class_name"})
        demo = compare_samples(legacy, table, confidence, class_names)
        checks["legacy_demo_predictions_identical"] = demo["prediction_mismatches"] == 0
        checks["legacy_demo_confidences_within_tolerance"] = (
            demo["max_calibrated_confidence_difference"] <= CONFIDENCE_TOLERANCE
        )
        result["legacy_demo_reference"] = demo
    result["checks"] = checks
    result["all_checks_passed"] = all(checks.values())
    return result


def reproduce(args: argparse.Namespace) -> dict[str, object]:
    maps, labels, assignments = validate_inputs(args.maps, args.labels, args.assignments)
    model, class_names, checkpoint = load_checkpoint(args.checkpoint)
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    if policy.get("fitted_on") != "validation" or policy.get("selection_uses_test"):
        raise ValueError("불확실성 정책은 validation에서만 결정되어야 합니다.")
    if list(policy["class_names"]) != class_names:
        raise ValueError("불확실성 정책과 체크포인트 클래스 순서가 다릅니다.")
    torch.manual_seed(0)
    test_indices = (
        assignments.loc[assignments["split"] == "test", "array_index"]
        .sort_values()
        .to_numpy(dtype=np.int64)
    )
    probabilities = predict_split(model, maps, test_indices, args.batch_size)
    table = build_prediction_table(assignments, probabilities, class_names)
    legacy_manifest = (
        None
        if args.legacy_demo_manifest is None
        else json.loads(args.legacy_demo_manifest.read_text(encoding="utf-8"))
    )
    verification = verify_against_committed(
        table,
        probabilities,
        class_names,
        policy,
        pd.read_csv(args.reference_report, index_col=0),
        pd.read_csv(args.reference_confusion_matrix, index_col=0).to_numpy(),
        load_xai_reference(args.reference_xai_bundle),
        legacy_manifest,
    )
    if not verification["all_checks_passed"]:
        raise RuntimeError(
            "재현한 test 예측이 커밋된 배포 모델 증거와 다릅니다: "
            + json.dumps(verification, ensure_ascii=False)
        )

    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_predictions, index=False, lineterminator="\n")
    references = {
        "classification_report": {"file_name": args.reference_report.name, **text_hashes(args.reference_report)},
        "confusion_matrix": {"file_name": args.reference_confusion_matrix.name, **text_hashes(args.reference_confusion_matrix)},
        "xai_bundle": {"file_name": args.reference_xai_bundle.name, "sha256": file_sha256(args.reference_xai_bundle)},
    }
    if args.legacy_demo_manifest is not None:
        references["legacy_demo_manifest"] = {
            "description": args.legacy_demo_description,
            **text_hashes(args.legacy_demo_manifest),
        }
    summary = {
        "schema_version": 2,
        "purpose": "reproduce_deployed_test_predictions_for_post_selection_demo_samples",
        "used_for_model_selection": False,
        "split": "test",
        "rows": int(len(table)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_best_epoch": int(checkpoint["best_epoch"]),
        "maps_sha256": file_sha256(args.maps),
        "labels_sha256": file_sha256(args.labels),
        "maps_shape": list(maps.shape),
        "assignments": {"file_name": args.assignments.name, **text_hashes(args.assignments)},
        "policy_sha256": file_sha256(args.policy),
        "temperature": float(policy["temperature"]),
        "review_threshold": float(policy["review_threshold"]),
        "predictions_file_name": args.output_predictions.name,
        "predictions_sha256": file_sha256(args.output_predictions),
        "predictions_tracked_in_git": False,
        "inference_device": "cpu",
        "references": references,
        "verification": verification,
        "limitations": [
            "예측표는 로컬 전처리 배열에서 다시 계산했으며 Git에는 해시만 기록합니다.",
            "대표 샘플 선정 전용이며 성능 재평가나 모델·임계값 조정에 사용하지 않습니다.",
            "CPU 재계산이라 Colab GPU 기록과 보정 신뢰도가 1e-5 이하로 다를 수 있습니다.",
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--assignments",
        type=Path,
        default=WM_DIR / "split_results" / "split_assignments.csv",
    )
    parser.add_argument("--checkpoint", type=Path, default=SELECTED_DIR / "best_model.pt")
    parser.add_argument(
        "--policy", type=Path, default=SELECTED_DIR / "uncertainty_policy.json"
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        default=SELECTED_DIR / "test_classification_report.csv",
    )
    parser.add_argument(
        "--reference-confusion-matrix",
        type=Path,
        default=SELECTED_DIR / "test_confusion_matrix.csv",
    )
    parser.add_argument(
        "--reference-xai-bundle",
        type=Path,
        default=WM_DIR / "colab_가져오기" / "xai_deployed_model_results.zip",
    )
    parser.add_argument(
        "--legacy-demo-manifest",
        type=Path,
        help="이 스크립트 결과로 만든 적 없는 이전 대표 샘플 manifest(선택)",
    )
    parser.add_argument("--legacy-demo-description", default="")
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=WM_DIR / "local_reproduction" / "deployed_test_predictions.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SELECTED_DIR / "test_prediction_reproduction.json",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    summary = reproduce(parse_args())
    print(json.dumps(summary["verification"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
