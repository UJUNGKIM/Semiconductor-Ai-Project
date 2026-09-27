"""Select reproducible post-selection WM-811K examples for a dashboard gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_uncertainty_policy import (
    CLASS_NAMES,
    temperature_scale,
    validate_predictions,
)


def sample_record(
    row: pd.Series,
    selection_case: str,
    confidence: float,
    review_required: bool,
) -> dict[str, object]:
    class_slug = str(row["failure_type"]).lower().replace("-", "_")
    return {
        "sample_id": f"{class_slug}_{selection_case}_{int(row['array_index'])}",
        "array_index": int(row["array_index"]),
        "source_row": int(row["source_row"]),
        "true_label_id": int(row["label_id"]),
        "true_class": str(row["failure_type"]),
        "predicted_label_id": int(row["predicted_label_id"]),
        "predicted_class": str(row["predicted_failure_type"]),
        "selection_case": selection_case,
        "correct": bool(row["predicted_label_id"] == row["label_id"]),
        "calibrated_confidence": float(confidence),
        "review_required": bool(review_required),
        "lot_name": str(row["lot_name"]),
        "wafer_index": float(row["wafer_index"]),
        "original_height": int(row["original_height"]),
        "original_width": int(row["original_width"]),
    }


# The three original cases are chosen first so a regenerated manifest keeps the
# earlier 27-sample gallery as an exact subset; the two added cases then draw
# only from rows that are still unused.
LEGACY_CASES = (
    "high_confidence_correct",
    "low_confidence_correct",
    "representative_error",
)
ADDED_CASES = ("typical_correct", "boundary_correct")
CASE_DISPLAY_ORDER = (
    "high_confidence_correct",
    "typical_correct",
    "boundary_correct",
    "low_confidence_correct",
    "representative_error",
)
SELECTION_RULES = {
    "high_confidence_correct": "클래스별 보정 신뢰도가 가장 높은 정답 1개",
    "typical_correct": "클래스 정답의 보정 신뢰도 중앙값에 가장 가까운 미선정 정답 1개",
    "boundary_correct": "Validation에서 고정한 검토 임계값에 가장 가까운 미선정 정답 1개",
    "low_confidence_correct": "클래스별 검토 대상 정답 중 보정 신뢰도가 가장 낮은 1개",
    "representative_error": "가장 흔한 오분류 방향에서 보정 신뢰도가 가장 높은 1개",
}
MISSING_CASE_REASONS = {
    "high_confidence_correct": "정답 예측이 없음",
    "typical_correct": "미선정 정답 예측이 없음",
    "boundary_correct": "미선정 정답 예측이 없음",
    "low_confidence_correct": "검토 대상 정답 예측이 없음",
    "representative_error": "오분류가 없음",
}
MIN_SAMPLES_PER_CLASS = 4
MAX_SAMPLES_PER_CLASS = 5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _closest(rows: pd.DataFrame, target: float) -> pd.Series | None:
    if rows.empty:
        return None
    return rows.assign(
        distance=(rows["calibrated_confidence"] - target).abs()
    ).sort_values(["distance", "array_index"]).iloc[0]


def select_class_cases(
    class_rows: pd.DataFrame, review_threshold: float
) -> dict[str, pd.Series]:
    """Pick one row per case for one true class without reusing a wafer."""
    correct_rows = class_rows.loc[
        class_rows["predicted_label_id"] == class_rows["label_id"]
    ]
    chosen: dict[str, pd.Series] = {}

    def unused(rows: pd.DataFrame) -> pd.DataFrame:
        used = {int(row["array_index"]) for row in chosen.values()}
        return rows.loc[~rows["array_index"].isin(used)]

    if not correct_rows.empty:
        chosen["high_confidence_correct"] = correct_rows.sort_values(
            ["calibrated_confidence", "array_index"], ascending=[False, True]
        ).iloc[0]
    review_correct = unused(correct_rows.loc[correct_rows["review_required"]])
    if not review_correct.empty:
        chosen["low_confidence_correct"] = review_correct.sort_values(
            ["calibrated_confidence", "array_index"], ascending=[True, True]
        ).iloc[0]
    errors = class_rows.loc[class_rows["predicted_label_id"] != class_rows["label_id"]]
    if not errors.empty:
        most_common_prediction = int(
            errors["predicted_label_id"].value_counts().sort_index().idxmax()
        )
        chosen["representative_error"] = errors.loc[
            errors["predicted_label_id"] == most_common_prediction
        ].sort_values(
            ["calibrated_confidence", "array_index"], ascending=[False, True]
        ).iloc[0]

    if not correct_rows.empty:
        typical = _closest(
            unused(correct_rows), float(correct_rows["calibrated_confidence"].median())
        )
        if typical is not None:
            chosen["typical_correct"] = typical
    boundary = _closest(unused(correct_rows), review_threshold)
    if boundary is not None:
        chosen["boundary_correct"] = boundary
    return chosen


def select_demo_samples(
    predictions_path: Path, policy_path: Path
) -> dict[str, object]:
    """Select four to five deterministic post-selection examples per class."""
    frame = pd.read_csv(predictions_path)
    raw_probabilities, labels = validate_predictions(frame, predictions_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("fitted_on") != "validation" or policy.get("selection_uses_test"):
        raise ValueError("불확실성 정책은 validation에서만 결정되어야 합니다.")
    if set(frame.get("split", pd.Series(["test"]))) != {"test"}:
        raise ValueError("대표 샘플은 test 예측에서만 선정합니다.")
    probabilities = temperature_scale(raw_probabilities, float(policy["temperature"]))
    confidence = probabilities.max(axis=1)
    review_threshold = float(policy["review_threshold"])
    review = confidence < review_threshold
    frame = frame.copy()
    frame["calibrated_confidence"] = confidence
    frame["review_required"] = review

    selected: list[dict[str, object]] = []
    class_counts: dict[str, int] = {}
    missing_cases: dict[str, dict[str, str]] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_rows = frame.loc[frame["label_id"] == class_id]
        chosen = select_class_cases(class_rows, review_threshold)
        for case in CASE_DISPLAY_ORDER:
            if case in chosen:
                row = chosen[case]
                selected.append(
                    sample_record(
                        row,
                        case,
                        float(row["calibrated_confidence"]),
                        bool(row["review_required"]),
                    )
                )
        class_counts[class_name] = len(chosen)
        absent = [case for case in CASE_DISPLAY_ORDER if case not in chosen]
        if absent:
            missing_cases[class_name] = {
                case: MISSING_CASE_REASONS[case] for case in absent
            }
        if not MIN_SAMPLES_PER_CLASS <= len(chosen) <= MAX_SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"{class_name} 대표 샘플이 {len(chosen)}개입니다. "
                f"실제 test 예측에서 클래스별 {MIN_SAMPLES_PER_CLASS}~"
                f"{MAX_SAMPLES_PER_CLASS}개를 선정할 수 없습니다."
            )

    if len({item["array_index"] for item in selected}) != len(selected):
        raise RuntimeError("대표 샘플 인덱스가 중복되었습니다.")
    return {
        "schema_version": 2,
        "source_split": "test",
        "purpose": "qualitative_dashboard_demo_after_model_selection",
        "used_for_model_selection": False,
        "strategy": policy["strategy"],
        "temperature": float(policy["temperature"]),
        "review_threshold": review_threshold,
        "selection_rules": dict(SELECTION_RULES),
        "legacy_cases": list(LEGACY_CASES),
        "added_cases": list(ADDED_CASES),
        "case_display_order": list(CASE_DISPLAY_ORDER),
        "samples_per_class_range": [MIN_SAMPLES_PER_CLASS, MAX_SAMPLES_PER_CLASS],
        "class_sample_counts": class_counts,
        "missing_cases": missing_cases,
        "source_predictions": {
            "file_name": predictions_path.name,
            "sha256": file_sha256(predictions_path),
            "rows": int(len(frame)),
        },
        "sample_count": len(selected),
        "samples": selected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = select_demo_samples(args.test_predictions, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "sample_count": manifest["sample_count"],
                "class_counts": pd.Series(
                    [item["true_class"] for item in manifest["samples"]]
                ).value_counts().sort_index().to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
