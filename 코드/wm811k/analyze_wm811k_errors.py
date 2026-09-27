"""Create post-selection WM-811K class and confusion diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_uncertainty_policy import (
    CLASS_NAMES,
    PROBABILITY_COLUMNS,
    temperature_scale,
    validate_predictions,
)


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def analyze_errors(predictions_path: Path, policy_path: Path) -> dict[str, object]:
    """Analyze a frozen selected model on test predictions only."""
    frame = pd.read_csv(predictions_path)
    raw_probabilities, labels = validate_predictions(frame, predictions_path)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("fitted_on") != "validation" or policy.get("selection_uses_test"):
        raise ValueError("불확실성 정책은 validation에서만 결정되어야 합니다.")
    if policy.get("class_names") != list(CLASS_NAMES):
        raise ValueError("불확실성 정책의 클래스 순서가 예측 결과와 다릅니다.")

    probabilities = temperature_scale(raw_probabilities, float(policy["temperature"]))
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    review = confidence < float(policy["review_threshold"])

    class_analysis: list[dict[str, object]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        actual_mask = labels == class_id
        predicted_mask = predicted == class_id
        true_positive = int((actual_mask & predicted_mask).sum())
        support = int(actual_mask.sum())
        predicted_count = int(predicted_mask.sum())
        class_errors = actual_mask & ~correct
        error_count = int(class_errors.sum())
        reviewed_errors = int((class_errors & review).sum())
        automatic_mask = actual_mask & ~review
        automatic_count = int(automatic_mask.sum())
        class_analysis.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support": support,
                "predicted_count": predicted_count,
                "correct_count": true_positive,
                "error_count": error_count,
                "precision": safe_ratio(true_positive, predicted_count),
                "recall": safe_ratio(true_positive, support),
                "error_rate": safe_ratio(error_count, support),
                "mean_confidence": (
                    float(confidence[actual_mask].mean()) if support else 0.0
                ),
                "review_count": int((actual_mask & review).sum()),
                "review_rate": (
                    float(review[actual_mask].mean()) if support else 0.0
                ),
                "automatic_count": automatic_count,
                "automatic_accuracy": (
                    float(correct[automatic_mask].mean()) if automatic_count else 0.0
                ),
                "reviewed_errors": reviewed_errors,
                "error_capture_rate": safe_ratio(reviewed_errors, error_count),
            }
        )

    error_frame = pd.DataFrame(
        {
            "true_class": np.asarray(CLASS_NAMES, dtype=object)[labels[~correct]],
            "predicted_class": np.asarray(CLASS_NAMES, dtype=object)[predicted[~correct]],
            "confidence": confidence[~correct],
            "review_required": review[~correct],
        }
    )
    pairs = (
        error_frame.groupby(["true_class", "predicted_class"], sort=False)
        .agg(
            count=("confidence", "size"),
            mean_confidence=("confidence", "mean"),
            reviewed_errors=("review_required", "sum"),
        )
        .reset_index()
    )
    pairs["share_of_all_errors"] = pairs["count"] / max(len(error_frame), 1)
    pairs["review_capture_rate"] = pairs["reviewed_errors"] / pairs["count"]
    pairs = pairs.sort_values(
        ["count", "true_class", "predicted_class"],
        ascending=[False, True, True],
    )

    return {
        "schema_version": 1,
        "strategy": policy["strategy"],
        "source_split": "test",
        "purpose": "post_selection_diagnostics",
        "used_for_model_selection": False,
        "temperature": float(policy["temperature"]),
        "review_threshold": float(policy["review_threshold"]),
        "total_rows": int(len(frame)),
        "total_errors": int((~correct).sum()),
        "accuracy": float(correct.mean()),
        "class_analysis": class_analysis,
        "confusion_pairs": pairs.to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_errors(args.test_predictions, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "total_rows": result["total_rows"],
                "total_errors": result["total_errors"],
                "top_confusion_pairs": result["confusion_pairs"][:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
