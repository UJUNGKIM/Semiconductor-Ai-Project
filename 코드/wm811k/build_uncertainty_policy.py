"""Build a validation-only uncertainty review policy for WM-811K predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


CLASS_NAMES = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
)
PROBABILITY_COLUMNS = tuple(
    "probability_" + name.lower().replace("-", "_") for name in CLASS_NAMES
)


def validate_predictions(frame: pd.DataFrame, source: Path) -> tuple[np.ndarray, np.ndarray]:
    """Validate a saved prediction table and return probabilities and labels."""
    required = {"label_id", "predicted_label_id", *PROBABILITY_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} 필수 열 누락: {sorted(missing)}")
    probabilities = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    labels = frame["label_id"].to_numpy(dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"{source} 확률 배열 크기가 올바르지 않습니다.")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{source} 확률에 NaN 또는 무한대가 있습니다.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"{source} 확률이 0~1 범위를 벗어났습니다.")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=3e-7)
    if np.any(labels < 0) or np.any(labels >= len(CLASS_NAMES)):
        raise ValueError(f"{source} label_id가 클래스 범위를 벗어났습니다.")
    predicted = probabilities.argmax(axis=1)
    if not np.array_equal(predicted, frame["predicted_label_id"].to_numpy(dtype=int)):
        raise ValueError(f"{source} 저장 예측과 확률 argmax가 다릅니다.")
    return probabilities, labels


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply multiclass temperature scaling to saved softmax probabilities."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature는 0보다 큰 유한한 값이어야 합니다.")
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    log_probabilities -= log_probabilities.max(axis=1, keepdims=True)
    exponentials = np.exp(log_probabilities)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Return mean multiclass negative log-likelihood."""
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Fit one temperature on validation predictions only."""
    result = minimize_scalar(
        lambda value: negative_log_likelihood(
            temperature_scale(probabilities, float(value)), labels
        ),
        bounds=(0.25, 5.0),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"온도 보정 최적화 실패: {result.message}")
    return float(result.x)


def selective_metrics(
    probabilities: np.ndarray, labels: np.ndarray, review_threshold: float
) -> dict[str, float | int]:
    """Evaluate the fixed review threshold without changing predictions."""
    predicted = probabilities.argmax(axis=1)
    correct = predicted == labels
    confidence = probabilities.max(axis=1)
    automatic = confidence >= review_threshold
    review = ~automatic
    errors = ~correct
    automatic_count = int(automatic.sum())
    review_count = int(review.sum())
    total_errors = int(errors.sum())
    captured_errors = int((review & errors).sum())
    return {
        "rows": int(len(labels)),
        "automatic_count": automatic_count,
        "review_count": review_count,
        "automatic_coverage": float(automatic.mean()),
        "review_rate": float(review.mean()),
        "overall_accuracy": float(correct.mean()),
        "automatic_accuracy": (
            float(correct[automatic].mean()) if automatic_count else 0.0
        ),
        "total_errors": total_errors,
        "captured_errors": captured_errors,
        "error_capture_rate": (
            captured_errors / total_errors if total_errors else 1.0
        ),
        "review_error_rate": float(errors[review].mean()) if review_count else 0.0,
    }


def build_policy(
    validation_path: Path,
    test_path: Path,
    target_automatic_coverage: float = 0.90,
) -> dict[str, object]:
    """Fit calibration and threshold on validation, then evaluate frozen policy on test."""
    if not 0 < target_automatic_coverage < 1:
        raise ValueError("target_automatic_coverage는 0과 1 사이여야 합니다.")
    validation_frame = pd.read_csv(validation_path)
    test_frame = pd.read_csv(test_path)
    validation_raw, validation_labels = validate_predictions(
        validation_frame, validation_path
    )
    test_raw, test_labels = validate_predictions(test_frame, test_path)

    temperature = fit_temperature(validation_raw, validation_labels)
    validation_probabilities = temperature_scale(validation_raw, temperature)
    test_probabilities = temperature_scale(test_raw, temperature)
    validation_confidence = validation_probabilities.max(axis=1)
    review_threshold = float(
        np.quantile(
            validation_confidence,
            1.0 - target_automatic_coverage,
            method="higher",
        )
    )
    return {
        "schema_version": 1,
        "strategy": "ce_sqrt_balanced",
        "method": "temperature_scaled_max_probability",
        "fitted_on": "validation",
        "selection_uses_test": False,
        "target_validation_automatic_coverage": target_automatic_coverage,
        "temperature": temperature,
        "review_threshold": review_threshold,
        "validation_nll_before": negative_log_likelihood(
            validation_raw, validation_labels
        ),
        "validation_nll_after": negative_log_likelihood(
            validation_probabilities, validation_labels
        ),
        "class_names": list(CLASS_NAMES),
        "probability_columns": list(PROBABILITY_COLUMNS),
        "validation_metrics": selective_metrics(
            validation_probabilities, validation_labels, review_threshold
        ),
        "test_evaluation": selective_metrics(
            test_probabilities, test_labels, review_threshold
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-automatic-coverage", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = build_policy(
        args.validation_predictions,
        args.test_predictions,
        args.target_automatic_coverage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(policy, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
