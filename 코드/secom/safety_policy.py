"""Leakage-safe policy utilities for SECOM human review prioritisation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def model_alert_mask(
    probability: Iterable[float], threshold: float
) -> np.ndarray:
    """Apply a tiny tolerance for probabilities serialized through CSV."""
    values = np.asarray(probability, dtype=float)
    tolerance = max(abs(float(threshold)) * 1e-12, np.finfo(float).eps * 8)
    return values >= float(threshold) - tolerance


def normalized_risk_ratio(
    catboost_probability: Iterable[float],
    xgboost_probability: Iterable[float],
    catboost_threshold: float,
    xgboost_threshold: float,
) -> np.ndarray:
    """Return the largest model probability relative to its OOF threshold."""
    cat = np.asarray(catboost_probability, dtype=float)
    xgb = np.asarray(xgboost_probability, dtype=float)
    if cat.shape != xgb.shape:
        raise ValueError("두 모델 확률 배열의 크기가 다릅니다.")
    return np.maximum(
        cat / max(float(catboost_threshold), 1e-12),
        xgb / max(float(xgboost_threshold), 1e-12),
    )


def locked_review_mask(
    catboost_probability: Iterable[float],
    xgboost_probability: Iterable[float],
    catboost_threshold: float,
    xgboost_threshold: float,
    *,
    margin: float = 0.25,
    ood_status: Iterable[str] | None = None,
) -> np.ndarray:
    """Apply the predeclared alert, boundary, and severe-OOD review rule."""
    if not 0 <= margin < 1:
        raise ValueError("margin은 0 이상 1 미만이어야 합니다.")
    cat = np.asarray(catboost_probability, dtype=float)
    xgb = np.asarray(xgboost_probability, dtype=float)
    ratio = normalized_risk_ratio(
        cat, xgb, catboost_threshold, xgboost_threshold
    )
    alert = model_alert_mask(cat, catboost_threshold) | model_alert_mask(
        xgb, xgboost_threshold
    )
    review = alert | (ratio >= 1.0 - margin)
    if ood_status is not None:
        status = np.asarray(ood_status, dtype=str)
        if status.shape != review.shape:
            raise ValueError("OOD 상태 배열의 크기가 다릅니다.")
        review |= status == "out_of_distribution"
    return review

def policy_metrics(
    y_true: Iterable[int],
    automated_prediction: Iterable[int],
    review_mask: Iterable[bool],
) -> dict[str, float | int]:
    """Measure safety benefit and manual workload for a review policy."""
    y = np.asarray(y_true, dtype=int)
    prediction = np.asarray(automated_prediction, dtype=int)
    review = np.asarray(review_mask, dtype=bool)
    if not (y.shape == prediction.shape == review.shape):
        raise ValueError("정답, 자동판정, 검토 배열의 크기가 다릅니다.")
    if y.size == 0:
        raise ValueError("평가할 행이 없습니다.")

    errors = prediction != y
    false_negatives = (y == 1) & (prediction == 0)
    false_positives = (y == 0) & (prediction == 1)
    auto = ~review

    def capture(mask: np.ndarray) -> float:
        count = int(mask.sum())
        return float((review & mask).sum() / count) if count else 1.0

    return {
        "rows": int(y.size),
        "reviewed_rows": int(review.sum()),
        "review_workload": float(review.mean()),
        "auto_rows": int(auto.sum()),
        "auto_coverage": float(auto.mean()),
        "auto_accuracy": float((prediction[auto] == y[auto]).mean())
        if auto.any()
        else 1.0,
        "errors": int(errors.sum()),
        "captured_errors": int((review & errors).sum()),
        "error_capture": capture(errors),
        "false_negatives": int(false_negatives.sum()),
        "captured_false_negatives": int((review & false_negatives).sum()),
        "false_negative_capture": capture(false_negatives),
        "false_positives": int(false_positives.sum()),
        "captured_false_positives": int((review & false_positives).sum()),
        "false_positive_capture": capture(false_positives),
        "post_review_recall": float((review & (y == 1)).sum() / (y == 1).sum())
        if (y == 1).any()
        else 1.0,
    }


def budget_review_mask(
    catboost_probability: Iterable[float],
    xgboost_probability: Iterable[float],
    catboost_threshold: float,
    xgboost_threshold: float,
    additional_normal_budget: float,
    *,
    ood_status: Iterable[str] | None = None,
) -> np.ndarray:
    """Review alerts plus a fixed fraction of highest-risk consensus normals."""
    if not 0 <= additional_normal_budget <= 1:
        raise ValueError("additional_normal_budget은 0~1이어야 합니다.")
    cat = np.asarray(catboost_probability, dtype=float)
    xgb = np.asarray(xgboost_probability, dtype=float)
    ratio = normalized_risk_ratio(
        cat, xgb, catboost_threshold, xgboost_threshold
    )
    review = model_alert_mask(cat, catboost_threshold) | model_alert_mask(
        xgb, xgboost_threshold
    )
    if ood_status is not None:
        status = np.asarray(ood_status, dtype=str)
        if status.shape != review.shape:
            raise ValueError("OOD 상태 배열의 크기가 다릅니다.")
        review |= status == "out_of_distribution"

    candidates = np.flatnonzero(~review)
    extra_rows = int(math.ceil(len(candidates) * additional_normal_budget))
    if extra_rows:
        order = np.argsort(-ratio[candidates], kind="stable")
        review[candidates[order[:extra_rows]]] = True
    return review


def score_cutoff_review_mask(
    catboost_probability: Iterable[float],
    xgboost_probability: Iterable[float],
    catboost_threshold: float,
    xgboost_threshold: float,
    risk_cutoff: float,
    *,
    ood_status: Iterable[str] | None = None,
) -> np.ndarray:
    """Apply a deployable OOF-locked normalized-risk cutoff."""
    if not 0 <= risk_cutoff <= 1:
        raise ValueError("risk_cutoff은 0~1이어야 합니다.")
    cat = np.asarray(catboost_probability, dtype=float)
    xgb = np.asarray(xgboost_probability, dtype=float)
    ratio = normalized_risk_ratio(
        cat, xgb, catboost_threshold, xgboost_threshold
    )
    review = model_alert_mask(cat, catboost_threshold) | model_alert_mask(
        xgb, xgboost_threshold
    )
    review |= ratio >= risk_cutoff
    if ood_status is not None:
        status = np.asarray(ood_status, dtype=str)
        if status.shape != review.shape:
            raise ValueError("OOD 상태 배열의 크기가 다릅니다.")
        review |= status == "out_of_distribution"
    return review
