"""Adaptive batch-level circuit breaker for SECOM diagnosis."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


SIGNALS = ("ood_any_rate", "ood_severe_rate", "model_disagreement_rate")


def adaptive_limits(
    baseline_rate: float,
    batch_rows: int,
    warning_floor: float,
    stop_floor: float,
) -> tuple[float, float]:
    """Create conservative one-sided binomial limits for the current batch size."""
    if batch_rows <= 0:
        raise ValueError("batch_rows는 양수여야 합니다.")
    rate = float(np.clip(baseline_rate, 0.0, 1.0))
    standard_error = math.sqrt(max(rate * (1.0 - rate), 1e-9) / batch_rows)
    warning = max(float(warning_floor), rate + 2.326 * standard_error)
    stop = max(float(stop_floor), rate + 3.090 * standard_error)
    stop = max(stop, warning + 1.0 / batch_rows)
    return min(warning, 1.0), min(stop, 1.0)


def evaluate_gate_from_rates(
    rates: Mapping[str, float],
    batch_rows: int,
    reference: Mapping,
) -> dict:
    """Evaluate PASS/CAUTION/STOP from aggregate batch signals."""
    missing = [signal for signal in SIGNALS if signal not in rates]
    if missing:
        raise ValueError(f"배치 안전 신호가 누락됐습니다: {missing}")
    if batch_rows <= 0:
        raise ValueError("빈 배치는 평가할 수 없습니다.")

    minimum_rows = int(reference["minimum_batch_rows"])
    details = {}
    stop_reasons = []
    warning_reasons = []
    for signal in SIGNALS:
        config = reference["signals"][signal]
        warning_limit, stop_limit = adaptive_limits(
            float(config["baseline_rate"]),
            batch_rows,
            float(config["warning_floor"]),
            float(config["stop_floor"]),
        )
        value = float(rates[signal])
        details[signal] = {
            "value": value,
            "warning_limit": warning_limit,
            "stop_limit": stop_limit,
        }
        if batch_rows >= minimum_rows:
            if value >= stop_limit:
                stop_reasons.append(signal)
            elif value >= warning_limit:
                warning_reasons.append(signal)

    if batch_rows < minimum_rows:
        severe_rows = int(round(float(rates["ood_severe_rate"]) * batch_rows))
        status = "CAUTION" if severe_rows else "ROW_LEVEL_ONLY"
        reasons = (
            ["소규모 배치의 severe OOD 행을 개별 보류"]
            if severe_rows
            else [f"배치 통계 최소 {minimum_rows}행 미만 · 행 단위 안전장치 사용"]
        )
    elif stop_reasons:
        status = "STOP"
        reasons = [f"{signal} 중단 한계 초과" for signal in stop_reasons]
    elif warning_reasons:
        status = "CAUTION"
        reasons = [f"{signal} 경고 한계 초과" for signal in warning_reasons]
    else:
        status = "PASS"
        reasons = ["모든 배치 안전 신호가 허용 범위 이내"]

    return {
        "status": status,
        "batch_rows": int(batch_rows),
        "automatic_decision_allowed": status in {"PASS", "ROW_LEVEL_ONLY"},
        "reasons": reasons,
        "signals": details,
    }


def evaluate_batch_gate(
    predictions: pd.DataFrame,
    ood: pd.DataFrame,
    reference: Mapping,
) -> dict:
    """Evaluate a live dashboard batch from model and OOD outputs."""
    if len(predictions) != len(ood):
        raise ValueError("예측과 OOD 결과의 행 수가 다릅니다.")
    disagreement = predictions["종합 판정"].isin(
        ("CatBoost만 불량", "XGBoost만 불량")
    )
    rates = {
        "ood_any_rate": float((ood["OOD 상태"] != "in_distribution").mean()),
        "ood_severe_rate": float(
            (ood["OOD 상태"] == "out_of_distribution").mean()
        ),
        "model_disagreement_rate": float(disagreement.mean()),
    }
    return evaluate_gate_from_rates(rates, len(predictions), reference)
