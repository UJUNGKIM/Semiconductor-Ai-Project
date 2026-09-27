"""Paired, label-grounded CatBoost/XGBoost comparison with a no-auto-promotion gate."""

from __future__ import annotations

import json
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def _as_bool(series: pd.Series) -> np.ndarray:
    values = series.astype(str).str.strip().str.lower()
    if (~values.isin(("true", "false", "1", "0"))).any():
        raise ValueError("모델 경보 열은 true/false여야 합니다.")
    return values.isin(("true", "1")).to_numpy()


def _metrics(labels: np.ndarray, alerts: np.ndarray) -> dict:
    tp = int(np.sum((labels == 1) & alerts))
    fp = int(np.sum((labels == 0) & alerts))
    fn = int(np.sum((labels == 1) & ~alerts))
    tn = int(np.sum((labels == 0) & ~alerts))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    f2_denominator = 5 * tp + 4 * fn + fp
    f2 = 5 * tp / f2_denominator if f2_denominator else float("nan")
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(fpr),
        "f2": float(f2),
    }


def _paired_stratified_bootstrap(
    labels: np.ndarray,
    champion: np.ndarray,
    challenger: np.ndarray,
    *,
    draws: int,
    random_state: int,
) -> dict:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("paired bootstrap에는 정상과 불량 라벨이 모두 필요합니다.")
    rng = np.random.default_rng(random_state)
    differences = {"recall": [], "false_positive_rate": [], "f2": []}
    for _ in range(draws):
        sampled = np.concatenate(
            (
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            )
        )
        champion_metrics = _metrics(labels[sampled], champion[sampled])
        challenger_metrics = _metrics(labels[sampled], challenger[sampled])
        for metric in differences:
            differences[metric].append(challenger_metrics[metric] - champion_metrics[metric])
    return {
        metric: {
            "mean": float(np.mean(values)),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        }
        for metric, values in differences.items()
    }


def evaluate_champion_challenger(
    completed_feedback: pd.DataFrame,
    config: Mapping,
    *,
    allow_promotion_review: bool,
    bootstrap_draws: int | None = None,
) -> dict:
    """Compare models on the same fully labelled rows; never promotes automatically."""
    required = {"batch_id", "actual_label", "catboost_alert", "xgboost_alert"}
    missing = sorted(required - set(completed_feedback.columns))
    if missing:
        raise ValueError(f"모델 비교 필수 열 누락: {missing}")
    if completed_feedback["actual_label"].isna().any():
        raise ValueError("전체 행 라벨이 확인된 파일만 모델 비교에 사용할 수 있습니다.")
    labels = pd.to_numeric(completed_feedback["actual_label"], errors="raise").astype(int).to_numpy()
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("actual_label은 0/1이어야 합니다.")
    predictions = {
        "CatBoost": _as_bool(completed_feedback["catboost_alert"]),
        "XGBoost": _as_bool(completed_feedback["xgboost_alert"]),
    }
    champion_name = str(config["champion"])
    challenger_name = str(config["challenger"])
    champion = predictions[champion_name]
    challenger = predictions[challenger_name]
    champion_metrics = _metrics(labels, champion)
    challenger_metrics = _metrics(labels, challenger)
    champion_correct = champion == labels
    challenger_correct = challenger == labels
    champion_only_correct = int(np.sum(champion_correct & ~challenger_correct))
    challenger_only_correct = int(np.sum(~champion_correct & challenger_correct))
    discordant = champion_only_correct + challenger_only_correct
    mcnemar_p = (
        float(
            binomtest(
                min(champion_only_correct, challenger_only_correct),
                discordant,
                0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    draws = int(bootstrap_draws or config["bootstrap_draws"])
    intervals = _paired_stratified_bootstrap(
        labels,
        champion,
        challenger,
        draws=draws,
        random_state=int(config["random_state"]),
    )
    minimum = config["minimum_evidence"]
    batch_count = int(completed_feedback["batch_id"].astype(str).nunique())
    defect_count = int(labels.sum())
    ready = (
        batch_count >= int(minimum["batches"])
        and len(labels) >= int(minimum["rows"])
        and defect_count >= int(minimum["defects"])
        and discordant >= int(minimum["discordant_predictions"])
    )
    guardrails = config["guardrails"]
    recall_safe = intervals["recall"]["ci_low"] >= -float(
        guardrails["recall_noninferiority_margin"]
    )
    false_positive_safe = intervals["false_positive_rate"]["ci_high"] <= float(
        guardrails["maximum_fpr_increase"]
    )
    improvement = (
        intervals["f2"]["ci_low"] > 0
        or (
            mcnemar_p < float(guardrails["mcnemar_alpha"])
            and challenger_only_correct > champion_only_correct
        )
    )
    reasons = []
    if not allow_promotion_review:
        decision = "HISTORICAL_DEMO_ONLY"
        reasons.append("OOF 재생 자료는 실제 모델 승격 근거가 아닙니다.")
    elif not ready:
        decision = "INSUFFICIENT_EVIDENCE"
        reasons.append(
            f"최소 근거 미충족: {batch_count}/{minimum['batches']}배치, "
            f"{len(labels)}/{minimum['rows']}행, 불량 {defect_count}/{minimum['defects']}건, "
            f"불일치 {discordant}/{minimum['discordant_predictions']}건"
        )
    elif not recall_safe or not false_positive_safe:
        decision = "KEEP_CHAMPION"
        if not recall_safe:
            reasons.append("challenger 재현율 비열등성 기준을 충족하지 못했습니다.")
        if not false_positive_safe:
            reasons.append("challenger 정상 오탐률 증가 상한을 초과했습니다.")
    elif improvement:
        decision = "REVIEW_PROMOTION"
        reasons.append("안전 기준과 개선 근거를 충족해 사람의 승격 검토가 가능합니다.")
    else:
        decision = "KEEP_COLLECTING"
        reasons.append("안전 기준은 통과했지만 우월성 근거가 충분하지 않습니다.")
    return {
        "decision": decision,
        "automatic_promotion_performed": False,
        "champion": champion_name,
        "challenger": challenger_name,
        "batch_count": batch_count,
        "rows": len(labels),
        "defects": defect_count,
        "discordant_predictions": discordant,
        "champion_only_correct": champion_only_correct,
        "challenger_only_correct": challenger_only_correct,
        "mcnemar_exact_p_value": mcnemar_p,
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "paired_bootstrap_delta_challenger_minus_champion": intervals,
        "recall_noninferiority_pass": recall_safe,
        "false_positive_guardrail_pass": false_positive_safe,
        "improvement_evidence": improvement,
        "minimum_evidence_met": ready,
        "reasons": reasons,
    }


def comparison_json_bytes(report: Mapping) -> bytes:
    return json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
