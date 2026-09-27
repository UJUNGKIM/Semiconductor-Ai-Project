"""Label-grounded monitoring and conservative retraining review for SECOM."""

from __future__ import annotations

import json
import math
from typing import Mapping

import pandas as pd

from monitoring_log import validate_monitoring_log


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def analyze_feedback_history(frame: pd.DataFrame, reference: Mapping) -> dict:
    """Evaluate only outcome-confirmed rows from the latest model cohort."""
    checked = validate_monitoring_log(frame).sort_values("recorded_at")
    has_aggregates = checked["outcome_confirmed"] & checked["confirmed_alerted_defects"].notna()
    verified_evidence = checked["feedback_evidence"].isin(("row_level_complete", "oof_replay"))
    confirmed = checked.loc[has_aggregates & verified_evidence].copy()
    if confirmed.empty:
        if bool(has_aggregates.any()):
            return {
                "status": "UNVERIFIED_LABELS",
                "ready": False,
                "reasons": ["숫자 직접 입력 결과는 참고용이며 행 단위 전체 검수 근거가 아닙니다."],
                "confirmed_batches": 0,
                "confirmed_rows": 0,
                "confirmed_defects": 0,
            }
        return {
            "status": "NO_LABELS",
            "ready": False,
            "reasons": ["경보와 실제 불량의 교차 검수 결과가 없습니다."],
            "confirmed_batches": 0,
            "confirmed_rows": 0,
            "confirmed_defects": 0,
        }

    latest = confirmed.iloc[-1]
    expected_hashes = reference["model_hashes"]
    reference_matches = (
        str(latest["profile"]) == str(reference["profile"])
        and str(latest["catboost_model_hash"]) == str(expected_hashes["CatBoost"])
        and str(latest["xgboost_model_hash"]) == str(expected_hashes["XGBoost"])
    )
    if not reference_matches:
        return {
            "status": "REFERENCE_MISMATCH",
            "ready": False,
            "reasons": ["현재 모델 SHA-256 또는 임계값 프로필과 OOF 기준이 다릅니다."],
            "profile": str(latest["profile"]),
            "confirmed_batches": 0,
            "confirmed_rows": 0,
            "confirmed_defects": 0,
        }
    cohort_mask = (
        confirmed["profile"].eq(latest["profile"])
        & confirmed["catboost_model_hash"].eq(latest["catboost_model_hash"])
        & confirmed["xgboost_model_hash"].eq(latest["xgboost_model_hash"])
    )
    cohort = confirmed.loc[cohort_mask].copy()
    rows = int(cohort["row_count"].sum())
    actual = int(cohort["confirmed_defects"].sum())
    alerted = int((cohort["both_models_defect"] + cohort["one_model_defect"]).sum())
    true_positive = int(cohort["confirmed_alerted_defects"].sum())
    false_positive = alerted - true_positive
    false_negative = actual - true_positive
    true_negative = rows - true_positive - false_positive - false_negative
    if min(false_positive, false_negative, true_negative) < 0:
        raise ValueError("검수 집계로 만든 혼동행렬에 음수 값이 생겼습니다.")

    recall = _safe_divide(true_positive, actual)
    precision = _safe_divide(true_positive, alerted)
    recall_low, recall_high = wilson_interval(true_positive, actual)
    actual_rate = _safe_divide(actual, rows)
    minimums = reference["minimum_evidence"]
    ready = (
        len(cohort) >= int(minimums["batches"])
        and rows >= int(minimums["rows"])
        and actual >= int(minimums["defects"])
    )
    reference_recall = float(reference["oof_reference"]["recall"])
    recall_limit = reference_recall - float(reference["guardrails"]["recall_drop_margin"])
    prevalence = float(reference["oof_reference"]["prevalence"])
    prevalence_ratio = actual_rate / prevalence if prevalence else float("inf")
    reasons: list[str] = []

    batch_recall = cohort.apply(
        lambda row: _safe_divide(
            int(row["confirmed_alerted_defects"]), int(row["confirmed_defects"])
        ),
        axis=1,
    )
    eligible_recent = batch_recall.loc[cohort["confirmed_defects"] > 0].tail(
        int(reference["guardrails"]["persistent_batches"])
    )
    persistent = (
        len(eligible_recent) >= int(reference["guardrails"]["persistent_batches"])
        and bool((eligible_recent < recall_limit).all())
    )

    if not ready:
        status = "INSUFFICIENT"
        reasons.append(
            f"최소 근거 미충족: {len(cohort)}/{minimums['batches']}배치, "
            f"{rows}/{minimums['rows']}행, 불량 {actual}/{minimums['defects']}건"
        )
    elif recall_high < recall_limit or persistent:
        status = "REVIEW_RETRAINING"
        if recall_high < recall_limit:
            reasons.append("재현율 95% 신뢰구간 상한이 사전 고정 하한보다 낮습니다.")
        if persistent:
            reasons.append("최근 검수 배치에서 재현율 저하가 연속 발생했습니다.")
    elif recall < recall_limit or (
        prevalence_ratio >= float(reference["guardrails"]["prevalence_ratio_warning"])
    ):
        status = "WATCH"
        if recall < recall_limit:
            reasons.append("재현율 점추정치가 사전 고정 하한보다 낮습니다.")
        if prevalence_ratio >= float(reference["guardrails"]["prevalence_ratio_warning"]):
            reasons.append("실제 불량률이 OOF 기준보다 크게 증가했습니다.")
    else:
        status = "STABLE"
        reasons.append("사전 고정된 재현율·불량률 감시 기준 이내입니다.")

    false_negative_batches = cohort.loc[
        cohort["confirmed_defects"] > cohort["confirmed_alerted_defects"], "batch_id"
    ].astype(str).tolist()
    return {
        "status": status,
        "ready": ready,
        "reasons": reasons,
        "profile": str(latest["profile"]),
        "catboost_model_hash": str(latest["catboost_model_hash"]),
        "xgboost_model_hash": str(latest["xgboost_model_hash"]),
        "confirmed_batches": len(cohort),
        "confirmed_rows": rows,
        "confirmed_defects": actual,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "recall_ci_low": recall_low,
        "recall_ci_high": recall_high,
        "actual_defect_rate": actual_rate,
        "reference_recall": reference_recall,
        "recall_guardrail": recall_limit,
        "reference_prevalence": prevalence,
        "prevalence_ratio": prevalence_ratio,
        "candidate_batch_ids": false_negative_batches,
    }


def build_candidate_manifest(summary: Mapping) -> dict:
    """Create a retrieval manifest; it never contains raw sensors or starts training."""
    return {
        "version": 1,
        "decision": summary["status"],
        "automatic_retraining_started": False,
        "profile": summary.get("profile"),
        "catboost_model_hash": summary.get("catboost_model_hash"),
        "xgboost_model_hash": summary.get("xgboost_model_hash"),
        "candidate_batch_ids": list(summary.get("candidate_batch_ids", [])),
        "selection_reason": list(summary.get("reasons", [])),
        "required_next_steps": [
            "보안 원본 저장소에서 batch_id로 센서와 행 단위 라벨을 별도 조회",
            "라벨 품질·중복·시간 누수 점검",
            "기존 모델과 동일한 고정 test 외 별도 미래 검증 세트에서 후보 모델 평가",
            "사람의 승인 후에만 모델 교체",
        ],
    }


def manifest_json_bytes(summary: Mapping) -> bytes:
    return json.dumps(build_candidate_manifest(summary), ensure_ascii=False, indent=2).encode(
        "utf-8"
    )
