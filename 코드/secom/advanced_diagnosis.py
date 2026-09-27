"""Safety and decision-support utilities for the SECOM dashboard.

The functions in this module do not claim physical process causality. They add
distribution checks, constrained model counterfactuals, failure-signature
assignment, human-review prioritisation, and an auditable HTML report around the
existing CatBoost/XGBoost models.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from diagnose_secom import shap_matrix


MODEL_NAMES = ("CatBoost", "XGBoost")
OOD_LABELS = {
    "in_distribution": "학습 분포 내부",
    "review": "분포 경계 · 검토",
    "out_of_distribution": "분포 이탈 · 판정 보류",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _filled_retained(X_raw: pd.DataFrame, reference: dict[str, Any]) -> np.ndarray:
    features = reference["input_features"]
    missing = [feature for feature in features if feature not in X_raw.columns]
    if missing:
        raise ValueError(f"OOD 입력에 필요한 센서가 없습니다: {missing[:5]}")
    values = X_raw[features].to_numpy(dtype=float, copy=True)
    medians = np.asarray(reference["medians"], dtype=float)
    rows, columns = np.where(np.isnan(values))
    values[rows, columns] = medians[columns]
    return values


def assess_ood(X_raw: pd.DataFrame, reference: dict[str, Any]) -> pd.DataFrame:
    """Assess row-level missingness, range violations, and neighbour distance."""
    retained = X_raw[reference["input_features"]].to_numpy(dtype=float, copy=True)
    observed = np.isfinite(retained)
    lower = np.asarray(reference["q01"], dtype=float)
    upper = np.asarray(reference["q99"], dtype=float)
    outside = observed & ((retained < lower) | (retained > upper))
    observed_count = np.maximum(observed.sum(axis=1), 1)
    range_rate = outside.sum(axis=1) / observed_count

    raw_features = [
        feature for feature in reference["raw_features"] if feature in X_raw.columns
    ]
    missing_rate = X_raw[raw_features].isna().mean(axis=1).to_numpy(dtype=float)

    filled = _filled_retained(X_raw, reference)
    scaled = reference["ood_scaler"].transform(filled)
    embedded = reference["ood_pca"].transform(scaled)
    distances = pairwise_distances(
        embedded,
        np.asarray(reference["reference_embedding"]),
        metric="euclidean",
    ).min(axis=1)

    thresholds = reference["ood_thresholds"]
    rows = []
    for index in range(len(X_raw)):
        severe_reasons = []
        review_reasons = []
        if distances[index] > thresholds["distance_outlier"]:
            severe_reasons.append("학습 샘플과 거리가 매우 큼")
        elif distances[index] > thresholds["distance_review"]:
            review_reasons.append("학습 분포 거리 경계")
        if missing_rate[index] > thresholds["missing_outlier"]:
            severe_reasons.append("결측률이 학습 범위를 크게 초과")
        elif missing_rate[index] > thresholds["missing_review"]:
            review_reasons.append("결측률이 학습 분포 상위권")
        if range_rate[index] > thresholds["range_outlier"]:
            severe_reasons.append("센서 범위 이탈이 많음")
        elif range_rate[index] > thresholds["range_review"]:
            review_reasons.append("센서 범위 이탈 감지")

        if severe_reasons:
            status = "out_of_distribution"
            reasons = severe_reasons + review_reasons
        elif review_reasons:
            status = "review"
            reasons = review_reasons
        else:
            status = "in_distribution"
            reasons = ["주요 분포 검사 통과"]

        distance_ratio = distances[index] / max(thresholds["distance_outlier"], 1e-12)
        missing_ratio = missing_rate[index] / max(thresholds["missing_outlier"], 1e-12)
        range_ratio = range_rate[index] / max(thresholds["range_outlier"], 1e-12)
        rows.append(
            {
                "행 번호": int(index),
                "입력 신뢰도": OOD_LABELS[status],
                "OOD 상태": status,
                "OOD 점수": float(min(max(distance_ratio, missing_ratio, range_ratio), 9.999)),
                "최근접 거리": float(distances[index]),
                "결측률": float(missing_rate[index]),
                "범위 이탈률": float(range_rate[index]),
                "검토 사유": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def model_row_probabilities(
    X_row: pd.DataFrame,
    bundles: dict[str, dict[str, Any]],
    profile: str,
) -> tuple[dict[str, float], dict[str, float]]:
    probabilities: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    for model_name in MODEL_NAMES:
        bundle = bundles[model_name]
        values = bundle["imputer"].transform(X_row[bundle["input_features"]])
        ready = bundle["selector"].transform(values)
        probabilities[model_name] = float(bundle["model"].predict_proba(ready)[0, 1])
        thresholds[model_name] = float(bundle["operating_thresholds"][profile])
    return probabilities, thresholds


def assign_failure_signature(
    X_row: pd.DataFrame,
    catboost_bundle: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Assign the closest SHAP-derived failure pattern candidate."""
    values = catboost_bundle["imputer"].transform(
        X_row[catboost_bundle["input_features"]]
    )
    ready = catboost_bundle["selector"].transform(values)
    shap_values = shap_matrix(catboost_bundle["model"], ready)
    signature_values = shap_values[:, reference["signature_feature_indices"]]
    scaled = reference["signature_scaler"].transform(signature_values)
    embedded = reference["signature_pca"].transform(scaled)
    cluster = int(reference["signature_kmeans"].predict(embedded)[0])
    details = reference["signature_details"][cluster]
    centroid = reference["signature_kmeans"].cluster_centers_[cluster]
    distance = float(np.linalg.norm(embedded[0] - centroid))
    return {
        "cluster": cluster,
        "pattern": details["pattern"],
        "training_defect_rows": int(details["training_defect_rows"]),
        "top_risk_features": list(details["top_risk_features"]),
        "top_protective_features": list(details["top_protective_features"]),
        "distance_to_pattern": distance,
        "shap_values": shap_values[0],
    }


def _objective(
    probabilities: dict[str, float], thresholds: dict[str, float]
) -> float:
    return max(
        probabilities[model_name] / max(thresholds[model_name], 1e-12)
        for model_name in MODEL_NAMES
    )


def constrained_counterfactual(
    X_row: pd.DataFrame,
    bundles: dict[str, dict[str, Any]],
    reference: dict[str, Any],
    profile: str,
    *,
    max_changes: int = 4,
    max_candidates: int = 14,
    max_iqr_change: float = 3.0,
) -> dict[str, Any]:
    """Greedy, observed-range counterfactual for two-model consensus.

    Only SHAP-positive candidate features are considered, each target is an
    observed training quantile, and at most ``max_changes`` sensors change.
    The result is a model-based inspection hypothesis, never a process recipe.
    """
    if len(X_row) != 1:
        raise ValueError("반사실적 탐색은 한 행만 받을 수 있습니다.")
    current = X_row.copy()
    before, thresholds = model_row_probabilities(current, bundles, profile)
    current_probabilities = before.copy()

    contribution: dict[str, float] = {}
    for model_name in MODEL_NAMES:
        bundle = bundles[model_name]
        values = bundle["imputer"].transform(current[bundle["input_features"]])
        ready = bundle["selector"].transform(values)
        shap_values = shap_matrix(bundle["model"], ready)[0]
        for feature, value in zip(bundle["selected_features"], shap_values):
            if value > 0:
                contribution[feature] = contribution.get(feature, 0.0) + float(value)
    candidates = [
        feature
        for feature, _ in sorted(
            contribution.items(), key=lambda item: item[1], reverse=True
        )[:max_candidates]
        if feature in reference["feature_index"]
    ]

    changes = []
    used: set[str] = set()
    current_objective = _objective(current_probabilities, thresholds)
    quantile_names = ("q05", "q25", "median", "q75", "q95")
    for _ in range(max_changes):
        if all(
            current_probabilities[name] < thresholds[name] for name in MODEL_NAMES
        ):
            break
        best: dict[str, Any] | None = None
        for feature in candidates:
            if feature in used:
                continue
            feature_index = int(reference["feature_index"][feature])
            original_value = float(current.iloc[0][feature])
            if not np.isfinite(original_value):
                original_value = float(reference["medians"][feature_index])
            scale = max(float(reference["iqr"][feature_index]), 1e-9)
            for quantile_name in quantile_names:
                target = float(reference[quantile_name][feature_index])
                if not np.isfinite(target) or np.isclose(target, original_value):
                    continue
                trial = current.copy()
                trial.loc[trial.index[0], feature] = target
                probabilities, _ = model_row_probabilities(trial, bundles, profile)
                objective = _objective(probabilities, thresholds)
                normalized_change = abs(target - original_value) / scale
                if normalized_change > max_iqr_change:
                    continue
                penalized = objective + 0.01 * normalized_change
                candidate = {
                    "feature": feature,
                    "from": original_value,
                    "to": target,
                    "quantile": quantile_name,
                    "normalized_change": normalized_change,
                    "probabilities": probabilities,
                    "objective": objective,
                    "penalized": penalized,
                    "trial": trial,
                }
                if objective < current_objective - 1e-8 and (
                    best is None or candidate["penalized"] < best["penalized"]
                ):
                    best = candidate
        if best is None:
            break
        current = best.pop("trial")
        current_probabilities = best.pop("probabilities")
        current_objective = float(best.pop("objective"))
        best.pop("penalized")
        used.add(str(best["feature"]))
        changes.append(best)

    success = all(
        current_probabilities[name] < thresholds[name] for name in MODEL_NAMES
    )
    return {
        "success": success,
        "before": before,
        "after": current_probabilities,
        "thresholds": thresholds,
        "changes": changes,
        "candidate_features": candidates,
        "objective_before": _objective(before, thresholds),
        "objective_after": current_objective,
        "warning": (
            "관측 train 범위 안에서 찾은 모델 기반 점검 후보이며 물리적 공정 처방이나 "
            "인과관계를 의미하지 않습니다. 센서 간 결합 제약은 반영되지 않았습니다."
        ),
    }


def build_review_queue(
    predictions: pd.DataFrame,
    ood: pd.DataFrame,
) -> pd.DataFrame:
    """Combine model decisions, margins, and OOD into a review queue."""
    queue = predictions.merge(ood, on="행 번호", how="left", validate="one_to_one")
    cat_ratio = queue["CatBoost 점수"] / queue["CatBoost 임계값"].clip(lower=1e-12)
    xgb_ratio = queue["XGBoost 점수"] / queue["XGBoost 임계값"].clip(lower=1e-12)
    boundary = np.minimum(abs(cat_ratio - 1.0), abs(xgb_ratio - 1.0))
    both_defect = queue["종합 판정"] == "두 모델 모두 불량"
    disagreement = queue["종합 판정"].isin(
        ("CatBoost만 불량", "XGBoost만 불량")
    )
    severe_ood = queue["OOD 상태"] == "out_of_distribution"
    review_ood = queue["OOD 상태"] == "review"
    near_boundary = boundary <= 0.25

    queue["검토 우선순위 점수"] = (
        4.0 * both_defect.astype(float)
        + 3.0 * disagreement.astype(float)
        + 3.0 * severe_ood.astype(float)
        + 1.5 * review_ood.astype(float)
        + 1.0 * near_boundary.astype(float)
        + np.minimum(queue["우선 확인 점수"], 5.0) / 5.0
    )
    queue["처리 권고"] = "자동 처리 후보"
    queue.loc[near_boundary, "처리 권고"] = "경계값 검토"
    queue.loc[disagreement, "처리 권고"] = "모델 불일치 검토"
    queue.loc[both_defect, "처리 권고"] = "불량 긴급 검토"
    queue.loc[severe_ood, "처리 권고"] = "자동판정 보류 · OOD 검토"
    return queue.sort_values(
        ["검토 우선순위 점수", "우선 확인 점수"], ascending=False
    ).reset_index(drop=True)


def build_html_report(
    *,
    source_name: str,
    profile_label: str,
    predictions: pd.DataFrame,
    ood: pd.DataFrame,
    review_queue: pd.DataFrame,
    model_hashes: dict[str, str],
) -> bytes:
    """Build a self-contained, escaped HTML audit report for one batch."""
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    normalized = predictions.drop(columns=["우선 확인 점수"], errors="ignore").to_csv(
        index=False
    )
    input_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest().upper()
    both_defect = int((predictions["종합 판정"] == "두 모델 모두 불량").sum())
    disagreement = int(
        predictions["종합 판정"].isin(
            ("CatBoost만 불량", "XGBoost만 불량")
        ).sum()
    )
    ood_count = int((ood["OOD 상태"] == "out_of_distribution").sum())
    report_columns = [
        "행 번호",
        "처리 권고",
        "종합 판정",
        "입력 신뢰도",
        "CatBoost 점수",
        "XGBoost 점수",
        "검토 사유",
    ]
    table = review_queue[report_columns].head(50).to_html(
        index=False,
        escape=True,
        float_format=lambda value: f"{value:.5f}",
        border=0,
    )
    model_rows = "".join(
        f"<li>{html.escape(name)} SHA-256: <code>{html.escape(value)}</code></li>"
        for name, value in model_hashes.items()
    )
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>SECOM 진단 감사 보고서</title>
<style>
body{{font-family:Arial,'Malgun Gothic',sans-serif;max-width:1100px;margin:32px auto;color:#1f2937;line-height:1.55}}
h1,h2{{color:#123b5d}} .cards{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{border:1px solid #d1d5db;border-radius:10px;padding:12px 18px;min-width:150px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid #ddd;padding:7px;text-align:left}}
.warning{{background:#fff3cd;border-left:5px solid #e0a800;padding:12px}} code{{word-break:break-all}}
</style></head><body>
<h1>SECOM 설명가능 AI 진단 감사 보고서</h1>
<p>생성 시각: {html.escape(generated)}<br>입력: {html.escape(source_name)}<br>
운영 임계값: {html.escape(profile_label)}<br>정규화 진단 결과 SHA-256: <code>{input_digest}</code></p>
<div class="cards"><div class="card"><b>전체 행</b><br>{len(predictions):,}</div>
<div class="card"><b>두 모델 모두 불량</b><br>{both_defect:,}</div>
<div class="card"><b>모델 불일치</b><br>{disagreement:,}</div>
<div class="card"><b>OOD 판정 보류</b><br>{ood_count:,}</div></div>
<h2>모델 무결성</h2><ul>{model_rows}</ul>
<h2>전문가 검토 우선순위 상위 50행</h2>{table}
<h2>해석 제한</h2><div class="warning">SECOM 센서는 익명 변수입니다. SHAP, OOD,
불량 패턴 및 반사실적 결과는 모델 기반 점검 후보이며 물리적 원인이나 공정 조정 지시를
증명하지 않습니다. 신규 장비·기간·lot에는 외부 검증이 필요합니다.</div>
</body></html>"""
    return document.encode("utf-8")
