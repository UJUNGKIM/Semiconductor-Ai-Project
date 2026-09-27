"""Metrics for post-selection SECOM SHAP stability and deletion fidelity audits."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not 1 <= int(k) <= len(values):
        raise ValueError("values는 1차원이고 k는 변수 수 범위여야 합니다.")
    return np.argsort(np.abs(values), kind="stable")[-int(k) :][::-1]


def explanation_similarity(
    reference: np.ndarray, candidate: np.ndarray, *, top_k: int
) -> dict:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("두 SHAP 벡터는 같은 길이의 1차원 배열이어야 합니다.")
    reference_top = top_k_indices(reference, top_k)
    candidate_top = top_k_indices(candidate, top_k)
    reference_set = set(reference_top.tolist())
    candidate_set = set(candidate_top.tolist())
    union = reference_set | candidate_set
    jaccard = len(reference_set & candidate_set) / len(union)
    sign_agreement = float(
        np.mean(np.sign(reference[reference_top]) == np.sign(candidate[reference_top]))
    )
    correlation = spearmanr(np.abs(reference), np.abs(candidate)).statistic
    if not np.isfinite(correlation):
        correlation = 1.0 if np.allclose(reference, candidate) else 0.0
    return {
        "top_k_jaccard": float(jaccard),
        "top_k_sign_agreement": sign_agreement,
        "absolute_rank_spearman": float(correlation),
    }


def perturb_within_reference(
    values: np.ndarray,
    iqr: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    noise_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    iqr = np.asarray(iqr, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if values.ndim != 2 or any(
        vector.shape != (values.shape[1],) for vector in (iqr, lower, upper)
    ):
        raise ValueError("참조 통계의 변수 수가 입력 행렬과 다릅니다.")
    if noise_fraction < 0:
        raise ValueError("noise_fraction은 음수일 수 없습니다.")
    safe_iqr = np.where(np.isfinite(iqr) & (iqr > 0), iqr, 0.0)
    perturbed = values + rng.normal(
        0.0, float(noise_fraction), size=values.shape
    ) * safe_iqr
    return np.clip(perturbed, lower, upper)


def build_masked_matrices(
    values: np.ndarray,
    shap_values: np.ndarray,
    replacement: np.ndarray,
    *,
    top_k: int,
    random_repeats: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]]:
    values = np.asarray(values, dtype=float)
    shap_values = np.asarray(shap_values, dtype=float)
    replacement = np.asarray(replacement, dtype=float)
    if values.shape != shap_values.shape or values.ndim != 2:
        raise ValueError("입력값과 SHAP 행렬의 크기가 같아야 합니다.")
    if replacement.shape != (values.shape[1],):
        raise ValueError("대체값의 변수 수가 다릅니다.")
    if not 1 <= int(top_k) <= values.shape[1] or random_repeats < 1:
        raise ValueError("top_k와 random_repeats를 확인하세요.")
    top_masked = values.copy()
    top_indices = []
    for row in range(len(values)):
        selected = top_k_indices(shap_values[row], top_k)
        top_indices.append(set(selected.tolist()))
        top_masked[row, selected] = replacement[selected]
    random_matrices = []
    all_indices = np.arange(values.shape[1])
    for _ in range(int(random_repeats)):
        masked = values.copy()
        for row in range(len(values)):
            alternatives = np.array(
                [index for index in all_indices if index not in top_indices[row]],
                dtype=int,
            )
            pool = alternatives if len(alternatives) >= top_k else all_indices
            selected = rng.choice(pool, size=top_k, replace=False)
            masked[row, selected] = replacement[selected]
        random_matrices.append(masked)
    return top_masked, random_matrices


def summarize_model_metrics(per_sample_rows: list[dict]) -> dict:
    if not per_sample_rows:
        raise ValueError("요약할 설명 신뢰성 행이 없습니다.")
    import pandas as pd

    frame = pd.DataFrame(per_sample_rows)
    summary = {
        "rows": int(frame["test_position"].nunique()),
        "stability_repeats": int(frame["repeat"].nunique()),
        "median_top10_jaccard": float(frame["top_k_jaccard"].median()),
        "p10_top10_jaccard": float(frame["top_k_jaccard"].quantile(0.10)),
        "median_top10_sign_agreement": float(
            frame["top_k_sign_agreement"].median()
        ),
        "median_absolute_rank_spearman": float(
            frame["absolute_rank_spearman"].median()
        ),
        "median_probability_change_under_noise": float(
            frame["probability_change_under_noise"].median()
        ),
        "prediction_flip_rate_under_noise": float(
            frame["prediction_flipped_under_noise"].mean()
        ),
        "median_top_shap_mask_effect": float(frame["top_shap_mask_effect"].median()),
        "median_random_mask_effect": float(frame["random_mask_effect"].median()),
        "median_fidelity_advantage": float(frame["fidelity_advantage"].median()),
        "top_shap_mask_win_rate": float(frame["top_shap_mask_wins"].mean()),
    }
    guardrails = {
        "median_top10_jaccard_at_least_0_70": summary["median_top10_jaccard"]
        >= 0.70,
        "median_sign_agreement_at_least_0_80": summary[
            "median_top10_sign_agreement"
        ]
        >= 0.80,
        "median_fidelity_advantage_positive": summary["median_fidelity_advantage"]
        > 0,
        "top_shap_mask_win_rate_above_0_50": summary["top_shap_mask_win_rate"]
        > 0.50,
    }
    summary["guardrails"] = guardrails
    summary["status"] = "PASS" if all(guardrails.values()) else "WARN"
    return summary
