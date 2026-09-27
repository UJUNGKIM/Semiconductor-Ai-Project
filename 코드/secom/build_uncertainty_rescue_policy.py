"""Build an OOF-selected safety-first review profile for SECOM."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from safety_policy import (
    budget_review_mask,
    locked_review_mask,
    model_alert_mask,
    normalized_risk_ratio,
    policy_metrics,
    score_cutoff_review_mask,
)


PROFILE = "balanced_f2"
TARGET_OOF_FALSE_NEGATIVE_CAPTURE = 0.70
BUDGET_GRID = tuple(np.round(np.arange(0.0, 1.001, 0.05), 2))
LOCKED_MARGIN = 0.25


def meta_features(cat_ratio: np.ndarray, xgb_ratio: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            cat_ratio,
            xgb_ratio,
            np.maximum(cat_ratio, xgb_ratio),
            np.minimum(cat_ratio, xgb_ratio),
            np.abs(cat_ratio - xgb_ratio),
            np.log1p(cat_ratio),
            np.log1p(xgb_ratio),
            cat_ratio * xgb_ratio,
        ]
    )


def repeated_meta_oof(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=10, random_state=42
    )
    probability_sum = np.zeros(len(y), dtype=float)
    prediction_count = np.zeros(len(y), dtype=int)
    for split_number, (fit_index, validation_index) in enumerate(cv.split(X, y)):
        model = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            random_state=42 + split_number,
        )
        model.fit(X[fit_index], y[fit_index])
        probability_sum[validation_index] += model.predict_proba(
            X[validation_index]
        )[:, 1]
        prediction_count[validation_index] += 1
    if not np.all(prediction_count == 10):
        raise RuntimeError("메타 모델 OOF 예측 횟수가 완전하지 않습니다.")
    return probability_sum / prediction_count


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 1.0, 1.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def plot_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    test = comparison.loc[comparison["dataset"] == "fixed test"].copy()
    labels = ["Standard", "Safety-first"]
    colors = ["#2878b5", "#d1495b"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axes[0].bar(labels, test["review_workload"], color=colors)
    axes[0].set_title("Manual-review workload")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Rate")

    values = test["false_negative_capture"].to_numpy(dtype=float)
    lower = test["fn_capture_ci_low"].to_numpy(dtype=float)
    upper = test["fn_capture_ci_high"].to_numpy(dtype=float)
    axes[1].bar(labels, values, color=colors)
    axes[1].errorbar(
        labels,
        values,
        yerr=np.vstack([values - lower, upper - values]),
        fmt="none",
        ecolor="#222222",
        capsize=5,
    )
    axes[1].set_title("False-negative capture (95% Wilson CI)")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle("SECOM standard versus OOF-selected safety-first policy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def with_interval(metrics: dict) -> dict:
    low, high = wilson_interval(
        int(metrics["captured_false_negatives"]),
        int(metrics["false_negatives"]),
    )
    return {**metrics, "fn_capture_ci_low": low, "fn_capture_ci_high": high}


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    model_dir = project_dir / "결과물" / "secom" / "dual_model_results"
    safety_dir = project_dir / "결과물" / "secom" / "safety_policy_results"
    output_dir = (
        project_dir / "결과물" / "secom" / "uncertainty_rescue_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] OOF 정상 합의 행의 미탐 후보를 구성합니다...")
    cat_bundle = joblib.load(model_dir / "catboost_model.joblib")
    xgb_bundle = joblib.load(model_dir / "xgboost_model.joblib")
    cat_threshold = float(cat_bundle["operating_thresholds"][PROFILE])
    xgb_threshold = float(xgb_bundle["operating_thresholds"][PROFILE])
    oof = pd.read_csv(model_dir / "oof_predictions.csv")
    y_oof = oof["label"].to_numpy(dtype=int)
    cat_oof = oof["catboost_probability"].to_numpy(dtype=float)
    xgb_oof = oof["xgboost_probability"].to_numpy(dtype=float)
    cat_alert = model_alert_mask(cat_oof, cat_threshold)
    xgb_alert = model_alert_mask(xgb_oof, xgb_threshold)
    normal_consensus = ~(cat_alert | xgb_alert)
    ratio_oof = normalized_risk_ratio(
        cat_oof, xgb_oof, cat_threshold, xgb_threshold
    )

    print("[2/5] 단순 위험도와 규제 메타 모델을 OOF에서 비교합니다...")
    candidate_y = y_oof[normal_consensus]
    cat_ratio = cat_oof[normal_consensus] / cat_threshold
    xgb_ratio = xgb_oof[normal_consensus] / xgb_threshold
    logistic_oof = repeated_meta_oof(
        meta_features(cat_ratio, xgb_ratio), candidate_y
    )
    heuristic_score = ratio_oof[normal_consensus]
    comparison_rows = [
        {
            "method": "normalized_max_ratio",
            "oof_average_precision": average_precision_score(
                candidate_y, heuristic_score
            ),
            "oof_roc_auc": roc_auc_score(candidate_y, heuristic_score),
            "deployable": True,
        },
        {
            "method": "regularized_logistic_meta",
            "oof_average_precision": average_precision_score(
                candidate_y, logistic_oof
            ),
            "oof_roc_auc": roc_auc_score(candidate_y, logistic_oof),
            "deployable": False,
        },
    ]
    method_comparison = pd.DataFrame(comparison_rows)
    selected_method = str(
        method_comparison.sort_values(
            ["oof_average_precision", "oof_roc_auc"], ascending=False
        ).iloc[0]["method"]
    )
    if selected_method != "normalized_max_ratio":
        raise RuntimeError(
            "메타 모델이 우세해졌습니다. 별도 외부 검증 전에는 자동 배포하지 않습니다."
        )
    method_comparison["selected"] = (
        method_comparison["method"] == selected_method
    )
    method_comparison.to_csv(output_dir / "method_comparison.csv", index=False)

    print("[3/5] OOF에서 목표 미탐 포착률의 최소 검토 예산을 선택합니다...")
    automated_oof = (cat_alert & xgb_alert).astype(int)
    budget_rows = []
    for budget in BUDGET_GRID:
        mask = budget_review_mask(
            cat_oof,
            xgb_oof,
            cat_threshold,
            xgb_threshold,
            float(budget),
        )
        metrics = policy_metrics(y_oof, automated_oof, mask)
        budget_rows.append({"additional_normal_budget": budget, **metrics})
    budget_frame = pd.DataFrame(budget_rows)
    eligible = budget_frame.loc[
        budget_frame["false_negative_capture"]
        >= TARGET_OOF_FALSE_NEGATIVE_CAPTURE
    ]
    if eligible.empty:
        raise RuntimeError("목표 미탐 포착률을 만족하는 OOF 검토 예산이 없습니다.")
    chosen_budget = float(eligible.iloc[0]["additional_normal_budget"])
    candidate_indices = np.flatnonzero(normal_consensus)
    extra_count = int(math.ceil(len(candidate_indices) * chosen_budget))
    ranked = candidate_indices[
        np.argsort(-ratio_oof[candidate_indices], kind="stable")
    ]
    risk_cutoff = float(ratio_oof[ranked[extra_count - 1]])
    oof_safety_mask = score_cutoff_review_mask(
        cat_oof,
        xgb_oof,
        cat_threshold,
        xgb_threshold,
        risk_cutoff,
    )
    oof_standard_mask = locked_review_mask(
        cat_oof,
        xgb_oof,
        cat_threshold,
        xgb_threshold,
        margin=LOCKED_MARGIN,
    )
    oof_standard = with_interval(
        policy_metrics(y_oof, automated_oof, oof_standard_mask)
    )
    oof_safety = with_interval(
        policy_metrics(y_oof, automated_oof, oof_safety_mask)
    )
    budget_frame.to_csv(output_dir / "oof_budget_selection.csv", index=False)

    print("[4/5] 잠긴 컷오프를 고정 test에 한 번 적용합니다...")
    test = pd.read_csv(safety_dir / "review_queue_test.csv").sort_values("test_row")
    y_test = test["label"].to_numpy(dtype=int)
    cat_test = test["catboost_probability"].to_numpy(dtype=float)
    xgb_test = test["xgboost_probability"].to_numpy(dtype=float)
    ood_status = test["ood_status"].to_numpy(dtype=str)
    cat_test_alert = model_alert_mask(cat_test, cat_threshold)
    xgb_test_alert = model_alert_mask(xgb_test, xgb_threshold)
    automated_test = (cat_test_alert & xgb_test_alert).astype(int)
    standard_test_mask = locked_review_mask(
        cat_test,
        xgb_test,
        cat_threshold,
        xgb_threshold,
        margin=LOCKED_MARGIN,
        ood_status=ood_status,
    )
    safety_test_mask = score_cutoff_review_mask(
        cat_test,
        xgb_test,
        cat_threshold,
        xgb_threshold,
        risk_cutoff,
        ood_status=ood_status,
    )
    test_standard = with_interval(
        policy_metrics(y_test, automated_test, standard_test_mask)
    )
    test_safety = with_interval(
        policy_metrics(y_test, automated_test, safety_test_mask)
    )

    test_queue = test.copy()
    test_queue["safety_first_review"] = safety_test_mask
    test_queue["newly_added_by_rescue"] = safety_test_mask & ~standard_test_mask
    test_queue.to_csv(output_dir / "safety_first_test_queue.csv", index=False)

    records = []
    for dataset, standard, safety in (
        ("OOF train", oof_standard, oof_safety),
        ("fixed test", test_standard, test_safety),
    ):
        records.append({"dataset": dataset, "policy": "standard", **standard})
        records.append({"dataset": dataset, "policy": "safety_first", **safety})
    policy_comparison = pd.DataFrame(records)
    policy_comparison.to_csv(output_dir / "policy_comparison.csv", index=False)
    plot_comparison(policy_comparison, output_dir / "policy_comparison.png")

    summary = {
        "version": 1,
        "profile": PROFILE,
        "test_used_for_policy_selection": False,
        "selection_source": "repeated_5x3_primary_model_oof_train_only",
        "candidate_rows": int(normal_consensus.sum()),
        "candidate_defects": int(candidate_y.sum()),
        "selected_method": selected_method,
        "rejected_method": "regularized_logistic_meta",
        "rejection_reason": "lower_repeated_oof_average_precision",
        "target_oof_false_negative_capture": TARGET_OOF_FALSE_NEGATIVE_CAPTURE,
        "chosen_additional_normal_budget": chosen_budget,
        "normalized_risk_cutoff": risk_cutoff,
        "thresholds": {
            "CatBoost": cat_threshold,
            "XGBoost": xgb_threshold,
        },
        "oof_standard": oof_standard,
        "oof_safety_first": oof_safety,
        "test_standard": test_standard,
        "test_safety_first": test_safety,
        "test_delta": {
            "review_workload": test_safety["review_workload"]
            - test_standard["review_workload"],
            "false_negative_capture": test_safety["false_negative_capture"]
            - test_standard["false_negative_capture"],
            "error_capture": test_safety["error_capture"]
            - test_standard["error_capture"],
            "auto_accuracy": test_safety["auto_accuracy"]
            - test_standard["auto_accuracy"],
        },
        "limitation": (
            "고정 test 불량이 21건, 자동판정 미탐이 9건뿐이므로 포착률 신뢰구간이 "
            "넓습니다. 개선은 탐색적 결과이며 외부 lot 전향 검증이 필요합니다."
        ),
    }
    (output_dir / "uncertainty_rescue_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = f"""# SECOM 미탐 예방 안전 우선 정책

- 선택 데이터: 반복 OOF train only (고정 test는 선택에 사용하지 않음)
- 정상 합의 후보: {int(normal_consensus.sum())}행, 실제 불량 {int(candidate_y.sum())}행
- 배포 점수: 두 모델 중 큰 정규화 위험도
- 기각한 대안: 규제 로지스틱 메타 모델(OOF 평균정밀도가 더 낮음)
- OOF 목표: 미탐 포착률 {TARGET_OOF_FALSE_NEGATIVE_CAPTURE:.0%} 이상
- 선택된 추가 검토 예산: 정상 합의 행의 {chosen_budget:.0%}
- 고정 위험도 컷오프: {risk_cutoff:.4f}

## 고정 test 확인

- 검토 업무량: {test_standard['review_workload']:.1%} → {test_safety['review_workload']:.1%}
- 미탐 포착률: {test_standard['false_negative_capture']:.1%} → {test_safety['false_negative_capture']:.1%}
- 전체 오분류 포착률: {test_standard['error_capture']:.1%} → {test_safety['error_capture']:.1%}
- 자동 처리 정확도: {test_standard['auto_accuracy']:.1%} → {test_safety['auto_accuracy']:.1%}
- 안전 우선 미탐 포착 95% Wilson 신뢰구간: {test_safety['fn_capture_ci_low']:.1%}~{test_safety['fn_capture_ci_high']:.1%}

고정 test 자동판정 미탐이 9건뿐이므로 신뢰구간이 넓습니다. 개선은 탐색적 결과이며
신규 장비·기간·lot의 전향적 외부 검증 전에는 실제 생산 성능으로 해석하면 안 됩니다.
"""
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")

    print("[5/5] 완료")
    print(
        f"  OOF 선택: 추가 정상 검토 {chosen_budget:.0%}, "
        f"위험도 컷오프 {risk_cutoff:.4f}"
    )
    print(
        f"  test 표준 → 안전우선: 업무량 "
        f"{test_standard['review_workload']:.1%} → {test_safety['review_workload']:.1%}, "
        f"미탐 포착 {test_standard['false_negative_capture']:.1%} → "
        f"{test_safety['false_negative_capture']:.1%}"
    )


if __name__ == "__main__":
    main()
