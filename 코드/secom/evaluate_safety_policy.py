"""Evaluate the locked SECOM human-review policy without test leakage."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from advanced_diagnosis import assess_ood
from safety_policy import (
    budget_review_mask,
    locked_review_mask,
    model_alert_mask,
    normalized_risk_ratio,
    policy_metrics,
)
from train_compare_models import RANDOM_STATE, TEST_SIZE, load_data, structural_filter


MODEL_NAMES = ("CatBoost", "XGBoost")
PROFILE = "balanced_f2"
LOCKED_MARGIN = 0.25
BUDGETS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0)


def model_probability(X: pd.DataFrame, bundle: dict) -> np.ndarray:
    values = bundle["imputer"].transform(X[bundle["input_features"]])
    ready = bundle["selector"].transform(values)
    return np.asarray(bundle["model"].predict_proba(ready)[:, 1], dtype=float)


def evaluate_dataset(
    *,
    dataset: str,
    y: np.ndarray,
    cat_probability: np.ndarray,
    xgb_probability: np.ndarray,
    cat_threshold: float,
    xgb_threshold: float,
    ood_status: np.ndarray | None,
) -> tuple[dict, list[dict]]:
    automated = (
        model_alert_mask(cat_probability, cat_threshold)
        & model_alert_mask(xgb_probability, xgb_threshold)
    ).astype(int)
    locked = locked_review_mask(
        cat_probability,
        xgb_probability,
        cat_threshold,
        xgb_threshold,
        margin=LOCKED_MARGIN,
        ood_status=ood_status,
    )
    locked_metrics = policy_metrics(y, automated, locked)
    locked_metrics.update(
        {
            "dataset": dataset,
            "policy": "locked_25pct_boundary",
            "additional_normal_budget": None,
        }
    )

    curve = []
    for budget in BUDGETS:
        review = budget_review_mask(
            cat_probability,
            xgb_probability,
            cat_threshold,
            xgb_threshold,
            budget,
            ood_status=ood_status,
        )
        row = policy_metrics(y, automated, review)
        row.update(
            {
                "dataset": dataset,
                "policy": "risk_rank_budget",
                "additional_normal_budget": budget,
            }
        )
        curve.append(row)
    return locked_metrics, curve


def review_reason(
    cat_alert: bool,
    xgb_alert: bool,
    risk_ratio: float,
    ood_status: str,
) -> str:
    if ood_status == "out_of_distribution":
        return "분포 이탈 · 자동판정 보류"
    if cat_alert and xgb_alert:
        return "두 모델 모두 불량 경보"
    if cat_alert or xgb_alert:
        return "모델 불일치"
    if risk_ratio >= 1.0 - LOCKED_MARGIN:
        return "정상 합의이나 임계값 경계"
    return "자동 처리 후보"


def build_plot(frame: pd.DataFrame, output_path: Path) -> None:
    colors = {"OOF train": "#2878b5", "fixed test": "#d1495b"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for dataset, rows in frame.groupby("dataset"):
        rows = rows.sort_values("review_workload")
        axes[0].plot(
            rows["review_workload"],
            rows["error_capture"],
            marker="o",
            label=dataset,
            color=colors[dataset],
        )
        axes[1].plot(
            rows["review_workload"],
            rows["false_negative_capture"],
            marker="o",
            label=dataset,
            color=colors[dataset],
        )
    axes[0].set_title("All classification errors")
    axes[1].set_title("False negatives only")
    for axis in axes:
        axis.set_xlabel("Manual-review workload")
        axis.set_ylabel("Capture rate")
        axis.set_xlim(0, 1.01)
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("SECOM review workload versus safety capture")
    fig.tight_layout()
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    model_dir = project_dir / "결과물" / "secom" / "dual_model_results"
    advanced_dir = project_dir / "결과물" / "secom" / "advanced_diagnostics"
    output_dir = project_dir / "결과물" / "secom" / "safety_policy_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] OOF 예측과 잠긴 운영 임계값을 불러옵니다...")
    bundles = {
        name: joblib.load(model_dir / f"{name.lower()}_model.joblib")
        for name in MODEL_NAMES
    }
    thresholds = {
        name: float(bundles[name]["operating_thresholds"][PROFILE])
        for name in MODEL_NAMES
    }
    oof = pd.read_csv(model_dir / "oof_predictions.csv")
    oof_locked, oof_curve = evaluate_dataset(
        dataset="OOF train",
        y=oof["label"].to_numpy(dtype=int),
        cat_probability=oof["catboost_probability"].to_numpy(dtype=float),
        xgb_probability=oof["xgboost_probability"].to_numpy(dtype=float),
        cat_threshold=thresholds["CatBoost"],
        xgb_threshold=thresholds["XGBoost"],
        ood_status=None,
    )

    print("[2/5] random_state=42 고정 테스트셋을 재구성합니다...")
    X_raw, y, _ = load_data(project_dir)
    X_filtered, _, _ = structural_filter(X_raw)
    if X_filtered.shape[1] != 446:
        raise RuntimeError(f"전처리 변수 수가 446개가 아닙니다: {X_filtered.shape[1]}")
    _, X_test, _, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    source_rows = X_test.index.to_numpy(dtype=int)
    X_test = X_test.reset_index(drop=True)
    y_array = y_test.to_numpy(dtype=int)

    print("[3/5] 두 모델 예측과 train-only OOD 검사를 실행합니다...")
    cat_probability = model_probability(X_test, bundles["CatBoost"])
    xgb_probability = model_probability(X_test, bundles["XGBoost"])
    reference = joblib.load(advanced_dir / "advanced_reference.joblib")
    ood = assess_ood(X_test, reference)
    ood_status = ood["OOD 상태"].to_numpy(dtype=str)
    test_locked, test_curve = evaluate_dataset(
        dataset="fixed test",
        y=y_array,
        cat_probability=cat_probability,
        xgb_probability=xgb_probability,
        cat_threshold=thresholds["CatBoost"],
        xgb_threshold=thresholds["XGBoost"],
        ood_status=ood_status,
    )

    print("[4/5] 대기열과 검토량-안전성 곡선을 저장합니다...")
    ratio = normalized_risk_ratio(
        cat_probability,
        xgb_probability,
        thresholds["CatBoost"],
        thresholds["XGBoost"],
    )
    cat_alert = model_alert_mask(cat_probability, thresholds["CatBoost"])
    xgb_alert = model_alert_mask(xgb_probability, thresholds["XGBoost"])
    automated = (cat_alert & xgb_alert).astype(int)
    locked = locked_review_mask(
        cat_probability,
        xgb_probability,
        thresholds["CatBoost"],
        thresholds["XGBoost"],
        margin=LOCKED_MARGIN,
        ood_status=ood_status,
    )
    queue = pd.DataFrame(
        {
            "test_row": np.arange(len(X_test), dtype=int),
            "source_row_index": source_rows,
            "label": y_array,
            "catboost_probability": cat_probability,
            "xgboost_probability": xgb_probability,
            "normalized_risk_ratio": ratio,
            "catboost_alert": cat_alert,
            "xgboost_alert": xgb_alert,
            "automated_prediction": automated,
            "ood_status": ood_status,
            "locked_review": locked,
            "automated_correct": automated == y_array,
        }
    )
    queue["review_reason"] = [
        review_reason(cat, xgb, risk, status)
        for cat, xgb, risk, status in zip(
            cat_alert, xgb_alert, ratio, ood_status
        )
    ]
    queue = queue.sort_values(
        ["locked_review", "normalized_risk_ratio"], ascending=False
    )
    queue.to_csv(output_dir / "review_queue_test.csv", index=False)

    tradeoff = pd.DataFrame(oof_curve + test_curve)
    tradeoff.to_csv(output_dir / "workload_tradeoff.csv", index=False)
    build_plot(tradeoff, output_dir / "safety_policy_curve.png")

    summary = {
        "version": 1,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "model_threshold_source": "repeated_5x3_oof_train_only",
        "policy_selection_uses_test_labels": False,
        "automated_defect_rule": "both_models_above_threshold",
        "locked_review_rule": (
            "either_model_alert OR normalized_risk_ratio>=0.75 "
            "OR severe_OOD"
        ),
        "locked_boundary_margin": LOCKED_MARGIN,
        "thresholds": thresholds,
        "oof_locked_policy": oof_locked,
        "test_locked_policy": test_locked,
        "test_ood_counts": {
            key: int(value)
            for key, value in pd.Series(ood_status).value_counts().items()
        },
        "interpretation": (
            "검토된 행을 전문가가 올바르게 판정한다고 가정한 정책 시뮬레이션이며, "
            "실제 생산 성능은 전향적 외부 검증이 필요합니다."
        ),
    }
    (output_dir / "safety_policy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    test = test_locked
    markdown = f"""# SECOM 전문가 검토 정책 검증

- 임계값 출처: 반복 5-fold OOF(train only), 테스트 라벨로 정책 조정하지 않음
- 자동 불량 판정: CatBoost와 XGBoost가 모두 임계값 이상
- 잠긴 검토 규칙: 한 모델 이상 경보, 정상 합의 중 임계값 25% 경계, severe OOD
- 고정 test: {test['rows']}행, 검토 {test['reviewed_rows']}행 ({test['review_workload']:.1%})
- 전체 오분류 포착: {test['captured_errors']}/{test['errors']} ({test['error_capture']:.1%})
- 미탐 포착: {test['captured_false_negatives']}/{test['false_negatives']} ({test['false_negative_capture']:.1%})
- 자동 처리: {test['auto_rows']}행, 자동 처리 정확도 {test['auto_accuracy']:.1%}

`workload_tradeoff.csv`는 경보 행을 기본 검토하고, 나머지 정상 합의 행 중 위험도
상위 비율을 추가 검토할 때의 업무량과 포착률을 OOF/test별로 보여줍니다. test 곡선은
사후 설명용이며 정책 선택에 사용하면 안 됩니다.

검토된 행을 전문가가 올바르게 판정한다고 가정한 정책 시뮬레이션입니다. 실제 생산
성능을 의미하지 않으며 신규 장비·기간·lot에 대한 전향적 외부 검증이 필요합니다.
"""
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")

    print("[5/5] 완료")
    print(
        f"  test 검토량={test['review_workload']:.1%}, "
        f"오분류 포착={test['error_capture']:.1%}, "
        f"미탐 포착={test['false_negative_capture']:.1%}, "
        f"자동 처리 정확도={test['auto_accuracy']:.1%}"
    )
    print(f"  결과: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"안전 정책 평가 실패: {exc}", file=sys.stderr)
        raise
