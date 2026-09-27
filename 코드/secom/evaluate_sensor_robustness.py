"""Stress-test fixed SECOM models against realistic sensor corruption."""

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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from advanced_diagnosis import assess_ood
from safety_policy import (
    locked_review_mask,
    model_alert_mask,
    policy_metrics,
    score_cutoff_review_mask,
)
from train_compare_models import RANDOM_STATE, TEST_SIZE, load_data


PROFILE = "balanced_f2"
REPLICATES = 5
SPIKE_MAGNITUDE_IQR = 5.0
MAX_RECALL_DROP = 0.20
MAX_FLIP_RATE = 0.20
MIN_SAFETY_AUTO_ACCURACY = 0.90
SCENARIOS = (
    ("random_missing", 0.05, "무작위 결측 5%"),
    ("random_missing", 0.10, "무작위 결측 10%"),
    ("random_missing", 0.20, "무작위 결측 20%"),
    ("gaussian_noise", 0.25, "가우시안 노이즈 0.25 IQR"),
    ("gaussian_noise", 0.50, "가우시안 노이즈 0.50 IQR"),
    ("gaussian_noise", 1.00, "가우시안 노이즈 1.00 IQR"),
    ("sensor_spike", 0.01, "센서 스파이크 1% · 5 IQR"),
    ("sensor_spike", 0.05, "센서 스파이크 5% · 5 IQR"),
    ("column_dropout", 0.05, "센서 열 드롭아웃 5%"),
    ("column_dropout", 0.10, "센서 열 드롭아웃 10%"),
)


def perturb_sensors(
    X_raw: pd.DataFrame,
    reference: dict,
    kind: str,
    severity: float,
    seed: int,
) -> pd.DataFrame:
    """Perturb retained sensors using train-only IQR values."""
    if kind not in {
        "random_missing",
        "gaussian_noise",
        "sensor_spike",
        "column_dropout",
    }:
        raise ValueError(f"지원하지 않는 교란 종류입니다: {kind}")
    if severity <= 0:
        raise ValueError("교란 강도는 0보다 커야 합니다.")

    rng = np.random.default_rng(seed)
    corrupted = X_raw.copy(deep=True)
    features = list(reference["input_features"])
    values = corrupted[features].to_numpy(dtype=float, copy=True)
    observed = np.isfinite(values)
    iqr = np.maximum(np.asarray(reference["iqr"], dtype=float), 1e-12)

    if kind == "random_missing":
        mask = (rng.random(values.shape) < severity) & observed
        values[mask] = np.nan
    elif kind == "gaussian_noise":
        noise = rng.normal(0.0, severity, size=values.shape) * iqr
        values[observed] += noise[observed]
    elif kind == "sensor_spike":
        mask = (rng.random(values.shape) < severity) & observed
        signs = rng.choice((-1.0, 1.0), size=values.shape)
        offsets = signs * SPIKE_MAGNITUDE_IQR * iqr
        values[mask] += np.broadcast_to(offsets, values.shape)[mask]
    else:
        column_count = int(math.ceil(len(features) * severity))
        selected = rng.choice(len(features), size=column_count, replace=False)
        values[:, selected] = np.nan

    corrupted.loc[:, features] = values
    return corrupted


def model_probability(X: pd.DataFrame, bundle: dict) -> np.ndarray:
    values = bundle["imputer"].transform(X[bundle["input_features"]])
    ready = bundle["selector"].transform(values)
    return np.asarray(bundle["model"].predict_proba(ready)[:, 1], dtype=float)


def evaluate_once(
    X: pd.DataFrame,
    y: np.ndarray,
    bundles: dict[str, dict],
    thresholds: dict[str, float],
    reference: dict,
    risk_cutoff: float,
    baseline_prediction: np.ndarray,
    baseline_cat_probability: np.ndarray,
    baseline_xgb_probability: np.ndarray,
) -> dict[str, float | int]:
    cat_probability = model_probability(X, bundles["CatBoost"])
    xgb_probability = model_probability(X, bundles["XGBoost"])
    cat_alert = model_alert_mask(cat_probability, thresholds["CatBoost"])
    xgb_alert = model_alert_mask(xgb_probability, thresholds["XGBoost"])
    prediction = (cat_alert & xgb_alert).astype(int)
    score = np.minimum(
        cat_probability / thresholds["CatBoost"],
        xgb_probability / thresholds["XGBoost"],
    )
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()

    ood = assess_ood(X, reference)
    ood_status = ood["OOD 상태"].to_numpy(dtype=str)
    standard_mask = locked_review_mask(
        cat_probability,
        xgb_probability,
        thresholds["CatBoost"],
        thresholds["XGBoost"],
        margin=0.25,
        ood_status=ood_status,
    )
    safety_mask = score_cutoff_review_mask(
        cat_probability,
        xgb_probability,
        thresholds["CatBoost"],
        thresholds["XGBoost"],
        risk_cutoff,
        ood_status=ood_status,
    )
    standard = policy_metrics(y, prediction, standard_mask)
    safety = policy_metrics(y, prediction, safety_mask)
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f2": float(fbeta_score(y, prediction, beta=2, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "prediction_flip_rate": float((prediction != baseline_prediction).mean()),
        "model_disagreement_rate": float((cat_alert != xgb_alert).mean()),
        "cat_probability_mae": float(
            np.abs(cat_probability - baseline_cat_probability).mean()
        ),
        "xgb_probability_mae": float(
            np.abs(xgb_probability - baseline_xgb_probability).mean()
        ),
        "ood_review_rate": float((ood_status != "in_distribution").mean()),
        "ood_severe_rate": float(
            (ood_status == "out_of_distribution").mean()
        ),
        "standard_review_workload": float(standard["review_workload"]),
        "standard_false_negative_capture": float(
            standard["false_negative_capture"]
        ),
        "standard_auto_accuracy": float(standard["auto_accuracy"]),
        "safety_review_workload": float(safety["review_workload"]),
        "safety_false_negative_capture": float(safety["false_negative_capture"]),
        "safety_auto_accuracy": float(safety["auto_accuracy"]),
    }


def aggregate_results(raw: pd.DataFrame, baseline_recall: float) -> pd.DataFrame:
    metric_columns = [
        column
        for column in raw.columns
        if column not in {"scenario", "kind", "severity", "replicate"}
    ]
    rows = []
    for keys, group in raw.groupby(
        ["scenario", "kind", "severity"], sort=False, dropna=False
    ):
        row = {
            "scenario": keys[0],
            "kind": keys[1],
            "severity": keys[2],
            "replicates": len(group),
        }
        for column in metric_columns:
            values = group[column].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_ci_low"] = mean - half
            row[f"{column}_ci_high"] = mean + half
        row["recall_drop"] = baseline_recall - row["recall_mean"]
        row["guardrail_pass"] = bool(
            row["recall_drop"] <= MAX_RECALL_DROP
            and row["prediction_flip_rate_mean"] <= MAX_FLIP_RATE
            and row["safety_auto_accuracy_mean"] >= MIN_SAFETY_AUTO_ACCURACY
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_dashboard(summary: pd.DataFrame, output_path: Path) -> None:
    def compact_label(row: pd.Series) -> str:
        severity = float(row["severity"])
        if row["kind"] == "baseline":
            return "Baseline"
        if row["kind"] == "random_missing":
            return f"Missing {severity:.0%}"
        if row["kind"] == "gaussian_noise":
            return f"Noise {severity:.2f} IQR"
        if row["kind"] == "sensor_spike":
            return f"Spike {severity:.0%}"
        return f"Column loss {severity:.0%}"

    labels = [compact_label(row) for _, row in summary.iterrows()]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panels = (
        ("recall_mean", "Consensus recall", (0, 1)),
        ("prediction_flip_rate_mean", "Prediction flip rate", (0, 1)),
        ("ood_review_rate_mean", "OOD review-or-hold rate", (0, 1)),
        ("safety_auto_accuracy_mean", "Safety-first auto accuracy", (0, 1)),
    )
    for axis, (column, title, limits) in zip(axes.flat, panels):
        colors = [
            "#2878b5" if passed else "#d1495b"
            for passed in summary["guardrail_pass"]
        ]
        axis.bar(x, summary[column], color=colors)
        axis.set_title(title)
        axis.set_ylim(*limits)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle("SECOM fixed-test sensor corruption stress test")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    model_dir = project_dir / "결과물" / "secom" / "dual_model_results"
    advanced_dir = project_dir / "결과물" / "secom" / "advanced_diagnostics"
    rescue_dir = project_dir / "결과물" / "secom" / "uncertainty_rescue_results"
    output_dir = project_dir / "결과물" / "secom" / "robustness_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 고정 test와 train-only 교란 기준을 불러옵니다...")
    X_raw, y, _ = load_data(project_dir)
    _, X_test, _, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_test = X_test.reset_index(drop=True)
    y_array = y_test.to_numpy(dtype=int)
    bundles = {
        name: joblib.load(model_dir / f"{name.lower()}_model.joblib")
        for name in ("CatBoost", "XGBoost")
    }
    thresholds = {
        name: float(bundles[name]["operating_thresholds"][PROFILE])
        for name in bundles
    }
    reference = joblib.load(advanced_dir / "advanced_reference.joblib")
    rescue = json.loads(
        (rescue_dir / "uncertainty_rescue_summary.json").read_text(encoding="utf-8")
    )
    risk_cutoff = float(rescue["normalized_risk_cutoff"])

    baseline_cat = model_probability(X_test, bundles["CatBoost"])
    baseline_xgb = model_probability(X_test, bundles["XGBoost"])
    baseline_prediction = (
        model_alert_mask(baseline_cat, thresholds["CatBoost"])
        & model_alert_mask(baseline_xgb, thresholds["XGBoost"])
    ).astype(int)
    baseline = evaluate_once(
        X_test,
        y_array,
        bundles,
        thresholds,
        reference,
        risk_cutoff,
        baseline_prediction,
        baseline_cat,
        baseline_xgb,
    )
    rows = [
        {
            "scenario": "원본 기준",
            "kind": "baseline",
            "severity": 0.0,
            "replicate": 0,
            **baseline,
        }
    ]

    print(f"[2/5] {len(SCENARIOS)}개 교란 조건 × {REPLICATES}회 평가합니다...")
    for scenario_number, (kind, severity, label) in enumerate(SCENARIOS, 1):
        for replicate in range(REPLICATES):
            corrupted = perturb_sensors(
                X_test,
                reference,
                kind,
                severity,
                seed=RANDOM_STATE + scenario_number * 100 + replicate,
            )
            metrics = evaluate_once(
                corrupted,
                y_array,
                bundles,
                thresholds,
                reference,
                risk_cutoff,
                baseline_prediction,
                baseline_cat,
                baseline_xgb,
            )
            rows.append(
                {
                    "scenario": label,
                    "kind": kind,
                    "severity": severity,
                    "replicate": replicate + 1,
                    **metrics,
                }
            )
        print(f"  {scenario_number:02d}/{len(SCENARIOS)} {label}")

    print("[3/5] 반복 결과와 95% 신뢰구간을 집계합니다...")
    raw = pd.DataFrame(rows)
    aggregate = aggregate_results(raw, baseline_recall=float(baseline["recall"]))
    raw.to_csv(output_dir / "stress_test_replicates.csv", index=False)
    aggregate.to_csv(output_dir / "stress_test_summary.csv", index=False)

    print("[4/5] 대시보드 그림과 검증 메타데이터를 저장합니다...")
    plot_dashboard(aggregate, output_dir / "robustness_dashboard.png")
    stressed = aggregate.loc[aggregate["kind"] != "baseline"]
    worst = stressed.sort_values("recall_mean").iloc[0]
    failed = stressed.loc[~stressed["guardrail_pass"]]
    metadata = {
        "version": 1,
        "random_state": RANDOM_STATE,
        "test_rows": len(X_test),
        "test_defects": int(y_array.sum()),
        "replicates_per_scenario": REPLICATES,
        "test_used_for_model_or_threshold_selection": False,
        "perturbation_statistics_source": "fixed_train_only_iqr",
        "scenarios_predeclared": True,
        "spike_magnitude_iqr": SPIKE_MAGNITUDE_IQR,
        "guardrails": {
            "maximum_recall_drop": MAX_RECALL_DROP,
            "maximum_prediction_flip_rate": MAX_FLIP_RATE,
            "minimum_safety_auto_accuracy": MIN_SAFETY_AUTO_ACCURACY,
        },
        "baseline": baseline,
        "scenario_count": len(SCENARIOS),
        "guardrail_pass_count": int(stressed["guardrail_pass"].sum()),
        "guardrail_fail_count": int((~stressed["guardrail_pass"]).sum()),
        "failed_scenarios": failed["scenario"].tolist(),
        "worst_recall_scenario": str(worst["scenario"]),
        "worst_recall_mean": float(worst["recall_mean"]),
        "limitation": (
            "합성 센서 교란은 실제 장비 고장 분포를 재현하지 않습니다. 고정 test의 "
            "민감도 분석이며 신규 장비·lot 전향 검증을 대체하지 않습니다."
        ),
    }
    (output_dir / "robustness_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = f"""# SECOM 센서 오류 강건성 스트레스 테스트

- 고정 test: {len(X_test)}행, 불량 {int(y_array.sum())}행
- 교란 조건: {len(SCENARIOS)}개, 조건당 {REPLICATES}회
- 교란 크기 기준: fixed train에서 계산한 IQR만 사용
- 모델·임계값·안전정책 선택에 test 결과를 사용하지 않음
- 사전 정의 guardrail 통과: {metadata['guardrail_pass_count']}/{len(SCENARIOS)}개
- 최저 재현율 조건: {metadata['worst_recall_scenario']} ({metadata['worst_recall_mean']:.1%})

교란은 무작위 결측, 전체 센서 열 드롭아웃, IQR 기반 가우시안 노이즈, 5 IQR
센서 스파이크를 포함합니다. `stress_test_summary.csv`에는 성능, 판정 뒤집힘, OOD
감지, 표준·안전 우선 검토 정책의 평균과 95% 신뢰구간이 저장됩니다.

합성 센서 교란은 실제 장비 고장 분포를 재현하지 않으며 신규 장비·lot 전향 검증을
대체하지 않습니다.
"""
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")

    print("[5/5] 완료")
    print(
        f"  guardrail 통과={metadata['guardrail_pass_count']}/{len(SCENARIOS)}, "
        f"최저 재현율={metadata['worst_recall_scenario']} "
        f"{metadata['worst_recall_mean']:.1%}"
    )
    print(f"  결과: {output_dir}")


if __name__ == "__main__":
    main()
