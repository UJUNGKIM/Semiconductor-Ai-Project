"""Chronological SECOM stress test with held-out probability calibration.

Timeline:
    first 60% -> fit preprocessing and classifier
    next 20%  -> select/finalize calibration and operating threshold
    last 20%  -> future-period evaluation only

The current CatBoost/XGBoost hyperparameters were selected in earlier random-CV
experiments spanning the full date range.  Consequently this is a retrospective
temporal stress test, not a pristine external validation.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from train_compare_models import RANDOM_STATE, load_data
from tune_weighted_models import build_tuned_model, fit_preprocessor


FIT_FRACTION = 0.60
CALIBRATION_FRACTION = 0.20
MIN_RECALL = 0.60
CALIBRATION_FOLDS = 5
BRIER_TIE_TOLERANCE = 0.001
MODELS = ("CatBoost", "XGBoost")
METHODS = ("uncalibrated", "sigmoid", "isotonic")
EPSILON = 1e-7


def fit_structural_filter(X_fit: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    high_missing = X_fit.columns[X_fit.isna().mean() >= 0.5].tolist()
    remaining = X_fit.drop(columns=high_missing)
    constants = remaining.columns[remaining.nunique(dropna=True) <= 1].tolist()
    kept = remaining.columns[~remaining.columns.isin(constants)].tolist()
    return kept, high_missing, constants


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "ece_10": expected_calibration_error(y_true, probabilities, 10),
        "mean_predicted_probability": float(np.mean(probabilities)),
        "observed_prevalence": float(np.mean(y_true)),
    }


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for bin_index in range(bins):
        mask = assignments == bin_index
        if np.any(mask):
            error += float(np.mean(mask)) * abs(
                float(np.mean(probabilities[mask])) - float(np.mean(y_true[mask]))
            )
    return float(error)


def fit_calibrator(method: str, probabilities: np.ndarray, y: np.ndarray):
    if method == "uncalibrated":
        return None
    if method == "sigmoid":
        scores = logit(np.clip(probabilities, EPSILON, 1.0 - EPSILON)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=RANDOM_STATE)
        calibrator.fit(scores, y)
        return calibrator
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(probabilities, y)
        return calibrator
    raise ValueError(method)


def apply_calibrator(method: str, calibrator, probabilities: np.ndarray) -> np.ndarray:
    if method == "uncalibrated":
        return probabilities.copy()
    if method == "sigmoid":
        scores = logit(np.clip(probabilities, EPSILON, 1.0 - EPSILON)).reshape(-1, 1)
        return calibrator.predict_proba(scores)[:, 1]
    if method == "isotonic":
        return np.asarray(calibrator.predict(probabilities), dtype=float)
    raise ValueError(method)


def cross_validated_calibration(
    method: str, raw_probabilities: np.ndarray, y: np.ndarray
) -> np.ndarray:
    if method == "uncalibrated":
        return raw_probabilities.copy()
    cv = StratifiedKFold(
        n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    calibrated = np.zeros_like(raw_probabilities, dtype=float)
    for fit_index, validation_index in cv.split(raw_probabilities, y):
        calibrator = fit_calibrator(
            method, raw_probabilities[fit_index], y[fit_index]
        )
        calibrated[validation_index] = apply_calibrator(
            method, calibrator, raw_probabilities[validation_index]
        )
    return calibrated


def select_f2_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    precision = precision[:-1]
    recall = recall[:-1]
    f2 = 5.0 * precision * recall / (4.0 * precision + recall + 1e-12)
    eligible = np.flatnonzero(recall >= MIN_RECALL)
    if eligible.size == 0:
        return float(thresholds[int(np.argmax(recall))])
    best = np.max(f2[eligible])
    tied = eligible[np.isclose(f2[eligible], best)]
    return float(np.max(thresholds[tied]))


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict:
    prediction = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f2": float(fbeta_score(y_true, prediction, beta=2, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def save_reliability_plot(
    output_dir: Path,
    y_future: np.ndarray,
    future_probabilities: dict[str, dict[str, np.ndarray]],
    selected_methods: dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    colors = {"uncalibrated": "#888888", "selected": "#d1495b"}
    for ax, model_name in zip(axes, MODELS):
        method = selected_methods[model_name]
        curves = (
            ("uncalibrated", future_probabilities[model_name]["uncalibrated"]),
            (method, future_probabilities[model_name][method]),
        )
        for position, (label, probabilities) in enumerate(curves):
            observed, predicted = calibration_curve(
                y_future, probabilities, n_bins=5, strategy="quantile"
            )
            display_label = label if position == 0 else f"selected: {method}"
            ax.plot(
                predicted,
                observed,
                marker="o",
                linewidth=2,
                color=colors["uncalibrated" if position == 0 else "selected"],
                label=display_label,
            )
        ax.plot([0, 1], [0, 1], "--", color="#333333", linewidth=1)
        ax.set_title(model_name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0.0, 0.20)
        ax.set_ylim(0.0, 0.20)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Observed defect rate")
    fig.suptitle("Future-period reliability (5 quantile bins)")
    fig.tight_layout()
    fig.savefig(output_dir / "future_reliability_diagram.png", dpi=190)
    plt.close(fig)


def save_temporal_prevalence_plot(
    output_dir: Path,
    timestamps: pd.Series,
    labels: pd.Series,
    fit_end: int,
    calibration_end: int,
) -> None:
    rolling = labels.rolling(window=100, min_periods=30).mean()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(timestamps, rolling, color="#6042a6", linewidth=2)
    ax.axvline(
        timestamps.iloc[fit_end], color="#2878b5", linestyle="--", label="calibration starts"
    )
    ax.axvline(
        timestamps.iloc[calibration_end],
        color="#d1495b",
        linestyle="--",
        label="future evaluation starts",
    )
    ax.set(
        xlabel="Timestamp",
        ylabel="Rolling defect rate",
        title="SECOM defect prevalence over time (100-sample rolling window)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "temporal_defect_prevalence.png", dpi=190)
    plt.close(fig)


def save_probability_plot(
    output_dir: Path,
    y_future: np.ndarray,
    future_probabilities: dict[str, dict[str, np.ndarray]],
    selected_methods: dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, model_name in zip(axes, MODELS):
        method = selected_methods[model_name]
        probabilities = future_probabilities[model_name][method]
        ax.hist(
            probabilities[y_future == 0],
            bins=20,
            alpha=0.7,
            color="#2878b5",
            label="normal",
        )
        ax.hist(
            probabilities[y_future == 1],
            bins=12,
            alpha=0.75,
            color="#d1495b",
            label="defect",
        )
        ax.set_title(f"{model_name} / {method}")
        ax.set_xlabel("Calibrated probability")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Count")
    fig.suptitle("Future-period calibrated probability distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "future_probability_distribution.png", dpi=190)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "결과물" / "secom" / "temporal_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/7] Loading and sorting SECOM by timestamp...")
    X_raw, y_raw, timestamps = load_data(project_dir)
    order = timestamps.sort_values(kind="stable").index
    X_sorted = X_raw.loc[order].reset_index(drop=False).rename(
        columns={"index": "original_row_index"}
    )
    y_sorted = y_raw.loc[order].reset_index(drop=True)
    time_sorted = timestamps.loc[order].reset_index(drop=True)
    original_indices = X_sorted.pop("original_row_index").to_numpy()
    fit_end = int(len(X_sorted) * FIT_FRACTION)
    calibration_end = int(
        len(X_sorted) * (FIT_FRACTION + CALIBRATION_FRACTION)
    )

    X_fit_raw = X_sorted.iloc[:fit_end].copy()
    X_cal_raw = X_sorted.iloc[fit_end:calibration_end].copy()
    X_future_raw = X_sorted.iloc[calibration_end:].copy()
    y_fit = y_sorted.iloc[:fit_end].copy()
    y_cal = y_sorted.iloc[fit_end:calibration_end].copy()
    y_future = y_sorted.iloc[calibration_end:].copy()
    print(
        f"  fit={len(y_fit)} ({int(y_fit.sum())} defects), "
        f"calibration={len(y_cal)} ({int(y_cal.sum())}), "
        f"future={len(y_future)} ({int(y_future.sum())})"
    )

    print("[2/7] Fitting missing/constant filtering on the earliest 60% only...")
    kept_features, high_missing, constants = fit_structural_filter(X_fit_raw)
    X_fit = X_fit_raw[kept_features]
    X_cal = X_cal_raw[kept_features]
    X_future = X_future_raw[kept_features]
    print(
        f"  590 -> {len(kept_features)} features "
        f"(high-missing={len(high_missing)}, constant={len(constants)})"
    )

    print("[3/7] Loading fixed finalist configurations...")
    configs = json.loads(
        (
            project_dir
            / "결과물"
            / "secom"
            / "dual_model_results"
            / "selected_configurations.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    calibration_rows = []
    future_rows = []
    bundle_records = {}
    future_probabilities: dict[str, dict[str, np.ndarray]] = {}
    selected_methods = {}

    print("[4/7] Training on the earliest period and producing calibration scores...")
    for model_name in MODELS:
        config = configs[model_name]
        fit_ready, cal_ready, imputer, selector, selected_features = fit_preprocessor(
            X_fit, y_fit, X_cal, config["top_k"]
        )
        future_ready = selector.transform(imputer.transform(X_future))
        ratio = float((y_fit == 0).sum() / (y_fit == 1).sum())
        model = build_tuned_model(
            config, ratio * float(config["weight_multiplier"]), RANDOM_STATE
        )
        model.fit(fit_ready, y_fit)
        raw_cal = model.predict_proba(cal_ready)[:, 1]
        raw_future = model.predict_proba(future_ready)[:, 1]

        print(f"[5/7] Comparing calibration methods for {model_name}...")
        cv_probabilities = {}
        fitted_calibrators = {}
        model_future_probabilities = {}
        for method in METHODS:
            cv_probability = cross_validated_calibration(
                method, raw_cal, y_cal.to_numpy()
            )
            calibrator = fit_calibrator(method, raw_cal, y_cal.to_numpy())
            fitted_calibration_probability = apply_calibrator(
                method, calibrator, raw_cal
            )
            future_probability = apply_calibrator(method, calibrator, raw_future)
            # Calibration CV is used only to choose the calibration method.  A
            # single final mapping is then fitted on the entire calibration
            # period, and the operating threshold is chosen on that consistent
            # probability scale.  The future period remains untouched.
            threshold = select_f2_threshold(
                y_cal.to_numpy(), fitted_calibration_probability
            )
            calibration_rows.append(
                {
                    "model": model_name,
                    "method": method,
                    **probability_metrics(y_cal.to_numpy(), cv_probability),
                    **classification_metrics(
                        y_cal.to_numpy(), fitted_calibration_probability, threshold
                    ),
                }
            )
            future_rows.append(
                {
                    "model": model_name,
                    "method": method,
                    **probability_metrics(y_future.to_numpy(), future_probability),
                    **classification_metrics(
                        y_future.to_numpy(), future_probability, threshold
                    ),
                }
            )
            cv_probabilities[method] = cv_probability
            fitted_calibrators[method] = calibrator
            model_future_probabilities[method] = future_probability

        model_calibration = pd.DataFrame(
            [row for row in calibration_rows if row["model"] == model_name]
        )
        best_brier = float(model_calibration["brier"].min())
        # With only 11 calibration defects, tiny Brier differences are noise-prone.
        # Among practically tied methods, prefer the lower log loss; this avoids
        # selecting a flexible isotonic mapping for a negligible Brier gain.
        competitive = model_calibration.loc[
            model_calibration["brier"] <= best_brier + BRIER_TIE_TOLERANCE
        ].sort_values(["log_loss", "brier"])
        selected_method = str(competitive.iloc[0]["method"])
        selected_methods[model_name] = selected_method
        future_probabilities[model_name] = model_future_probabilities
        selected_threshold = float(
            model_calibration.loc[
                model_calibration["method"] == selected_method, "threshold"
            ].iloc[0]
        )
        print(
            f"  selected={selected_method}, calibration CV "
            f"Brier={competitive.iloc[0]['brier']:.4f}, "
            f"threshold={selected_threshold:.4f}"
        )
        bundle = {
            "model": model,
            "calibrator": fitted_calibrators[selected_method],
            "calibration_method": selected_method,
            "imputer": imputer,
            "selector": selector,
            "threshold": selected_threshold,
            "model_name": model_name,
            "config": config,
            "input_features": kept_features,
            "selected_features": selected_features,
            "removed_high_missing_features": high_missing,
            "removed_constant_features": constants,
            "fit_period_end": str(time_sorted.iloc[fit_end - 1]),
            "calibration_period_end": str(time_sorted.iloc[calibration_end - 1]),
            "evaluation_note": (
                "Model fit on first 60%, calibrator/threshold on next 20%; "
                "last 20% reserved for retrospective temporal evaluation."
            ),
        }
        bundle_records[model_name] = bundle
        joblib.dump(
            bundle, output_dir / f"{model_name.lower()}_temporal_calibrated.joblib"
        )

    print("[6/7] Evaluating once on the last 20% future period...")
    calibration_table = pd.DataFrame(calibration_rows)
    future_table = pd.DataFrame(future_rows)
    calibration_table["selected"] = calibration_table.apply(
        lambda row: row["method"] == selected_methods[row["model"]], axis=1
    )
    future_table["selected"] = future_table.apply(
        lambda row: row["method"] == selected_methods[row["model"]], axis=1
    )
    selected_future = future_table.loc[future_table["selected"]]
    for _, row in selected_future.iterrows():
        print(
            f"  {row['model']}/{row['method']}: Brier={row['brier']:.4f}, "
            f"PR-AUC={row['pr_auc']:.4f}, precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, F2={row['f2']:.4f}"
        )

    print("[7/7] Saving reproducible artifacts and report...")
    prediction_frame = pd.DataFrame(
        {
            "original_row_index": original_indices[calibration_end:],
            "timestamp": time_sorted.iloc[calibration_end:].astype(str).to_numpy(),
            "label": y_future.to_numpy(),
        }
    )
    for model_name in MODELS:
        key = model_name.lower()
        method = selected_methods[model_name]
        probability = future_probabilities[model_name][method]
        threshold = float(bundle_records[model_name]["threshold"])
        prediction_frame[f"{key}_calibration_method"] = method
        prediction_frame[f"{key}_probability"] = probability
        prediction_frame[f"{key}_threshold"] = threshold
        prediction_frame[f"{key}_prediction"] = (probability >= threshold).astype(int)

    split_summary = {
        "total_rows": len(X_sorted),
        "fit": {
            "rows": len(y_fit),
            "defects": int(y_fit.sum()),
            "start": str(time_sorted.iloc[0]),
            "end": str(time_sorted.iloc[fit_end - 1]),
        },
        "calibration": {
            "rows": len(y_cal),
            "defects": int(y_cal.sum()),
            "start": str(time_sorted.iloc[fit_end]),
            "end": str(time_sorted.iloc[calibration_end - 1]),
        },
        "future": {
            "rows": len(y_future),
            "defects": int(y_future.sum()),
            "start": str(time_sorted.iloc[calibration_end]),
            "end": str(time_sorted.iloc[-1]),
        },
        "features": {
            "raw": X_raw.shape[1],
            "kept": len(kept_features),
            "removed_high_missing": len(high_missing),
            "removed_constant": len(constants),
        },
        "selected_calibration_methods": selected_methods,
        "limitation": (
            "Retrospective temporal stress test: hyperparameters were selected in "
            "earlier random-CV experiments spanning the full timestamp range."
        ),
    }
    calibration_table.to_csv(output_dir / "calibration_cv_metrics.csv", index=False)
    future_table.to_csv(output_dir / "future_metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "future_predictions.csv", index=False)
    (output_dir / "split_summary.json").write_text(
        json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_reliability_plot(
        output_dir, y_future.to_numpy(), future_probabilities, selected_methods
    )
    save_probability_plot(
        output_dir, y_future.to_numpy(), future_probabilities, selected_methods
    )
    save_temporal_prevalence_plot(
        output_dir, time_sorted, y_sorted, fit_end, calibration_end
    )

    lines = [
        "# SECOM 시간순 검증 및 확률 보정",
        "",
        "## 시간 분할",
        "",
        f"- 모델 학습: {len(y_fit)}개 / 불량 {int(y_fit.sum())}개 / "
        f"{time_sorted.iloc[0]} ~ {time_sorted.iloc[fit_end - 1]}",
        f"- 보정·임계값 결정: {len(y_cal)}개 / 불량 {int(y_cal.sum())}개 / "
        f"{time_sorted.iloc[fit_end]} ~ {time_sorted.iloc[calibration_end - 1]}",
        f"- 미래 평가: {len(y_future)}개 / 불량 {int(y_future.sum())}개 / "
        f"{time_sorted.iloc[calibration_end]} ~ {time_sorted.iloc[-1]}",
        f"- 구조 필터: 590 → {len(kept_features)}개 "
        f"(결측률 제거 {len(high_missing)}, 상수 제거 {len(constants)})",
        "",
        "## 보정 구간 교차검증에서 선택한 방법",
        "",
        f"Brier 차이가 {BRIER_TIE_TOLERANCE:.3f} 이내이면 작은 보정 표본의 "
        "과적합을 피하기 위해 log loss가 낮은 방법을 선택했습니다.",
        "",
        "| 모델 | 선택 방법 | Brier | Log loss | 평균 예측확률 | 실제 불량률 | 임계값 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        row = calibration_table.loc[
            (calibration_table["model"] == model_name)
            & calibration_table["selected"]
        ].iloc[0]
        lines.append(
            f"| {model_name} | {row['method']} | {row['brier']:.4f} | "
            f"{row['log_loss']:.4f} | {row['mean_predicted_probability']:.4f} | "
            f"{row['observed_prevalence']:.4f} | {row['threshold']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 마지막 20% 미래 구간 결과",
            "",
            "| 모델 | 보정 | PR-AUC | ROC-AUC | Brier | 평균 확률 | 실제 불량률 | Precision | Recall | F2 | TP/FP/FN/TN |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in selected_future.iterrows():
        lines.append(
            f"| {row['model']} | {row['method']} | {row['pr_auc']:.4f} | "
            f"{row['roc_auc']:.4f} | {row['brier']:.4f} | "
            f"{row['mean_predicted_probability']:.4f} | "
            f"{row['observed_prevalence']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f2']:.4f} | "
            f"{int(row['tp'])}/{int(row['fp'])}/{int(row['fn'])}/{int(row['tn'])} |"
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "기존 하이퍼파라미터는 전체 날짜 범위를 포함한 이전 랜덤 교차검증에서 선택됐습니다.",
            "따라서 이번 분석은 시간 변화에 대한 후향적 스트레스 테스트이며, 완전히 새로운",
            "외부 데이터 검증을 대체하지 않습니다. 보정 구간의 불량도 11개뿐이라 보정 방법의",
            "선택 불확실성이 큽니다.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  results={output_dir}")


if __name__ == "__main__":
    main()
