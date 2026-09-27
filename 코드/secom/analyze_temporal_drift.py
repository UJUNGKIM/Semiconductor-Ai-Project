"""Analyze sensor, missingness, model-score, and explanation drift in SECOM."""

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
from scipy.stats import ks_2samp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diagnose_secom import shap_matrix
from temporal_calibration_evaluation import apply_calibrator
from train_compare_models import RANDOM_STATE, load_data


FIT_FRACTION = 0.60
CALIBRATION_FRACTION = 0.20
PSI_BINS = 10
MODELS = ("CatBoost", "XGBoost")


def population_stability_index(reference: pd.Series, current: pd.Series) -> float:
    """PSI using reference-derived quantile bins plus an explicit missing bin."""
    ref = reference.dropna().to_numpy(dtype=float)
    cur = current.dropna().to_numpy(dtype=float)
    if ref.size == 0 or np.unique(ref).size <= 1:
        return 0.0
    quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, PSI_BINS + 1)))
    if quantiles.size <= 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges = quantiles.copy()
        edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_counts = np.append(ref_counts, reference.isna().sum())
    cur_counts = np.append(cur_counts, current.isna().sum())
    ref_share = np.clip(ref_counts / len(reference), 1e-6, None)
    cur_share = np.clip(cur_counts / len(current), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def standardized_median_shift(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        return 0.0
    q1, q3 = np.quantile(ref, [0.25, 0.75])
    scale = float(q3 - q1)
    if scale <= 1e-12:
        scale = float(np.std(ref))
    if scale <= 1e-12:
        return 0.0
    return float(abs(np.median(cur) - np.median(ref)) / scale)


def severity(psi: float, missing_delta: float) -> str:
    if psi >= 0.25 or abs(missing_delta) >= 0.10:
        return "high"
    if psi >= 0.10 or abs(missing_delta) >= 0.05:
        return "moderate"
    return "low"


def future_shap_importance(
    bundle: dict, X_future: pd.DataFrame
) -> tuple[dict[str, float], dict[str, int]]:
    X_model = X_future[bundle["input_features"]]
    X_imputed = bundle["imputer"].transform(X_model)
    X_ready = bundle["selector"].transform(X_imputed)
    values = shap_matrix(bundle["model"], X_ready)
    importance = np.mean(np.abs(values), axis=0)
    normalized = importance / max(float(importance.sum()), 1e-12)
    frame = pd.DataFrame(
        {"feature": bundle["selected_features"], "importance": normalized}
    ).sort_values("importance", ascending=False)
    importance_map = dict(zip(frame["feature"], frame["importance"]))
    rank_map = {
        feature: rank for rank, feature in enumerate(frame["feature"], start=1)
    }
    return importance_map, rank_map


def model_period_scores(
    bundle: dict, periods: dict[str, pd.DataFrame]
) -> tuple[dict[str, np.ndarray], dict]:
    scores = {}
    for period_name, frame in periods.items():
        X_model = frame[bundle["input_features"]]
        X_ready = bundle["selector"].transform(bundle["imputer"].transform(X_model))
        raw = bundle["model"].predict_proba(X_ready)[:, 1]
        scores[period_name] = apply_calibrator(
            bundle["calibration_method"], bundle["calibrator"], raw
        )
    statistic, p_value = ks_2samp(scores["calibration"], scores["future"])
    summary = {
        "fit_mean": float(np.mean(scores["fit"])),
        "calibration_mean": float(np.mean(scores["calibration"])),
        "future_mean": float(np.mean(scores["future"])),
        "calibration_to_future_ks": float(statistic),
        "calibration_to_future_p_value": float(p_value),
    }
    return scores, summary


def save_top_drift_plot(output_dir: Path, drift: pd.DataFrame) -> None:
    top = drift.nlargest(20, "psi").sort_values("psi")
    colors = top["severity"].map(
        {"high": "#d1495b", "moderate": "#f2a541", "low": "#2878b5"}
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["psi"], color=colors)
    ax.axvline(0.10, color="#f2a541", linestyle="--", linewidth=1, label="PSI 0.10")
    ax.axvline(0.25, color="#d1495b", linestyle="--", linewidth=1, label="PSI 0.25")
    ax.set(xlabel="Population Stability Index", title="Top feature distribution shifts")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "top_feature_psi.png", dpi=190)
    plt.close(fig)


def save_missingness_plot(output_dir: Path, drift: pd.DataFrame) -> None:
    top = drift.assign(abs_delta=drift["missing_rate_delta"].abs()).nlargest(
        20, "abs_delta"
    ).sort_values("missing_rate_delta")
    colors = np.where(top["missing_rate_delta"] >= 0, "#d1495b", "#2878b5")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["missing_rate_delta"] * 100, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set(
        xlabel="Future minus fit missing rate (percentage points)",
        title="Largest missingness shifts",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "missingness_shift.png", dpi=190)
    plt.close(fig)


def save_drift_importance_plot(output_dir: Path, drift: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sizes = 30 + 2500 * drift["max_future_shap_importance"]
    colors = drift["severity"].map(
        {"high": "#d1495b", "moderate": "#f2a541", "low": "#2878b5"}
    )
    ax.scatter(
        drift["psi"],
        drift["max_future_shap_importance"],
        s=sizes,
        c=colors,
        alpha=0.68,
        edgecolor="white",
        linewidth=0.5,
    )
    priority = drift.nlargest(12, "monitor_priority_score")
    for _, row in priority.iterrows():
        ax.annotate(
            row["feature"],
            (row["psi"], row["max_future_shap_importance"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axvline(0.10, color="#777777", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Population Stability Index",
        ylabel="Maximum normalized future SHAP importance",
        title="Drift × model-importance monitoring priorities",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "drift_shap_priority.png", dpi=190)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "결과물" / "secom" / "drift_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/6] Loading SECOM and recreating chronological periods...")
    X_raw, y, timestamps = load_data(project_dir)
    order = timestamps.sort_values(kind="stable").index
    X = X_raw.loc[order].reset_index(drop=True)
    y = y.loc[order].reset_index(drop=True)
    times = timestamps.loc[order].reset_index(drop=True)
    fit_end = int(len(X) * FIT_FRACTION)
    calibration_end = int(len(X) * (FIT_FRACTION + CALIBRATION_FRACTION))
    X_fit = X.iloc[:fit_end]
    X_cal = X.iloc[fit_end:calibration_end]
    X_future = X.iloc[calibration_end:]

    bundles = {
        model_name: joblib.load(
            project_dir
            / "결과물"
            / "secom"
            / "temporal_results"
            / f"{model_name.lower()}_temporal_calibrated.joblib"
        )
        for model_name in MODELS
    }
    kept_features = list(bundles["CatBoost"]["input_features"])
    print(
        f"  fit={len(X_fit)}, calibration={len(X_cal)}, future={len(X_future)}, "
        f"monitored_features={len(kept_features)}"
    )

    print("[2/6] Computing PSI, KS, median, and missingness shifts...")
    rows = []
    for feature in kept_features:
        reference = X_fit[feature]
        current = X_future[feature]
        ref_nonmissing = reference.dropna()
        cur_nonmissing = current.dropna()
        if ref_nonmissing.empty or cur_nonmissing.empty:
            ks_statistic, ks_pvalue = 0.0, 1.0
        else:
            ks_statistic, ks_pvalue = ks_2samp(ref_nonmissing, cur_nonmissing)
        psi = population_stability_index(reference, current)
        missing_delta = float(current.isna().mean() - reference.isna().mean())
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "ks_statistic": float(ks_statistic),
                "ks_p_value": float(ks_pvalue),
                "standardized_median_shift": standardized_median_shift(
                    reference, current
                ),
                "fit_missing_rate": float(reference.isna().mean()),
                "future_missing_rate": float(current.isna().mean()),
                "missing_rate_delta": missing_delta,
                "severity": severity(psi, missing_delta),
            }
        )
    drift = pd.DataFrame(rows)
    drift["ks_q_value"] = benjamini_hochberg(drift["ks_p_value"].to_numpy())
    drift["ks_significant_fdr_05"] = drift["ks_q_value"] < 0.05

    print("[3/6] Connecting drift to future-period SHAP importance...")
    for model_name in MODELS:
        importance, ranks = future_shap_importance(bundles[model_name], X_future)
        key = model_name.lower()
        drift[f"{key}_future_shap_importance"] = drift["feature"].map(
            importance
        ).fillna(0.0)
        drift[f"{key}_future_shap_rank"] = drift["feature"].map(ranks)
        drift[f"selected_by_{key}"] = drift["feature"].isin(
            bundles[model_name]["selected_features"]
        )
    drift["max_future_shap_importance"] = drift[
        ["catboost_future_shap_importance", "xgboost_future_shap_importance"]
    ].max(axis=1)
    drift["monitor_priority_score"] = (
        drift["psi"]
        * (1.0 + 20.0 * drift["max_future_shap_importance"])
        + 2.0 * drift["missing_rate_delta"].abs()
    )
    drift = drift.sort_values("monitor_priority_score", ascending=False)

    print("[4/6] Testing whether periods are distinguishable as a whole...")
    domain_X = pd.concat([X_fit[kept_features], X_future[kept_features]], ignore_index=True)
    domain_y = np.concatenate(
        [np.zeros(len(X_fit), dtype=int), np.ones(len(X_future), dtype=int)]
    )
    domain_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    domain_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    domain_probability = cross_val_predict(
        domain_pipeline,
        domain_X,
        domain_y,
        cv=domain_cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    domain_auc = float(roc_auc_score(domain_y, domain_probability))
    domain_pipeline.fit(domain_X, domain_y)
    coefficients = np.abs(
        domain_pipeline.named_steps["classifier"].coef_[0]
    )
    coefficient_map = dict(zip(kept_features, coefficients))
    drift["domain_logistic_abs_coefficient"] = drift["feature"].map(coefficient_map)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    RocCurveDisplay.from_predictions(domain_y, domain_probability, ax=ax)
    ax.set_title(f"Early-fit vs future domain classifier (AUC={domain_auc:.3f})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "domain_classifier_roc.png", dpi=190)
    plt.close(fig)

    print("[5/6] Measuring calibrated model-score drift...")
    periods = {"fit": X_fit, "calibration": X_cal, "future": X_future}
    score_summary = {}
    score_frame = pd.DataFrame(
        {
            "timestamp": times.iloc[calibration_end:].astype(str).to_numpy(),
            "label": y.iloc[calibration_end:].to_numpy(),
        }
    )
    for model_name in MODELS:
        scores, summary = model_period_scores(bundles[model_name], periods)
        score_summary[model_name] = summary
        score_frame[f"{model_name.lower()}_calibrated_probability"] = scores["future"]

    print("[6/6] Saving tables, figures, and report...")
    counts = drift["severity"].value_counts().to_dict()
    priority_features = drift.head(15)["feature"].tolist()
    summary = {
        "fit_rows": len(X_fit),
        "future_rows": len(X_future),
        "monitored_features": len(kept_features),
        "severity_counts": {key: int(value) for key, value in counts.items()},
        "ks_significant_after_fdr": int(drift["ks_significant_fdr_05"].sum()),
        "domain_classifier_auc": domain_auc,
        "model_score_drift": score_summary,
        "top_monitoring_priorities": priority_features,
        "psi_note": "PSI >=0.10 moderate, >=0.25 high; heuristics, not universal laws.",
    }
    drift.to_csv(output_dir / "feature_drift.csv", index=False)
    score_frame.to_csv(output_dir / "future_model_scores.csv", index=False)
    (output_dir / "drift_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_top_drift_plot(output_dir, drift)
    save_missingness_plot(output_dir, drift)
    save_drift_importance_plot(output_dir, drift)

    if domain_auc >= 0.80:
        domain_level = "강함"
    elif domain_auc >= 0.65:
        domain_level = "중간"
    else:
        domain_level = "낮음"
    lines = [
        "# SECOM 시간 드리프트 분석",
        "",
        f"- 비교: 초기 학습 {len(X_fit)}개 vs 미래 평가 {len(X_future)}개",
        f"- 감시 변수: {len(kept_features)}개",
        f"- high drift: {counts.get('high', 0)}개",
        f"- moderate drift: {counts.get('moderate', 0)}개",
        f"- KS FDR 5% 유의 변수: {int(drift['ks_significant_fdr_05'].sum())}개",
        f"- 기간 판별 모델 OOF ROC-AUC: {domain_auc:.4f} ({domain_level} 드리프트 신호)",
        "",
        "## 우선 감시 변수",
        "",
        ", ".join(priority_features),
        "",
        "우선순위는 PSI, 결측률 변화, 미래 구간 SHAP 중요도를 함께 반영했습니다.",
        "",
        "## 모델 점수 변화",
        "",
        "| 모델 | 학습 평균 | 보정 평균 | 미래 평균 | 보정→미래 KS | p-value |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        row = score_summary[model_name]
        lines.append(
            f"| {model_name} | {row['fit_mean']:.4f} | "
            f"{row['calibration_mean']:.4f} | {row['future_mean']:.4f} | "
            f"{row['calibration_to_future_ks']:.4f} | "
            f"{row['calibration_to_future_p_value']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## 주의",
            "",
            "PSI 0.10/0.25 기준은 실무 휴리스틱이며 절대 기준이 아닙니다. 익명 센서",
            "변수이므로 공정 의미를 연결하기 전에는 원인으로 단정할 수 없습니다. 기간 판별",
            "모델은 드리프트 존재 여부를 보조하는 진단 도구이며 불량 예측 모델이 아닙니다.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  domain_auc={domain_auc:.4f}, high={counts.get('high', 0)}, moderate={counts.get('moderate', 0)}")
    print(f"  results={output_dir}")


if __name__ == "__main__":
    main()
