"""Compare drift-aware feature filters without using the future evaluation period.

Feature stability and strategy selection use only the first 80% of the timeline:
the first 60% fits models and the next 20% estimates drift, calibrates scores, and
selects an operating threshold.  The last 20% is evaluated only after selection.
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
from sklearn.metrics import average_precision_score, precision_recall_curve

from analyze_temporal_drift import population_stability_index
from temporal_calibration_evaluation import (
    CALIBRATION_FRACTION,
    FIT_FRACTION,
    apply_calibrator,
    classification_metrics,
    fit_calibrator,
    fit_structural_filter,
    probability_metrics,
    select_f2_threshold,
)
from train_compare_models import RANDOM_STATE, load_data
from tune_weighted_models import build_tuned_model, fit_preprocessor


MODELS = ("CatBoost", "XGBoost")
STRATEGY_ORDER = ("baseline_all", "stable_psi_025", "stable_psi_010")
BOOTSTRAP_SAMPLES = 2000


def stable_feature_sets(
    X_fit: pd.DataFrame, X_calibration: pd.DataFrame, base_features: list[str]
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    rows = []
    for feature in base_features:
        psi = population_stability_index(X_fit[feature], X_calibration[feature])
        missing_delta = float(
            X_calibration[feature].isna().mean() - X_fit[feature].isna().mean()
        )
        rows.append(
            {
                "feature": feature,
                "fit_to_calibration_psi": psi,
                "fit_missing_rate": float(X_fit[feature].isna().mean()),
                "calibration_missing_rate": float(
                    X_calibration[feature].isna().mean()
                ),
                "absolute_missing_rate_delta": abs(missing_delta),
            }
        )
    stability = pd.DataFrame(rows)
    strategies = {
        "baseline_all": base_features,
        "stable_psi_025": stability.loc[
            (stability["fit_to_calibration_psi"] < 0.25)
            & (stability["absolute_missing_rate_delta"] < 0.10),
            "feature",
        ].tolist(),
        "stable_psi_010": stability.loc[
            (stability["fit_to_calibration_psi"] < 0.10)
            & (stability["absolute_missing_rate_delta"] < 0.05),
            "feature",
        ].tolist(),
    }
    return strategies, stability


def stratified_paired_bootstrap(
    y_true: np.ndarray,
    baseline_probability: np.ndarray,
    selected_probability: np.ndarray,
) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    differences = []
    for _ in range(BOOTSTRAP_SAMPLES):
        index = np.concatenate(
            [
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            ]
        )
        differences.append(
            average_precision_score(y_true[index], selected_probability[index])
            - average_precision_score(y_true[index], baseline_probability[index])
        )
    values = np.asarray(differences)
    return {
        "mean_pr_auc_difference": float(values.mean()),
        "ci_95_low": float(np.quantile(values, 0.025)),
        "ci_95_high": float(np.quantile(values, 0.975)),
        "probability_selected_greater": float(np.mean(values > 0)),
    }


def save_strategy_plot(
    output_dir: Path, calibration: pd.DataFrame, future: pd.DataFrame
) -> None:
    labels = [
        "All 444",
        "Stable PSI<0.25",
        "Stable PSI<0.10",
    ]
    x = np.arange(len(STRATEGY_ORDER))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {"CatBoost": "#d1495b", "XGBoost": "#2878b5"}
    for ax, table, title in (
        (axes[0], calibration, "Calibration-period selection"),
        (axes[1], future, "Future-period diagnostic comparison"),
    ):
        for model_index, model_name in enumerate(MODELS):
            rows = (
                table.loc[table["model"] == model_name]
                .set_index("strategy")
                .loc[list(STRATEGY_ORDER)]
            )
            offset = (model_index - 0.5) * width
            ax.bar(
                x + offset,
                rows["pr_auc"],
                width,
                color=colors[model_name],
                label=model_name,
            )
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylabel("PR-AUC")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_pr_auc_comparison.png", dpi=190)
    plt.close(fig)


def save_future_pr_plot(
    output_dir: Path,
    y_future: np.ndarray,
    probabilities: dict[str, dict[str, np.ndarray]],
    selected_strategies: dict[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = {"CatBoost": "#d1495b", "XGBoost": "#2878b5"}
    for model_name in MODELS:
        strategy = selected_strategies[model_name]
        probability = probabilities[model_name][strategy]
        precision, recall, _ = precision_recall_curve(y_future, probability)
        ap = average_precision_score(y_future, probability)
        ax.plot(
            recall,
            precision,
            linewidth=2,
            color=colors[model_name],
            label=f"{model_name}/{strategy} (AP={ap:.3f})",
        )
    ax.axhline(
        np.mean(y_future),
        color="#666666",
        linestyle="--",
        label=f"baseline={np.mean(y_future):.3f}",
    )
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Future-period PR curves: calibration-selected strategies",
        xlim=(0, 1.01),
        ylim=(0, 1.01),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "selected_future_pr_curves.png", dpi=190)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "결과물" / "secom" / "drift_robust_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/7] Recreating the chronological 60/20/20 split...")
    X_raw, y_raw, timestamps = load_data(project_dir)
    order = timestamps.sort_values(kind="stable").index
    X = X_raw.loc[order].reset_index(drop=True)
    y = y_raw.loc[order].reset_index(drop=True)
    times = timestamps.loc[order].reset_index(drop=True)
    fit_end = int(len(X) * FIT_FRACTION)
    calibration_end = int(len(X) * (FIT_FRACTION + CALIBRATION_FRACTION))
    X_fit_raw, y_fit = X.iloc[:fit_end], y.iloc[:fit_end]
    X_cal_raw, y_cal = X.iloc[fit_end:calibration_end], y.iloc[fit_end:calibration_end]
    X_future_raw, y_future = X.iloc[calibration_end:], y.iloc[calibration_end:]

    print("[2/7] Learning structural and drift filters without future data...")
    base_features, high_missing, constants = fit_structural_filter(X_fit_raw)
    strategies, stability = stable_feature_sets(
        X_fit_raw, X_cal_raw, base_features
    )
    for strategy in STRATEGY_ORDER:
        print(f"  {strategy}: {len(strategies[strategy])} features")
    stability.to_csv(output_dir / "fit_calibration_feature_stability.csv", index=False)

    configs = json.loads(
        (
            project_dir
            / "결과물"
            / "secom"
            / "dual_model_results"
            / "selected_configurations.json"
        ).read_text(encoding="utf-8")
    )
    calibration_rows = []
    future_rows = []
    trained = {}
    future_probabilities = {model_name: {} for model_name in MODELS}

    print("[3/7] Training both models under three feature strategies...")
    for model_name in MODELS:
        trained[model_name] = {}
        for strategy in STRATEGY_ORDER:
            features = strategies[strategy]
            config = dict(configs[model_name])
            original_top_k = config["top_k"]
            if original_top_k == "all":
                effective_top_k = "all"
            else:
                effective_top_k = min(int(original_top_k), len(features))
            config["top_k"] = effective_top_k
            X_fit = X_fit_raw[features]
            X_cal = X_cal_raw[features]
            X_future = X_future_raw[features]
            fit_ready, cal_ready, imputer, selector, selected_features = fit_preprocessor(
                X_fit, y_fit, X_cal, effective_top_k
            )
            future_ready = selector.transform(imputer.transform(X_future))
            ratio = float((y_fit == 0).sum() / (y_fit == 1).sum())
            model = build_tuned_model(
                config,
                ratio * float(config["weight_multiplier"]),
                RANDOM_STATE,
            )
            model.fit(fit_ready, y_fit)
            raw_cal = model.predict_proba(cal_ready)[:, 1]
            raw_future = model.predict_proba(future_ready)[:, 1]
            calibrator = fit_calibrator("sigmoid", raw_cal, y_cal.to_numpy())
            cal_probability = apply_calibrator(
                "sigmoid", calibrator, raw_cal
            )
            future_probability = apply_calibrator(
                "sigmoid", calibrator, raw_future
            )
            threshold = select_f2_threshold(y_cal.to_numpy(), cal_probability)
            calibration_rows.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "candidate_feature_count": len(features),
                    "selected_feature_count": len(selected_features),
                    **probability_metrics(y_cal.to_numpy(), cal_probability),
                    **classification_metrics(
                        y_cal.to_numpy(), cal_probability, threshold
                    ),
                }
            )
            future_rows.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "candidate_feature_count": len(features),
                    "selected_feature_count": len(selected_features),
                    **probability_metrics(y_future.to_numpy(), future_probability),
                    **classification_metrics(
                        y_future.to_numpy(), future_probability, threshold
                    ),
                }
            )
            future_probabilities[model_name][strategy] = future_probability
            trained[model_name][strategy] = {
                "model": model,
                "calibrator": calibrator,
                "calibration_method": "sigmoid",
                "imputer": imputer,
                "selector": selector,
                "threshold": threshold,
                "model_name": model_name,
                "strategy": strategy,
                "parameters": config,
                "input_features": features,
                "selected_features": selected_features,
                "removed_high_missing_features": high_missing,
                "removed_constant_features": constants,
                "fit_period_end": str(times.iloc[fit_end - 1]),
                "calibration_period_end": str(times.iloc[calibration_end - 1]),
            }
            print(
                f"  {model_name}/{strategy}: cal AP="
                f"{calibration_rows[-1]['pr_auc']:.4f}"
            )

    print("[4/7] Selecting one feature strategy per model on calibration AP only...")
    calibration_table = pd.DataFrame(calibration_rows)
    future_table = pd.DataFrame(future_rows)
    selected_strategies = {}
    for model_name in MODELS:
        candidates = calibration_table.loc[
            calibration_table["model"] == model_name
        ].sort_values(["pr_auc", "f2", "candidate_feature_count"], ascending=[False, False, True])
        selected_strategies[model_name] = str(candidates.iloc[0]["strategy"])
        print(
            f"  {model_name}: {selected_strategies[model_name]} "
            f"(cal AP={candidates.iloc[0]['pr_auc']:.4f})"
        )
    calibration_table["selected_on_calibration"] = calibration_table.apply(
        lambda row: row["strategy"] == selected_strategies[row["model"]], axis=1
    )
    future_table["selected_on_calibration"] = future_table.apply(
        lambda row: row["strategy"] == selected_strategies[row["model"]], axis=1
    )

    print("[5/7] Evaluating the calibration-selected strategies on future data...")
    bootstrap_summary = {}
    for model_name in MODELS:
        selected = selected_strategies[model_name]
        selected_row = future_table.loc[
            (future_table["model"] == model_name)
            & (future_table["strategy"] == selected)
        ].iloc[0]
        print(
            f"  {model_name}/{selected}: AP={selected_row['pr_auc']:.4f}, "
            f"precision={selected_row['precision']:.4f}, "
            f"recall={selected_row['recall']:.4f}, F2={selected_row['f2']:.4f}"
        )
        bootstrap_summary[model_name] = stratified_paired_bootstrap(
            y_future.to_numpy(),
            future_probabilities[model_name]["baseline_all"],
            future_probabilities[model_name][selected],
        )
        bundle = trained[model_name][selected]
        bundle["selection_note"] = (
            "Feature strategy selected on the middle 20% calibration period; "
            "future 20% was not used for selection."
        )
        joblib.dump(
            bundle, output_dir / f"{model_name.lower()}_drift_aware.joblib"
        )

    print("[6/7] Saving comparison tables and figures...")
    calibration_table.to_csv(output_dir / "calibration_strategy_metrics.csv", index=False)
    future_table.to_csv(output_dir / "future_strategy_metrics.csv", index=False)
    (output_dir / "selection_and_bootstrap.json").write_text(
        json.dumps(
            {
                "selected_strategies": selected_strategies,
                "future_pr_auc_selected_minus_baseline_bootstrap": bootstrap_summary,
                "selection_data": "middle 20% only",
                "future_data_used_for_selection": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_strategy_plot(output_dir, calibration_table, future_table)
    save_future_pr_plot(
        output_dir,
        y_future.to_numpy(),
        future_probabilities,
        selected_strategies,
    )

    print("[7/7] Writing report...")
    selected_future = future_table.loc[future_table["selected_on_calibration"]]
    lines = [
        "# 드리프트 대응 안정 변수 재학습",
        "",
        "- 변수 안정성 계산: 초기 60%와 보정 20%만 사용",
        "- 전략 선택·sigmoid 보정·임계값 결정: 보정 20%만 사용",
        "- 마지막 미래 20%는 선택 완료 후 평가에만 사용",
        "",
        "## 안정 변수 후보",
        "",
        f"- baseline_all: {len(strategies['baseline_all'])}개",
        f"- stable_psi_025: {len(strategies['stable_psi_025'])}개",
        f"- stable_psi_010: {len(strategies['stable_psi_010'])}개",
        "",
        "## 보정 구간에서 선택된 전략",
        "",
        "| 모델 | 전략 | 후보 변수 | 최종 선택 변수 | Calibration PR-AUC |",
        "|---|---|---:|---:|---:|",
    ]
    for model_name in MODELS:
        row = calibration_table.loc[
            (calibration_table["model"] == model_name)
            & calibration_table["selected_on_calibration"]
        ].iloc[0]
        lines.append(
            f"| {model_name} | {row['strategy']} | "
            f"{int(row['candidate_feature_count'])} | "
            f"{int(row['selected_feature_count'])} | {row['pr_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 미래 20% 결과 — 선택된 전략",
            "",
            "| 모델 | 전략 | PR-AUC | Brier | Precision | Recall | F2 | TP/FP/FN/TN |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in selected_future.iterrows():
        lines.append(
            f"| {row['model']} | {row['strategy']} | {row['pr_auc']:.4f} | "
            f"{row['brier']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f2']:.4f} | "
            f"{int(row['tp'])}/{int(row['fp'])}/{int(row['fn'])}/{int(row['tn'])} |"
        )
    lines.extend(
        [
            "",
            "## 선택 편향 방지",
            "",
            "미래 구간 결과가 더 좋아 보이는 다른 전략이 있더라도 그것으로 다시 선택하지",
            "않았습니다. 선택은 보정 구간 PR-AUC로 고정했습니다. 미래 불량이 17개뿐이므로",
            "bootstrap 구간이 넓을 수 있으며 신규 lot 검증이 여전히 필요합니다.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  results={output_dir}")


if __name__ == "__main__":
    main()
