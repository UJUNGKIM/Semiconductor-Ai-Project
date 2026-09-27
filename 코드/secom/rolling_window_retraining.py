"""Walk-forward comparison of expanding, rolling, and recency-weighted training.

Strategy selection uses four chronological validation blocks ending at 65% of
the timeline.  The next 15% calibrates probabilities and selects thresholds.
The final 20% is used once for evaluation and never for strategy selection.
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
from sklearn.metrics import average_precision_score, roc_auc_score

from temporal_calibration_evaluation import (
    apply_calibrator,
    classification_metrics,
    fit_calibrator,
    fit_structural_filter,
    probability_metrics,
    select_f2_threshold,
)
from train_compare_models import RANDOM_STATE, load_data
from tune_weighted_models import build_tuned_model, fit_preprocessor


SELECTION_END_FRACTION = 0.65
CALIBRATION_END_FRACTION = 0.80
INITIAL_TRAIN_ROWS = 400
WALK_FORWARD_FOLDS = 4
RECENCY_HALF_LIFE = 250.0
STRATEGIES = ("expanding", "rolling_300", "rolling_500", "exp_decay_250")
MODELS = ("CatBoost", "XGBoost")
BOOTSTRAP_SAMPLES = 2000


def strategy_indices_and_weights(
    cutoff: int, strategy: str
) -> tuple[np.ndarray, np.ndarray | None]:
    if strategy == "rolling_300":
        indices = np.arange(max(0, cutoff - 300), cutoff)
        return indices, None
    if strategy == "rolling_500":
        indices = np.arange(max(0, cutoff - 500), cutoff)
        return indices, None
    indices = np.arange(cutoff)
    if strategy == "expanding":
        return indices, None
    if strategy == "exp_decay_250":
        age = cutoff - 1 - indices
        weights = np.power(0.5, age / RECENCY_HALF_LIFE)
        weights = weights / np.mean(weights)
        return indices, weights
    raise ValueError(strategy)


def fit_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    cutoff: int,
    validation_indices: np.ndarray,
    strategy: str,
    config: dict,
    seed: int,
) -> tuple[np.ndarray, dict]:
    train_indices, sample_weight = strategy_indices_and_weights(cutoff, strategy)
    X_train_raw = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    X_validation_raw = X.iloc[validation_indices]
    kept_features, high_missing, constants = fit_structural_filter(X_train_raw)
    effective_config = dict(config)
    if effective_config["top_k"] != "all":
        effective_config["top_k"] = min(
            int(effective_config["top_k"]), len(kept_features)
        )
    train_ready, validation_ready, imputer, selector, selected_features = fit_preprocessor(
        X_train_raw[kept_features],
        y_train,
        X_validation_raw[kept_features],
        effective_config["top_k"],
    )
    if sample_weight is None:
        ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    else:
        ratio = float(
            sample_weight[y_train.to_numpy() == 0].sum()
            / sample_weight[y_train.to_numpy() == 1].sum()
        )
    model = build_tuned_model(
        effective_config,
        ratio * float(effective_config["weight_multiplier"]),
        seed,
    )
    fit_kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
    model.fit(train_ready, y_train, **fit_kwargs)
    probability = model.predict_proba(validation_ready)[:, 1]
    artifacts = {
        "model": model,
        "imputer": imputer,
        "selector": selector,
        "input_features": kept_features,
        "selected_features": selected_features,
        "parameters": effective_config,
        "removed_high_missing_features": high_missing,
        "removed_constant_features": constants,
        "training_rows": len(train_indices),
        "sample_weighted": sample_weight is not None,
    }
    return probability, artifacts


def transform_and_predict(artifacts: dict, X_evaluation: pd.DataFrame) -> np.ndarray:
    X_model = X_evaluation[artifacts["input_features"]]
    X_ready = artifacts["selector"].transform(
        artifacts["imputer"].transform(X_model)
    )
    return artifacts["model"].predict_proba(X_ready)[:, 1]


def paired_bootstrap(
    y_true: np.ndarray,
    baseline_probability: np.ndarray,
    selected_probability: np.ndarray,
) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    differences = []
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = np.concatenate(
            [
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            ]
        )
        differences.append(
            average_precision_score(y_true[indices], selected_probability[indices])
            - average_precision_score(y_true[indices], baseline_probability[indices])
        )
    values = np.asarray(differences)
    return {
        "mean_pr_auc_difference": float(values.mean()),
        "ci_95_low": float(np.quantile(values, 0.025)),
        "ci_95_high": float(np.quantile(values, 0.975)),
        "probability_selected_greater": float(np.mean(values > 0)),
    }


def save_walk_forward_plot(output_dir: Path, summary: pd.DataFrame) -> None:
    labels = {
        "expanding": "All history",
        "rolling_300": "Recent 300",
        "rolling_500": "Recent 500",
        "exp_decay_250": "Decay half-life 250",
    }
    x = np.arange(len(STRATEGIES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"CatBoost": "#d1495b", "XGBoost": "#2878b5"}
    for model_index, model_name in enumerate(MODELS):
        rows = summary.loc[summary["model"] == model_name].set_index("strategy").loc[
            list(STRATEGIES)
        ]
        ax.bar(
            x + (model_index - 0.5) * width,
            rows["mean_pr_auc"],
            width,
            yerr=rows["std_pr_auc"],
            capsize=3,
            color=colors[model_name],
            label=model_name,
        )
    ax.set_xticks(x, [labels[value] for value in STRATEGIES], rotation=18, ha="right")
    ax.set_ylabel("Mean walk-forward PR-AUC")
    ax.set_title("Training-window strategy selection (4 chronological folds)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "walk_forward_strategy_selection.png", dpi=190)
    plt.close(fig)


def save_future_comparison_plot(output_dir: Path, future: pd.DataFrame) -> None:
    labels = []
    values = []
    colors = []
    for model_name in MODELS:
        for role in ("baseline", "selected"):
            row = future.loc[
                (future["model"] == model_name) & (future["role"] == role)
            ].iloc[0]
            labels.append(f"{model_name}\n{role}: {row['strategy']}")
            values.append(row["pr_auc"])
            colors.append("#888888" if role == "baseline" else "#d1495b")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0541, color="#333333", linestyle="--", label="future prevalence")
    ax.set_ylabel("Future PR-AUC")
    ax.set_title("Future evaluation: expanding baseline vs selected strategy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "future_selected_vs_baseline.png", dpi=190)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "결과물" / "secom" / "rolling_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/7] Loading and sorting SECOM by timestamp...")
    X_raw, y_raw, timestamps = load_data(project_dir)
    order = timestamps.sort_values(kind="stable").index
    X = X_raw.loc[order].reset_index(drop=True)
    y = y_raw.loc[order].reset_index(drop=True)
    times = timestamps.loc[order].reset_index(drop=True)
    selection_end = int(len(X) * SELECTION_END_FRACTION)
    calibration_end = int(len(X) * CALIBRATION_END_FRACTION)
    edges = np.linspace(
        INITIAL_TRAIN_ROWS, selection_end, WALK_FORWARD_FOLDS + 1, dtype=int
    )
    print(
        f"  walk-forward={INITIAL_TRAIN_ROWS}:{selection_end}, "
        f"calibration={selection_end}:{calibration_end}, future={calibration_end}:{len(X)}"
    )

    configs = json.loads(
        (
            project_dir
            / "결과물"
            / "secom"
            / "dual_model_results"
            / "selected_configurations.json"
        ).read_text(encoding="utf-8")
    )

    print("[2/7] Running four chronological strategy-selection folds...")
    fold_rows = []
    for fold_number in range(WALK_FORWARD_FOLDS):
        cutoff = int(edges[fold_number])
        validation_indices = np.arange(cutoff, int(edges[fold_number + 1]))
        y_validation = y.iloc[validation_indices]
        for model_name in MODELS:
            for strategy in STRATEGIES:
                probability, artifacts = fit_and_predict(
                    X,
                    y,
                    cutoff,
                    validation_indices,
                    strategy,
                    configs[model_name],
                    RANDOM_STATE + fold_number,
                )
                fold_rows.append(
                    {
                        "fold": fold_number + 1,
                        "train_cutoff": cutoff,
                        "validation_start": int(validation_indices[0]),
                        "validation_end": int(validation_indices[-1]),
                        "validation_rows": len(validation_indices),
                        "validation_defects": int(y_validation.sum()),
                        "model": model_name,
                        "strategy": strategy,
                        "training_rows": artifacts["training_rows"],
                        "pr_auc": float(
                            average_precision_score(y_validation, probability)
                        ),
                        "roc_auc": float(roc_auc_score(y_validation, probability)),
                    }
                )
        print(
            f"  fold {fold_number + 1}/{WALK_FORWARD_FOLDS}: "
            f"train_end={cutoff}, validation_defects={int(y_validation.sum())}"
        )
    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby(["model", "strategy"], as_index=False)
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            std_pr_auc=("pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"),
        )
    )

    print("[3/7] Selecting one training strategy per model without future data...")
    selected_strategies = {}
    for model_name in MODELS:
        candidates = summary.loc[summary["model"] == model_name].sort_values(
            ["mean_pr_auc", "mean_roc_auc"], ascending=False
        )
        selected_strategies[model_name] = str(candidates.iloc[0]["strategy"])
        print(
            f"  {model_name}: {selected_strategies[model_name]} "
            f"(mean AP={candidates.iloc[0]['mean_pr_auc']:.4f})"
        )
    summary["selected"] = summary.apply(
        lambda row: row["strategy"] == selected_strategies[row["model"]], axis=1
    )

    print("[4/7] Refitting selected and expanding baselines through 65%...")
    calibration_indices = np.arange(selection_end, calibration_end)
    future_indices = np.arange(calibration_end, len(X))
    y_calibration = y.iloc[calibration_indices]
    y_future = y.iloc[future_indices]
    future_rows = []
    future_probabilities = {model_name: {} for model_name in MODELS}
    saved_bundles = {}
    for model_name in MODELS:
        selected = selected_strategies[model_name]
        roles = {"baseline": "expanding", "selected": selected}
        trained_cache = {}
        for role, strategy in roles.items():
            if strategy in trained_cache:
                raw_calibration, artifacts = trained_cache[strategy]
            else:
                raw_calibration, artifacts = fit_and_predict(
                    X,
                    y,
                    selection_end,
                    calibration_indices,
                    strategy,
                    configs[model_name],
                    RANDOM_STATE,
                )
                trained_cache[strategy] = (raw_calibration, artifacts)
            calibrator = fit_calibrator(
                "sigmoid", raw_calibration, y_calibration.to_numpy()
            )
            calibration_probability = apply_calibrator(
                "sigmoid", calibrator, raw_calibration
            )
            threshold = select_f2_threshold(
                y_calibration.to_numpy(), calibration_probability
            )
            raw_future = transform_and_predict(artifacts, X.iloc[future_indices])
            future_probability = apply_calibrator("sigmoid", calibrator, raw_future)
            future_probabilities[model_name][role] = future_probability
            future_rows.append(
                {
                    "model": model_name,
                    "role": role,
                    "strategy": strategy,
                    **probability_metrics(y_future.to_numpy(), future_probability),
                    **classification_metrics(
                        y_future.to_numpy(), future_probability, threshold
                    ),
                }
            )
            if role == "selected":
                saved_bundles[model_name] = {
                    **artifacts,
                    "calibrator": calibrator,
                    "calibration_method": "sigmoid",
                    "threshold": threshold,
                    "model_name": model_name,
                    "strategy": strategy,
                    "walk_forward_mean_pr_auc": float(
                        summary.loc[
                            (summary["model"] == model_name)
                            & (summary["strategy"] == strategy),
                            "mean_pr_auc",
                        ].iloc[0]
                    ),
                    "selection_note": (
                        "Selected by four walk-forward folds ending at 65%; "
                        "calibrated on 65-80%; evaluated on final 20%."
                    ),
                }

    print("[5/7] Evaluating once on the final future period...")
    future = pd.DataFrame(future_rows)
    bootstrap = {}
    for model_name in MODELS:
        selected_row = future.loc[
            (future["model"] == model_name) & (future["role"] == "selected")
        ].iloc[0]
        baseline_row = future.loc[
            (future["model"] == model_name) & (future["role"] == "baseline")
        ].iloc[0]
        bootstrap[model_name] = paired_bootstrap(
            y_future.to_numpy(),
            future_probabilities[model_name]["baseline"],
            future_probabilities[model_name]["selected"],
        )
        print(
            f"  {model_name}: baseline AP={baseline_row['pr_auc']:.4f}, "
            f"selected AP={selected_row['pr_auc']:.4f}, "
            f"selected recall={selected_row['recall']:.4f}, F2={selected_row['f2']:.4f}"
        )
        joblib.dump(
            saved_bundles[model_name],
            output_dir / f"{model_name.lower()}_rolling_selected.joblib",
        )

    print("[6/7] Saving tables and figures...")
    folds.to_csv(output_dir / "walk_forward_folds.csv", index=False)
    summary.to_csv(output_dir / "strategy_selection_summary.csv", index=False)
    future.to_csv(output_dir / "future_evaluation.csv", index=False)
    (output_dir / "selection_bootstrap.json").write_text(
        json.dumps(
            {
                "selected_strategies": selected_strategies,
                "selected_minus_expanding_future_pr_auc": bootstrap,
                "future_used_for_selection": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_walk_forward_plot(output_dir, summary)
    save_future_comparison_plot(output_dir, future)

    print("[7/7] Writing report...")
    lines = [
        "# 최근 데이터 중심 walk-forward 재학습",
        "",
        f"- 전략 선택: 첫 {selection_end}개 안에서 {WALK_FORWARD_FOLDS}개 순차 검증 fold",
        f"- 확률 보정·임계값: 다음 {len(calibration_indices)}개 / 불량 {int(y_calibration.sum())}개",
        f"- 미래 평가: 마지막 {len(future_indices)}개 / 불량 {int(y_future.sum())}개",
        "- 미래 데이터는 전략 선택에 사용하지 않음",
        "",
        "## Walk-forward 전략 선택",
        "",
        "| 모델 | 선택 전략 | 평균 PR-AUC | 표준편차 |",
        "|---|---|---:|---:|",
    ]
    for model_name in MODELS:
        row = summary.loc[
            (summary["model"] == model_name) & summary["selected"]
        ].iloc[0]
        lines.append(
            f"| {model_name} | {row['strategy']} | "
            f"{row['mean_pr_auc']:.4f} | {row['std_pr_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 미래 20%: 누적 기준과 선택 전략 비교",
            "",
            "| 모델 | 역할 | 전략 | PR-AUC | Brier | Precision | Recall | F2 | TP/FP/FN/TN |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in future.iterrows():
        lines.append(
            f"| {row['model']} | {row['role']} | {row['strategy']} | "
            f"{row['pr_auc']:.4f} | {row['brier']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f2']:.4f} | "
            f"{int(row['tp'])}/{int(row['fp'])}/{int(row['fn'])}/{int(row['tn'])} |"
        )
    lines.extend(
        [
            "",
            "## 제한",
            "",
            "전략은 미래 구간 없이 선택했지만 기존 모델 하이퍼파라미터는 앞선 전체 기간",
            "랜덤 교차검증의 영향을 받았습니다. 마지막 미래 구간의 불량도 17개뿐이므로",
            "우열은 bootstrap 구간과 신규 lot 검증을 함께 봐야 합니다.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  results={output_dir}")


if __name__ == "__main__":
    main()
