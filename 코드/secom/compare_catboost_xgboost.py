"""Fair CatBoost/XGBoost operating-point comparison for SECOM.

The fixed 20% holdout is never used to choose a model or threshold.  Candidate
configurations are read from the earlier training-only tuning results.  Every
learned preprocessing step is fitted inside each repeated CV training fold.
"""

from __future__ import annotations

import json
import time
import warnings
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from train_compare_models import RANDOM_STATE, TEST_SIZE, load_data, structural_filter
from tune_weighted_models import build_tuned_model, fit_preprocessor


N_SPLITS = 5
N_REPEATS = 3
MIN_RECALL = 0.60
TARGET_RECALLS = (0.50, 0.60, 0.70, 0.80, 0.90)
BOOTSTRAP_SAMPLES = 2000
MODELS = ("CatBoost", "XGBoost")


def native(value):
    return value.item() if isinstance(value, np.generic) else value


def load_selected_configs(project_dir: Path) -> dict[str, dict]:
    """Select the best previously tuned configuration within each model family."""
    path = (
        project_dir
        / "결과물"
        / "secom"
        / "tuning_results"
        / "tuning_summary.csv"
    )
    summary = pd.read_csv(path)
    selected: dict[str, dict] = {}
    for model_name in MODELS:
        candidates = summary.loc[summary["model"] == model_name].copy()
        eligible = candidates.loc[candidates["oof_recall"] >= MIN_RECALL]
        if not eligible.empty:
            candidates = eligible
        row = candidates.sort_values(
            ["oof_f2", "oof_pr_auc"], ascending=False
        ).iloc[0]
        selected[model_name] = json.loads(row["parameters_json"])
    return selected


def metrics_at_threshold(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    normal_count = int((y_true == 0).sum())
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "false_alarms_per_100_normal": float(100.0 * fp / normal_count),
        "alert_count": int(tp + fp),
    }


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_recall: float,
    objective: str,
) -> float:
    """Choose a threshold on OOF predictions only.

    ``precision`` finds the highest precision satisfying the recall target.
    ``f2`` maximizes F2 while satisfying the recall target.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    precision = precision[:-1]
    recall = recall[:-1]
    f2 = 5.0 * precision * recall / (4.0 * precision + recall + 1e-12)
    eligible = np.flatnonzero(recall >= target_recall)
    if eligible.size == 0:
        return float(thresholds[int(np.argmax(recall))])
    scores = precision if objective == "precision" else f2
    best_score = np.max(scores[eligible])
    tied = eligible[np.isclose(scores[eligible], best_score)]
    # Prefer the highest threshold in an exact tie to avoid unnecessary alerts.
    return float(np.max(thresholds[tied]))


def paired_stratified_bootstrap(
    y_true: np.ndarray, probabilities: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(RANDOM_STATE)
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    records = []
    for bootstrap_id in range(BOOTSTRAP_SAMPLES):
        indices = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        record = {"bootstrap_id": bootstrap_id}
        for model_name in MODELS:
            record[model_name] = average_precision_score(
                y_true[indices], probabilities[model_name][indices]
            )
        record["catboost_minus_xgboost"] = (
            record["CatBoost"] - record["XGBoost"]
        )
        records.append(record)
    draws = pd.DataFrame(records)
    summary = {}
    for column in (*MODELS, "catboost_minus_xgboost"):
        values = draws[column].to_numpy()
        summary[column] = {
            "mean": float(np.mean(values)),
            "ci_95_low": float(np.quantile(values, 0.025)),
            "ci_95_high": float(np.quantile(values, 0.975)),
        }
    summary["probability_catboost_pr_auc_greater"] = float(
        np.mean(draws["catboost_minus_xgboost"] > 0)
    )
    return draws, summary


def save_plots(
    output_dir: Path,
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    operating_rows: pd.DataFrame,
) -> None:
    colors = {"CatBoost": "#d1495b", "XGBoost": "#2878b5"}
    prevalence = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for model_name in MODELS:
        precision, recall, _ = precision_recall_curve(
            y_true, probabilities[model_name]
        )
        ap = average_precision_score(y_true, probabilities[model_name])
        ax.plot(
            recall,
            precision,
            color=colors[model_name],
            linewidth=2,
            label=f"{model_name} (PR-AUC={ap:.3f})",
        )
    ax.axhline(
        prevalence,
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        label=f"Random baseline={prevalence:.3f}",
    )
    ax.set(xlabel="Recall", ylabel="Precision", title="Repeated OOF precision-recall")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "oof_precision_recall_comparison.png", dpi=190)
    plt.close(fig)

    target_rows = operating_rows.loc[operating_rows["operating_point"] != "balanced_f2"]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for model_name in MODELS:
        rows = target_rows.loc[target_rows["model"] == model_name].sort_values(
            "target_recall"
        )
        ax.plot(
            rows["recall"],
            rows["false_alarms_per_100_normal"],
            marker="o",
            linewidth=2,
            color=colors[model_name],
            label=model_name,
        )
        for _, row in rows.iterrows():
            ax.annotate(
                f"{int(row['target_recall'] * 100)}%",
                (row["recall"], row["false_alarms_per_100_normal"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set(
        xlabel="OOF recall",
        ylabel="False alarms per 100 normal wafers",
        title="Recall versus false-alarm burden",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "recall_false_alarm_tradeoff.png", dpi=190)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "결과물" / "secom" / "dual_model_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/7] Loading data and recreating the fixed stratified split...")
    X_raw, y, timestamps = load_data(project_dir)
    X, high_missing, constants = structural_filter(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(
        f"  train={len(y_train)} ({int(y_train.sum())} defects), "
        f"test={len(y_test)} ({int(y_test.sum())} defects), features={X.shape[1]}"
    )

    print("[2/7] Loading the best training-only configuration for each model...")
    configs = load_selected_configs(project_dir)
    for model_name, config in configs.items():
        print(
            f"  {model_name}: {config['config_id']}, top_k={config['top_k']}, "
            f"weight_multiplier={config['weight_multiplier']}"
        )

    print("[3/7] Rebuilding repeated 5-fold OOF probabilities...")
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    sums = {model_name: np.zeros(len(y_train)) for model_name in MODELS}
    counts = {model_name: np.zeros(len(y_train), dtype=int) for model_name in MODELS}
    fold_rows = []
    for split_number, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        repeat = (split_number - 1) // N_SPLITS + 1
        fold = (split_number - 1) % N_SPLITS + 1
        X_fit, X_val = X_train.iloc[fit_idx], X_train.iloc[val_idx]
        y_fit, y_val = y_train.iloc[fit_idx], y_train.iloc[val_idx]
        base_ratio = float((y_fit == 0).sum() / (y_fit == 1).sum())
        started = time.perf_counter()
        for model_name in MODELS:
            config = configs[model_name]
            fit_ready, val_ready, _, _, _ = fit_preprocessor(
                X_fit, y_fit, X_val, config["top_k"]
            )
            weight = base_ratio * float(config["weight_multiplier"])
            model = build_tuned_model(config, weight, RANDOM_STATE + split_number)
            model.fit(fit_ready, y_fit)
            fold_probability = model.predict_proba(val_ready)[:, 1]
            sums[model_name][val_idx] += fold_probability
            counts[model_name][val_idx] += 1
            fold_rows.append(
                {
                    "model": model_name,
                    "config_id": config["config_id"],
                    "repeat": repeat,
                    "fold": fold,
                    "pr_auc": average_precision_score(y_val, fold_probability),
                    "roc_auc": roc_auc_score(y_val, fold_probability),
                }
            )
        print(
            f"  repeat {repeat}/{N_REPEATS}, fold {fold}/{N_SPLITS}: "
            f"{time.perf_counter() - started:.1f}s"
        )

    probabilities = {}
    for model_name in MODELS:
        if not np.all(counts[model_name] == N_REPEATS):
            raise RuntimeError(f"Incomplete OOF coverage for {model_name}")
        probabilities[model_name] = sums[model_name] / counts[model_name]

    print("[4/7] Selecting operating thresholds from OOF predictions only...")
    operating_rows = []
    operating_thresholds: dict[str, dict[str, float]] = {}
    for model_name in MODELS:
        model_thresholds = {}
        balanced = select_threshold(
            y_train.to_numpy(), probabilities[model_name], MIN_RECALL, "f2"
        )
        model_thresholds["balanced_f2"] = balanced
        operating_rows.append(
            {
                "model": model_name,
                "operating_point": "balanced_f2",
                "target_recall": MIN_RECALL,
                **metrics_at_threshold(
                    y_train.to_numpy(), probabilities[model_name], balanced
                ),
            }
        )
        for target in TARGET_RECALLS:
            threshold = select_threshold(
                y_train.to_numpy(), probabilities[model_name], target, "precision"
            )
            key = f"recall_{int(target * 100)}"
            model_thresholds[key] = threshold
            operating_rows.append(
                {
                    "model": model_name,
                    "operating_point": key,
                    "target_recall": target,
                    **metrics_at_threshold(
                        y_train.to_numpy(), probabilities[model_name], threshold
                    ),
                }
            )
        operating_thresholds[model_name] = model_thresholds
    operating_table = pd.DataFrame(operating_rows)

    print("[5/7] Quantifying PR-AUC uncertainty with paired bootstrap...")
    bootstrap_draws, bootstrap_summary = paired_stratified_bootstrap(
        y_train.to_numpy(), probabilities
    )
    delta = bootstrap_summary["catboost_minus_xgboost"]
    print(
        f"  CatBoost-XGBoost PR-AUC delta={delta['mean']:.4f} "
        f"(95% CI {delta['ci_95_low']:.4f} to {delta['ci_95_high']:.4f})"
    )

    print("[6/7] Fitting both models on full train and confirming on fixed test...")
    test_rows = []
    model_bundles = {}
    for model_name in MODELS:
        config = configs[model_name]
        fit_ready, test_ready, imputer, selector, selected_features = fit_preprocessor(
            X_train, y_train, X_test, config["top_k"]
        )
        base_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
        weight = base_ratio * float(config["weight_multiplier"])
        model = build_tuned_model(config, weight, RANDOM_STATE)
        model.fit(fit_ready, y_train)
        test_probability = model.predict_proba(test_ready)[:, 1]
        for point_name, threshold in operating_thresholds[model_name].items():
            target = MIN_RECALL if point_name == "balanced_f2" else int(point_name[-2:]) / 100
            test_rows.append(
                {
                    "model": model_name,
                    "operating_point": point_name,
                    "target_recall": target,
                    **metrics_at_threshold(y_test.to_numpy(), test_probability, threshold),
                }
            )
        bundle = {
            "model": model,
            "imputer": imputer,
            "selector": selector,
            "model_name": model_name,
            "config_id": config["config_id"],
            "parameters": config,
            "primary_threshold": operating_thresholds[model_name]["balanced_f2"],
            "operating_thresholds": operating_thresholds[model_name],
            "input_features": X.columns.tolist(),
            "selected_features": selected_features,
            "removed_high_missing_features": high_missing,
            "removed_constant_features": constants,
            "label_mapping": {-1: 0, 1: 1},
            "random_state": RANDOM_STATE,
        }
        model_bundles[model_name] = bundle
        joblib.dump(bundle, output_dir / f"{model_name.lower()}_model.joblib")

    print("[7/7] Saving tables, plots, and report...")
    test_table = pd.DataFrame(test_rows)
    oof_frame = pd.DataFrame(
        {
            "train_row_index": X_train.index,
            "label": y_train.to_numpy(),
            "catboost_probability": probabilities["CatBoost"],
            "xgboost_probability": probabilities["XGBoost"],
        }
    )
    operating_table.to_csv(output_dir / "oof_operating_points.csv", index=False)
    test_table.to_csv(output_dir / "test_confirmation_by_threshold.csv", index=False)
    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "repeated_cv_fold_auc.csv", index=False)
    bootstrap_draws.to_csv(output_dir / "bootstrap_pr_auc_draws.csv", index=False)
    (output_dir / "bootstrap_summary.json").write_text(
        json.dumps(bootstrap_summary, indent=2), encoding="utf-8"
    )
    (output_dir / "selected_configurations.json").write_text(
        json.dumps(configs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plots(output_dir, y_train.to_numpy(), probabilities, operating_table)

    balanced_oof = operating_table.loc[
        operating_table["operating_point"] == "balanced_f2"
    ].set_index("model")
    balanced_test = test_table.loc[
        test_table["operating_point"] == "balanced_f2"
    ].set_index("model")
    report_lines = [
        "# CatBoost vs XGBoost 최종 후보 비교",
        "",
        f"- 검증: RepeatedStratifiedKFold {N_SPLITS}-fold × {N_REPEATS}회",
        "- 모든 median 대체와 변수 선택은 각 fold 학습 부분에서만 적합",
        "- 모델과 임계값 선택에는 고정 test를 사용하지 않음",
        f"- 양성 비율 기준선: {float(y_train.mean()):.4f}",
        "",
        "## 반복 OOF 균형형 임계값",
        "",
        "| 모델 | PR-AUC | threshold | precision | recall | F2 | FP | FN | 정상 100개당 오탐 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODELS:
        row = balanced_oof.loc[model_name]
        report_lines.append(
            f"| {model_name} | {row['pr_auc']:.4f} | {row['threshold']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f2']:.4f} | "
            f"{int(row['fp'])} | {int(row['fn'])} | "
            f"{row['false_alarms_per_100_normal']:.1f} |"
        )
    report_lines.extend(
        [
            "",
            "## PR-AUC 차이의 불확실성",
            "",
            f"- CatBoost − XGBoost 평균 차이: {delta['mean']:.4f}",
            f"- paired stratified bootstrap 95% CI: "
            f"[{delta['ci_95_low']:.4f}, {delta['ci_95_high']:.4f}]",
            f"- CatBoost PR-AUC가 더 높았던 bootstrap 비율: "
            f"{bootstrap_summary['probability_catboost_pr_auc_greater']:.3f}",
            "",
            "## 고정 test 확인 — 균형형 임계값",
            "",
            "| 모델 | PR-AUC | precision | recall | F2 | TP | FP | FN | TN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in MODELS:
        row = balanced_test.loc[model_name]
        report_lines.append(
            f"| {model_name} | {row['pr_auc']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f2']:.4f} | {int(row['tp'])} | "
            f"{int(row['fp'])} | {int(row['fn'])} | {int(row['tn'])} |"
        )
    report_lines.extend(
        [
            "",
            "고정 test는 앞선 실험에서 이미 관찰했으므로 외부 검증으로 간주할 수 없습니다.",
            "최종 운영 평가는 신규 시간대 또는 신규 lot 데이터로 다시 수행해야 합니다.",
            "",
            "## 권장 역할",
            "",
            "- CatBoost: 정밀도와 F2를 고려한 기본 균형형 후보",
            "- XGBoost: 낮은 임계값에서 불량 누락을 줄이는 고재현율 후보",
            "- 실제 배포 전에는 정상 100개당 허용 가능한 오탐 수로 운영 임계값을 확정",
        ]
    )
    (output_dir / "summary.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(f"  results={output_dir}")


if __name__ == "__main__":
    main()
