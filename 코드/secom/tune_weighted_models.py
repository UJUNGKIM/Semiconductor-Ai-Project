"""Repeated-CV tuning for weighted SECOM defect classifiers.

Model selection and the operating threshold use only the fixed training split.
The previously defined 20% holdout is evaluated once after configuration choice.
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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterSampler,
    RepeatedStratifiedKFold,
    train_test_split,
)
from xgboost import XGBClassifier

from train_compare_models import RANDOM_STATE, TEST_SIZE, load_data, structural_filter


N_SPLITS = 5
N_REPEATS = 3
N_RANDOM_CONFIGS = 5
MIN_RECALL = 0.60
TOP_K_OPTIONS = (50, 100, 200, "all")


def native(value):
    return value.item() if isinstance(value, np.generic) else value


def make_configurations() -> list[dict]:
    """Create one reference and five reproducible random configs per model."""
    spaces = {
        "XGBoost": {
            "top_k": TOP_K_OPTIONS,
            "weight_multiplier": (0.75, 1.0, 1.25),
            "n_estimators": (250, 400, 600),
            "learning_rate": (0.02, 0.04, 0.07),
            "max_depth": (3, 4, 5),
            "min_child_weight": (1, 3, 6),
            "subsample": (0.7, 0.85, 1.0),
            "colsample_bytree": (0.5, 0.7, 0.9),
            "reg_alpha": (0.0, 0.1, 0.5),
            "reg_lambda": (1.0, 5.0, 10.0),
        },
        "CatBoost": {
            "top_k": TOP_K_OPTIONS,
            "weight_multiplier": (0.75, 1.0, 1.25),
            "iterations": (250, 400, 600),
            "learning_rate": (0.02, 0.04, 0.07),
            "depth": (4, 6, 8),
            "l2_leaf_reg": (3.0, 7.0, 12.0),
            "random_strength": (0.5, 1.0, 2.0),
            "border_count": (64, 128),
        },
        "LightGBM": {
            "top_k": TOP_K_OPTIONS,
            "weight_multiplier": (0.75, 1.0, 1.25),
            "n_estimators": (250, 400, 600),
            "learning_rate": (0.02, 0.04, 0.07),
            "num_leaves": (15, 31, 63),
            "max_depth": (-1, 5, 8),
            "min_child_samples": (10, 20, 35),
            "subsample": (0.7, 0.85, 1.0),
            "colsample_bytree": (0.5, 0.7, 0.9),
            "reg_alpha": (0.0, 0.1, 0.5),
            "reg_lambda": (1.0, 5.0, 10.0),
        },
    }
    references = {
        "XGBoost": {
            "top_k": 50,
            "weight_multiplier": 1.0,
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 2,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        "CatBoost": {
            "top_k": 50,
            "weight_multiplier": 1.0,
            "iterations": 400,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "border_count": 128,
        },
        "LightGBM": {
            "top_k": 50,
            "weight_multiplier": 1.0,
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    }

    configurations: list[dict] = []
    for model_index, model_name in enumerate(("XGBoost", "CatBoost", "LightGBM")):
        candidates = [references[model_name]]
        candidates.extend(
            ParameterSampler(
                spaces[model_name],
                n_iter=N_RANDOM_CONFIGS,
                random_state=RANDOM_STATE + model_index,
            )
        )
        for number, candidate in enumerate(candidates):
            config = {key: native(value) for key, value in candidate.items()}
            config.update(
                {
                    "model": model_name,
                    "config_id": f"{model_name.lower()}_{number:02d}",
                }
            )
            configurations.append(config)
    return configurations


def fit_preprocessor(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
    top_k: int | str,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, SelectKBest, list[str]]:
    imputer = SimpleImputer(strategy="median")
    X_fit_imputed = imputer.fit_transform(X_fit)
    X_eval_imputed = imputer.transform(X_eval)
    selector = SelectKBest(score_func=f_classif, k=top_k)
    X_fit_selected = selector.fit_transform(X_fit_imputed, y_fit)
    X_eval_selected = selector.transform(X_eval_imputed)
    selected_features = X_fit.columns[selector.get_support()].tolist()
    return X_fit_selected, X_eval_selected, imputer, selector, selected_features


def build_tuned_model(config: dict, scale_pos_weight: float, seed: int):
    model_name = config["model"]
    ignored = {"model", "config_id", "top_k", "weight_multiplier"}
    params = {key: value for key, value in config.items() if key not in ignored}
    if model_name == "XGBoost":
        return XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    if model_name == "CatBoost":
        return CatBoostClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=seed,
        )
    if model_name == "LightGBM":
        return LGBMClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            objective="binary",
            verbosity=-1,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(model_name)


def constrained_f2_threshold(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5, 0.0
    precision = precision[:-1]
    recall = recall[:-1]
    f2 = 5.0 * precision * recall / (4.0 * precision + recall + 1e-12)
    eligible = np.flatnonzero(recall >= MIN_RECALL)
    if eligible.size == 0:
        index = int(np.nanargmax(recall))
    else:
        eligible_f2 = f2[eligible]
        index = int(eligible[int(np.nanargmax(eligible_f2))])
    return float(thresholds[index]), float(f2[index])


def threshold_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "threshold": threshold,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def save_plot(output_dir: Path, summary: pd.DataFrame) -> None:
    top = summary.head(12).iloc[::-1]
    labels = (top["model"] + " / " + top["config_id"]).tolist()
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = ["#d1495b" if value >= MIN_RECALL else "#4472a3" for value in top["oof_recall"]]
    ax.barh(labels, top["oof_f2"], color=colors)
    ax.set_xlabel("Repeated out-of-fold F2")
    ax.set_title(f"Weighted-model tuning (red: recall >= {MIN_RECALL:.2f})")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "repeated_cv_f2_ranking.png", dpi=180)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "결과물" / "secom" / "tuning_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/6] Loading SECOM and recreating the fixed split...")
    X_raw, y, _ = load_data(project_dir)
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

    print("[2/6] Building deterministic hyperparameter candidates...")
    configs = make_configurations()
    config_by_id = {config["config_id"]: config for config in configs}
    print(f"  {len(configs)} configs x {N_SPLITS} folds x {N_REPEATS} repeats")

    print("[3/6] Running repeated stratified out-of-fold evaluation...")
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    probability_sums = {
        config["config_id"]: np.zeros(len(y_train), dtype=float) for config in configs
    }
    prediction_counts = {
        config["config_id"]: np.zeros(len(y_train), dtype=int) for config in configs
    }
    fold_rows: list[dict] = []
    unique_top_k = sorted(
        {config["top_k"] for config in configs},
        key=lambda value: 9999 if value == "all" else int(value),
    )

    for split_number, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        repeat = (split_number - 1) // N_SPLITS + 1
        fold = (split_number - 1) % N_SPLITS + 1
        X_fit, X_val = X_train.iloc[fit_idx], X_train.iloc[val_idx]
        y_fit, y_val = y_train.iloc[fit_idx], y_train.iloc[val_idx]
        base_ratio = float((y_fit == 0).sum() / (y_fit == 1).sum())
        prepared = {}
        for top_k in unique_top_k:
            fit_ready, val_ready, _, _, _ = fit_preprocessor(
                X_fit, y_fit, X_val, top_k
            )
            prepared[str(top_k)] = (fit_ready, val_ready)

        started_split = time.perf_counter()
        for config in configs:
            fit_ready, val_ready = prepared[str(config["top_k"])]
            weight = base_ratio * float(config["weight_multiplier"])
            model = build_tuned_model(
                config, weight, RANDOM_STATE + split_number
            )
            started = time.perf_counter()
            model.fit(fit_ready, y_fit)
            elapsed = time.perf_counter() - started
            probabilities = model.predict_proba(val_ready)[:, 1]
            config_id = config["config_id"]
            probability_sums[config_id][val_idx] += probabilities
            prediction_counts[config_id][val_idx] += 1
            fold_threshold, _ = constrained_f2_threshold(
                y_val.to_numpy(), probabilities
            )
            fold_metrics = threshold_metrics(
                y_val.to_numpy(), probabilities, fold_threshold
            )
            fold_rows.append(
                {
                    "config_id": config_id,
                    "model": config["model"],
                    "repeat": repeat,
                    "fold": fold,
                    "top_k": config["top_k"],
                    "weight_multiplier": config["weight_multiplier"],
                    "fit_seconds": elapsed,
                    **fold_metrics,
                }
            )
        print(
            f"  repeat {repeat}/{N_REPEATS}, fold {fold}/{N_SPLITS}: "
            f"{time.perf_counter() - started_split:.1f}s"
        )

    print("[4/6] Selecting configuration and threshold from averaged OOF predictions...")
    summary_rows = []
    for config in configs:
        config_id = config["config_id"]
        counts = prediction_counts[config_id]
        if not np.all(counts == N_REPEATS):
            raise RuntimeError(f"Incomplete OOF coverage for {config_id}")
        probabilities = probability_sums[config_id] / counts
        threshold, _ = constrained_f2_threshold(y_train.to_numpy(), probabilities)
        metrics = threshold_metrics(y_train.to_numpy(), probabilities, threshold)
        summary_rows.append(
            {
                "config_id": config_id,
                "model": config["model"],
                "top_k": config["top_k"],
                "weight_multiplier": config["weight_multiplier"],
                **metrics,
                "parameters_json": json.dumps(config, ensure_ascii=False),
            }
        )

    summary = pd.DataFrame(summary_rows).rename(
        columns={
            "pr_auc": "oof_pr_auc",
            "roc_auc": "oof_roc_auc",
            "threshold": "oof_threshold",
            "precision": "oof_precision",
            "recall": "oof_recall",
            "f2": "oof_f2",
            "tp": "oof_tp",
            "fp": "oof_fp",
            "fn": "oof_fn",
            "tn": "oof_tn",
        }
    )
    summary["meets_recall_target"] = summary["oof_recall"] >= MIN_RECALL
    summary = summary.sort_values(
        ["meets_recall_target", "oof_f2", "oof_pr_auc"],
        ascending=[False, False, False],
    )
    best = summary.iloc[0]
    best_config = config_by_id[str(best["config_id"])]
    print(
        f"  selected={best['model']}/{best['config_id']}, "
        f"OOF PR-AUC={best['oof_pr_auc']:.4f}, "
        f"recall={best['oof_recall']:.4f}, F2={best['oof_f2']:.4f}, "
        f"threshold={best['oof_threshold']:.4f}"
    )

    print("[5/6] Fitting selected model on full train and confirming on fixed test...")
    X_fit, X_eval, imputer, selector, selected_features = fit_preprocessor(
        X_train, y_train, X_test, best_config["top_k"]
    )
    final_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    final_weight = final_ratio * float(best_config["weight_multiplier"])
    final_model = build_tuned_model(best_config, final_weight, RANDOM_STATE)
    final_model.fit(X_fit, y_train)
    test_probabilities = final_model.predict_proba(X_eval)[:, 1]
    test_metrics = threshold_metrics(
        y_test.to_numpy(), test_probabilities, float(best["oof_threshold"])
    )
    print(
        f"  test PR-AUC={test_metrics['pr_auc']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, F2={test_metrics['f2']:.4f}"
    )
    print(
        f"  confusion: TP={test_metrics['tp']}, FP={test_metrics['fp']}, "
        f"FN={test_metrics['fn']}, TN={test_metrics['tn']}"
    )

    print("[6/6] Saving reproducible artifacts...")
    bundle = {
        "model": final_model,
        "imputer": imputer,
        "selector": selector,
        "threshold": float(best["oof_threshold"]),
        "model_name": best_config["model"],
        "config_id": best_config["config_id"],
        "parameters": best_config,
        "input_features": X.columns.tolist(),
        "selected_features": selected_features,
        "label_mapping": {-1: 0, 1: 1},
        "minimum_recall_target": MIN_RECALL,
        "random_state": RANDOM_STATE,
    }
    joblib.dump(bundle, output_dir / "tuned_weighted_model.joblib")
    pd.DataFrame(fold_rows).to_csv(output_dir / "repeated_cv_folds.csv", index=False)
    summary.to_csv(output_dir / "tuning_summary.csv", index=False)
    (output_dir / "best_parameters.json").write_text(
        json.dumps(best_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "selected_features.json").write_text(
        json.dumps(selected_features, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    test_record = {
        "model": best_config["model"],
        "config_id": best_config["config_id"],
        "oof_metrics": {
            key: native(best[key])
            for key in (
                "oof_pr_auc",
                "oof_roc_auc",
                "oof_threshold",
                "oof_precision",
                "oof_recall",
                "oof_f2",
            )
        },
        "test_metrics": test_metrics,
        "test_note": "Fixed holdout was observed in earlier experiments; treat as confirmation, not pristine external validation.",
    }
    (output_dir / "test_confirmation.json").write_text(
        json.dumps(test_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plot(output_dir, summary)

    report = f"""# 반복 교차검증 모델 최적화 결과

- 비교: XGBoost, CatBoost, LightGBM 가중치 모델
- 탐색: 모델별 6개 설정, 총 {len(configs)}개
- 검증: RepeatedStratifiedKFold {N_SPLITS}-fold × {N_REPEATS}회
- 목표: OOF recall ≥ {MIN_RECALL:.2f} 조건에서 F2 최대화
- median/변수선택은 각 fold 학습 데이터에서만 적합

## 선택 모델

- 모델: **{best_config['model']}** (`{best_config['config_id']}`)
- 선택 변수 수: {len(selected_features)}
- OOF PR-AUC: {best['oof_pr_auc']:.4f}
- OOF precision: {best['oof_precision']:.4f}
- OOF recall: {best['oof_recall']:.4f}
- OOF F2: {best['oof_f2']:.4f}
- OOF 기반 임계값: {best['oof_threshold']:.4f}

## 고정 test 확인

- PR-AUC: {test_metrics['pr_auc']:.4f}
- ROC-AUC: {test_metrics['roc_auc']:.4f}
- precision: {test_metrics['precision']:.4f}
- recall: {test_metrics['recall']:.4f}
- F2: {test_metrics['f2']:.4f}
- 혼동행렬: TP={test_metrics['tp']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}, TN={test_metrics['tn']}

고정 test는 이전 실험에서 이미 결과를 확인했으므로 완전히 새로운 외부 검증 세트는 아닙니다.
최종 성능 주장은 반복 OOF 결과와 향후 외부 검증을 중심으로 판단해야 합니다.
"""
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(f"  results={output_dir}")


if __name__ == "__main__":
    main()
