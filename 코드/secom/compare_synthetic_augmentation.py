"""Compare CTGAN, TVAE, SMOTE, and class weighting on UCI SECOM.

This is a leakage-safe exploratory experiment:
- the real test set is never augmented;
- imputation, feature selection, scaling, and synthesis are fitted inside each
  training fold only;
- final test thresholds are medians of validation-fold F1-optimal thresholds.
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
from ctgan import CTGAN, TVAE
from imblearn.over_sampling import SMOTE
from scipy.stats import ks_2samp
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from train_compare_models import (
    MODEL_NAMES,
    N_SPLITS,
    RANDOM_STATE,
    TEST_SIZE,
    build_model,
    classification_metrics,
    load_data,
    optimal_f1_threshold,
    structural_filter,
)


TOP_K_FEATURES = 50
SYNTH_EPOCHS = 75
SYNTH_RATIOS = (0.50, 0.75, 1.00)


def learned_preprocess(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    SimpleImputer,
    SelectKBest,
    StandardScaler,
    list[str],
]:
    """Fit median, supervised selector, and scaler on training rows only."""
    imputer = SimpleImputer(strategy="median")
    X_fit_imputed = imputer.fit_transform(X_fit)
    X_eval_imputed = imputer.transform(X_eval)

    selector = SelectKBest(score_func=f_classif, k=TOP_K_FEATURES)
    X_fit_selected = selector.fit_transform(X_fit_imputed, y_fit)
    X_eval_selected = selector.transform(X_eval_imputed)
    selected_features = X_fit.columns[selector.get_support()].tolist()

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit_selected)
    X_eval_scaled = scaler.transform(X_eval_selected)
    return (
        X_fit_scaled,
        X_eval_scaled,
        imputer,
        selector,
        scaler,
        selected_features,
    )


def fit_generator(
    method: str, real_minority: pd.DataFrame, seed: int
) -> CTGAN | TVAE:
    """Fit a compact CPU generator to minority-only, fold-local data."""
    if method == "CTGAN":
        generator: CTGAN | TVAE = CTGAN(
            embedding_dim=64,
            generator_dim=(128, 128),
            discriminator_dim=(128, 128),
            batch_size=50,
            epochs=SYNTH_EPOCHS,
            pac=10,
            verbose=False,
            enable_gpu=False,
        )
    elif method == "TVAE":
        generator = TVAE(
            embedding_dim=64,
            compress_dims=(128, 128),
            decompress_dims=(128, 128),
            batch_size=50,
            epochs=SYNTH_EPOCHS,
            verbose=False,
            enable_gpu=False,
        )
    else:
        raise ValueError(f"Unknown generator: {method}")

    generator.set_random_state(seed)
    generator.fit(real_minority, discrete_columns=())
    return generator


def sample_and_clip(
    generator: CTGAN | TVAE,
    real_minority: pd.DataFrame,
    num_rows: int,
) -> pd.DataFrame:
    """Sample minority rows and clip each value to observed fold-train bounds."""
    synthetic = generator.sample(num_rows).astype(float)
    lower = real_minority.min(axis=0)
    upper = real_minority.max(axis=0)
    return synthetic.clip(lower=lower, upper=upper, axis="columns")


def synthetic_quality(
    real_minority: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> dict[str, float]:
    """Compute distribution, correlation, and exact-copy diagnostics."""
    ks_scores = [
        1.0 - ks_2samp(real_minority[column], synthetic[column]).statistic
        for column in real_minority.columns
    ]
    real_corr = np.nan_to_num(real_minority.corr().to_numpy(), nan=0.0)
    synth_corr = np.nan_to_num(synthetic.corr().to_numpy(), nan=0.0)
    correlation_mae = float(np.mean(np.abs(real_corr - synth_corr)))

    neighbors = NearestNeighbors(n_neighbors=1).fit(real_minority.to_numpy())
    distances = neighbors.kneighbors(synthetic.to_numpy(), return_distance=True)[0][:, 0]
    return {
        "ks_similarity_mean": float(np.mean(ks_scores)),
        "correlation_similarity": float(max(0.0, 1.0 - correlation_mae / 2.0)),
        "nearest_real_distance_median": float(np.median(distances)),
        "exact_copy_rate": float(np.mean(distances < 1e-8)),
    }


def target_synthetic_count(y: np.ndarray, ratio: float) -> int:
    majority = int((y == 0).sum())
    minority = int((y == 1).sum())
    return max(0, int(round(majority * ratio)) - minority)


def make_training_sets(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    seed: int,
    context: str,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    list[dict[str, float | int | str]],
    dict[str, CTGAN | TVAE],
    dict[str, pd.DataFrame],
]:
    """Create all augmentation variants once for one fold/final training split."""
    sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "scale_pos_weight": (X_fit, y_fit)
    }
    quality_rows: list[dict[str, float | int | str]] = []
    generators: dict[str, CTGAN | TVAE] = {}
    max_samples: dict[str, pd.DataFrame] = {}

    smote = SMOTE(sampling_strategy=1.0, random_state=seed)
    sets["SMOTE_1.00"] = smote.fit_resample(X_fit, y_fit)

    columns = [f"selected_{i}" for i in range(X_fit.shape[1])]
    real_minority = pd.DataFrame(X_fit[y_fit == 1], columns=columns)
    max_needed = target_synthetic_count(y_fit, max(SYNTH_RATIOS))

    for offset, method in enumerate(("CTGAN", "TVAE"), start=1):
        started = time.perf_counter()
        generator = fit_generator(method, real_minority, seed + offset * 1000)
        fit_seconds = time.perf_counter() - started
        synthetic_max = sample_and_clip(generator, real_minority, max_needed)
        generators[method] = generator
        max_samples[method] = synthetic_max

        quality = synthetic_quality(real_minority, synthetic_max)
        quality_rows.append(
            {
                "context": context,
                "generator": method,
                "real_minority_rows": len(real_minority),
                "synthetic_rows": len(synthetic_max),
                "fit_seconds": fit_seconds,
                **quality,
            }
        )
        print(
            f"    {method}: fit={fit_seconds:.1f}s, generated={max_needed}, "
            f"KS={quality['ks_similarity_mean']:.3f}, "
            f"corr={quality['correlation_similarity']:.3f}, "
            f"copies={quality['exact_copy_rate']:.3f}"
        )

        for ratio in SYNTH_RATIOS:
            needed = target_synthetic_count(y_fit, ratio)
            synthetic = synthetic_max.iloc[:needed].to_numpy()
            augmented_X = np.vstack([X_fit, synthetic])
            augmented_y = np.concatenate([y_fit, np.ones(needed, dtype=int)])
            sets[f"{method}_{ratio:.2f}"] = (augmented_X, augmented_y)

    return sets, quality_rows, generators, max_samples


def evaluate_fold(
    model_name: str,
    strategy: str,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold: int,
) -> dict[str, float | int | str]:
    ratio = float((y_fit == 0).sum() / (y_fit == 1).sum())
    model_strategy = "scale_pos_weight" if strategy == "scale_pos_weight" else strategy
    model = build_model(model_name, model_strategy, ratio, RANDOM_STATE + fold)
    started = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - started
    probabilities = model.predict_proba(X_val)[:, 1]
    threshold = optimal_f1_threshold(y_val, probabilities)
    metrics = classification_metrics(y_val, probabilities, threshold)
    return {
        "model": model_name,
        "strategy": strategy,
        "fold": fold,
        "fit_seconds": fit_seconds,
        "train_rows_after": len(y_fit),
        "train_positives_after": int(y_fit.sum()),
        **metrics,
    }


def summarize_cv(fold_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "mcc",
        "threshold",
        "fit_seconds",
    ]
    summary = fold_results.groupby(["model", "strategy"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values(
        ["pr_auc_mean", "f1_mean"], ascending=False
    )


def strategy_family(strategy: str) -> str:
    return strategy.split("_", maxsplit=1)[0]


def save_plots(
    output_dir: Path, cv_summary: pd.DataFrame, test_results: pd.DataFrame
) -> None:
    top = cv_summary.head(12).iloc[::-1]
    labels = (top["model"] + " / " + top["strategy"]).tolist()
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(labels, top["pr_auc_mean"], xerr=top["pr_auc_std"], color="#386fa4")
    ax.set_xlabel("5-fold PR-AUC (mean +/- SD)")
    ax.set_title("Top synthetic-augmentation configurations")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "cv_top_pr_auc.png", dpi=180)
    plt.close(fig)

    best_families = (
        test_results.sort_values("cv_pr_auc_mean", ascending=False)
        .groupby("family", as_index=False)
        .first()
    )
    labels = (best_families["model"] + "\n" + best_families["strategy"]).tolist()
    positions = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, metric in enumerate(("pr_auc", "precision", "recall", "f1")):
        ax.bar(
            positions + (index - 1.5) * width,
            best_families[metric],
            width,
            label=metric,
        )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Untouched test score")
    ax.set_title("CV-selected best configuration in each augmentation family")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "test_best_by_family.png", dpi=180)
    plt.close(fig)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "결과물" / "secom" / "synthetic_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("[1/7] Loading and filtering SECOM...")
    X_raw, y, timestamps = load_data(project_dir)
    X, high_missing, constants = structural_filter(X_raw)
    print(
        f"  rows={len(y)}, raw={X_raw.shape[1]}, retained={X.shape[1]}, "
        f"normal={(y == 0).sum()}, defect={(y == 1).sum()}"
    )

    print("[2/7] Locking untouched stratified test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(
        f"  train={len(y_train)} ({int(y_train.sum())} defects), "
        f"test={len(y_test)} ({int(y_test.sum())} defects)"
    )

    print("[3-5/7] Running leakage-safe 5-fold generation and model comparison...")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: list[dict[str, float | int | str]] = []
    quality_rows: list[dict[str, float | int | str]] = []

    for fold, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        print(f"\n  Fold {fold}/{N_SPLITS}")
        X_fit, X_val = X_train.iloc[fit_idx], X_train.iloc[val_idx]
        y_fit, y_val = y_train.iloc[fit_idx], y_train.iloc[val_idx]
        X_fit_ready, X_val_ready, _, _, _, selected = learned_preprocess(
            X_fit, y_fit, X_val
        )
        print(f"    selected={len(selected)} fold-train features")
        training_sets, fold_quality, _, _ = make_training_sets(
            X_fit_ready,
            y_fit.to_numpy(),
            RANDOM_STATE + fold,
            context=f"fold_{fold}",
        )
        quality_rows.extend(fold_quality)

        for strategy, (X_augmented, y_augmented) in training_sets.items():
            for model_name in MODEL_NAMES:
                row = evaluate_fold(
                    model_name,
                    strategy,
                    X_augmented,
                    y_augmented,
                    X_val_ready,
                    y_val.to_numpy(),
                    fold,
                )
                fold_rows.append(row)
        current = pd.DataFrame(fold_rows)
        fold_best = current[current["fold"] == fold].sort_values("pr_auc", ascending=False).iloc[0]
        print(
            f"    fold best={fold_best['model']}/{fold_best['strategy']}, "
            f"PR-AUC={fold_best['pr_auc']:.3f}, F1={fold_best['f1']:.3f}"
        )

    fold_results = pd.DataFrame(fold_rows)
    cv_summary = summarize_cv(fold_results)
    quality_results = pd.DataFrame(quality_rows)
    fold_results.to_csv(output_dir / "cv_fold_results.csv", index=False)
    cv_summary.to_csv(output_dir / "cv_summary.csv", index=False)
    quality_results.to_csv(output_dir / "synthetic_quality.csv", index=False)

    print("\n[6/7] Refitting generators and all configurations on full train...")
    (
        X_train_ready,
        X_test_ready,
        imputer,
        selector,
        scaler,
        selected_features,
    ) = learned_preprocess(X_train, y_train, X_test)
    final_sets, final_quality, generators, synthetic_samples = make_training_sets(
        X_train_ready,
        y_train.to_numpy(),
        RANDOM_STATE,
        context="full_train",
    )
    quality_results = pd.concat(
        [quality_results, pd.DataFrame(final_quality)], ignore_index=True
    )
    quality_results.to_csv(output_dir / "synthetic_quality.csv", index=False)

    for method, generator in generators.items():
        generator.save(output_dir / f"{method.lower()}_generator.pkl")
        original_scale = scaler.inverse_transform(synthetic_samples[method])
        pd.DataFrame(original_scale, columns=selected_features).to_csv(
            output_dir / f"{method.lower()}_synthetic_defects.csv", index=False
        )

    test_rows: list[dict[str, float | int | str]] = []
    bundles: dict[tuple[str, str], dict] = {}
    for _, cv_row in cv_summary.iterrows():
        model_name = str(cv_row["model"])
        strategy = str(cv_row["strategy"])
        thresholds = fold_results.loc[
            (fold_results["model"] == model_name)
            & (fold_results["strategy"] == strategy),
            "threshold",
        ]
        threshold = float(thresholds.median())
        X_fit, y_fit = final_sets[strategy]
        ratio = float((y_fit == 0).sum() / (y_fit == 1).sum())
        model_strategy = "scale_pos_weight" if strategy == "scale_pos_weight" else strategy
        model = build_model(model_name, model_strategy, ratio, RANDOM_STATE)
        model.fit(X_fit, y_fit)
        probabilities = model.predict_proba(X_test_ready)[:, 1]
        metrics = classification_metrics(y_test.to_numpy(), probabilities, threshold)
        test_rows.append(
            {
                "model": model_name,
                "strategy": strategy,
                "family": strategy_family(strategy),
                "cv_pr_auc_mean": float(cv_row["pr_auc_mean"]),
                "train_rows_after": len(y_fit),
                "train_positives_after": int(y_fit.sum()),
                **metrics,
            }
        )
        bundles[(model_name, strategy)] = {
            "model": model,
            "imputer": imputer,
            "selector": selector,
            "scaler": scaler,
            "threshold": threshold,
            "model_name": model_name,
            "strategy": strategy,
            "input_features": X.columns.tolist(),
            "selected_features": selected_features,
            "label_mapping": {-1: 0, 1: 1},
            "random_state": RANDOM_STATE,
        }

    test_results = pd.DataFrame(test_rows).sort_values(
        "cv_pr_auc_mean", ascending=False
    )
    test_results.to_csv(output_dir / "test_results.csv", index=False)
    best = test_results.iloc[0]
    joblib.dump(
        bundles[(str(best["model"]), str(best["strategy"]))],
        output_dir / "best_cv_selected_model.joblib",
    )
    (output_dir / "selected_features.json").write_text(
        json.dumps(selected_features, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    save_plots(output_dir, cv_summary, test_results)

    best_by_family = (
        test_results.groupby("family", as_index=False, sort=False).first()
    )
    display = best_by_family[
        [
            "family",
            "model",
            "strategy",
            "cv_pr_auc_mean",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
    ].copy()
    numeric = display.select_dtypes(include="number").columns
    display[numeric] = display[numeric].round(4)
    report = f"""# 생성형 증강 비교 결과

- 실제 입력: {len(y)}행, 불량 {int(y.sum())}건
- test: {len(y_test)}행, 불량 {int(y_test.sum())}건 — 합성/증강 미적용
- fold 내부 전처리: median → ANOVA 상위 {TOP_K_FEATURES}개 → 표준화
- 생성 모델: CTGAN/TVAE, minority-only, {SYNTH_EPOCHS} epochs
- 생성 목표 비율: {', '.join(str(value) for value in SYNTH_RATIOS)}
- 임계값: fold별 F1 최대, 최종은 fold 임계값 중앙값

## 전략군별 CV 1위와 실제 test 성능

{display.to_markdown(index=False)}

## 전체 CV 1위

**{best['model']} / {best['strategy']}**  
CV PR-AUC={best['cv_pr_auc_mean']:.4f}, test PR-AUC={best['pr_auc']:.4f},
precision={best['precision']:.4f}, recall={best['recall']:.4f}, F1={best['f1']:.4f}

합성 행 수는 실측 정보량과 동일하지 않습니다. 품질 지표와 untouched real test의
downstream 성능을 함께 보고 사용 여부를 결정해야 합니다.
"""
    (output_dir / "summary.md").write_text(report, encoding="utf-8")

    metadata = {
        "rows": len(y),
        "raw_features": X_raw.shape[1],
        "retained_features": X.shape[1],
        "selected_features": TOP_K_FEATURES,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "test_defects": int(y_test.sum()),
        "synth_epochs": SYNTH_EPOCHS,
        "synth_ratios": SYNTH_RATIOS,
        "random_state": RANDOM_STATE,
        "best_model": str(best["model"]),
        "best_strategy": str(best["strategy"]),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n[7/7] Complete")
    print(display.to_string(index=False))
    print(f"  CV winner: {best['model']} / {best['strategy']}")
    print(f"  Results: {output_dir}")


if __name__ == "__main__":
    main()
