"""Strengthen SECOM augmentation with a fold-local diffusion experiment.

The script intentionally keeps every validation/test row real. Median
imputation, feature selection, scaling, diffusion fitting, quality gating, and
threshold discovery are all learned from the current training fold only.

This is a compact DDPM-style research baseline, not a claim that synthetic rows
replace newly collected semiconductor lots.
"""

from __future__ import annotations

import json
import math
import random
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from compare_synthetic_augmentation import (
    TOP_K_FEATURES,
    learned_preprocess,
    target_synthetic_count,
)
from train_compare_models import (
    N_SPLITS,
    RANDOM_STATE,
    TEST_SIZE,
    build_model,
    classification_metrics,
    load_data,
    optimal_f1_threshold,
    structural_filter,
)


MODEL_NAMES = ("CatBoost", "XGBoost")
SYNTH_RATIOS = (0.25, 0.50, 0.75)
DIFFUSION_STEPS = 40
DIFFUSION_EPOCHS = 300
POOL_MULTIPLIER = 3
MIN_CV_GAIN = 0.005


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TimeEmbedding(nn.Module):
    """Deterministic sinusoidal embedding for a diffusion time step."""

    def __init__(self, dimension: int = 32) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, time_index: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10_000) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=time_index.device, dtype=torch.float32)
            * -scale
        )
        angles = time_index.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class DenoisingNetwork(nn.Module):
    """Small CPU-friendly MLP used by the minority-class diffusion model."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.time_embedding = TimeEmbedding(32)
        self.network = nn.Sequential(
            nn.Linear(feature_count + 32, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, feature_count),
        )

    def forward(self, noisy: torch.Tensor, time_index: torch.Tensor) -> torch.Tensor:
        embedded = self.time_embedding(time_index)
        return self.network(torch.cat((noisy, embedded), dim=1))


class MinorityDiffusion:
    """A compact DDPM-style generator trained on fold-local defect rows."""

    def __init__(
        self,
        feature_count: int,
        seed: int,
        steps: int = DIFFUSION_STEPS,
        epochs: int = DIFFUSION_EPOCHS,
    ) -> None:
        self.feature_count = feature_count
        self.seed = seed
        self.steps = steps
        self.epochs = epochs
        self.model = DenoisingNetwork(feature_count)
        self.betas = torch.linspace(1e-4, 0.02, steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.final_loss = float("nan")

    def fit(self, rows: np.ndarray) -> "MinorityDiffusion":
        seed_everything(self.seed)
        self.model = DenoisingNetwork(self.feature_count)
        values = torch.as_tensor(rows, dtype=torch.float32)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-5
        )
        batch_size = min(64, len(values))
        self.model.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(len(values))
            epoch_losses: list[float] = []
            for start in range(0, len(values), batch_size):
                clean = values[permutation[start : start + batch_size]]
                time_index = torch.randint(0, self.steps, (len(clean),))
                noise = torch.randn_like(clean)
                alpha_bar = self.alpha_bars[time_index].unsqueeze(1)
                noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
                predicted_noise = self.model(noisy, time_index)
                loss = nn.functional.mse_loss(predicted_noise, noise)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            self.final_loss = float(np.mean(epoch_losses))
        return self

    @torch.no_grad()
    def sample(self, row_count: int, seed: int | None = None) -> np.ndarray:
        sample_seed = self.seed + 100_000 if seed is None else seed
        generator = torch.Generator().manual_seed(sample_seed)
        values = torch.randn(
            (row_count, self.feature_count), generator=generator
        )
        self.model.eval()
        for step in reversed(range(self.steps)):
            time_index = torch.full((row_count,), step, dtype=torch.long)
            predicted_noise = self.model(values, time_index)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            mean = (
                values - (1.0 - alpha) / (1.0 - alpha_bar).sqrt() * predicted_noise
            ) / alpha.sqrt()
            if step > 0:
                previous_bar = self.alpha_bars[step - 1]
                variance = self.betas[step] * (1.0 - previous_bar) / (1.0 - alpha_bar)
                noise = torch.randn(values.shape, generator=generator)
                values = mean + variance.clamp_min(1e-8).sqrt() * noise
            else:
                values = mean
        return values.cpu().numpy()

    def state_bundle(self) -> dict:
        return {
            "state_dict": self.model.state_dict(),
            "feature_count": self.feature_count,
            "seed": self.seed,
            "steps": self.steps,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
        }


def quality_gate(
    real_minority: np.ndarray,
    synthetic_pool: np.ndarray,
    needed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Reject copies and extreme samples using only fold-train defect geometry."""
    real_neighbors = NearestNeighbors(n_neighbors=2).fit(real_minority)
    real_distances = real_neighbors.kneighbors(real_minority)[0][:, 1]
    synthetic_distances = NearestNeighbors(n_neighbors=1).fit(
        real_minority
    ).kneighbors(synthetic_pool)[0][:, 0]

    reference_median = float(np.median(real_distances))
    lower = max(1e-6, float(np.quantile(real_distances, 0.10)))
    upper = float(np.quantile(real_distances, 0.90))
    accepted = np.flatnonzero(
        (synthetic_distances >= lower) & (synthetic_distances <= upper)
    )

    closeness = np.abs(
        np.log((synthetic_distances + 1e-8) / (reference_median + 1e-8))
    )
    ranked = np.argsort(closeness)
    accepted_set = set(accepted.tolist())
    selected = [int(index) for index in ranked if int(index) in accepted_set]
    accepted_selected = min(len(selected), needed)
    if len(selected) < needed:
        selected_set = set(selected)
        selected.extend(
            int(index) for index in ranked if int(index) not in selected_set
        )
    selected = selected[:needed]
    return synthetic_pool[selected], {
        "pool_rows": len(synthetic_pool),
        "accepted_rows": len(accepted),
        "accepted_rate": float(len(accepted) / len(synthetic_pool)),
        "selected_rows": len(selected),
        "fallback_rows": max(0, needed - accepted_selected),
        "distance_lower_gate": lower,
        "distance_upper_gate": upper,
        "real_to_real_nn_median": reference_median,
    }


def _rbf_mmd(real: np.ndarray, synthetic: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    real = real[rng.choice(len(real), min(250, len(real)), replace=False)]
    synthetic = synthetic[
        rng.choice(len(synthetic), min(250, len(synthetic)), replace=False)
    ]
    cross_squared = cdist(real, synthetic, metric="sqeuclidean")
    positive = cross_squared[cross_squared > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / max(2.0 * bandwidth, 1e-8)
    real_kernel = np.exp(-gamma * cdist(real, real, metric="sqeuclidean"))
    synth_kernel = np.exp(
        -gamma * cdist(synthetic, synthetic, metric="sqeuclidean")
    )
    cross_kernel = np.exp(-gamma * cross_squared)
    return float(max(0.0, real_kernel.mean() + synth_kernel.mean() - 2 * cross_kernel.mean()))


def enhanced_quality_metrics(
    real_minority: np.ndarray,
    synthetic: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Measure fidelity, distinguishability, coverage, and memorization risk."""
    ks_scores = [
        1.0 - ks_2samp(real_minority[:, column], synthetic[:, column]).statistic
        for column in range(real_minority.shape[1])
    ]
    real_corr = np.nan_to_num(np.corrcoef(real_minority, rowvar=False), nan=0.0)
    synth_corr = np.nan_to_num(np.corrcoef(synthetic, rowvar=False), nan=0.0)
    corr_mae = float(np.mean(np.abs(real_corr - synth_corr)))
    correlation_similarity = max(0.0, 1.0 - corr_mae / 2.0)

    real_nn = NearestNeighbors(n_neighbors=2).fit(real_minority)
    real_to_real = real_nn.kneighbors(real_minority)[0][:, 1]
    synth_to_real = NearestNeighbors(n_neighbors=1).fit(
        real_minority
    ).kneighbors(synthetic)[0][:, 0]
    distance_ratio = float(
        np.median(synth_to_real) / max(np.median(real_to_real), 1e-8)
    )
    memorization_threshold = max(float(np.median(real_to_real)) * 0.10, 1e-8)

    real_coverage_distance = NearestNeighbors(n_neighbors=1).fit(
        synthetic
    ).kneighbors(real_minority)[0][:, 0]
    coverage = float(
        np.mean(real_coverage_distance <= np.quantile(real_to_real, 0.95))
    )

    rng = np.random.default_rng(seed)
    balanced_size = min(len(real_minority), len(synthetic))
    synth_subset = synthetic[
        rng.choice(len(synthetic), balanced_size, replace=False)
    ]
    discriminator_X = np.vstack((real_minority, synth_subset))
    discriminator_y = np.concatenate(
        (np.zeros(len(real_minority), dtype=int), np.ones(balanced_size, dtype=int))
    )
    discriminator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    discriminator_auc = float(
        cross_val_score(
            discriminator,
            discriminator_X,
            discriminator_y,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
        ).mean()
    )
    discriminator_auc = max(discriminator_auc, 1.0 - discriminator_auc)
    discriminator_similarity = max(0.0, 1.0 - 2.0 * abs(discriminator_auc - 0.5))
    nn_similarity = float(math.exp(-abs(math.log(max(distance_ratio, 1e-8)))))
    quality_score = (
        0.30 * float(np.mean(ks_scores))
        + 0.25 * correlation_similarity
        + 0.25 * discriminator_similarity
        + 0.10 * coverage
        + 0.10 * nn_similarity
    )
    return {
        "ks_similarity_mean": float(np.mean(ks_scores)),
        "correlation_similarity": correlation_similarity,
        "discriminator_auc": discriminator_auc,
        "discriminator_similarity": discriminator_similarity,
        "mmd_rbf": _rbf_mmd(real_minority, synthetic, seed),
        "real_coverage": coverage,
        "nn_distance_ratio": distance_ratio,
        "exact_copy_rate": float(np.mean(synth_to_real < 1e-8)),
        "memorization_risk_rate": float(
            np.mean(synth_to_real < memorization_threshold)
        ),
        "quality_score": float(quality_score),
    }


def build_training_sets(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    seed: int,
    context: str,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    MinorityDiffusion,
    np.ndarray,
    dict[str, float | int | str],
]:
    """Fit one generator and create ratio variants for a fold."""
    training_sets = {"scale_pos_weight": (X_fit, y_fit)}
    smote = SMOTE(sampling_strategy=1.0, random_state=seed)
    training_sets["SMOTE_1.00"] = smote.fit_resample(X_fit, y_fit)

    real_minority = X_fit[y_fit == 1]
    max_needed = target_synthetic_count(y_fit, max(SYNTH_RATIOS))
    started = time.perf_counter()
    diffusion = MinorityDiffusion(X_fit.shape[1], seed).fit(real_minority)
    fit_seconds = time.perf_counter() - started
    raw_pool = diffusion.sample(max_needed * POOL_MULTIPLIER, seed + 50_000)
    raw_pool = np.clip(raw_pool, real_minority.min(axis=0), real_minority.max(axis=0))
    filtered, gate = quality_gate(real_minority, raw_pool, max_needed)

    rng = np.random.default_rng(seed + 70_000)
    filtered = filtered[rng.permutation(len(filtered))]
    for ratio in SYNTH_RATIOS:
        needed = target_synthetic_count(y_fit, ratio)
        generated = filtered[:needed]
        training_sets[f"DDPM_{ratio:.2f}"] = (
            np.vstack((X_fit, generated)),
            np.concatenate((y_fit, np.ones(needed, dtype=int))),
        )

    quality = {
        "context": context,
        "generator": "DDPM",
        "real_minority_rows": len(real_minority),
        "synthetic_rows": len(filtered),
        "fit_seconds": fit_seconds,
        "final_training_loss": diffusion.final_loss,
        **gate,
        **enhanced_quality_metrics(real_minority, filtered, seed),
    }
    return training_sets, diffusion, filtered, quality


def evaluate_configuration(
    model_name: str,
    strategy: str,
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    fold: int,
) -> dict[str, float | int | str]:
    weighting = "scale_pos_weight" if strategy == "scale_pos_weight" else strategy
    ratio = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
    model = build_model(model_name, weighting, ratio, RANDOM_STATE + fold)
    started = time.perf_counter()
    model.fit(X_fit, y_fit)
    probabilities = model.predict_proba(X_eval)[:, 1]
    threshold = optimal_f1_threshold(y_eval, probabilities)
    return {
        "model": model_name,
        "strategy": strategy,
        "fold": fold,
        "fit_seconds": time.perf_counter() - started,
        "train_rows_after": len(y_fit),
        "train_positives_after": int(y_fit.sum()),
        **classification_metrics(y_eval, probabilities, threshold),
    }


def summarize_cv(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = (
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
    )
    summary = rows.groupby(["model", "strategy"])[list(metrics)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values(
        ["pr_auc_mean", "f1_mean"], ascending=False
    )


def save_plots(
    output_dir: Path,
    summary: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    plotting = summary.sort_values("pr_auc_mean").copy()
    labels = (plotting["model"] + " / " + plotting["strategy"]).tolist()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(labels, plotting["pr_auc_mean"], xerr=plotting["pr_auc_std"], color="#2962a3")
    ax.set_xlabel("Real-validation 5-fold PR-AUC (mean +/- SD)")
    ax.set_title("SECOM: diffusion augmentation comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "cv_pr_auc_comparison.png", dpi=180)
    plt.close(fig)

    metrics = [
        "ks_similarity_mean",
        "correlation_similarity",
        "discriminator_similarity",
        "real_coverage",
        "quality_score",
    ]
    quality_mean = quality[metrics].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(quality_mean.index, quality_mean.values, color="#3a8d7d")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Fold-local DDPM synthetic-data quality")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_quality_dashboard.png", dpi=180)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    project_dir = Path(__file__).resolve().parents[2]
    output_dir = project_dir / "결과물" / "secom" / "enhanced_synthetic_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] SECOM 로드 및 구조 필터링")
    X_raw, y, _ = load_data(project_dir)
    X, _, _ = structural_filter(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(
        f"  train={len(y_train)} (defect={int(y_train.sum())}), "
        f"untouched test={len(y_test)} (defect={int(y_test.sum())})"
    )

    print("[2/6] Fold-local DDPM 생성 + 품질 게이트 + 실제 validation 평가")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: list[dict[str, float | int | str]] = []
    quality_rows: list[dict[str, float | int | str]] = []
    for fold, (fit_index, val_index) in enumerate(cv.split(X_train, y_train), 1):
        print(f"  fold {fold}/{N_SPLITS}")
        X_fit, X_val = X_train.iloc[fit_index], X_train.iloc[val_index]
        y_fit, y_val = y_train.iloc[fit_index], y_train.iloc[val_index]
        X_fit_ready, X_val_ready, _, _, _, _ = learned_preprocess(
            X_fit, y_fit, X_val
        )
        sets, _, _, quality = build_training_sets(
            X_fit_ready,
            y_fit.to_numpy(),
            RANDOM_STATE + fold,
            f"fold_{fold}",
        )
        quality_rows.append(quality)
        print(
            f"    DDPM fit={quality['fit_seconds']:.1f}s, "
            f"quality={quality['quality_score']:.3f}, "
            f"C2ST AUC={quality['discriminator_auc']:.3f}, "
            f"memorization={quality['memorization_risk_rate']:.3f}"
        )
        for strategy, (X_augmented, y_augmented) in sets.items():
            for model_name in MODEL_NAMES:
                fold_rows.append(
                    evaluate_configuration(
                        model_name,
                        strategy,
                        X_augmented,
                        y_augmented,
                        X_val_ready,
                        y_val.to_numpy(),
                        fold,
                    )
                )

    fold_results = pd.DataFrame(fold_rows)
    cv_summary = summarize_cv(fold_results)
    quality_results = pd.DataFrame(quality_rows)
    fold_results.to_csv(output_dir / "cv_fold_results.csv", index=False)
    cv_summary.to_csv(output_dir / "cv_summary.csv", index=False)

    print("[3/6] 전체 train으로 최종 DDPM 재학습")
    (
        X_train_ready,
        X_test_ready,
        imputer,
        selector,
        scaler,
        selected_features,
    ) = learned_preprocess(X_train, y_train, X_test)
    final_sets, diffusion, final_synthetic, final_quality = build_training_sets(
        X_train_ready, y_train.to_numpy(), RANDOM_STATE, "full_train"
    )
    quality_results = pd.concat(
        (quality_results, pd.DataFrame([final_quality])), ignore_index=True
    )
    quality_results.to_csv(output_dir / "synthetic_quality.csv", index=False)
    torch.save(diffusion.state_bundle(), output_dir / "ddpm_generator.pt")
    original_scale = scaler.inverse_transform(final_synthetic)
    pd.DataFrame(original_scale, columns=selected_features).to_csv(
        output_dir / "ddpm_synthetic_defects.csv", index=False
    )

    print("[4/6] CV 임계값을 고정하고 untouched real test 평가")
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
        ratio = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
        weighting = "scale_pos_weight" if strategy == "scale_pos_weight" else strategy
        model = build_model(model_name, weighting, ratio, RANDOM_STATE)
        model.fit(X_fit, y_fit)
        probabilities = model.predict_proba(X_test_ready)[:, 1]
        test_rows.append(
            {
                "model": model_name,
                "strategy": strategy,
                "cv_pr_auc_mean": float(cv_row["pr_auc_mean"]),
                "cv_pr_auc_std": float(cv_row["pr_auc_std"]),
                **classification_metrics(y_test.to_numpy(), probabilities, threshold),
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
    test_results = pd.DataFrame(test_rows).sort_values("cv_pr_auc_mean", ascending=False)
    test_results.to_csv(output_dir / "test_results.csv", index=False)

    print("[5/6] 증강 채택 여부 판정 및 결과 저장")
    baseline = cv_summary[cv_summary["strategy"] == "scale_pos_weight"].iloc[0]
    synthetic_candidates = cv_summary[cv_summary["strategy"].str.startswith("DDPM")]
    best_synthetic = synthetic_candidates.iloc[0]
    baseline_fold = fold_results[
        (fold_results["model"] == baseline["model"])
        & (fold_results["strategy"] == baseline["strategy"])
    ].set_index("fold")
    synthetic_fold = fold_results[
        (fold_results["model"] == best_synthetic["model"])
        & (fold_results["strategy"] == best_synthetic["strategy"])
    ].set_index("fold")
    paired_wins = int(
        (synthetic_fold["pr_auc"] > baseline_fold["pr_auc"]).sum()
    )
    cv_gain = float(best_synthetic["pr_auc_mean"] - baseline["pr_auc_mean"])
    adopt = bool(cv_gain >= MIN_CV_GAIN and paired_wins >= 3)
    recommendation = {
        "selection_uses_test": False,
        "baseline_model": str(baseline["model"]),
        "baseline_strategy": str(baseline["strategy"]),
        "baseline_cv_pr_auc": float(baseline["pr_auc_mean"]),
        "best_ddpm_model": str(best_synthetic["model"]),
        "best_ddpm_strategy": str(best_synthetic["strategy"]),
        "best_ddpm_cv_pr_auc": float(best_synthetic["pr_auc_mean"]),
        "cv_pr_auc_gain": cv_gain,
        "paired_fold_wins": paired_wins,
        "minimum_gain": MIN_CV_GAIN,
        "adopt_for_primary_model": adopt,
        "decision": "adopt_ddpm" if adopt else "retain_scale_pos_weight",
    }
    (output_dir / "recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best_row = cv_summary.iloc[0]
    joblib.dump(
        bundles[(str(best_row["model"]), str(best_row["strategy"]))],
        output_dir / "best_cv_selected_model.joblib",
    )
    (output_dir / "selected_features.json").write_text(
        json.dumps(selected_features, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plots(output_dir, cv_summary, quality_results)

    top_test = test_results.iloc[0]
    quality_mean = quality_results[quality_results["context"].str.startswith("fold_")]
    report = f"""# SECOM 생성형 증강 강화 결과

- 실제 데이터: {len(y)}건 (불량 {int(y.sum())}건)
- 고정 test: {len(y_test)}건 (불량 {int(y_test.sum())}건), 합성 데이터 미사용
- fold 내부 처리: median → ANOVA 상위 {TOP_K_FEATURES}개 → 표준화 → DDPM 학습/품질 게이트
- 비교 모델: CatBoost, XGBoost
- DDPM 목표 불량/정상 비율: {', '.join(f'{ratio:.2f}' for ratio in SYNTH_RATIOS)}
- 채택 규칙: CV PR-AUC +{MIN_CV_GAIN:.3f} 이상이고 5개 fold 중 3개 이상 승리

## 결론

**{recommendation['decision']}**

기준: {baseline['model']} / {baseline['strategy']} (CV PR-AUC {baseline['pr_auc_mean']:.4f})

최상 DDPM: {best_synthetic['model']} / {best_synthetic['strategy']} (CV PR-AUC {best_synthetic['pr_auc_mean']:.4f}, 차이 {cv_gain:+.4f}, fold 승리 {paired_wins}/5)

## 합성 데이터 품질 (5-fold 평균)

- KS 유사도: {quality_mean['ks_similarity_mean'].mean():.4f}
- 상관 유사도: {quality_mean['correlation_similarity'].mean():.4f}
- real-vs-synthetic 판별 AUC: {quality_mean['discriminator_auc'].mean():.4f} (0.5에 가까울수록 구별 어려움)
- 실제 불량 커버리지: {quality_mean['real_coverage'].mean():.4f}
- 암기 위험률: {quality_mean['memorization_risk_rate'].mean():.4f}
- 종합 품질 점수: {quality_mean['quality_score'].mean():.4f}

## 실제 test 참고 결과

CV 1위 구성 {top_test['model']} / {top_test['strategy']}: PR-AUC {top_test['pr_auc']:.4f}, precision {top_test['precision']:.4f}, recall {top_test['recall']:.4f}, F1 {top_test['f1']:.4f}

모델/증강 선택은 test가 아니라 교차검증 결과로 결정했습니다. 합성 행 수는 실측 정보량과 같지 않으며, 외부 lot 검증 전에는 연구용 프로토타입으로 해석해야 합니다.
"""
    (output_dir / "summary.md").write_text(report, encoding="utf-8")

    metadata = {
        "random_state": RANDOM_STATE,
        "folds": N_SPLITS,
        "top_k_features": TOP_K_FEATURES,
        "diffusion_steps": DIFFUSION_STEPS,
        "diffusion_epochs": DIFFUSION_EPOCHS,
        "pool_multiplier": POOL_MULTIPLIER,
        "synth_ratios": SYNTH_RATIOS,
        "test_rows": len(y_test),
        "test_defects": int(y_test.sum()),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[6/6] 완료")
    print(cv_summary[["model", "strategy", "pr_auc_mean", "pr_auc_std", "recall_mean", "f1_mean"]].to_string(index=False))
    print(
        f"  decision={recommendation['decision']}, CV gain={cv_gain:+.4f}, "
        f"paired wins={paired_wins}/5"
    )
    print(f"  results={output_dir}")


if __name__ == "__main__":
    main()
