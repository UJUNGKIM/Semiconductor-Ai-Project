"""Select a WM-811K SSL none-class bias from train-lot OOF predictions.

The calibration value is selected without validation or test labels. Three
StratifiedGroupKFold models are trained on the frozen training split with a
fixed epoch count, initialized from the SSL encoder. The selected shared bias
is then evaluated once on the already-produced three-seed validation
predictions. The fixed test split is never loaded.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from calibrate_wm811k_robust_flat_bias import (
    CLASS_NAMES,
    NONE_LABEL,
    PROBABILITY_COLUMNS,
    WEAK_LABELS,
    metric_bundle,
    predict_with_none_bias,
)
from train_wm811k_cnn import (
    WaferCNN,
    WaferMapDataset,
    evaluate,
    seed_everything,
    sqrt_sampling_multipliers,
    train_epoch,
)


SEEDS = (17, 42, 2026)


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def _loader(
    maps: Path,
    frame: pd.DataFrame,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    train: bool,
    sampling_max_multiplier: float,
) -> DataLoader:
    labels = frame["label_id"].to_numpy(dtype=np.int64)
    dataset = WaferMapDataset(
        maps,
        frame["array_index"].to_numpy(dtype=np.int64),
        labels,
        augment=train,
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if train:
        _, multipliers = sqrt_sampling_multipliers(
            labels, len(CLASS_NAMES), sampling_max_multiplier
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(multipliers[labels], dtype=torch.double),
            num_samples=len(labels),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_seed,
        generator=generator,
    )


def build_train_oof_predictions(
    maps_path: Path,
    assignments: pd.DataFrame,
    encoder_state: dict[str, torch.Tensor],
    *,
    folds: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    weight_decay: float,
    sampling_max_multiplier: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Train fixed-epoch train-lot folds and return aligned OOF probabilities."""
    train_frame = assignments.loc[assignments["split"] == "train"].copy()
    train_frame = train_frame.sort_values("array_index").reset_index(drop=True)
    labels = train_frame["label_id"].to_numpy(dtype=np.int64)
    groups = train_frame["lot_name"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    probabilities = np.full(
        (len(train_frame), len(CLASS_NAMES)), np.nan, dtype=np.float32
    )
    fold_rows: list[dict[str, float | int]] = []
    placeholder = np.zeros(len(train_frame), dtype=np.uint8)
    for fold, (fit_positions, holdout_positions) in enumerate(
        splitter.split(placeholder, labels, groups)
    ):
        fold_seed = seed + fold
        seed_everything(fold_seed)
        fit_frame = train_frame.iloc[fit_positions]
        holdout_frame = train_frame.iloc[holdout_positions]
        fit_lots = set(fit_frame["lot_name"].astype(str))
        holdout_lots = set(holdout_frame["lot_name"].astype(str))
        if fit_lots & holdout_lots:
            raise AssertionError(f"fold {fold}: lot leakage")
        if set(fit_frame["label_id"]) != set(range(len(CLASS_NAMES))):
            raise ValueError(f"fold {fold}: fit split is missing a class")
        if set(holdout_frame["label_id"]) != set(range(len(CLASS_NAMES))):
            raise ValueError(f"fold {fold}: holdout split is missing a class")

        train_loader = _loader(
            maps_path,
            fit_frame,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            seed=fold_seed,
            train=True,
            sampling_max_multiplier=sampling_max_multiplier,
        )
        holdout_loader = _loader(
            maps_path,
            holdout_frame,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            seed=fold_seed,
            train=False,
            sampling_max_multiplier=sampling_max_multiplier,
        )
        model = WaferCNN(num_classes=len(CLASS_NAMES)).to(device)
        model.features.load_state_dict(copy.deepcopy(encoder_state))
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        final_train_loss = 0.0
        for epoch in range(1, epochs + 1):
            final_train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                loss_function,
                device,
                scaler,
            )
            print(
                json.dumps(
                    {
                        "phase": "train_oof",
                        "fold": fold,
                        "epoch": epoch,
                        "train_loss": final_train_loss,
                    },
                    ensure_ascii=False,
                )
            )
        metrics, true, _, fold_probabilities = evaluate(
            model, holdout_loader, loss_function, device
        )
        if not np.array_equal(true, labels[holdout_positions]):
            raise AssertionError(f"fold {fold}: holdout order mismatch")
        probabilities[holdout_positions] = fold_probabilities.astype(np.float32)
        fold_rows.append(
            {
                "fold": fold,
                "fit_rows": len(fit_positions),
                "holdout_rows": len(holdout_positions),
                "fit_lots": len(fit_lots),
                "holdout_lots": len(holdout_lots),
                "fixed_epochs": epochs,
                "final_train_loss": final_train_loss,
                "holdout_accuracy": metrics["accuracy"],
                "holdout_balanced_accuracy": metrics["balanced_accuracy"],
                "holdout_macro_f1": metrics["macro_f1"],
            }
        )
        del model, optimizer, train_loader, holdout_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.isfinite(probabilities).all():
        raise AssertionError("OOF matrix contains unfilled rows")
    return labels, probabilities, pd.DataFrame(fold_rows)


def select_train_oof_bias(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray,
    *,
    target_none_recall: float,
    weak_recall_tolerance: float,
) -> tuple[float | None, pd.DataFrame]:
    """Select the smallest train-OOF bias satisfying fixed safety targets."""
    baseline = metric_bundle(true_labels, probabilities.argmax(axis=1))
    rows: list[dict[str, float | bool]] = []
    for raw_bias in grid:
        bias = float(np.round(raw_bias, 6))
        metrics = metric_bundle(
            true_labels, predict_with_none_bias(probabilities, bias)
        )
        eligible = (
            metrics["none_recall"] >= target_none_recall
            and metrics["weak_recall"]
            >= baseline["weak_recall"] - weak_recall_tolerance
        )
        rows.append(
            {
                "none_logit_bias": bias,
                **metrics,
                "delta_macro_f1": metrics["macro_f1"] - baseline["macro_f1"],
                "delta_balanced_accuracy": (
                    metrics["balanced_accuracy"] - baseline["balanced_accuracy"]
                ),
                "delta_weak_recall": (
                    metrics["weak_recall"] - baseline["weak_recall"]
                ),
                "delta_none_recall": (
                    metrics["none_recall"] - baseline["none_recall"]
                ),
                "eligible": eligible,
            }
        )
    sweep = pd.DataFrame(rows)
    eligible = sweep.loc[sweep["eligible"]].sort_values(
        ["none_logit_bias", "macro_f1"],
        ascending=[True, False],
    )
    selected = None if eligible.empty else float(eligible.iloc[0]["none_logit_bias"])
    return selected, sweep


def load_validation_predictions(
    root: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    loaded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    reference_indices: np.ndarray | None = None
    reference_labels: np.ndarray | None = None
    for seed in SEEDS:
        path = root / "seeds" / f"seed_{seed}" / "validation_predictions.csv"
        frame = pd.read_csv(path)
        required = {"array_index", "label_id", *PROBABILITY_COLUMNS}
        missing = required.difference(frame.columns)
        if missing or frame["array_index"].duplicated().any():
            raise ValueError(f"seed {seed}: invalid validation predictions")
        indices = frame["array_index"].to_numpy(dtype=np.int64)
        true = frame["label_id"].to_numpy(dtype=np.int64)
        probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError(f"seed {seed}: invalid probability simplex")
        if reference_indices is None:
            reference_indices = indices
            reference_labels = true
        elif not np.array_equal(reference_indices, indices):
            raise ValueError("Validation rows differ across seeds")
        elif not np.array_equal(reference_labels, true):
            raise ValueError("Validation labels differ across seeds")
        loaded[seed] = true, probabilities
    return loaded


def validation_gate(
    loaded: dict[int, tuple[np.ndarray, np.ndarray]],
    bias: float,
    baseline_summary: dict,
    baseline_recalls: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, float | int]] = []
    class_recalls: dict[str, list[float]] = {name: [] for name in CLASS_NAMES}
    for seed in SEEDS:
        true, probabilities = loaded[seed]
        predicted = predict_with_none_bias(probabilities, bias)
        metrics = metric_bundle(true, predicted)
        recalls = []
        for label in range(len(CLASS_NAMES)):
            mask = true == label
            recalls.append(float(np.mean(predicted[mask] == label)))
        rows.append({"seed": seed, **metrics})
        for name, value in zip(CLASS_NAMES, recalls):
            class_recalls[name].append(value)
    frame = pd.DataFrame(rows)
    candidate = {
        metric: {
            "mean": float(frame[metric].mean()),
            "std": float(frame[metric].std(ddof=1)),
            "min": float(frame[metric].min()),
            "max": float(frame[metric].max()),
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    }
    recall_summary = {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
        for name, values in class_recalls.items()
    }
    baseline_macro = float(
        baseline_summary["metrics"]["validation_macro_f1"]["mean"]
    )
    baseline_balanced = float(
        baseline_summary["metrics"]["validation_balanced_accuracy"]["mean"]
    )
    checks = {
        "macro_f1_mean_improves_by_0_002": (
            candidate["macro_f1"]["mean"] >= baseline_macro + 0.002
        ),
        "balanced_accuracy_drop_within_0_002": (
            candidate["balanced_accuracy"]["mean"] >= baseline_balanced - 0.002
        ),
        "normal_recall_drop_within_0_002": (
            recall_summary["none"]["mean"] >= baseline_recalls["none"] - 0.002
        ),
    }
    for name in ("Edge-Loc", "Loc", "Scratch"):
        checks[f"{name}_recall_drop_within_0_01"] = (
            recall_summary[name]["mean"] >= baseline_recalls[name] - 0.01
        )
    return frame, {
        "candidate_validation_metrics": candidate,
        "candidate_validation_class_recall": recall_summary,
        "guardrail_checks": checks,
        "all_guardrails_passed": all(checks.values()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.oof_folds < 2 or args.epochs <= 0:
        raise ValueError("oof-folds >= 2 and epochs > 0 are required")
    assignments = pd.read_csv(args.assignments).sort_values("array_index")
    if set(assignments["split"]) != {"train", "validation", "test"}:
        raise ValueError("Frozen assignments must contain train/validation/test")
    train_lots = set(
        assignments.loc[assignments["split"] == "train", "lot_name"].astype(str)
    )
    held_out_lots = set(
        assignments.loc[assignments["split"].isin(["validation", "test"]), "lot_name"]
        .astype(str)
    )
    if train_lots & held_out_lots:
        raise ValueError("Frozen assignments contain lot leakage")
    maps = np.load(args.maps, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    if len(maps) != len(assignments) or len(labels) != len(assignments):
        raise ValueError("Maps, labels, and assignments have different lengths")
    train_frame = assignments.loc[assignments["split"] == "train"]
    train_indices = train_frame["array_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        labels[train_indices], train_frame["label_id"].to_numpy(dtype=np.int64)
    ):
        raise ValueError("Train maps, labels, and assignments are not aligned")
    del maps, labels
    encoder = torch.load(args.ssl_encoder, map_location="cpu", weights_only=True)
    if (
        encoder.get("held_out_lots_used") is not False
        or encoder.get("labels_used_during_pretraining") is not False
    ):
        raise ValueError("SSL encoder provenance is unsafe")
    encoder_state = encoder["features_state_dict"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"device={device}")

    true, oof_probabilities, folds = build_train_oof_predictions(
        args.maps,
        assignments,
        encoder_state,
        folds=args.oof_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        sampling_max_multiplier=args.sampling_max_multiplier,
        seed=args.seed,
        device=device,
    )
    grid = np.arange(
        args.bias_min, args.bias_max + args.bias_step / 2, args.bias_step
    )
    selected_bias, sweep = select_train_oof_bias(
        true,
        oof_probabilities,
        grid,
        target_none_recall=args.target_none_recall,
        weak_recall_tolerance=args.weak_recall_tolerance,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_dir / "oof_fold_metrics.csv", index=False)
    sweep.to_csv(args.output_dir / "oof_bias_sweep.csv", index=False)

    if selected_bias is None:
        summary = {
            "schema_version": 1,
            "experiment_type": "wm811k_ssl_train_oof_none_bias",
            "selection_source": "train_lot_oof_only",
            "selected_none_logit_bias": None,
            "oof_target_met": False,
            "test_evaluated": False,
            "test_used_for_selection": False,
            "validation_evaluated": False,
            "validation_used_for_bias_selection": False,
            "deployment_model_changed": False,
            "decision": "retain_existing_model",
        }
    else:
        baseline_summary = json.loads(
            args.baseline_summary.read_text(encoding="utf-8")
        )
        baseline_recall_rows = json.loads(
            args.baseline_class_recall.read_text(encoding="utf-8")
        )
        baseline_recalls = {
            row["class_name"]: float(row["recall_mean"])
            for row in baseline_recall_rows
            if row["split"] == "validation"
        }
        validation_predictions = load_validation_predictions(
            args.validation_run_root
        )
        seed_metrics, validation = validation_gate(
            validation_predictions,
            selected_bias,
            baseline_summary,
            baseline_recalls,
        )
        seed_metrics.to_csv(
            args.output_dir / "calibrated_validation_seed_metrics.csv",
            index=False,
        )
        passed = bool(validation["all_guardrails_passed"])
        summary = {
            "schema_version": 1,
            "experiment_type": "wm811k_ssl_train_oof_none_bias",
            "selection_source": "train_lot_oof_only",
            "oof_folds": args.oof_folds,
            "oof_seed": args.seed,
            "fixed_epochs": args.epochs,
            "target_none_recall": args.target_none_recall,
            "weak_recall_tolerance": args.weak_recall_tolerance,
            "selected_none_logit_bias": selected_bias,
            "oof_target_met": True,
            **validation,
            "test_evaluated": False,
            "test_used_for_selection": False,
            "validation_used_for_bias_selection": False,
            "deployment_model_changed": False,
            "decision": (
                "validation_supported_external_holdout_required"
                if passed
                else "retain_existing_model"
            ),
        }
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--ssl-encoder", type=Path, required=True)
    parser.add_argument("--validation-run-root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-class-recall", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sampling-max-multiplier", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bias-min", type=float, default=0.0)
    parser.add_argument("--bias-max", type=float, default=0.5)
    parser.add_argument("--bias-step", type=float, default=0.025)
    parser.add_argument("--target-none-recall", type=float, default=0.995)
    parser.add_argument("--weak-recall-tolerance", type=float, default=0.01)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
