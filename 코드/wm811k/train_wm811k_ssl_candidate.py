"""Validation-only WM-811K self-supervised pretraining experiment.

The encoder is contrastively pretrained with (1) unlabeled wafer maps from
frozen training lots and (2) labeled training maps. Fine-tuning uses labeled
training rows and model selection uses validation rows only. The fixed test
split is deliberately never loaded or evaluated by this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_wm811k_cnn import (
    WaferCNN,
    WaferMapDataset,
    evaluate,
    plot_history,
    save_evaluation_artifacts,
    seed_everything,
    sqrt_sampling_multipliers,
    train_epoch,
)


WEAK_CLASSES = ("Edge-Loc", "Loc", "Scratch")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def wafer_channels(wafer: np.ndarray) -> np.ndarray:
    array = np.asarray(wafer, dtype=np.uint8)
    return np.stack((array > 0, array == 2), axis=0).astype(np.float32)


def contrastive_view(channels: np.ndarray, dropout_probability: float) -> torch.Tensor:
    view = np.asarray(channels)
    view = np.rot90(view, k=np.random.randint(0, 4), axes=(1, 2))
    if np.random.random() < 0.5:
        view = np.flip(view, axis=1)
    if np.random.random() < 0.5:
        view = np.flip(view, axis=2)
    view = np.ascontiguousarray(view.copy())
    if dropout_probability > 0:
        active = view[0] > 0
        dropped = active & (np.random.random(active.shape) < dropout_probability)
        view[:, dropped] = 0
    return torch.from_numpy(view)


class ContrastiveWaferDataset(Dataset):
    """Lazy union of train-lot unlabeled maps and labeled training maps."""

    def __init__(
        self,
        unlabeled_maps: Path,
        labeled_maps: Path,
        labeled_train_indices: np.ndarray,
        *,
        dropout_probability: float,
    ) -> None:
        self.unlabeled_path = str(unlabeled_maps)
        self.labeled_path = str(labeled_maps)
        self.labeled_train_indices = np.asarray(labeled_train_indices, dtype=np.int64)
        self.dropout_probability = float(dropout_probability)
        self._unlabeled: np.ndarray | None = None
        self._labeled: np.ndarray | None = None
        probe = np.load(unlabeled_maps, mmap_mode="r")
        self.unlabeled_count = len(probe)
        del probe

    def __len__(self) -> int:
        return self.unlabeled_count + len(self.labeled_train_indices)

    def _maps(self, unlabeled: bool) -> np.ndarray:
        if unlabeled:
            if self._unlabeled is None:
                self._unlabeled = np.load(self.unlabeled_path, mmap_mode="r")
            return self._unlabeled
        if self._labeled is None:
            self._labeled = np.load(self.labeled_path, mmap_mode="r")
        return self._labeled

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        if position < self.unlabeled_count:
            wafer = self._maps(True)[position]
        else:
            labeled_position = position - self.unlabeled_count
            source_index = self.labeled_train_indices[labeled_position]
            wafer = self._maps(False)[source_index]
        channels = wafer_channels(wafer)
        return (
            contrastive_view(channels, self.dropout_probability),
            contrastive_view(channels, self.dropout_probability),
        )


class ContrastiveEncoder(nn.Module):
    def __init__(self, projection_dimension: int = 128) -> None:
        super().__init__()
        base = WaferCNN()
        self.features = base.features
        self.pool = base.pool
        self.projector = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, projection_dimension),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features).flatten(1)
        return self.projector(pooled)


def nt_xent_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Contrastive embeddings must be equally shaped 2-D tensors")
    if len(first) < 2:
        raise ValueError("At least two samples are required for contrastive loss")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    embeddings = F.normalize(torch.cat((first, second), dim=0), dim=1)
    logits = embeddings @ embeddings.T / temperature
    diagonal = torch.eye(len(logits), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
    batch_size = len(first)
    targets = torch.arange(2 * batch_size, device=logits.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, targets)


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def validate_inputs(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame]:
    labeled = np.load(args.maps, mmap_mode="r")
    labels = np.load(args.labels, mmap_mode="r")
    unlabeled = np.load(args.unlabeled_maps, mmap_mode="r")
    assignments = pd.read_csv(args.assignments).sort_values("array_index")
    required = {"array_index", "label_id", "failure_type", "lot_name", "split"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Assignments are missing columns: {sorted(missing)}")
    if labeled.shape[1:] != (64, 64) or labeled.dtype != np.uint8:
        raise ValueError(f"Invalid labeled maps: shape={labeled.shape}, dtype={labeled.dtype}")
    if unlabeled.shape[1:] != (64, 64) or unlabeled.dtype != np.uint8:
        raise ValueError(
            f"Invalid unlabeled maps: shape={unlabeled.shape}, dtype={unlabeled.dtype}"
        )
    if len(labeled) != len(labels) or len(labeled) != len(assignments):
        raise ValueError("Labeled maps, labels, and assignments are not aligned")
    if not np.array_equal(labels, assignments["label_id"].to_numpy()):
        raise ValueError("labels.npy and assignments are not aligned")
    if set(assignments["split"]) != {"train", "validation", "test"}:
        raise ValueError("Frozen assignments must contain all three split names")
    train_lots = set(assignments.loc[assignments["split"] == "train", "lot_name"].astype(str))
    held_out_lots = set(
        assignments.loc[assignments["split"].isin(["validation", "test"]), "lot_name"]
        .astype(str)
    )
    if train_lots & held_out_lots:
        raise ValueError("Lot leakage exists in frozen assignments")

    receipt = json.loads(args.unlabeled_summary.read_text(encoding="utf-8"))
    if receipt.get("validation_or_test_lots_used") is not False:
        raise ValueError("Unlabeled receipt does not prove held-out lot exclusion")
    if receipt.get("labels_used") is not False:
        raise ValueError("Unlabeled receipt does not prove label-free extraction")
    if receipt.get("valid_selected_rows") != len(unlabeled):
        raise ValueError("Unlabeled receipt row count does not match maps")
    if receipt.get("split_assignments_sha256") != sha256_file(args.assignments):
        raise ValueError("Unlabeled receipt was created from different split assignments")
    del labeled, unlabeled
    return np.asarray(labels), assignments


def pretrain_encoder(
    args: argparse.Namespace,
    assignments: pd.DataFrame,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int]], float]:
    seed_everything(args.pretrain_seed)
    train_indices = assignments.loc[
        assignments["split"] == "train", "array_index"
    ].to_numpy(dtype=np.int64)
    dataset = ContrastiveWaferDataset(
        args.unlabeled_maps,
        args.maps,
        train_indices,
        dropout_probability=args.die_dropout,
    )
    generator = torch.Generator().manual_seed(args.pretrain_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.pretrain_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_seed,
        generator=generator,
    )
    model = ContrastiveEncoder(args.projection_dimension).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.pretrain_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.pretrain_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for first, second in loader:
            first = first.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                first_embedding = model(first)
                second_embedding = model(second)
                loss = nt_xent_loss(
                    first_embedding, second_embedding, args.temperature
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(first)
            total_rows += len(first)
        scheduler.step()
        row = {
            "epoch": epoch,
            "contrastive_loss": total_loss / total_rows,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps({"phase": "pretrain", **row}, ensure_ascii=False))
    elapsed = time.perf_counter() - started
    return copy.deepcopy(model.features.state_dict()), history, elapsed


def make_validation_only_loaders(
    maps_path: Path,
    assignments: pd.DataFrame,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    sampling_max_multiplier: float,
) -> tuple[dict[str, DataLoader], np.ndarray]:
    generator = torch.Generator().manual_seed(seed)
    num_classes = int(assignments["label_id"].max()) + 1
    loaders: dict[str, DataLoader] = {}
    multipliers = np.ones(num_classes, dtype=np.float32)
    for split_name in ("train", "validation"):
        subset = assignments.loc[assignments["split"] == split_name]
        subset_labels = subset["label_id"].to_numpy(dtype=np.int64)
        dataset = WaferMapDataset(
            maps_path,
            subset["array_index"].to_numpy(dtype=np.int64),
            subset_labels,
            augment=split_name == "train",
        )
        sampler = None
        if split_name == "train":
            _, multipliers = sqrt_sampling_multipliers(
                subset_labels, num_classes, sampling_max_multiplier
            )
            weights = torch.as_tensor(
                multipliers[subset_labels], dtype=torch.double
            )
            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(weights),
                replacement=True,
                generator=generator,
            )
        loaders[split_name] = DataLoader(
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
    return loaders, multipliers


def finetune_one_seed(
    args: argparse.Namespace,
    assignments: pd.DataFrame,
    encoder_state: dict[str, torch.Tensor],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(seed)
    seed_dir = args.output_dir / "seeds" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    class_table = (
        assignments[["label_id", "failure_type"]]
        .drop_duplicates()
        .sort_values("label_id")
    )
    class_names = class_table["failure_type"].tolist()
    loaders, multipliers = make_validation_only_loaders(
        args.maps,
        assignments,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
        device=device,
        sampling_max_multiplier=args.sampling_max_multiplier,
    )
    model = WaferCNN(num_classes=len(class_names)).to(device)
    model.features.load_state_dict(encoder_state)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = seed_dir / "best_validation_model.pt"
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, loaders["train"], optimizer, loss_function, device, scaler
        )
        metrics, _, _, _ = evaluate(
            model, loaders["validation"], loss_function, device
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_loss": metrics["loss"],
            "validation_accuracy": metrics["accuracy"],
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_weighted_f1": metrics["weighted_f1"],
        }
        history.append(row)
        print(json.dumps({"phase": "finetune", "seed": seed, **row}, ensure_ascii=False))
        if metrics["macro_f1"] > best_f1 + args.minimum_delta:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "num_classes": len(class_names),
                    "input_channels": 2,
                    "image_size": 64,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_f1,
                    "sampling": "sqrt_balanced",
                    "sampling_multipliers": multipliers.tolist(),
                    "seed": seed,
                    "initialization": "train_lot_unlabeled_contrastive",
                    "test_evaluated": False,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, true, predicted, probabilities = evaluate(
        model, loaders["validation"], loss_function, device
    )
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(seed_dir / "training_history.csv", index=False)
    plot_history(history_frame, seed_dir / "training_curves.png")
    save_evaluation_artifacts(
        assignments,
        "validation",
        true,
        predicted,
        probabilities,
        class_names,
        seed_dir,
    )
    per_class_recall = recall_score(
        true,
        predicted,
        labels=np.arange(len(class_names)),
        average=None,
        zero_division=0,
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "experiment_type": "wm811k_ssl_validation_only",
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "validation_metrics": metrics,
        "validation_class_recall": {
            name: float(per_class_recall[index])
            for index, name in enumerate(class_names)
        },
        "test_evaluated": False,
        "selection_split": "validation",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (seed_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def compare_with_baseline(
    runs: list[dict[str, object]],
    baseline_summary_path: Path,
    baseline_recall_path: Path,
) -> dict[str, object]:
    baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    baseline_recalls_raw = json.loads(baseline_recall_path.read_text(encoding="utf-8"))
    baseline_recalls = {
        row["class_name"]: float(row["recall_mean"])
        for row in baseline_recalls_raw
        if row["split"] == "validation"
    }
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    candidate_metrics: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray(
            [float(run["validation_metrics"][name]) for run in runs],
            dtype=np.float64,
        )
        candidate_metrics[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    candidate_recalls: dict[str, dict[str, float]] = {}
    class_names = list(runs[0]["validation_class_recall"])
    for name in class_names:
        values = np.asarray(
            [float(run["validation_class_recall"][name]) for run in runs]
        )
        candidate_recalls[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    baseline_macro = float(baseline["metrics"]["validation_macro_f1"]["mean"])
    baseline_balanced = float(
        baseline["metrics"]["validation_balanced_accuracy"]["mean"]
    )
    checks = {
        "three_or_more_seeds": len(runs) >= 3,
        "macro_f1_mean_improves_by_0_002": (
            candidate_metrics["macro_f1"]["mean"] >= baseline_macro + 0.002
        ),
        "balanced_accuracy_drop_within_0_002": (
            candidate_metrics["balanced_accuracy"]["mean"] >= baseline_balanced - 0.002
        ),
        "normal_recall_drop_within_0_002": (
            candidate_recalls["none"]["mean"] >= baseline_recalls["none"] - 0.002
        ),
    }
    for class_name in WEAK_CLASSES:
        checks[f"{class_name}_recall_drop_within_0_01"] = (
            candidate_recalls[class_name]["mean"]
            >= baseline_recalls[class_name] - 0.01
        )
    eligible = all(checks.values())
    return {
        "schema_version": 1,
        "purpose": "validation_only_candidate_screening",
        "candidate": "train_lot_unlabeled_contrastive_pretraining",
        "baseline": "ce_sqrt_balanced_random_initialization",
        "seed_count": len(runs),
        "candidate_validation_metrics": candidate_metrics,
        "baseline_validation_metrics": {
            "macro_f1": baseline["metrics"]["validation_macro_f1"],
            "balanced_accuracy": baseline["metrics"][
                "validation_balanced_accuracy"
            ],
        },
        "candidate_validation_class_recall": candidate_recalls,
        "baseline_validation_class_recall": baseline_recalls,
        "guardrail_checks": checks,
        "all_guardrails_passed": eligible,
        "decision": (
            "validation_supported_external_holdout_required"
            if eligible
            else "retain_existing_model"
        ),
        "fixed_test_reopened": False,
        "deployment_model_changed": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if len(set(args.finetune_seeds)) != len(args.finetune_seeds):
        raise ValueError("finetune seeds must be unique")
    if args.pretrain_epochs <= 0 or args.epochs <= 0:
        raise ValueError("epoch counts must be positive")
    labels, assignments = validate_inputs(args)
    del labels
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    encoder_state, pretrain_history, pretrain_seconds = pretrain_encoder(
        args, assignments, device
    )
    torch.save(
        {
            "features_state_dict": encoder_state,
            "pretrain_seed": args.pretrain_seed,
            "unlabeled_maps_sha256": sha256_file(args.unlabeled_maps),
            "split_assignments_sha256": sha256_file(args.assignments),
            "held_out_lots_used": False,
            "labels_used_during_pretraining": False,
        },
        args.output_dir / "ssl_encoder.pt",
    )
    pd.DataFrame(pretrain_history).to_csv(
        args.output_dir / "pretraining_history.csv", index=False
    )

    runs = [
        finetune_one_seed(args, assignments, encoder_state, seed, device)
        for seed in args.finetune_seeds
    ]
    comparison = compare_with_baseline(
        runs, args.baseline_summary, args.baseline_class_recall
    )
    comparison.update(
        {
            "pretrain_seconds": pretrain_seconds,
            "pretrain_epochs": args.pretrain_epochs,
            "pretrain_rows": int(
                json.loads(args.unlabeled_summary.read_text(encoding="utf-8"))[
                    "valid_selected_rows"
                ]
                + int((assignments["split"] == "train").sum())
            ),
            "finetune_seeds": args.finetune_seeds,
            "split_assignments_sha256": sha256_file(args.assignments),
            "unlabeled_maps_sha256": sha256_file(args.unlabeled_maps),
            "test_used_for_selection": False,
        }
    )
    (args.output_dir / "ssl_candidate_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--unlabeled-maps", type=Path, required=True)
    parser.add_argument("--unlabeled-summary", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-class-recall", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--pretrain-batch-size", type=int, default=512)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--pretrain-seed", type=int, default=42)
    parser.add_argument("--projection-dimension", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--die-dropout", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sampling-max-multiplier", type=float, default=8.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--finetune-seeds", type=int, nargs="+", default=[17, 42, 2026]
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
