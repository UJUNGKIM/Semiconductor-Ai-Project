"""Train a compact explainable CNN on leakage-free WM-811K splits."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SPLIT_NAMES = ("train", "validation", "test")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WaferMapDataset(Dataset):
    """Lazy memory-mapped two-channel wafer-map dataset."""

    def __init__(
        self,
        maps_path: Path,
        indices: np.ndarray,
        labels: np.ndarray,
        *,
        augment: bool,
    ) -> None:
        self.maps_path = str(maps_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.augment = augment
        self._maps: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _open_maps(self) -> np.ndarray:
        if self._maps is None:
            self._maps = np.load(self.maps_path, mmap_mode="r")
        return self._maps

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        wafer = np.asarray(self._open_maps()[self.indices[position]], dtype=np.uint8)
        channels = np.stack((wafer > 0, wafer == 2), axis=0).astype(np.float32)
        if self.augment:
            channels = np.rot90(channels, k=np.random.randint(0, 4), axes=(1, 2))
            if np.random.random() < 0.5:
                channels = np.flip(channels, axis=1)
            if np.random.random() < 0.5:
                channels = np.flip(channels, axis=2)
        image = torch.from_numpy(np.ascontiguousarray(channels))
        label = torch.tensor(self.labels[position], dtype=torch.long)
        return image, label


class WaferCNN(nn.Module):
    """Compact CNN whose final feature map can later be used for Grad-CAM."""

    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.30), nn.Linear(256, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


class FocalLoss(nn.Module):
    def __init__(self, weights: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        target_probabilities = probabilities.gather(1, targets[:, None]).squeeze(1)
        cross_entropy = F.nll_loss(
            log_probabilities, targets, weight=self.weights, reduction="none"
        )
        return (((1.0 - target_probabilities) ** self.gamma) * cross_entropy).mean()


def effective_class_weights(
    labels: np.ndarray, num_classes: int, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training split is missing classes: {np.where(counts == 0)[0]}")
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights /= weights.mean()
    return counts.astype(np.int64), weights.astype(np.float32)


def sqrt_sampling_multipliers(
    labels: np.ndarray, num_classes: int, maximum: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return clipped square-root inverse-frequency sampling multipliers."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training split is missing classes: {np.where(counts == 0)[0]}")
    if maximum < 1.0:
        raise ValueError("Sampling maximum multiplier must be at least 1")
    multipliers = np.sqrt(counts.max() / counts)
    multipliers = np.minimum(multipliers, maximum)
    return counts.astype(np.int64), multipliers.astype(np.float32)


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def make_loaders(
    maps_path: Path,
    assignments: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
    sampling: str = "shuffle",
    sampling_max_multiplier: float = 8.0,
) -> tuple[dict[str, DataLoader], np.ndarray]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders: dict[str, DataLoader] = {}
    num_classes = int(assignments["label_id"].max()) + 1
    sampling_multipliers = np.ones(num_classes, dtype=np.float32)
    for split_name in SPLIT_NAMES:
        subset = assignments.loc[assignments["split"] == split_name]
        subset_labels = subset["label_id"].to_numpy()
        dataset = WaferMapDataset(
            maps_path,
            subset["array_index"].to_numpy(),
            subset_labels,
            augment=split_name == "train",
        )
        sampler = None
        if split_name == "train" and sampling == "sqrt_balanced":
            _, sampling_multipliers = sqrt_sampling_multipliers(
                subset_labels, num_classes, sampling_max_multiplier
            )
            sample_weights = torch.as_tensor(
                sampling_multipliers[subset_labels], dtype=torch.double
            )
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
                generator=generator,
            )
        elif split_name == "train" and sampling != "shuffle":
            raise ValueError(f"Unknown sampling strategy: {sampling}")
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train" and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            worker_init_fn=_worker_seed,
            generator=generator,
        )
    return loaders, sampling_multipliers


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(inputs)
            loss = loss_function(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * len(targets)
        total_rows += len(targets)
    return total_loss / total_rows


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = loss_function(logits, targets)
        batch_probabilities = torch.softmax(logits, dim=1)
        total_loss += float(loss) * len(targets)
        total_rows += len(targets)
        labels.append(targets.cpu().numpy())
        predictions.append(batch_probabilities.argmax(dim=1).cpu().numpy())
        probabilities.append(batch_probabilities.cpu().numpy())

    true_labels = np.concatenate(labels)
    predicted_labels = np.concatenate(predictions)
    predicted_probabilities = np.concatenate(probabilities)
    metrics = {
        "loss": total_loss / total_rows,
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels),
        "macro_f1": f1_score(
            true_labels, predicted_labels, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            true_labels, predicted_labels, average="weighted", zero_division=0
        ),
    }
    return metrics, true_labels, predicted_labels, predicted_probabilities


def plot_history(history: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["validation_macro_f1"], label="Macro-F1")
    axes[1].plot(
        history["epoch"],
        history["validation_balanced_accuracy"],
        label="Balanced accuracy",
    )
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    split_name: str = "test",
) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"Normalized {split_name} confusion matrix")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = normalized[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_evaluation_artifacts(
    assignments: pd.DataFrame,
    split_name: str,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    predicted_probabilities: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Save one split's class report, confusion matrix, and row predictions."""
    num_classes = len(class_names)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=np.arange(num_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / f"{split_name}_classification_report.csv"
    )
    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=np.arange(num_classes)
    )
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output_dir / f"{split_name}_confusion_matrix.csv"
    )
    plot_confusion_matrix(
        matrix,
        class_names,
        output_dir / f"{split_name}_confusion_matrix.png",
        split_name=split_name,
    )

    rows = assignments.loc[assignments["split"] == split_name].copy()
    rows["predicted_label_id"] = predicted_labels
    rows["predicted_failure_type"] = [
        class_names[label] for label in predicted_labels
    ]
    rows["correct"] = true_labels == predicted_labels
    for label_id, class_name in enumerate(class_names):
        safe_name = class_name.lower().replace("-", "_").replace(" ", "_")
        rows[f"probability_{safe_name}"] = predicted_probabilities[:, label_id]
    rows.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)


def validate_inputs(
    maps_path: Path, labels_path: Path, assignments_path: Path
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    maps = np.load(maps_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    assignments = pd.read_csv(assignments_path).sort_values("array_index")
    required = {"array_index", "label_id", "failure_type", "lot_name", "split"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Split assignments are missing columns: {sorted(missing)}")
    if maps.ndim != 3 or maps.shape[1:] != (64, 64):
        raise ValueError(f"Expected maps shaped (N, 64, 64), got {maps.shape}")
    if maps.dtype != np.uint8:
        raise ValueError(f"Expected uint8 maps, got {maps.dtype}")
    if len(maps) != len(labels) or len(maps) != len(assignments):
        raise ValueError(
            f"Row mismatch: maps={len(maps)}, labels={len(labels)}, "
            f"assignments={len(assignments)}"
        )
    expected_indices = np.arange(len(assignments))
    if not np.array_equal(assignments["array_index"].to_numpy(), expected_indices):
        raise ValueError("array_index is not contiguous")
    if not np.array_equal(labels, assignments["label_id"].to_numpy()):
        raise ValueError("labels.npy and split assignments are not aligned")
    if set(assignments["split"]) != set(SPLIT_NAMES):
        raise ValueError(f"Unexpected split names: {sorted(set(assignments['split']))}")
    lot_sets = {
        name: set(assignments.loc[assignments["split"] == name, "lot_name"])
        for name in SPLIT_NAMES
    }
    if (
        lot_sets["train"] & lot_sets["validation"]
        or lot_sets["train"] & lot_sets["test"]
        or lot_sets["validation"] & lot_sets["test"]
    ):
        raise ValueError("Lot leakage detected")
    return maps, labels, assignments


def train(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps, labels, assignments = validate_inputs(
        args.maps, args.labels, args.assignments
    )
    del maps

    class_table = (
        assignments[["label_id", "failure_type"]]
        .drop_duplicates()
        .sort_values("label_id")
    )
    class_names = class_table["failure_type"].tolist()
    num_classes = len(class_names)
    train_labels = assignments.loc[
        assignments["split"] == "train", "label_id"
    ].to_numpy()
    class_counts, class_weights = effective_class_weights(
        train_labels, num_classes, args.class_weight_beta
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"class_counts={class_counts.tolist()}")
    print(f"class_weights={class_weights.tolist()}")

    sampling = getattr(args, "sampling", "shuffle")
    sampling_max_multiplier = float(
        getattr(args, "sampling_max_multiplier", 8.0)
    )
    loaders, sampling_multipliers = make_loaders(
        args.maps,
        assignments,
        args.batch_size,
        args.num_workers,
        args.seed,
        device,
        sampling=sampling,
        sampling_max_multiplier=sampling_max_multiplier,
    )
    print(f"sampling={sampling}")
    print(f"sampling_multipliers={sampling_multipliers.tolist()}")
    model = WaferCNN(num_classes=num_classes).to(device)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    if args.loss == "focal":
        loss_function: nn.Module = FocalLoss(weights_tensor, gamma=args.focal_gamma)
    elif args.loss == "weighted_ce":
        loss_function = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history_rows: list[dict[str, float | int]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path = args.output_dir / "best_model.pt"
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, loaders["train"], optimizer, loss_function, device, scaler
        )
        validation_metrics, _, _, _ = evaluate(
            model, loaders["validation"], loss_function, device
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_weighted_f1": validation_metrics["weighted_f1"],
        }
        history_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

        current_macro_f1 = validation_metrics["macro_f1"]
        if current_macro_f1 > best_macro_f1 + args.minimum_delta:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "input_channels": 2,
                    "image_size": 64,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_macro_f1,
                    "loss": args.loss,
                    "sampling": sampling,
                    "sampling_max_multiplier": sampling_max_multiplier,
                    "sampling_multipliers": sampling_multipliers.tolist(),
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= args.patience:
            print(f"early_stopping epoch={epoch}")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    (
        validation_metrics,
        validation_true,
        validation_predicted,
        validation_probabilities,
    ) = evaluate(
        model, loaders["validation"], loss_function, device
    )
    test_metrics, test_true, test_predicted, test_probabilities = evaluate(
        model, loaders["test"], loss_function, device
    )

    history = pd.DataFrame(history_rows)
    history.to_csv(args.output_dir / "training_history.csv", index=False)
    plot_history(history, args.output_dir / "training_curves.png")

    save_evaluation_artifacts(
        assignments,
        "validation",
        validation_true,
        validation_predicted,
        validation_probabilities,
        class_names,
        args.output_dir,
    )
    save_evaluation_artifacts(
        assignments,
        "test",
        test_true,
        test_predicted,
        test_probabilities,
        class_names,
        args.output_dir,
    )

    elapsed_seconds = time.perf_counter() - started
    run_summary: dict[str, object] = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seed": args.seed,
        "loss": args.loss,
        "sampling": sampling,
        "sampling_max_multiplier": sampling_max_multiplier,
        "sampling_multipliers": sampling_multipliers.tolist(),
        "focal_gamma": args.focal_gamma,
        "class_weight_beta": args.class_weight_beta,
        "class_names": class_names,
        "class_counts": class_counts.tolist(),
        "class_weights": class_weights.tolist(),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "elapsed_seconds": elapsed_seconds,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    return run_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss",
        choices=("focal", "weighted_ce", "cross_entropy"),
        default="focal",
    )
    parser.add_argument(
        "--sampling", choices=("shuffle", "sqrt_balanced"), default="shuffle"
    )
    parser.add_argument("--sampling-max-multiplier", type=float, default=8.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-weight-beta", type=float, default=0.9999)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
