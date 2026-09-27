"""Validation-only WM-811K robustness and hierarchical-classifier experiment.

The script compares three candidates on the same lot-aware split and three fixed
seeds.  The test split is intentionally never loaded for inference or selection.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
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
    f1_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_wm811k_cnn import (
    WaferCNN,
    evaluate,
    plot_history,
    save_evaluation_artifacts,
    seed_everything,
    sqrt_sampling_multipliers,
    train_epoch,
    validate_inputs,
)


SEEDS = (17, 42, 2026)
NONE_LABEL = 8
WEAK_LABELS = (2, 4, 7)  # Edge-Loc, Loc, Scratch
CANDIDATES = ("baseline", "robust_flat", "robust_hierarchical")
SHIFT_PIXELS = 4
DROPOUT_FRACTION = 0.01


def zero_fill_shift(channels: np.ndarray, row_shift: int, column_shift: int) -> np.ndarray:
    """Translate CHW data without wraparound."""
    shifted = np.zeros_like(channels)
    height, width = channels.shape[1:]
    source_r0 = max(0, -row_shift)
    source_r1 = min(height, height - row_shift)
    source_c0 = max(0, -column_shift)
    source_c1 = min(width, width - column_shift)
    target_r0 = max(0, row_shift)
    target_r1 = target_r0 + max(0, source_r1 - source_r0)
    target_c0 = max(0, column_shift)
    target_c1 = target_c0 + max(0, source_c1 - source_c0)
    if source_r1 > source_r0 and source_c1 > source_c0:
        shifted[:, target_r0:target_r1, target_c0:target_c1] = channels[
            :, source_r0:source_r1, source_c0:source_c1
        ]
    return shifted


def drop_active_dies(
    channels: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Drop active die locations from both channels while retaining one defect."""
    result = channels.copy()
    active = np.argwhere(result[0] > 0)
    if len(active) == 0 or fraction <= 0:
        return result
    count = min(len(active) - 1, max(1, int(round(len(active) * fraction))))
    if count <= 0:
        return result
    selected = rng.choice(len(active), size=count, replace=False)
    rows, columns = active[selected].T
    original_defects = np.argwhere(result[1] > 0)
    result[:, rows, columns] = 0
    if len(original_defects) and not np.any(result[1] > 0):
        restore_row, restore_column = original_defects[0]
        result[0, restore_row, restore_column] = 1
        result[1, restore_row, restore_column] = 1
    return result


class RobustWaferDataset(Dataset):
    """Memory-mapped wafer dataset with train-only or deterministic stress transforms."""

    def __init__(
        self,
        maps_path: Path,
        indices: np.ndarray,
        labels: np.ndarray,
        *,
        task: str,
        train_augment: bool = False,
        robust_train: bool = False,
        stress_mode: str = "clean",
        seed: int = 42,
    ) -> None:
        if task not in {"flat", "binary", "defect"}:
            raise ValueError(f"Unknown task: {task}")
        if stress_mode not in {"clean", "shift4", "dropout1"}:
            raise ValueError(f"Unknown stress mode: {stress_mode}")
        self.maps_path = str(maps_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.task = task
        self.train_augment = bool(train_augment)
        self.robust_train = bool(robust_train)
        self.stress_mode = stress_mode
        self.seed = int(seed)
        self._maps: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _open_maps(self) -> np.ndarray:
        if self._maps is None:
            self._maps = np.load(self.maps_path, mmap_mode="r")
        return self._maps

    def close(self) -> None:
        if isinstance(self._maps, np.memmap) and getattr(self._maps, "_mmap", None) is not None:
            self._maps._mmap.close()
        self._maps = None

    def __del__(self) -> None:
        self.close()

    def _target(self, label: int) -> int:
        if self.task == "binary":
            return int(label != NONE_LABEL)
        return label

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        array_index = int(self.indices[position])
        wafer = np.asarray(self._open_maps()[array_index], dtype=np.uint8)
        channels = np.stack((wafer > 0, wafer == 2), axis=0).astype(np.float32)

        if self.train_augment:
            channels = np.rot90(channels, k=np.random.randint(0, 4), axes=(1, 2))
            if np.random.random() < 0.5:
                channels = np.flip(channels, axis=1)
            if np.random.random() < 0.5:
                channels = np.flip(channels, axis=2)
            if self.robust_train and np.random.random() < 0.70:
                row_shift = int(np.random.randint(-SHIFT_PIXELS, SHIFT_PIXELS + 1))
                column_shift = int(np.random.randint(-SHIFT_PIXELS, SHIFT_PIXELS + 1))
                channels = zero_fill_shift(channels, row_shift, column_shift)
            if self.robust_train and np.random.random() < 0.50:
                fraction = float(np.random.uniform(0.0025, 0.02))
                transform_seed = int(np.random.randint(0, np.iinfo(np.uint32).max))
                channels = drop_active_dies(
                    channels, fraction, np.random.default_rng(transform_seed)
                )

        if self.stress_mode == "shift4":
            shifts = ((SHIFT_PIXELS, 0), (-SHIFT_PIXELS, 0), (0, SHIFT_PIXELS), (0, -SHIFT_PIXELS))
            row_shift, column_shift = shifts[position % len(shifts)]
            channels = zero_fill_shift(channels, row_shift, column_shift)
        elif self.stress_mode == "dropout1":
            rng = np.random.default_rng(self.seed * 1_000_003 + array_index)
            channels = drop_active_dies(channels, DROPOUT_FRACTION, rng)

        return (
            torch.from_numpy(np.ascontiguousarray(channels)),
            torch.tensor(self._target(int(self.labels[position])), dtype=torch.long),
        )


def task_rows(assignments: pd.DataFrame, split: str, task: str) -> pd.DataFrame:
    rows = assignments.loc[assignments["split"].eq(split)].copy()
    if task == "defect":
        rows = rows.loc[rows["label_id"].ne(NONE_LABEL)]
    return rows


def task_targets(labels: np.ndarray, task: str) -> np.ndarray:
    if task == "binary":
        return (labels != NONE_LABEL).astype(np.int64)
    return labels.astype(np.int64)


def make_loader(
    maps_path: Path,
    assignments: pd.DataFrame,
    split: str,
    task: str,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
    *,
    train_augment: bool = False,
    robust_train: bool = False,
    stress_mode: str = "clean",
) -> DataLoader:
    rows = task_rows(assignments, split, task)
    labels = rows["label_id"].to_numpy(dtype=np.int64)
    dataset = RobustWaferDataset(
        maps_path,
        rows["array_index"].to_numpy(dtype=np.int64),
        labels,
        task=task,
        train_augment=train_augment,
        robust_train=robust_train,
        stress_mode=stress_mode,
        seed=seed,
    )
    sampler = None
    shuffle = False
    generator = torch.Generator().manual_seed(seed)
    if split == "train":
        targets = task_targets(labels, task)
        class_count = 2 if task == "binary" else (8 if task == "defect" else 9)
        maximum = 4.0 if task == "binary" else 8.0
        _, multipliers = sqrt_sampling_multipliers(targets, class_count, maximum)
        sample_weights = torch.as_tensor(multipliers[targets], dtype=torch.double)
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def train_stage(
    maps_path: Path,
    assignments: pd.DataFrame,
    output_dir: Path,
    task: str,
    robust_train: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    summary_path = output_dir / "stage_summary.json"
    if checkpoint_path.is_file() and summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("seed") == args.seed and cached.get("task") == task and cached.get("robust_train") == robust_train:
            print(f"reuse stage={task} seed={args.seed} robust={robust_train}", flush=True)
            return checkpoint_path

    seed_everything(args.seed)
    num_classes = 2 if task == "binary" else (8 if task == "defect" else 9)
    train_loader = make_loader(
        maps_path, assignments, "train", task, args.batch_size, args.num_workers,
        args.seed, device, train_augment=True, robust_train=robust_train,
    )
    validation_loader = make_loader(
        maps_path, assignments, "validation", task, args.batch_size,
        args.num_workers, args.seed, device,
    )
    model = WaferCNN(num_classes=num_classes).to(device)
    loss_function: nn.Module = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_function, device, scaler)
        metrics, _, _, _ = evaluate(model, validation_loader, loss_function, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            **{f"validation_{key}": float(value) for key, value in metrics.items()},
        }
        history.append(row)
        print(json.dumps({"task": task, "seed": args.seed, **row}, ensure_ascii=False), flush=True)
        if metrics["macro_f1"] > best_macro_f1 + args.minimum_delta:
            best_macro_f1 = float(metrics["macro_f1"])
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_role": task,
                    "num_classes": num_classes,
                    "input_channels": 2,
                    "image_size": 64,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_macro_f1,
                    "robust_train": robust_train,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    plot_history(history_frame, output_dir / "training_curves.png")
    summary = {
        "seed": args.seed,
        "task": task,
        "robust_train": robust_train,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_macro_f1": best_macro_f1,
        "test_evaluated": False,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    del model, train_loader, validation_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint_path


def load_model(checkpoint_path: Path, device: torch.device) -> WaferCNN:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WaferCNN(num_classes=int(checkpoint["num_classes"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def model_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for inputs, targets in loader:
        logits = model(inputs.to(device, non_blocking=True))
        labels.append(targets.numpy())
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def combine_hierarchical_probabilities(binary: np.ndarray, defect: np.ndarray) -> np.ndarray:
    if binary.ndim != 2 or binary.shape[1] != 2:
        raise ValueError("Binary probabilities must have shape (N, 2)")
    if defect.ndim != 2 or defect.shape[1] != 8 or len(binary) != len(defect):
        raise ValueError("Defect probabilities must have shape (N, 8) with matching rows")
    combined = np.empty((len(binary), 9), dtype=np.float32)
    combined[:, :8] = binary[:, 1:2] * defect
    combined[:, NONE_LABEL] = binary[:, 0]
    combined /= combined.sum(axis=1, keepdims=True)
    return combined


def metric_bundle(true: np.ndarray, probabilities: np.ndarray, class_names: list[str]) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    report = classification_report(
        true,
        predicted,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, predicted, average="weighted", zero_division=0)),
        "weak_recall": float(np.mean([report[class_names[index]]["recall"] for index in WEAK_LABELS])),
        "none_recall": float(report[class_names[NONE_LABEL]]["recall"]),
    }


def candidate_probabilities(
    candidate_dir: Path,
    candidate: str,
    maps_path: Path,
    assignments: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    stress_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    clean_loader = make_loader(
        maps_path, assignments, "validation", "flat", args.batch_size,
        args.num_workers, args.seed, device, stress_mode=stress_mode,
    )
    if candidate != "robust_hierarchical":
        model = load_model(candidate_dir / "flat" / "best_model.pt", device)
        true, probabilities = model_probabilities(model, clean_loader, device)
        del model
    else:
        binary_model = load_model(candidate_dir / "binary" / "best_model.pt", device)
        defect_model = load_model(candidate_dir / "defect" / "best_model.pt", device)
        true, binary_probabilities = model_probabilities(binary_model, clean_loader, device)
        _, defect_probabilities = model_probabilities(defect_model, clean_loader, device)
        probabilities = combine_hierarchical_probabilities(binary_probabilities, defect_probabilities)
        del binary_model, defect_model
    del clean_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return true, probabilities


def train_candidate(
    candidate: str,
    seed_root: Path,
    maps_path: Path,
    assignments: pd.DataFrame,
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    candidate_dir = seed_root / candidate
    candidate_dir.mkdir(parents=True, exist_ok=True)
    robust_train = candidate != "baseline"
    if candidate == "robust_hierarchical":
        train_stage(maps_path, assignments, candidate_dir / "binary", "binary", True, args, device)
        train_stage(maps_path, assignments, candidate_dir / "defect", "defect", True, args, device)
    else:
        train_stage(maps_path, assignments, candidate_dir / "flat", "flat", robust_train, args, device)

    results: dict[str, dict[str, float]] = {}
    clean_probabilities: np.ndarray | None = None
    clean_true: np.ndarray | None = None
    clean_predictions: np.ndarray | None = None
    for stress_mode in ("clean", "shift4", "dropout1"):
        true, probabilities = candidate_probabilities(
            candidate_dir, candidate, maps_path, assignments, args, device, stress_mode
        )
        results[stress_mode] = metric_bundle(true, probabilities, class_names)
        if stress_mode == "clean":
            clean_true = true
            clean_probabilities = probabilities
            clean_predictions = probabilities.argmax(axis=1)
        else:
            results[stress_mode]["prediction_stability"] = float(
                np.mean(probabilities.argmax(axis=1) == clean_predictions)
            )
    assert clean_true is not None and clean_probabilities is not None
    save_evaluation_artifacts(
        assignments,
        "validation",
        clean_true,
        clean_probabilities.argmax(axis=1),
        clean_probabilities,
        class_names,
        candidate_dir,
    )
    summary: dict[str, object] = {
        "candidate": candidate,
        "seed": args.seed,
        "architecture": "two_stage_binary_then_defect" if candidate == "robust_hierarchical" else "flat_9_class",
        "robust_train_augmentation": robust_train,
        "clean": results["clean"],
        "shift4": results["shift4"],
        "dropout1": results["dropout1"],
        "selection_split": "validation",
        "test_evaluated": False,
    }
    (candidate_dir / "candidate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def flatten_summary(summary: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "seed": summary["seed"],
        "candidate": summary["candidate"],
        "architecture": summary["architecture"],
        "test_evaluated": summary["test_evaluated"],
    }
    for stress_mode in ("clean", "shift4", "dropout1"):
        for metric, value in dict(summary[stress_mode]).items():
            row[f"{stress_mode}_{metric}"] = value
    return row


def candidate_is_complete(candidate_dir: Path, candidate: str) -> bool:
    required = [
        candidate_dir / "candidate_summary.json",
        candidate_dir / "validation_classification_report.csv",
        candidate_dir / "validation_predictions.csv",
    ]
    if candidate == "robust_hierarchical":
        required.extend(
            [
                candidate_dir / "binary" / "best_model.pt",
                candidate_dir / "binary" / "stage_summary.json",
                candidate_dir / "defect" / "best_model.pt",
                candidate_dir / "defect" / "stage_summary.json",
            ]
        )
    else:
        required.extend(
            [
                candidate_dir / "flat" / "best_model.pt",
                candidate_dir / "flat" / "stage_summary.json",
            ]
        )
    return all(path.is_file() for path in required)


def select_candidate(metrics: pd.DataFrame) -> dict[str, object]:
    baseline = metrics.loc[metrics["candidate"].eq("baseline")].set_index("seed")
    decision_rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        current = metrics.loc[metrics["candidate"].eq(candidate)].set_index("seed")
        if list(current.index) != list(baseline.index):
            current = current.reindex(baseline.index)
        deltas = current.select_dtypes(include=[np.number]) - baseline.select_dtypes(include=[np.number])
        if candidate == "baseline":
            checks = {
                "mean_weak_recall_improved": True,
                "weak_recall_improved_at_least_two_seeds": True,
                "macro_f1_guardrail_all_seeds": True,
                "none_recall_guardrail_all_seeds": True,
                "balanced_accuracy_guardrail_all_seeds": True,
                "shift_macro_f1_mean_improved": True,
                "dropout_macro_f1_mean_improved": True,
            }
        else:
            checks = {
                "mean_weak_recall_improved": float(deltas["clean_weak_recall"].mean()) > 0,
                "weak_recall_improved_at_least_two_seeds": int((deltas["clean_weak_recall"] > 0).sum()) >= 2,
                "macro_f1_guardrail_all_seeds": bool((deltas["clean_macro_f1"] >= -0.005).all()),
                "none_recall_guardrail_all_seeds": bool((deltas["clean_none_recall"] >= -0.002).all()),
                "balanced_accuracy_guardrail_all_seeds": bool((deltas["clean_balanced_accuracy"] >= -0.005).all()),
                "shift_macro_f1_mean_improved": float(deltas["shift4_macro_f1"].mean()) > 0,
                "dropout_macro_f1_mean_improved": float(deltas["dropout1_macro_f1"].mean()) > 0,
            }
        eligible = all(checks.values())
        score = float(
            current["clean_weak_recall"].mean()
            + current["shift4_macro_f1"].mean()
            + current["dropout1_macro_f1"].mean()
        )
        decision_rows.append(
            {
                "candidate": candidate,
                "eligible": eligible,
                "score": score,
                "checks": checks,
                "mean_deltas": {
                    column: float(deltas[column].mean())
                    for column in (
                        "clean_macro_f1", "clean_balanced_accuracy", "clean_weak_recall",
                        "clean_none_recall", "shift4_macro_f1", "dropout1_macro_f1",
                    )
                },
            }
        )
    eligible_rows = [row for row in decision_rows if row["eligible"]]
    selected = max(eligible_rows, key=lambda row: (row["score"], row["candidate"]))
    return {
        "selected_candidate": selected["candidate"],
        "deployment_model_changed": False,
        "test_evaluated": False,
        "used_test_for_selection": False,
        "guardrails": {
            "macro_f1_tolerance_per_seed": 0.005,
            "none_recall_tolerance_per_seed": 0.002,
            "balanced_accuracy_tolerance_per_seed": 0.005,
        },
        "candidate_decisions": decision_rows,
        "next_step": "run_one_locked_test_evaluation_only_if_a_challenger_is_selected",
    }


def plot_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = (
        ("clean_weak_recall", "Clean weak-class recall"),
        ("shift4_macro_f1", "4 px shift macro-F1"),
        ("dropout1_macro_f1", "1% die dropout macro-F1"),
    )
    for axis, (column, title) in zip(axes, panels):
        grouped = metrics.groupby("candidate")[column].agg(["mean", "std"]).reindex(CANDIDATES)
        axis.bar(grouped.index, grouped["mean"], yerr=grouped["std"], capsize=4)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    maps, labels, assignments = validate_inputs(args.maps, args.labels, args.assignments)
    for reference in (maps, labels):
        if isinstance(reference, np.memmap) and getattr(reference, "_mmap", None) is not None:
            reference._mmap.close()
    class_names = (
        assignments[["label_id", "failure_type"]]
        .drop_duplicates()
        .sort_values("label_id")["failure_type"]
        .tolist()
    )
    if len(class_names) != 9 or class_names[NONE_LABEL] != "none":
        raise ValueError(f"Unexpected class order: {class_names}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        seed_root = args.output_dir / "runs" / f"seed_{seed}"
        for candidate in CANDIDATES:
            args.seed = seed
            summary_path = seed_root / candidate / "candidate_summary.json"
            candidate_dir = seed_root / candidate
            if candidate_is_complete(candidate_dir, candidate):
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("seed") != seed or summary.get("candidate") != candidate:
                    raise ValueError(f"Cached summary contract mismatch: {summary_path}")
                print(f"reuse candidate={candidate} seed={seed}", flush=True)
            else:
                summary = train_candidate(
                    candidate, seed_root, args.maps, assignments, class_names, args, device
                )
            rows.append(flatten_summary(summary))

    metrics = pd.DataFrame(rows).sort_values(["seed", "candidate"]).reset_index(drop=True)
    metrics.to_csv(args.output_dir / "seed_candidate_metrics.csv", index=False)
    metric_columns = [
        column
        for column in metrics.columns
        if column not in {"seed", "candidate", "architecture", "test_evaluated"}
        and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    aggregate = metrics.groupby("candidate")[metric_columns].agg(
        ["mean", "std", "min", "max"]
    )
    aggregate.to_csv(args.output_dir / "metric_summary.csv")
    decision = select_candidate(metrics)
    protocol = {
        "schema_version": 1,
        "experiment_type": "wm811k_robust_hierarchical_multiseed_validation",
        "candidates": list(CANDIDATES),
        "seeds": list(args.seeds),
        "selection_split": "validation",
        "test_evaluated": False,
        "used_test_for_selection": False,
        "fixed_lot_aware_split": True,
        "stress_tests": {
            "translation_pixels": SHIFT_PIXELS,
            "active_die_dropout_fraction": DROPOUT_FRACTION,
        },
        "train_only_robust_augmentation": {
            "translation_probability": 0.70,
            "translation_max_pixels": SHIFT_PIXELS,
            "active_die_dropout_probability": 0.50,
            "active_die_dropout_range": [0.0025, 0.02],
        },
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {**protocol, **decision}
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_comparison(metrics, args.output_dir / "candidate_comparison.png")
    (args.output_dir / "README.md").write_text(
        "# WM-811K 견고성·2단계 분류 후보 실험\n\n"
        "동일한 lot-aware validation과 seed 17/42/2026에서 baseline, 견고성 증강 flat CNN, "
        "정상/불량→불량유형 2단계 CNN을 비교합니다. 후보 선택 과정에서는 test 추론을 수행하지 않습니다.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    args.seeds = tuple(args.seeds)
    if args.seeds != SEEDS:
        raise ValueError(f"Registered protocol requires seeds {SEEDS}")
    return args


if __name__ == "__main__":
    run(parse_args())
