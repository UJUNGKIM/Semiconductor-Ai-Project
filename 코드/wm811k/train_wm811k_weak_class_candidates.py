"""Compare train-only WM-811K weak-class interventions on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_wm811k_cnn import (
    WaferCNN,
    evaluate,
    save_evaluation_artifacts,
    seed_everything,
    sqrt_sampling_multipliers,
    train_epoch,
    validate_inputs,
)


WEAK_LABELS = (2, 4, 7)  # Edge-Loc, Loc, Scratch
CANDIDATES = (
    {"name": "baseline", "defect_dropout_max": 0.0, "hard_negative_multiplier": 1.0},
    {"name": "weak_defect_thinning", "defect_dropout_max": 0.12, "hard_negative_multiplier": 1.0},
    {"name": "weak_thinning_hard_negative", "defect_dropout_max": 0.12, "hard_negative_multiplier": 2.0},
)


class TrainOnlyWaferDataset(Dataset):
    """Two-channel dataset with optional weak-class defect thinning."""

    def __init__(
        self,
        maps_path: Path,
        indices: np.ndarray,
        labels: np.ndarray,
        *,
        augment: bool,
        defect_dropout_max: float = 0.0,
    ) -> None:
        if not 0 <= defect_dropout_max <= 0.25:
            raise ValueError("defect_dropout_max must be between 0 and 0.25")
        self.maps_path = str(maps_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.augment = augment
        self.defect_dropout_max = float(defect_dropout_max)
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

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        wafer = np.asarray(self._open_maps()[self.indices[position]], dtype=np.uint8)
        channels = np.stack((wafer > 0, wafer == 2), axis=0).astype(np.float32)
        if self.augment:
            channels = np.rot90(channels, k=np.random.randint(0, 4), axes=(1, 2))
            if np.random.random() < 0.5:
                channels = np.flip(channels, axis=1)
            if np.random.random() < 0.5:
                channels = np.flip(channels, axis=2)
            label = int(self.labels[position])
            defect_locations = np.argwhere(channels[1] > 0)
            if (
                label in WEAK_LABELS
                and self.defect_dropout_max > 0
                and len(defect_locations) >= 8
            ):
                dropout_rate = np.random.uniform(0.03, self.defect_dropout_max)
                drop_count = min(len(defect_locations) - 1, int(round(len(defect_locations) * dropout_rate)))
                if drop_count > 0:
                    chosen = np.random.choice(len(defect_locations), size=drop_count, replace=False)
                    rows, columns = defect_locations[chosen].T
                    channels[1, rows, columns] = 0.0
        return (
            torch.from_numpy(np.ascontiguousarray(channels)),
            torch.tensor(self.labels[position], dtype=torch.long),
        )


def hard_negative_indices(
    maps_path: Path,
    assignments: pd.DataFrame,
    fraction: float,
) -> np.ndarray:
    """Select the train-only none wafers with the highest defective-die ratio."""
    if not 0 < fraction <= 0.5:
        raise ValueError("hard-negative fraction must be in (0, 0.5]")
    subset = assignments.loc[
        assignments["split"].eq("train") & assignments["label_id"].eq(8),
        "array_index",
    ].to_numpy(dtype=np.int64)
    maps = np.load(maps_path, mmap_mode="r")
    ratios = np.empty(len(subset), dtype=np.float32)
    for start in range(0, len(subset), 4096):
        indices = subset[start : start + 4096]
        batch = np.asarray(maps[indices], dtype=np.uint8)
        active = (batch > 0).sum(axis=(1, 2))
        defect = (batch == 2).sum(axis=(1, 2))
        ratios[start : start + len(indices)] = np.divide(
            defect, active, out=np.zeros_like(defect, dtype=np.float32), where=active > 0
        )
    count = max(1, int(np.ceil(len(subset) * fraction)))
    order = np.lexsort((subset, -ratios))
    selected = np.sort(subset[order[:count]])
    if isinstance(maps, np.memmap) and getattr(maps, "_mmap", None) is not None:
        maps._mmap.close()
    return selected


def make_validation_only_loaders(
    maps_path: Path,
    assignments: pd.DataFrame,
    hard_indices: np.ndarray,
    candidate: dict[str, float | str],
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, np.ndarray]:
    generator = torch.Generator().manual_seed(seed)
    train = assignments.loc[assignments["split"].eq("train")]
    validation = assignments.loc[assignments["split"].eq("validation")]
    train_labels = train["label_id"].to_numpy(dtype=np.int64)
    _, multipliers = sqrt_sampling_multipliers(train_labels, 9, 8.0)
    sample_weights = multipliers[train_labels].astype(np.float64)
    hard_multiplier = float(candidate["hard_negative_multiplier"])
    if hard_multiplier > 1:
        sample_weights[np.isin(train["array_index"].to_numpy(), hard_indices)] *= hard_multiplier
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    train_dataset = TrainOnlyWaferDataset(
        maps_path,
        train["array_index"].to_numpy(),
        train_labels,
        augment=True,
        defect_dropout_max=float(candidate["defect_dropout_max"]),
    )
    validation_dataset = TrainOnlyWaferDataset(
        maps_path,
        validation["array_index"].to_numpy(),
        validation["label_id"].to_numpy(),
        augment=False,
    )
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "generator": generator,
    }
    return (
        DataLoader(train_dataset, sampler=sampler, **common),
        DataLoader(validation_dataset, shuffle=False, **common),
        sample_weights,
    )


def choose_candidate(
    comparison: pd.DataFrame,
    macro_f1_tolerance: float = 0.005,
    none_recall_tolerance: float = 0.002,
) -> tuple[str, pd.DataFrame]:
    baseline = comparison.loc[comparison["candidate"].eq("baseline")]
    if len(baseline) != 1:
        raise ValueError("Exactly one baseline row is required")
    base = baseline.iloc[0]
    result = comparison.copy()
    result["macro_f1_guardrail"] = result["validation_macro_f1"] >= base["validation_macro_f1"] - macro_f1_tolerance
    result["none_recall_guardrail"] = result["validation_none_recall"] >= base["validation_none_recall"] - none_recall_tolerance
    result["eligible"] = result["macro_f1_guardrail"] & result["none_recall_guardrail"]
    eligible = result.loc[result["eligible"]].sort_values(
        ["validation_weak_recall", "validation_macro_f1", "candidate"],
        ascending=[False, False, True],
    )
    return str(eligible.iloc[0]["candidate"]), result


def run_candidate(
    candidate: dict[str, float | str],
    args: argparse.Namespace,
    assignments: pd.DataFrame,
    class_names: list[str],
    hard_indices: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(args.seed)
    output_dir = args.output_dir / str(candidate["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, sample_weights = make_validation_only_loaders(
        args.maps, assignments, hard_indices, candidate,
        args.batch_size, args.num_workers, args.seed, device,
    )
    model = WaferCNN(num_classes=len(class_names)).to(device)
    loss_function: nn.Module = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_macro_f1 = -1.0
    best_epoch = 0
    stale = 0
    history = []
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_function, device, scaler)
        metrics, _, _, _ = evaluate(model, validation_loader, loss_function, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"validation_{key}": value for key, value in metrics.items()}})
        print(json.dumps({"candidate": candidate["name"], **history[-1]}, ensure_ascii=False), flush=True)
        if metrics["macro_f1"] > best_macro_f1 + args.minimum_delta:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(), "class_names": class_names,
                "num_classes": len(class_names), "input_channels": 2, "image_size": 64,
                "best_epoch": best_epoch, "best_validation_macro_f1": best_macro_f1,
                "loss": "cross_entropy", "sampling": "sqrt_balanced_targeted",
                "candidate": candidate, "seed": args.seed,
            }, checkpoint_path)
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, true, predicted, probabilities = evaluate(model, validation_loader, loss_function, device)
    save_evaluation_artifacts(assignments, "validation", true, predicted, probabilities, class_names, output_dir)
    report = pd.read_csv(output_dir / "validation_classification_report.csv", index_col=0)
    weak_recall = float(report.loc[[class_names[index] for index in WEAK_LABELS], "recall"].mean())
    none_recall = float(report.loc["none", "recall"])
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    summary = {
        "candidate": candidate["name"], "seed": args.seed,
        "defect_dropout_max": candidate["defect_dropout_max"],
        "hard_negative_multiplier": candidate["hard_negative_multiplier"],
        "best_epoch": best_epoch, "epochs_ran": len(history),
        "hard_negative_count": len(hard_indices),
        "sample_weight_min": float(sample_weights.min()),
        "sample_weight_max": float(sample_weights.max()),
        "validation_accuracy": metrics["accuracy"],
        "validation_balanced_accuracy": metrics["balanced_accuracy"],
        "validation_macro_f1": metrics["macro_f1"],
        "validation_weighted_f1": metrics["weighted_f1"],
        "validation_weak_recall": weak_recall,
        "validation_none_recall": none_recall,
        "test_evaluated": False,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def reusable_candidate_summary(
    output_root: Path,
    candidate: dict[str, float | str],
    seed: int,
) -> dict[str, object] | None:
    candidate_dir = output_root / str(candidate["name"])
    summary_path = candidate_dir / "run_summary.json"
    required = (
        candidate_dir / "best_model.pt",
        candidate_dir / "validation_classification_report.csv",
        candidate_dir / "validation_predictions.csv",
    )
    if not summary_path.is_file() or not all(path.is_file() for path in required):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = {
        "candidate": candidate["name"],
        "seed": seed,
        "defect_dropout_max": candidate["defect_dropout_max"],
        "hard_negative_multiplier": candidate["hard_negative_multiplier"],
        "test_evaluated": False,
    }
    if any(summary.get(key) != value for key, value in contract.items()):
        return None
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    maps_reference, labels_reference, assignments = validate_inputs(
        args.maps, args.labels, args.assignments
    )
    for reference in (maps_reference, labels_reference):
        if isinstance(reference, np.memmap) and getattr(reference, "_mmap", None) is not None:
            reference._mmap.close()
    class_names = assignments[["label_id", "failure_type"]].drop_duplicates().sort_values("label_id")["failure_type"].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hard_indices = hard_negative_indices(args.maps, assignments, args.hard_negative_fraction)
    pd.DataFrame({"array_index": hard_indices}).to_csv(args.output_dir / "train_hard_negative_indices.csv", index=False)
    rows = []
    for candidate in CANDIDATES:
        cached = reusable_candidate_summary(args.output_dir, candidate, args.seed)
        if cached is not None:
            print(f"reuse_completed_candidate={candidate['name']}", flush=True)
            rows.append(cached)
        else:
            rows.append(
                run_candidate(
                    candidate, args, assignments, class_names, hard_indices, device
                )
            )
    selected, comparison = choose_candidate(pd.DataFrame(rows))
    comparison.to_csv(args.output_dir / "validation_candidate_comparison.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    comparison.plot.bar(x="candidate", y=["validation_macro_f1", "validation_weak_recall", "validation_none_recall"], ax=axes[0], ylim=(0.65, 1), rot=15)
    axes[0].set_title("Validation-only candidate metrics")
    guardrail_plot = comparison[["candidate", "macro_f1_guardrail", "none_recall_guardrail"]].copy()
    guardrail_plot[["macro_f1_guardrail", "none_recall_guardrail"]] = guardrail_plot[["macro_f1_guardrail", "none_recall_guardrail"]].astype(int)
    guardrail_plot.plot.bar(x="candidate", y=["macro_f1_guardrail", "none_recall_guardrail"], ax=axes[1], ylim=(0, 1.1), rot=15)
    axes[1].set_title("Guardrails (1=pass)")
    figure.tight_layout(); figure.savefig(args.output_dir / "validation_candidate_dashboard.png", dpi=170); plt.close(figure)
    summary = {
        "schema_version": 1,
        "purpose": "validation_only_weak_class_candidate_comparison",
        "used_test_for_selection": False,
        "test_evaluated": False,
        "selected_candidate": selected,
        "selection_metric": "validation_weak_recall_with_macro_f1_and_none_recall_guardrails",
        "macro_f1_tolerance": 0.005,
        "none_recall_tolerance": 0.002,
        "hard_negative_fraction": args.hard_negative_fraction,
        "hard_negative_source": "train_none_only",
        "candidates": comparison.to_dict(orient="records"),
        "next_step": "replicate_selected_candidate_across_seeds_before_any_test_evaluation",
    }
    (args.output_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# WM-811K 취약 클래스 개선 후보 비교\n\n"
        "세 후보를 train으로 학습하고 validation으로만 비교합니다. Test 추론은 수행하지 않습니다.\n\n"
        f"Validation 선택 후보: {selected}\n\n"
        "선택 후보는 다음 단계에서 여러 seed로 재현한 뒤에만 test 최종 평가 대상으로 검토합니다.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
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
    parser.add_argument("--hard-negative-fraction", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())
