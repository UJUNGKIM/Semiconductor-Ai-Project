"""Compare conservative weak-vs-none margin candidates on WM-811K validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_wm811k_cnn import (
    WaferCNN,
    evaluate,
    save_evaluation_artifacts,
    seed_everything,
    sqrt_sampling_multipliers,
    train_epoch,
    validate_inputs,
)
from train_wm811k_weak_class_candidates import TrainOnlyWaferDataset, choose_candidate


WEAK_LABELS = (2, 4, 7)
NONE_LABEL = 8
CANDIDATES = (
    {"name": "baseline", "weak_margin_lambda": 0.0, "weak_margin_value": 0.20},
    {"name": "weak_none_margin_005", "weak_margin_lambda": 0.05, "weak_margin_value": 0.20},
    {"name": "weak_none_margin_010", "weak_margin_lambda": 0.10, "weak_margin_value": 0.20},
)


class WeakNoneMarginLoss(nn.Module):
    """Cross-entropy plus a weak-class margin against the none logit."""

    def __init__(self, penalty_weight: float, margin: float) -> None:
        super().__init__()
        if penalty_weight < 0 or margin < 0:
            raise ValueError("penalty_weight and margin must be non-negative")
        self.penalty_weight = float(penalty_weight)
        self.margin = float(margin)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = F.cross_entropy(logits, targets)
        if self.penalty_weight == 0:
            return base
        weak_mask = torch.zeros_like(targets, dtype=torch.bool)
        for label in WEAK_LABELS:
            weak_mask |= targets.eq(label)
        if not bool(weak_mask.any()):
            return base
        true_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        penalty = F.relu(logits[:, NONE_LABEL] - true_logits + self.margin)
        return base + self.penalty_weight * penalty[weak_mask].mean()


def make_loaders(
    maps_path: Path,
    assignments: pd.DataFrame,
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
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights), replacement=True, generator=generator,
    )
    train_dataset = TrainOnlyWaferDataset(
        maps_path, train["array_index"].to_numpy(), train_labels,
        augment=True, defect_dropout_max=0.0,
    )
    validation_dataset = TrainOnlyWaferDataset(
        maps_path, validation["array_index"].to_numpy(),
        validation["label_id"].to_numpy(), augment=False,
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


def reusable_summary(output_root: Path, candidate: dict[str, float | str], seed: int) -> dict | None:
    directory = output_root / str(candidate["name"])
    summary_path = directory / "run_summary.json"
    required = (
        directory / "best_model.pt",
        directory / "validation_classification_report.csv",
        directory / "validation_predictions.csv",
    )
    if not summary_path.is_file() or not all(path.is_file() for path in required):
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = {
        "candidate": candidate["name"],
        "seed": seed,
        "weak_margin_lambda": candidate["weak_margin_lambda"],
        "weak_margin_value": candidate["weak_margin_value"],
        "test_evaluated": False,
    }
    return summary if all(summary.get(key) == value for key, value in contract.items()) else None


def import_baseline(source: Path, output_root: Path, seed: int) -> dict:
    source_summary = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
    if source_summary.get("candidate") != "baseline" or source_summary.get("seed") != seed:
        raise ValueError("Baseline source candidate or seed mismatch")
    if source_summary.get("test_evaluated") is not False:
        raise ValueError("Baseline source must be validation-only")
    required = (
        "best_model.pt", "run_summary.json", "training_history.csv",
        "validation_classification_report.csv", "validation_confusion_matrix.csv",
        "validation_confusion_matrix.png", "validation_predictions.csv",
    )
    if not all((source / name).is_file() for name in required):
        raise FileNotFoundError("Baseline source is incomplete")
    destination = output_root / "baseline"
    destination.mkdir(parents=True, exist_ok=True)
    for name in required:
        shutil.copy2(source / name, destination / name)
    source_summary.update({
        "weak_margin_lambda": 0.0,
        "weak_margin_value": 0.20,
        "baseline_reused": True,
    })
    (destination / "run_summary.json").write_text(
        json.dumps(source_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return source_summary


def train_candidate(
    candidate: dict[str, float | str], args: argparse.Namespace,
    assignments: pd.DataFrame, class_names: list[str], device: torch.device,
) -> dict[str, object]:
    seed_everything(args.seed)
    output_dir = args.output_dir / str(candidate["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, sample_weights = make_loaders(
        args.maps, assignments, args.batch_size, args.num_workers, args.seed, device
    )
    model = WaferCNN(num_classes=len(class_names)).to(device)
    loss_function = WeakNoneMarginLoss(
        float(candidate["weak_margin_lambda"]), float(candidate["weak_margin_value"])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_macro_f1, best_epoch, stale = -1.0, 0, 0
    history = []
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_function, device, scaler)
        validation_metrics, _, _, _ = evaluate(
            model, validation_loader, loss_function, device
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        })
        print(json.dumps({"candidate": candidate["name"], **history[-1]}), flush=True)
        if validation_metrics["macro_f1"] > best_macro_f1 + args.minimum_delta:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch, stale = epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(), "class_names": class_names,
                "num_classes": len(class_names), "input_channels": 2, "image_size": 64,
                "best_epoch": best_epoch, "best_validation_macro_f1": best_macro_f1,
                "loss": "weak_none_margin", "sampling": "sqrt_balanced",
                "candidate": candidate, "seed": args.seed,
            }, checkpoint_path)
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, true, predicted, probabilities = evaluate(
        model, validation_loader, loss_function, device
    )
    save_evaluation_artifacts(
        assignments, "validation", true, predicted, probabilities, class_names, output_dir
    )
    report = pd.read_csv(output_dir / "validation_classification_report.csv", index_col=0)
    summary = {
        "candidate": candidate["name"], "seed": args.seed,
        "weak_margin_lambda": candidate["weak_margin_lambda"],
        "weak_margin_value": candidate["weak_margin_value"],
        "best_epoch": best_epoch, "epochs_ran": len(history),
        "sample_weight_min": float(sample_weights.min()),
        "sample_weight_max": float(sample_weights.max()),
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_weighted_f1": validation_metrics["weighted_f1"],
        "validation_weak_recall": float(
            report.loc[[class_names[index] for index in WEAK_LABELS], "recall"].mean()
        ),
        "validation_none_recall": float(report.loc["none", "recall"]),
        "test_evaluated": False,
        "baseline_reused": False,
    }
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    maps_reference, labels_reference, assignments = validate_inputs(
        args.maps, args.labels, args.assignments
    )
    for reference in (maps_reference, labels_reference):
        if isinstance(reference, np.memmap) and getattr(reference, "_mmap", None) is not None:
            reference._mmap.close()
    class_names = (
        assignments[["label_id", "failure_type"]].drop_duplicates()
        .sort_values("label_id")["failure_type"].tolist()
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in CANDIDATES:
        cached = reusable_summary(args.output_dir, candidate, args.seed)
        if cached is not None:
            print(f"reuse_completed_candidate={candidate['name']}", flush=True)
            rows.append(cached)
        elif candidate["name"] == "baseline" and args.baseline_source is not None:
            rows.append(import_baseline(args.baseline_source, args.output_dir, args.seed))
            print("reused_validation_only_baseline", flush=True)
        else:
            rows.append(train_candidate(candidate, args, assignments, class_names, device))
    selected, comparison = choose_candidate(pd.DataFrame(rows))
    comparison.to_csv(args.output_dir / "validation_margin_candidate_comparison.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    comparison.plot.bar(
        x="candidate", y=["validation_macro_f1", "validation_weak_recall", "validation_none_recall"],
        ax=axes[0], ylim=(0.65, 1.0), rot=15,
    )
    axes[0].set_title("Validation-only margin candidates")
    guardrails = comparison[["candidate", "macro_f1_guardrail", "none_recall_guardrail"]].copy()
    guardrails[["macro_f1_guardrail", "none_recall_guardrail"]] = guardrails[
        ["macro_f1_guardrail", "none_recall_guardrail"]
    ].astype(int)
    guardrails.plot.bar(
        x="candidate", y=["macro_f1_guardrail", "none_recall_guardrail"],
        ax=axes[1], ylim=(0, 1.1), rot=15,
    )
    axes[1].set_title("Guardrails (1=pass)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "validation_margin_dashboard.png", dpi=170)
    plt.close(figure)
    summary = {
        "schema_version": 1,
        "purpose": "validation_only_weak_none_margin_candidate_comparison",
        "used_test_for_selection": False,
        "test_evaluated": False,
        "selected_candidate": selected,
        "selection_metric": "validation_weak_recall_with_macro_f1_and_none_recall_guardrails",
        "macro_f1_tolerance": 0.005,
        "none_recall_tolerance": 0.002,
        "candidates": comparison.to_dict(orient="records"),
        "next_step": "replicate_nonbaseline_winner_across_seeds_or_keep_baseline",
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# WM-811K weak-vs-none margin 후보\n\n"
        "픽셀을 훼손하지 않고 취약 클래스의 true logit이 none logit보다 높아지도록 "
        "train loss에만 작은 margin penalty를 적용합니다. Test 추론은 수행하지 않습니다.\n\n"
        f"Validation 선택 후보: {selected}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
