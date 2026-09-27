"""Run paired multi-seed WM-811K baseline vs weak-margin validation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train_wm811k_cnn import validate_inputs
from train_wm811k_weak_margin_candidates import (
    CANDIDATES,
    reusable_summary,
    train_candidate,
)


STRATEGIES = ("baseline", "weak_none_margin_005")
METRICS = (
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_macro_f1",
    "validation_weighted_f1",
    "validation_weak_recall",
    "validation_none_recall",
)
REQUIRED_RUN_FILES = (
    "best_model.pt", "run_summary.json", "training_history.csv",
    "validation_classification_report.csv", "validation_confusion_matrix.csv",
    "validation_confusion_matrix.png", "validation_predictions.csv",
)


def candidate_definition(name: str) -> dict[str, float | str]:
    return next(candidate for candidate in CANDIDATES if candidate["name"] == name)


def reuse_seed42(source_root: Path, destination_root: Path, strategy: str) -> dict:
    source = source_root / strategy
    destination = destination_root / strategy
    summary = json.loads((source / "run_summary.json").read_text(encoding="utf-8"))
    candidate = candidate_definition(strategy)
    contract = {
        "candidate": strategy,
        "seed": 42,
        "weak_margin_lambda": candidate["weak_margin_lambda"],
        "weak_margin_value": candidate["weak_margin_value"],
        "test_evaluated": False,
    }
    if any(summary.get(key) != value for key, value in contract.items()):
        raise ValueError(f"Seed 42 source contract mismatch for {strategy}")
    if not all((source / name).is_file() for name in REQUIRED_RUN_FILES):
        raise FileNotFoundError(f"Seed 42 source is incomplete for {strategy}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_RUN_FILES:
        shutil.copy2(source / name, destination / name)
    summary["multiseed_reused"] = True
    (destination / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def paired_decision(
    deltas: pd.DataFrame,
    macro_tolerance: float = 0.005,
    none_tolerance: float = 0.002,
    balanced_tolerance: float = 0.005,
) -> dict[str, object]:
    weak_improved = int((deltas["delta_validation_weak_recall"] > 0).sum())
    checks = {
        "mean_weak_recall_improved": float(deltas["delta_validation_weak_recall"].mean()) > 0,
        "weak_recall_improved_at_least_two_seeds": weak_improved >= 2,
        "macro_f1_guardrail_all_seeds": bool(
            (deltas["delta_validation_macro_f1"] >= -macro_tolerance).all()
        ),
        "none_recall_guardrail_all_seeds": bool(
            (deltas["delta_validation_none_recall"] >= -none_tolerance).all()
        ),
        "balanced_accuracy_guardrail_all_seeds": bool(
            (deltas["delta_validation_balanced_accuracy"] >= -balanced_tolerance).all()
        ),
    }
    checks["eligible"] = all(checks.values())
    return {
        "selected_candidate": "weak_none_margin_005" if checks["eligible"] else "baseline",
        "weak_recall_improved_seed_count": weak_improved,
        "checks": checks,
    }


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
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        seed_root = args.output_dir / "runs" / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        for strategy in STRATEGIES:
            candidate = candidate_definition(strategy)
            cached = reusable_summary(seed_root, candidate, seed)
            if cached is not None:
                print(f"reuse_completed seed={seed} strategy={strategy}", flush=True)
                summary = cached
            elif seed == 42 and args.seed42_source is not None:
                summary = reuse_seed42(args.seed42_source, seed_root, strategy)
                print(f"reuse_seed42 strategy={strategy}", flush=True)
            else:
                run_args = SimpleNamespace(
                    maps=args.maps, output_dir=seed_root, seed=seed,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                    epochs=args.epochs, minimum_delta=args.minimum_delta,
                    patience=args.patience,
                )
                summary = train_candidate(
                    candidate, run_args, assignments, class_names, device
                )
            rows.append({"seed": seed, "strategy": strategy, **summary})

    run_table = pd.DataFrame(rows).sort_values(["seed", "strategy"]).reset_index(drop=True)
    if len(run_table) != len(args.seeds) * len(STRATEGIES):
        raise ValueError("Incomplete seed-strategy result grid")
    run_table.to_csv(args.output_dir / "seed_candidate_metrics.csv", index=False)

    summary_rows = []
    for strategy in STRATEGIES:
        subset = run_table.loc[run_table["strategy"].eq(strategy)]
        row: dict[str, object] = {"strategy": strategy, "seed_count": len(subset)}
        for metric in METRICS:
            values = subset[metric].astype(float)
            row.update({
                f"{metric}_mean": float(values.mean()),
                f"{metric}_std": float(values.std(ddof=1)),
                f"{metric}_min": float(values.min()),
                f"{metric}_max": float(values.max()),
            })
        summary_rows.append(row)
    metric_summary = pd.DataFrame(summary_rows)
    metric_summary.to_csv(args.output_dir / "metric_summary.csv", index=False)

    baseline = run_table.loc[run_table["strategy"].eq("baseline")].set_index("seed")
    margin = run_table.loc[run_table["strategy"].eq("weak_none_margin_005")].set_index("seed")
    delta_rows = []
    for seed in args.seeds:
        row: dict[str, object] = {"seed": seed}
        for metric in METRICS:
            row[f"baseline_{metric}"] = float(baseline.loc[seed, metric])
            row[f"margin_{metric}"] = float(margin.loc[seed, metric])
            row[f"delta_{metric}"] = float(margin.loc[seed, metric] - baseline.loc[seed, metric])
        delta_rows.append(row)
    paired = pd.DataFrame(delta_rows)
    paired.to_csv(args.output_dir / "paired_seed_deltas.csv", index=False)
    decision = paired_decision(paired)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(args.seeds))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        baseline.loc[list(args.seeds), "validation_weak_recall"],
        width, label="baseline",
    )
    axes[0].bar(
        x + width / 2,
        margin.loc[list(args.seeds), "validation_weak_recall"],
        width, label="margin 0.05",
    )
    axes[0].set_xticks(x, [str(seed) for seed in args.seeds])
    axes[0].set_ylim(0.75, 1.0)
    axes[0].set_title("Validation weak-class recall by seed")
    axes[0].legend()
    paired.plot.bar(
        x="seed",
        y=[
            "delta_validation_weak_recall", "delta_validation_macro_f1",
            "delta_validation_none_recall", "delta_validation_balanced_accuracy",
        ],
        ax=axes[1], rot=0,
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Paired margin - baseline deltas")
    figure.tight_layout()
    figure.savefig(args.output_dir / "multiseed_margin_dashboard.png", dpi=170)
    plt.close(figure)

    result = {
        "schema_version": 1,
        "purpose": "validation_only_paired_weak_margin_multiseed_reproducibility",
        "seeds": list(args.seeds),
        "strategies": list(STRATEGIES),
        "fixed_split": True,
        "test_evaluated": False,
        "used_test_for_selection": False,
        "deployment_changed": False,
        "primary_metric": "validation_weak_recall",
        "paired_guardrails": {
            "macro_f1_tolerance_per_seed": 0.005,
            "none_recall_tolerance_per_seed": 0.002,
            "balanced_accuracy_tolerance_per_seed": 0.005,
        },
        **decision,
        "mean_deltas": {
            metric: float(paired[f"delta_{metric}"].mean()) for metric in METRICS
        },
        "next_step": "keep_baseline_or_run_final_governed_evaluation_after_multiseed_gate",
    }
    (args.output_dir / "reproducibility_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# WM-811K weak margin 다중 seed 재현\n\n"
        "동일한 고정 split에서 baseline과 λ=0.05 margin 후보를 seed 17, 42, 2026으로 "
        "쌍 비교합니다. Seed 42의 검증된 결과는 재사용합니다. Test 추론은 수행하지 않습니다.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed42-source", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 2026])
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
    if args.seeds != (17, 42, 2026):
        raise ValueError("The registered reproducibility protocol requires seeds 17, 42, 2026")
    run(args)


if __name__ == "__main__":
    main()
