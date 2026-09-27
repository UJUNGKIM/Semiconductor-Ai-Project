"""Replay validation stress tests for the bias-calibrated WM-811K robust-flat model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_wm811k_cnn import validate_inputs
from train_wm811k_robust_hierarchical_candidates import (
    NONE_LABEL,
    make_loader,
    load_model,
    metric_bundle,
    model_probabilities,
)


SEEDS = (17, 42, 2026)
STRESS_MODES = ("clean", "shift4", "dropout1")


def adjusted_metrics(
    true: np.ndarray,
    probabilities: np.ndarray,
    bias: float,
    class_names: list[str],
) -> tuple[dict[str, float], np.ndarray]:
    adjusted = probabilities.copy()
    adjusted[:, NONE_LABEL] *= np.exp(float(bias))
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    predicted = adjusted.argmax(axis=1)
    return metric_bundle(true, adjusted, class_names), predicted


def evaluate_guardrails(frame: pd.DataFrame) -> dict[str, bool]:
    clean = frame.loc[frame["stress_mode"].eq("clean")]
    shift = frame.loc[frame["stress_mode"].eq("shift4")]
    dropout = frame.loc[frame["stress_mode"].eq("dropout1")]
    return {
        "clean_weak_recall_mean_improved": bool(clean["delta_weak_recall"].mean() > 0),
        "clean_macro_f1_guardrail_all_seeds": bool((clean["delta_macro_f1"] >= -0.005).all()),
        "clean_none_recall_guardrail_all_seeds": bool((clean["delta_none_recall"] >= -0.002).all()),
        "clean_balanced_accuracy_guardrail_all_seeds": bool((clean["delta_balanced_accuracy"] >= -0.005).all()),
        "shift_macro_f1_mean_improved": bool(shift["delta_macro_f1"].mean() > 0),
        "shift_none_recall_guardrail_all_seeds": bool((shift["delta_none_recall"] >= -0.002).all()),
        "shift_balanced_accuracy_guardrail_all_seeds": bool((shift["delta_balanced_accuracy"] >= -0.005).all()),
        "dropout_macro_f1_mean_improved": bool(dropout["delta_macro_f1"].mean() > 0),
        "dropout_none_recall_guardrail_all_seeds": bool((dropout["delta_none_recall"] >= -0.002).all()),
        "dropout_balanced_accuracy_guardrail_all_seeds": bool((dropout["delta_balanced_accuracy"] >= -0.005).all()),
    }


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
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        baseline_dir = args.experiment_root / "runs" / f"seed_{seed}" / "baseline"
        robust_dir = args.experiment_root / "runs" / f"seed_{seed}" / "robust_flat"
        baseline_summary_path = baseline_dir / "candidate_summary.json"
        checkpoint_path = robust_dir / "flat" / "best_model.pt"
        if not baseline_summary_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing seed {seed} baseline summary or robust checkpoint"
            )
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
        model = load_model(checkpoint_path, device)
        clean_predictions: np.ndarray | None = None
        for stress_mode in STRESS_MODES:
            loader = make_loader(
                args.maps, assignments, "validation", "flat", args.batch_size,
                args.num_workers, seed, device, stress_mode=stress_mode,
            )
            true, probabilities = model_probabilities(model, loader, device)
            robust_metrics, predicted = adjusted_metrics(
                true, probabilities, args.none_logit_bias, class_names
            )
            if stress_mode == "clean":
                clean_predictions = predicted
            else:
                robust_metrics["prediction_stability"] = float(
                    np.mean(predicted == clean_predictions)
                )
            baseline_metrics = baseline_summary[stress_mode]
            row: dict[str, object] = {
                "seed": seed,
                "stress_mode": stress_mode,
                "none_logit_bias": args.none_logit_bias,
            }
            for metric in (
                "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                "weak_recall", "none_recall",
            ):
                row[f"baseline_{metric}"] = float(baseline_metrics[metric])
                row[f"robust_{metric}"] = float(robust_metrics[metric])
                row[f"delta_{metric}"] = float(robust_metrics[metric] - baseline_metrics[metric])
            if stress_mode != "clean":
                row["baseline_prediction_stability"] = float(
                    baseline_metrics["prediction_stability"]
                )
                row["robust_prediction_stability"] = float(
                    robust_metrics["prediction_stability"]
                )
                row["delta_prediction_stability"] = float(
                    robust_metrics["prediction_stability"]
                    - baseline_metrics["prediction_stability"]
                )
            rows.append(row)
            del loader
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows).sort_values(["seed", "stress_mode"])
    checks = evaluate_guardrails(frame)
    passed = all(checks.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "stress_replay_metrics.csv", index=False)
    mean_deltas = {}
    for mode in STRESS_MODES:
        subset = frame.loc[frame["stress_mode"].eq(mode)]
        mean_deltas[mode] = {
            metric: float(subset[f"delta_{metric}"].mean())
            for metric in (
                "accuracy", "balanced_accuracy", "macro_f1", "weak_recall", "none_recall"
            )
        }
        if mode != "clean":
            mean_deltas[mode]["prediction_stability"] = float(
                subset["delta_prediction_stability"].mean()
            )
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment_type": "wm811k_robust_flat_bias_stress_replay",
        "seeds": list(SEEDS),
        "selection_split": "validation",
        "test_evaluated": False,
        "used_test_for_selection": False,
        "selected_none_logit_bias": args.none_logit_bias,
        "stress_modes": list(STRESS_MODES),
        "stress_gate_passed": passed,
        "checks": checks,
        "mean_deltas": mean_deltas,
        "deployment_model_changed": False,
        "next_step": (
            "run_one_locked_seed42_test_evaluation"
            if passed else "keep_deployed_baseline"
        ),
    }
    (args.output_dir / "stress_replay_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--none-logit-bias", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
