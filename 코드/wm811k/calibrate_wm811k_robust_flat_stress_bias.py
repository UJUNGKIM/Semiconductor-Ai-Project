"""Select one shared WM-811K none-class bias across clean and stressed validation."""

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
    load_model,
    make_loader,
    metric_bundle,
    model_probabilities,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Numeric calibration remains usable without plotting.
    plt = None


SEEDS = (17, 42, 2026)
STRESS_MODES = ("clean", "shift4", "dropout1")
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "weak_recall",
    "none_recall",
)


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
    checks: dict[str, bool] = {}
    for mode in STRESS_MODES:
        subset = frame.loc[frame["stress_mode"].eq(mode)]
        checks[f"{mode}_weak_recall_mean_improved"] = bool(
            subset["delta_weak_recall"].mean() > 0
        )
        checks[f"{mode}_macro_f1_mean_improved"] = bool(
            subset["delta_macro_f1"].mean() > 0
        )
        checks[f"{mode}_none_recall_guardrail_all_seeds"] = bool(
            (subset["delta_none_recall"] >= -0.002).all()
        )
        checks[f"{mode}_balanced_accuracy_guardrail_all_seeds"] = bool(
            (subset["delta_balanced_accuracy"] >= -0.005).all()
        )
        checks[f"{mode}_accuracy_guardrail_all_seeds"] = bool(
            (subset["delta_accuracy"] >= -0.005).all()
        )
    return checks


def evaluate_bias(
    cached: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    baselines: dict[tuple[int, str], dict[str, float]],
    bias: float,
    class_names: list[str],
) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        clean_prediction: np.ndarray | None = None
        for mode in STRESS_MODES:
            true, probabilities = cached[seed, mode]
            robust, predicted = adjusted_metrics(
                true, probabilities, bias, class_names
            )
            baseline = baselines[seed, mode]
            row: dict[str, object] = {
                "seed": seed,
                "stress_mode": mode,
                "none_logit_bias": bias,
            }
            for metric in METRICS:
                row[f"baseline_{metric}"] = float(baseline[metric])
                row[f"robust_{metric}"] = float(robust[metric])
                row[f"delta_{metric}"] = float(robust[metric] - baseline[metric])
            if mode == "clean":
                clean_prediction = predicted
            else:
                if clean_prediction is None:
                    raise RuntimeError("clean predictions must be evaluated first")
                row["baseline_prediction_stability"] = float(
                    baseline["prediction_stability"]
                )
                row["robust_prediction_stability"] = float(
                    np.mean(predicted == clean_prediction)
                )
                row["delta_prediction_stability"] = float(
                    row["robust_prediction_stability"]
                    - row["baseline_prediction_stability"]
                )
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["seed", "stress_mode"])
    return frame, evaluate_guardrails(frame)


def choose_best_bias(sweep: pd.DataFrame) -> float | None:
    eligible = sweep.loc[sweep["eligible"]].sort_values(
        [
            "worst_mode_mean_delta_weak_recall",
            "mean_delta_weak_recall",
            "mean_delta_macro_f1",
            "minimum_delta_none_recall",
            "none_logit_bias",
        ],
        ascending=[False, False, False, False, True],
    )
    return None if eligible.empty else float(eligible.iloc[0]["none_logit_bias"])


def select_stress_aware_bias(
    cached: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    baselines: dict[tuple[int, str], dict[str, float]],
    class_names: list[str],
    grid: np.ndarray,
) -> tuple[float | None, pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    if len(grid) == 0:
        raise ValueError("Bias grid must contain at least one value")
    sweep_rows: list[dict[str, object]] = []
    details: dict[float, tuple[pd.DataFrame, dict[str, bool]]] = {}
    for raw_bias in grid:
        bias = float(np.round(raw_bias, 6))
        frame, checks = evaluate_bias(
            cached, baselines, bias, class_names
        )
        mode_weak = frame.groupby("stress_mode")["delta_weak_recall"].mean()
        sweep_rows.append(
            {
                "none_logit_bias": bias,
                "eligible": all(checks.values()),
                "worst_mode_mean_delta_weak_recall": float(mode_weak.min()),
                "mean_delta_weak_recall": float(frame["delta_weak_recall"].mean()),
                "mean_delta_macro_f1": float(frame["delta_macro_f1"].mean()),
                "mean_delta_balanced_accuracy": float(
                    frame["delta_balanced_accuracy"].mean()
                ),
                "mean_delta_none_recall": float(frame["delta_none_recall"].mean()),
                "minimum_delta_none_recall": float(
                    frame["delta_none_recall"].min()
                ),
            }
        )
        details[bias] = frame, checks
    sweep = pd.DataFrame(sweep_rows)
    selected = choose_best_bias(sweep)
    fallback = float(np.round(grid[0], 6)) if selected is None else selected
    frame, checks = details[fallback]
    return selected, sweep, frame, checks


def plot_sweep(sweep: pd.DataFrame, selected: float | None, path: Path) -> None:
    if plt is None:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(
        sweep["none_logit_bias"],
        sweep["worst_mode_mean_delta_weak_recall"],
        label="worst-mode weak recall",
    )
    axes[0].plot(
        sweep["none_logit_bias"],
        sweep["mean_delta_macro_f1"],
        label="all-mode macro-F1",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].legend()
    axes[1].plot(
        sweep["none_logit_bias"], sweep["minimum_delta_none_recall"]
    )
    axes[1].axhline(-0.002, color="red", linestyle="--", label="guardrail")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Shared none logit bias")
        if selected is not None:
            axis.axvline(selected, color="green", linestyle=":")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    maps, labels, assignments = validate_inputs(
        args.maps, args.labels, args.assignments
    )
    for reference in (maps, labels):
        if (
            isinstance(reference, np.memmap)
            and getattr(reference, "_mmap", None) is not None
        ):
            reference._mmap.close()
    class_names = (
        assignments[["label_id", "failure_type"]]
        .drop_duplicates()
        .sort_values("label_id")["failure_type"]
        .tolist()
    )
    if len(class_names) != 9 or class_names[NONE_LABEL] != "none":
        raise ValueError(f"Unexpected class order: {class_names}")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"device={device}", flush=True)
    cached: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    baselines: dict[tuple[int, str], dict[str, float]] = {}
    for seed in SEEDS:
        seed_dir = args.experiment_root / "runs" / f"seed_{seed}"
        summary_path = seed_dir / "baseline" / "candidate_summary.json"
        checkpoint_path = seed_dir / "robust_flat" / "flat" / "best_model.pt"
        if not summary_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing seed {seed} baseline summary or robust checkpoint"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model = load_model(checkpoint_path, device)
        for mode in STRESS_MODES:
            loader = make_loader(
                args.maps,
                assignments,
                "validation",
                "flat",
                args.batch_size,
                args.num_workers,
                seed,
                device,
                stress_mode=mode,
            )
            cached[seed, mode] = model_probabilities(model, loader, device)
            baselines[seed, mode] = summary[mode]
            del loader
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected, sweep, seed_metrics, checks = select_stress_aware_bias(
        cached,
        baselines,
        class_names,
        np.linspace(args.bias_min, args.bias_max, args.bias_steps),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output_dir / "stress_bias_sweep.csv", index=False)
    seed_metrics.to_csv(
        args.output_dir / "selected_stress_seed_metrics.csv", index=False
    )
    plot_sweep(sweep, selected, args.output_dir / "stress_bias_sweep.png")
    passed = selected is not None and all(checks.values())
    mean_deltas: dict[str, dict[str, float]] = {}
    for mode in STRESS_MODES:
        subset = seed_metrics.loc[seed_metrics["stress_mode"].eq(mode)]
        mean_deltas[mode] = {
            metric: float(subset[f"delta_{metric}"].mean())
            for metric in METRICS
        }
        if mode != "clean":
            mean_deltas[mode]["prediction_stability"] = float(
                subset["delta_prediction_stability"].mean()
            )
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment_type": "wm811k_robust_flat_joint_stress_bias_validation",
        "seeds": list(SEEDS),
        "selection_split": "validation",
        "stress_modes": list(STRESS_MODES),
        "test_evaluated": False,
        "used_test_for_selection": False,
        "selected_none_logit_bias": selected,
        "eligible_bias_count": int(sweep["eligible"].sum()),
        "stress_calibration_gate_passed": passed,
        "checks": checks,
        "mean_deltas": mean_deltas,
        "deployment_model_changed": False,
        "next_step": (
            "run_one_locked_seed42_test_evaluation"
            if passed
            else "keep_deployed_baseline"
        ),
    }
    (args.output_dir / "stress_bias_calibration_summary.json").write_text(
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
    parser.add_argument("--bias-min", type=float, default=0.0)
    parser.add_argument("--bias-max", type=float, default=2.0)
    parser.add_argument("--bias-steps", type=int, default=401)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
