"""Select a baseline/robust WM-811K ensemble using validation stress conditions."""

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
    WEAK_LABELS,
    load_model,
    make_loader,
    model_probabilities,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Numeric selection remains usable without plotting.
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


def metrics_from_predictions(
    true: np.ndarray, predicted: np.ndarray, class_count: int = 9
) -> dict[str, float]:
    confusion = np.bincount(
        class_count * true.astype(np.int64) + predicted.astype(np.int64),
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(class_count, dtype=np.float64),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(class_count, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(class_count, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / support.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "weak_recall": float(np.mean(recall[list(WEAK_LABELS)])),
        "none_recall": float(recall[NONE_LABEL]),
    }


def ensemble_predictions(
    baseline_probabilities: np.ndarray,
    robust_probabilities: np.ndarray,
    robust_weight: float,
    none_logit_bias: float,
) -> np.ndarray:
    if not 0.0 <= robust_weight <= 1.0:
        raise ValueError("robust_weight must be between 0 and 1")
    scores = (
        (1.0 - robust_weight) * baseline_probabilities
        + robust_weight * robust_probabilities
    )
    scores = scores.copy()
    scores[:, NONE_LABEL] *= np.exp(float(none_logit_bias))
    return scores.argmax(axis=1)


def guardrail_status(frame: pd.DataFrame) -> tuple[dict[str, bool], float]:
    checks: dict[str, bool] = {}
    margins: list[float] = []
    for mode in STRESS_MODES:
        subset = frame.loc[frame["stress_mode"].eq(mode)]
        weak_margin = float(subset["delta_weak_recall"].mean())
        macro_margin = float(subset["delta_macro_f1"].mean())
        none_margin = float(subset["delta_none_recall"].min() + 0.002)
        balanced_margin = float(
            subset["delta_balanced_accuracy"].min() + 0.005
        )
        accuracy_margin = float(subset["delta_accuracy"].min() + 0.005)
        checks[f"{mode}_weak_recall_mean_improved"] = weak_margin > 0
        checks[f"{mode}_macro_f1_mean_improved"] = macro_margin > 0
        checks[f"{mode}_none_recall_guardrail_all_seeds"] = none_margin >= 0
        checks[f"{mode}_balanced_accuracy_guardrail_all_seeds"] = (
            balanced_margin >= 0
        )
        checks[f"{mode}_accuracy_guardrail_all_seeds"] = accuracy_margin >= 0
        margins.extend(
            [
                weak_margin,
                macro_margin,
                none_margin,
                balanced_margin,
                accuracy_margin,
            ]
        )
    return checks, min(margins)


def evaluate_configuration(
    cached: dict[
        tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]
    ],
    robust_weight: float,
    none_logit_bias: float,
) -> tuple[pd.DataFrame, dict[str, bool], float]:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        clean_baseline: np.ndarray | None = None
        clean_ensemble: np.ndarray | None = None
        for mode in STRESS_MODES:
            true, baseline_probabilities, robust_probabilities = cached[seed, mode]
            baseline_prediction = baseline_probabilities.argmax(axis=1)
            ensemble_prediction = ensemble_predictions(
                baseline_probabilities,
                robust_probabilities,
                robust_weight,
                none_logit_bias,
            )
            baseline = metrics_from_predictions(true, baseline_prediction)
            ensemble = metrics_from_predictions(true, ensemble_prediction)
            row: dict[str, object] = {
                "seed": seed,
                "stress_mode": mode,
                "robust_weight": robust_weight,
                "none_logit_bias": none_logit_bias,
            }
            for metric in METRICS:
                row[f"baseline_{metric}"] = baseline[metric]
                row[f"ensemble_{metric}"] = ensemble[metric]
                row[f"delta_{metric}"] = ensemble[metric] - baseline[metric]
            if mode == "clean":
                clean_baseline = baseline_prediction
                clean_ensemble = ensemble_prediction
            else:
                if clean_baseline is None or clean_ensemble is None:
                    raise RuntimeError("clean predictions must be evaluated first")
                row["baseline_prediction_stability"] = float(
                    np.mean(baseline_prediction == clean_baseline)
                )
                row["ensemble_prediction_stability"] = float(
                    np.mean(ensemble_prediction == clean_ensemble)
                )
                row["delta_prediction_stability"] = float(
                    row["ensemble_prediction_stability"]
                    - row["baseline_prediction_stability"]
                )
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["seed", "stress_mode"])
    checks, minimum_margin = guardrail_status(frame)
    return frame, checks, minimum_margin


def sweep_row(
    frame: pd.DataFrame,
    checks: dict[str, bool],
    minimum_margin: float,
    robust_weight: float,
    none_logit_bias: float,
) -> dict[str, object]:
    mode_weak = frame.groupby("stress_mode")["delta_weak_recall"].mean()
    row: dict[str, object] = {
        "robust_weight": robust_weight,
        "none_logit_bias": none_logit_bias,
        "eligible": all(checks.values()),
        "minimum_guardrail_margin": minimum_margin,
        "worst_mode_mean_delta_weak_recall": float(mode_weak.min()),
        "mean_delta_weak_recall": float(frame["delta_weak_recall"].mean()),
        "mean_delta_macro_f1": float(frame["delta_macro_f1"].mean()),
        "mean_delta_balanced_accuracy": float(
            frame["delta_balanced_accuracy"].mean()
        ),
        "mean_delta_none_recall": float(frame["delta_none_recall"].mean()),
        "minimum_delta_none_recall": float(frame["delta_none_recall"].min()),
    }
    row.update(checks)
    return row


def select_best_configuration(sweep: pd.DataFrame) -> tuple[float, float] | None:
    eligible = sweep.loc[sweep["eligible"]].sort_values(
        [
            "worst_mode_mean_delta_weak_recall",
            "mean_delta_weak_recall",
            "mean_delta_macro_f1",
            "minimum_delta_none_recall",
            "robust_weight",
            "none_logit_bias",
        ],
        ascending=[False, False, False, False, True, True],
    )
    if eligible.empty:
        return None
    best = eligible.iloc[0]
    return float(best["robust_weight"]), float(best["none_logit_bias"])


def configuration_grid(
    alpha_center: float | None = None,
    bias_center: float | None = None,
) -> list[tuple[float, float]]:
    if alpha_center is None or bias_center is None:
        alphas = np.linspace(0.0, 1.0, 21)
        biases = np.linspace(0.0, 2.0, 41)
    else:
        alphas = np.arange(
            max(0.0, alpha_center - 0.05),
            min(1.0, alpha_center + 0.05) + 0.0001,
            0.01,
        )
        biases = np.arange(
            max(0.0, bias_center - 0.05),
            min(2.0, bias_center + 0.05) + 0.0001,
            0.01,
        )
    return [
        (float(np.round(alpha, 6)), float(np.round(bias, 6)))
        for alpha in alphas
        for bias in biases
    ]


def search_configurations(
    cached: dict[
        tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]
    ]
) -> tuple[tuple[float, float] | None, pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    evaluated: dict[
        tuple[float, float], tuple[pd.DataFrame, dict[str, bool], float]
    ] = {}

    def evaluate_many(configurations: list[tuple[float, float]]) -> None:
        for configuration in configurations:
            if configuration in evaluated:
                continue
            alpha, bias = configuration
            evaluated[configuration] = evaluate_configuration(
                cached, alpha, bias
            )

    evaluate_many(configuration_grid())
    coarse_rows = [
        sweep_row(frame, checks, margin, alpha, bias)
        for (alpha, bias), (frame, checks, margin) in evaluated.items()
    ]
    coarse = pd.DataFrame(coarse_rows).sort_values(
        ["minimum_guardrail_margin", "worst_mode_mean_delta_weak_recall"],
        ascending=[False, False],
    )
    for candidate in coarse.head(3).itertuples(index=False):
        evaluate_many(
            configuration_grid(candidate.robust_weight, candidate.none_logit_bias)
        )
    sweep = pd.DataFrame(
        [
            sweep_row(frame, checks, margin, alpha, bias)
            for (alpha, bias), (frame, checks, margin) in evaluated.items()
        ]
    ).sort_values(["robust_weight", "none_logit_bias"])
    selected = select_best_configuration(sweep)
    if selected is None:
        fallback = sweep.sort_values(
            ["minimum_guardrail_margin", "worst_mode_mean_delta_weak_recall"],
            ascending=[False, False],
        ).iloc[0]
        key = (float(fallback["robust_weight"]), float(fallback["none_logit_bias"]))
    else:
        key = selected
    frame, checks, _ = evaluated[key]
    return selected, sweep, frame, checks


def plot_sweep(
    sweep: pd.DataFrame,
    selected: tuple[float, float] | None,
    output_path: Path,
) -> None:
    if plt is None:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    scatter = axes[0].scatter(
        sweep["robust_weight"],
        sweep["none_logit_bias"],
        c=sweep["minimum_guardrail_margin"],
        cmap="coolwarm",
        s=18,
    )
    figure.colorbar(scatter, ax=axes[0], label="minimum guardrail margin")
    axes[1].scatter(
        sweep["robust_weight"],
        sweep["worst_mode_mean_delta_weak_recall"],
        c=sweep["eligible"].map({True: "green", False: "gray"}),
        s=18,
    )
    if selected is not None:
        alpha, bias = selected
        axes[0].scatter([alpha], [bias], marker="*", s=180, color="gold")
        chosen = sweep.loc[
            sweep["robust_weight"].eq(alpha)
            & sweep["none_logit_bias"].eq(bias)
        ].iloc[0]
        axes[1].scatter(
            [alpha],
            [chosen["worst_mode_mean_delta_weak_recall"]],
            marker="*",
            s=180,
            color="gold",
        )
    axes[0].set(xlabel="robust weight", ylabel="none logit bias")
    axes[1].set(
        xlabel="robust weight",
        ylabel="worst-mode mean weak-recall delta",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def load_validation_cache(
    args: argparse.Namespace,
    assignments: pd.DataFrame,
    device: torch.device,
) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cached: dict[
        tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for seed in SEEDS:
        seed_dir = args.experiment_root / "runs" / f"seed_{seed}"
        checkpoints = {
            "baseline": seed_dir / "baseline" / "flat" / "best_model.pt",
            "robust": seed_dir / "robust_flat" / "flat" / "best_model.pt",
        }
        seed_outputs: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for candidate, checkpoint in checkpoints.items():
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            model = load_model(checkpoint, device)
            seed_outputs[candidate] = {}
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
                seed_outputs[candidate][mode] = model_probabilities(
                    model, loader, device
                )
                del loader
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for mode in STRESS_MODES:
            baseline_true, baseline_probabilities = seed_outputs["baseline"][mode]
            robust_true, robust_probabilities = seed_outputs["robust"][mode]
            if not np.array_equal(baseline_true, robust_true):
                raise ValueError(f"Prediction alignment mismatch: seed={seed}, mode={mode}")
            cached[seed, mode] = (
                baseline_true,
                baseline_probabilities,
                robust_probabilities,
            )
    return cached


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
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"device={device}", flush=True)
    cached = load_validation_cache(args, assignments, device)
    selected, sweep, seed_metrics, checks = search_configurations(cached)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output_dir / "ensemble_sweep.csv", index=False)
    seed_metrics.to_csv(
        args.output_dir / "selected_ensemble_seed_metrics.csv", index=False
    )
    plot_sweep(sweep, selected, args.output_dir / "ensemble_sweep.png")
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
        "experiment_type": "wm811k_baseline_robust_validation_stress_ensemble",
        "seeds": list(SEEDS),
        "selection_split": "validation",
        "stress_modes": list(STRESS_MODES),
        "test_evaluated": False,
        "used_test_for_selection": False,
        "search_strategy": "coarse_0.05_then_top3_local_0.01",
        "evaluated_configuration_count": int(len(sweep)),
        "eligible_configuration_count": int(sweep["eligible"].sum()),
        "selected_robust_weight": None if selected is None else selected[0],
        "selected_none_logit_bias": None if selected is None else selected[1],
        "ensemble_validation_gate_passed": passed,
        "checks": checks,
        "mean_deltas": mean_deltas,
        "deployment_model_changed": False,
        "next_step": (
            "run_one_locked_seed42_test_evaluation"
            if passed
            else "keep_deployed_baseline"
        ),
    }
    (args.output_dir / "ensemble_selection_summary.json").write_text(
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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
