"""Choose one shared none-class logit bias from WM-811K validation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Numeric calibration remains usable in a minimal runtime.
    plt = None


SEEDS = (17, 42, 2026)
CANDIDATES = ("baseline", "robust_flat")
CLASS_NAMES = (
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
    "Near-full", "Random", "Scratch", "none",
)
PROBABILITY_COLUMNS = tuple(
    f"probability_{name.lower().replace('-', '_').replace(' ', '_')}"
    for name in CLASS_NAMES
)
WEAK_LABELS = (2, 4, 7)
NONE_LABEL = 8


def load_bundle(path: Path) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    expected = {
        f"seed_{seed}/{candidate}_validation_predictions.csv"
        for seed in SEEDS for candidate in CANDIDATES
    }
    loaded: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    indices: dict[tuple[int, str], np.ndarray] = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate entries")
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts or "\\" in name:
                raise ValueError(f"Unsafe ZIP path: {name}")
        if set(names) != expected:
            raise ValueError(
                f"Prediction bundle mismatch: missing={sorted(expected-set(names))}, "
                f"extra={sorted(set(names)-expected)}"
            )
        for seed in SEEDS:
            for candidate in CANDIDATES:
                name = f"seed_{seed}/{candidate}_validation_predictions.csv"
                with archive.open(name) as handle:
                    frame = pd.read_csv(handle)
                required = {"array_index", "label_id", *PROBABILITY_COLUMNS}
                missing = required.difference(frame.columns)
                if missing or frame["array_index"].duplicated().any():
                    raise ValueError(f"Invalid prediction CSV {name}: missing={sorted(missing)}")
                probabilities = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
                if (
                    not np.isfinite(probabilities).all()
                    or np.any(probabilities < 0)
                    or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
                ):
                    raise ValueError(f"Invalid probability simplex: {name}")
                indices[seed, candidate] = frame["array_index"].to_numpy(np.int64)
                loaded[seed, candidate] = (
                    frame["label_id"].to_numpy(np.int64), probabilities
                )
            if not np.array_equal(indices[seed, "baseline"], indices[seed, "robust_flat"]):
                raise ValueError(f"Prediction row alignment mismatch for seed {seed}")
            if not np.array_equal(loaded[seed, "baseline"][0], loaded[seed, "robust_flat"][0]):
                raise ValueError(f"Label alignment mismatch for seed {seed}")
    return loaded


def predict_with_none_bias(probabilities: np.ndarray, bias: float) -> np.ndarray:
    scores = probabilities.copy()
    scores[:, NONE_LABEL] *= np.exp(float(bias))
    return scores.argmax(axis=1)


def metric_bundle(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    recalls: list[float] = []
    f1_scores: list[float] = []
    supports: list[int] = []
    for label in range(len(CLASS_NAMES)):
        true_label = true == label
        predicted_label = predicted == label
        true_positive = int(np.sum(true_label & predicted_label))
        false_positive = int(np.sum(~true_label & predicted_label))
        support = int(np.sum(true_label))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
    return {
        "accuracy": float(np.mean(true == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "weighted_f1": float(np.average(f1_scores, weights=supports)),
        "weak_recall": float(np.mean([recalls[index] for index in WEAK_LABELS])),
        "none_recall": float(recalls[NONE_LABEL]),
    }


def evaluate_bias(
    loaded: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]], bias: float
) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows: list[dict[str, float | int]] = []
    for seed in SEEDS:
        true, baseline_probabilities = loaded[seed, "baseline"]
        _, robust_probabilities = loaded[seed, "robust_flat"]
        baseline = metric_bundle(true, baseline_probabilities.argmax(axis=1))
        robust = metric_bundle(
            true, predict_with_none_bias(robust_probabilities, bias)
        )
        row: dict[str, float | int] = {"seed": seed, "none_logit_bias": bias}
        for metric in baseline:
            row[f"baseline_{metric}"] = baseline[metric]
            row[f"robust_{metric}"] = robust[metric]
            row[f"delta_{metric}"] = robust[metric] - baseline[metric]
        rows.append(row)
    frame = pd.DataFrame(rows)
    checks = {
        "mean_weak_recall_improved": bool(frame["delta_weak_recall"].mean() > 0),
        "weak_recall_improved_at_least_two_seeds": bool((frame["delta_weak_recall"] > 0).sum() >= 2),
        "macro_f1_guardrail_all_seeds": bool((frame["delta_macro_f1"] >= -0.005).all()),
        "none_recall_guardrail_all_seeds": bool((frame["delta_none_recall"] >= -0.002).all()),
        "balanced_accuracy_guardrail_all_seeds": bool((frame["delta_balanced_accuracy"] >= -0.005).all()),
        "accuracy_guardrail_all_seeds": bool((frame["delta_accuracy"] >= -0.005).all()),
    }
    return frame, checks


def select_shared_bias(
    loaded: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]], grid: np.ndarray
) -> tuple[float | None, pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    sweep_rows = []
    details: dict[float, tuple[pd.DataFrame, dict[str, bool]]] = {}
    for raw_bias in grid:
        bias = float(np.round(raw_bias, 6))
        frame, checks = evaluate_bias(loaded, bias)
        row = {
            "none_logit_bias": bias,
            "eligible": all(checks.values()),
            "mean_delta_weak_recall": float(frame["delta_weak_recall"].mean()),
            "mean_delta_macro_f1": float(frame["delta_macro_f1"].mean()),
            "mean_delta_balanced_accuracy": float(frame["delta_balanced_accuracy"].mean()),
            "mean_delta_none_recall": float(frame["delta_none_recall"].mean()),
            "minimum_delta_none_recall": float(frame["delta_none_recall"].min()),
        }
        sweep_rows.append(row)
        details[bias] = frame, checks
    sweep = pd.DataFrame(sweep_rows)
    eligible = sweep.loc[sweep["eligible"]].sort_values(
        ["mean_delta_weak_recall", "mean_delta_macro_f1", "mean_delta_balanced_accuracy", "none_logit_bias"],
        ascending=[False, False, False, True],
    )
    if eligible.empty:
        frame, checks = details[0.0]
        return None, sweep, frame, checks
    selected = float(eligible.iloc[0]["none_logit_bias"])
    frame, checks = details[selected]
    return selected, sweep, frame, checks


def plot_sweep(sweep: pd.DataFrame, selected: float | None, path: Path) -> None:
    if plt is None:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(sweep["none_logit_bias"], sweep["mean_delta_weak_recall"], label="weak recall")
    axes[0].plot(sweep["none_logit_bias"], sweep["mean_delta_macro_f1"], label="macro-F1")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].legend()
    axes[0].set_title("Mean robust-flat minus baseline")
    axes[1].plot(sweep["none_logit_bias"], sweep["minimum_delta_none_recall"])
    axes[1].axhline(-0.002, color="red", linestyle="--", label="guardrail")
    axes[1].legend()
    axes[1].set_title("Worst-seed none recall delta")
    for axis in axes:
        axis.set_xlabel("Shared none logit bias")
        if selected is not None:
            axis.axvline(selected, color="green", linestyle=":")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    loaded = load_bundle(args.predictions_zip)
    selected, sweep, seed_metrics, checks = select_shared_bias(
        loaded, np.linspace(0.0, 2.0, 401)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.output_dir / "bias_sweep.csv", index=False)
    seed_metrics.to_csv(args.output_dir / "selected_seed_metrics.csv", index=False)
    plot_sweep(sweep, selected, args.output_dir / "bias_sweep.png")
    passed = selected is not None and all(checks.values())
    result = {
        "schema_version": 1,
        "experiment_type": "wm811k_robust_flat_shared_none_bias_validation",
        "seeds": list(SEEDS),
        "selection_split": "validation",
        "test_evaluated": False,
        "used_test_for_selection": False,
        "selected_none_logit_bias": selected,
        "eligible_bias_count": int(sweep["eligible"].sum()),
        "clean_validation_gate_passed": passed,
        "checks": checks,
        "mean_deltas": {
            metric: float(seed_metrics[f"delta_{metric}"].mean())
            for metric in (
                "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                "weak_recall", "none_recall",
            )
        },
        "deployment_model_changed": False,
        "next_step": (
            "replay_validation_shift4_and_dropout1_with_selected_bias"
            if passed else "keep_deployed_baseline"
        ),
    }
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
