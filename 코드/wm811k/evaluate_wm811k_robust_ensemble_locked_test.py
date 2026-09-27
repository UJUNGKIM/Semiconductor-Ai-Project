"""Run the one-time locked WM-811K test for the validation-selected ensemble."""

from __future__ import annotations

import argparse
import hashlib
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


LOCKED_SEED = 42
LOCKED_ROBUST_WEIGHT = 0.5
LOCKED_NONE_LOGIT_BIAS = 0.1
EXPECTED_DEPLOYED_SHA256 = (
    "77ea8394f593ccd0e1ffa85ce1b8e59c52fcd44b109d7e3587c66f7290d4557b"
)
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "weak_recall",
    "none_recall",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confusion_matrix(
    true: np.ndarray, predicted: np.ndarray, class_count: int = 9
) -> np.ndarray:
    return np.bincount(
        class_count * true.astype(np.int64) + predicted.astype(np.int64),
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(len(confusion), dtype=np.float64),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(len(confusion), dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(confusion), dtype=np.float64),
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


def class_metric_rows(
    confusion: np.ndarray,
    class_names: list[str],
    model_name: str,
) -> list[dict[str, object]]:
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(len(confusion), dtype=np.float64),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(len(confusion), dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(confusion), dtype=np.float64),
        where=(precision + recall) > 0,
    )
    return [
        {
            "model": model_name,
            "label_id": index,
            "class_name": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    ]


def ensemble_predictions(
    baseline_probabilities: np.ndarray,
    robust_probabilities: np.ndarray,
) -> np.ndarray:
    scores = (
        (1.0 - LOCKED_ROBUST_WEIGHT) * baseline_probabilities
        + LOCKED_ROBUST_WEIGHT * robust_probabilities
    )
    scores = scores.copy()
    scores[:, NONE_LABEL] *= np.exp(LOCKED_NONE_LOGIT_BIAS)
    return scores.argmax(axis=1)


def deployment_gate(
    deployed: dict[str, float], candidate: dict[str, float]
) -> tuple[dict[str, bool], dict[str, float]]:
    deltas = {
        metric: float(candidate[metric] - deployed[metric])
        for metric in METRICS
    }
    checks = {
        "weak_recall_improved": deltas["weak_recall"] > 0,
        "macro_f1_improved": deltas["macro_f1"] > 0,
        "balanced_accuracy_improved": deltas["balanced_accuracy"] > 0,
        "accuracy_guardrail": deltas["accuracy"] >= -0.005,
        "weighted_f1_guardrail": deltas["weighted_f1"] >= -0.005,
        "none_recall_guardrail": deltas["none_recall"] >= -0.002,
    }
    return checks, deltas


def infer_checkpoint(
    checkpoint: Path,
    maps_path: Path,
    assignments: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = load_model(checkpoint, device)
    loader = make_loader(
        maps_path,
        assignments,
        "test",
        "flat",
        batch_size,
        workers,
        LOCKED_SEED,
        device,
        stress_mode="clean",
    )
    return model_probabilities(model, loader, device)


def existing_result(output_dir: Path) -> dict[str, object] | None:
    summary_path = output_dir / "locked_test_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "selected_robust_weight": LOCKED_ROBUST_WEIGHT,
        "selected_none_logit_bias": LOCKED_NONE_LOGIT_BIAS,
        "locked_seed": LOCKED_SEED,
        "test_evaluated": True,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"Existing locked result has unexpected {key}")
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = existing_result(args.output_dir)
    if previous is not None:
        print("Existing locked test result found; inference will not be repeated.")
        print(json.dumps(previous, ensure_ascii=False, indent=2))
        return previous

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

    experiment_baseline = (
        args.experiment_root
        / "runs"
        / f"seed_{LOCKED_SEED}"
        / "baseline"
        / "flat"
        / "best_model.pt"
    )
    robust_checkpoint = (
        args.experiment_root
        / "runs"
        / f"seed_{LOCKED_SEED}"
        / "robust_flat"
        / "flat"
        / "best_model.pt"
    )
    for path in (
        args.deployed_checkpoint,
        experiment_baseline,
        robust_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
    deployed_sha256 = sha256_file(args.deployed_checkpoint)
    if deployed_sha256 != EXPECTED_DEPLOYED_SHA256:
        raise ValueError(
            "Uploaded deployed checkpoint does not match the registered model: "
            f"{deployed_sha256}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"device={device}", flush=True)
    deployed_true, deployed_probabilities = infer_checkpoint(
        args.deployed_checkpoint,
        args.maps,
        assignments,
        device,
        args.batch_size,
        args.num_workers,
    )
    baseline_true, baseline_probabilities = infer_checkpoint(
        experiment_baseline,
        args.maps,
        assignments,
        device,
        args.batch_size,
        args.num_workers,
    )
    robust_true, robust_probabilities = infer_checkpoint(
        robust_checkpoint,
        args.maps,
        assignments,
        device,
        args.batch_size,
        args.num_workers,
    )
    if not (
        np.array_equal(deployed_true, baseline_true)
        and np.array_equal(deployed_true, robust_true)
    ):
        raise ValueError("Test prediction rows are not aligned")

    deployed_prediction = deployed_probabilities.argmax(axis=1)
    candidate_prediction = ensemble_predictions(
        baseline_probabilities, robust_probabilities
    )
    deployed_confusion = confusion_matrix(deployed_true, deployed_prediction)
    candidate_confusion = confusion_matrix(deployed_true, candidate_prediction)
    deployed_metrics = metrics_from_confusion(deployed_confusion)
    candidate_metrics = metrics_from_confusion(candidate_confusion)
    checks, deltas = deployment_gate(deployed_metrics, candidate_metrics)
    passed = all(checks.values())

    metrics_frame = pd.DataFrame(
        [
            {"model": "deployed_baseline", **deployed_metrics},
            {"model": "locked_ensemble_candidate", **candidate_metrics},
        ]
    )
    metrics_frame.to_csv(args.output_dir / "locked_test_metrics.csv", index=False)
    class_rows = class_metric_rows(
        deployed_confusion, class_names, "deployed_baseline"
    ) + class_metric_rows(
        candidate_confusion, class_names, "locked_ensemble_candidate"
    )
    pd.DataFrame(class_rows).to_csv(
        args.output_dir / "locked_test_class_metrics.csv", index=False
    )
    confusion_rows = []
    for model_name, matrix in (
        ("deployed_baseline", deployed_confusion),
        ("locked_ensemble_candidate", candidate_confusion),
    ):
        for true_index, true_name in enumerate(class_names):
            for predicted_index, predicted_name in enumerate(class_names):
                confusion_rows.append(
                    {
                        "model": model_name,
                        "true_label_id": true_index,
                        "true_class": true_name,
                        "predicted_label_id": predicted_index,
                        "predicted_class": predicted_name,
                        "count": int(matrix[true_index, predicted_index]),
                    }
                )
    pd.DataFrame(confusion_rows).to_csv(
        args.output_dir / "locked_test_confusion_long.csv", index=False
    )

    result: dict[str, object] = {
        "schema_version": 1,
        "experiment_type": "wm811k_locked_ensemble_test",
        "selection_split": "validation",
        "evaluation_split": "test",
        "configuration_locked_from_validation": True,
        "test_evaluated": True,
        "used_test_for_selection": False,
        "locked_seed": LOCKED_SEED,
        "selected_robust_weight": LOCKED_ROBUST_WEIGHT,
        "selected_none_logit_bias": LOCKED_NONE_LOGIT_BIAS,
        "test_sample_count": int(len(deployed_true)),
        "checkpoint_sha256": {
            "deployed_baseline": deployed_sha256,
            "experiment_baseline_seed42": sha256_file(experiment_baseline),
            "robust_flat_seed42": sha256_file(robust_checkpoint),
        },
        "split_assignments_sha256": sha256_file(args.assignments),
        "deployed_metrics": deployed_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_minus_deployed": deltas,
        "checks": checks,
        "locked_test_gate_passed": passed,
        "deployment_model_changed": False,
        "next_step": (
            "calibrate_ensemble_uncertainty_and_ood_before_deployment"
            if passed
            else "keep_deployed_baseline"
        ),
    }
    (args.output_dir / "locked_test_summary.json").write_text(
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
    parser.add_argument("--deployed-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
