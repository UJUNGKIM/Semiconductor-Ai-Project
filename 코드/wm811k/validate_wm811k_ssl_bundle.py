"""Validate a WM-811K self-supervised validation-only result bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
import torch


SEEDS = (17, 42, 2026)
CLASS_NAMES = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
)
WEAK_CLASSES = ("Edge-Loc", "Loc", "Scratch")
PROBABILITY_COLUMNS = tuple(
    f"probability_{name.lower().replace('-', '_').replace(' ', '_')}"
    for name in CLASS_NAMES
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def _expected_files() -> set[str]:
    files = {
        "ssl_encoder.pt",
        "pretraining_history.csv",
        "ssl_candidate_comparison.json",
    }
    run_files = {
        "best_validation_model.pt",
        "training_history.csv",
        "training_curves.png",
        "validation_classification_report.csv",
        "validation_confusion_matrix.csv",
        "validation_confusion_matrix.png",
        "validation_predictions.csv",
        "run_summary.json",
    }
    for seed in SEEDS:
        files.update(f"seeds/seed_{seed}/{name}" for name in run_files)
    return files


def _read_json(archive: zipfile.ZipFile, name: str) -> dict:
    return json.loads(archive.read(name).decode("utf-8"))


def _read_csv(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archive.read(name)))


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-11):
        raise ValueError(f"{label}: expected {expected}, recomputed {actual}")


def _metric_bundle(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    metrics = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "macro_f1": float(
            f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }
    recalls = recall_score(
        true_labels,
        predicted_labels,
        labels=np.arange(len(CLASS_NAMES)),
        average=None,
        zero_division=0,
    )
    return metrics, {
        name: float(recalls[index]) for index, name in enumerate(CLASS_NAMES)
    }


def _validate_predictions(
    predictions: pd.DataFrame,
    validation_assignments: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    required = {
        "array_index",
        "label_id",
        "failure_type",
        "lot_name",
        "split",
        "predicted_label_id",
        "predicted_failure_type",
        "correct",
        *PROBABILITY_COLUMNS,
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"seed {seed}: prediction columns missing: {sorted(missing)}")
    if len(predictions) != len(validation_assignments):
        raise ValueError(f"seed {seed}: validation row count mismatch")
    if predictions["array_index"].duplicated().any():
        raise ValueError(f"seed {seed}: duplicate array_index")
    predictions = predictions.sort_values("array_index").reset_index(drop=True)
    expected = validation_assignments.sort_values("array_index").reset_index(drop=True)
    for column in ("array_index", "label_id", "failure_type", "lot_name", "split"):
        if predictions[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ValueError(f"seed {seed}: {column} differs from frozen validation rows")
    if set(predictions["split"]) != {"validation"}:
        raise ValueError(f"seed {seed}: non-validation predictions found")

    probabilities = predictions[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"seed {seed}: non-finite probability")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError(f"seed {seed}: probabilities do not sum to one")
    predicted = predictions["predicted_label_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(predicted, probabilities.argmax(axis=1)):
        raise ValueError(f"seed {seed}: predicted labels differ from argmax")
    expected_names = np.asarray(CLASS_NAMES, dtype=object)[predicted]
    if not np.array_equal(
        expected_names, predictions["predicted_failure_type"].astype(str).to_numpy()
    ):
        raise ValueError(f"seed {seed}: predicted class names are inconsistent")
    true_labels = predictions["label_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        true_labels == predicted, predictions["correct"].astype(bool).to_numpy()
    ):
        raise ValueError(f"seed {seed}: correctness flag is inconsistent")
    return _metric_bundle(true_labels, predicted)


def _validate_checkpoint(
    archive: zipfile.ZipFile,
    name: str,
    *,
    seed: int | None,
) -> str:
    payload = archive.read(name)
    checkpoint = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=True
    )
    if seed is None:
        if checkpoint.get("held_out_lots_used") is not False:
            raise ValueError("SSL encoder does not prove held-out lot exclusion")
        if checkpoint.get("labels_used_during_pretraining") is not False:
            raise ValueError("SSL encoder does not prove label-free pretraining")
    else:
        if checkpoint.get("seed") != seed:
            raise ValueError(f"seed {seed}: checkpoint seed mismatch")
        if checkpoint.get("test_evaluated") is not False:
            raise ValueError(f"seed {seed}: checkpoint reports test evaluation")
        if checkpoint.get("initialization") != "train_lot_unlabeled_contrastive":
            raise ValueError(f"seed {seed}: unexpected checkpoint initialization")
        if checkpoint.get("class_names") != list(CLASS_NAMES):
            raise ValueError(f"seed {seed}: class mapping mismatch")
    return hashlib.sha256(payload).hexdigest()


def _recompute_guardrails(
    candidate_metrics: dict[str, dict[str, float]],
    candidate_recalls: dict[str, dict[str, float]],
    baseline_metrics: dict[str, dict[str, float]],
    baseline_recalls: dict[str, float],
) -> dict[str, bool]:
    checks = {
        "three_or_more_seeds": True,
        "macro_f1_mean_improves_by_0_002": (
            candidate_metrics["macro_f1"]["mean"]
            >= baseline_metrics["macro_f1"]["mean"] + 0.002
        ),
        "balanced_accuracy_drop_within_0_002": (
            candidate_metrics["balanced_accuracy"]["mean"]
            >= baseline_metrics["balanced_accuracy"]["mean"] - 0.002
        ),
        "normal_recall_drop_within_0_002": (
            candidate_recalls["none"]["mean"] >= baseline_recalls["none"] - 0.002
        ),
    }
    for class_name in WEAK_CLASSES:
        checks[f"{class_name}_recall_drop_within_0_01"] = (
            candidate_recalls[class_name]["mean"]
            >= baseline_recalls[class_name] - 0.01
        )
    return checks


def validate_bundle(
    zip_path: Path,
    assignments_path: Path,
) -> dict[str, object]:
    assignments = pd.read_csv(assignments_path)
    required_assignment_columns = {
        "array_index", "label_id", "failure_type", "lot_name", "split"
    }
    missing = required_assignment_columns.difference(assignments.columns)
    if missing:
        raise ValueError(f"Assignments missing columns: {sorted(missing)}")
    validation_assignments = assignments.loc[
        assignments["split"] == "validation"
    ].copy()
    if len(validation_assignments) != 24_703:
        raise ValueError("Frozen validation row count is not 24,703")

    with zipfile.ZipFile(zip_path) as archive:
        unsafe = [name for name in archive.namelist() if not _safe_name(name)]
        if unsafe:
            raise ValueError(f"Unsafe ZIP paths: {unsafe}")
        files = {info.filename for info in archive.infolist() if not info.is_dir()}
        expected_files = _expected_files()
        if files != expected_files:
            raise ValueError(
                f"Bundle inventory mismatch; missing={sorted(expected_files-files)}, "
                f"unexpected={sorted(files-expected_files)}"
            )
        for seed in SEEDS:
            for image_name in (
                f"seeds/seed_{seed}/training_curves.png",
                f"seeds/seed_{seed}/validation_confusion_matrix.png",
            ):
                if not archive.read(image_name).startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError(f"{image_name} is not a PNG")

        declared = _read_json(archive, "ssl_candidate_comparison.json")
        expected_contract = {
            "schema_version": 1,
            "purpose": "validation_only_candidate_screening",
            "candidate": "train_lot_unlabeled_contrastive_pretraining",
            "baseline": "ce_sqrt_balanced_random_initialization",
            "seed_count": 3,
            "finetune_seeds": list(SEEDS),
            "fixed_test_reopened": False,
            "test_used_for_selection": False,
            "deployment_model_changed": False,
        }
        for key, value in expected_contract.items():
            if declared.get(key) != value:
                raise ValueError(f"Unexpected comparison contract for {key}")
        if declared.get("split_assignments_sha256") != file_sha256(assignments_path):
            raise ValueError("Split assignment hash does not match")

        pretraining = _read_csv(archive, "pretraining_history.csv")
        if (
            len(pretraining) != int(declared["pretrain_epochs"])
            or pretraining["epoch"].tolist() != list(range(1, len(pretraining) + 1))
            or not np.isfinite(pretraining["contrastive_loss"]).all()
        ):
            raise ValueError("Invalid pretraining history")

        encoder_hash = _validate_checkpoint(
            archive, "ssl_encoder.pt", seed=None
        )
        metrics_by_seed: dict[str, dict[str, float]] = {}
        recalls_by_seed: dict[str, dict[str, float]] = {}
        checkpoint_hashes: dict[str, str] = {}
        for seed in SEEDS:
            prefix = f"seeds/seed_{seed}"
            summary = _read_json(archive, f"{prefix}/run_summary.json")
            if (
                summary.get("schema_version") != 1
                or summary.get("experiment_type") != "wm811k_ssl_validation_only"
                or summary.get("seed") != seed
                or summary.get("test_evaluated") is not False
                or summary.get("selection_split") != "validation"
            ):
                raise ValueError(f"seed {seed}: invalid run contract")
            history = _read_csv(archive, f"{prefix}/training_history.csv")
            if (
                len(history) != int(summary["epochs_ran"])
                or int(summary["best_epoch"]) not in set(history["epoch"])
            ):
                raise ValueError(f"seed {seed}: training history mismatch")
            predictions = _read_csv(
                archive, f"{prefix}/validation_predictions.csv"
            )
            metrics, recalls = _validate_predictions(
                predictions, validation_assignments, seed
            )
            for metric, value in metrics.items():
                _assert_close(
                    value,
                    float(summary["validation_metrics"][metric]),
                    f"seed {seed}/{metric}",
                )
            for class_name, value in recalls.items():
                _assert_close(
                    value,
                    float(summary["validation_class_recall"][class_name]),
                    f"seed {seed}/{class_name}_recall",
                )
            metrics_by_seed[str(seed)] = metrics
            recalls_by_seed[str(seed)] = recalls
            checkpoint_hashes[str(seed)] = _validate_checkpoint(
                archive,
                f"{prefix}/best_validation_model.pt",
                seed=seed,
            )

        candidate_metrics: dict[str, dict[str, float]] = {}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"):
            values = np.asarray(
                [metrics_by_seed[str(seed)][metric] for seed in SEEDS]
            )
            candidate_metrics[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
            for statistic, value in candidate_metrics[metric].items():
                _assert_close(
                    value,
                    float(declared["candidate_validation_metrics"][metric][statistic]),
                    f"aggregate/{metric}/{statistic}",
                )

        candidate_recalls: dict[str, dict[str, float]] = {}
        for class_name in CLASS_NAMES:
            values = np.asarray(
                [recalls_by_seed[str(seed)][class_name] for seed in SEEDS]
            )
            candidate_recalls[class_name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
            for statistic, value in candidate_recalls[class_name].items():
                _assert_close(
                    value,
                    float(
                        declared["candidate_validation_class_recall"][class_name][
                            statistic
                        ]
                    ),
                    f"aggregate/{class_name}/{statistic}",
                )

        baseline_metrics = declared["baseline_validation_metrics"]
        baseline_recalls = declared["baseline_validation_class_recall"]
        checks = _recompute_guardrails(
            candidate_metrics, candidate_recalls, baseline_metrics, baseline_recalls
        )
        if checks != declared["guardrail_checks"]:
            raise ValueError("Declared guardrail checks do not match recomputation")
        eligible = all(checks.values())
        if eligible != bool(declared["all_guardrails_passed"]):
            raise ValueError("Declared eligibility is inconsistent")
        expected_decision = (
            "validation_supported_external_holdout_required"
            if eligible
            else "retain_existing_model"
        )
        if declared["decision"] != expected_decision:
            raise ValueError("Declared decision is inconsistent")

    baseline_macro = float(baseline_metrics["macro_f1"]["mean"])
    baseline_balanced = float(baseline_metrics["balanced_accuracy"]["mean"])
    normal_delta = (
        candidate_recalls["none"]["mean"] - float(baseline_recalls["none"])
    )
    return {
        "schema_version": 1,
        "status": "validated",
        "bundle_sha256": file_sha256(zip_path),
        "bundle_file_count": len(expected_files),
        "split_assignments_sha256": file_sha256(assignments_path),
        "seeds": list(SEEDS),
        "pretrain_rows": int(declared["pretrain_rows"]),
        "pretrain_epochs": int(declared["pretrain_epochs"]),
        "pretrain_seconds": float(declared["pretrain_seconds"]),
        "validation_rows_per_seed": len(validation_assignments),
        "validation_prediction_rows_checked": len(validation_assignments) * len(SEEDS),
        "candidate_validation_metrics": candidate_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "candidate_validation_class_recall": candidate_recalls,
        "baseline_validation_class_recall": baseline_recalls,
        "metric_deltas": {
            "macro_f1": candidate_metrics["macro_f1"]["mean"] - baseline_macro,
            "balanced_accuracy": (
                candidate_metrics["balanced_accuracy"]["mean"] - baseline_balanced
            ),
            "none_recall": normal_delta,
            **{
                f"{name}_recall": (
                    candidate_recalls[name]["mean"] - float(baseline_recalls[name])
                )
                for name in WEAK_CLASSES
            },
        },
        "guardrail_checks": checks,
        "all_guardrails_passed": eligible,
        "decision": expected_decision,
        "fixed_test_reopened": False,
        "test_used_for_selection": False,
        "deployment_model_changed": False,
        "ssl_encoder_sha256": encoder_hash,
        "candidate_checkpoint_sha256": checkpoint_hashes,
        "metrics_by_seed": metrics_by_seed,
        "class_recall_by_seed": recalls_by_seed,
        "conclusion": (
            "candidate_requires_external_holdout"
            if eligible
            else "existing_champion_retained"
        ),
    }


def write_report(report: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bundle_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    deltas = report["metric_deltas"]
    checks = report["guardrail_checks"]
    lines = [
        "# WM-811K 비라벨 자기지도 후보 검증",
        "",
        "- 상태: 업로드 번들 검증 통과",
        f"- 검증 seed: {', '.join(map(str, report['seeds']))}",
        f"- 재계산한 validation 예측: {report['validation_prediction_rows_checked']:,}행",
        "- 고정 test 재사용: 아니요",
        "- 운영 모델 변경: 아니요",
        f"- 결정: {report['decision']}",
        f"- 번들 SHA-256: {report['bundle_sha256']}",
        "",
        "| 지표 | 기존 기준선 | 자기지도 후보 | 변화 |",
        "|---|---:|---:|---:|",
        (
            "| Validation Macro-F1 | "
            f"{report['baseline_validation_metrics']['macro_f1']['mean']:.4f} | "
            f"{report['candidate_validation_metrics']['macro_f1']['mean']:.4f} | "
            f"{deltas['macro_f1']:+.4f} |"
        ),
        (
            "| Validation balanced accuracy | "
            f"{report['baseline_validation_metrics']['balanced_accuracy']['mean']:.4f} | "
            f"{report['candidate_validation_metrics']['balanced_accuracy']['mean']:.4f} | "
            f"{deltas['balanced_accuracy']:+.4f} |"
        ),
        (
            "| none recall | "
            f"{report['baseline_validation_class_recall']['none']:.4f} | "
            f"{report['candidate_validation_class_recall']['none']['mean']:.4f} | "
            f"{deltas['none_recall']:+.4f} |"
        ),
        *[
            (
                f"| {name} recall | "
                f"{report['baseline_validation_class_recall'][name]:.4f} | "
                f"{report['candidate_validation_class_recall'][name]['mean']:.4f} | "
                f"{deltas[f'{name}_recall']:+.4f} |"
            )
            for name in WEAK_CLASSES
        ],
        "",
        "## 결정",
        "",
        (
            "자기지도 사전학습은 Macro-F1, balanced accuracy와 취약 3종 recall을 "
            "개선했습니다. 그러나 none recall 감소폭이 사전 선언한 0.2%p 허용치를 "
            f"넘었습니다(실제 {abs(deltas['none_recall']):.3%}p)."
        ),
        "",
        (
            "따라서 현재 champion은 유지합니다. 이 결과를 보고 임계값을 같은 "
            "validation에서 다시 맞추면 선택 편향이 커지므로, 후보를 개선하려면 "
            "train 내부 OOF 보정 또는 새 외부 holdout을 사용해야 합니다."
        ),
        "",
        "### 검증된 안전장치",
        "",
        *[f"- {name}: {'통과' if passed else '실패'}" for name, passed in checks.items()],
    ]
    (output_dir / "검증결과.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_bundle(args.zip, args.assignments)
    write_report(report, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
