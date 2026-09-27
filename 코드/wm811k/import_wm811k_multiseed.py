"""Validate and import compact WM-811K reproducibility evidence, without replacing models."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from validate_wm811k_multiseed_bundle import (
    EXPECTED_SEEDS,
    REQUIRED_FILES,
    sha256_file,
    validate_bundle,
)


def inspect_class_reports(bundle_path: Path, assignments_path: Path) -> list[dict]:
    """Reconcile class counts and report aggregates against the fixed split."""
    assignments = pd.read_csv(assignments_path)
    class_rows = assignments[["label_id", "failure_type"]].drop_duplicates()
    class_rows = class_rows.sort_values("label_id")
    if class_rows["label_id"].tolist() != list(range(9)):
        raise ValueError("고정 분할의 클래스 매핑이 잘못되었습니다.")
    class_names = class_rows["failure_type"].tolist()
    collected = {split: {name: [] for name in class_names} for split in ("validation", "test")}
    expected_train_counts = (
        assignments.loc[assignments["split"] == "train", "failure_type"]
        .value_counts().reindex(class_names).to_numpy(dtype=int)
    )
    with zipfile.ZipFile(bundle_path) as archive:
        protocol = json.loads(archive.read("protocol.json"))
        for key, value in {"fixed_split": True, "epochs": 20, "batch_size": 256, "patience": 4, "minimum_delta": 0.0001}.items():
            if protocol.get(key) != value:
                raise ValueError(f"학습 설정 계약 불일치: {key}")
        for seed in EXPECTED_SEEDS:
            run = json.loads(archive.read(f"seeds/seed_{seed}/run_summary.json"))
            if run.get("class_names") != class_names:
                raise ValueError(f"seed {seed}의 클래스 순서가 분할과 다릅니다.")
            if not np.array_equal(run.get("class_counts"), expected_train_counts):
                raise ValueError(f"seed {seed}의 train 클래스 수가 분할과 다릅니다.")
            best_epoch, epochs_ran = run["best_epoch"], run["epochs_ran"]
            if type(best_epoch) is not int or type(epochs_ran) is not int or not 1 <= best_epoch <= epochs_ran <= 20:
                raise ValueError(f"seed {seed}의 epoch 기록이 잘못되었습니다.")
            if not np.isfinite(run["elapsed_seconds"]) or run["elapsed_seconds"] <= 0:
                raise ValueError(f"seed {seed}의 실행 시간이 잘못되었습니다.")
            for split in ("validation", "test"):
                report = pd.read_csv(
                    io.BytesIO(archive.read(f"seeds/seed_{seed}/{split}_classification_report.csv")),
                    index_col=0,
                )
                if report.index.tolist() != class_names + ["accuracy", "macro avg", "weighted avg"]:
                    raise ValueError(f"seed {seed} {split} 보고서 클래스 구성이 다릅니다.")
                if set(report.columns) != {"precision", "recall", "f1-score", "support"}:
                    raise ValueError(f"seed {seed} {split} 보고서 열 구성이 다릅니다.")
                classes = report.loc[class_names]
                values = classes[["precision", "recall", "f1-score"]].to_numpy(dtype=float)
                if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                    raise ValueError(f"seed {seed} {split} 클래스 지표가 유효하지 않습니다.")
                expected_counts = (
                    assignments.loc[assignments["split"] == split, "failure_type"]
                    .value_counts().reindex(class_names).to_numpy(dtype=int)
                )
                if not np.array_equal(classes["support"].to_numpy(), expected_counts):
                    raise ValueError(f"seed {seed} {split} 클래스 support가 분할과 다릅니다.")
                precision, recall, f1 = values.T
                expected_f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(9), where=(precision + recall) > 0)
                if not np.allclose(f1, expected_f1, rtol=0, atol=1e-12):
                    raise ValueError(f"seed {seed} {split} 클래스 F1 계산이 잘못되었습니다.")
                checks = {
                    "macro_f1": (float(np.mean(f1)), float(report.loc["macro avg", "f1-score"])),
                    "balanced_accuracy": (float(np.mean(recall)), float(report.loc["macro avg", "recall"])),
                    "accuracy": (float(np.average(recall, weights=expected_counts)), float(report.loc["accuracy", "precision"])),
                }
                for metric, (recomputed, reported) in checks.items():
                    saved = float(run[f"{split}_metrics"][metric])
                    if not np.allclose([reported, saved], recomputed, rtol=0, atol=1e-12):
                        raise ValueError(f"seed {seed} {split} {metric} 집계가 보고서와 다릅니다.")
                for name, value in zip(class_names, recall):
                    collected[split][name].append(float(value))

    result = []
    for split in ("validation", "test"):
        counts = assignments.loc[assignments["split"] == split, "failure_type"].value_counts()
        for name in class_names:
            values = np.array(collected[split][name])
            result.append({
                "split": split,
                "class_name": name,
                "support_per_seed": int(counts[name]),
                "recall_mean": float(values.mean()),
                "recall_std": float(values.std(ddof=1)),
                "recall_min": float(values.min()),
                "recall_max": float(values.max()),
                "recall_by_seed": {str(seed): float(value) for seed, value in zip(EXPECTED_SEEDS, values)},
            })
    return result


def import_bundle(bundle_path: Path, assignments_path: Path, checkpoint_path: Path, output_dir: Path) -> dict:
    validation = validate_bundle(bundle_path, assignments_path)
    class_recall = inspect_class_reports(bundle_path, assignments_path)
    checkpoint_hash_before = sha256_file(checkpoint_path)
    with zipfile.ZipFile(bundle_path) as archive:
        seed_metrics = pd.read_csv(io.BytesIO(archive.read("seed_metrics.csv")))
        seed42_hash = str(seed_metrics.loc[seed_metrics["seed"] == 42, "checkpoint_sha256"].iloc[0])
        if seed42_hash != checkpoint_hash_before:
            raise ValueError("시드 42 재실행 체크포인트 해시가 현재 배포 모델과 다릅니다.")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"기존 결과를 덮어쓰지 않습니다: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(REQUIRED_FILES):
            destination = output_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))

    validation.update({
        "class_reports_reconciled": True,
        "seed42_checkpoint_matches_deployment": True,
        "deployed_checkpoint_sha256": checkpoint_hash_before,
        "class_recall_rows": len(class_recall),
        "test_used_for_selection": False,
    })
    validation["bundle"] = bundle_path.name
    (output_dir / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "class_recall_summary.json").write_text(json.dumps(class_recall, ensure_ascii=False, indent=2), encoding="utf-8")
    if sha256_file(checkpoint_path) != checkpoint_hash_before:
        raise RuntimeError("가져오기 중 배포 체크포인트가 변경되었습니다.")

    summary = json.loads((output_dir / "reproducibility_summary.json").read_text(encoding="utf-8"))
    metrics = summary["metrics"]
    lines = [
        "# WM-811K 다중 시드 재현성 검증 결과", "",
        "- 시드: 17, 42, 2026 / 고정 lot 비중복 분할 / ce_sqrt_balanced",
        "- ZIP 안전 경로, 중복·필수 파일, 분할 SHA-256, 시드별 설정, 평균·표본 표준편차(ddof=1) 확인",
        "- 클래스 순서, train 수, validation/test support, 클래스 F1 및 전체 지표 집계 일치",
        "- 시드 42 체크포인트 SHA-256은 현재 배포 모델과 동일. 모델과 불확실성/OOD 정책은 변경하지 않음.", "",
        "| 시드 | Validation Macro-F1 | 고정 Test Macro-F1 | 최적 epoch |", "| --- | ---: | ---: | ---: |",
    ]
    for row in seed_metrics.itertuples(index=False):
        lines.append(f"| {row.seed} | {row.validation_macro_f1:.6f} | {row.test_macro_f1:.6f} | {row.best_epoch} |")
    lines += ["", f"Validation Macro-F1: **{metrics['validation_macro_f1']['mean']:.6f} ± {metrics['validation_macro_f1']['std']:.6f}**.", f"고정 Test Macro-F1: **{metrics['test_macro_f1']['mean']:.6f} ± {metrics['test_macro_f1']['std']:.6f}**.", "", "## 해석과 한계", "", "Test Macro-F1은 세 시드에서 0.8787~0.8829로 가까웠습니다. 클래스별 재현율은 더 변동하며 Scratch는 71.76~81.76%, Loc는 71.54~79.53%, Random은 86.29~96.77%였습니다. 특히 Near-full test support는 시드당 동일한 21개뿐입니다.", "", "이는 같은 고정 데이터 분할에서의 학습 초기값 안정성 근거입니다. 세 시드는 서로 독립적인 신규 test 표본이 아니며 평균 ± 표준편차는 신뢰구간이나 외부 공정 검증이 아닙니다. 가장 좋은 시드를 선택하지 않았고 test 분석으로 모델·임계값을 변경하지 않았습니다. 보정 신뢰도/OOD 정책을 모든 시드에서 재검증한 실험도 아닙니다.", "", "전달 ZIP은 요약 번들이므로 시드 17·2026의 전체 예측과 체크포인트를 독립 재실행하지는 않았습니다. 전체 산출물은 팀원의 Drive에 보존됩니다. 계약 기록과 보고서 간 일치를 확인했으며 학습 이력 자체를 외부 감사로 증명하지는 않습니다.", "", f"ZIP SHA-256: `{validation['zip_sha256']}`", f"분할 SHA-256: `{validation['assignments_sha256']}`", f"배포 체크포인트 SHA-256: `{checkpoint_hash_before}`", ""]
    (output_dir / "검증결과.md").write_text("\n".join(lines), encoding="utf-8")
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(import_bundle(args.bundle, args.assignments, args.checkpoint, args.output_dir), ensure_ascii=False, indent=2))
