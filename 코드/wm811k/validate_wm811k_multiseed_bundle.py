"""Validate the compact WM-811K multi-seed Colab result bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


EXPECTED_SEEDS = (17, 42, 2026)
EXPECTED_STRATEGY = "ce_sqrt_balanced"
EXPECTED_LOSS = "cross_entropy"
EXPECTED_SAMPLING = "sqrt_balanced"
EXPECTED_METRICS = (
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_macro_f1",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
)
BASE_FILES = {
    "protocol.json",
    "seed_metrics.csv",
    "metric_summary.csv",
    "reproducibility_summary.json",
    "reproducibility_metrics.png",
}
SEED_FILES = {
    f"seeds/seed_{seed}/{name}"
    for seed in EXPECTED_SEEDS
    for name in (
        "run_summary.json",
        "validation_classification_report.csv",
        "test_classification_report.csv",
    )
}
REQUIRED_FILES = BASE_FILES | SEED_FILES
MAX_ENTRIES = 25
MAX_UNCOMPRESSED_BYTES = 25 * 1024 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(info: zipfile.ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    unix_mode = info.external_attr >> 16
    return (
        bool(info.filename)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in info.filename
        and ":" not in info.filename
        and not stat.S_ISLNK(unix_mode)
    )


def _require_protocol(payload: dict[str, object], assignments_hash: str) -> None:
    expected = {
        "schema_version": 1,
        "experiment_type": "wm811k_multiseed_reproducibility",
        "strategy_id": EXPECTED_STRATEGY,
        "loss": EXPECTED_LOSS,
        "sampling": EXPECTED_SAMPLING,
        "sampling_max_multiplier": 8.0,
        "selection_split": "validation",
        "primary_metric": "validation_macro_f1",
        "test_used_for_selection": False,
        "deployment_model_changed": False,
        "split_assignments_sha256": assignments_hash,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"protocol 계약 불일치: {key}")
    if tuple(payload.get("seeds", [])) != EXPECTED_SEEDS:
        raise ValueError("protocol의 시드 구성이 잘못되었습니다.")


def _finite_metric(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} 지표가 0~1의 유한값이 아닙니다.")
    return number


def validate_bundle(bundle_path: Path, assignments_path: Path) -> dict[str, object]:
    for path in (bundle_path, assignments_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    assignments_hash = sha256_file(assignments_path)

    with zipfile.ZipFile(bundle_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos if not info.is_dir()]
        if len(infos) > MAX_ENTRIES:
            raise ValueError(f"ZIP 항목이 너무 많습니다: {len(infos)}")
        if len(names) != len(set(names)):
            raise ValueError("ZIP에 중복 파일명이 있습니다.")
        unsafe = [info.filename for info in infos if not _safe_member(info)]
        if unsafe:
            raise ValueError(f"안전하지 않은 ZIP 경로가 있습니다: {unsafe}")
        total_size = sum(info.file_size for info in infos)
        if total_size > MAX_UNCOMPRESSED_BYTES:
            raise ValueError(f"압축 해제 크기가 제한을 넘습니다: {total_size:,} bytes")
        missing = REQUIRED_FILES.difference(names)
        extra = set(names).difference(REQUIRED_FILES)
        if missing:
            raise ValueError(f"필수 결과 파일 누락: {sorted(missing)}")
        if extra:
            raise ValueError(f"예상하지 않은 결과 파일: {sorted(extra)}")

        protocol = json.loads(archive.read("protocol.json").decode("utf-8"))
        summary = json.loads(
            archive.read("reproducibility_summary.json").decode("utf-8")
        )
        seed_metrics = pd.read_csv(io.BytesIO(archive.read("seed_metrics.csv")))
        metric_summary = pd.read_csv(io.BytesIO(archive.read("metric_summary.csv")))
        png_signature = archive.read("reproducibility_metrics.png")[:8]
        run_summaries = {
            seed: json.loads(
                archive.read(f"seeds/seed_{seed}/run_summary.json").decode("utf-8")
            )
            for seed in EXPECTED_SEEDS
        }

    if png_signature != b"\x89PNG\r\n\x1a\n":
        raise ValueError("reproducibility_metrics.png가 유효한 PNG가 아닙니다.")
    _require_protocol(protocol, assignments_hash)
    _require_protocol(summary, assignments_hash)
    if summary.get("purpose") != "stability_evidence_not_model_selection":
        raise ValueError("재현성 실험 목적이 잘못 기록되었습니다.")
    if summary.get("std_ddof") != 1:
        raise ValueError("표준편차는 표본 표준편차(ddof=1)여야 합니다.")
    if summary.get("existing_deployment_seed") != 42:
        raise ValueError("기존 배포 시드 기록이 잘못되었습니다.")

    required_columns = {
        "seed",
        "best_epoch",
        "epochs_ran",
        "elapsed_seconds",
        "checkpoint_sha256",
        *EXPECTED_METRICS,
    }
    if not required_columns.issubset(seed_metrics.columns):
        raise ValueError("seed_metrics.csv 필수 열이 누락되었습니다.")
    if len(seed_metrics) != len(EXPECTED_SEEDS):
        raise ValueError("seed_metrics.csv 행 수가 시드 수와 다릅니다.")
    if tuple(sorted(seed_metrics["seed"].astype(int))) != EXPECTED_SEEDS:
        raise ValueError("seed_metrics.csv 시드 구성이 잘못되었습니다.")
    seed_metrics = seed_metrics.set_index(seed_metrics["seed"].astype(int))

    for seed in EXPECTED_SEEDS:
        row = seed_metrics.loc[seed]
        run = run_summaries[seed]
        if int(run.get("seed", -1)) != seed:
            raise ValueError(f"seed {seed} run_summary의 시드가 다릅니다.")
        if run.get("loss") != EXPECTED_LOSS or run.get("sampling") != EXPECTED_SAMPLING:
            raise ValueError(f"seed {seed}의 학습 전략이 고정 계약과 다릅니다.")
        if float(run.get("sampling_max_multiplier", -1)) != 8.0:
            raise ValueError(f"seed {seed}의 sampling multiplier가 다릅니다.")
        if int(row["best_epoch"]) != int(run["best_epoch"]):
            raise ValueError(f"seed {seed}의 best epoch가 요약과 다릅니다.")
        if int(row["epochs_ran"]) != int(run["epochs_ran"]):
            raise ValueError(f"seed {seed}의 실행 epoch가 요약과 다릅니다.")
        if not SHA256_PATTERN.fullmatch(str(row["checkpoint_sha256"])):
            raise ValueError(f"seed {seed} 체크포인트 SHA-256 형식이 잘못되었습니다.")
        for prefix, source_key in (("validation", "validation_metrics"), ("test", "test_metrics")):
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                column = f"{prefix}_{metric}"
                actual = _finite_metric(row[column], column)
                expected = _finite_metric(run[source_key][metric], column)
                if not np.isclose(actual, expected, rtol=0, atol=1e-12):
                    raise ValueError(f"seed {seed}의 {column}이 run_summary와 다릅니다.")

    if set(metric_summary.columns) != {"metric", "mean", "std", "min", "max"}:
        raise ValueError("metric_summary.csv 열 구성이 잘못되었습니다.")
    if set(metric_summary["metric"]) != set(EXPECTED_METRICS):
        raise ValueError("metric_summary.csv 지표 구성이 잘못되었습니다.")
    metric_summary = metric_summary.set_index("metric")
    summary_metrics = summary.get("metrics", {})
    for metric in EXPECTED_METRICS:
        values = seed_metrics[metric].astype(float).to_numpy()
        expected_values = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for statistic, expected in expected_values.items():
            csv_value = float(metric_summary.loc[metric, statistic])
            json_value = float(summary_metrics[metric][statistic])
            if not np.isclose(csv_value, expected, rtol=0, atol=1e-12):
                raise ValueError(f"{metric} {statistic} CSV 집계가 잘못되었습니다.")
            if not np.isclose(json_value, expected, rtol=0, atol=1e-12):
                raise ValueError(f"{metric} {statistic} JSON 집계가 잘못되었습니다.")

    validation_values = seed_metrics["validation_macro_f1"].astype(float).to_numpy()
    return {
        "bundle": str(bundle_path),
        "zip_sha256": sha256_file(bundle_path),
        "entries": len(names),
        "uncompressed_bytes": int(total_size),
        "seeds": list(EXPECTED_SEEDS),
        "strategy_id": EXPECTED_STRATEGY,
        "validation_macro_f1_mean": float(np.mean(validation_values)),
        "validation_macro_f1_std": float(np.std(validation_values, ddof=1)),
        "assignments_sha256": assignments_hash,
        "deployment_model_changed": False,
        "validation_passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = validate_bundle(args.bundle, args.assignments)
    print(json.dumps(report, ensure_ascii=False, indent=2))
