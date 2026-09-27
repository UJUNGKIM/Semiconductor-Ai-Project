"""Validate a WM-811K OOD Colab ZIP before extracting or deploying it."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


REQUIRED_FILES = {
    "ood_reference.json",
    "ood_reference.npz",
    "ood_summary.json",
    "ood_split_metrics.csv",
    "ood_score_records.csv",
    "ood_score_distribution.png",
    "README.md",
}
MAX_ENTRIES = 20
MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024


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


def validate_bundle(
    bundle_path: Path,
    checkpoint_path: Path,
    assignments_path: Path,
) -> dict[str, object]:
    for path in (bundle_path, checkpoint_path, assignments_path):
        if not path.is_file():
            raise FileNotFoundError(path)

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

        summary = json.loads(archive.read("ood_summary.json").decode("utf-8"))
        reference = json.loads(archive.read("ood_reference.json").decode("utf-8"))
        with np.load(
            io.BytesIO(archive.read("ood_reference.npz")), allow_pickle=False
        ) as arrays:
            array_names = set(arrays.files)
            if array_names != {"class_means", "class_variances", "class_counts"}:
                raise ValueError(f"OOD NPZ 배열 구성이 잘못되었습니다: {sorted(array_names)}")
            class_means = arrays["class_means"].copy()
            class_variances = arrays["class_variances"].copy()
            class_counts = arrays["class_counts"].copy()
        metrics = pd.read_csv(io.BytesIO(archive.read("ood_split_metrics.csv")))
        records = pd.read_csv(io.BytesIO(archive.read("ood_score_records.csv")))
        png_signature = archive.read("ood_score_distribution.png")[:8]

    if png_signature != b"\x89PNG\r\n\x1a\n":
        raise ValueError("ood_score_distribution.png가 유효한 PNG가 아닙니다.")
    if summary.get("schema_version") != 1 or reference.get("schema_version") != 1:
        raise ValueError("지원하지 않는 OOD 결과 schema입니다.")
    selection = summary.get("threshold_selection", {})
    if selection.get("calibration_split") != "validation":
        raise ValueError("OOD 임계값은 validation에서 고정해야 합니다.")
    if selection.get("test_used_for_threshold") is not False:
        raise ValueError("test가 OOD 임계값 선택에 사용되었습니다.")
    if reference.get("fit_split") != "train":
        raise ValueError("OOD 특징 기준은 train으로 적합해야 합니다.")
    if reference.get("calibration_split") != "validation":
        raise ValueError("reference calibration split이 validation이 아닙니다.")
    if reference.get("test_used_for_threshold") is not False:
        raise ValueError("reference가 test 사용 금지 계약을 위반합니다.")

    thresholds = reference.get("thresholds", {})
    review_threshold = float(thresholds["review_threshold"])
    ood_threshold = float(thresholds["ood_threshold"])
    if not (0.0 < review_threshold < ood_threshold < np.inf):
        raise ValueError("OOD 임계값 순서 또는 값이 잘못되었습니다.")
    if not np.isclose(review_threshold, selection["review_threshold"]):
        raise ValueError("summary와 reference의 review 임계값이 다릅니다.")
    if not np.isclose(ood_threshold, selection["ood_threshold"]):
        raise ValueError("summary와 reference의 OOD 임계값이 다릅니다.")

    class_names = list(reference.get("class_names", []))
    if len(class_names) != 9 or class_means.shape != (9, 256):
        raise ValueError(
            f"OOD 클래스 또는 평균 형상이 잘못되었습니다: {len(class_names)}, {class_means.shape}"
        )
    if class_variances.shape != class_means.shape or class_counts.shape != (9,):
        raise ValueError("OOD 분산 또는 클래스 개수 형상이 잘못되었습니다.")
    if not np.isfinite(class_means).all() or not np.isfinite(class_variances).all():
        raise ValueError("OOD 기준 배열에 유한하지 않은 값이 있습니다.")
    if np.any(class_variances <= 0) or np.any(class_counts < 2):
        raise ValueError("OOD 분산 또는 클래스 train 개수가 유효하지 않습니다.")

    expected_checkpoint_hash = sha256_file(checkpoint_path)
    expected_assignments_hash = sha256_file(assignments_path)
    for payload_name, payload in (("reference", reference), ("summary", summary)):
        if payload.get("checkpoint_sha256") != expected_checkpoint_hash:
            raise ValueError(f"{payload_name}의 선택 체크포인트 해시가 다릅니다.")
        if payload.get("assignments_sha256") != expected_assignments_hash:
            raise ValueError(f"{payload_name}의 split 해시가 다릅니다.")

    required_record_columns = {
        "split",
        "array_index",
        "true_label_id",
        "predicted_label_id",
        "nearest_reference_label_id",
        "ood_score",
        "ood_status",
    }
    if not required_record_columns.issubset(records.columns):
        raise ValueError("OOD score records 열이 누락되었습니다.")
    if not np.isfinite(records["ood_score"]).all():
        raise ValueError("OOD score records에 유한하지 않은 값이 있습니다.")
    expected_splits = {"validation", "test", "spatial_shuffle_stress"}
    if set(records["split"].unique()) != expected_splits:
        raise ValueError("OOD score records의 split 구성이 잘못되었습니다.")
    values = records["ood_score"].to_numpy()
    recalculated_status = np.where(
        values >= ood_threshold,
        "out_of_distribution",
        np.where(values >= review_threshold, "review", "in_distribution"),
    )
    if not np.array_equal(recalculated_status, records["ood_status"].to_numpy()):
        raise ValueError("저장된 OOD 상태가 고정 임계값과 일치하지 않습니다.")
    split_rows = summary.get("split_rows", {})
    for split_name in ("validation", "test"):
        actual_rows = int((records["split"] == split_name).sum())
        if actual_rows != int(split_rows[split_name]):
            raise ValueError(f"{split_name} score 행 수가 summary와 다릅니다.")
    if set(metrics["split"]) != {"train", "validation", "test", "spatial_shuffle_stress"}:
        raise ValueError("OOD split metrics 구성이 잘못되었습니다.")

    return {
        "bundle": str(bundle_path),
        "zip_sha256": sha256_file(bundle_path),
        "entries": len(names),
        "uncompressed_bytes": int(total_size),
        "method": reference["method"],
        "feature_dim": int(reference["feature_dim"]),
        "review_threshold": review_threshold,
        "ood_threshold": ood_threshold,
        "split_rows": split_rows,
        "split_metrics": summary["split_metrics"],
        "stress_metrics": summary["stress_metrics"],
        "checkpoint_sha256": expected_checkpoint_hash,
        "assignments_sha256": expected_assignments_hash,
        "validation_passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = validate_bundle(args.bundle, args.checkpoint, args.assignments)
    print(json.dumps(report, ensure_ascii=False, indent=2))
