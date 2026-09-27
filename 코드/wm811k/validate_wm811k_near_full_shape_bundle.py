"""Validate a WM-811K Near-full global-shape result ZIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import stat
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


REQUIRED_FILES = {
    "near_full_global_shape_samples.csv",
    "near_full_global_shape_summary.json",
    "near_full_global_shape_dashboard.png",
    "README.md",
}
IDENTITY_COLUMNS = {
    "array_index",
    "source_row",
    "lot_name",
    "wafer_index",
    "original_height",
    "original_width",
}
METRIC_COLUMNS = [
    "defect_ratio",
    "component_count_8",
    "largest_component_fraction",
    "mean_normalized_radius",
    "center_defect_fraction",
    "edge_defect_fraction",
    "boundary_defect_coverage",
    "defect_bbox_fraction",
    "defect_centroid_offset",
    "radial_entropy_4bin",
]
COUNT_COLUMNS = {"active_die_count", "defect_die_count", "component_count_8"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(info: zipfile.ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    return (
        bool(info.filename)
        and not path.is_absolute()
        and ".." not in path.parts
        and chr(92) not in info.filename
        and ":" not in info.filename
        and not stat.S_ISLNK(mode)
    )


def _close(actual: float, expected: float, name: str) -> None:
    if (
        not math.isfinite(actual)
        or not np.isclose(actual, expected, rtol=0, atol=1e-12)
    ):
        raise ValueError(f"{name} 집계가 표본 CSV와 일치하지 않습니다.")


def validate_bundle(bundle: Path, assignments_path: Path) -> dict[str, object]:
    if not bundle.is_file() or not assignments_path.is_file():
        raise FileNotFoundError(bundle if not bundle.is_file() else assignments_path)
    with zipfile.ZipFile(bundle) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos if not info.is_dir()]
        if len(infos) > 8 or sum(info.file_size for info in infos) > 5 * 1024 * 1024:
            raise ValueError("ZIP 항목 수 또는 압축 해제 크기가 제한을 넘습니다.")
        if len(names) != len(set(names)) or any(not _safe(info) for info in infos):
            raise ValueError("ZIP에 중복 또는 안전하지 않은 경로가 있습니다.")
        if set(names) != REQUIRED_FILES:
            raise ValueError(
                f"ZIP 파일 구성이 다릅니다: {sorted(set(names) ^ REQUIRED_FILES)}"
            )
        summary = json.loads(
            archive.read("near_full_global_shape_summary.json").decode("utf-8")
        )
        records = pd.read_csv(
            io.BytesIO(archive.read("near_full_global_shape_samples.csv"))
        )
        if (
            archive.read("near_full_global_shape_dashboard.png")[:8]
            != b"\x89PNG\r\n\x1a\n"
        ):
            raise ValueError("대시보드 이미지가 유효한 PNG가 아닙니다.")

    contract = {
        "schema_version": 1,
        "purpose": "post_selection_near_full_global_shape_diagnostic",
        "used_for_model_selection": False,
        "thresholds_retuned": False,
        "deployment_model_changed": False,
        "split": "test",
        "class_name": "Near-full",
        "class_id": 5,
        "sample_count": 21,
        "connectivity": "8-neighbor",
        "center_band": "normalized radius <= 0.35",
        "edge_band": "normalized radius >= 0.70",
    }
    for key, expected in contract.items():
        if summary.get(key) != expected:
            raise ValueError(f"실험 계약 불일치: {key}")

    required_columns = (
        IDENTITY_COLUMNS
        | {"active_die_count", "defect_die_count"}
        | set(METRIC_COLUMNS)
    )
    if not required_columns.issubset(records.columns):
        raise ValueError(
            f"표본 CSV 필수 열이 없습니다: {sorted(required_columns - set(records.columns))}"
        )
    if len(records) != 21 or records["array_index"].duplicated().any():
        raise ValueError("표본 CSV는 중복 없는 Near-full 21행이어야 합니다.")
    if records.isna().any().any():
        raise ValueError("표본 CSV에 결측값이 있습니다.")
    numeric_columns = list(COUNT_COLUMNS | (set(METRIC_COLUMNS) - {"component_count_8"}))
    if not np.isfinite(records[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("표본 CSV에 유한하지 않은 수치가 있습니다.")
    for column in COUNT_COLUMNS:
        values = records[column].to_numpy(dtype=float)
        if np.any(values < 0) or not np.allclose(values, np.rint(values)):
            raise ValueError(f"{column}은 음이 아닌 정수여야 합니다.")
    if not (records["active_die_count"] > 0).all():
        raise ValueError("활성 die 수는 양수여야 합니다.")
    if not (records["defect_die_count"] <= records["active_die_count"]).all():
        raise ValueError("결함 die 수가 활성 die 수를 넘습니다.")
    bounded = [column for column in METRIC_COLUMNS if column != "component_count_8"]
    for column in bounded:
        if not records[column].between(0, 1).all():
            raise ValueError(f"{column} 범위가 0~1을 벗어났습니다.")

    assignments = pd.read_csv(assignments_path)
    expected = assignments.loc[
        assignments["split"].eq("test") & assignments["label_id"].eq(5),
        list(IDENTITY_COLUMNS),
    ]
    if len(expected) != 21 or expected["array_index"].duplicated().any():
        raise ValueError("원본 분할표의 Near-full test 계약이 21개가 아닙니다.")
    joined = records.merge(
        expected,
        on="array_index",
        suffixes=("_bundle", "_source"),
        validate="one_to_one",
    )
    if len(joined) != 21:
        raise ValueError("표본 CSV가 고정 test의 Near-full 표본과 다릅니다.")
    for column in IDENTITY_COLUMNS - {"array_index"}:
        if not joined[f"{column}_bundle"].astype(str).eq(
            joined[f"{column}_source"].astype(str)
        ).all():
            raise ValueError(f"{column}이 원본 분할표와 다릅니다.")

    saved_metrics = summary.get("metrics")
    if set(saved_metrics or {}) != set(METRIC_COLUMNS):
        raise ValueError("요약 JSON의 지표 구성이 잘못되었습니다.")
    for column in METRIC_COLUMNS:
        computed = {
            "mean": float(records[column].mean()),
            "median": float(records[column].median()),
            "min": float(records[column].min()),
            "max": float(records[column].max()),
        }
        for statistic, expected_value in computed.items():
            _close(
                float(saved_metrics[column][statistic]),
                expected_value,
                f"{column}.{statistic}",
            )

    return {
        "validation_passed": True,
        "bundle": str(bundle),
        "zip_sha256": sha256_file(bundle),
        "sample_count": len(records),
        "unique_lot_count": int(records["lot_name"].nunique()),
        "mean_defect_ratio": float(records["defect_ratio"].mean()),
        "mean_largest_component_fraction": float(
            records["largest_component_fraction"].mean()
        ),
        "mean_boundary_defect_coverage": float(
            records["boundary_defect_coverage"].mean()
        ),
        "mean_defect_bbox_fraction": float(records["defect_bbox_fraction"].mean()),
        "deployment_model_changed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            validate_bundle(arguments.bundle, arguments.assignments),
            ensure_ascii=False,
            indent=2,
        )
    )
