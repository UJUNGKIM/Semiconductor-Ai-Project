"""Validate the historical WM-811K Grad-CAM versus Integrated Gradients result ZIP.

This bundle predates the dashboard's Gradient SHAP explanation. It is evidence
about the two methods it compared, not a validation of Gradient SHAP.
"""

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

from hash_evidence import compare_recorded_text_hash


CLASS_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
METHODS = ["Grad-CAM", "Integrated Gradients"]
REQUIRED_FILES = {
    "xai_selected_test_samples.csv",
    "xai_method_records.csv",
    "xai_method_class_summary.csv",
    "xai_method_summary.json",
    "xai_method_comparison_dashboard.png",
    "README.md",
}
METRICS = [
    "active_attribution_mass",
    "active_attribution_lift",
    "top10_confidence_drop",
    "random10_confidence_drop_mean",
    "bottom10_confidence_drop",
    "top_vs_random_advantage",
    "top_vs_bottom_advantage",
    "top10_prediction_changed",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(info: zipfile.ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    return bool(info.filename) and not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename and ":" not in info.filename and not stat.S_ISLNK(mode)


def _close(actual: float, expected: float, name: str) -> None:
    if not math.isfinite(actual) or not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise ValueError(f"{name} 집계가 원자료와 일치하지 않습니다.")


def validate_bundle(
    bundle: Path, assignments_path: Path, checkpoint_path: Path | None = None
) -> dict[str, object]:
    if not bundle.is_file() or not assignments_path.is_file():
        raise FileNotFoundError(bundle if not bundle.is_file() else assignments_path)
    with zipfile.ZipFile(bundle) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos if not info.is_dir()]
        if len(infos) > 12 or sum(info.file_size for info in infos) > 10 * 1024 * 1024:
            raise ValueError("ZIP 항목 수 또는 압축 해제 크기가 제한을 넘습니다.")
        if len(names) != len(set(names)) or any(not _safe(info) for info in infos):
            raise ValueError("ZIP에 중복 또는 안전하지 않은 경로가 있습니다.")
        if set(names) != REQUIRED_FILES:
            raise ValueError(f"ZIP 파일 구성이 다릅니다: {sorted(set(names) ^ REQUIRED_FILES)}")
        summary = json.loads(archive.read("xai_method_summary.json").decode("utf-8"))
        selected = pd.read_csv(io.BytesIO(archive.read("xai_selected_test_samples.csv")))
        records = pd.read_csv(io.BytesIO(archive.read("xai_method_records.csv")))
        class_summary = pd.read_csv(io.BytesIO(archive.read("xai_method_class_summary.csv")))
        if archive.read("xai_method_comparison_dashboard.png")[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("대시보드 이미지가 유효한 PNG가 아닙니다.")

    contract = {
        "purpose": "post_selection_class_balanced_xai_comparison",
        "used_for_model_selection": False,
        "thresholds_retuned": False,
        "population_representative": False,
        "sample_count": 421,
        "per_class_cap": 50,
        "mask_fraction": 0.1,
        "random_repeats": 5,
        "integrated_gradients_steps": 24,
        "random_seed": 42,
        "device": "cuda",
    }
    for key, expected in contract.items():
        if summary.get(key) != expected:
            raise ValueError(f"실험 계약 불일치: {key}")
    if summary.get("schema_version") not in {1, 2}:
        raise ValueError("지원하지 않는 XAI 결과 버전입니다.")
    deployment_evidence_verified = False
    assignments_hash_check = None
    if summary["schema_version"] == 2:
        checkpoint_hash = summary.get("checkpoint_sha256")
        if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint_hash
        ):
            raise ValueError("체크포인트 SHA-256 형식이 잘못되었습니다.")
        assignments_hash_check = compare_recorded_text_hash(
            assignments_path,
            summary.get("assignments_sha256"),
            summary.get("assignments_canonical_sha256"),
        )
        if not assignments_hash_check["canonical_content_match"]:
            raise ValueError(
                "XAI 결과의 분할표 SHA-256이 다르고 줄바꿈만 다른 동일 내용으로도 "
                "확인되지 않습니다."
            )
        _close(float(summary.get("temperature", 0)), 1.047171711723721, "temperature")
        if checkpoint_path is not None:
            if summary.get("checkpoint_sha256") != sha256_file(checkpoint_path):
                raise ValueError("XAI 결과가 현재 배포 체크포인트와 다릅니다.")
            deployment_evidence_verified = True
    if set(summary.get("methods", {})) != set(METHODS):
        raise ValueError("설명 방법 구성이 잘못되었습니다.")

    required_selected = {"array_index", "label_id", "failure_type", "split", "selection_seed", "class_test_count", "class_sample_count"}
    if not required_selected.issubset(selected.columns) or len(selected) != 421 or selected["array_index"].duplicated().any():
        raise ValueError("선정 표본 파일의 열, 행 수 또는 고유키가 잘못되었습니다.")
    if not selected["split"].eq("test").all() or not selected["selection_seed"].eq(42).all():
        raise ValueError("선정 표본의 split 또는 seed가 잘못되었습니다.")
    expected_counts = {class_id: (21 if class_id == 5 else 50) for class_id in range(9)}
    if selected.groupby("label_id").size().to_dict() != expected_counts:
        raise ValueError("클래스별 표본 수가 고정 계약과 다릅니다.")

    assignments = pd.read_csv(assignments_path).set_index("array_index")
    joined = selected.join(assignments[["label_id", "failure_type", "split"]], on="array_index", rsuffix="_source", validate="one_to_one")
    for column in ("label_id", "failure_type", "split"):
        if not joined[column].astype(str).eq(joined[f"{column}_source"].astype(str)).all():
            raise ValueError(f"선정 표본의 {column}이 원본 분할표와 다릅니다.")

    required_records = {"method", "array_index", "true_class", "predicted_class", "correct", "baseline_confidence", *METRICS}
    if not required_records.issubset(records.columns) or len(records) != 842:
        raise ValueError("설명 기록의 열 또는 행 수가 잘못되었습니다.")
    if records.duplicated(["method", "array_index"]).any() or set(records["method"]) != set(METHODS):
        raise ValueError("설명 기록의 방법·표본 키가 중복되거나 누락되었습니다.")
    for method in METHODS:
        if set(records.loc[records["method"].eq(method), "array_index"]) != set(selected["array_index"]):
            raise ValueError(f"{method} 표본 구성이 선정표와 다릅니다.")
    numeric_columns = ["baseline_confidence", *METRICS]
    if not np.isfinite(records[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("설명 기록에 유한하지 않은 수치가 있습니다.")
    tolerance = 1e-6
    confidence_in_range = records["baseline_confidence"].between(-tolerance, 1 + tolerance).all()
    mass_in_range = records["active_attribution_mass"].between(-tolerance, 1 + tolerance).all()
    if not confidence_in_range or not mass_in_range:
        raise ValueError("확률 또는 attribution mass 범위가 잘못되었습니다.")

    expected_class_rows = {(method, name) for method in METHODS for name in CLASS_NAMES}
    if len(class_summary) != 18 or set(map(tuple, class_summary[["method", "true_class"]].to_numpy())) != expected_class_rows:
        raise ValueError("클래스 요약 구성이 잘못되었습니다.")
    recomputed = records.groupby(["method", "true_class"], sort=False)[METRICS].mean().reset_index()
    merged = class_summary.merge(recomputed, on=["method", "true_class"], suffixes=("_saved", "_computed"), validate="one_to_one")
    for metric in METRICS:
        if not np.allclose(merged[f"{metric}_saved"], merged[f"{metric}_computed"], rtol=0, atol=1e-12):
            raise ValueError(f"클래스별 {metric} 요약이 원자료와 다릅니다.")

    for method in METHODS:
        part = records.loc[records["method"].eq(method)]
        class_part = class_summary.loc[class_summary["method"].eq(method)]
        saved = summary["methods"][method]
        _close(float(saved["macro_top_vs_random_advantage"]), float(class_part["top_vs_random_advantage"].mean()), f"{method} macro top-random")
        _close(float(saved["macro_top_vs_bottom_advantage"]), float(class_part["top_vs_bottom_advantage"].mean()), f"{method} macro top-bottom")
        _close(float(saved["macro_active_attribution_lift"]), float(class_part["active_attribution_lift"].mean()), f"{method} macro active lift")
        _close(float(saved["top_exceeds_random_rate"]), float((part["top_vs_random_advantage"] > 0).mean()), f"{method} top>random")
        _close(float(saved["top_exceeds_bottom_rate"]), float((part["top_vs_bottom_advantage"] > 0).mean()), f"{method} top>bottom")
        interval = np.asarray(saved["top_vs_random_95ci"], dtype=float)
        if interval.shape != (2,) or not np.isfinite(interval).all() or not interval[0] <= saved["macro_top_vs_random_advantage"] <= interval[1]:
            raise ValueError(f"{method} bootstrap 구간이 잘못되었습니다.")

    paired = records.pivot(index="array_index", columns="method", values="top_vs_random_advantage")
    paired_delta = float((paired["Integrated Gradients"] - paired["Grad-CAM"]).mean())
    _close(float(summary["paired_ig_minus_gradcam"]), paired_delta, "paired IG-Grad-CAM")
    paired_interval = np.asarray(summary["paired_ig_minus_gradcam_95ci"], dtype=float)
    if paired_interval.shape != (2,) or not paired_interval[0] <= paired_delta <= paired_interval[1]:
        raise ValueError("paired bootstrap 구간이 잘못되었습니다.")

    return {
        "validation_passed": True,
        "deployment_evidence_verified": deployment_evidence_verified,
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "bundle": str(bundle),
        "zip_sha256": sha256_file(bundle),
        "sample_count": len(selected),
        "record_count": len(records),
        "methods": METHODS,
        "gradcam_macro_top_vs_random": summary["methods"]["Grad-CAM"]["macro_top_vs_random_advantage"],
        "integrated_gradients_macro_top_vs_random": summary["methods"]["Integrated Gradients"]["macro_top_vs_random_advantage"],
        "paired_ig_minus_gradcam": summary["paired_ig_minus_gradcam"],
        "deployment_model_changed": False,
        "evidence_scope": "historical_integrated_gradients_vs_gradcam_comparison",
        "validates_gradient_shap": False,
        "assignments_hash_check": assignments_hash_check,
        "warnings": [] if assignments_hash_check is None or "warning" not in assignments_hash_check else [assignments_hash_check["warning"]],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(validate_bundle(arguments.bundle, arguments.assignments, arguments.checkpoint), ensure_ascii=False, indent=2))
