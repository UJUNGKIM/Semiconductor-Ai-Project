"""Validate WM-811K Gradient SHAP evidence produced by evaluate_wm811k_xai_methods.py.

The summary is not trusted on its own: per-wafer additivity residuals, pass
flags, class means, pass rates, and residual quantiles are recomputed from the
raw records, and provenance is checked against the deployed checkpoint and the
split table (exact bytes and newline-canonical content are reported apart).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from hash_evidence import compare_recorded_text_hash


METHODS = ("Grad-CAM", "Integrated Gradients", "Gradient SHAP")
REQUIRED_FILES = (
    "xai_selected_test_samples.csv",
    "xai_method_records.csv",
    "xai_method_class_summary.csv",
    "xai_method_summary.json",
    "xai_method_comparison_dashboard.png",
    "README.md",
)
LEGACY_METRICS = [
    "active_attribution_mass",
    "active_attribution_lift",
    "top10_confidence_drop",
    "random10_confidence_drop_mean",
    "bottom10_confidence_drop",
    "top_vs_random_advantage",
    "top_vs_bottom_advantage",
    "top10_prediction_changed",
]
FLIP_METRICS = [
    "flip_top10_confidence_drop",
    "flip_random10_confidence_drop_mean",
    "flip_bottom10_confidence_drop",
    "flip_top_vs_random_advantage",
    "flip_top_vs_bottom_advantage",
]
GRADIENT_SHAP_COLUMNS = [
    "gradient_shap_samples",
    "selected_logit_delta",
    "raw_attribution_sum",
    "raw_additivity_residual",
    "relative_additivity_residual",
    "additivity_tolerance",
    "additivity_check_passed",
]
CONTRACT = {
    "purpose": "post_selection_class_balanced_xai_comparison",
    "used_for_model_selection": False,
    "thresholds_retuned": False,
    "population_representative": False,
    "schema_version": 3,
}
GRADIENT_SHAP_CONTRACT = {
    "baseline": "same_geometry_all_active_dies_normal",
    "estimator": "stratified_expected_gradients",
    "rescaled_to_logit_delta": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(actual: float, expected: float, name: str, tolerance: float = 1e-9) -> None:
    if not (math.isfinite(actual) and math.isfinite(expected)) or abs(actual - expected) > tolerance:
        raise ValueError(f"{name} 집계가 원자료와 일치하지 않습니다: {actual} != {expected}")


def validate_gradient_shap_results(
    result_dir: Path, assignments_path: Path, checkpoint_path: Path | None = None
) -> dict[str, object]:
    missing = [name for name in REQUIRED_FILES if not (result_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Gradient SHAP 검증 결과 파일 누락: {missing}")
    summary = json.loads((result_dir / "xai_method_summary.json").read_text(encoding="utf-8"))
    for key, expected in CONTRACT.items():
        if summary.get(key) != expected:
            raise ValueError(f"실험 계약 불일치: {key}")
    if tuple(summary.get("methods_compared", ())) != METHODS or set(summary.get("methods", {})) != set(METHODS):
        raise ValueError("설명 방법 구성이 잘못되었습니다.")
    shap_summary = summary.get("gradient_shap", {})
    for key, expected in GRADIENT_SHAP_CONTRACT.items():
        if shap_summary.get(key) != expected:
            raise ValueError(f"Gradient SHAP 계약 불일치: {key}")
    relative_tolerance = float(shap_summary["relative_tolerance"])
    absolute_tolerance = float(shap_summary["absolute_tolerance"])

    selected = pd.read_csv(result_dir / "xai_selected_test_samples.csv")
    records = pd.read_csv(result_dir / "xai_method_records.csv")
    class_summary = pd.read_csv(result_dir / "xai_method_class_summary.csv")
    if (result_dir / "xai_method_comparison_dashboard.png").read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("대시보드 이미지가 유효한 PNG가 아닙니다.")

    sample_count = int(summary["sample_count"])
    if len(selected) != sample_count or selected["array_index"].duplicated().any():
        raise ValueError("선정 표본 수 또는 고유키가 잘못되었습니다.")
    if not selected["split"].eq("test").all():
        raise ValueError("선정 표본은 모두 test split이어야 합니다.")
    assignments = pd.read_csv(assignments_path).set_index("array_index")
    joined = selected.join(
        assignments[["label_id", "failure_type", "split"]],
        on="array_index",
        rsuffix="_source",
        validate="one_to_one",
    )
    for column in ("label_id", "failure_type", "split"):
        if not joined[column].astype(str).eq(joined[f"{column}_source"].astype(str)).all():
            raise ValueError(f"선정 표본의 {column}이 원본 분할표와 다릅니다.")

    if len(records) != sample_count * len(METHODS) or records.duplicated(["method", "array_index"]).any():
        raise ValueError("설명 기록의 행 수 또는 방법·표본 키가 잘못되었습니다.")
    for method in METHODS:
        if set(records.loc[records["method"].eq(method), "array_index"]) != set(selected["array_index"]):
            raise ValueError(f"{method} 표본 구성이 선정표와 다릅니다.")
    if not np.isfinite(records[["baseline_confidence", *LEGACY_METRICS]].to_numpy(dtype=float)).all():
        raise ValueError("설명 기록에 유한하지 않은 수치가 있습니다.")

    shap_records = records.loc[records["method"].eq("Gradient SHAP")].copy()
    if shap_records[GRADIENT_SHAP_COLUMNS].isna().any().any():
        raise ValueError("Gradient SHAP 가산성 기록이 비어 있습니다.")
    residual = shap_records["raw_attribution_sum"] - shap_records["selected_logit_delta"]
    if not np.allclose(residual, shap_records["raw_additivity_residual"], rtol=0, atol=1e-9):
        raise ValueError("원시 가산성 잔차가 원시 합계와 로짓 변화로 재계산되지 않습니다.")
    tolerance = np.maximum(absolute_tolerance, relative_tolerance * shap_records["selected_logit_delta"].abs())
    if not np.allclose(tolerance, shap_records["additivity_tolerance"], rtol=0, atol=1e-12):
        raise ValueError("가산성 허용오차가 계약과 다릅니다.")
    recomputed_pass = residual.abs() <= tolerance
    if not recomputed_pass.eq(shap_records["additivity_check_passed"].astype(bool)).all():
        raise ValueError("가산성 통과 여부가 원시 잔차와 일치하지 않습니다.")
    delta = shap_records["selected_logit_delta"].abs()
    relative = np.where(delta > 0, residual.abs() / delta.where(delta > 0, 1.0), np.where(residual == 0, 0.0, np.inf))
    if not np.allclose(relative, shap_records["relative_additivity_residual"], rtol=1e-9, atol=1e-12):
        raise ValueError("상대 가산성 잔차가 원자료와 일치하지 않습니다.")
    if not shap_records["gradient_shap_samples"].between(
        int(shap_summary["initial_samples"]), int(shap_summary["max_samples"])
    ).all():
        raise ValueError("Gradient SHAP 표본 수가 설정 범위를 벗어났습니다.")
    samples_used = {
        str(key): int(value)
        for key, value in shap_records["gradient_shap_samples"].astype(int).value_counts().sort_index().items()
    }
    if samples_used != shap_summary["samples_used_counts"]:
        raise ValueError("Gradient SHAP 표본 수 분포가 원자료와 다릅니다.")
    _close(float(shap_summary["additivity_pass_count"]), float(recomputed_pass.sum()), "가산성 통과 수", 0)
    _close(float(shap_summary["additivity_pass_rate"]), float(recomputed_pass.mean()), "가산성 통과율")
    _close(float(shap_summary["relative_residual_median"]), float(np.median(relative)), "상대 잔차 중앙값")
    _close(float(shap_summary["relative_residual_p90"]), float(np.quantile(relative, 0.9)), "상대 잔차 90%")
    _close(float(shap_summary["relative_residual_max"]), float(relative.max()), "상대 잔차 최대")

    recomputed = records.groupby(["method", "true_class"], sort=False)[LEGACY_METRICS + FLIP_METRICS].mean().reset_index()
    merged = class_summary.merge(recomputed, on=["method", "true_class"], suffixes=("_saved", "_computed"), validate="one_to_one")
    if len(merged) != len(class_summary) or len(class_summary) != len(METHODS) * selected["label_id"].nunique():
        raise ValueError("클래스 요약 구성이 잘못되었습니다.")
    for metric in LEGACY_METRICS + FLIP_METRICS:
        saved = merged[f"{metric}_saved"].to_numpy(dtype=float)
        computed = merged[f"{metric}_computed"].to_numpy(dtype=float)
        if not np.allclose(saved, computed, rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError(f"클래스별 {metric} 요약이 원자료와 다릅니다.")
    for method in METHODS:
        part = records.loc[records["method"].eq(method)]
        class_part = class_summary.loc[class_summary["method"].eq(method)]
        saved = summary["methods"][method]
        _close(float(saved["macro_top_vs_random_advantage"]), float(class_part["top_vs_random_advantage"].mean()), f"{method} macro top-random")
        _close(float(saved["top_exceeds_random_rate"]), float((part["top_vs_random_advantage"] > 0).mean()), f"{method} top>random")
        flip = saved["defect_flip"]
        flip_part = part.dropna(subset=["flip_top_vs_random_advantage"])
        _close(float(flip["evaluated_wafers"]), float(len(flip_part)), f"{method} defect-flip rows", 0)
        _close(float(flip["macro_top_vs_random_advantage"]), float(class_part["flip_top_vs_random_advantage"].mean(skipna=True)), f"{method} defect-flip macro")
        _close(float(flip["top_exceeds_random_rate"]), float((flip_part["flip_top_vs_random_advantage"] > 0).mean()), f"{method} defect-flip top>random")
        interval = np.asarray(flip["top_vs_random_95ci"], dtype=float)
        if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] > interval[1]:
            raise ValueError(f"{method} 결함 교란 bootstrap 구간이 잘못되었습니다.")

    checkpoint_matches = None
    if checkpoint_path is not None:
        checkpoint_matches = summary.get("checkpoint_sha256") == sha256_file(checkpoint_path)
        if not checkpoint_matches:
            raise ValueError("Gradient SHAP 검증 결과가 현재 배포 체크포인트와 다릅니다.")
    assignments_check = compare_recorded_text_hash(
        assignments_path,
        summary.get("assignments_sha256"),
        summary.get("assignments_canonical_sha256"),
    )
    if not assignments_check["canonical_content_match"]:
        raise ValueError("Gradient SHAP 검증 결과의 분할표 내용이 현재 분할표와 다릅니다.")

    return {
        "validation_passed": True,
        "evidence_scope": "gradient_shap_post_selection_validation",
        "validates_gradient_shap": True,
        "deployment_evidence_verified": bool(checkpoint_matches),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "assignments_hash_check": assignments_check,
        "warnings": [assignments_check["warning"]] if "warning" in assignments_check else [],
        "device": summary.get("device"),
        "sample_count": sample_count,
        "record_count": int(len(records)),
        "gradient_shap": {
            "additivity_pass_count": int(recomputed_pass.sum()),
            "additivity_pass_rate": float(recomputed_pass.mean()),
            "relative_residual_median": float(np.median(relative)),
            "relative_residual_p90": float(np.quantile(relative, 0.9)),
            "relative_residual_max": float(relative.max()),
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "samples_used_counts": shap_summary["samples_used_counts"],
        },
        "defect_flip": {
            method: summary["methods"][method]["defect_flip"] for method in METHODS
        },
        "die_removal": {
            method: {
                key: summary["methods"][method][key]
                for key in ("macro_top_vs_random_advantage", "top_vs_random_95ci", "top_exceeds_random_rate")
            }
            for method in METHODS
        },
        "paired_gradient_shap_minus_integrated_gradients_defect_flip": summary[
            "paired_gradient_shap_minus_integrated_gradients_defect_flip"
        ],
        "paired_gradient_shap_minus_integrated_gradients_defect_flip_95ci": summary[
            "paired_gradient_shap_minus_integrated_gradients_defect_flip_95ci"
        ],
        "paired_gradient_shap_minus_gradcam_defect_flip": summary[
            "paired_gradient_shap_minus_gradcam_defect_flip"
        ],
        "paired_gradient_shap_minus_gradcam_defect_flip_95ci": summary[
            "paired_gradient_shap_minus_gradcam_defect_flip_95ci"
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(validate_gradient_shap_results(arguments.result_dir, arguments.assignments, arguments.checkpoint), ensure_ascii=False, indent=2))
