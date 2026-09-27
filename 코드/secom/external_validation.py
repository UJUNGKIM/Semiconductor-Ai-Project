"""Pre-registered prospective validation utilities for independently labelled SECOM lots."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from model_registry import load_verified_bundles, validate_registry
from safety_policy import model_alert_mask


FEATURE_COLUMNS = [f"feature_{index}" for index in range(590)]
REQUIRED_METADATA_COLUMNS = ["sample_id", "lot_id", "captured_at", "label"]
OPTIONAL_METADATA_COLUMNS = ["equipment_id", "source_system"]
REQUIRED_DECLARATIONS = (
    "prospective_collection",
    "independent_source",
    "labels_finalized_before_scoring",
    "model_outputs_hidden_from_labelers",
    "data_collected_after_protocol_freeze",
)
MAX_EXTERNAL_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_EXTERNAL_ROWS = 100_000
MAX_DECLARATION_BYTES = 64 * 1024


def canonical_json_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _row_fingerprint(values: np.ndarray) -> str:
    tokens = ["NA" if pd.isna(value) else format(float(value), ".17g") for value in values]
    return hashlib.sha256("\x1f".join(tokens).encode("utf-8")).hexdigest().upper()


def row_fingerprints(sensors: pd.DataFrame) -> list[str]:
    return [_row_fingerprint(row) for row in sensors[FEATURE_COLUMNS].to_numpy()]


def build_protocol(project: Path) -> dict:
    registry_path = project / "결과물" / "secom" / "model_failover" / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry, project)
    bundles = load_verified_bundles(registry_path, project)
    profile = "balanced_f2"
    model_hashes = {
        str(entry["model_name"]): str(entry["artifact_sha256"]).upper()
        for entry in registry["models"]
    }
    thresholds = {
        name: float(bundle["operating_thresholds"][profile])
        for name, bundle in bundles.items()
    }
    core = {
        "protocol_version": 1,
        "registered_at": "2026-09-17T00:00:00+09:00",
        "scope": "independent_future_equipment_period_or_lot",
        "profile": profile,
        "decision_rule": "CatBoost OR XGBoost fixed-threshold alert",
        "expected_sensor_columns": FEATURE_COLUMNS,
        "label_contract": {"normal": 0, "defect": 1},
        "model_artifact_sha256": model_hashes,
        "operating_thresholds": thresholds,
        "minimum_evidence": {"rows": 500, "distinct_lots": 3, "defects": 30},
        "acceptance": {
            "recall_wilson_95_lower_min": 0.60,
            "false_positive_rate_wilson_95_upper_max": 0.15,
            "review_rate_wilson_95_upper_max": 0.25,
        },
        "selection_prohibitions": [
            "model_selection",
            "threshold_retuning",
            "feature_selection",
            "calibration_refit",
            "synthetic_augmentation_fit",
        ],
        "raw_rows_persisted_in_results": False,
    }
    return {**core, "protocol_id": canonical_json_hash(core)[:20]}


def validate_declarations(declarations: Mapping[str, object]) -> None:
    expected = set(REQUIRED_DECLARATIONS)
    actual = set(declarations)
    missing_keys = sorted(expected - actual)
    unknown_keys = sorted(actual - expected)
    if missing_keys or unknown_keys:
        raise ValueError(
            f"외부 전향 검증 선언 키가 정확하지 않습니다: "
            f"누락={missing_keys}, 알 수 없음={unknown_keys}"
        )
    missing = [name for name in REQUIRED_DECLARATIONS if declarations.get(name) is not True]
    if missing:
        raise ValueError(f"외부 전향 검증 선언이 참이 아닙니다: {missing}")


def parse_declarations_bytes(payload: bytes) -> dict:
    """Parse a small, exact declaration document and require every claim to be true."""
    if not payload:
        raise ValueError("독립성 선언 JSON이 비어 있습니다.")
    if len(payload) > MAX_DECLARATION_BYTES:
        raise ValueError("독립성 선언 JSON은 64KB를 초과할 수 없습니다.")
    try:
        value = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("독립성 선언 JSON 형식이 올바르지 않습니다.") from error
    if not isinstance(value, dict):
        raise ValueError("독립성 선언 JSON의 최상위 값은 객체여야 합니다.")
    validate_declarations(value)
    return value


def parse_external_csv_bytes(payload: bytes) -> pd.DataFrame:
    """Parse an uncompressed CSV with conservative memory and row-count limits."""
    if not payload:
        raise ValueError("외부 lot CSV가 비어 있습니다.")
    if len(payload) > MAX_EXTERNAL_UPLOAD_BYTES:
        raise ValueError("외부 lot CSV는 50MB를 초과할 수 없습니다.")
    try:
        frame = pd.read_csv(BytesIO(payload), low_memory=False)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError("외부 lot CSV 형식이 올바르지 않습니다.") from error
    if frame.empty:
        raise ValueError("외부 lot CSV에는 데이터 행이 필요합니다.")
    if len(frame) > MAX_EXTERNAL_ROWS:
        raise ValueError(f"외부 lot CSV는 {MAX_EXTERNAL_ROWS:,}행을 초과할 수 없습니다.")
    return frame


def summary_json_bytes(summary: Mapping[str, object]) -> bytes:
    return json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def _csv_safe_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    if value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def per_lot_csv_bytes(per_lot: pd.DataFrame) -> bytes:
    """Export aggregate lot metrics while preventing spreadsheet formula execution."""
    safe = per_lot.copy()
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(_csv_safe_text)
    return safe.to_csv(index=False).encode("utf-8-sig")


def validate_external_frame(frame: pd.DataFrame, protocol: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(REQUIRED_METADATA_COLUMNS + FEATURE_COLUMNS)
    allowed = required | set(OPTIONAL_METADATA_COLUMNS)
    missing = sorted(required - set(frame.columns))
    unknown = sorted(set(frame.columns) - allowed)
    if missing:
        raise ValueError(f"외부 검증 필수 열 누락: {missing[:10]}")
    if unknown:
        raise ValueError(f"외부 검증 알 수 없는 열: {unknown[:10]}")
    metadata = frame[REQUIRED_METADATA_COLUMNS + [c for c in OPTIONAL_METADATA_COLUMNS if c in frame]].copy()
    if metadata["sample_id"].isna().any() or metadata["sample_id"].astype(str).str.strip().eq("").any():
        raise ValueError("sample_id는 비어 있을 수 없습니다.")
    if metadata["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id는 중복될 수 없습니다.")
    if metadata["lot_id"].isna().any() or metadata["lot_id"].astype(str).str.strip().eq("").any():
        raise ValueError("lot_id는 비어 있을 수 없습니다.")
    timestamp_text = metadata["captured_at"].astype(str)
    if not timestamp_text.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True).all():
        raise ValueError("captured_at에는 UTC Z 또는 명시적 시간대 오프셋이 필요합니다.")
    captured = pd.to_datetime(timestamp_text, utc=True, errors="raise")
    freeze_time = pd.Timestamp(str(protocol["registered_at"])).tz_convert("UTC")
    if (captured <= freeze_time).any():
        raise ValueError("모든 외부 검증 행은 프로토콜 등록 이후에 수집되어야 합니다.")
    labels = pd.to_numeric(metadata["label"], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("label은 결측 없는 0(정상)/1(불량)이어야 합니다.")
    metadata["label"] = labels.astype(int)
    metadata["captured_at"] = captured
    sensors = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid = sensors.isna() & frame[FEATURE_COLUMNS].notna()
    if invalid.any().any():
        row, column = np.argwhere(invalid.to_numpy())[0]
        raise ValueError(f"숫자가 아닌 센서 값: row={int(row)}, {FEATURE_COLUMNS[int(column)]}")
    if np.isinf(sensors.to_numpy(dtype=float)).any():
        raise ValueError("센서 값에는 무한대를 사용할 수 없습니다.")
    if sensors.isna().all(axis=1).any():
        raise ValueError("590개 센서가 모두 결측인 행은 사용할 수 없습니다.")
    return sensors, metadata


def known_data_overlap_count(sensors: pd.DataFrame, project: Path) -> int:
    raw = pd.read_csv(
        project / "데이터" / "SECOM 데이터셋" / "raw" / "secom.data",
        sep=r"\s+",
        header=None,
        names=FEATURE_COLUMNS,
    )
    known = set(row_fingerprints(raw))
    return sum(fingerprint in known for fingerprint in row_fingerprints(sensors))


def _metric_summary(labels: np.ndarray, alerts: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    alerts = np.asarray(alerts, dtype=bool)
    tp = int(np.sum((labels == 1) & alerts))
    fn = int(np.sum((labels == 1) & ~alerts))
    fp = int(np.sum((labels == 0) & alerts))
    tn = int(np.sum((labels == 0) & ~alerts))
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    review_rate = (tp + fp) / max(len(labels), 1)
    recall_ci = wilson_interval(tp, tp + fn)
    fpr_ci = wilson_interval(fp, fp + tn)
    review_ci = wilson_interval(tp + fp, len(labels))
    return {
        "rows": int(len(labels)),
        "defects": int(tp + fn),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall": recall,
        "precision": precision,
        "false_positive_rate": fpr,
        "review_rate": review_rate,
        "recall_wilson_95": {"lower": recall_ci[0], "upper": recall_ci[1]},
        "false_positive_rate_wilson_95": {"lower": fpr_ci[0], "upper": fpr_ci[1]},
        "review_rate_wilson_95": {"lower": review_ci[0], "upper": review_ci[1]},
    }


def evaluate_external_frame(
    frame: pd.DataFrame,
    declarations: Mapping[str, object],
    project: Path,
    protocol: Mapping[str, object] | None = None,
) -> tuple[dict, pd.DataFrame]:
    protocol = dict(protocol or build_protocol(project))
    validate_declarations(declarations)
    sensors, metadata = validate_external_frame(frame, protocol)
    registry_path = project / "결과물" / "secom" / "model_failover" / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    bundles = load_verified_bundles(registry_path, project)
    thresholds = {key: float(value) for key, value in protocol["operating_thresholds"].items()}
    model_alerts = {}
    for name, bundle in bundles.items():
        prepared = bundle["selector"].transform(
            bundle["imputer"].transform(sensors[bundle["input_features"]])
        )
        probability = bundle["model"].predict_proba(prepared)[:, 1]
        model_alerts[name] = model_alert_mask(probability, thresholds[name])
    alerts = np.logical_or(model_alerts["CatBoost"], model_alerts["XGBoost"])
    labels = metadata["label"].to_numpy(dtype=int)
    overall = _metric_summary(labels, alerts)
    overlap_count = known_data_overlap_count(sensors, project)
    per_lot = []
    lot_values = metadata["lot_id"].astype(str).to_numpy()
    for lot_id in sorted(set(lot_values)):
        mask = lot_values == lot_id
        per_lot.append({"lot_id": lot_id, **_metric_summary(labels[mask], alerts[mask])})
    per_lot_frame = pd.DataFrame(per_lot)
    minimum = protocol["minimum_evidence"]
    acceptance = protocol["acceptance"]
    gates = {
        "minimum_rows": overall["rows"] >= int(minimum["rows"]),
        "minimum_distinct_lots": len(per_lot) >= int(minimum["distinct_lots"]),
        "minimum_defects": overall["defects"] >= int(minimum["defects"]),
        "no_known_secom_row_overlap": overlap_count == 0,
        "recall_lower_bound": overall["recall_wilson_95"]["lower"]
        >= float(acceptance["recall_wilson_95_lower_min"]),
        "false_positive_upper_bound": overall["false_positive_rate_wilson_95"]["upper"]
        <= float(acceptance["false_positive_rate_wilson_95_upper_max"]),
        "review_rate_upper_bound": overall["review_rate_wilson_95"]["upper"]
        <= float(acceptance["review_rate_wilson_95_upper_max"]),
    }
    registry_hashes = {
        str(entry["model_name"]): str(entry["artifact_sha256"]).upper()
        for entry in registry["models"]
    }
    dataset_digest = canonical_json_hash(
        {
            "sample_id": metadata["sample_id"].astype(str).tolist(),
            "row_fingerprints": row_fingerprints(sensors),
            "labels": labels.tolist(),
        }
    )
    summary = {
        "summary_version": 1,
        "validation_status": "PASS" if all(gates.values()) else "BLOCKED",
        "protocol_id": protocol["protocol_id"],
        "prospective_collection": True,
        "independent_source": True,
        "labels_finalized_before_scoring": True,
        "model_outputs_hidden_from_labelers": True,
        "data_collected_after_protocol_freeze": True,
        "model_or_threshold_selection_used": False,
        "thresholds_retuned": False,
        "raw_rows_persisted": False,
        "dataset_digest": dataset_digest,
        "known_secom_exact_row_overlap_count": overlap_count,
        "model_artifact_sha256": registry_hashes,
        "operating_thresholds": thresholds,
        "distinct_lots": len(per_lot),
        "overall": overall,
        "gates": gates,
    }
    return summary, per_lot_frame


def validated_summary_is_release_evidence(summary: Mapping[str, object], protocol: Mapping[str, object]) -> bool:
    try:
        overall = summary["overall"]
        minimum = protocol["minimum_evidence"]
        acceptance = protocol["acceptance"]
        recomputed_gates = {
            "minimum_rows": int(overall["rows"]) >= int(minimum["rows"]),
            "minimum_distinct_lots": int(summary["distinct_lots"])
            >= int(minimum["distinct_lots"]),
            "minimum_defects": int(overall["defects"]) >= int(minimum["defects"]),
            "no_known_secom_row_overlap": int(summary["known_secom_exact_row_overlap_count"]) == 0,
            "recall_lower_bound": float(overall["recall_wilson_95"]["lower"])
            >= float(acceptance["recall_wilson_95_lower_min"]),
            "false_positive_upper_bound": float(
                overall["false_positive_rate_wilson_95"]["upper"]
            ) <= float(acceptance["false_positive_rate_wilson_95_upper_max"]),
            "review_rate_upper_bound": float(overall["review_rate_wilson_95"]["upper"])
            <= float(acceptance["review_rate_wilson_95_upper_max"]),
        }
    except (KeyError, TypeError, ValueError):
        return False
    gates = summary.get("gates")
    dataset_digest = str(summary.get("dataset_digest", ""))
    return bool(
        summary.get("validation_status") == "PASS"
        and summary.get("protocol_id") == protocol.get("protocol_id")
        and summary.get("prospective_collection") is True
        and summary.get("independent_source") is True
        and summary.get("labels_finalized_before_scoring") is True
        and summary.get("model_outputs_hidden_from_labelers") is True
        and summary.get("data_collected_after_protocol_freeze") is True
        and summary.get("model_or_threshold_selection_used") is False
        and summary.get("thresholds_retuned") is False
        and summary.get("raw_rows_persisted") is False
        and summary.get("model_artifact_sha256") == protocol.get("model_artifact_sha256")
        and summary.get("operating_thresholds") == protocol.get("operating_thresholds")
        and len(dataset_digest) == 64
        and all(character in "0123456789ABCDEF" for character in dataset_digest)
        and isinstance(gates, Mapping)
        and dict(gates) == recomputed_gates
        and all(recomputed_gates.values())
    )
