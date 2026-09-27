"""Validation contract for owner-supplied SECOM sensor semantics."""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import re
from pathlib import Path
from typing import Mapping

import pandas as pd


FEATURES = [f"feature_{index}" for index in range(590)]
REQUIRED_COLUMNS = [
    "feature",
    "sensor_name",
    "unit",
    "process_step",
    "equipment_scope",
    "description",
    "valid_min",
    "valid_max",
    "owner",
    "source_reference",
    "verification_status",
    "verified_at",
]
STRING_COLUMNS = [
    "feature",
    "sensor_name",
    "unit",
    "process_step",
    "equipment_scope",
    "description",
    "owner",
    "source_reference",
    "verification_status",
    "verified_at",
]
VERIFICATION_STATUSES = {"UNMAPPED", "PROVISIONAL", "VERIFIED"}
MAX_DICTIONARY_UPLOAD_BYTES = 5 * 1024 * 1024
PLACEHOLDERS = {
    "unknown",
    "tbd",
    "todo",
    "n/a",
    "na",
    "임시",
    "미정",
    "모름",
    "placeholder",
}
PROVISIONAL_REQUIRED = ["sensor_name", "process_step", "description", "source_reference"]
VERIFIED_REQUIRED = [
    "sensor_name",
    "unit",
    "process_step",
    "equipment_scope",
    "description",
    "owner",
    "source_reference",
    "verified_at",
]


def empty_dictionary_template() -> pd.DataFrame:
    frame = pd.DataFrame({"feature": FEATURES})
    for column in REQUIRED_COLUMNS[1:]:
        frame[column] = ""
    frame["verification_status"] = "UNMAPPED"
    return frame[REQUIRED_COLUMNS]


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _canonical_records(frame: pd.DataFrame) -> list[dict]:
    records = []
    for record in frame[REQUIRED_COLUMNS].to_dict(orient="records"):
        clean = {}
        for key, value in record.items():
            if pd.isna(value):
                clean[key] = None if key in {"valid_min", "valid_max"} else ""
            elif key in {"valid_min", "valid_max"}:
                clean[key] = float(value)
            else:
                clean[key] = str(value)
        records.append(clean)
    return records


def dictionary_digest(frame: pd.DataFrame) -> str:
    encoded = json.dumps(
        _canonical_records(frame),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def validate_sensor_dictionary(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    unknown = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))
    if missing:
        raise ValueError(f"센서 사전 필수 열 누락: {missing}")
    if unknown:
        raise ValueError(f"센서 사전 알 수 없는 열: {unknown}")
    if len(frame) != len(FEATURES):
        raise ValueError(f"센서 사전은 정확히 590행이어야 합니다: {len(frame)}")
    output = frame[REQUIRED_COLUMNS].copy()
    for column in STRING_COLUMNS:
        output[column] = output[column].map(_clean_text)
    if output["feature"].duplicated().any():
        raise ValueError("센서 사전 feature가 중복되었습니다.")
    if set(output["feature"]) != set(FEATURES):
        missing_features = sorted(set(FEATURES) - set(output["feature"]))
        unknown_features = sorted(set(output["feature"]) - set(FEATURES))
        raise ValueError(
            f"feature_0~feature_589 계약 불일치: missing={missing_features[:5]}, "
            f"unknown={unknown_features[:5]}"
        )
    output = output.set_index("feature").loc[FEATURES].reset_index()
    statuses = set(output["verification_status"])
    if not statuses <= VERIFICATION_STATUSES:
        raise ValueError(f"허용되지 않은 verification_status: {sorted(statuses - VERIFICATION_STATUSES)}")
    for column in ("valid_min", "valid_max"):
        original = output[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = numeric.isna() & original.map(_clean_text).ne("")
        if invalid.any():
            feature = output.loc[invalid, "feature"].iloc[0]
            raise ValueError(f"{feature}의 {column}이 숫자가 아닙니다.")
        output[column] = numeric
    one_sided_range = output["valid_min"].isna() ^ output["valid_max"].isna()
    if one_sided_range.any():
        raise ValueError("valid_min과 valid_max는 둘 다 입력하거나 둘 다 비워야 합니다.")
    invalid_range = (
        output["valid_min"].notna()
        & output["valid_max"].notna()
        & (output["valid_min"] >= output["valid_max"])
    )
    if invalid_range.any():
        feature = output.loc[invalid_range, "feature"].iloc[0]
        raise ValueError(f"{feature}의 valid_min은 valid_max보다 작아야 합니다.")

    for index, row in output.iterrows():
        status = row["verification_status"]
        required = VERIFIED_REQUIRED if status == "VERIFIED" else (
            PROVISIONAL_REQUIRED if status == "PROVISIONAL" else []
        )
        absent = [column for column in required if not str(row[column]).strip()]
        if absent:
            raise ValueError(f"{row['feature']} {status} 필수값 누락: {absent}")
        if status in {"PROVISIONAL", "VERIFIED"}:
            for column in required:
                if str(row[column]).strip().lower() in PLACEHOLDERS:
                    raise ValueError(f"{row['feature']}의 {column}에 placeholder를 사용할 수 없습니다.")
        if status == "VERIFIED":
            if pd.isna(row["valid_min"]) or pd.isna(row["valid_max"]):
                raise ValueError(f"{row['feature']} VERIFIED에는 유효 범위가 필요합니다.")
            verified_at = str(row["verified_at"])
            if not re.search(r"(?:Z|[+-]\d{2}:\d{2})$", verified_at):
                raise ValueError(f"{row['feature']} verified_at에는 시간대가 필요합니다.")
            pd.to_datetime(verified_at, utc=True, errors="raise")

    counts = output["verification_status"].value_counts().to_dict()
    verified_count = int(counts.get("VERIFIED", 0))
    provisional_count = int(counts.get("PROVISIONAL", 0))
    unmapped_count = int(counts.get("UNMAPPED", 0))
    complete = verified_count == len(FEATURES)
    report = {
        "report_version": 1,
        "validation_status": "VERIFIED" if complete else "AWAITING_MAPPING",
        "row_count": len(output),
        "expected_feature_count": len(FEATURES),
        "verified_count": verified_count,
        "provisional_count": provisional_count,
        "unmapped_count": unmapped_count,
        "verified_coverage": verified_count / len(FEATURES),
        "complete_owner_verified_mapping": complete,
        "dictionary_digest": dictionary_digest(output),
        "raw_sensor_values_present": False,
    }
    return output, report


def parse_sensor_dictionary_bytes(payload: bytes) -> tuple[pd.DataFrame, dict]:
    """Parse and validate an owner-supplied dictionary without installing it."""
    if not payload:
        raise ValueError("센서 사전 CSV가 비어 있습니다.")
    if len(payload) > MAX_DICTIONARY_UPLOAD_BYTES:
        raise ValueError("센서 사전 CSV는 5MB를 초과할 수 없습니다.")
    try:
        frame = pd.read_csv(BytesIO(payload), keep_default_na=False)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError("센서 사전 CSV 형식이 올바르지 않습니다.") from error
    return validate_sensor_dictionary(frame)


def sensor_report_json_bytes(report: Mapping[str, object]) -> bytes:
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def _csv_safe_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    if value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def normalized_dictionary_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Export a normalized dictionary while preventing spreadsheet formulas."""
    safe = frame[REQUIRED_COLUMNS].copy()
    for column in STRING_COLUMNS:
        safe[column] = safe[column].map(_csv_safe_text)
    return safe.to_csv(index=False).encode("utf-8-sig")


def dictionary_is_release_evidence(path: Path, saved_report: Mapping[str, object]) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path, keep_default_na=False)
        _, actual = validate_sensor_dictionary(frame)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return bool(
        actual["validation_status"] == "VERIFIED"
        and actual["complete_owner_verified_mapping"] is True
        and actual["verified_count"] == 590
        and actual["dictionary_digest"] == saved_report.get("dictionary_digest")
        and saved_report.get("validation_status") == "VERIFIED"
        and saved_report.get("complete_owner_verified_mapping") is True
        and saved_report.get("raw_sensor_values_present") is False
    )
