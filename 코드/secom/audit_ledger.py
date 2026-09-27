"""Tamper-evident, aggregate-only audit ledger for SECOM operations."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

import pandas as pd


GENESIS_HASH = "0" * 64
LEDGER_COLUMNS = (
    "sequence",
    "recorded_at",
    "event_type",
    "actor",
    "batch_id",
    "payload_json",
    "previous_hash",
    "event_hash",
)
ALLOWED_EVENT_TYPES = {
    "BATCH_DIAGNOSIS",
    "LABEL_FEEDBACK",
    "MODEL_COMPARISON",
    "RELEASE_READINESS",
    "MANUAL_NOTE",
}
FORBIDDEN_PAYLOAD_KEYS = {
    "sensor_values",
    "raw_sensor_values",
    "feature_values",
    "raw_feature_values",
    "input_matrix",
    "raw_input",
    "sensor_rows",
    "features",
}
MAX_LEDGER_BYTES = 5 * 1024 * 1024
MAX_LEDGER_ENTRIES = 100_000
MAX_PAYLOAD_BYTES = 32 * 1024


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"JSON으로 변환할 수 없는 값입니다: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"중복 JSON 키가 있습니다: {key}")
        output[key] = value
    return output


def _check_payload_keys(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(
                    f"감사 원장에는 센서 원본·특징값을 저장할 수 없습니다: {path}.{key_text}"
                )
            _check_payload_keys(child, f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_payload_keys(child, f"{path}[{index}]")


def _canonical_payload(payload: Mapping[str, Any]) -> str:
    if not isinstance(payload, Mapping):
        raise ValueError("payload는 JSON 객체여야 합니다.")
    _check_payload_keys(payload)
    encoded = _canonical_json(dict(payload))
    if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError("감사 이벤트 payload는 32KB 이하여야 합니다.")
    return encoded


def _hash_material(record: Mapping[str, Any]) -> str:
    material = {
        "sequence": int(record["sequence"]),
        "recorded_at": str(record["recorded_at"]),
        "event_type": str(record["event_type"]),
        "actor": str(record["actor"]),
        "batch_id": str(record["batch_id"]),
        "payload_json": str(record["payload_json"]),
        "previous_hash": str(record["previous_hash"]),
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def validate_ledger(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return a canonical ledger or raise when its chain/content is invalid."""
    if frame is None or frame.empty:
        return empty_ledger()
    if len(frame) > MAX_LEDGER_ENTRIES:
        raise ValueError(f"감사 원장은 최대 {MAX_LEDGER_ENTRIES:,}개 이벤트까지 지원합니다.")
    missing = [column for column in LEDGER_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"감사 원장 필수 열 누락: {missing}")
    output = frame.loc[:, LEDGER_COLUMNS].copy().reset_index(drop=True)
    expected_previous = GENESIS_HASH
    previous_time = None
    for index, row in output.iterrows():
        row_number = index + 1
        try:
            sequence = int(row["sequence"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{row_number}행 sequence가 정수가 아닙니다.") from error
        if sequence != row_number:
            raise ValueError(
                f"{row_number}행 sequence가 연속적이지 않습니다: {sequence}"
            )
        event_type = str(row["event_type"])
        if event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"{row_number}행 event_type이 허용되지 않습니다: {event_type}")
        try:
            timestamp = pd.Timestamp(row["recorded_at"])
        except Exception as error:
            raise ValueError(f"{row_number}행 기록 시각이 올바르지 않습니다.") from error
        if timestamp.tzinfo is None:
            raise ValueError(f"{row_number}행 기록 시각에 시간대가 없습니다.")
        if previous_time is not None and timestamp < previous_time:
            raise ValueError(f"{row_number}행 기록 시각이 이전 이벤트보다 빠릅니다.")
        previous_time = timestamp
        actor = str(row["actor"])
        batch_id = str(row["batch_id"])
        if len(actor) > 80 or len(batch_id) > 120:
            raise ValueError(f"{row_number}행 actor 또는 batch_id가 너무 깁니다.")
        try:
            payload = json.loads(
                str(row["payload_json"]), object_pairs_hook=_reject_duplicate_keys
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{row_number}행 payload JSON이 올바르지 않습니다: {error}") from error
        payload_json = _canonical_payload(payload)
        if payload_json != str(row["payload_json"]):
            raise ValueError(f"{row_number}행 payload JSON이 정규 형식이 아닙니다.")
        previous_hash = str(row["previous_hash"]).lower()
        event_hash = str(row["event_hash"]).lower()
        if previous_hash != expected_previous:
            raise ValueError(f"{row_number}행 이전 해시가 연결되지 않습니다.")
        canonical_row = {
            "sequence": sequence,
            "recorded_at": str(row["recorded_at"]),
            "event_type": event_type,
            "actor": actor,
            "batch_id": batch_id,
            "payload_json": payload_json,
            "previous_hash": previous_hash,
        }
        expected_hash = _hash_material(canonical_row)
        if event_hash != expected_hash:
            raise ValueError(f"{row_number}행 이벤트 해시가 일치하지 않습니다.")
        output.loc[index] = [
            sequence,
            canonical_row["recorded_at"],
            event_type,
            actor,
            batch_id,
            payload_json,
            previous_hash,
            event_hash,
        ]
        expected_previous = event_hash
    output["sequence"] = output["sequence"].astype(int)
    return output


def verify_ledger(frame: pd.DataFrame | None) -> dict:
    ledger = validate_ledger(frame)
    counts = Counter(ledger["event_type"].tolist())
    return {
        "valid": True,
        "entry_count": len(ledger),
        "latest_hash": GENESIS_HASH if ledger.empty else str(ledger.iloc[-1]["event_hash"]),
        "event_type_counts": dict(sorted(counts.items())),
        "contains_raw_sensor_values": False,
        "algorithm": "SHA-256",
    }


def append_event(
    frame: pd.DataFrame | None,
    *,
    event_type: str,
    payload: Mapping[str, Any],
    actor: str = "",
    batch_id: str = "",
    recorded_at: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    ledger = validate_ledger(frame)
    event_type = str(event_type).strip().upper()
    if event_type not in ALLOWED_EVENT_TYPES:
        raise ValueError(f"허용되지 않은 감사 이벤트입니다: {event_type}")
    actor = str(actor).strip()[:80]
    batch_id = str(batch_id).strip()[:120]
    timestamp = pd.Timestamp.now(tz="Asia/Seoul") if recorded_at is None else pd.Timestamp(recorded_at)
    if timestamp.tzinfo is None:
        raise ValueError("감사 이벤트 기록 시각에는 시간대가 필요합니다.")
    if not ledger.empty and timestamp < pd.Timestamp(ledger.iloc[-1]["recorded_at"]):
        raise ValueError("새 이벤트 시각은 기존 마지막 이벤트보다 빠를 수 없습니다.")
    sequence = len(ledger) + 1
    row = {
        "sequence": sequence,
        "recorded_at": timestamp.isoformat(),
        "event_type": event_type,
        "actor": actor,
        "batch_id": batch_id,
        "payload_json": _canonical_payload(payload),
        "previous_hash": GENESIS_HASH if ledger.empty else str(ledger.iloc[-1]["event_hash"]),
    }
    row["event_hash"] = _hash_material(row)
    return validate_ledger(pd.concat([ledger, pd.DataFrame([row])], ignore_index=True))


def ledger_jsonl_bytes(frame: pd.DataFrame | None) -> bytes:
    ledger = validate_ledger(frame)
    lines = []
    for row in ledger.to_dict(orient="records"):
        export_row = {key: row[key] for key in LEDGER_COLUMNS if key != "payload_json"}
        export_row["payload"] = json.loads(row["payload_json"])
        lines.append(_canonical_json(export_row))
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def parse_ledger_jsonl(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_LEDGER_BYTES:
        raise ValueError("감사 원장 JSONL은 5MB 이하여야 합니다.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("감사 원장은 UTF-8 JSONL이어야 합니다.") from error
    rows = []
    for line_number, line in enumerate(io.StringIO(text), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"JSONL {line_number}행을 읽지 못했습니다: {error}") from error
        if not isinstance(record, dict) or "payload" not in record:
            raise ValueError(f"JSONL {line_number}행에 payload 객체가 없습니다.")
        unknown = set(record) - (set(LEDGER_COLUMNS) - {"payload_json"}) - {"payload"}
        if unknown:
            raise ValueError(f"JSONL {line_number}행에 알 수 없는 키가 있습니다: {sorted(unknown)}")
        payload = record.pop("payload")
        record["payload_json"] = _canonical_payload(payload)
        rows.append(record)
    if len(rows) > MAX_LEDGER_ENTRIES:
        raise ValueError(f"감사 원장은 최대 {MAX_LEDGER_ENTRIES:,}개 이벤트까지 지원합니다.")
    return validate_ledger(pd.DataFrame(rows, columns=LEDGER_COLUMNS))


def monitoring_record_payload(record: Mapping[str, Any]) -> dict:
    """Select aggregate, non-sensitive monitoring evidence for one audit event."""
    allowed = (
        "profile",
        "row_count",
        "both_models_defect",
        "one_model_defect",
        "both_models_normal",
        "defect_alert_rate",
        "model_disagreement_rate",
        "ood_any_rate",
        "ood_severe_rate",
        "gate_status",
        "automatic_decision_allowed",
        "review_target_count",
        "input_digest",
        "catboost_model_hash",
        "xgboost_model_hash",
        "outcome_confirmed",
        "confirmed_defects",
        "confirmed_alerted_defects",
        "feedback_evidence",
    )
    return {key: record.get(key) for key in allowed}
