"""Tamper-checked row-level label feedback without storing raw SECOM sensors."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import pandas as pd


IMMUTABLE_COLUMNS = (
    "batch_id",
    "input_digest",
    "profile",
    "catboost_model_hash",
    "xgboost_model_hash",
    "row_number",
    "catboost_alert",
    "xgboost_alert",
    "predicted_alert",
    "consensus",
    "ood_status",
)
EDITABLE_COLUMNS = ("actual_label", "label_source", "reviewer", "reviewed_at", "note")
REQUIRED_COLUMNS = ("feedback_schema_version", *IMMUTABLE_COLUMNS, *EDITABLE_COLUMNS)


def _diagnosis_digest(results: pd.DataFrame) -> str:
    text = results.sort_index(axis=1).to_csv(index=False, float_format="%.12g")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_label_template(
    results: pd.DataFrame,
    ood: pd.DataFrame,
    *,
    batch_id: str,
    profile: str,
    model_hashes: Mapping[str, str],
) -> pd.DataFrame:
    """Create an editable label sheet containing no sensor measurements."""
    if not batch_id.strip():
        raise ValueError("배치 ID를 먼저 입력하세요.")
    if len(results) == 0 or len(results) != len(ood):
        raise ValueError("예측과 OOD 결과의 행 수가 일치해야 합니다.")
    row_numbers = (
        results["행 번호"].astype(int)
        if "행 번호" in results
        else pd.Series(range(len(results)), index=results.index, dtype=int)
    )
    if "행 번호" in ood and not row_numbers.reset_index(drop=True).equals(
        ood["행 번호"].astype(int).reset_index(drop=True)
    ):
        raise ValueError("예측과 OOD 결과의 행 번호 순서가 다릅니다.")
    cat_alert = results["CatBoost 판정"].astype(str).eq("불량")
    xgb_alert = results["XGBoost 판정"].astype(str).eq("불량")
    digest = _diagnosis_digest(results)
    template = pd.DataFrame(
        {
            "feedback_schema_version": 1,
            "batch_id": batch_id.strip(),
            "input_digest": digest,
            "profile": profile,
            "catboost_model_hash": str(model_hashes.get("CatBoost", "")),
            "xgboost_model_hash": str(model_hashes.get("XGBoost", "")),
            "row_number": row_numbers.to_numpy(),
            "catboost_alert": cat_alert.to_numpy(),
            "xgboost_alert": xgb_alert.to_numpy(),
            "predicted_alert": (cat_alert | xgb_alert).to_numpy(),
            "consensus": results["종합 판정"].astype(str).to_numpy(),
            "ood_status": ood["OOD 상태"].astype(str).to_numpy(),
            "actual_label": pd.NA,
            "label_source": "",
            "reviewer": "",
            "reviewed_at": "",
            "note": "",
        },
        columns=REQUIRED_COLUMNS,
    )
    return template


def template_csv_bytes(template: pd.DataFrame) -> bytes:
    return template.to_csv(index=False).encode("utf-8-sig")


def _parse_actual_label(value) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    normalized = str(value).strip().lower()
    mapping = {"0": 0, "0.0": 0, "정상": 0, "normal": 0, "-1": 0, "-1.0": 0,
               "1": 1, "1.0": 1, "불량": 1, "defect": 1}
    if normalized not in mapping:
        raise ValueError(f"actual_label은 정상/불량 또는 0/1이어야 합니다: {value}")
    return mapping[normalized]


def validate_completed_feedback(
    uploaded: pd.DataFrame, expected_template: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    missing = [column for column in REQUIRED_COLUMNS if column not in uploaded.columns]
    if missing:
        raise ValueError(f"행 단위 검수 파일 필수 열 누락: {missing}")
    if uploaded.empty:
        raise ValueError("검수 파일에 행이 없습니다.")
    if len(uploaded) > len(expected_template):
        raise ValueError("검수 행 수가 현재 배치 행 수를 초과합니다.")
    completed = uploaded.loc[:, REQUIRED_COLUMNS].copy()
    versions = pd.to_numeric(completed["feedback_schema_version"], errors="raise")
    if versions.isna().any() or not bool((versions == 1).all()):
        raise ValueError("지원하지 않는 feedback_schema_version입니다.")
    completed["row_number"] = pd.to_numeric(
        completed["row_number"], errors="raise"
    ).astype(int)
    if completed["row_number"].duplicated().any():
        raise ValueError("중복된 row_number가 있습니다.")
    expected = expected_template.loc[:, IMMUTABLE_COLUMNS].copy()
    merged = completed.merge(
        expected, on="row_number", how="left", suffixes=("", "_expected"), validate="one_to_one"
    )
    if merged["batch_id_expected"].isna().any():
        raise ValueError("현재 배치에 없는 row_number가 있습니다.")
    for column in IMMUTABLE_COLUMNS:
        if column == "row_number":
            continue
        left = merged[column].astype(str).str.strip().str.lower()
        right = merged[f"{column}_expected"].astype(str).str.strip().str.lower()
        if not left.equals(right):
            raise ValueError(f"예측 또는 배치 식별 열이 변경됐습니다: {column}")
    completed["actual_label"] = completed["actual_label"].map(_parse_actual_label)
    labeled = completed["actual_label"].notna()
    for column in ("label_source", "reviewer"):
        if completed.loc[labeled, column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"라벨이 있는 행에는 {column}가 필요합니다.")
    reviewed_at = pd.to_datetime(completed["reviewed_at"], errors="coerce", utc=True)
    if reviewed_at.loc[labeled].isna().any():
        raise ValueError("라벨이 있는 행에는 ISO 형식의 reviewed_at이 필요합니다.")
    labeled_rows = int(labeled.sum())
    total_rows = len(expected_template)
    labeled_frame = completed.loc[labeled].copy()
    actual = labeled_frame["actual_label"].astype(int)
    predicted = labeled_frame["predicted_alert"].astype(str).str.lower().eq("true")
    true_positive = int(((actual == 1) & predicted).sum())
    false_positive = int(((actual == 0) & predicted).sum())
    false_negative = int(((actual == 1) & ~predicted).sum())
    true_negative = int(((actual == 0) & ~predicted).sum())
    full_alert_rate = float(expected_template["predicted_alert"].mean())
    labeled_alert_rate = float(predicted.mean()) if labeled_rows else float("nan")
    complete = labeled_rows == total_rows and len(completed) == total_rows
    summary = {
        "complete": complete,
        "eligible_for_performance_monitoring": complete,
        "total_rows": total_rows,
        "submitted_rows": len(completed),
        "labeled_rows": labeled_rows,
        "label_coverage": labeled_rows / total_rows,
        "full_alert_rate": full_alert_rate,
        "labeled_alert_rate": labeled_alert_rate,
        "selection_bias_warning": not complete,
        "audit_metadata_complete": True,
        "confirmed_defects": int((actual == 1).sum()),
        "confirmed_alerted_defects": true_positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }
    return completed, summary


def hard_example_manifest(completed: pd.DataFrame, summary: Mapping) -> dict:
    labeled = completed.loc[completed["actual_label"].notna()].copy()
    actual = labeled["actual_label"].astype(int)
    predicted = labeled["predicted_alert"].astype(str).str.lower().eq("true")
    false_negative_rows = labeled.loc[(actual == 1) & ~predicted, "row_number"].astype(int).tolist()
    false_positive_rows = labeled.loc[(actual == 0) & predicted, "row_number"].astype(int).tolist()
    first = completed.iloc[0]
    return {
        "version": 1,
        "batch_id": str(first["batch_id"]),
        "input_digest": str(first["input_digest"]),
        "profile": str(first["profile"]),
        "catboost_model_hash": str(first["catboost_model_hash"]),
        "xgboost_model_hash": str(first["xgboost_model_hash"]),
        "complete_batch_labels": bool(summary["complete"]),
        "automatic_retraining_started": False,
        "false_negative_row_numbers": false_negative_rows,
        "false_positive_row_numbers": false_positive_rows,
        "contains_raw_sensor_values": False,
    }


def manifest_json_bytes(completed: pd.DataFrame, summary: Mapping) -> bytes:
    return json.dumps(
        hard_example_manifest(completed, summary), ensure_ascii=False, indent=2
    ).encode("utf-8")
