"""Freeze the SECOM prospective external-validation protocol before data intake."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from external_validation import (
    FEATURE_COLUMNS,
    OPTIONAL_METADATA_COLUMNS,
    REQUIRED_METADATA_COLUMNS,
    build_protocol,
)


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "결과물" / "secom" / "external_validation"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(PROJECT)
    schema = {
        "schema_version": 1,
        "required_columns": REQUIRED_METADATA_COLUMNS + FEATURE_COLUMNS,
        "optional_columns": OPTIONAL_METADATA_COLUMNS,
        "label_values": [0, 1],
        "captured_at": "ISO-8601 with Z or explicit UTC offset; strictly after registered_at",
        "missing_sensor_values": "allowed, but not all 590 sensors in one row",
        "unknown_columns_allowed": False,
    }
    status = {
        "status": "AWAITING_INDEPENDENT_DATA",
        "protocol_id": protocol["protocol_id"],
        "validated_summary_present": False,
        "production_claim_allowed": False,
        "next_action": "신규 장비·기간·lot에서 라벨 확정 후 evaluate_external_validation.py 실행",
    }
    declarations_template = {
        "prospective_collection": False,
        "independent_source": False,
        "labels_finalized_before_scoring": False,
        "model_outputs_hidden_from_labelers": False,
        "data_collected_after_protocol_freeze": False,
    }
    (OUTPUT_DIR / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "input_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "declarations_template.json").write_text(
        json.dumps(declarations_template, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(columns=REQUIRED_METADATA_COLUMNS + OPTIONAL_METADATA_COLUMNS + FEATURE_COLUMNS).to_csv(
        OUTPUT_DIR / "external_lot_template.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "README.md").write_text(
        "# SECOM 외부 신규 lot 전향 검증\n\n"
        f"프로토콜 `{protocol['protocol_id']}`는 외부 데이터를 보기 전에 모델 해시, 임계값, "
        "최소 표본과 합격 기준을 고정합니다. 현재 실제 독립 데이터가 없어 상태는 "
        "`AWAITING_INDEPENDENT_DATA`입니다.\n\n"
        "`external_lot_template.csv`의 헤더를 사용해 신규 데이터 파일을 만들고, 라벨 담당자가 "
        "모델 출력을 보지 않은 상태에서 라벨을 확정한 뒤 `declarations_template.json`의 항목을 "
        "사실인 경우에만 `true`로 바꾸고 평가 스크립트를 실행합니다. 기존 SECOM "
        "행과 정확히 겹치면 통과할 수 없습니다. 결과에는 원시 센서 행을 저장하지 않습니다.\n",
        encoding="utf-8",
    )
    print(f"외부 전향 검증 프로토콜 고정 완료: {protocol['protocol_id']}")


if __name__ == "__main__":
    main()
