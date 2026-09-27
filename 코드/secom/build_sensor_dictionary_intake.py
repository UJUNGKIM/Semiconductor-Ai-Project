"""Create an honest, unmapped SECOM sensor-dictionary intake package."""

from __future__ import annotations

import json
from pathlib import Path

from sensor_semantics import REQUIRED_COLUMNS, empty_dictionary_template, validate_sensor_dictionary


PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "데이터" / "SECOM 데이터셋"
OUTPUT_DIR = PROJECT / "결과물" / "secom" / "sensor_semantics"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    template = empty_dictionary_template()
    normalized, report = validate_sensor_dictionary(template)
    normalized.to_csv(DATA_DIR / "sensor_dictionary_template.csv", index=False, encoding="utf-8-sig")
    schema = {
        "schema_version": 1,
        "required_columns": REQUIRED_COLUMNS,
        "expected_rows": 590,
        "feature_contract": "feature_0 through feature_589 exactly once",
        "statuses": ["UNMAPPED", "PROVISIONAL", "VERIFIED"],
        "verified_requires": [
            "actual sensor name and unit",
            "process step and equipment scope",
            "physical description and valid_min/valid_max",
            "named owner and source reference",
            "timezone-aware verified_at",
        ],
        "placeholder_values_for_mapped_rows_allowed": False,
        "raw_sensor_values_in_dictionary_allowed": False,
    }
    status = {
        **report,
        "status": "AWAITING_OWNER_MAPPING",
        "production_claim_allowed": False,
        "template_path": "데이터/SECOM 데이터셋/sensor_dictionary_template.csv",
        "installed_dictionary_path": "데이터/SECOM 데이터셋/sensor_dictionary.csv",
        "next_action": "장비·공정 담당자가 실제 의미와 근거를 입력하고 검증 스크립트를 실행",
    }
    (OUTPUT_DIR / "input_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "README.md").write_text(
        "# SECOM 센서 의미 사전 입력 패키지\n\n"
        "공개 SECOM의 590개 변수는 익명이므로 센서명·단위·공정 단계를 추정해 채우지 "
        "않습니다. 템플릿의 모든 행은 `UNMAPPED`로 시작합니다.\n\n"
        "실제 장비·공정 담당자가 센서명, 단위, 공정, 장비 범위, 설명, 물리 유효 범위, "
        "책임자, 근거 문서와 시간대가 있는 검증 시각을 입력하고 `VERIFIED`로 바꿔야 합니다. "
        "590개 전체가 검증되기 전에는 운영 준비로 인정하지 않습니다.\n",
        encoding="utf-8",
    )
    print("센서 사전 입력 패키지 생성 완료: verified=0/590, production_ready=False")


if __name__ == "__main__":
    main()
