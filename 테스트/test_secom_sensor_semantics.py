import json
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from release_readiness import build_release_readiness  # noqa: E402
from sensor_semantics import (  # noqa: E402
    empty_dictionary_template,
    normalized_dictionary_csv_bytes,
    parse_sensor_dictionary_bytes,
    sensor_report_json_bytes,
    validate_sensor_dictionary,
)


class SensorSemanticsTests(unittest.TestCase):
    def test_template_is_complete_but_honestly_unmapped(self):
        normalized, report = validate_sensor_dictionary(empty_dictionary_template())
        self.assertEqual(len(normalized), 590)
        self.assertEqual(report["validation_status"], "AWAITING_MAPPING")
        self.assertEqual(report["unmapped_count"], 590)
        self.assertEqual(report["verified_count"], 0)
        self.assertFalse(report["complete_owner_verified_mapping"])

    def test_missing_duplicate_and_one_sided_range_are_rejected(self):
        frame = empty_dictionary_template().drop(index=589)
        with self.assertRaisesRegex(ValueError, "590행"):
            validate_sensor_dictionary(frame)
        frame = empty_dictionary_template()
        frame.loc[1, "feature"] = "feature_0"
        with self.assertRaisesRegex(ValueError, "중복"):
            validate_sensor_dictionary(frame)
        frame = empty_dictionary_template()
        frame.loc[0, "valid_min"] = 0.0
        with self.assertRaisesRegex(ValueError, "둘 다"):
            validate_sensor_dictionary(frame)

    def test_verified_rows_require_real_metadata_range_and_timezone(self):
        frame = empty_dictionary_template()
        frame.loc[0, "verification_status"] = "VERIFIED"
        with self.assertRaisesRegex(ValueError, "필수값"):
            validate_sensor_dictionary(frame)
        values = {
            "sensor_name": "Chamber pressure A",
            "unit": "Pa",
            "process_step": "etch",
            "equipment_scope": "tool-family-A",
            "description": "Pressure measured at chamber inlet",
            "valid_min": 1.0,
            "valid_max": 10.0,
            "owner": "process-engineering",
            "source_reference": "controlled-spec-001",
            "verified_at": "2026-10-01T09:00:00+09:00",
        }
        for key, value in values.items():
            frame.loc[0, key] = value
        _, report = validate_sensor_dictionary(frame)
        self.assertEqual(report["verified_count"], 1)
        self.assertEqual(report["validation_status"], "AWAITING_MAPPING")

    def test_placeholder_cannot_be_promoted_to_verified(self):
        frame = empty_dictionary_template()
        frame.loc[0, "verification_status"] = "PROVISIONAL"
        frame.loc[0, "sensor_name"] = "TBD"
        frame.loc[0, "process_step"] = "etch"
        frame.loc[0, "description"] = "candidate mapping"
        frame.loc[0, "source_reference"] = "interview-001"
        with self.assertRaisesRegex(ValueError, "placeholder"):
            validate_sensor_dictionary(frame)

    def test_uploaded_dictionary_is_validated_without_installing(self):
        payload = empty_dictionary_template().to_csv(index=False).encode("utf-8")
        normalized, report = parse_sensor_dictionary_bytes(payload)
        self.assertEqual(len(normalized), 590)
        self.assertEqual(report["validation_status"], "AWAITING_MAPPING")
        self.assertFalse(report["complete_owner_verified_mapping"])
        self.assertFalse(
            (PROJECT / "데이터" / "SECOM 데이터셋" / "sensor_dictionary.csv").exists()
        )
        with self.assertRaisesRegex(ValueError, "비어"):
            parse_sensor_dictionary_bytes(b"")

    def test_validation_exports_exclude_raw_values_and_escape_formulas(self):
        frame = empty_dictionary_template()
        frame.loc[0, "sensor_name"] = "=FORMULA()"
        normalized, report = validate_sensor_dictionary(frame)
        report_text = sensor_report_json_bytes(report).decode("utf-8")
        self.assertNotIn("FORMULA", report_text)
        self.assertNotIn("sensor_name", report_text)
        exported = normalized_dictionary_csv_bytes(normalized).decode("utf-8-sig")
        self.assertIn("'=FORMULA()", exported)

    def test_committed_intake_and_release_blocker(self):
        template_path = PROJECT / "데이터" / "SECOM 데이터셋" / "sensor_dictionary_template.csv"
        template = pd.read_csv(template_path, keep_default_na=False)
        _, report = validate_sensor_dictionary(template)
        status = json.loads(
            (PROJECT / "결과물" / "secom" / "sensor_semantics" / "status.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(status["dictionary_digest"], report["dictionary_digest"])
        self.assertFalse(status["production_claim_allowed"])
        self.assertFalse(
            (PROJECT / "데이터" / "SECOM 데이터셋" / "sensor_dictionary.csv").exists()
        )
        release, _ = build_release_readiness(PROJECT)
        check = next(item for item in release["checks"] if item["check_id"] == "SENSOR_SEMANTICS")
        self.assertEqual(check["status"], "BLOCK")
        self.assertIn("입력 템플릿", check["evidence"])


if __name__ == "__main__":
    unittest.main()
