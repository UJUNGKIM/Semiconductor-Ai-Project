import json
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


class SecomGovernanceDashboardTests(unittest.TestCase):
    def test_dashboard_exposes_external_validation_and_sensor_intake(self):
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        for marker in (
            "EXTERNAL_VALIDATION_DIR",
            "SENSOR_SEMANTICS_DIR",
            "외부 신규 lot 검증 준비",
            "센서 의미·단위 사전 준비",
            "secom_external_lot_template.csv",
            "secom_external_validation_declarations.json",
            "secom_external_validation_protocol.json",
            "독립 신규 lot 검증 실행",
            "run_secom_external_validation",
            "secom_external_validation_summary.json",
            "secom_external_validation_per_lot.csv",
            "secom_sensor_dictionary_template.csv",
            "secom_sensor_dictionary_schema.json",
            "작성된 센서 사전 검증",
            "run_secom_sensor_dictionary_validation",
            "secom_sensor_dictionary_validation.json",
            "secom_sensor_dictionary_normalized.csv",
        ):
            self.assertIn(marker, app)

    def test_download_artifacts_are_honest_and_present(self):
        external = PROJECT / "결과물" / "secom" / "external_validation"
        sensor = PROJECT / "결과물" / "secom" / "sensor_semantics"
        declarations = json.loads(
            (external / "declarations_template.json").read_text(encoding="utf-8")
        )
        sensor_status = json.loads((sensor / "status.json").read_text(encoding="utf-8"))
        self.assertTrue(all(value is False for value in declarations.values()))
        self.assertEqual(sensor_status["verified_count"], 0)
        self.assertFalse(sensor_status["production_claim_allowed"])
        self.assertTrue(
            (PROJECT / "데이터" / "SECOM 데이터셋" / "sensor_dictionary_template.csv").is_file()
        )


if __name__ == "__main__":
    unittest.main()
