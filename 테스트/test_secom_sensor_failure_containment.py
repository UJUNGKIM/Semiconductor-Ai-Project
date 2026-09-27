import json
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from sensor_failure_containment import (  # noqa: E402
    build_sensor_failure_containment,
    validate_sensor_failure_containment,
)


class SensorFailureContainmentTests(unittest.TestCase):
    def test_committed_failure_is_fully_contained_without_claiming_robustness(self):
        path = (
            PROJECT
            / "결과물"
            / "secom"
            / "sensor_failure_containment"
            / "containment_report.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        validation = validate_sensor_failure_containment(report, PROJECT)
        self.assertEqual(validation["status"], "PASS")
        self.assertTrue(validation["all_failed_scenarios_contained"])
        self.assertEqual(report["model_robustness_guardrail_status"], "WARN")
        self.assertFalse(report["model_robustness_improved_by_this_audit"])
        self.assertFalse(report["actual_hardware_faults_validated"])
        self.assertFalse(report["raw_sensor_values_persisted"])
        failed = report["failed_scenario_containment"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["scenario"], "가우시안 노이즈 1.00 IQR")
        self.assertEqual(failed[0]["stop_rate"], 1.0)

    def test_current_inputs_rebuild_same_deterministic_core(self):
        committed = json.loads(
            (
                PROJECT
                / "결과물"
                / "secom"
                / "sensor_failure_containment"
                / "containment_report.json"
            ).read_text(encoding="utf-8")
        )
        rebuilt = build_sensor_failure_containment(
            PROJECT, committed["evaluated_at"]
        )
        self.assertEqual(committed, rebuilt)

    def test_validation_detects_forged_pass(self):
        report = build_sensor_failure_containment(PROJECT, "2026-09-18T00:00:00Z")
        report["failed_scenario_containment"][0]["stop_count"] = 0
        with self.assertRaisesRegex(ValueError, "현재 입력과 다릅니다"):
            validate_sensor_failure_containment(report, PROJECT)

    def test_source_rows_are_one_to_one(self):
        stress = pd.read_csv(
            PROJECT
            / "결과물"
            / "secom"
            / "robustness_results"
            / "stress_test_replicates.csv"
        )
        gate = pd.read_csv(
            PROJECT
            / "결과물"
            / "secom"
            / "batch_safety_gate"
            / "gate_stress_validation.csv"
        )
        keys = ["kind", "severity", "replicate"]
        self.assertFalse(stress.duplicated(keys).any())
        self.assertFalse(gate.duplicated(keys).any())
        self.assertEqual(
            set(map(tuple, stress[keys].to_numpy())),
            set(map(tuple, gate[keys].to_numpy())),
        )
        merged = stress.merge(gate, on=keys, validate="one_to_one")
        self.assertTrue(
            (merged["scenario_x"] == merged["scenario_korean"]).all()
        )

    def test_dashboard_contract(self):
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("센서 실패 안전격리", app)
        self.assertIn("secom_sensor_failure_containment.json", app)


if __name__ == "__main__":
    unittest.main()
