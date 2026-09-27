import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from model_registry import (
    load_verified_bundles,
    simulated_corruption_is_detected,
    validate_registry,
)


class ModelFailoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = PROJECT / "결과물" / "secom" / "model_failover"
        cls.registry = json.loads(
            (cls.output / "model_registry.json").read_text(encoding="utf-8")
        )
        cls.drill = json.loads(
            (cls.output / "failover_drill.json").read_text(encoding="utf-8")
        )

    def test_registry_verifies_both_models_and_contract(self):
        result = validate_registry(self.registry, PROJECT)
        self.assertTrue(result["valid"])
        self.assertEqual(result["active_model"], "CatBoost")
        self.assertEqual(result["standby_model"], "XGBoost")
        self.assertTrue(result["input_schema_compatible"])
        self.assertFalse(self.registry["automatic_failover_allowed"])
        self.assertTrue(self.registry["activation_requires_human_approval"])

    def test_path_traversal_and_hash_tampering_are_rejected(self):
        unsafe = copy.deepcopy(self.registry)
        unsafe["models"][0]["artifact_path"] = "../outside.joblib"
        with self.assertRaisesRegex(ValueError, "프로젝트 범위"):
            validate_registry(unsafe, PROJECT)
        altered = copy.deepcopy(self.registry)
        altered["models"][0]["artifact_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_registry(altered, PROJECT)

    def test_corruption_drill_never_modifies_real_model(self):
        active = next(
            item for item in self.registry["models"] if item["role"] == "ACTIVE"
        )
        path = PROJECT / active["artifact_path"]
        before = path.read_bytes()
        self.assertTrue(simulated_corruption_is_detected(path, active["artifact_sha256"]))
        self.assertEqual(before, path.read_bytes())
        self.assertTrue(self.drill["simulated_active_corruption_detected"])
        self.assertFalse(self.drill["real_model_file_modified"])

    def test_hash_failure_stops_before_deserialization(self):
        altered = copy.deepcopy(self.registry)
        altered["models"][0]["artifact_sha256"] = "0" * 64
        registry_path = self.output / "invalid_registry_for_test.json"
        registry_path.write_text(json.dumps(altered), encoding="utf-8")
        try:
            with patch("model_registry.joblib.load") as mocked_load:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    load_verified_bundles(registry_path, PROJECT)
                mocked_load.assert_not_called()
        finally:
            registry_path.unlink(missing_ok=True)

    def test_failover_is_human_controlled_and_post_selection(self):
        self.assertEqual(self.drill["status"], "PASS")
        self.assertFalse(self.drill["automatic_failover_allowed"])
        self.assertEqual(
            self.drill["recommended_action"],
            "HALT_AND_REQUIRE_HUMAN_APPROVAL_FOR_STANDBY",
        )
        self.assertTrue(self.drill["fixed_test_post_selection_audit_only"])
        self.assertEqual(self.drill["test_rows"], 314)
        self.assertGreater(self.drill["model_prediction_disagreement_rate"], 0)

    def test_artifacts_and_dashboard_contract(self):
        for name in (
            "artifact_inventory.csv",
            "fixed_test_behavior.csv",
            "summary.md",
        ):
            self.assertTrue((self.output / name).is_file(), name)
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("모델 무결성·장애 복구 훈련", app)
        self.assertIn("secom_model_registry.json", app)
        self.assertIn("자동 전환하지 않으며", app)


if __name__ == "__main__":
    unittest.main()
