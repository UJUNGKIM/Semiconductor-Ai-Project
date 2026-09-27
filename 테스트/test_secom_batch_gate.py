"""Regression tests for the SECOM batch safety circuit breaker."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "코드" / "secom"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_safety_gate import adaptive_limits, evaluate_gate_from_rates  # noqa: E402


class SecomBatchSafetyGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result_dir = PROJECT_DIR / "결과물" / "secom" / "batch_safety_gate"
        cls.reference = json.loads(
            (cls.result_dir / "batch_gate_reference.json").read_text(
                encoding="utf-8"
            )
        )
        cls.summary = json.loads(
            (cls.result_dir / "batch_gate_summary.json").read_text(encoding="utf-8")
        )

    def test_adaptive_limits_are_more_conservative_for_small_batches(self) -> None:
        warning_small, stop_small = adaptive_limits(0.1, 20, 0.2, 0.35)
        warning_large, stop_large = adaptive_limits(0.1, 500, 0.2, 0.35)
        self.assertGreaterEqual(warning_small, warning_large)
        self.assertGreaterEqual(stop_small, stop_large)
        self.assertLess(warning_small, stop_small)

    def test_clean_and_corrupted_batch_decisions(self) -> None:
        clean = evaluate_gate_from_rates(
            {
                "ood_any_rate": 0.08,
                "ood_severe_rate": 0.01,
                "model_disagreement_rate": 0.22,
            },
            314,
            self.reference,
        )
        corrupted = evaluate_gate_from_rates(
            {
                "ood_any_rate": 0.95,
                "ood_severe_rate": 0.80,
                "model_disagreement_rate": 0.40,
            },
            314,
            self.reference,
        )
        self.assertEqual(clean["status"], "PASS")
        self.assertTrue(clean["automatic_decision_allowed"])
        self.assertEqual(corrupted["status"], "STOP")
        self.assertFalse(corrupted["automatic_decision_allowed"])

    def test_small_batch_uses_row_level_protection(self) -> None:
        result = evaluate_gate_from_rates(
            {
                "ood_any_rate": 0.0,
                "ood_severe_rate": 0.0,
                "model_disagreement_rate": 0.0,
            },
            5,
            self.reference,
        )
        self.assertEqual(result["status"], "ROW_LEVEL_ONLY")
        self.assertTrue(result["automatic_decision_allowed"])

    def test_artifact_and_dashboard_contract(self) -> None:
        self.assertFalse(self.reference["test_used_for_threshold_selection"])
        self.assertEqual(
            self.reference["fit_scope"], "fixed_train_and_repeated_oof_only"
        )
        self.assertFalse(self.reference["automatic_decision_on_stop"])
        self.assertEqual(self.summary["clean_test_decision"]["status"], "PASS")
        self.assertGreater(self.summary["stress_stop_count"], 0)
        for name in (
            "gate_condition_summary.csv",
            "gate_stress_validation.csv",
            "gate_stress_validation.png",
            "summary.md",
        ):
            self.assertTrue((self.result_dir / name).is_file(), name)
        app_source = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("배치 안전 차단기 STOP", app_source)
        self.assertIn("배치 차단 · 전체 검토", app_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
