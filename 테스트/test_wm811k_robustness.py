"""Tests for WM-811K post-selection input-error stress evaluation."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
if str(WM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPT_DIR))

from diagnose_wm811k import artificial_demo_map  # noqa: E402
from evaluate_wm811k_robustness import SCENARIOS, perturb_wafer  # noqa: E402


class Wm811kRobustnessTests(unittest.TestCase):
    def test_perturbations_are_deterministic_valid_and_non_mutating(self) -> None:
        wafer = artificial_demo_map(64)
        original = wafer.copy()
        for scenario in SCENARIOS:
            first = perturb_wafer(wafer, kind=scenario["kind"], amount=scenario["amount"], rng=np.random.default_rng(42))
            second = perturb_wafer(wafer, kind=scenario["kind"], amount=scenario["amount"], rng=np.random.default_rng(42))
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.shape, (64, 64))
            self.assertTrue(set(np.unique(first)).issubset({0, 1, 2}))
        np.testing.assert_array_equal(wafer, original)

    def test_guardrails_are_declared_before_evaluation(self) -> None:
        self.assertEqual(len(SCENARIOS), 6)
        for scenario in SCENARIOS:
            self.assertIn(scenario["guardrail_metric"], {"prediction_stability", "changed_prediction_capture"})
            self.assertGreaterEqual(scenario["guardrail_minimum"], 0.8)

    def test_committed_artifact_contract(self) -> None:
        output_dir = PROJECT_DIR / "결과물" / "wm811k" / "robustness_results"
        summary = json.loads((output_dir / "robustness_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["used_for_model_selection"])
        self.assertFalse(summary["thresholds_retuned"])
        self.assertFalse(summary["representative_population_estimate"])
        self.assertEqual(summary["sample_count"], 27)
        self.assertEqual(summary["scenario_count"], 6)
        self.assertEqual(summary["evaluated_rows"], 27 * 6 * 5)
        self.assertTrue(summary["guardrails_declared_before_evaluation"])
        for digest in summary["artifact_sha256"].values():
            self.assertEqual(len(digest), 64)
        self.assertTrue((output_dir / "stress_test_summary.csv").is_file())
        self.assertTrue((output_dir / "stress_test_records.csv").is_file())
        self.assertTrue((output_dir / "robustness_dashboard.png").is_file())
        records = pd.read_csv(output_dir / "stress_test_records.csv")
        self.assertEqual(len(records), 810)
        for scenario in summary["scenarios"]:
            selected = records[records["scenario"] == scenario["scenario"]]
            missed = selected["prediction_changed"] & ~selected["review_or_hold"]
            self.assertEqual(int(missed.sum()), scenario["unreviewed_changed_count"])

    def test_dashboard_exposes_robustness_results(self) -> None:
        code = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("WM_ROBUSTNESS_DIR", code)
        self.assertIn("입력 오류 강건성 스트레스 테스트", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
