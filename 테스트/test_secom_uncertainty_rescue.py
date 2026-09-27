"""Regression tests for the SECOM uncertainty-rescue policy."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]


class SecomUncertaintyRescueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result_dir = (
            PROJECT_DIR / "결과물" / "secom" / "uncertainty_rescue_results"
        )
        cls.summary = json.loads(
            (cls.result_dir / "uncertainty_rescue_summary.json").read_text(
                encoding="utf-8"
            )
        )

    def test_policy_selection_is_train_only(self) -> None:
        self.assertFalse(self.summary["test_used_for_policy_selection"])
        self.assertEqual(
            self.summary["selection_source"],
            "repeated_5x3_primary_model_oof_train_only",
        )
        self.assertEqual(self.summary["selected_method"], "normalized_max_ratio")
        self.assertEqual(
            self.summary["rejected_method"], "regularized_logistic_meta"
        )

    def test_oof_target_selects_minimum_budget(self) -> None:
        budget = pd.read_csv(self.result_dir / "oof_budget_selection.csv")
        chosen = float(self.summary["chosen_additional_normal_budget"])
        target = float(self.summary["target_oof_false_negative_capture"])
        selected = budget.loc[budget["additional_normal_budget"] == chosen].iloc[0]
        self.assertGreaterEqual(selected["false_negative_capture"], target)
        earlier = budget.loc[budget["additional_normal_budget"] < chosen]
        self.assertTrue((earlier["false_negative_capture"] < target).all())

    def test_safety_first_tradeoff_and_uncertainty_are_explicit(self) -> None:
        standard = self.summary["test_standard"]
        safety = self.summary["test_safety_first"]
        self.assertGreater(safety["review_workload"], standard["review_workload"])
        self.assertGreater(
            safety["false_negative_capture"],
            standard["false_negative_capture"],
        )
        self.assertLessEqual(
            safety["fn_capture_ci_low"], safety["false_negative_capture"]
        )
        self.assertGreaterEqual(
            safety["fn_capture_ci_high"], safety["false_negative_capture"]
        )
        self.assertIn("외부 lot", self.summary["limitation"])

    def test_runtime_artifacts_exist(self) -> None:
        required = (
            "method_comparison.csv",
            "oof_budget_selection.csv",
            "policy_comparison.csv",
            "policy_comparison.png",
            "safety_first_test_queue.csv",
            "summary.md",
        )
        for name in required:
            self.assertTrue((self.result_dir / name).is_file(), name)
        source = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("안전 우선 · 미탐 예방", source)
        self.assertIn("normalized_risk_cutoff", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
