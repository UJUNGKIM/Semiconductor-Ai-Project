"""Regression tests for SECOM sensor-corruption stress testing."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "코드" / "secom"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_sensor_robustness import perturb_sensors  # noqa: E402
from train_compare_models import load_data  # noqa: E402


class SecomRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result_dir = PROJECT_DIR / "결과물" / "secom" / "robustness_results"
        cls.metadata = json.loads(
            (cls.result_dir / "robustness_summary.json").read_text(encoding="utf-8")
        )
        cls.reference = joblib.load(
            PROJECT_DIR
            / "결과물"
            / "secom"
            / "advanced_diagnostics"
            / "advanced_reference.joblib"
        )
        cls.X, _, _ = load_data(PROJECT_DIR)

    def test_perturbation_is_deterministic_and_does_not_mutate_input(self) -> None:
        original = self.X.iloc[:8].copy(deep=True)
        first = perturb_sensors(
            original, self.reference, "random_missing", 0.10, seed=123
        )
        second = perturb_sensors(
            original, self.reference, "random_missing", 0.10, seed=123
        )
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(original, self.X.iloc[:8])
        self.assertGreater(first.isna().sum().sum(), original.isna().sum().sum())

    def test_column_dropout_uses_requested_train_feature_fraction(self) -> None:
        sample = self.X.iloc[:10].copy()
        corrupted = perturb_sensors(
            sample, self.reference, "column_dropout", 0.05, seed=42
        )
        features = self.reference["input_features"]
        fully_missing = int(corrupted[features].isna().all(axis=0).sum())
        self.assertGreaterEqual(fully_missing, math.ceil(len(features) * 0.05))

    def test_robustness_artifact_contract(self) -> None:
        self.assertFalse(
            self.metadata["test_used_for_model_or_threshold_selection"]
        )
        self.assertEqual(
            self.metadata["perturbation_statistics_source"],
            "fixed_train_only_iqr",
        )
        self.assertTrue(self.metadata["scenarios_predeclared"])
        self.assertEqual(self.metadata["scenario_count"], 10)
        self.assertEqual(self.metadata["replicates_per_scenario"], 5)
        self.assertEqual(
            self.metadata["guardrail_pass_count"]
            + self.metadata["guardrail_fail_count"],
            10,
        )
        summary = pd.read_csv(self.result_dir / "stress_test_summary.csv")
        self.assertEqual(len(summary), 11)
        self.assertIn(False, set(summary["guardrail_pass"]))

    def test_dashboard_and_result_files_exist(self) -> None:
        for name in (
            "robustness_dashboard.png",
            "stress_test_replicates.csv",
            "stress_test_summary.csv",
            "summary.md",
        ):
            self.assertTrue((self.result_dir / name).is_file(), name)
        app_source = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("센서 오류 강건성 스트레스 테스트", app_source)
        self.assertIn("ROBUSTNESS_DIR", app_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
