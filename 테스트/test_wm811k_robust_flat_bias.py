"""Tests for WM-811K robust-flat shared none-bias calibration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
if str(WM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPT_DIR))

from calibrate_wm811k_robust_flat_bias import (  # noqa: E402
    SEEDS,
    metric_bundle,
    predict_with_none_bias,
    select_shared_bias,
)


class Wm811kRobustFlatBiasTests(unittest.TestCase):
    def test_none_bias_changes_only_decision_boundary(self) -> None:
        probabilities = np.zeros((2, 9), dtype=np.float64)
        probabilities[0, 2] = 0.55
        probabilities[0, 8] = 0.45
        probabilities[1, 7] = 0.90
        probabilities[1, 8] = 0.10
        before = predict_with_none_bias(probabilities, 0.0)
        after = predict_with_none_bias(probabilities, 0.5)
        np.testing.assert_array_equal(before, np.array([2, 7]))
        np.testing.assert_array_equal(after, np.array([8, 7]))
        np.testing.assert_array_equal(probabilities.sum(axis=1), np.ones(2))

    def test_metric_bundle_reports_weak_and_none_recall(self) -> None:
        true = np.repeat(np.arange(9), 2)
        predicted = true.copy()
        predicted[4] = 8  # one Edge-Loc error
        result = metric_bundle(true, predicted)
        self.assertAlmostEqual(result["none_recall"], 1.0)
        self.assertAlmostEqual(result["weak_recall"], (0.5 + 1.0 + 1.0) / 3)

    def test_shared_bias_selector_requires_all_seed_guardrails(self) -> None:
        loaded = {}
        true = np.repeat(np.arange(9), 5)
        for seed in SEEDS:
            baseline = np.full((len(true), 9), 0.001)
            baseline[np.arange(len(true)), true] = 0.992
            baseline /= baseline.sum(axis=1, keepdims=True)
            robust = baseline.copy()
            loaded[seed, "baseline"] = (true, baseline)
            loaded[seed, "robust_flat"] = (true, robust)
        selected, sweep, _, checks = select_shared_bias(
            loaded, np.array([0.0, 0.5])
        )
        self.assertIsNone(selected)
        self.assertFalse(checks["mean_weak_recall_improved"])
        self.assertFalse(bool(sweep["eligible"].any()))

    def test_received_bundle_selects_registered_bias(self) -> None:
        bundle = (
            PROJECT_DIR / "결과물" / "wm811k" / "colab_가져오기"
            / "wm811k_robust_flat_predictions.zip"
        )
        from calibrate_wm811k_robust_flat_bias import load_bundle

        loaded = load_bundle(bundle)
        selected, sweep, metrics, checks = select_shared_bias(
            loaded, np.linspace(0.0, 2.0, 401)
        )
        self.assertEqual(selected, 1.155)
        self.assertEqual(int(sweep["eligible"].sum()), 49)
        self.assertTrue(all(checks.values()))
        self.assertGreater(float(metrics["delta_weak_recall"].mean()), 0.04)
        self.assertGreater(float(metrics["delta_macro_f1"].mean()), 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
