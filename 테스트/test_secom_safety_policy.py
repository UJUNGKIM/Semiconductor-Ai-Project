"""Tests for leakage-safe SECOM human-review policy evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "코드" / "secom"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from safety_policy import (  # noqa: E402
    budget_review_mask,
    locked_review_mask,
    model_alert_mask,
    policy_metrics,
    score_cutoff_review_mask,
)


class SecomSafetyPolicyTests(unittest.TestCase):
    def test_serialized_threshold_probability_remains_an_alert(self) -> None:
        threshold = 0.07179839120789556
        probability = np.array([0.0717983912078955])
        self.assertTrue(model_alert_mask(probability, threshold)[0])

    def test_locked_policy_reviews_alert_boundary_and_severe_ood(self) -> None:
        cat = np.array([0.08, 0.06, 0.01, 0.01])
        xgb = np.array([0.01, 0.01, 0.005, 0.005])
        review = locked_review_mask(
            cat,
            xgb,
            0.07,
            0.02,
            margin=0.25,
            ood_status=[
                "in_distribution",
                "in_distribution",
                "out_of_distribution",
                "in_distribution",
            ],
        )
        np.testing.assert_array_equal(review, [True, True, True, False])

    def test_policy_metrics_count_captured_errors(self) -> None:
        metrics = policy_metrics(
            y_true=[0, 1, 1, 0],
            automated_prediction=[0, 0, 1, 1],
            review_mask=[False, True, True, True],
        )
        self.assertEqual(metrics["errors"], 2)
        self.assertEqual(metrics["captured_errors"], 2)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["captured_false_negatives"], 1)
        self.assertAlmostEqual(metrics["review_workload"], 0.75)
        self.assertEqual(metrics["auto_accuracy"], 1.0)

    def test_budget_review_is_monotonic_and_label_free(self) -> None:
        cat = np.array([0.08, 0.06, 0.04, 0.01])
        xgb = np.array([0.01, 0.01, 0.01, 0.005])
        masks = [
            budget_review_mask(cat, xgb, 0.07, 0.02, budget)
            for budget in (0.0, 0.5, 1.0)
        ]
        self.assertLessEqual(masks[0].sum(), masks[1].sum())
        self.assertLessEqual(masks[1].sum(), masks[2].sum())
        self.assertTrue(np.all(masks[0] <= masks[1]))
        self.assertTrue(np.all(masks[1] <= masks[2]))

    def test_score_cutoff_policy_is_row_independent(self) -> None:
        review = score_cutoff_review_mask(
            [0.01, 0.04, 0.08],
            [0.005, 0.015, 0.005],
            0.07,
            0.02,
            0.6,
        )
        np.testing.assert_array_equal(review, [False, True, True])

    def test_committed_artifacts_preserve_test_lock(self) -> None:
        result_dir = (
            PROJECT_DIR / "결과물" / "secom" / "safety_policy_results"
        )
        metadata = json.loads(
            (result_dir / "safety_policy_summary.json").read_text(encoding="utf-8")
        )
        self.assertFalse(metadata["policy_selection_uses_test_labels"])
        self.assertEqual(metadata["random_state"], 42)
        self.assertEqual(metadata["test_locked_policy"]["rows"], 314)
        self.assertEqual(metadata["locked_boundary_margin"], 0.25)
        tradeoff = pd.read_csv(result_dir / "workload_tradeoff.csv")
        self.assertEqual(set(tradeoff["dataset"]), {"OOF train", "fixed test"})
        for _, rows in tradeoff.groupby("dataset"):
            ordered = rows.sort_values("additional_normal_budget")
            self.assertTrue(ordered["review_workload"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main(verbosity=2)
