"""Tests for validation-only WM-811K weak-class score adjustment audit."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "코드" / "wm811k"))

from analyze_wm811k_weak_score_adjustment import (  # noqa: E402
    adjusted_predictions,
    choose_candidate,
    stable_lot_folds,
)


class Wm811kWeakScoreAdjustmentTests(unittest.TestCase):
    def test_adjustment_boosts_only_weak_labels_before_argmax(self) -> None:
        probabilities = np.array([[0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.60]])
        self.assertEqual(int(adjusted_predictions(probabilities, 1.0)[0]), 8)
        self.assertEqual(int(adjusted_predictions(probabilities, 2.0)[0]), 2)

    def test_candidate_selection_respects_guardrails(self) -> None:
        frame = pd.DataFrame([
            {"multiplier": 1.0, "macro_f1": 0.90, "weak_recall": 0.80, "none_recall": 0.99},
            {"multiplier": 1.1, "macro_f1": 0.897, "weak_recall": 0.82, "none_recall": 0.989},
            {"multiplier": 1.2, "macro_f1": 0.89, "weak_recall": 0.90, "none_recall": 0.989},
        ])
        selected, evaluated = choose_candidate(frame)
        self.assertEqual(selected, 1.1)
        self.assertFalse(bool(evaluated.loc[evaluated["multiplier"].eq(1.2), "eligible"].iloc[0]))

    def test_lot_fold_assignment_is_stable_and_grouped(self) -> None:
        lots = pd.Series(["lot1", "lot1", "lot2", "lot3"])
        first = stable_lot_folds(lots)
        second = stable_lot_folds(lots)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(int(first[0]), int(first[1]))

    def test_saved_result_is_exploratory_and_not_deployed(self) -> None:
        result_dir = PROJECT_DIR / "결과물" / "wm811k" / "weak_score_adjustment_results"
        summary = json.loads((result_dir / "score_adjustment_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["selected_multiplier"], 1.2)
        self.assertFalse(summary["test_evaluated"])
        self.assertFalse(summary["deployment_changed"])
        self.assertGreater(summary["deltas"]["weak_recall"], 0)
        self.assertEqual(summary["weak_recall_not_worse_folds"], 5)
        self.assertTrue((result_dir / "score_adjustment_dashboard.png").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
