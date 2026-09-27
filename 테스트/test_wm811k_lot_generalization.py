"""Tests for held-out lot generalization diagnostics."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_wm811k_lot_generalization import cluster_bootstrap_accuracy, wilson_interval  # noqa: E402


class Wm811kLotGeneralizationTests(unittest.TestCase):
    def test_wilson_interval_contains_rate(self) -> None:
        low, high = wilson_interval(18, 20)
        self.assertLess(low, 0.9)
        self.assertGreater(high, 0.9)
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        lots = pd.DataFrame({"correct_count": [18, 20, 15], "support": [20, 20, 20]})
        first = cluster_bootstrap_accuracy(lots, 200, 42)
        second = cluster_bootstrap_accuracy(lots, 200, 42)
        np.testing.assert_allclose(first, second)

    def test_committed_artifact_contract(self) -> None:
        result_dir = PROJECT_DIR / "결과물" / "wm811k" / "lot_generalization_results"
        summary = json.loads((result_dir / "lot_generalization_summary.json").read_text(encoding="utf-8"))
        lots = pd.read_csv(result_dir / "lot_metrics.csv")
        geometries = pd.read_csv(result_dir / "geometry_metrics.csv")
        classes = pd.read_csv(result_dir / "class_lot_stability.csv")
        self.assertEqual(summary["source_split"], "test")
        self.assertFalse(summary["used_for_model_selection"])
        self.assertFalse(summary["used_for_threshold_selection"])
        self.assertEqual(summary["rows"], 24705)
        self.assertEqual(summary["lot_count"], len(lots))
        self.assertEqual(summary["geometry_count"], len(geometries))
        self.assertEqual(len(classes), 9)
        self.assertAlmostEqual(summary["global_accuracy"], 0.9772920461445052)
        self.assertLessEqual(summary["lot_support_max"], 25)
        self.assertTrue((result_dir / "lot_generalization_dashboard.png").is_file())

    def test_dashboard_contract(self) -> None:
        code = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("Held-out lot 일반화 감사", code)
        self.assertIn("개별 lot은 최대", code)
        self.assertIn("used_for_model_selection=false", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
