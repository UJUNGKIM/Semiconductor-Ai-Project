"""Tests for the WM-811K Near-full post-selection diagnostic."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "코드" / "wm811k"))

from analyze_wm811k_near_full import bootstrap_ci  # noqa: E402


class Wm811kNearFullAnalysisTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_contains_mean(self) -> None:
        values = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        first = bootstrap_ci(values, seed=42, repeats=500)
        second = bootstrap_ci(values, seed=42, repeats=500)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], values.mean())
        self.assertGreaterEqual(first[1], values.mean())

    def test_result_contract(self) -> None:
        result_dir = PROJECT_DIR / "결과물" / "wm811k" / "near_full_diagnostic_results"
        summary = json.loads((result_dir / "near_full_summary.json").read_text(encoding="utf-8"))
        records = pd.read_csv(result_dir / "near_full_sample_diagnostics.csv")
        geometry = pd.read_csv(result_dir / "near_full_geometry_summary.csv")
        self.assertEqual(summary["test_sample_count"], 21)
        self.assertEqual(len(records), 21)
        self.assertEqual(int(geometry["support"].sum()), 21)
        self.assertAlmostEqual(summary["test_accuracy"], 19 / 21)
        self.assertFalse(summary["used_for_model_selection"])
        self.assertFalse(summary["thresholds_retuned"])
        self.assertFalse(summary["deployment_model_changed"])
        self.assertEqual(summary["combined_review_count"], 6)
        self.assertEqual(summary["ood_review_count"], 3)
        self.assertLessEqual(summary["paired_ig_minus_gradcam_95ci"][0], 0)
        self.assertGreaterEqual(summary["paired_ig_minus_gradcam_95ci"][1], 0)
        self.assertTrue((result_dir / "near_full_diagnostic_dashboard.png").is_file())

    def test_dashboard_states_global_pattern_limitation(self) -> None:
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("Near-full 집중 진단", app)
        self.assertIn("웨이퍼 전반에 퍼진 전역 패턴", app)
        self.assertIn("모델·임계값은 변경하지 않습니다", app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
