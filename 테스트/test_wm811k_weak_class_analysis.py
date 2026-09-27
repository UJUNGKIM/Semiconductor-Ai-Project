"""Tests for deployed WM-811K weak-class diagnostics."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "코드" / "wm811k"))

from analyze_wm811k_weak_classes import calibrated_probabilities, file_sha256  # noqa: E402


class Wm811kWeakClassTests(unittest.TestCase):
    def test_temperature_calibration_preserves_probability_contract(self) -> None:
        columns = {
            "probability_center": [0.6], "probability_donut": [0.05],
            "probability_edge_loc": [0.05], "probability_edge_ring": [0.05],
            "probability_loc": [0.05], "probability_near_full": [0.05],
            "probability_random": [0.05], "probability_scratch": [0.05],
            "probability_none": [0.05],
        }
        calibrated = calibrated_probabilities(pd.DataFrame(columns), 1.047171711723721)
        self.assertEqual(calibrated.shape, (1, 9))
        self.assertAlmostEqual(float(calibrated.sum()), 1.0)
        self.assertEqual(int(calibrated.argmax(axis=1)[0]), 0)

    def test_result_contract_and_source_provenance(self) -> None:
        result_dir = PROJECT_DIR / "결과물" / "wm811k" / "weak_class_results"
        summary = json.loads((result_dir / "weak_class_summary.json").read_text(encoding="utf-8"))
        classes = pd.read_csv(result_dir / "weak_class_summary.csv")
        samples = pd.read_csv(result_dir / "weak_class_sample_diagnostics.csv")
        predictions = PROJECT_DIR / "결과물" / "wm811k" / "colab_가져오기" / "wm811k_strategy_comparison_results" / "ce_sqrt_balanced" / "test_predictions.csv"
        checkpoint = PROJECT_DIR / "결과물" / "wm811k" / "selected_model_results" / "best_model.pt"
        self.assertEqual(summary["checkpoint_sha256"], file_sha256(checkpoint))
        self.assertEqual(summary["prediction_source_sha256"], file_sha256(predictions))
        self.assertEqual(summary["weak_class_rows"], 1424)
        self.assertEqual(summary["weak_class_errors"], 310)
        self.assertEqual(summary["errors_to_none"], 208)
        self.assertEqual(summary["automatic_error_count"], 25)
        self.assertEqual(set(classes["class_name"]), {"Scratch", "Loc", "Edge-Loc"})
        self.assertEqual(len(samples), 1424)
        self.assertFalse(samples["array_index"].duplicated().any())
        self.assertTrue((result_dir / "weak_class_dashboard.png").is_file())
        self.assertFalse(summary["used_for_model_selection"])
        self.assertFalse(summary["thresholds_retuned"])

    def test_dashboard_contract(self) -> None:
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("Scratch·Loc·Edge-Loc 집중 분석", app)
        self.assertIn("none hard-negative sampling", app)
        self.assertIn("test는 최종 확인까지 잠급니다", app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
