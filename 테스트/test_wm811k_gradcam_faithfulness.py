"""Tests for WM-811K Grad-CAM sanity diagnostics."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "코드" / "wm811k"))

from diagnose_wm811k import artificial_demo_map  # noqa: E402
from evaluate_wm811k_gradcam_faithfulness import attribution_mask  # noqa: E402


class Wm811kGradcamFaithfulnessTests(unittest.TestCase):
    def test_attribution_mask_is_ranked_and_non_mutating(self) -> None:
        wafer = artificial_demo_map(64)
        original = wafer.copy()
        heatmap = np.arange(64 * 64, dtype=float).reshape(64, 64)
        top = attribution_mask(wafer, heatmap, 0.1, largest=True)
        bottom = attribution_mask(wafer, heatmap, 0.1, largest=False)
        self.assertEqual(int((top == 0).sum()), int((bottom == 0).sum()))
        self.assertFalse(np.array_equal(top, bottom))
        np.testing.assert_array_equal(wafer, original)

    def test_artifact_and_dashboard_contract(self) -> None:
        result_dir = PROJECT_DIR / "결과물" / "wm811k" / "gradcam_faithfulness_results"
        summary = json.loads((result_dir / "gradcam_faithfulness_summary.json").read_text(encoding="utf-8"))
        records = pd.read_csv(result_dir / "gradcam_faithfulness_records.csv")
        self.assertEqual(summary["sample_count"], 27)
        self.assertFalse(summary["used_for_model_selection"])
        self.assertFalse(summary["thresholds_retuned"])
        self.assertFalse(summary["representative_population_estimate"])
        self.assertEqual(len(records), 27)
        self.assertTrue(records["heatmap_nonzero"].all())
        self.assertLess(summary["mean_top_vs_random_advantage"], 0)
        self.assertLess(summary["top_drop_exceeds_random_rate"], 0.5)
        self.assertGreater(summary["top_drop_exceeds_bottom_rate"], 0.5)
        self.assertTrue((result_dir / "gradcam_faithfulness_dashboard.png").is_file())
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("Grad-CAM 충실도·sanity 점검", app)
        self.assertIn("인과관계를 증명하지 않습니다", app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
