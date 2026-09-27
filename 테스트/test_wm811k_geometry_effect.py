"""Tests for class-mix-adjusted geometry analysis."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "코드" / "wm811k"))

from analyze_wm811k_geometry_effect import adjusted_geometry_metrics  # noqa: E402


class Wm811kGeometryEffectTests(unittest.TestCase):
    def test_class_mix_adjustment_removes_composition_only_difference(self) -> None:
        rows = pd.DataFrame({"original_height": [10] * 4 + [20] * 4, "original_width": [10] * 4 + [20] * 4, "label_id": [0, 0, 1, 1] * 2, "predicted_label_id": [0, 0, 1, 0] * 2})
        result = adjusted_geometry_metrics(rows, minimum_support=2)
        self.assertEqual(len(result), 2)
        self.assertTrue((result["class_adjusted_gap"].abs() < 1e-12).all())

    def test_artifact_and_dashboard_contract(self) -> None:
        result_dir = PROJECT_DIR / "결과물" / "wm811k" / "geometry_adjusted_results"
        summary = json.loads((result_dir / "geometry_candidate_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["selection_split"], "validation")
        self.assertFalse(summary["test_used_for_candidate_selection"])
        self.assertFalse(summary["threshold_preregistered_before_dataset_analysis"])
        self.assertTrue(summary["candidate_only"])
        self.assertFalse(summary["deployed"])
        self.assertEqual(set(summary["selected_geometries"]), {"26x30", "43x42"})
        self.assertTrue(summary["all_selected_geometries_negative_on_test"])
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("웨이퍼 크기 구성 보정", app)
        self.assertIn("후보 안전 규칙이며 현재 판정에는 적용하지 않았습니다", app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
