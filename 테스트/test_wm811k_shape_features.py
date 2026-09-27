"""Tests for WM-811K global shape descriptors and integration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
sys.path.insert(0, str(WM_SCRIPT_DIR))

from wafer_shape import wafer_shape_features  # noqa: E402
from analyze_wm811k_near_full_shape import analyze  # noqa: E402
from validate_wm811k_near_full_shape_bundle import validate_bundle  # noqa: E402


class Wm811kShapeFeatureTests(unittest.TestCase):
    def test_connected_center_block(self) -> None:
        wafer = np.ones((12, 12), dtype=np.uint8)
        wafer[4:8, 4:8] = 2
        features = wafer_shape_features(wafer)
        self.assertEqual(features["active_die_count"], 144)
        self.assertEqual(features["defect_die_count"], 16)
        self.assertEqual(features["component_count_8"], 1)
        self.assertEqual(features["largest_component_fraction"], 1.0)
        self.assertEqual(features["edge_defect_fraction"], 0.0)
        self.assertEqual(features["boundary_defect_coverage"], 0.0)

    def test_two_components_and_empty_defect_case(self) -> None:
        wafer = np.ones((10, 10), dtype=np.uint8)
        wafer[2, 2] = 2
        wafer[7, 7] = 2
        features = wafer_shape_features(wafer)
        self.assertEqual(features["component_count_8"], 2)
        self.assertEqual(features["largest_component_fraction"], 0.5)
        clean = wafer_shape_features(np.ones((10, 10), dtype=np.uint8))
        self.assertEqual(clean["defect_die_count"], 0)
        self.assertEqual(clean["component_count_8"], 0)
        self.assertEqual(clean["radial_entropy_4bin"], 0.0)

    def test_metrics_are_bounded(self) -> None:
        wafer = np.zeros((16, 16), dtype=np.uint8)
        yy, xx = np.indices(wafer.shape)
        radius = np.sqrt((yy - 7.5) ** 2 + (xx - 7.5) ** 2)
        wafer[radius <= 7] = 1
        wafer[(radius >= 5.5) & (radius <= 7)] = 2
        features = wafer_shape_features(wafer)
        for name in (
            "defect_ratio",
            "largest_component_fraction",
            "mean_normalized_radius",
            "center_defect_fraction",
            "edge_defect_fraction",
            "boundary_defect_coverage",
            "defect_bbox_fraction",
            "defect_centroid_offset",
            "radial_entropy_4bin",
        ):
            self.assertGreaterEqual(float(features[name]), 0.0, name)
            self.assertLessEqual(float(features[name]), 1.0 + 1e-12, name)

    def test_colab_and_dashboard_contract(self) -> None:
        notebook_path = (
            PROJECT_DIR / "노트북" / "WM811K_08_Near-full_전역형태_Colab.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        notebook_text = notebook_path.read_text(encoding="utf-8")
        self.assertIn("analyze_wm811k_near_full_shape.py", notebook_text)
        self.assertIn("near_full_global_shape_results", notebook_text)
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("전역 형태 지표", app)
        self.assertIn("canonical_wafer_shape_features", app)
        self.assertIn("8-이웃 연결 영역 수", app)

    def test_analysis_writes_one_row_per_fixed_test_wafer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            maps = np.ones((21, 8, 8), dtype=np.uint8)
            for index in range(21):
                maps[index, 2 + index % 3, 2 + (index * 2) % 4] = 2
            labels = np.full(21, 5, dtype=np.int64)
            assignments = pd.DataFrame(
                {
                    "array_index": np.arange(21),
                    "split": ["test"] * 21,
                    "label_id": [5] * 21,
                    "lot_name": [f"lot-{index // 2}" for index in range(21)],
                }
            )
            maps_path = root / "maps.npy"
            labels_path = root / "labels.npy"
            assignments_path = root / "assignments.csv"
            output_dir = root / "output"
            np.save(maps_path, maps)
            np.save(labels_path, labels)
            assignments.to_csv(assignments_path, index=False)
            summary = analyze(
                maps_path, labels_path, assignments_path, output_dir
            )
            records = pd.read_csv(
                output_dir / "near_full_global_shape_samples.csv"
            )
            self.assertEqual(summary["sample_count"], 21)
            self.assertEqual(len(records), 21)
            self.assertEqual(records["array_index"].nunique(), 21)
            self.assertTrue(
                (output_dir / "near_full_global_shape_dashboard.png").is_file()
            )

    def test_received_bundle_and_dashboard_contract(self) -> None:
        bundle = (
            PROJECT_DIR
            / "결과물"
            / "wm811k"
            / "colab_가져오기"
            / "near_full_global_shape_results.zip"
        )
        assignments = (
            PROJECT_DIR
            / "결과물"
            / "wm811k"
            / "split_results"
            / "split_assignments.csv"
        )
        report = validate_bundle(bundle, assignments)
        self.assertTrue(report["validation_passed"])
        self.assertEqual(report["sample_count"], 21)
        self.assertEqual(report["unique_lot_count"], 20)
        self.assertFalse(report["deployment_model_changed"])
        result_dir = (
            PROJECT_DIR
            / "결과물"
            / "wm811k"
            / "near_full_global_shape_results"
        )
        self.assertTrue(
            (result_dir / "near_full_global_shape_dashboard.png").is_file()
        )
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("Near-full 전역 형태 근거", app)
        self.assertIn("결함 die 평균", app)
        self.assertIn("shape_metrics['defect_ratio']['mean']", app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
