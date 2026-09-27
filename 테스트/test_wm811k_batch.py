"""Tests for WM-811K batch inference and dashboard integration."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
if str(WM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPT_DIR))

from diagnose_wm811k import (  # noqa: E402
    artificial_demo_map,
    load_checkpoint,
    predict_wafer_batch,
    predict_with_gradcam,
    predict_with_integrated_gradients,
)


class Wm811kBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result_dir = PROJECT_DIR / "결과물" / "wm811k" / "selected_model_results"
        cls.demo_dir = PROJECT_DIR / "결과물" / "wm811k" / "demo_samples"
        cls.model, cls.class_names, _ = load_checkpoint(
            cls.result_dir / "best_model.pt"
        )
        cls.policy = json.loads(
            (cls.result_dir / "uncertainty_policy.json").read_text(encoding="utf-8")
        )

    def test_batch_prediction_contract(self) -> None:
        wafers = [artificial_demo_map(24), artificial_demo_map(32)]
        predictions = predict_wafer_batch(
            self.model,
            self.class_names,
            wafers,
            temperature=float(self.policy["temperature"]),
            batch_size=1,
        )
        self.assertEqual(len(predictions), 2)
        for prediction in predictions:
            self.assertIn(prediction["predicted_label"], self.class_names)
            self.assertGreaterEqual(float(prediction["confidence"]), 0.0)
            self.assertLessEqual(float(prediction["confidence"]), 1.0)
            self.assertAlmostEqual(
                float(np.asarray(prediction["probabilities"]).sum()), 1.0, places=5
            )
        self.assertEqual(predict_wafer_batch(self.model, self.class_names, []), [])
        with self.assertRaisesRegex(ValueError, "batch_size"):
            predict_wafer_batch(self.model, self.class_names, wafers, batch_size=0)

    def test_real_demo_batch_reproduces_saved_predictions(self) -> None:
        manifest = json.loads(
            (self.demo_dir / "manifest.json").read_text(encoding="utf-8")
        )
        wafers = [
            np.load(self.demo_dir / item["npy_file"], allow_pickle=False)
            for item in manifest["samples"]
        ]
        predictions = predict_wafer_batch(
            self.model,
            self.class_names,
            wafers,
            temperature=float(self.policy["temperature"]),
        )
        self.assertEqual(len(predictions), len(manifest["samples"]))
        class_counts = pd.Series([item["true_class"] for item in manifest["samples"]]).value_counts()
        self.assertEqual(len(class_counts), 9)
        self.assertTrue(class_counts.between(4, 5).all())
        for item, prediction in zip(manifest["samples"], predictions):
            self.assertEqual(prediction["predicted_label"], item["predicted_class"])
            self.assertAlmostEqual(
                float(prediction["confidence"]),
                float(item["calibrated_confidence"]),
                places=4,
            )

    def test_integrated_gradients_matches_prediction_and_returns_heatmap(self) -> None:
        wafer = artificial_demo_map(32)
        temperature = float(self.policy["temperature"])
        gradcam = predict_with_gradcam(
            self.model, self.class_names, wafer, temperature=temperature
        )
        integrated = predict_with_integrated_gradients(
            self.model,
            self.class_names,
            wafer,
            temperature=temperature,
            steps=4,
        )
        self.assertEqual(integrated["predicted_class"], gradcam["predicted_class"])
        np.testing.assert_allclose(integrated["probabilities"], gradcam["probabilities"], atol=1e-7)
        self.assertEqual(integrated["heatmap"].shape, (64, 64))
        self.assertTrue(np.isfinite(integrated["heatmap"]).all())
        self.assertTrue(np.isfinite(integrated["completeness_delta"]))
        with self.assertRaisesRegex(ValueError, "steps"):
            predict_with_integrated_gradients(self.model, self.class_names, wafer, steps=1)

    def test_dashboard_exposes_batch_workflow(self) -> None:
        code = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn('"배치 진단"', code)
        self.assertIn("accept_multiple_files=True", code)
        self.assertIn("wm811k_batch_diagnosis.csv", code)
        self.assertIn("predict_wafer_batch", code)
        self.assertIn("predict_with_gradient_shap", code)
        self.assertIn("same_geometry_all_active_dies_normal", (
            PROJECT_DIR / "코드" / "wm811k" / "diagnose_wm811k.py"
        ).read_text(encoding="utf-8"))

    def test_dashboard_runs_real_demo_batch(self) -> None:
        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=30)
        app.run()
        app.switch_page("dashboard_ui/site_pages/wm811k.py")
        app.run()
        app.button(key="wm811k_batch_run").click()
        app.run()
        self.assertEqual(len(app.exception), 0)
        batch_metrics = {metric.label: metric.value for metric in app.metric}
        manifest = json.loads(
            (self.demo_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(batch_metrics["처리 완료"], f"{len(manifest['samples'])}개")


if __name__ == "__main__":
    unittest.main(verbosity=2)
