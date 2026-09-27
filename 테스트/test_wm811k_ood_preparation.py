"""Contract tests for the WM-811K OOD reference preparation."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
if str(WM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPT_DIR))

from build_wm811k_ood import (  # noqa: E402
    WaferCNN,
    calibrate_score_thresholds,
    embedding_distance_scores,
    fit_class_reference,
    score_status,
)
from diagnose_wm811k import (  # noqa: E402
    assess_wafer_ood_batch,
    load_checkpoint,
    load_ood_reference,
)


class Wm811kOodPreparationTests(unittest.TestCase):
    def test_builder_is_self_contained_for_colab(self) -> None:
        script = (WM_SCRIPT_DIR / "build_wm811k_ood.py").read_text(encoding="utf-8")
        self.assertNotIn("from diagnose_wm811k import", script)
        self.assertNotIn("joblib.dump", script)
        self.assertIn("np.savez_compressed", script)
        self.assertIn("ood_reference.json", script)
        model = WaferCNN(num_classes=9)
        self.assertEqual(model.classifier[-1].out_features, 9)

    def test_reference_math_contract(self) -> None:
        features = np.array(
            [
                [0.0, 0.1],
                [0.1, 0.0],
                [-0.1, 0.0],
                [5.0, 5.1],
                [5.1, 5.0],
                [4.9, 5.0],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 0, 1, 1, 1])
        means, variances, counts = fit_class_reference(features, labels, 2)
        self.assertEqual(means.shape, (2, 2))
        self.assertTrue(np.all(variances > 0))
        np.testing.assert_array_equal(counts, [3, 3])
        scores, nearest = embedding_distance_scores(features, means, variances)
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_array_equal(nearest, labels)
        chunked_scores, chunked_nearest = embedding_distance_scores(
            features, means, variances, chunk_size=2
        )
        np.testing.assert_allclose(chunked_scores, scores)
        np.testing.assert_array_equal(chunked_nearest, nearest)

    def test_validation_only_threshold_contract(self) -> None:
        validation_scores = np.arange(1, 101, dtype=float)
        thresholds = calibrate_score_thresholds(validation_scores, 0.95, 0.99)
        self.assertLess(thresholds["review_threshold"], thresholds["ood_threshold"])
        statuses = score_status(
            np.array([0.0, thresholds["review_threshold"], thresholds["ood_threshold"]]),
            thresholds,
        )
        self.assertEqual(
            statuses.tolist(), ["in_distribution", "review", "out_of_distribution"]
        )

    def test_colab_notebook_contract(self) -> None:
        notebook_path = PROJECT_DIR / "노트북" / "WM811K_05_OOD_기준생성_Colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertIn("torch.cuda.is_available()", code)
        self.assertIn("build_wm811k_ood.py", code)
        self.assertIn("split_assignments.csv", code)
        self.assertIn("best_model.pt", code)
        self.assertIn("wm811k_ood_results", code)
        self.assertIn("ood_reference.npz", code)
        self.assertIn("files.download", code)
        self.assertIn("'--device', 'cuda'", code)
        self.assertIn("next(iter(uploaded.items()))", code)
        self.assertIn("renamed_copy", code)

    def test_deployed_ood_reference_and_demo_assessment(self) -> None:
        ood_dir = PROJECT_DIR / "결과물" / "wm811k" / "ood_results"
        result_dir = PROJECT_DIR / "결과물" / "wm811k" / "selected_model_results"
        demo_dir = PROJECT_DIR / "결과물" / "wm811k" / "demo_samples"
        reference = load_ood_reference(
            ood_dir / "ood_reference.json", ood_dir / "ood_reference.npz"
        )
        model, class_names, _ = load_checkpoint(result_dir / "best_model.pt")
        manifest = json.loads(
            (demo_dir / "manifest.json").read_text(encoding="utf-8")
        )
        wafers = [
            np.load(demo_dir / item["npy_file"], allow_pickle=False)
            for item in manifest["samples"]
        ]
        assessments = assess_wafer_ood_batch(
            model, class_names, wafers, reference, batch_size=7
        )
        self.assertEqual(len(assessments), len(manifest["samples"]))
        self.assertTrue(
            all(
                item["ood_status"]
                in {"in_distribution", "review", "out_of_distribution"}
                for item in assessments
            )
        )
        self.assertTrue(all(np.isfinite(item["ood_score"]) for item in assessments))
        self.assertEqual(assess_wafer_ood_batch(model, class_names, [], reference), [])

        app_code = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("입력 분포 이탈(OOD) 안전 기준", app_code)
        self.assertIn("자동판정 보류 · 입력 확인", app_code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
