"""Tests for validation-only WM-811K weak-class candidate training."""

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

from train_wm811k_weak_class_candidates import (  # noqa: E402
    TrainOnlyWaferDataset,
    choose_candidate,
    hard_negative_indices,
    reusable_candidate_summary,
)


class Wm811kWeakClassCandidateTests(unittest.TestCase):
    def test_defect_thinning_preserves_active_map_and_at_least_one_defect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "maps.npy"
            maps = np.zeros((1, 64, 64), dtype=np.uint8)
            maps[0, 8:56, 8:56] = 1
            maps[0, 20:30, 20:30] = 2
            np.save(path, maps)
            np.random.seed(42)
            dataset = TrainOnlyWaferDataset(path, np.array([0]), np.array([7]), augment=True, defect_dropout_max=0.12)
            image, label = dataset[0]
            self.assertEqual(int(label), 7)
            self.assertEqual(int(image[0].sum()), int((maps[0] > 0).sum()))
            self.assertGreater(int(image[1].sum()), 0)
            self.assertLessEqual(int(image[1].sum()), int((maps[0] == 2).sum()))
            dataset.close()

    def test_hard_negatives_use_train_none_only_and_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "maps.npy"
            maps = np.ones((6, 64, 64), dtype=np.uint8)
            for index, defect_count in enumerate((1, 3, 5, 10, 20, 30)):
                maps[index].flat[:defect_count] = 2
            np.save(path, maps)
            assignments = pd.DataFrame({
                "array_index": range(6), "label_id": [8, 8, 8, 8, 8, 8],
                "split": ["train", "train", "train", "train", "validation", "test"],
            })
            first = hard_negative_indices(path, assignments, 0.25)
            second = hard_negative_indices(path, assignments, 0.25)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first, np.array([3]))

    def test_candidate_selection_applies_both_guardrails(self) -> None:
        comparison = pd.DataFrame([
            {"candidate": "baseline", "validation_macro_f1": 0.90, "validation_none_recall": 0.995, "validation_weak_recall": 0.75},
            {"candidate": "unsafe", "validation_macro_f1": 0.91, "validation_none_recall": 0.990, "validation_weak_recall": 0.90},
            {"candidate": "safe", "validation_macro_f1": 0.897, "validation_none_recall": 0.994, "validation_weak_recall": 0.80},
        ])
        selected, evaluated = choose_candidate(comparison)
        self.assertEqual(selected, "safe")
        self.assertFalse(bool(evaluated.loc[evaluated["candidate"].eq("unsafe"), "eligible"].iloc[0]))

    def test_completed_candidate_can_resume_without_retraining(self) -> None:
        candidate = {"name": "baseline", "defect_dropout_max": 0.0, "hard_negative_multiplier": 1.0}
        with tempfile.TemporaryDirectory() as temporary:
            candidate_dir = Path(temporary) / "baseline"
            candidate_dir.mkdir()
            summary = {
                "candidate": candidate["name"],
                "defect_dropout_max": candidate["defect_dropout_max"],
                "hard_negative_multiplier": candidate["hard_negative_multiplier"],
                "seed": 42,
                "test_evaluated": False,
            }
            (candidate_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            for name in ("best_model.pt", "validation_classification_report.csv", "validation_predictions.csv"):
                (candidate_dir / name).write_bytes(b"fixture")
            self.assertEqual(reusable_candidate_summary(Path(temporary), candidate, 42), summary)
            self.assertIsNone(reusable_candidate_summary(Path(temporary), candidate, 7))

    def test_colab_notebook_and_script_lock_test_split(self) -> None:
        notebook = json.loads((PROJECT_DIR / "노트북" / "WM811K_09_취약클래스_개선후보_Colab.ipynb").read_text(encoding="utf-8"))
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("train_wm811k_weak_class_candidates.py", text)
        self.assertIn("wm811k_weak_class_validation_results", text)
        self.assertIn("Test 추론은 수행하지 않습니다", text)
        self.assertIn("stderr=subprocess.STDOUT", text)
        script = (PROJECT_DIR / "코드" / "wm811k" / "train_wm811k_weak_class_candidates.py").read_text(encoding="utf-8")
        self.assertNotIn('loaders["test"]', script)
        self.assertIn('"test_evaluated": False', script)
        self.assertIn('"used_test_for_selection": False', script)
        self.assertIn("guardrail_plot", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
