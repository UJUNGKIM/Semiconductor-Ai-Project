"""Contract tests for the WM-811K robust hierarchical candidate experiment."""

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
if str(WM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPT_DIR))

from train_wm811k_robust_hierarchical_candidates import (  # noqa: E402
    CANDIDATES,
    RobustWaferDataset,
    combine_hierarchical_probabilities,
    drop_active_dies,
    select_candidate,
    zero_fill_shift,
)


class Wm811kRobustHierarchicalCandidateTests(unittest.TestCase):
    def test_zero_fill_shift_never_wraps(self) -> None:
        channels = np.zeros((2, 6, 6), dtype=np.float32)
        channels[:, 0, 0] = 1
        shifted = zero_fill_shift(channels, -2, -2)
        self.assertEqual(float(shifted.sum()), 0.0)
        channels.fill(0)
        channels[:, 2, 2] = 1
        shifted = zero_fill_shift(channels, 2, 1)
        self.assertTrue(np.all(shifted[:, 4, 3] == 1))
        self.assertTrue(np.all(shifted[:, 2, 1] == 0))

    def test_active_die_dropout_is_deterministic_and_preserves_a_defect(self) -> None:
        channels = np.zeros((2, 20, 20), dtype=np.float32)
        channels[0, 2:18, 2:18] = 1
        channels[1, 8:10, 8:10] = 1
        first = drop_active_dies(channels, 0.20, np.random.default_rng(42))
        second = drop_active_dies(channels, 0.20, np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)
        self.assertGreater(int(first[1].sum()), 0)
        self.assertTrue(np.all(first[1] <= first[0]))
        self.assertLess(int(first[0].sum()), int(channels[0].sum()))

    def test_hierarchical_probabilities_are_coherent(self) -> None:
        binary = np.array([[0.80, 0.20], [0.10, 0.90]], dtype=np.float32)
        defect = np.zeros((2, 8), dtype=np.float32)
        defect[0, 2] = 1
        defect[1, 7] = 1
        combined = combine_hierarchical_probabilities(binary, defect)
        np.testing.assert_allclose(combined.sum(axis=1), 1.0)
        self.assertEqual(int(combined[0].argmax()), 8)
        self.assertEqual(int(combined[1].argmax()), 7)
        self.assertAlmostEqual(float(combined[1, 7]), 0.90, places=6)

    def test_dataset_stress_transform_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            maps_path = Path(temporary) / "maps.npy"
            maps = np.zeros((1, 64, 64), dtype=np.uint8)
            maps[0, 8:56, 8:56] = 1
            maps[0, 20:30, 20:30] = 2
            np.save(maps_path, maps)
            first = RobustWaferDataset(
                maps_path, np.array([0]), np.array([7]), task="flat",
                stress_mode="dropout1", seed=42,
            )
            second = RobustWaferDataset(
                maps_path, np.array([0]), np.array([7]), task="flat",
                stress_mode="dropout1", seed=42,
            )
            first_image, _ = first[0]
            second_image, _ = second[0]
            np.testing.assert_array_equal(first_image.numpy(), second_image.numpy())
            first.close()
            second.close()

    def test_selection_rejects_clean_guardrail_failure(self) -> None:
        rows = []
        for seed in (17, 42, 2026):
            for candidate in CANDIDATES:
                weak_gain = 0.03 if candidate != "baseline" else 0.0
                macro_loss = -0.02 if candidate == "robust_hierarchical" else 0.0
                rows.append(
                    {
                        "seed": seed,
                        "candidate": candidate,
                        "clean_macro_f1": 0.88 + macro_loss,
                        "clean_balanced_accuracy": 0.86,
                        "clean_weak_recall": 0.75 + weak_gain,
                        "clean_none_recall": 0.994,
                        "shift4_macro_f1": 0.70 + weak_gain,
                        "dropout1_macro_f1": 0.80 + weak_gain,
                    }
                )
        result = select_candidate(pd.DataFrame(rows))
        decisions = {row["candidate"]: row for row in result["candidate_decisions"]}
        self.assertFalse(decisions["robust_hierarchical"]["eligible"])
        self.assertEqual(result["selected_candidate"], "robust_flat")
        self.assertFalse(result["test_evaluated"])

    def test_notebook_and_script_lock_test_split(self) -> None:
        notebook = json.loads(
            (PROJECT_DIR / "노트북" / "WM811K_12_견고성_2단계분류_Colab.ipynb")
            .read_text(encoding="utf-8")
        )
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertIn("train_wm811k_robust_hierarchical_candidates.py", text)
        self.assertIn("seed 17, 42, 2026", text)
        self.assertIn("test 추론은 수행하지 않습니다", text)
        self.assertIn("wm811k_robust_hierarchical_results", text)
        script = (
            PROJECT_DIR / "코드" / "wm811k" / "train_wm811k_robust_hierarchical_candidates.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"used_test_for_selection": False', script)
        self.assertIn('"test_evaluated": False', script)
        self.assertNotIn('"test", "flat"', script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
