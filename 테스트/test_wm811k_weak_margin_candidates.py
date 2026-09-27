"""Tests for conservative WM-811K weak-vs-none margin candidates."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "코드" / "wm811k"))

from train_wm811k_weak_margin_candidates import (  # noqa: E402
    CANDIDATES,
    WeakNoneMarginLoss,
)


class Wm811kWeakMarginCandidateTests(unittest.TestCase):
    def test_margin_penalizes_weak_sample_when_none_logit_is_higher(self) -> None:
        logits = torch.zeros((1, 9), dtype=torch.float32)
        logits[0, 8] = 1.0
        target = torch.tensor([2])
        baseline = WeakNoneMarginLoss(0.0, 0.2)(logits, target)
        adjusted = WeakNoneMarginLoss(0.1, 0.2)(logits, target)
        self.assertGreater(float(adjusted), float(baseline))

    def test_margin_does_not_penalize_none_sample(self) -> None:
        logits = torch.zeros((1, 9), dtype=torch.float32)
        target = torch.tensor([8])
        baseline = WeakNoneMarginLoss(0.0, 0.2)(logits, target)
        adjusted = WeakNoneMarginLoss(0.1, 0.2)(logits, target)
        self.assertAlmostEqual(float(adjusted), float(baseline), places=6)

    def test_candidates_are_conservative_and_keep_baseline(self) -> None:
        self.assertEqual(CANDIDATES[0]["name"], "baseline")
        self.assertEqual([row["weak_margin_lambda"] for row in CANDIDATES], [0.0, 0.05, 0.10])
        self.assertTrue(all(row["weak_margin_value"] == 0.20 for row in CANDIDATES))

    def test_colab_notebook_locks_test_and_reuses_baseline(self) -> None:
        notebook = json.loads(
            (PROJECT_DIR / "노트북" / "WM811K_10_취약클래스_margin_Colab.ipynb").read_text(encoding="utf-8")
        )
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("train_wm811k_weak_margin_candidates.py", text)
        self.assertIn("--baseline-source", text)
        self.assertIn("Test 추론은 수행하지 않습니다", text)
        script = (PROJECT_DIR / "코드" / "wm811k" / "train_wm811k_weak_margin_candidates.py").read_text(encoding="utf-8")
        self.assertNotIn('loaders["test"]', script)
        self.assertIn('"test_evaluated": False', script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
