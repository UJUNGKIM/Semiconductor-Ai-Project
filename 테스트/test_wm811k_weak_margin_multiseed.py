"""Tests for paired WM-811K weak-margin multi-seed experiment."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "코드" / "wm811k"))

from run_wm811k_weak_margin_multiseed import paired_decision  # noqa: E402


class Wm811kWeakMarginMultiseedTests(unittest.TestCase):
    def test_paired_gate_accepts_consistent_safe_improvement(self) -> None:
        frame = pd.DataFrame({
            "delta_validation_weak_recall": [0.01, 0.02, 0.0],
            "delta_validation_macro_f1": [0.001, -0.002, 0.0],
            "delta_validation_none_recall": [0.0, -0.001, 0.001],
            "delta_validation_balanced_accuracy": [0.001, -0.004, 0.002],
        })
        decision = paired_decision(frame)
        self.assertEqual(decision["selected_candidate"], "weak_none_margin_005")
        self.assertTrue(decision["checks"]["eligible"])

    def test_paired_gate_rejects_one_seed_guardrail_failure(self) -> None:
        frame = pd.DataFrame({
            "delta_validation_weak_recall": [0.02, 0.02, 0.02],
            "delta_validation_macro_f1": [0.0, -0.006, 0.0],
            "delta_validation_none_recall": [0.0, 0.0, 0.0],
            "delta_validation_balanced_accuracy": [0.0, 0.0, 0.0],
        })
        decision = paired_decision(frame)
        self.assertEqual(decision["selected_candidate"], "baseline")
        self.assertFalse(decision["checks"]["macro_f1_guardrail_all_seeds"])

    def test_colab_notebook_preserves_protocol_and_test_lock(self) -> None:
        notebook = json.loads(
            (PROJECT_DIR / "노트북" / "WM811K_11_margin_다중시드_Colab.ipynb").read_text(encoding="utf-8")
        )
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("run_wm811k_weak_margin_multiseed.py", text)
        self.assertIn("17", text)
        self.assertIn("2026", text)
        self.assertIn("--seed42-source", text)
        self.assertIn("Test 추론은 수행하지 않습니다", text)
        script = (PROJECT_DIR / "코드" / "wm811k" / "run_wm811k_weak_margin_multiseed.py").read_text(encoding="utf-8")
        self.assertNotIn('split"].eq("test")', script)
        self.assertIn('"test_evaluated": False', script)
        self.assertIn("balanced_accuracy_guardrail_all_seeds", script)


if __name__ == "__main__":
    unittest.main(verbosity=2)
