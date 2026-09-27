"""Contract tests for bias-calibrated WM-811K validation stress replay."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
if str(WM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPT_DIR))

from replay_wm811k_robust_flat_stress import (  # noqa: E402
    adjusted_metrics,
    evaluate_guardrails,
)


class Wm811kRobustFlatStressReplayTests(unittest.TestCase):
    def test_adjusted_probabilities_apply_registered_none_bias(self) -> None:
        true = np.array([8, 7])
        probabilities = np.zeros((2, 9), dtype=np.float64)
        probabilities[0, 2], probabilities[0, 8] = 0.55, 0.45
        probabilities[1, 7], probabilities[1, 8] = 0.90, 0.10
        metrics, predicted = adjusted_metrics(
            true, probabilities, 1.155,
            ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"],
        )
        np.testing.assert_array_equal(predicted, true)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_stress_gate_requires_each_none_recall_guardrail(self) -> None:
        rows = []
        for seed in (17, 42, 2026):
            for mode in ("clean", "shift4", "dropout1"):
                rows.append(
                    {
                        "seed": seed,
                        "stress_mode": mode,
                        "delta_weak_recall": 0.02,
                        "delta_macro_f1": 0.01,
                        "delta_none_recall": -0.003 if seed == 42 and mode == "shift4" else 0.0,
                        "delta_balanced_accuracy": 0.01,
                    }
                )
        checks = evaluate_guardrails(pd.DataFrame(rows))
        self.assertFalse(checks["shift_none_recall_guardrail_all_seeds"])
        self.assertFalse(all(checks.values()))

    def test_notebook_locks_bias_and_test_split(self) -> None:
        notebook = json.loads(
            (PROJECT_DIR / "노트북" / "WM811K_13_보정후_견고성재검증_Colab.ipynb")
            .read_text(encoding="utf-8")
        )
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertIn("--none-logit-bias', '1.155'", text)
        self.assertIn("test 추론은 수행하지 않습니다", text)
        self.assertIn("replay_wm811k_robust_flat_stress.py", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
