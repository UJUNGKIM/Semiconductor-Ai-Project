"""Contract tests for the WM-811K validation-stress ensemble selector."""

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

from select_wm811k_robust_ensemble import (  # noqa: E402
    ensemble_predictions,
    guardrail_status,
    metrics_from_predictions,
    select_best_configuration,
)


class Wm811kRobustEnsembleTests(unittest.TestCase):
    def test_ensemble_endpoints_match_component_predictions(self) -> None:
        baseline = np.zeros((2, 9), dtype=np.float64)
        robust = np.zeros((2, 9), dtype=np.float64)
        baseline[0, 8], baseline[1, 2] = 1.0, 1.0
        robust[0, 2], robust[1, 7] = 1.0, 1.0
        np.testing.assert_array_equal(
            ensemble_predictions(baseline, robust, 0.0, 0.0),
            np.array([8, 2]),
        )
        np.testing.assert_array_equal(
            ensemble_predictions(baseline, robust, 1.0, 0.0),
            np.array([2, 7]),
        )

    def test_metrics_match_perfect_nine_class_predictions(self) -> None:
        true = np.arange(9, dtype=np.int64)
        result = metrics_from_predictions(true, true.copy())
        for value in result.values():
            self.assertAlmostEqual(value, 1.0)

    def test_guardrail_rejects_single_seed_normal_recall_drop(self) -> None:
        rows = []
        for seed in (17, 42, 2026):
            for mode in ("clean", "shift4", "dropout1"):
                rows.append(
                    {
                        "seed": seed,
                        "stress_mode": mode,
                        "delta_accuracy": 0.001,
                        "delta_balanced_accuracy": 0.01,
                        "delta_macro_f1": 0.01,
                        "delta_weak_recall": 0.02,
                        "delta_none_recall": (
                            -0.003 if seed == 42 and mode == "shift4" else 0.0
                        ),
                    }
                )
        checks, minimum_margin = guardrail_status(pd.DataFrame(rows))
        self.assertFalse(checks["shift4_none_recall_guardrail_all_seeds"])
        self.assertLess(minimum_margin, 0)

    def test_selector_uses_worst_mode_weak_recall(self) -> None:
        sweep = pd.DataFrame(
            [
                {
                    "robust_weight": 0.4,
                    "none_logit_bias": 0.5,
                    "eligible": True,
                    "worst_mode_mean_delta_weak_recall": 0.01,
                    "mean_delta_weak_recall": 0.05,
                    "mean_delta_macro_f1": 0.03,
                    "minimum_delta_none_recall": -0.001,
                },
                {
                    "robust_weight": 0.5,
                    "none_logit_bias": 0.6,
                    "eligible": True,
                    "worst_mode_mean_delta_weak_recall": 0.02,
                    "mean_delta_weak_recall": 0.04,
                    "mean_delta_macro_f1": 0.02,
                    "minimum_delta_none_recall": -0.0015,
                },
            ]
        )
        self.assertEqual(select_best_configuration(sweep), (0.5, 0.6))

    def test_notebook_keeps_test_locked(self) -> None:
        notebook = json.loads(
            (
                PROJECT_DIR
                / "노트북"
                / "WM811K_15_견고성앙상블선택_Colab.ipynb"
            ).read_text(encoding="utf-8")
        )
        text = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertIn("test 추론은 수행하지 않습니다", text)
        self.assertIn("select_wm811k_robust_ensemble.py", text)
        self.assertIn("ensemble_validation_gate_passed", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
