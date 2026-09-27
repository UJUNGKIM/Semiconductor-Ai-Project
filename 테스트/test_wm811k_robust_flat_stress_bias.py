"""Contract tests for WM-811K joint validation-stress bias calibration."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
if str(WM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPT_DIR))

from calibrate_wm811k_robust_flat_stress_bias import (  # noqa: E402
    STRESS_MODES,
    choose_best_bias,
    evaluate_guardrails,
)


class Wm811kRobustFlatStressBiasTests(unittest.TestCase):
    def test_guardrail_rejects_one_shift_none_recall_failure(self) -> None:
        rows = []
        for seed in (17, 42, 2026):
            for mode in STRESS_MODES:
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
        checks = evaluate_guardrails(pd.DataFrame(rows))
        self.assertFalse(checks["shift4_none_recall_guardrail_all_seeds"])
        self.assertFalse(all(checks.values()))

    def test_selector_prioritizes_worst_stress_mode_weak_recall(self) -> None:
        sweep = pd.DataFrame(
            [
                {
                    "none_logit_bias": 1.1,
                    "eligible": True,
                    "worst_mode_mean_delta_weak_recall": 0.02,
                    "mean_delta_weak_recall": 0.06,
                    "mean_delta_macro_f1": 0.04,
                    "minimum_delta_none_recall": -0.001,
                },
                {
                    "none_logit_bias": 1.2,
                    "eligible": True,
                    "worst_mode_mean_delta_weak_recall": 0.03,
                    "mean_delta_weak_recall": 0.04,
                    "mean_delta_macro_f1": 0.03,
                    "minimum_delta_none_recall": -0.0019,
                },
            ]
        )
        self.assertEqual(choose_best_bias(sweep), 1.2)

    def test_notebook_never_opens_test_split(self) -> None:
        notebook = json.loads(
            (
                PROJECT_DIR
                / "노트북"
                / "WM811K_14_스트레스공동보정_Colab.ipynb"
            ).read_text(encoding="utf-8")
        )
        text = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertIn("--bias-max', '2.0'", text)
        self.assertIn("test 추론은 수행하지 않습니다", text)
        self.assertIn("stress_calibration_gate_passed", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
