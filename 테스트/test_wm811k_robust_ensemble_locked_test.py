"""Contract tests for the one-time locked WM-811K ensemble test."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
if str(WM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPT_DIR))

from evaluate_wm811k_robust_ensemble_locked_test import (  # noqa: E402
    EXPECTED_DEPLOYED_SHA256,
    LOCKED_NONE_LOGIT_BIAS,
    LOCKED_ROBUST_WEIGHT,
    LOCKED_SEED,
    confusion_matrix,
    deployment_gate,
    ensemble_predictions,
    existing_result,
    metrics_from_confusion,
)


class Wm811kRobustEnsembleLockedTestTests(unittest.TestCase):
    def test_locked_configuration_is_registered(self) -> None:
        self.assertEqual(LOCKED_SEED, 42)
        self.assertEqual(LOCKED_ROBUST_WEIGHT, 0.5)
        self.assertEqual(LOCKED_NONE_LOGIT_BIAS, 0.1)
        self.assertEqual(len(EXPECTED_DEPLOYED_SHA256), 64)

    def test_metrics_are_perfect_for_matching_predictions(self) -> None:
        true = np.arange(9, dtype=np.int64)
        metrics = metrics_from_confusion(confusion_matrix(true, true.copy()))
        for value in metrics.values():
            self.assertAlmostEqual(value, 1.0)

    def test_candidate_ensemble_uses_locked_parameters(self) -> None:
        baseline = np.zeros((1, 9), dtype=np.float64)
        robust = np.zeros((1, 9), dtype=np.float64)
        baseline[0, 8] = 0.6
        baseline[0, 2] = 0.4
        robust[0, 8] = 0.4
        robust[0, 2] = 0.6
        self.assertEqual(int(ensemble_predictions(baseline, robust)[0]), 8)

    def test_deployment_gate_protects_normal_recall(self) -> None:
        deployed = {
            "accuracy": 0.95,
            "balanced_accuracy": 0.80,
            "macro_f1": 0.78,
            "weighted_f1": 0.94,
            "weak_recall": 0.60,
            "none_recall": 0.99,
        }
        candidate = {
            "accuracy": 0.951,
            "balanced_accuracy": 0.82,
            "macro_f1": 0.80,
            "weighted_f1": 0.941,
            "weak_recall": 0.65,
            "none_recall": 0.987,
        }
        checks, _ = deployment_gate(deployed, candidate)
        self.assertFalse(checks["none_recall_guardrail"])
        self.assertFalse(all(checks.values()))

    def test_existing_locked_result_prevents_repeat_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            summary = {
                "selected_robust_weight": 0.5,
                "selected_none_logit_bias": 0.1,
                "locked_seed": 42,
                "test_evaluated": True,
                "locked_test_gate_passed": True,
            }
            (output_dir / "locked_test_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            self.assertEqual(existing_result(output_dir), summary)

    def test_notebook_runs_one_locked_test(self) -> None:
        notebook = json.loads(
            (
                PROJECT_DIR
                / "노트북"
                / "WM811K_16_앙상블고정Test_Colab.ipynb"
            ).read_text(encoding="utf-8")
        )
        text = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        self.assertIn("단 한 번", text)
        self.assertIn("evaluate_wm811k_robust_ensemble_locked_test.py", text)
        self.assertIn("locked_test_gate_passed", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
