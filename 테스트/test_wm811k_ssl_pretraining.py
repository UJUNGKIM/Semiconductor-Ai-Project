from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "코드" / "wm811k"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from extract_wm811k_train_lot_unlabeled import (
    select_train_lot_unlabeled_indices,
)
from train_wm811k_ssl_candidate import (
    compare_with_baseline,
    contrastive_view,
    nt_xent_loss,
    wafer_channels,
)
from validate_wm811k_ssl_bundle import _safe_name
from calibrate_wm811k_ssl_train_oof import (
    select_train_oof_bias,
)
from validate_wm811k_ssl_oof_calibration import _safe_name as oof_safe_name


def _wrapped(value: str | None) -> object:
    return np.asarray([] if value is None else [[value]], dtype=object)


class Wm811kSslPretrainingTests(unittest.TestCase):
    def test_unlabeled_selection_uses_train_lots_only(self):
        rows = []
        for lot, label in [
            ("train-a", None),
            ("validation-a", None),
            ("test-a", None),
            ("train-a", "Loc"),
            ("train-b", None),
            (None, None),
        ]:
            rows.append(
                {
                    "waferMap": np.asarray([[0, 1], [2, 1]], dtype=np.uint8),
                    "dieSize": 4,
                    "lotName": _wrapped(lot),
                    "waferIndex": _wrapped("1"),
                    "trianTestLabel": _wrapped("Training"),
                    "failureType": _wrapped(label),
                }
            )
        source = pd.DataFrame(rows)
        assignments = pd.DataFrame(
            {
                "lot_name": ["train-a", "train-b", "validation-a", "test-a"],
                "split": ["train", "train", "validation", "test"],
            }
        )
        selected, summary = select_train_lot_unlabeled_indices(
            source, assignments, maximum_samples=10, seed=42
        )
        self.assertEqual(selected.tolist(), [0, 4])
        self.assertEqual(summary["held_out_lot_overlap"], {"validation": 0, "test": 0})
        self.assertEqual(summary["eligible_unlabeled_train_lot_rows"], 2)
        self.assertEqual(summary["unknown_or_non_train_lot_unlabeled_excluded"], 3)

    def test_unlabeled_cap_is_deterministic(self):
        source = pd.DataFrame(
            [
                {
                    "waferMap": np.asarray([[0, 1], [2, 1]], dtype=np.uint8),
                    "dieSize": 4,
                    "lotName": _wrapped("train-a"),
                    "waferIndex": _wrapped(str(index)),
                    "trianTestLabel": _wrapped("Training"),
                    "failureType": _wrapped(None),
                }
                for index in range(20)
            ]
        )
        assignments = pd.DataFrame(
            {
                "lot_name": ["train-a", "validation-a", "test-a"],
                "split": ["train", "validation", "test"],
            }
        )
        first, _ = select_train_lot_unlabeled_indices(
            source, assignments, maximum_samples=5, seed=17
        )
        second, _ = select_train_lot_unlabeled_indices(
            source, assignments, maximum_samples=5, seed=17
        )
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 5)

    def test_contrastive_transform_preserves_shape_and_binary_values(self):
        wafer = np.asarray(
            [[0, 0, 0, 0], [0, 1, 2, 0], [0, 2, 1, 0], [0, 0, 0, 0]],
            dtype=np.uint8,
        )
        channels = wafer_channels(wafer)
        view = contrastive_view(channels, dropout_probability=0.2)
        self.assertEqual(tuple(view.shape), (2, 4, 4))
        self.assertTrue(set(torch.unique(view).tolist()).issubset({0.0, 1.0}))
        self.assertTrue(torch.all(view[1] <= view[0]))

    def test_nt_xent_prefers_matching_pairs(self):
        first = torch.eye(4)
        matching = torch.eye(4)
        mismatched = torch.roll(torch.eye(4), shifts=1, dims=0)
        good = nt_xent_loss(first, matching, temperature=0.2)
        bad = nt_xent_loss(first, mismatched, temperature=0.2)
        self.assertLess(float(good), float(bad))

    def test_candidate_comparison_never_promotes_directly(self):
        baseline = {
            "metrics": {
                "validation_macro_f1": {"mean": 0.89},
                "validation_balanced_accuracy": {"mean": 0.90},
            }
        }
        recall_rows = [
            {"split": "validation", "class_name": name, "recall_mean": value}
            for name, value in {
                "Center": 0.95,
                "Donut": 0.93,
                "Edge-Loc": 0.78,
                "Edge-Ring": 0.98,
                "Loc": 0.77,
                "Near-full": 0.92,
                "Random": 0.91,
                "Scratch": 0.86,
                "none": 0.991,
            }.items()
        ]
        run = {
            "validation_metrics": {
                "accuracy": 0.98,
                "balanced_accuracy": 0.905,
                "macro_f1": 0.895,
                "weighted_f1": 0.98,
            },
            "validation_class_recall": {
                row["class_name"]: row["recall_mean"] for row in recall_rows
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_path = root / "baseline.json"
            recall_path = root / "recall.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            recall_path.write_text(json.dumps(recall_rows), encoding="utf-8")
            result = compare_with_baseline(
                [run, run, run], baseline_path, recall_path
            )
        self.assertTrue(result["all_guardrails_passed"])
        self.assertEqual(
            result["decision"], "validation_supported_external_holdout_required"
        )
        self.assertFalse(result["fixed_test_reopened"])
        self.assertFalse(result["deployment_model_changed"])

    def test_training_script_has_no_test_evaluation_path(self):
        script = (
            SCRIPTS / "train_wm811k_ssl_candidate.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('loaders["test"]', script)
        self.assertNotIn("save_evaluation_artifacts(\n        assignments,\n        \"test\"", script)
        self.assertIn('"fixed_test_reopened": False', script)
        self.assertIn('"test_evaluated": False', script)

    def test_colab_notebook_documents_leakage_contract(self):
        notebook = (
            PROJECT / "노트북" / "WM811K_17_비라벨_자기지도_Colab.ipynb"
        ).read_text(encoding="utf-8")
        self.assertIn("train lot", notebook)
        self.assertIn("validation-only", notebook)
        self.assertIn("test", notebook)
        self.assertIn("extract_wm811k_train_lot_unlabeled.py", notebook)
        self.assertIn("train_wm811k_ssl_candidate.py", notebook)

    def test_colab_bundle_is_flat_complete_and_validation_only(self):
        bundle = PROJECT / "노트북" / "wm811k_ssl_colab_bundle.zip"
        required = {
            "preprocess_wm811k.py",
            "extract_wm811k_train_lot_unlabeled.py",
            "train_wm811k_cnn.py",
            "train_wm811k_ssl_candidate.py",
            "split_assignments.csv",
            "baseline_reproducibility_summary.json",
            "baseline_class_recall_summary.json",
        }
        with zipfile.ZipFile(bundle) as archive:
            self.assertEqual(set(archive.namelist()), required)
            baseline = json.loads(
                archive.read("baseline_reproducibility_summary.json")
            )
            recalls = json.loads(
                archive.read("baseline_class_recall_summary.json")
            )
        self.assertFalse(baseline["test_metrics_included"])
        self.assertFalse(any("test" in key for key in baseline["metrics"]))
        self.assertTrue(recalls)
        self.assertEqual({row["split"] for row in recalls}, {"validation"})

    def test_result_bundle_path_safety(self):
        self.assertTrue(_safe_name("seeds/seed_17/run_summary.json"))
        self.assertFalse(_safe_name("../best_model.pt"))
        self.assertFalse(_safe_name("/absolute/checkpoint.pt"))

    def test_verified_result_and_dashboard_contract(self):
        result_path = (
            PROJECT
            / "결과물"
            / "wm811k"
            / "ssl_validation_results"
            / "bundle_validation.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["bundle_file_count"], 27)
        self.assertEqual(len(result["bundle_sha256"]), 64)
        self.assertEqual(result["validation_prediction_rows_checked"], 74_109)
        self.assertGreater(result["metric_deltas"]["macro_f1"], 0)
        self.assertGreater(result["metric_deltas"]["balanced_accuracy"], 0)
        self.assertLess(result["metric_deltas"]["none_recall"], -0.002)
        self.assertFalse(result["all_guardrails_passed"])
        self.assertEqual(result["decision"], "retain_existing_model")
        self.assertFalse(result["fixed_test_reopened"])
        self.assertFalse(result["deployment_model_changed"])
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("WM_SSL_VALIDATION_DIR", app)
        self.assertIn("비라벨 자기지도 사전학습 후보", app)
        self.assertIn("기존 champion을 유지", app)

    def test_train_oof_bias_selection_is_deterministic_and_conservative(self):
        true = np.repeat(np.arange(9), 20)
        probabilities = np.full((len(true), 9), 0.005, dtype=np.float64)
        probabilities[np.arange(len(true)), true] = 0.96
        none_rows = np.flatnonzero(true == 8)
        probabilities[none_rows[:2], :] = 0.001
        probabilities[none_rows[:2], 0] = 0.55
        probabilities[none_rows[:2], 8] = 0.442
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        grid = np.asarray([0.0, 0.25, 0.5])
        first, first_sweep = select_train_oof_bias(
            true,
            probabilities,
            grid,
            target_none_recall=0.99,
            weak_recall_tolerance=0.01,
        )
        second, second_sweep = select_train_oof_bias(
            true,
            probabilities,
            grid,
            target_none_recall=0.99,
            weak_recall_tolerance=0.01,
        )
        self.assertEqual(first, 0.25)
        self.assertEqual(first, second)
        pd.testing.assert_frame_equal(first_sweep, second_sweep)
        selected_row = first_sweep.loc[
            first_sweep["none_logit_bias"].eq(first)
        ].iloc[0]
        self.assertTrue(selected_row["eligible"])
        self.assertGreaterEqual(selected_row["none_recall"], 0.99)

    def test_oof_calibration_script_and_notebook_keep_test_locked(self):
        script = (
            SCRIPTS / "calibrate_wm811k_ssl_train_oof.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('loaders["test"]', script)
        self.assertIn("labels[train_indices]", script)
        self.assertNotIn('labels, assignments["label_id"]', script)
        self.assertIn('"selection_source": "train_lot_oof_only"', script)
        self.assertIn('"validation_used_for_bias_selection": False', script)
        self.assertIn('"test_evaluated": False', script)
        notebook_path = (
            PROJECT / "노트북" / "WM811K_18_SSL_Train_OOF_보정_Colab.ipynb"
        )
        notebook = notebook_path.read_text(encoding="utf-8")
        self.assertIn("train lot", notebook)
        self.assertIn("OOF", notebook)
        self.assertIn("test", notebook)
        self.assertIn("wm811k_ssl_oof_calibration_results", notebook)

    def test_oof_calibration_bundle_is_compact_and_complete(self):
        bundle = PROJECT / "노트북" / "wm811k_ssl_oof_calibration_bundle.zip"
        expected = {
            "train_wm811k_cnn.py",
            "calibrate_wm811k_robust_flat_bias.py",
            "calibrate_wm811k_ssl_train_oof.py",
            "split_assignments.csv",
            "baseline_reproducibility_summary.json",
            "baseline_class_recall_summary.json",
        }
        with zipfile.ZipFile(bundle) as archive:
            self.assertEqual(set(archive.namelist()), expected)
        self.assertLess(bundle.stat().st_size, 3 * 1024 * 1024)

    def test_received_oof_calibration_result_is_fail_safe(self):
        result_path = (
            PROJECT
            / "결과물"
            / "wm811k"
            / "ssl_oof_calibration_results"
            / "bundle_validation.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["bundle_file_count"], 3)
        self.assertEqual(len(result["bundle_sha256"]), 64)
        self.assertEqual(result["selection_source"], "train_lot_oof_only")
        self.assertEqual(result["oof_folds"], 3)
        self.assertEqual(result["oof_rows_checked"], 123_542)
        self.assertEqual(result["eligible_bias_count"], 0)
        self.assertIsNone(result["selected_none_logit_bias"])
        self.assertLess(
            result["maximum_none_recall_candidate"]["none_recall"],
            result["target_none_recall"],
        )
        self.assertLess(
            result["maximum_none_recall_candidate"]["delta_weak_recall"],
            -result["weak_recall_tolerance"],
        )
        self.assertFalse(result["validation_evaluated"])
        self.assertFalse(result["test_evaluated"])
        self.assertFalse(result["deployment_model_changed"])
        self.assertEqual(result["decision"], "retain_existing_model")
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("WM_SSL_OOF_CALIBRATION_DIR", app)
        self.assertIn("train OOF 정상 클래스 보정", app)

    def test_oof_result_validator_rejects_unsafe_paths(self):
        self.assertTrue(oof_safe_name("oof_bias_sweep.csv"))
        self.assertFalse(oof_safe_name("../calibration_summary.json"))
        self.assertFalse(oof_safe_name("/absolute/oof_fold_metrics.csv"))


if __name__ == "__main__":
    unittest.main()
