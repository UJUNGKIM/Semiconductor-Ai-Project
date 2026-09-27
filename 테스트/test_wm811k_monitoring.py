from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "코드" / "wm811k"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from wm811k_monitoring import assess_batch_drift, load_monitoring_reference


class Wm811kMonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_path = (
            PROJECT
            / "결과물"
            / "wm811k"
            / "monitoring_reference"
            / "monitoring_reference.json"
        )
        cls.reference = load_monitoring_reference(cls.reference_path)

    def test_reference_is_validation_only_and_hash_linked(self):
        reference = self.reference
        self.assertEqual(reference["source_split"], "validation")
        self.assertEqual(reference["validation_sample_count"], 24703)
        self.assertFalse(reference["test_split_used"])
        self.assertFalse(reference["labels_required_during_monitoring"])
        self.assertEqual(len(reference["source_record_sha256"]), 64)
        checkpoint = (
            PROJECT / "결과물" / "wm811k" / "selected_model_results" / "best_model.pt"
        )
        self.assertEqual(
            reference["model_checkpoint_sha256"],
            hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper(),
        )
        self.assertAlmostEqual(sum(reference["class_probabilities"].values()), 1.0)

    def test_matching_batch_is_stable_and_deterministic(self):
        names = self.reference["class_names"]
        probabilities = np.asarray(
            [self.reference["class_probabilities"][name] for name in names]
        )
        counts = np.floor(probabilities * 1000).astype(int)
        counts[np.argmax(counts)] += 1000 - int(counts.sum())
        labels = [name for name, count in zip(names, counts) for _ in range(int(count))]
        statuses = ["in_distribution"] * 1000
        scores = [self.reference["ood_score_quantiles"]["p50"]] * 1000
        first = assess_batch_drift(
            predicted_labels=labels,
            ood_statuses=statuses,
            ood_scores=scores,
            reference=self.reference,
        )
        second = assess_batch_drift(
            predicted_labels=labels,
            ood_statuses=statuses,
            ood_scores=scores,
            reference=self.reference,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "STABLE")
        self.assertTrue(first["automatic_batch_decision_allowed"])

    def test_concentrated_predictions_trigger_batch_hold(self):
        count = 100
        result = assess_batch_drift(
            predicted_labels=[self.reference["class_names"][0]] * count,
            ood_statuses=["out_of_distribution"] * count,
            ood_scores=[self.reference["ood_score_quantiles"]["p99"] + 1.0] * count,
            reference=self.reference,
        )
        self.assertEqual(result["status"], "HOLD")
        self.assertFalse(result["automatic_batch_decision_allowed"])
        self.assertTrue(result["signals"]["predicted_class_mix"])
        self.assertTrue(result["signals"]["severe_ood_rate"])

    def test_small_batch_is_not_overinterpreted(self):
        result = assess_batch_drift(
            predicted_labels=[self.reference["class_names"][0]] * 5,
            ood_statuses=["in_distribution"] * 5,
            ood_scores=[0.5] * 5,
            reference=self.reference,
        )
        self.assertEqual(result["status"], "INSUFFICIENT")
        self.assertFalse(result["automatic_batch_decision_allowed"])

    def test_dashboard_exposes_monitoring_and_hold_guardrail(self):
        app = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("assess_batch_drift", app)
        self.assertIn("배치 분포 변화 감지", app)
        self.assertIn("배치 자동판정 보류 · 분포 변화 검토", app)
        self.assertIn("wm811k_batch_monitoring.json", app)


if __name__ == "__main__":
    unittest.main()
