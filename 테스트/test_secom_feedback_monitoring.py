import json
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from feedback_monitoring import (
    analyze_feedback_history,
    build_candidate_manifest,
    wilson_interval,
)
from monitoring_log import validate_monitoring_log


class FeedbackMonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result_dir = PROJECT / "결과물" / "secom" / "feedback_monitoring"
        cls.reference = json.loads(
            (cls.result_dir / "feedback_reference.json").read_text(encoding="utf-8")
        )
        cls.replay = pd.read_csv(cls.result_dir / "oof_feedback_replay.csv")

    def test_reference_is_train_only_and_never_auto_retrains(self):
        self.assertEqual(self.reference["created_from"], "repeated_5x3_oof_train_only")
        self.assertFalse(self.reference["test_labels_used_for_guardrails"])
        self.assertFalse(self.reference["automatic_retraining_allowed"])

    def test_oof_replay_is_stable(self):
        summary = analyze_feedback_history(self.replay, self.reference)
        self.assertEqual(summary["status"], "STABLE")
        self.assertEqual(summary["confirmed_rows"], 1253)
        self.assertAlmostEqual(summary["recall"], 69 / 83)

    def test_low_recall_requests_review_but_not_training(self):
        degraded = self.replay.copy()
        degraded["confirmed_alerted_defects"] = 0
        summary = analyze_feedback_history(degraded, self.reference)
        self.assertEqual(summary["status"], "REVIEW_RETRAINING")
        manifest = build_candidate_manifest(summary)
        self.assertFalse(manifest["automatic_retraining_started"])
        self.assertNotIn("sensor_values", manifest)

    def test_latest_model_version_is_not_mixed(self):
        changed = self.replay.copy()
        changed.loc[changed.index[-1], "catboost_model_hash"] = "new-version"
        summary = analyze_feedback_history(changed, self.reference)
        self.assertEqual(summary["status"], "REFERENCE_MISMATCH")
        self.assertEqual(summary["confirmed_batches"], 0)

    def test_version_one_csv_is_migrated_without_fake_overlap(self):
        old = self.replay.drop(columns=["log_schema_version", "confirmed_alerted_defects"])
        checked = validate_monitoring_log(old)
        self.assertTrue(checked["confirmed_alerted_defects"].isna().all())
        self.assertTrue((checked["log_schema_version"] == 1).all())

    def test_manual_aggregate_is_not_retraining_evidence(self):
        manual = self.replay.copy()
        manual["feedback_evidence"] = "manual_aggregate"
        summary = analyze_feedback_history(manual, self.reference)
        self.assertEqual(summary["status"], "UNVERIFIED_LABELS")
        self.assertFalse(summary["ready"])

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(69, 83)
        self.assertLess(low, 69 / 83)
        self.assertGreater(high, 69 / 83)


if __name__ == "__main__":
    unittest.main()
