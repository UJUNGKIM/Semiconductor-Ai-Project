import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "코드" / "secom"))

from row_label_feedback import (
    build_label_template,
    hard_example_manifest,
    validate_completed_feedback,
)


class RowLabelFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.results = pd.DataFrame(
            {
                "행 번호": [0, 1, 2],
                "CatBoost 점수": [0.8, 0.1, 0.2],
                "CatBoost 판정": ["불량", "정상", "정상"],
                "XGBoost 판정": ["불량", "정상", "정상"],
                "종합 판정": ["두 모델 모두 불량", "두 모델 모두 정상", "두 모델 모두 정상"],
            }
        )
        self.ood = pd.DataFrame(
            {"행 번호": [0, 1, 2], "OOD 상태": ["in_distribution"] * 3}
        )
        self.template = build_label_template(
            self.results,
            self.ood,
            batch_id="lot-1",
            profile="balanced_f2",
            model_hashes={"CatBoost": "cat", "XGBoost": "xgb"},
        )

    def complete(self):
        frame = self.template.copy()
        frame["actual_label"] = [1, 0, 1]
        frame["label_source"] = "final_inspection"
        frame["reviewer"] = "reviewer-a"
        frame["reviewed_at"] = "2026-09-16T12:00:00+09:00"
        return frame

    def test_template_contains_no_sensor_values(self):
        self.assertEqual(len(self.template), 3)
        self.assertFalse(any(column.startswith("feature_") for column in self.template))

    def test_complete_feedback_builds_confusion_counts(self):
        completed, summary = validate_completed_feedback(self.complete(), self.template)
        self.assertTrue(summary["eligible_for_performance_monitoring"])
        self.assertEqual(summary["true_positive"], 1)
        self.assertEqual(summary["false_negative"], 1)
        manifest = hard_example_manifest(completed, summary)
        self.assertEqual(manifest["false_negative_row_numbers"], [2])
        self.assertFalse(manifest["contains_raw_sensor_values"])

    def test_prediction_tampering_is_rejected(self):
        changed = self.complete()
        changed.loc[0, "predicted_alert"] = False
        with self.assertRaises(ValueError):
            validate_completed_feedback(changed, self.template)

    def test_partial_feedback_is_not_performance_evidence(self):
        partial = self.complete().iloc[:2].copy()
        _, summary = validate_completed_feedback(partial, self.template)
        self.assertFalse(summary["eligible_for_performance_monitoring"])
        self.assertTrue(summary["selection_bias_warning"])

    def test_missing_audit_metadata_is_rejected(self):
        changed = self.complete()
        changed.loc[0, "reviewer"] = ""
        with self.assertRaises(ValueError):
            validate_completed_feedback(changed, self.template)

    def test_dashboard_contract(self):
        text = (PROJECT / "app.py").read_text(encoding="utf-8")
        self.assertIn("행 단위 검수 양식 다운로드", text)
        self.assertIn('feedback_evidence="row_level_complete"', text)


if __name__ == "__main__":
    unittest.main()
