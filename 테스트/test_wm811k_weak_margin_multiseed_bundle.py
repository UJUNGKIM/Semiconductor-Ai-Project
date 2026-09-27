"""Tests for the imported WM-811K weak-margin multi-seed bundle."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "코드" / "wm811k"))

from validate_wm811k_weak_margin_multiseed_bundle import validate_bundle  # noqa: E402


class Wm811kWeakMarginMultiseedBundleTests(unittest.TestCase):
    def test_bundle_recomputes_all_predictions_and_keeps_baseline(self) -> None:
        result_dir = (
            PROJECT_DIR / "결과물" / "wm811k" / "weak_margin_multiseed_results"
        )
        report = validate_bundle(
            PROJECT_DIR / "결과물" / "wm811k" / "colab_가져오기"
            / "wm811k_weak_margin_multiseed_results.zip",
            PROJECT_DIR / "결과물" / "wm811k" / "split_results"
            / "split_assignments.csv",
            PROJECT_DIR / "결과물" / "wm811k" / "weak_margin_results",
        )
        saved = json.loads(
            (result_dir / "bundle_validation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report, saved)
        self.assertEqual(report["selected_candidate"], "baseline")
        self.assertFalse(report["eligible"])
        self.assertFalse(report["test_evaluated"])
        self.assertFalse(report["deployment_changed"])
        self.assertEqual(report["validation_prediction_rows_checked"], 148218)
        self.assertEqual(
            report["seed42_reuse_sha_matches"],
            {"baseline": True, "weak_none_margin_005": True},
        )
        self.assertGreater(
            report["mean_deltas"]["validation_weak_recall"], 0.03
        )
        self.assertFalse(report["checks"]["macro_f1_guardrail_all_seeds"])
        self.assertFalse(report["checks"]["none_recall_guardrail_all_seeds"])
        self.assertFalse(
            report["checks"]["balanced_accuracy_guardrail_all_seeds"]
        )
        self.assertTrue((result_dir / "검증결과.md").is_file())

    def test_dashboard_explains_multiseed_rejection(self) -> None:
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("WM_WEAK_MARGIN_MULTISEED_DIR", app)
        self.assertIn("Margin 다중 seed 재현성 검증 · Validation 전용", app)
        self.assertIn("따라서 기존 baseline을 유지합니다", app)
        self.assertIn("Test 미사용 · 배포 모델 변경 없음", app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
