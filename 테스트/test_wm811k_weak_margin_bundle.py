"""Tests for the imported WM-811K weak-margin validation bundle."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "코드" / "wm811k"))

from validate_wm811k_weak_margin_bundle import validate_bundle  # noqa: E402


class Wm811kWeakMarginBundleTests(unittest.TestCase):
    def test_imported_margin_bundle_is_reproducible_and_validation_only(self) -> None:
        result_dir = PROJECT_DIR / "결과물" / "wm811k" / "weak_margin_results"
        report = validate_bundle(
            PROJECT_DIR / "결과물" / "wm811k" / "colab_가져오기" / "wm811k_weak_margin_validation_results.zip",
            PROJECT_DIR / "결과물" / "wm811k" / "split_results" / "split_assignments.csv",
            PROJECT_DIR / "결과물" / "wm811k" / "weak_class_validation_results" / "baseline" / "best_model.pt",
        )
        saved = json.loads((result_dir / "bundle_validation.json").read_text(encoding="utf-8"))
        self.assertEqual(report, saved)
        self.assertEqual(report["selected_candidate"], "weak_none_margin_005")
        self.assertEqual(report["selected_margin_lambda"], 0.05)
        self.assertTrue(report["baseline_reuse_sha_matches"])
        self.assertFalse(report["test_evaluated"])
        self.assertFalse(report["deployment_changed"])
        self.assertGreater(
            report["selected_deltas_vs_baseline"]["validation_weak_recall"], 0.02
        )
        self.assertGreater(
            report["selected_deltas_vs_baseline"]["validation_macro_f1"], 0
        )
        self.assertTrue((result_dir / "검증결과.md").is_file())

    def test_dashboard_declares_multiseed_requirement(self) -> None:
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("WM_WEAK_MARGIN_DIR", app)
        self.assertIn("취약 클래스 margin 후보 · Validation 전용", app)
        self.assertIn("다중 seed 재현 전에는 체크포인트", app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
