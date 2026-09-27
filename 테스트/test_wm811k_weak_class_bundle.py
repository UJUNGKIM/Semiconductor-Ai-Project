"""Tests for the imported WM-811K weak-class validation bundle."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
import zipfile


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "코드" / "wm811k"))

from validate_wm811k_weak_class_bundle import _safe_members, validate_bundle  # noqa: E402


class Wm811kWeakClassBundleTests(unittest.TestCase):
    def test_imported_bundle_is_validation_only_and_reproducible(self) -> None:
        result_dir = PROJECT_DIR / "결과물" / "wm811k" / "weak_class_validation_results"
        report = validate_bundle(
            PROJECT_DIR / "결과물" / "wm811k" / "colab_가져오기" / "wm811k_weak_class_validation_results.zip",
            PROJECT_DIR / "결과물" / "wm811k" / "split_results" / "split_assignments.csv",
        )
        saved = json.loads((result_dir / "bundle_validation.json").read_text(encoding="utf-8"))
        self.assertEqual(report, saved)
        self.assertEqual(report["selected_candidate"], "baseline")
        self.assertFalse(report["test_evaluated"])
        self.assertFalse(report["used_test_for_selection"])
        self.assertFalse(report["deployment_changed"])
        self.assertEqual(report["validation_rows_per_candidate"], 24703)
        self.assertLess(
            report["metrics"]["weak_defect_thinning"]["validation_weak_recall"],
            report["metrics"]["baseline"]["validation_weak_recall"],
        )
        self.assertTrue((result_dir / "검증결과.md").is_file())

    def test_unsafe_zip_path_is_rejected(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("../escape.txt", "unsafe")
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP path"):
                _safe_members(archive)

    def test_dashboard_declares_validation_only_result(self) -> None:
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("WM_WEAK_VALIDATION_DIR", app)
        self.assertIn("취약 클래스 개선 후보 · Validation 전용 비교", app)
        self.assertIn("이 실험 체크포인트는 배포에 사용하지 않습니다", app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
