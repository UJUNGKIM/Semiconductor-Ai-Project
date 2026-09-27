"""Tests for WM-811K audit-ready diagnosis reports."""

from __future__ import annotations

import sys
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
if str(WM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPT_DIR))

from diagnose_wm811k import (  # noqa: E402
    artificial_demo_map,
    build_wafer_audit_report,
    build_wafer_excel_report,
    canonical_wafer_sha256,
    sha256_file,
)


class Wm811kAuditTests(unittest.TestCase):
    def test_canonical_input_and_artifact_hashes(self) -> None:
        wafer = artificial_demo_map(32)
        first = canonical_wafer_sha256(wafer)
        second = canonical_wafer_sha256(wafer.copy())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        checkpoint = (
            PROJECT_DIR
            / "결과물"
            / "wm811k"
            / "selected_model_results"
            / "best_model.pt"
        )
        self.assertEqual(len(sha256_file(checkpoint)), 64)

    def test_html_report_escapes_user_controlled_text(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "파일명": "<script>alert(1)</script>.npy",
                    "입력 SHA-256": "A" * 64,
                    "판정": "불량",
                    "예측 클래스": "Scratch",
                    "보정 신뢰도": 0.81,
                    "입력 신뢰도": "분포 이탈 · 판정 보류",
                    "OOD 점수": 2.1,
                    "검토 필요": "예",
                    "처리 권고": "자동판정 보류 · 입력 확인",
                    "입력 품질 경고": "",
                }
            ]
        )
        report = build_wafer_audit_report(
            source_name="<script>alert(2)</script>",
            results=results,
            artifact_hashes={"model": "B" * 64},
        ).decode("utf-8")
        self.assertIn("WM-811K 설명가능 AI 진단 감사 보고서", report)
        self.assertNotIn("<script>alert(1)</script>", report)
        self.assertNotIn("<script>alert(2)</script>", report)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", report)
        self.assertIn("OOD 자동판정 보류", report)
        self.assertIn("SHA-256", report)

    def test_dashboard_exposes_audit_downloads(self) -> None:
        code = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        self.assertIn("wm811k_single_diagnosis_report.html", code)
        self.assertIn("wm811k_batch_audit_report.html", code)
        self.assertIn("wm811k_batch_diagnosis.xlsx", code)
        self.assertIn("build_wafer_audit_report", code)

    def test_excel_report_preserves_readable_layout(self) -> None:
        results = pd.DataFrame(
            [
                {
                    "파일명": "near_full_representative.npy",
                    "입력 SHA-256": "A" * 64,
                    "실제 클래스": "Near-full",
                    "판정": "불량",
                    "예측 클래스": "Near-full",
                    "보정 신뢰도": 0.9876,
                    "입력 신뢰도": "분포 경계 · 전문가 검토",
                    "OOD 점수": 1.2345,
                    "가까운 기준 클래스": "Near-full",
                    "검토 필요": "예",
                    "처리 권고": "전문가 검토",
                    "입력 품질 경고": "결함 die 비율이 train 범위를 벗어남",
                }
            ]
        )
        content = build_wafer_excel_report(results)
        self.assertTrue(content.startswith(b"PK"))
        with zipfile.ZipFile(BytesIO(content)) as workbook:
            sheet_xml = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")
            styles_xml = workbook.read("xl/styles.xml").decode("utf-8")
        self.assertIn('ySplit="1"', sheet_xml)
        self.assertIn('xSplit="2"', sheet_xml)
        self.assertIn('<autoFilter ref="A1:L2"', sheet_xml)
        self.assertIn('customWidth="1"', sheet_xml)
        self.assertIn('customHeight="1"', sheet_xml)
        self.assertIn('applyAlignment="1"', styles_xml)


if __name__ == "__main__":
    unittest.main(verbosity=2)
