"""Tests for non-causal WM-811K cause guidance."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
WM_SCRIPT_DIR = PROJECT_DIR / "코드" / "wm811k"
if str(WM_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(WM_SCRIPT_DIR))

from wafer_cause_guidance import (  # noqa: E402
    SUPPORTED_LABELS,
    build_wafer_cause_guidance,
)


FEATURES = {
    "defect_ratio": 0.18,
    "component_count_8": 3,
    "largest_component_fraction": 0.72,
    "center_defect_fraction": 0.10,
    "edge_defect_fraction": 0.65,
    "boundary_defect_coverage": 0.55,
    "defect_bbox_fraction": 0.80,
}


class Wm811kCauseGuidanceTests(unittest.TestCase):
    def test_every_defect_class_has_three_actionable_hypotheses(self) -> None:
        for label in SUPPORTED_LABELS:
            guidance = build_wafer_cause_guidance(label, FEATURES)
            if label == "none":
                self.assertEqual(guidance["candidates"], [])
                self.assertIn("정상 패턴", guidance["interpretation"])
                continue
            candidates = guidance["candidates"]
            self.assertEqual(len(candidates), 3, label)
            self.assertEqual(
                [item["priority"] for item in candidates], [1, 2, 3]
            )
            self.assertEqual(len({item["cause"] for item in candidates}), 3)
            self.assertTrue(all(item["check"] for item in candidates))
            self.assertIn("확정 불가", guidance["interpretation"])
            self.assertIn("원인 확률", guidance["disclaimer"])

    def test_shape_evidence_and_ood_limit_are_explicit(self) -> None:
        guidance = build_wafer_cause_guidance(
            "Edge-Ring", FEATURES, ood_status="out_of_distribution"
        )
        evidence = " · ".join(guidance["observations"])
        self.assertIn("결함 die 18.0%", evidence)
        self.assertIn("8-이웃 연결 영역 3개", evidence)
        self.assertIn("가장자리", evidence)
        self.assertIn("입력 재확인", guidance["evidence_level"])

    def test_unknown_class_and_invalid_features_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "지원하지 않는"):
            build_wafer_cause_guidance("Mystery", FEATURES)
        incomplete = dict(FEATURES)
        incomplete.pop("defect_ratio")
        with self.assertRaisesRegex(ValueError, "defect_ratio"):
            build_wafer_cause_guidance("Loc", incomplete)

    def test_dashboard_and_reports_expose_guidance(self) -> None:
        app = (PROJECT_DIR / "app.py").read_text(encoding="utf-8")
        diagnosis = (WM_SCRIPT_DIR / "diagnose_wm811k.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("가능한 불량 원인과 우선 점검", app)
        self.assertIn("build_wafer_cause_guidance", app)
        self.assertIn('"주요 원인 후보"', app)
        self.assertIn('"우선 확인 항목"', app)
        self.assertIn('"주요 원인 후보": 48', diagnosis)


if __name__ == "__main__":
    unittest.main(verbosity=2)
