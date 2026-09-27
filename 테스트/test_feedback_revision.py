"""Regression checks for the 2026-09-23 dashboard feedback revision."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


class FeedbackRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_source = (PROJECT / "app.py").read_text(encoding="utf-8")
        cls.wm_source = (
            PROJECT / "코드" / "wm811k" / "diagnose_wm811k.py"
        ).read_text(encoding="utf-8")

    def test_sources_parse(self) -> None:
        ast.parse(self.app_source)
        ast.parse(self.wm_source)

    def test_common_shap_decision_frame_is_visible_in_both_modules(self) -> None:
        self.assertIn("def render_unified_xai_frame", self.app_source)
        self.assertGreaterEqual(self.app_source.count("render_unified_xai_frame("), 3)
        for phrase in ("입력 신뢰도", "모델 판정", "SHAP 기여", "검토 행동"):
            self.assertIn(phrase, self.app_source)

    def test_wm_uses_gradient_shap_and_predicted_class_only(self) -> None:
        self.assertIn("def predict_with_gradient_shap", self.wm_source)
        self.assertIn("same_geometry_all_active_dies_normal", self.wm_source)
        self.assertIn("gradient_shap_result = explain_wafer_gradient_shap(", self.app_source)
        cached = self.app_source[
            self.app_source.index("def explain_wafer_gradient_shap"):
            self.app_source.index("def load_gradient_shap_evidence")
        ]
        self.assertIn("predict_with_gradient_shap(", cached)
        self.assertNotIn("target_class", cached)
        self.assertNotIn('"\uC124\uBA85\uD560 \uD074\uB798\uC2A4"', self.app_source)

    def test_gradient_shap_is_never_rescaled_to_pass_additivity(self) -> None:
        function = self.wm_source[
            self.wm_source.index("def predict_with_gradient_shap"):
            self.wm_source.index("def predict_wafer_batch")
        ]
        self.assertNotIn("local_accuracy_normalized", self.wm_source)
        self.assertNotIn("logit_delta / raw_attribution_sum", self.wm_source)
        self.assertIn('"rescaled_to_logit_delta": False', function)
        self.assertIn("\uC628\uB3C4\uB294", function)
        self.assertNotIn("\uB2E8\uB3C4\uB294", self.wm_source)
        diagnosis_start = self.app_source.index("def render_wm_diagnosis")
        diagnosis_end = self.app_source.index("def render_wm_batch")
        section = self.app_source[diagnosis_start:diagnosis_end]
        self.assertIn("raw_additivity_residual", section)
        self.assertIn('if gradient_shap_result["additivity_check_passed"]:', section)
        self.assertNotIn("['additivity_residual']", section)

    def test_wm_score_does_not_round_to_false_certainty(self) -> None:
        self.assertIn('return ">99.9%"', self.app_source)
        self.assertIn("실제 원인이나 정답이 100% 확실", self.app_source)

    def test_sample_gallery_shows_all_samples_and_explains_selection(self) -> None:
        self.assertIn("wm_gallery.render_sample_gallery(", self.app_source)
        self.assertNotIn("][:5]", self.app_source)
        self.assertIn("선정 근거", self.app_source)
        self.assertIn("대표 샘플 선택", self.app_source)

    def test_customer_navigation_hides_admin_by_default(self) -> None:
        access_source = (PROJECT / "dashboard_ui" / "access.py").read_text(encoding="utf-8")
        self.assertIn("if ACCESS.is_admin:", self.app_source)
        self.assertNotIn("ADMIN_MODE = os.getenv", self.app_source)
        self.assertIn('secom_navigation["\uAD00\uB9AC\uC790"]', self.app_source)
        self.assertIn('wm_navigation["\uAD00\uB9AC\uC790"]', self.app_source)
        self.assertIn("SHAPGPT_REQUIRE_LOGIN", self.app_source)
        self.assertIn('REQUIRE_LOGIN_ENV = "SHAPGPT_REQUIRE_LOGIN"', access_source)
        self.assertEqual(
            self.app_source.count("dashboard_access.section_allowed("), 2
        )

    def test_secom_quality_evidence_precedes_model_explanation(self) -> None:
        explanation_start = self.app_source.index("def render_secom_explanation")
        explanation_end = self.app_source.index("def render_secom_safety")
        section = self.app_source[explanation_start:explanation_end]
        self.assertLess(section.index("검토 우선순위"), section.index("TreeSHAP 기여"))
        self.assertIn("입력 데이터의 한계", section)

    def test_wm_failure_bug_no_longer_uses_cross_page_local(self) -> None:
        diagnosis_start = self.app_source.index("def render_wm_diagnosis")
        diagnosis_end = self.app_source.index("def render_wm_batch")
        section = self.app_source[diagnosis_start:diagnosis_end]
        self.assertNotIn("if xai_matches_deployment", section)
        self.assertIn("wm_xai_matches_deployment()", section)


if __name__ == "__main__":
    unittest.main(verbosity=2)
