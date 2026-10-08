"""Section navigation stays within Streamlit's widget-state rules.

Regression for the `전체 SHAP 근거 확인` button, which used to write the
`secom_nav_group` widget key after the widget existed and raised
StreamlitWidgetAlreadyInstantiatedError.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]
APP = str(PROJECT_DIR / "app.py")
PAGES = "dashboard_ui/site_pages"
CLEAN_ENV = {key: value for key, value in os.environ.items() if not key.startswith("SHAPGPT_")}
ROW = 7


def navigation_app() -> None:
    import streamlit as st

    from dashboard_ui.navigation import navigate_to_section, render_section_navigation

    if not st.session_state.get("hide_navigation"):
        section = render_section_navigation(
            {
                "진단": (("대기열", "queue"), ("근거", "evidence")),
                "안내": (("흐름", "flow"),),
            },
            module="demo",
            default_section="queue",
            key_prefix="demo_nav",
        )
        st.caption(f"section={section}")
    if st.button("script body", key="body_jump"):
        navigate_to_section(module="demo", section="evidence", row=3)
        st.rerun()
    st.button(
        "callback",
        key="callback_jump",
        on_click=navigate_to_section,
        kwargs={"module": "demo", "section": "flow"},
    )


def flat_navigation_app() -> None:
    import streamlit as st

    from dashboard_ui.navigation import render_flat_navigation

    area = render_flat_navigation(
        (("첫째", "first"), ("둘째", "second"), ("셋째", "third")),
        key="demo_area",
        label="데모 영역",
    )
    st.caption(f"area={area}")


def section_caption(app: AppTest) -> str:
    return next(c.value for c in app.caption if str(c.value).startswith(("section=", "area=")))


class NavigationHelperTests(unittest.TestCase):
    def test_script_body_jump_after_widgets_exist_no_longer_fails(self) -> None:
        app = AppTest.from_function(navigation_app, default_timeout=30).run()
        self.assertEqual(section_caption(app), "section=queue")
        app.button(key="body_jump").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(section_caption(app), "section=evidence")
        self.assertEqual(app.segmented_control(key="demo_nav_group").value, "진단")
        self.assertEqual(app.segmented_control(key="demo_nav_진단").value, "근거")
        self.assertEqual(app.query_params["section"], ["evidence"])

    def test_callback_jump_switches_group_and_section(self) -> None:
        app = AppTest.from_function(navigation_app, default_timeout=30).run()
        app.button(key="callback_jump").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(section_caption(app), "section=flow")
        self.assertEqual(app.segmented_control(key="demo_nav_group").value, "안내")
        app.button(key="body_jump").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(section_caption(app), "section=evidence")

    def test_group_selection_returns_after_widgets_were_dropped(self) -> None:
        app = AppTest.from_function(navigation_app, default_timeout=30).run()
        app.button(key="callback_jump").click().run()
        app.session_state["hide_navigation"] = True
        app.run()
        app.session_state["hide_navigation"] = False
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.segmented_control(key="demo_nav_group").value, "안내")
        self.assertEqual(section_caption(app), "section=flow")

    def test_flat_navigation_follows_links_and_clicks(self) -> None:
        app = AppTest.from_function(flat_navigation_app, default_timeout=30)
        app.query_params["section"] = "third"
        app.run()
        self.assertEqual(section_caption(app), "area=third")
        app.segmented_control(key="demo_area").set_value("둘째").run()
        self.assertEqual(section_caption(app), "area=second")
        self.assertEqual(app.query_params["section"], ["second"])
        unknown = AppTest.from_function(flat_navigation_app, default_timeout=30)
        unknown.query_params["section"] = "missing"
        unknown.run()
        self.assertEqual(section_caption(unknown), "area=first")

    def test_helper_never_writes_widget_keys(self) -> None:
        source = (PROJECT_DIR / "dashboard_ui" / "navigation.py").read_text(encoding="utf-8")
        helper = source[source.index("def navigate_to_section") :]
        self.assertNotIn("session_state", helper)
        app_source = Path(APP).read_text(encoding="utf-8")
        self.assertEqual(app_source.count("navigate_to_section("), 1)
        self.assertIn("on_click=open_row_explanation", app_source)


class SecomShapButtonTests(unittest.TestCase):
    """The real SECOM page: open the full SHAP evidence of a queue row."""

    @staticmethod
    def open_queue_row() -> AppTest:
        app = AppTest.from_file(APP, default_timeout=180)
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.run()
            app.switch_page(f"{PAGES}/secom.py").run()
            app.session_state["secom_selected_row"] = ROW
            app.run()
        return app

    def click_shap_button(self, app: AppTest) -> None:
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.button(key="open_selected_shap").click().run()
        self.assertEqual(len(app.exception), 0, [str(e.value) for e in app.exception])
        self.assertEqual(app.segmented_control(key="secom_nav_group").value, "진단")
        self.assertEqual(app.segmented_control(key="secom_nav_진단").value, "개별 SHAP 설명")
        self.assertEqual(app.selectbox(key="secom_explanation_row").value, ROW)
        self.assertEqual(app.query_params["section"], ["explanation"])
        self.assertEqual(app.query_params["row"], [str(ROW)])

    def test_button_opens_row_evidence_without_error(self) -> None:
        app = self.open_queue_row()
        self.assertIn("전체 SHAP 근거 확인", [button.label for button in app.button])
        self.click_shap_button(app)

    def test_evidence_page_survives_a_reload(self) -> None:
        app = self.open_queue_row()
        self.click_shap_button(app)
        reloaded = AppTest.from_file(APP, default_timeout=180)
        for key, values in app.query_params.items():
            reloaded.query_params[key] = values[-1]
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            reloaded.run()
        self.assertEqual(len(reloaded.exception), 0)
        self.assertEqual(reloaded.segmented_control(key="secom_nav_진단").value, "개별 SHAP 설명")
        self.assertEqual(reloaded.selectbox(key="secom_explanation_row").value, ROW)

    def test_button_keeps_working_after_changing_screens(self) -> None:
        app = self.open_queue_row()
        self.click_shap_button(app)
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.segmented_control(key="secom_nav_group").set_value("안내").run()
            self.assertEqual(len(app.exception), 0)
            app.segmented_control(key="secom_nav_group").set_value("진단").run()
            app.segmented_control(key="secom_nav_진단").set_value("검토 대기열").run()
            self.assertEqual(len(app.exception), 0)
        self.click_shap_button(app)
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.switch_page(f"{PAGES}/wm811k.py").run()
            app.switch_page(f"{PAGES}/secom.py").run()
            self.assertEqual(len(app.exception), 0)
            app.segmented_control(key="secom_nav_진단").set_value("검토 대기열").run()
        self.click_shap_button(app)


class SecomValidationRelocationTests(unittest.TestCase):
    @staticmethod
    def rendered_text(app: AppTest) -> str:
        chunks = []
        for name in ("title", "subheader", "markdown", "caption", "info", "warning"):
            chunks.extend(str(getattr(element, "value", "")) for element in getattr(app, name))
        return "\n".join(chunks)

    def test_diagnosis_page_keeps_only_run_tools(self) -> None:
        app = SecomShapButtonTests.open_queue_row()
        groups = app.segmented_control(key="secom_nav_group").options
        sections = app.segmented_control(key="secom_nav_진단").options
        self.assertEqual(groups, ["진단", "안내"])
        self.assertEqual(sections, ["검토 대기열", "개별 SHAP 설명", "안전·What-if"])
        for moved in ("증강 검증", "모델 검증", "검증 자료"):
            self.assertNotIn(moved, groups + sections)

    def test_old_links_open_the_project_page(self) -> None:
        for section, area_label in (
            ("validation", "SECOM 모델 검증"),
            ("augmentation", "SECOM 증강 검증"),
        ):
            with self.subTest(section=section):
                legacy = AppTest.from_file(APP, default_timeout=180)
                legacy.query_params["module"] = "secom"
                legacy.query_params["section"] = section
                with patch.dict(os.environ, CLEAN_ENV, clear=True):
                    legacy.run()
                self.assertEqual(len(legacy.exception), 0)
                self.assertIn("프로젝트·검증", [title.value for title in legacy.title])
                self.assertEqual(legacy.segmented_control(key="project_area").value, area_label)
                direct = AppTest.from_file(APP, default_timeout=180)
                with patch.dict(os.environ, CLEAN_ENV, clear=True):
                    direct.run()
                    direct.query_params["section"] = section
                    direct.switch_page(f"{PAGES}/secom.py").run()
                self.assertIn("프로젝트·검증", [title.value for title in direct.title])
                self.assertEqual(direct.segmented_control(key="project_area").value, area_label)

    def test_project_page_separates_the_four_validation_areas(self) -> None:
        app = AppTest.from_file(APP, default_timeout=180)
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.run()
            app.switch_page(f"{PAGES}/project.py").run()
        area = app.segmented_control(key="project_area")
        self.assertEqual(
            area.options,
            ["SECOM 모델 검증", "SECOM 증강 검증", "SHAP/XAI 검증 근거", "WM-811K 검증 자료"],
        )
        expected = {
            "SECOM 모델 검증": ("반복 교차검증 결과", "센서 오류 강건성 스트레스 테스트"),
            "SECOM 증강 검증": ("생성형 AI 증강 검증",),
            "SHAP/XAI 검증 근거": (
                "SHAP 설명 신뢰성 감사",
                "현재 배포 설명법 · Gradient SHAP 검증",
            ),
            "WM-811K 검증 자료": ("WM-811K 진단 페이지의 상세 검증 화면",),
        }
        for label, markers in expected.items():
            with self.subTest(area=label), patch.dict(os.environ, CLEAN_ENV, clear=True):
                app.segmented_control(key="project_area").set_value(label).run()
                self.assertEqual(len(app.exception), 0)
                text = self.rendered_text(app)
                for marker in markers:
                    self.assertIn(marker, text)
        self.assertIn("WM-811K 균형 정확도", [metric.label for metric in app.metric])

    def test_metric_cards_show_short_sources_and_keep_paths_in_evidence(self) -> None:
        app = AppTest.from_file(APP, default_timeout=180)
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.run()
            app.switch_page(f"{PAGES}/home.py").run()
        sources = [str(m.value) for m in app.markdown if "site-source" in str(m.value)]
        self.assertTrue(sources)
        self.assertTrue(all("결과물/" not in source for source in sources), sources)
        self.assertIn("검증 근거 보기", [expander.label for expander in app.expander])
        evidence = next(frame.value for frame in app.dataframe if "근거 파일" in frame.value.columns)
        files = " ".join(evidence["근거 파일"])
        self.assertIn("결과물/wm811k/selected_model_results/run_summary.json", files)
        self.assertIn("결과물/secom/dual_model_results/oof_operating_points.csv", files)
        self.assertEqual(
            set(evidence["출처"]),
            {"WM-811K test evaluation", "WM-811K Gradient SHAP validation", "SECOM repeated CV"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
