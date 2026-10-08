"""Administrator access for the dashboard: the local development mode only."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]

from dashboard_ui import access  # noqa: E402

CLEAN_ENV = {
    key: value for key, value in os.environ.items() if not key.startswith("SHAPGPT_")
}


def resolve(env=None, server_address=None):
    return access.resolve_access(env=env or {}, server_address=server_address)


class AccessDecisionTests(unittest.TestCase):
    def test_default_mode_has_no_admin(self) -> None:
        context = resolve()
        self.assertFalse(context.is_admin)
        self.assertEqual(context.admin_basis, "none")

    def test_local_admin_flag_needs_loopback_server(self) -> None:
        flag = {"SHAPGPT_ADMIN_MODE": "1"}
        self.assertFalse(resolve(flag).is_admin)
        self.assertFalse(resolve(flag, server_address="0.0.0.0").is_admin)
        for address in ("localhost", "127.0.0.1", "::1"):
            context = resolve(flag, server_address=address)
            self.assertTrue(context.is_admin)
            self.assertEqual(context.admin_basis, "local_development")
        self.assertFalse(
            resolve({"SHAPGPT_ADMIN_MODE": "0"}, server_address="localhost").is_admin
        )

    def test_admin_sections_are_guarded(self) -> None:
        viewer = resolve()
        admin = resolve({"SHAPGPT_ADMIN_MODE": "1"}, server_address="localhost")
        for section in access.SECOM_ADMIN_SECTIONS:
            self.assertFalse(access.section_allowed(section, access.SECOM_ADMIN_SECTIONS, viewer))
            self.assertTrue(access.section_allowed(section, access.SECOM_ADMIN_SECTIONS, admin))
        self.assertTrue(access.section_allowed("diagnosis", access.SECOM_ADMIN_SECTIONS, viewer))
        self.assertFalse(access.section_allowed("research_log", access.WM_ADMIN_SECTIONS, viewer))

    def test_no_hardcoded_email_addresses_in_access_code(self) -> None:
        pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+")
        for path in (PROJECT_DIR / "app.py", PROJECT_DIR / "dashboard_ui" / "access.py"):
            self.assertEqual(pattern.findall(path.read_text(encoding="utf-8")), [], path.name)


class AccessDashboardTests(unittest.TestCase):
    """Run the real app with environment variations."""

    @staticmethod
    def _run(env: dict[str, str], query: dict[str, str] | None = None) -> AppTest:
        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=120)
        for key, value in (query or {}).items():
            app.query_params[key] = value
        with patch.dict(os.environ, env, clear=True):
            app.run()
        return app

    @staticmethod
    def _texts(app: AppTest) -> list[str]:
        values = []
        for name in ("title", "subheader", "warning", "error", "info", "caption"):
            values.extend(str(element.value) for element in getattr(app, name))
        return values

    def test_default_mode_hides_admin_menu_and_blocks_deep_links(self) -> None:
        app = self._run(CLEAN_ENV, {"module": "secom", "section": "release"})
        self.assertEqual(len(app.exception), 0)
        texts = self._texts(app)
        self.assertIn("관리자 권한이 필요한 화면입니다. 기본 화면으로 이동했습니다.", texts)
        self.assertNotIn("SECOM 통합 릴리스 준비도", texts)
        self.assertNotIn("관리자", app.segmented_control[0].options)
        self.assertIn("SECOM 공정 센서 SHAP 진단", texts)

    def test_admin_flag_without_loopback_server_grants_nothing(self) -> None:
        env = {**CLEAN_ENV, "SHAPGPT_ADMIN_MODE": "1"}
        app = self._run(env, {"module": "secom", "section": "monitoring"})
        self.assertEqual(len(app.exception), 0)
        texts = self._texts(app)
        self.assertNotIn("배치 운영 기록과 안전 신호 추세", texts)
        self.assertNotIn("관리자", app.segmented_control[0].options)

    def test_wm_research_log_deep_link_is_blocked(self) -> None:
        app = self._run(CLEAN_ENV, {"module": "wm811k", "section": "research_log"})
        self.assertEqual(len(app.exception), 0)
        texts = self._texts(app)
        self.assertIn("관리자 권한이 필요한 화면입니다. 기본 화면으로 이동했습니다.", texts)
        self.assertNotIn("전체 실험 기록", texts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
