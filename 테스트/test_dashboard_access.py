"""Login and per-user administrator access for the B2C dashboard."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]

from dashboard_ui import access  # noqa: E402

ISSUER = "https://accounts.example.test"
ADMIN_CLAIMS = {
    "is_logged_in": True,
    "email": "Owner@Example.test",
    "email_verified": True,
    "iss": ISSUER,
    "sub": "admin-subject",
}
VIEWER_CLAIMS = {
    "is_logged_in": True,
    "email": "viewer@example.test",
    "email_verified": True,
    "iss": ISSUER,
    "sub": "viewer-subject",
}
CLEAN_ENV = {
    key: value for key, value in os.environ.items() if not key.startswith("SHAPGPT_")
}


def resolve(env=None, claims=None, secrets=None, server_address=None):
    return access.resolve_access(
        env=env or {}, claims=claims, secrets=secrets, server_address=server_address
    )


class AccessDecisionTests(unittest.TestCase):
    def test_demo_mode_has_no_login_and_no_admin(self) -> None:
        context = resolve()
        self.assertFalse(context.login_required)
        self.assertFalse(context.is_admin)
        self.assertEqual(context.admin_basis, "none")

    def test_local_admin_flag_needs_demo_mode_and_loopback_server(self) -> None:
        flag = {"SHAPGPT_ADMIN_MODE": "1"}
        self.assertFalse(resolve(flag).is_admin)
        self.assertIn("루프백", resolve(flag).notice)
        self.assertFalse(resolve(flag, server_address="0.0.0.0").is_admin)
        for address in ("localhost", "127.0.0.1", "::1"):
            context = resolve(flag, server_address=address)
            self.assertTrue(context.is_admin)
            self.assertEqual(context.admin_basis, "local_development")
        with_login = resolve(
            {**flag, "SHAPGPT_REQUIRE_LOGIN": "1"},
            claims=VIEWER_CLAIMS,
            server_address="localhost",
        )
        self.assertFalse(with_login.is_admin)
        self.assertIn("무시", with_login.notice)

    def test_login_required_blocks_anonymous_viewers(self) -> None:
        env = {"SHAPGPT_REQUIRE_LOGIN": "true", "SHAPGPT_ADMIN_EMAILS": "owner@example.test"}
        context = resolve(env, claims={"email": "owner@example.test"})
        self.assertTrue(context.login_required)
        self.assertFalse(context.logged_in)
        self.assertFalse(context.is_admin)

    def test_admin_requires_verified_allowlisted_email(self) -> None:
        env = {
            "SHAPGPT_REQUIRE_LOGIN": "1",
            "SHAPGPT_ADMIN_EMAILS": " owner@example.test ; other@example.test",
        }
        admin = resolve(env, claims=ADMIN_CLAIMS)
        self.assertTrue(admin.is_admin)
        self.assertEqual(admin.admin_basis, "verified_email_allowlist")
        self.assertEqual(admin.user_label, "owner@example.test")
        self.assertFalse(resolve(env, claims=VIEWER_CLAIMS).is_admin)
        unverified = {**ADMIN_CLAIMS, "email_verified": False}
        self.assertFalse(resolve(env, claims=unverified).is_admin)
        missing_claim = {key: value for key, value in ADMIN_CLAIMS.items() if key != "email_verified"}
        self.assertFalse(resolve(env, claims=missing_claim).is_admin)
        self.assertFalse(resolve({"SHAPGPT_REQUIRE_LOGIN": "1"}, claims=ADMIN_CLAIMS).is_admin)

    def test_admin_subject_allowlist_from_secrets(self) -> None:
        env = {"SHAPGPT_REQUIRE_LOGIN": "1"}
        secrets = {"shapgpt": {"admin_subjects": [f"{ISSUER}|admin-subject"]}}
        no_email_claim = {"is_logged_in": True, "iss": ISSUER, "sub": "admin-subject"}
        context = resolve(env, claims=no_email_claim, secrets=secrets)
        self.assertTrue(context.is_admin)
        self.assertEqual(context.admin_basis, "subject_allowlist")
        other_issuer = {**no_email_claim, "iss": "https://evil.example.test"}
        self.assertFalse(resolve(env, claims=other_issuer, secrets=secrets).is_admin)
        case_changed = {**no_email_claim, "sub": "ADMIN-SUBJECT"}
        self.assertFalse(resolve(env, claims=case_changed, secrets=secrets).is_admin)
        email_secrets = {"shapgpt": {"admin_emails": ["OWNER@example.test"]}}
        self.assertTrue(resolve(env, claims=ADMIN_CLAIMS, secrets=email_secrets).is_admin)

    def test_allowlist_parsing_ignores_malformed_entries(self) -> None:
        self.assertEqual(
            access.parse_admin_emails("A@x.test, not-an-email,\nb@y.test;"),
            frozenset({"a@x.test", "b@y.test"}),
        )
        self.assertEqual(
            access.parse_admin_subjects(["iss|sub", "missing-separator", "|sub", "a|b|c"]),
            frozenset({"iss|sub"}),
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
    """Run the real app with environment and identity variations."""

    @staticmethod
    def _run(env: dict[str, str], query: dict[str, str] | None = None, claims=None) -> AppTest:
        app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=120)
        for key, value in (query or {}).items():
            app.query_params[key] = value
        with patch.dict(os.environ, env, clear=True):
            if claims is None:
                app.run()
            else:
                with patch.object(access, "current_user_claims", return_value=claims):
                    app.run()
        return app

    @staticmethod
    def _texts(app: AppTest) -> list[str]:
        values = []
        for name in ("title", "subheader", "warning", "error", "info", "caption"):
            values.extend(str(element.value) for element in getattr(app, name))
        return values

    def test_demo_mode_hides_admin_menu_and_blocks_deep_links(self) -> None:
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

    def test_login_required_mode_shows_only_the_login_gate(self) -> None:
        env = {**CLEAN_ENV, "SHAPGPT_REQUIRE_LOGIN": "1"}
        app = self._run(env, {"module": "secom", "section": "release"})
        self.assertEqual(len(app.exception), 0)
        texts = self._texts(app)
        self.assertIn("SHAPGPT 로그인", texts)
        self.assertNotIn("SECOM 공정 센서 SHAP 진단", texts)
        self.assertNotIn("SECOM 통합 릴리스 준비도", texts)
        self.assertTrue(any("OIDC 로그인 설정이 없어" in text for text in texts))
        self.assertEqual(len(app.segmented_control), 0)

    def test_logged_in_admin_opens_admin_screen(self) -> None:
        env = {
            **CLEAN_ENV,
            "SHAPGPT_REQUIRE_LOGIN": "1",
            "SHAPGPT_ADMIN_EMAILS": "owner@example.test",
        }
        app = self._run(env, {"module": "secom", "section": "monitoring"}, claims=ADMIN_CLAIMS)
        self.assertEqual(len(app.exception), 0)
        texts = self._texts(app)
        self.assertIn("배치 운영 기록과 안전 신호 추세", texts)
        self.assertNotIn("관리자 권한이 필요한 화면입니다. 기본 화면으로 이동했습니다.", texts)
        self.assertIn("관리자", app.segmented_control[0].options)
        self.assertTrue(any("로그인: owner@example.test · 관리자" in text for text in texts))

    def test_logged_in_viewer_cannot_open_admin_screen(self) -> None:
        env = {
            **CLEAN_ENV,
            "SHAPGPT_REQUIRE_LOGIN": "1",
            "SHAPGPT_ADMIN_EMAILS": "owner@example.test",
        }
        app = self._run(env, {"module": "secom", "section": "release"}, claims=VIEWER_CLAIMS)
        self.assertEqual(len(app.exception), 0)
        texts = self._texts(app)
        self.assertIn("SECOM 공정 센서 SHAP 진단", texts)
        self.assertNotIn("SECOM 통합 릴리스 준비도", texts)
        self.assertNotIn("관리자", app.segmented_control[0].options)


if __name__ == "__main__":
    unittest.main(verbosity=2)
