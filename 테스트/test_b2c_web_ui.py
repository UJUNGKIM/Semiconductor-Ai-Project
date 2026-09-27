"""B2C site structure, WM-811K sample gallery and categorical map rendering."""

from __future__ import annotations

import json
import math
import os
import tomllib
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]

from dashboard_ui import access, wafer_view, wm_gallery  # noqa: E402

APP = str(PROJECT_DIR / "app.py")
PAGES = "dashboard_ui/site_pages"
DEMO_DIR = PROJECT_DIR / "결과물" / "wm811k" / "demo_samples"
MANIFEST = json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))
ISSUER = "https://accounts.example.test"
ADMIN_CLAIMS = {
    "is_logged_in": True,
    "name": "Owner Park",
    "email": "owner@example.test",
    "email_verified": True,
    "iss": ISSUER,
    "sub": "admin-subject",
}
VIEWER_CLAIMS = {
    "is_logged_in": True,
    "name": "Viewer Kim",
    "email": "viewer@example.test",
    "email_verified": True,
    "iss": ISSUER,
    "sub": "viewer-subject",
}
CLEAN_ENV = {key: value for key, value in os.environ.items() if not key.startswith("SHAPGPT_")}
LOGIN_ENV = {
    **CLEAN_ENV,
    "SHAPGPT_REQUIRE_LOGIN": "1",
    "SHAPGPT_ADMIN_EMAILS": "owner@example.test",
}
PICK = wm_gallery.PICK_BUTTON_LABEL


def rendered_text(app: AppTest) -> str:
    chunks = []
    for name in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "error"):
        chunks.extend(str(getattr(element, "value", "")) for element in getattr(app, name))
    return "\n".join(chunks)


def pick_buttons(app: AppTest) -> list:
    return [button for button in app.button if button.label == PICK]


def decode_rgb(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(png)).convert("RGB"))


class CategoricalRenderingTests(unittest.TestCase):
    def test_nearest_upscale_repeats_cells_without_new_values(self) -> None:
        wafer = np.random.default_rng(7).integers(0, 3, size=(13, 17)).astype(np.uint8)
        scaled = wafer_view.upscale_nearest(wafer, 5)
        self.assertEqual(scaled.shape, (65, 85))
        self.assertTrue(set(np.unique(scaled)) <= {0, 1, 2})
        blocks = scaled.reshape(13, 5, 17, 5)
        np.testing.assert_array_equal(blocks, wafer[:, None, :, None].repeat(5, 1).repeat(5, 3))
        with self.assertRaises(ValueError):
            wafer_view.upscale_nearest(wafer, 1.5)

    def test_png_holds_only_the_three_category_colors(self) -> None:
        sample = MANIFEST["samples"][0]
        wafer = np.load(DEMO_DIR / sample["npy_file"], allow_pickle=False)
        png, scale = wafer_view.categorical_png(wafer)
        image = Image.open(BytesIO(png))
        self.assertEqual(image.format, "PNG")
        self.assertEqual(image.size, (512, 512))
        self.assertEqual(scale["factor"], 8)
        palette = {tuple(color) for color in wafer_view.CATEGORY_COLORS.values()}
        colors = {tuple(pixel) for pixel in decode_rgb(png).reshape(-1, 3)}
        self.assertTrue(colors <= palette, colors - palette)
        codes = np.asarray(image)
        np.testing.assert_array_equal(codes, wafer_view.upscale_nearest(wafer.astype(np.uint8), 8))

    def test_display_size_uses_integer_blocks_within_streamlit_limit(self) -> None:
        for shape, factor in (((64, 64), 8), ((32, 32), 16), ((25, 27), 21), ((300, 200), 3)):
            wafer = np.ones(shape, dtype=np.uint8)
            _, scale = wafer_view.categorical_png(wafer)
            self.assertEqual(scale["factor"], factor, shape)
            self.assertGreaterEqual(min(scale["height"], scale["width"]), 512, shape)
            self.assertLessEqual(max(scale["height"], scale["width"]), wafer_view.MAX_IMAGE_SIDE)
        _, huge = wafer_view.categorical_png(np.ones((2000, 60), dtype=np.uint8))
        self.assertGreater(huge["step"], 1)
        self.assertLessEqual(huge["height"], wafer_view.MAX_IMAGE_SIDE)
        self.assertLess(wafer_view.MAX_IMAGE_SIDE, 1460)

    def test_out_of_category_or_fractional_maps_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "0, 1, 2"):
            wafer_view.categorical_png(np.full((8, 8), 3, dtype=np.uint8))
        with self.assertRaises(ValueError):
            wafer_view.categorical_png(np.full((8, 8), 1.0))

    def test_explanation_map_keeps_die_blocks_and_neutral_outside(self) -> None:
        wafer = np.ones((64, 64), dtype=np.uint8)
        wafer[:4, :] = 0
        values = np.random.default_rng(3).random((64, 64))
        png, scale = wafer_view.explanation_png(values, wafer)
        self.assertEqual(scale["factor"], 8)
        rgb = decode_rgb(png).reshape(64, 8, 64, 8, 3)
        self.assertTrue((rgb == rgb[:, :1, :, :1, :]).all(), "a die cell mixed colors")
        np.testing.assert_array_equal(rgb[:4, 0, :, 0], np.broadcast_to(wafer_view.CATEGORY_COLORS[0], (4, 64, 3)))
        with self.assertRaisesRegex(ValueError, "크기"):
            wafer_view.explanation_png(np.zeros((32, 32)), wafer)

    def test_images_reach_streamlit_as_lossless_png(self) -> None:
        png, _ = wafer_view.categorical_png(np.ones((16, 16), dtype=np.uint8))
        with patch.object(wafer_view.st, "image") as image:
            wafer_view.show_png(png, caption="map")
        args, kwargs = image.call_args
        self.assertIs(args[0], png)
        self.assertEqual(kwargs["output_format"], "PNG")
        self.assertEqual(kwargs["width"], "content")
        gallery_source = (PROJECT_DIR / "dashboard_ui" / "wm_gallery.py").read_text(encoding="utf-8")
        self.assertIn('output_format="PNG"', gallery_source)


class GalleryDataTests(unittest.TestCase):
    def test_every_class_shows_all_of_its_manifest_samples(self) -> None:
        for true_class in wm_gallery.class_order(MANIFEST):
            expected = {item["sample_id"] for item in MANIFEST["samples"] if item["true_class"] == true_class}
            cards = wm_gallery.cards_for_class(MANIFEST, true_class)
            self.assertEqual({card.sample_id for card in cards}, expected, true_class)
            order = MANIFEST["case_display_order"]
            self.assertEqual([card.case for card in cards], sorted((card.case for card in cards), key=order.index))

    def test_rows_hold_at_most_four_cards(self) -> None:
        cards = wm_gallery.cards_for_class(MANIFEST, "Loc")
        for count in range(1, 10):
            subset = (cards * 2)[:count]
            rows = wm_gallery.gallery_rows(subset)
            self.assertEqual(len(rows), math.ceil(count / 4))
            self.assertTrue(all(1 <= len(row) <= 4 for row in rows))
            self.assertEqual(sum(len(row) for row in rows), count)

    def test_selection_stays_inside_the_selected_class(self) -> None:
        cards = wm_gallery.cards_for_class(MANIFEST, "Donut")
        self.assertEqual(wm_gallery.resolve_selection(cards, cards[2].sample_id), cards[2].sample_id)
        other = wm_gallery.cards_for_class(MANIFEST, "Scratch")[0].sample_id
        self.assertEqual(wm_gallery.resolve_selection(cards, other), cards[0].sample_id)
        self.assertIsNone(wm_gallery.resolve_selection([], other))

    def test_gallery_cards_never_touch_the_model(self) -> None:
        source = (PROJECT_DIR / "dashboard_ui" / "wm_gallery.py").read_text(encoding="utf-8")
        for token in ("torch", "predict_", "diagnose_wm811k", "gradient_shap", "assess_wafer_ood"):
            self.assertNotIn(token, source)


def synthetic_gallery_app(count: int) -> None:
    import numpy as np
    import streamlit as st

    from dashboard_ui import wafer_view, wm_gallery

    cases = list(wm_gallery.CASE_LABELS)
    samples = [
        {
            "sample_id": f"demo_{index}",
            "true_class": "Loc",
            "predicted_class": "Loc" if index % 5 != 4 else "none",
            "selection_case": cases[index % len(cases)],
            "calibrated_confidence": 0.5 + index / 20,
            "review_required": index % 2 == 1,
            "npy_file": f"samples/demo_{index}.npy",
        }
        for index in range(count)
    ]
    manifest = {"samples": samples, "case_display_order": cases}
    png = wafer_view.categorical_png(np.ones((8, 8), dtype=np.uint8), min_side=64)[0]
    selected = wm_gallery.render_sample_gallery(
        wm_gallery.cards_for_class(manifest, "Loc"),
        state_key="picked",
        thumbnail=lambda card: png,
        score_formatter=lambda value: f"{value:.1%}",
    )
    st.caption(f"selected={selected}")


class GalleryRenderingTests(unittest.TestCase):
    def test_three_four_or_five_samples_render_as_full_rows(self) -> None:
        for count in (3, 4, 5):
            with self.subTest(count=count):
                app = AppTest.from_function(synthetic_gallery_app, args=(count,), default_timeout=30).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(pick_buttons(app)), count)
                self.assertEqual(len(app.image), count)
                self.assertEqual(len(app.columns), 4 * math.ceil(count / 4))
                self.assertIn("selected=demo_0", [caption.value for caption in app.caption])

    def test_clicking_a_card_selects_only_that_sample(self) -> None:
        app = AppTest.from_function(synthetic_gallery_app, args=(5,), default_timeout=30).run()
        app.button(key="wm_pick_demo_3").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("selected=demo_3", [caption.value for caption in app.caption])
        self.assertEqual(rendered_text(app).count("wm-badge--selected"), 1)
        primary = [button.key for button in pick_buttons(app) if button.proto.type == "primary"]
        self.assertEqual(primary, ["wm_pick_demo_3"])


class WmGalleryAppTests(unittest.TestCase):
    """The real WM-811K page with the committed demo manifest."""

    @staticmethod
    def open_diagnosis() -> AppTest:
        app = AppTest.from_file(APP, default_timeout=240)
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.run()
            app.query_params["section"] = "diagnosis"
            app.switch_page(f"{PAGES}/wm811k.py").run()
        return app

    def test_class_selection_shows_every_representative_sample_at_once(self) -> None:
        app = self.open_diagnosis()
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.selectbox(key="wm811k_demo_class").set_value("Loc").run()
        self.assertEqual(len(app.exception), 0)
        expected = [item for item in MANIFEST["samples"] if item["true_class"] == "Loc"]
        self.assertEqual(len(pick_buttons(app)), len(expected))
        self.assertEqual(
            {button.key for button in pick_buttons(app)},
            {f"wm_pick_{item['sample_id']}" for item in expected},
        )
        text = rendered_text(app)
        for item in expected:
            self.assertIn(wm_gallery.CASE_LABELS[item["selection_case"]], text)

    def test_picking_a_card_runs_detail_analysis_for_that_sample_only(self) -> None:
        app = self.open_diagnosis()
        target = wm_gallery.cards_for_class(MANIFEST, "Center")[3]
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.button(key=f"wm_pick_{target.sample_id}").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["wm811k_selected_sample"], target.sample_id)
        analysed = [info.value for info in app.info if str(info.value).startswith("분석 대상")]
        self.assertEqual(len(analysed), 1)
        self.assertIn(target.sample_id, analysed[0])
        self.assertEqual([metric.label for metric in app.metric].count("CNN 판정"), 1)
        self.assertFalse(any("CNN 진단에 실패" in str(error.value) for error in app.error))

    def test_upload_and_artificial_inputs_keep_working(self) -> None:
        app = self.open_diagnosis()
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.radio(key="wm811k_source").set_value("인공 데모 맵").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(pick_buttons(app), [])
        self.assertIn("CNN 판정", [metric.label for metric in app.metric])
        self.assertIn("32×32 격자의 각 칸을 16×16 픽셀 블록으로", rendered_text(app))
        upload = "\n".join(",".join(["0", "1", "1", "1", "1", "0"]) for _ in range(3))
        upload += "\n" + "\n".join(",".join(["1", "2", "2", "1", "1", "1"]) for _ in range(3))
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.radio(key="wm811k_source").set_value("파일 업로드").run()
            app.file_uploader(key="wm811k_upload").set_value(
                ("wafer.csv", upload.encode("utf-8"), "text/csv")
            ).run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("CNN 판정", [metric.label for metric in app.metric])
        self.assertIn("6×6 격자의 각 칸을 86×86 픽셀 블록으로", rendered_text(app))


class SiteAccessAppTests(unittest.TestCase):
    """Public pages, login gate and administrator page with identity variations."""

    @staticmethod
    def run_app(env: dict, claims=None, page: str | None = None, app: AppTest | None = None) -> AppTest:
        app = app or AppTest.from_file(APP, default_timeout=120)
        with patch.dict(os.environ, env, clear=True), patch.object(
            access, "current_user_claims", return_value=claims or {}
        ):
            if page is not None:
                app.run()
                app.switch_page(f"{PAGES}/{page}.py")
            app.run()
        return app

    def test_logged_out_visitor_sees_public_pages_but_cannot_diagnose(self) -> None:
        app = self.run_app(LOGIN_ENV)
        self.assertEqual(len(app.exception), 0)
        text = rendered_text(app)
        self.assertIn("설명 가능한 AI로 반도체 공정 데이터를 진단합니다", text)
        self.assertIn("입력 신뢰도", text)
        login = app.button(key="home_login")
        self.assertEqual(login.label, "Google로 로그인")
        self.assertTrue(login.disabled)
        self.assertIn("WM-811K Macro-F1", [metric.label for metric in app.metric])
        for page, forbidden in (("secom", "진단 실행"), ("wm811k", PICK)):
            with self.subTest(page=page):
                gated = self.run_app(LOGIN_ENV, page=page)
                self.assertEqual(len(gated.exception), 0)
                self.assertIn("SHAPGPT 로그인", [title.value for title in gated.title])
                self.assertNotIn(forbidden, [button.label for button in gated.button])
                self.assertNotIn("wm811k_batch_run", [button.key for button in gated.button])
                self.assertTrue(any("OIDC 로그인 설정이 없어" in str(e.value) for e in gated.error))
        project = self.run_app(LOGIN_ENV, page="project")
        self.assertIn("프로젝트·검증", [title.value for title in project.title])

    def test_logged_in_viewer_gets_no_admin_menu(self) -> None:
        app = self.run_app(LOGIN_ENV, claims=VIEWER_CLAIMS)
        self.assertEqual(len(app.exception), 0)
        self.assertIn("로그인: viewer@example.test · 일반 사용자", [c.value for c in app.caption])
        with self.assertRaises(ValueError):
            app.switch_page(f"{PAGES}/admin.py")
        secom = self.run_app(LOGIN_ENV, claims=VIEWER_CLAIMS, page="secom")
        self.assertIn("SECOM 공정 센서 SHAP 진단", [title.value for title in secom.title])
        self.assertNotIn("관리자", secom.segmented_control[0].options)

    def test_account_page_shows_profile_without_raw_claims(self) -> None:
        app = self.run_app(LOGIN_ENV, claims=VIEWER_CLAIMS, page="account")
        self.assertEqual(len(app.exception), 0)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["이름"], "Viewer Kim")
        self.assertEqual(metrics["이메일"], "viewer@example.test")
        self.assertEqual(metrics["역할"], "일반 사용자")
        self.assertIn("로그아웃", [button.label for button in app.button])
        text = rendered_text(app) + " ".join(metrics.values())
        self.assertNotIn("viewer-subject", text)
        self.assertNotIn(ISSUER, text)

    def test_allowlisted_admin_sees_admin_page(self) -> None:
        app = self.run_app(LOGIN_ENV, claims=ADMIN_CLAIMS, page="admin")
        self.assertEqual(len(app.exception), 0)
        self.assertIn("관리자", [title.value for title in app.title])
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["권한 근거"], "인증된 이메일 허용 목록")
        self.assertEqual(metrics["허용 목록"], "이메일 1개 · 주체 0개")
        self.assertNotIn("owner@example.test", rendered_text(app).replace("로그인: owner@example.test", ""))
        secom = self.run_app(LOGIN_ENV, claims=ADMIN_CLAIMS, page="secom")
        self.assertIn("관리자", secom.segmented_control[0].options)

    def test_admin_url_falls_back_for_a_regular_user(self) -> None:
        app = self.run_app(LOGIN_ENV, claims=ADMIN_CLAIMS, page="admin")
        self.assertIn("관리자", [title.value for title in app.title])
        app = self.run_app(LOGIN_ENV, claims=VIEWER_CLAIMS, app=app)
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("관리자", [title.value for title in app.title])
        self.assertNotIn("권한 근거", [metric.label for metric in app.metric])
        self.assertIn("설명 가능한 AI로 반도체 공정 데이터를 진단합니다", rendered_text(app))

    def test_legacy_module_links_open_the_matching_page(self) -> None:
        app = AppTest.from_file(APP, default_timeout=120)
        app.query_params["module"] = "wm811k"
        app.query_params["section"] = "evaluation"
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn("WM-811K 웨이퍼 맵 SHAP 진단", [title.value for title in app.title])
        self.assertIn("운영 성능 요약", rendered_text(app))


class LoginConfigurationTests(unittest.TestCase):
    SHARED = {"redirect_uri": "https://app.example.test/oauth2callback", "cookie_secret": "x" * 32}
    PROVIDER = {
        "client_id": "client",
        "client_secret": "secret",
        "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
    }

    def test_only_complete_configurations_enable_login(self) -> None:
        self.assertEqual(access.login_provider({**self.SHARED, "google": self.PROVIDER}), "google")
        self.assertEqual(access.login_provider({**self.SHARED, **self.PROVIDER}), "")
        self.assertIsNone(access.login_provider(None))
        self.assertIsNone(access.login_provider({"google": self.PROVIDER}))
        self.assertIsNone(access.login_provider({**self.SHARED, "google": {**self.PROVIDER, "client_secret": " "}}))
        self.assertIsNone(access.login_provider({**self.SHARED}))

    def test_login_callback_does_nothing_without_configuration(self) -> None:
        with patch.object(access, "configured_login_provider", return_value=None), patch(
            "streamlit.login"
        ) as login:
            access.start_login()
        login.assert_not_called()
        with patch.object(access, "configured_login_provider", return_value="google"), patch(
            "streamlit.login"
        ) as login:
            access.start_login()
        login.assert_called_once_with("google")

    def test_secrets_template_holds_placeholders_only(self) -> None:
        template_path = PROJECT_DIR / ".streamlit" / "secrets.toml.example"
        template = tomllib.loads(template_path.read_text(encoding="utf-8"))
        self.assertEqual(access.login_provider(template["auth"]), "google")
        google = template["auth"]["google"]
        for value in (
            template["auth"]["redirect_uri"],
            template["auth"]["cookie_secret"],
            google["client_id"],
            google["client_secret"],
        ):
            self.assertIn("<", value)
        self.assertEqual(google["server_metadata_url"], self.PROVIDER["server_metadata_url"])
        ignored = (PROJECT_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".streamlit/secrets.toml", ignored)
        self.assertNotIn(".streamlit/secrets.toml.example", ignored)


if __name__ == "__main__":
    unittest.main(verbosity=2)
