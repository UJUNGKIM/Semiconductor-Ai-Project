"""SHAPGPT start screen, its "#" glow mask, the dark sidebar and the page design."""

from __future__ import annotations

import json
import os
import re
import unittest
from base64 import b64decode
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]

from dashboard_ui import brand, site  # noqa: E402

APP = str(PROJECT_DIR / "app.py")
PAGES = "dashboard_ui/site_pages"
CLEAN_ENV = {key: value for key, value in os.environ.items() if not key.startswith("SHAPGPT_")}
# Menu names are English on the start screen and in the sidebar.
MENU = [
    ("Home", "home"),
    ("SECOM Diagnosis", "secom"),
    ("WM-811K Diagnosis", "wm811k"),
    ("Project & Validation", "project"),
]
BACKGROUND = (14, 23, 20)


def logo_rgb() -> np.ndarray:
    with Image.open(brand.LOGO_PATH) as image:
        return np.asarray(image.convert("RGB"))


def decode_png_uri(uri: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not uri.startswith(prefix):
        raise ValueError(uri[:40])
    return b64decode(uri[len(prefix) :])


def intro_mode(app: AppTest) -> str | None:
    for node in app.get("html"):
        match = re.search(r'class="shapgpt-intro" data-mode="(\w+)"', node.proto.body)
        if match:
            return match.group(1)
    return None


def page_links(block) -> list[tuple[str, str]]:
    return [(node.proto.label, node.proto.page) for node in block.get("page_link")]


def page_styles(app: AppTest) -> str:
    """CSS the page sent: brand styles travel as markdown, scripts as html."""
    bodies = [node.proto.body for node in app.get("html")]
    return " ".join(bodies + [str(element.value) for element in app.markdown])


def active_menu(app: AppTest) -> list[str]:
    return re.findall(
        r'\.st-key-brand_nav_(\w+) a\[data-testid="stPageLink-NavLink"\], ', page_styles(app)
    )


def page_marker(app: AppTest) -> list[str]:
    """Pages NAV_JS would see as arrived: the markers sent with the styles."""
    return re.findall(r'class="shapgpt-page" data-page="(\w+)"', page_styles(app))


def nav_ms(name: str) -> int:
    return int(re.search(rf"const {name} = (\d+);", brand.NAV_JS).group(1))


class HashGlowMaskTests(unittest.TestCase):
    def test_only_the_orange_hash_pixels_glow(self) -> None:
        rgb = logo_rgb()
        alpha = brand.hash_mask(rgb)
        self.assertEqual(alpha.shape, rgb.shape[:2])
        left, top, right, bottom = brand.HASH_REGION
        outside = np.ones(alpha.shape, dtype=bool)
        outside[top:bottom, left:right] = False
        self.assertFalse(alpha[outside].any())
        orange = np.all(rgb == brand.HASH_ORANGE, axis=-1)
        self.assertGreater(orange.sum(), 1000)
        self.assertTrue((alpha[orange] == 255).all())
        for colour in (brand.LOGO_MINT, BACKGROUND):
            self.assertFalse(alpha[np.all(rgb == colour, axis=-1)].any(), colour)
        # Anti-aliased edges keep a partial share instead of a hard cut.
        self.assertGreater(((alpha > 0) & (alpha < 255)).sum(), 100)

    def test_overlay_is_the_logo_itself_on_the_same_pixel_grid(self) -> None:
        logo_uri, hash_uri = brand.logo_layers(
            str(brand.LOGO_PATH), brand.LOGO_PATH.stat().st_mtime_ns
        )
        self.assertEqual(decode_png_uri(logo_uri), brand.LOGO_PATH.read_bytes())
        with Image.open(BytesIO(decode_png_uri(hash_uri))) as image:
            self.assertEqual(image.mode, "RGBA")
            overlay = np.asarray(image)
        rgb = logo_rgb()
        self.assertEqual(overlay.shape[:2], rgb.shape[:2])
        np.testing.assert_array_equal(overlay[..., 3], brand.hash_mask(rgb))
        lit = overlay[..., 3] > 0
        np.testing.assert_array_equal(overlay[lit][:, :3], rgb[lit])


class LogoCutoutTests(unittest.TestCase):
    def test_cutout_drops_the_dark_box_but_keeps_every_letter_pixel(self) -> None:
        png = site.logo_cutout_png(str(site.LOGO_PATH), site.LOGO_PATH.stat().st_mtime_ns)
        with Image.open(BytesIO(png)) as image:
            self.assertEqual(image.mode, "RGBA")
            cut = np.asarray(image)
        rgb = logo_rgb()
        self.assertEqual(cut.shape[:2], rgb.shape[:2])
        self.assertEqual(set(np.unique(cut[..., 3]).tolist()), {0, 255})
        opaque = cut[..., 3] == 255
        np.testing.assert_array_equal(cut[opaque][:, :3], rgb[opaque])
        # No frame is left around the letters ...
        for edge in (opaque[:8], opaque[-6:], opaque[:, :8], opaque[:, -8:]):
            self.assertFalse(edge.any())
        self.assertGreater((~opaque).mean(), 0.4)
        # ... while every bright letter pixel and the dark grid inside the
        # letters stay as they were.
        ink = np.abs(rgb.astype(np.int16) - np.array(BACKGROUND)).sum(axis=-1) > 120
        self.assertTrue(opaque[ink].all())
        np.testing.assert_array_equal(opaque, site.logo_outline(rgb))


def timing(css: str, name: str) -> tuple[int, int]:
    """(start, end) in ms of the first ``animation: <name> <duration> ... <delay>``."""
    match = re.search(rf"animation: {name} (\d+)ms [^;]*? (\d+)ms both", css)
    if match is None:
        match = re.search(rf"animation: {name} (\d+)ms [^;]*both", css)
        return 0, int(match.group(1))
    duration, delay = int(match.group(1)), int(match.group(2))
    return delay, delay + duration


class IntroTimingTests(unittest.TestCase):
    def test_sequence_follows_the_brand_timeline(self) -> None:
        css = brand.INTRO_PLAY_CSS
        self.assertEqual(timing(css, "shapgpt-logo-in"), (0, 600))
        glow_start, glow_end = timing(css, "shapgpt-hash-glow")
        lift_start, lift_end = timing(css, "shapgpt-lift")
        self.assertGreaterEqual(glow_start, 600)
        # The logo rises only after the "#" has glowed, and the menus spread
        # out while it is still rising.
        self.assertGreaterEqual(lift_start, glow_end - 150)
        menus = re.findall(r"brand_menu_(\d) \{ animation: shapgpt-wafer-in (\d+)ms .*? (\d+)ms both", css)
        self.assertEqual([index for index, _, _ in menus], ["1", "2", "3", "4"])
        starts = [int(delay) for _, _, delay in menus]
        self.assertTrue(lift_start <= starts[0] <= lift_end, (starts, lift_start, lift_end))
        self.assertTrue(all(90 <= b - a <= 120 for a, b in zip(starts, starts[1:])), starts)
        self.assertLessEqual(starts[-1] + int(menus[-1][1]), 2100)
        for block in (brand.INTRO_PLAY_CSS, brand.PAGE_ENTRANCE_CSS):
            self.assertIn("@media (prefers-reduced-motion: reduce)", block)

    def test_every_opening_plays_in_step(self) -> None:
        # No "played" memory: a reload plays the animation again.
        self.assertNotIn("sessionStorage", brand.INTRO_JS)
        self.assertIn('classList.remove("shapgpt-intro-instant")', brand.INTRO_JS)
        # It waits for the menus and the removal of the page just left.
        self.assertIn('classList.add("shapgpt-intro-wait")', brand.INTRO_JS)
        self.assertIn('[data-stale="true"]', brand.INTRO_JS)
        self.assertIn("html.shapgpt-intro-wait", brand.INTRO_PLAY_CSS)


class PageTransitionTests(unittest.TestCase):
    def test_timeline_fits_the_animations(self) -> None:
        css = brand.NAV_CSS
        # The page is opened right when its fade-out ends ...
        self.assertEqual(timing(css, "shapgpt-nav-out"), (0, nav_ms("OUT_MS")))
        intro_out = max(
            timing(css, name)[1]
            for name in ("shapgpt-intro-out", "shapgpt-wafer-out", "shapgpt-wafer-pick")
        )
        self.assertLessEqual(intro_out, nav_ms("OUT_INTRO_MS"))
        # ... and the arrival classes stay until the longest arrival is over.
        arrivals = [
            int(delay or 0) + int(duration)
            for duration, delay in re.findall(
                r"animation: shapgpt-(?:nav-in|rail-in) (\d+)ms [^;]*?(?:(\d+)ms )?both", css
            )
        ]
        self.assertLessEqual(max(arrivals), nav_ms("IN_MS"))
        self.assertLess(nav_ms("OUT_MS"), 300)
        self.assertLessEqual(nav_ms("SETTLE_MS"), 600)
        self.assertLessEqual(nav_ms("GIVE_UP_MS"), 5000)

    def test_links_still_open_pages_through_streamlit(self) -> None:
        js = brand.NAV_JS
        # The click is held, then the same Streamlit link is clicked again.
        self.assertIn('a[data-testid="stPageLink-NavLink"], button[data-testid="stLogoLink"]', js)
        self.assertIn("event.stopImmediatePropagation();", js)
        self.assertIn("again.click();", js)
        # Links to the page already shown, and reduced motion, go straight to Streamlit.
        self.assertIn("url.pathname === window.location.pathname", js)
        self.assertIn('matchMedia("(prefers-reduced-motion: reduce)")', js)
        self.assertIn("if (window.__shapgptNav)", js)
        # The styles travel inside the script, so they outlive the page left.
        self.assertIn(json.dumps(brand.NAV_CSS), js)
        self.assertTrue(js.isascii())


class StartScreenAppTests(unittest.TestCase):
    """The real app: start screen, skip, sidebar menu and the way back."""

    def run_app(self, app: AppTest | None = None, page: str | None = None) -> AppTest:
        if app is None:
            app = AppTest.from_file(APP, default_timeout=180)
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            if page is not None:
                # Menu links and the logo open a page without query parameters;
                # an old ?module= link on the start screen would redirect.
                app.query_params.clear()
                app.switch_page(f"{PAGES}/{page}.py")
            app.run()
        self.assertEqual(len(app.exception), 0, [str(error.value) for error in app.exception])
        return app

    def test_start_screen_holds_only_the_logo_and_four_menus(self) -> None:
        app = self.run_app()
        self.assertEqual(intro_mode(app), "play")
        self.assertEqual(page_marker(app), ["intro"])
        self.assertIn("window.__shapgptNav", page_styles(app))
        self.assertEqual(page_links(app.get_by_key("brand_menu")), MENU)
        self.assertEqual(len(app.sidebar.children), 0)
        self.assertEqual([(button.key, button.label) for button in app.button], [("brand_skip", "Skip")])
        for name in ("title", "header", "metric", "dataframe", "expander", "tabs"):
            self.assertEqual(len(getattr(app, name)), 0, name)

    def test_start_screen_text_is_english(self) -> None:
        app = self.run_app()
        intro = next(node.proto.body for node in app.get("html") if "shapgpt-intro" in node.proto.body)
        self.assertIn(brand.INTRO_KICKER, intro)
        self.assertIn("SHAPGPT", brand.INTRO_SUMMARY)
        self.assertIn("SHAP</b>", brand.INTRO_SUMMARY)
        shown = [label for label, _ in page_links(app)] + [button.label for button in app.button]
        shown += [node.proto.body for node in app.get("html")]
        for text in shown:
            self.assertTrue(text.isascii(), text[:80])

    def test_start_screen_loads_no_model(self) -> None:
        st.cache_resource.clear()
        with patch("joblib.load") as joblib_load, patch("torch.load") as torch_load:
            self.run_app()
        joblib_load.assert_not_called()
        torch_load.assert_not_called()

    def test_skip_finishes_the_intro_without_leaving(self) -> None:
        app = self.run_app()
        with patch.dict(os.environ, CLEAN_ENV, clear=True):
            app.button(key="brand_skip").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(intro_mode(app), "seen")
        self.assertEqual([button.key for button in app.button], [])
        self.assertEqual(page_links(app.get_by_key("brand_menu")), MENU)

    def test_returning_from_a_page_replays_the_intro(self) -> None:
        app = AppTest.from_file(APP, default_timeout=180)
        app.session_state[brand.LAST_PAGE_KEY] = "wm811k"
        self.run_app(app)
        self.assertEqual(intro_mode(app), "play")
        self.assertEqual([button.key for button in app.button], ["brand_skip"])
        self.assertIn("@keyframes shapgpt-wafer-in", page_styles(app))

    def test_pages_share_the_sidebar_menu_and_the_way_back(self) -> None:
        app = self.run_app()
        for _, slug in MENU:
            with self.subTest(page=slug):
                self.run_app(app, page=slug)
                self.assertIsNone(intro_mode(app))
                self.assertEqual(page_links(app.sidebar.get_by_key("brand_side_nav")), MENU)
                self.assertEqual(active_menu(app), [slug])
                self.assertEqual(page_marker(app), [slug])
                sidebar = " ".join(node.proto.body for node in app.sidebar.get("html"))
                self.assertIn('class="brand-nav__label">MENU</p>', sidebar)
                self.assertIn("window.__shapgptNav", sidebar)
                self.assertIn(".st-key-dashboard_hero::after", page_styles(app))
                # Pages opened from a menu are animated by NAV_JS, not by the
                # fade of the first page.
                self.assertNotIn("@keyframes shapgpt-page-in-", page_styles(app))
        # The logo opens the default page: the start screen plays again.
        self.run_app(app, page="intro")
        self.assertEqual(intro_mode(app), "play")
        self.assertEqual([button.key for button in app.button], ["brand_skip"])
        self.assertEqual(len(app.sidebar.children), 0)

    def test_first_page_of_a_visit_fades_in_on_its_own(self) -> None:
        app = AppTest.from_file(APP, default_timeout=180)
        self.run_app(app, page="home")
        self.assertIn("@keyframes shapgpt-page-in-home", page_styles(app))
        self.run_app(app, page="project")
        self.assertNotIn("@keyframes shapgpt-page-in-", page_styles(app))

    def test_secom_settings_and_results_survive_the_start_screen(self) -> None:
        app = self.run_app()
        self.run_app(app, page="secom")
        panel = app.sidebar.get_by_key("brand_side_panel")
        self.assertIn("진단 실행", [button.label for button in panel.get("button")])
        self.assertTrue(any(str(caption.value).startswith("실행 시각") for caption in panel.get("caption")))
        run_id = app.session_state["secom_diagnosis_run"]["run_id"]
        self.assertEqual(
            [app.get_by_key(f"xai_step_{index}").type for index in range(1, 5)],
            ["flex_container"] * 4,
        )
        self.run_app(app, page="intro")
        self.assertEqual(intro_mode(app), "play")
        self.run_app(app, page="secom")
        self.assertEqual(app.session_state["secom_diagnosis_run"]["run_id"], run_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
