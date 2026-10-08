"""SHAPGPT brand start screen and the dark sidebar of the content pages.

The start screen is its own page (``/``). The original logo appears in the
middle of the screen and the orange "#" pixels already in the artwork glow
once; then the logo glides up while an English introduction and four
wafer-shaped menus spread out below it. Menus and sidebar entries are
``st.page_link`` widgets, so choosing one runs the existing page code with the
current session state. JavaScript holds the start-screen animation until the
page is ready, animates the change from one page to the next (NAV_JS) and
closes the phone sidebar after a menu is chosen. The content pages get the
same look through ``PAGE_CSS``: styling only.
"""

from __future__ import annotations

import json
from base64 import b64encode
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from dashboard_ui.site import LOGO_PATH


LAST_PAGE_KEY = "brand_last_page"
# Box (left, top, right, bottom) of the orange "#" in the 1039x148 logo.
HASH_REGION = (678, 60, 752, 134)
# Every colour in the artwork mixes the orange of the "#" with a tone on the
# line from the dark background to the mint dies.
HASH_ORANGE = (232, 131, 95)
LOGO_MINT = (140, 199, 166)
MIN_HASH_SHARE = 0.08
INTRO_KICKER = "Explainable AI for semiconductor defect diagnosis"
INTRO_SUMMARY = (
    "SHAPGPT diagnoses semiconductor process data and wafer maps, then uses "
    "<b>SHAP</b> to reveal which sensors (SECOM) and which dies (WM-811K) drive "
    "every prediction."
)


@dataclass(frozen=True)
class MenuItem:
    slug: str
    label: str
    icon: str
    page: object


def _png_data_uri(raw: bytes) -> str:
    return "data:image/png;base64," + b64encode(raw).decode("ascii")


def hash_mask(rgb: np.ndarray) -> np.ndarray:
    """Alpha of the "#" pixels: each pixel's share of the orange colour.

    A least-squares split into the orange and mint tones gives anti-aliased
    edges their partial share and leaves letters, mint dies and background at
    zero. Pixels outside the "#" box never glow.
    """
    left, top, right, bottom = HASH_REGION
    box = rgb[top:bottom, left:right].reshape(-1, 3).T.astype(np.float64)
    basis = np.array([HASH_ORANGE, LOGO_MINT], dtype=np.float64).T
    share = np.linalg.lstsq(basis, box, rcond=None)[0][0]
    share = np.where(share >= MIN_HASH_SHARE, np.clip(share, 0.0, 1.0), 0.0)
    alpha = np.zeros(rgb.shape[:2], dtype=np.uint8)
    alpha[top:bottom, left:right] = np.round(share * 255).reshape(bottom - top, right - left)
    return alpha


@st.cache_data(show_spinner=False)
def logo_layers(path: str, modified_ns: int) -> tuple[str, str]:
    """The logo file as is, and a same-size overlay holding only its "#" pixels."""
    del modified_ns  # cache key only
    raw = Path(path).read_bytes()
    with Image.open(BytesIO(raw)) as image:
        rgb = np.asarray(image.convert("RGB"))
    alpha = hash_mask(rgb)
    overlay = np.dstack([np.where(alpha[..., None] > 0, rgb, 0), alpha]).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG", optimize=True)
    return _png_data_uri(raw), _png_data_uri(buffer.getvalue())


INTRO_CSS = """
:root {
    --brand-bg: #0e1714;
    --brand-mint: #8cc7a6;
    --brand-ink: #e9f2ee;
    --brand-accent: #e8835f;
    --wafer: 150px;
    --wafer-gap: 32px;
    --pitch: calc(var(--wafer) + var(--wafer-gap));
    --die: calc(var(--wafer) / 12);
    --menu-h: var(--wafer);
    --copy-gap: 28px;
    --copy-h: 84px;
    --menu-gap: 42px;
    /* While the logo is alone it sits this much lower, in the middle. */
    --rise: calc((var(--copy-gap) + var(--copy-h) + var(--menu-gap) + var(--menu-h)) / 2);
}

body .stApp,
body .stApp [data-testid="stAppViewContainer"],
body .stApp [data-testid="stMain"] {
    background: var(--brand-bg);
}

body [data-testid="stHeader"],
body [data-testid="stSidebar"] {
    display: none !important;
}

.stApp [data-testid="stMainBlockContainer"] {
    box-sizing: border-box;
    max-width: 1180px;
    min-height: 100vh;
    min-height: 100dvh;
    padding: 8vh 24px 10vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.stApp [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
    flex: 0 0 auto;
}

/* The app-wide style blocks render nothing; drop their layout gaps here. */
.stApp [data-testid="stMainBlockContainer"]
    [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) {
    display: none;
}

/* Elements of the page just left stay until Streamlit removes them, always
   after the menus; they must not push the logo out of the middle. */
.stApp [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]
    > [data-testid="stLayoutWrapper"]:has(> .st-key-brand_menu)
    ~ :not(.st-key-brand_skip) {
    display: none;
}

.shapgpt-intro {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.shapgpt-lift {
    width: min(64vw, 940px, 100%);
}

.shapgpt-logo {
    position: relative;
    width: 100%;
    aspect-ratio: 1039 / 148;
}

.shapgpt-logo > * {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    user-select: none;
    -webkit-user-drag: none;
}

/* The logo background equals the page background. Only the faint pixel grid
   at the outer edge (no letters there) fades out, so no rectangle shows. */
.shapgpt-logo__base {
    display: block;
    -webkit-mask-image:
        linear-gradient(90deg, transparent 0, #000 1%, #000 99%, transparent 100%),
        linear-gradient(180deg, transparent 0, #000 6%, #000 93%, transparent 100%);
    -webkit-mask-composite: source-in;
    mask-image:
        linear-gradient(90deg, transparent 0, #000 1%, #000 99%, transparent 100%),
        linear-gradient(180deg, transparent 0, #000 6%, #000 93%, transparent 100%);
    mask-composite: intersect;
}

.shapgpt-logo__hash,
.shapgpt-logo__sweep {
    opacity: 0;
}

.shapgpt-logo__sweep {
    -webkit-mask: var(--shapgpt-hash-mask) 0 0 / 100% 100% no-repeat;
    mask: var(--shapgpt-hash-mask) 0 0 / 100% 100% no-repeat;
    background: linear-gradient(
        100deg, transparent 38%, rgba(255, 236, 222, 0.9) 50%, transparent 62%
    ) 100% 0 / 300% 100% no-repeat;
}

.shapgpt-copy {
    box-sizing: border-box;
    min-height: var(--copy-h);
    max-width: 700px;
    margin-top: var(--copy-gap);
    padding: 0 8px;
    text-align: center;
}

.shapgpt-copy__kicker {
    margin: 0 0 12px;
    color: var(--brand-mint);
    font-size: 0.74rem;
    font-weight: 700;
    line-height: 1.4;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    text-wrap: balance;
}

.shapgpt-copy__summary {
    margin: 0;
    color: rgba(233, 242, 238, 0.8);
    font-size: 1.04rem;
    font-weight: 400;
    line-height: 1.6;
    text-wrap: balance;
}

.shapgpt-copy__summary b {
    color: var(--brand-ink);
    font-weight: 650;
}

/* Menus as wafers: a thin ring around a faint die grid, with the notch at the
   bottom. Hover or focus lights the grid and one die, like a SHAP highlight. */
.stApp .st-key-brand_menu {
    display: grid;
    grid-template-columns: repeat(4, var(--wafer));
    justify-content: center;
    gap: var(--wafer-gap);
    margin-top: calc(var(--menu-gap) - 1rem);
    counter-reset: wafer;
}

.stApp .st-key-brand_menu_1 { --from-x: calc(var(--pitch) * 1.5); --from-y: 0px; }
.stApp .st-key-brand_menu_2 { --from-x: calc(var(--pitch) * 0.5); --from-y: 0px; }
.stApp .st-key-brand_menu_3 { --from-x: calc(var(--pitch) * -0.5); --from-y: 0px; }
.stApp .st-key-brand_menu_4 { --from-x: calc(var(--pitch) * -1.5); --from-y: 0px; }

.stApp .st-key-brand_menu [data-testid="stElementContainer"] {
    margin: 0;
}

.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"] {
    position: relative;
    isolation: isolate;
    box-sizing: border-box;
    width: var(--wafer);
    height: var(--wafer);
    margin: 0;
    padding: 0 16px 6px;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    border-radius: 50%;
    background: transparent;
    color: var(--brand-ink);
    text-align: center;
    transition: transform 220ms ease, box-shadow 220ms ease;
}

.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"]::before,
.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"]::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: -1;
    border-radius: 50%;
}

.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"]::before {
    border: 1px solid rgba(140, 199, 166, 0.36);
    background:
        radial-gradient(circle at 50% 42%, rgba(140, 199, 166, 0.1), transparent 68%),
        linear-gradient(rgba(140, 199, 166, 0.07) 1px, transparent 1px)
            center / var(--die) var(--die),
        linear-gradient(90deg, rgba(140, 199, 166, 0.07) 1px, transparent 1px)
            center / var(--die) var(--die),
        #111b17;
    -webkit-mask: radial-gradient(circle at 50% 100%, transparent 6.5px, #000 7.5px);
    mask: radial-gradient(circle at 50% 100%, transparent 6.5px, #000 7.5px);
    transition: border-color 220ms ease;
}

.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"]::after {
    background:
        linear-gradient(var(--brand-accent), var(--brand-accent))
            calc(50% + var(--die) * 2) calc(50% - var(--die) * 3) /
            calc(var(--die) - 2px) calc(var(--die) - 2px) no-repeat,
        linear-gradient(rgba(140, 199, 166, 0.2) 1px, transparent 1px)
            center / var(--die) var(--die),
        linear-gradient(90deg, rgba(140, 199, 166, 0.2) 1px, transparent 1px)
            center / var(--die) var(--die);
    -webkit-mask: radial-gradient(circle at 50% 45%, #000 28%, transparent 70%);
    mask: radial-gradient(circle at 50% 45%, #000 28%, transparent 70%);
    opacity: 0;
    transition: opacity 260ms ease;
}

.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"]:hover,
.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"]:focus-visible {
    transform: translateY(-3px);
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.38);
}

.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"]:hover::before,
.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"]:focus-visible::before {
    border-color: rgba(140, 199, 166, 0.85);
}

.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"]:hover::after,
.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"]:focus-visible::after {
    opacity: 1;
}

.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"]:focus-visible {
    outline: 2px solid var(--brand-mint);
    outline-offset: 4px;
}

.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"] :is(span, div, p) {
    overflow: visible;
    color: inherit !important;
    white-space: normal;
}

.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"] p {
    max-width: 6.6rem;
    margin: 0 auto;
    font-size: 0.95rem;
    font-weight: 650;
    line-height: 1.3;
    letter-spacing: 0.01em;
}

.stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"] p::before {
    counter-increment: wafer;
    content: counter(wafer, decimal-leading-zero);
    display: block;
    margin-bottom: 7px;
    color: var(--brand-mint);
    font-family: "Source Code Pro", ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.2em;
}

.stApp .st-key-brand_skip {
    position: fixed;
    right: 22px;
    bottom: 18px;
    width: auto;
    z-index: 20;
}

.stApp .st-key-brand_skip button {
    min-height: 2rem;
    padding: 0.2rem 0.65rem;
    color: rgba(233, 242, 238, 0.62);
}

.stApp .st-key-brand_skip button :is(span, div, p) {
    color: inherit !important;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.stApp .st-key-brand_skip button:hover,
.stApp .st-key-brand_skip button:focus-visible {
    color: var(--brand-mint);
    box-shadow: none;
    transform: none;
}

.stApp .st-key-brand_skip button:focus-visible {
    outline: 2px solid var(--brand-mint);
    outline-offset: 2px;
}

@media (max-width: 900px) {
    :root {
        --wafer: 140px;
        --wafer-gap: 24px;
        --menu-h: calc(var(--wafer) * 2 + var(--wafer-gap));
        --copy-h: 92px;
        --menu-gap: 36px;
    }

    .shapgpt-lift {
        width: min(76vw, 100%);
    }

    .stApp .st-key-brand_menu {
        grid-template-columns: repeat(2, var(--wafer));
    }

    .stApp .st-key-brand_menu_1 { --from-x: calc(var(--pitch) * 0.5); --from-y: calc(var(--pitch) * 0.5); }
    .stApp .st-key-brand_menu_2 { --from-x: calc(var(--pitch) * -0.5); --from-y: calc(var(--pitch) * 0.5); }
    .stApp .st-key-brand_menu_3 { --from-x: calc(var(--pitch) * 0.5); --from-y: calc(var(--pitch) * -0.5); }
    .stApp .st-key-brand_menu_4 { --from-x: calc(var(--pitch) * -0.5); --from-y: calc(var(--pitch) * -0.5); }
}

@media (max-width: 560px) {
    :root {
        --wafer: min(40vw, 146px);
        --wafer-gap: 18px;
        --copy-gap: 22px;
        --copy-h: 132px;
        --menu-gap: 30px;
    }

    .stApp [data-testid="stMainBlockContainer"] {
        padding-left: 16px;
        padding-right: 16px;
    }

    .shapgpt-lift {
        width: min(88vw, 100%);
    }

    .shapgpt-copy__kicker {
        font-size: 0.68rem;
        letter-spacing: 0.2em;
    }

    .shapgpt-copy__summary {
        font-size: 0.95rem;
    }

    .stApp .st-key-brand_menu a[data-testid="stPageLink-NavLink"] p {
        font-size: 0.9rem;
    }
}
"""

# Sent when the start screen opens: first visit, reload, or back from a page.
INTRO_PLAY_CSS = """
.shapgpt-logo {
    animation: shapgpt-logo-in 600ms cubic-bezier(0.16, 1, 0.3, 1) both;
}

.shapgpt-logo__hash {
    animation: shapgpt-hash-glow 550ms ease-out 600ms both;
}

.shapgpt-logo__sweep {
    animation: shapgpt-hash-sweep 500ms ease-in-out 650ms both;
}

.shapgpt-lift {
    animation: shapgpt-lift 600ms cubic-bezier(0.65, 0, 0.35, 1) 1050ms both;
}

.shapgpt-copy__kicker {
    animation: shapgpt-copy-in 650ms cubic-bezier(0.22, 1, 0.36, 1) 1200ms both;
}

.shapgpt-copy__summary {
    animation: shapgpt-copy-in 650ms cubic-bezier(0.22, 1, 0.36, 1) 1300ms both;
}

.st-key-brand_menu_1 { animation: shapgpt-wafer-in 600ms cubic-bezier(0.16, 1, 0.3, 1) 1200ms both; }
.st-key-brand_menu_2 { animation: shapgpt-wafer-in 600ms cubic-bezier(0.16, 1, 0.3, 1) 1290ms both; }
.st-key-brand_menu_3 { animation: shapgpt-wafer-in 600ms cubic-bezier(0.16, 1, 0.3, 1) 1380ms both; }
.st-key-brand_menu_4 { animation: shapgpt-wafer-in 600ms cubic-bezier(0.16, 1, 0.3, 1) 1470ms both; }

.stApp .st-key-brand_skip {
    animation: shapgpt-skip-out 250ms ease 2100ms both;
}

@keyframes shapgpt-logo-in {
    from { opacity: 0; transform: translateY(10px) scale(0.97); }
    to { opacity: 1; transform: none; }
}

@keyframes shapgpt-hash-glow {
    0% { opacity: 0; filter: brightness(1); }
    35% {
        opacity: 1;
        filter: brightness(1.65) saturate(1.15)
            drop-shadow(0 0 2px rgba(255, 178, 140, 0.9))
            drop-shadow(0 0 6px rgba(232, 131, 95, 0.45));
    }
    100% { opacity: 0; filter: brightness(1); }
}

@keyframes shapgpt-hash-sweep {
    0% { opacity: 0; background-position: 100% 0; }
    30% { opacity: 1; }
    100% { opacity: 0; background-position: 0 0; }
}

@keyframes shapgpt-lift {
    from { transform: translateY(var(--rise)); }
    to { transform: none; }
}

@keyframes shapgpt-copy-in {
    from { opacity: 0; clip-path: inset(0 50% 0 50%); }
    to { opacity: 1; clip-path: inset(0 0 0 0); }
}

/* Wafers spread out from the middle; hidden ones are out of the focus order. */
@keyframes shapgpt-wafer-in {
    from {
        opacity: 0;
        transform: translate(var(--from-x), var(--from-y)) scale(0.7);
        visibility: hidden;
    }
    to { opacity: 1; transform: none; visibility: visible; }
}

@keyframes shapgpt-skip-out {
    to { opacity: 0; visibility: hidden; }
}

/* Held at the first frame until the menus are in place and the page just
   left is gone (see INTRO_JS), then everything starts together. */
html.shapgpt-intro-wait :is(
    .shapgpt-lift, .shapgpt-logo, .shapgpt-logo > *, .shapgpt-copy > *,
    [class*="st-key-brand_menu_"], .st-key-brand_skip
) {
    animation-play-state: paused;
}

html.shapgpt-intro-instant :is(
    .shapgpt-lift, .shapgpt-logo, .shapgpt-logo > *, .shapgpt-copy > *,
    [class*="st-key-brand_menu_"]
) {
    animation: none !important;
}

html.shapgpt-intro-instant .stApp .st-key-brand_skip {
    display: none;
}

@media (prefers-reduced-motion: reduce) {
    :is(
        .shapgpt-lift, .shapgpt-logo, .shapgpt-logo > *, .shapgpt-copy > *,
        [class*="st-key-brand_menu_"]
    ) {
        animation: none !important;
    }

    .stApp .st-key-brand_skip {
        display: none;
    }
}
"""

SHELL_CSS = """
:root {
    --shell-rail: #14271f;
    --shell-line: #24392f;
    --shell-text: #c5d5ce;
    --shell-strong: #f3f8f5;
    --shell-muted: #7d938a;
    --shell-mint: #8cc7a6;
}

.stApp {
    transition: background-color 320ms ease;
}

/* A light gray page instead of pure white, so the dark rail and the
   banners meet a soft tone rather than a hard black-and-white edge. */
body .stApp,
body .stApp [data-testid="stAppViewContainer"] {
    background: #f0f2f4;
}

/* Style blocks render nothing; hidden, they add no gaps. The extra top
   padding keeps the page where the gaps of the two app-wide blocks had it. */
.stApp [data-testid="stMainBlockContainer"]
    [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) {
    display: none;
}

.stApp [data-testid="stMainBlockContainer"] {
    padding-top: 4.6rem;
}

@media (max-width: 900px) {
    .stApp [data-testid="stMainBlockContainer"] {
        padding-top: 4.4rem;
    }
}

.stApp [data-testid="stSidebar"],
.stApp [data-testid="stSidebarContent"] {
    background: var(--shell-rail);
}

.stApp [data-testid="stSidebar"] {
    border-right: 1px solid var(--shell-line);
    box-shadow: 10px 0 30px rgba(17, 24, 39, 0.08);
}

/* The logo sits centred and lower; the collapse button moves to the corner. */
.stApp [data-testid="stSidebarHeader"] {
    position: relative;
    justify-content: center;
    height: auto;
    padding: 3rem 0 0.4rem;
}

.stApp [data-testid="stSidebar"] img[data-testid="stSidebarLogo"] {
    width: 228px;
}

.stApp [data-testid="stSidebarCollapseButton"] {
    position: absolute;
    top: 0.8rem;
    right: -0.35rem;
}

.stApp [data-testid="stSidebarCollapseButton"] button {
    color: var(--shell-muted);
}

.stApp [data-testid="stSidebarCollapseButton"] button:hover {
    color: var(--shell-strong);
    background: rgba(140, 199, 166, 0.1);
}

/* Streamlit's logo button opens the default page, the start screen. */
.stApp [data-testid="stLogoLink"]:focus-visible {
    outline: 2px solid #0f766e;
    outline-offset: 3px;
    border-radius: 4px;
}

.stApp [data-testid="stSidebar"] [data-testid="stLogoLink"]:focus-visible {
    outline-color: var(--shell-mint);
}

.brand-nav__label {
    margin: 2.4rem 0.85rem 0.45rem;
    color: var(--shell-muted);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
}

.stApp .st-key-brand_side_nav {
    gap: 0.2rem;
}

.stApp .st-key-brand_side_nav a[data-testid="stPageLink-NavLink"] {
    gap: 0.75rem;
    margin: 0;
    padding: 0.62rem 0.85rem;
    border-radius: 10px;
    color: var(--shell-text);
    background: transparent;
    transition: background-color 150ms ease, color 150ms ease;
}

.stApp .st-key-brand_side_nav [data-testid="stElementContainer"] {
    margin: 0;
}

.stApp .st-key-brand_side_nav a[data-testid="stPageLink-NavLink"] :is(span, div, p) {
    color: inherit !important;
}

.stApp .st-key-brand_side_nav a[data-testid="stPageLink-NavLink"] p {
    font-size: 0.95rem;
    font-weight: 600;
}

.stApp .st-key-brand_side_nav a[data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"] {
    color: #7fa592 !important;
}

.stApp .st-key-brand_side_nav a[data-testid="stPageLink-NavLink"]:hover {
    background: rgba(140, 199, 166, 0.08);
    color: var(--shell-strong);
}

.stApp .st-key-brand_side_nav a[data-testid="stPageLink-NavLink"]:focus-visible {
    outline: 2px solid var(--shell-mint);
    outline-offset: 2px;
}

.brand-nav__divider {
    height: 1px;
    margin: 0.55rem 0.85rem;
    border: 0;
    background: var(--shell-line);
}

/* Page settings (SECOM) sit on a charcoal panel that goes with the green
   rail. Every control in it is restyled for the dark surface; the dropdown
   list opens outside the panel and keeps the regular light theme. */
.stApp .st-key-brand_side_panel {
    --panel-bg: #242d29;
    --panel-field: #2f3a35;
    --panel-line: rgba(255, 255, 255, 0.1);
    --panel-text: #e8eeeb;
    --panel-muted: #b3bfb9;
    margin-top: 1.4rem;
    padding: 1.15rem 1rem 1.1rem;
    border: 1px solid var(--panel-line);
    border-radius: 14px;
    background: linear-gradient(180deg, #27312c, var(--panel-bg));
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.25);
    color: var(--panel-text);
}

.stApp .st-key-brand_side_panel :is(h1, h2, h3) {
    color: var(--panel-text) !important;
}

.stApp .st-key-brand_side_panel [data-testid="stWidgetLabel"] p,
.stApp .st-key-brand_side_panel [data-testid="stCaptionContainer"],
.stApp .st-key-brand_side_panel [data-testid="stCaptionContainer"] p,
.stApp .st-key-brand_side_panel [data-testid="stSliderTickBar"],
.stApp .st-key-brand_side_panel [data-testid="stSliderTickBar"] p,
.stApp .st-key-brand_side_panel [data-testid="stFileUploaderDropzoneInstructions"] :is(span, small, div, p) {
    color: var(--panel-muted) !important;
}

.stApp .st-key-brand_side_panel [data-testid="stRadioOption"] p,
.stApp .st-key-brand_side_panel [data-testid="stExpander"] :is(summary, summary *),
.stApp .st-key-brand_side_panel [data-testid="stExpanderDetails"],
.stApp .st-key-brand_side_panel [data-testid="stFileUploaderFile"] :is(span, small, div, p) {
    color: var(--panel-text);
}

/* Radio: a light ring when off, a mint disc with a dark dot when on. */
.stApp .st-key-brand_side_panel [data-testid="stRadioOption"] > div > div > div:first-child {
    background: rgba(255, 255, 255, 0.32);
}

.stApp .st-key-brand_side_panel [data-testid="stRadioOption"] > div > div > div:first-child > div {
    background: var(--panel-bg);
}

.stApp .st-key-brand_side_panel [data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child {
    background: var(--shell-mint);
}

.stApp .st-key-brand_side_panel [data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child > div {
    background: #1b2420;
}

.stApp .st-key-brand_side_panel [data-testid="stSelectbox"] [role="group"],
.stApp .st-key-brand_side_panel [data-testid="stFileUploaderDropzone"] {
    border: 1px solid var(--panel-line);
    background: var(--panel-field);
    color: var(--panel-text);
}

.stApp .st-key-brand_side_panel [data-testid="stFileUploaderDropzone"] {
    border-style: dashed;
    border-color: rgba(255, 255, 255, 0.22);
}

.stApp .st-key-brand_side_panel [data-testid="stSelectbox"] :is(input, button) {
    color: var(--panel-text);
}

/* The slider track's empty part is translucent; a light base keeps it seen. */
.stApp .st-key-brand_side_panel [data-testid="stSlider"] [role="group"] > div > div:first-child {
    background-color: rgba(255, 255, 255, 0.16);
}

.stApp .st-key-brand_side_panel [data-testid="stSliderThumbValue"],
.stApp .st-key-brand_side_panel [data-testid="stSliderThumbValue"] p {
    color: var(--shell-mint) !important;
}

.stApp .st-key-brand_side_panel [data-testid="stBaseButton-secondary"] {
    border-color: var(--panel-line);
    background: rgba(255, 255, 255, 0.05);
    color: var(--panel-text);
}

.stApp .st-key-brand_side_panel [data-testid="stBaseButton-secondary"]:hover {
    border-color: rgba(140, 199, 166, 0.6);
    background: rgba(140, 199, 166, 0.1);
    color: #ffffff;
}

.stApp .st-key-brand_side_panel [data-testid="stBaseButton-secondary"] :is(span, div, p) {
    color: inherit !important;
}

.stApp .st-key-brand_side_panel [data-testid="stExpander"],
.stApp .st-key-brand_side_panel [data-testid="stExpanderDetails"] {
    border-color: var(--panel-line);
    background: rgba(255, 255, 255, 0.03);
}

.stApp .st-key-brand_side_panel [data-testid="stExpander"] [data-testid="stIconMaterial"] {
    color: var(--shell-mint);
}

/* Streamlit lightens an open or hovered expander header; keep it dark. */
.stApp .st-key-brand_side_panel [data-testid="stExpander"] summary,
.stApp .st-key-brand_side_panel [data-testid="stExpander"] summary:is(:hover, :focus-visible),
.stApp .st-key-brand_side_panel [data-testid="stExpander"] details[open] > summary {
    background: rgba(255, 255, 255, 0.05);
}

.stApp .st-key-brand_side_panel [data-testid="stAlertContainer"] {
    border: 1px solid rgba(140, 199, 166, 0.3) !important;
    background: rgba(140, 199, 166, 0.1) !important;
    color: var(--panel-text) !important;
}

.stApp [data-testid="stHeader"] {
    background: rgba(240, 242, 244, 0.88);
    border-bottom-color: transparent;
    backdrop-filter: blur(8px);
}
"""

# Look of the content pages, matched to the start screen. Only styling: every
# text, number, table and chart stays as the page renders it.
PAGE_CSS = """
:root {
    --page-ink: #1b2420;
    --page-body: #4b5450;
    --page-soft: #6d7672;
    --page-line: #dfe3e6;
    --page-card: #ffffff;
    --page-mint: #8cc7a6;
    --page-teal: #0f766e;
    --page-deep: #1e4337;
    --page-dark: #0e1714;
    --page-accent: #e8835f;
}

/* Page banners: a deep emerald take on the start screen (softer than its
   near-black) with a die grid and a wafer peeking from the corner; one
   orange die marks the SHAP highlight. */
.stApp .st-key-site_hero,
.stApp .st-key-dashboard_hero {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    border: 1px solid rgba(140, 199, 166, 0.26);
    border-radius: 18px;
    background:
        radial-gradient(120% 160% at 0% 0%, rgba(140, 199, 166, 0.2), transparent 55%),
        linear-gradient(rgba(140, 199, 166, 0.06) 1px, transparent 1px) 0 0 / 22px 22px,
        linear-gradient(90deg, rgba(140, 199, 166, 0.06) 1px, transparent 1px) 0 0 / 22px 22px,
        linear-gradient(135deg, #17352c 0%, #1e4337 55%, #285243 100%);
    box-shadow: 0 16px 40px rgba(23, 53, 44, 0.16);
    color: #e9f2ee;
}

.stApp .st-key-dashboard_hero {
    --hero-wafer: 250px;
    padding: 1.75rem 2rem 1.6rem;
    margin-bottom: 1.4rem;
}

.stApp .st-key-site_hero {
    --hero-wafer: 380px;
    padding: 2.7rem 2.6rem 2.3rem;
}

.stApp .st-key-site_hero > *,
.stApp .st-key-dashboard_hero > * {
    position: relative;
    z-index: 1;
}

/* A lit top edge instead of the old side bar. */
.stApp .st-key-site_hero::before,
.stApp .st-key-dashboard_hero::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    z-index: 0;
    width: auto;
    height: 1px;
    background: linear-gradient(
        90deg, transparent, rgba(140, 199, 166, 0.7) 28%, rgba(140, 199, 166, 0.18) 72%, transparent
    );
}

.stApp .st-key-site_hero::after,
.stApp .st-key-dashboard_hero::after {
    --hero-die: calc(var(--hero-wafer) / 14);
    content: "";
    position: absolute;
    right: calc(var(--hero-wafer) * -0.3);
    bottom: calc(var(--hero-wafer) * -0.5);
    z-index: 0;
    width: var(--hero-wafer);
    height: var(--hero-wafer);
    border: 1px solid rgba(140, 199, 166, 0.26);
    border-radius: 50%;
    background:
        linear-gradient(var(--page-accent), var(--page-accent))
            calc(50% - var(--hero-die) * 2) calc(50% - var(--hero-die) * 2) /
            calc(var(--hero-die) - 3px) calc(var(--hero-die) - 3px) no-repeat,
        linear-gradient(rgba(232, 131, 95, 0.5), rgba(232, 131, 95, 0.5))
            calc(50% - var(--hero-die) * 3) calc(50% - var(--hero-die) * 2) /
            calc(var(--hero-die) - 3px) calc(var(--hero-die) - 3px) no-repeat,
        linear-gradient(rgba(140, 199, 166, 0.13) 1px, transparent 1px)
            center / var(--hero-die) var(--hero-die),
        linear-gradient(90deg, rgba(140, 199, 166, 0.13) 1px, transparent 1px)
            center / var(--hero-die) var(--hero-die),
        radial-gradient(circle at 40% 36%, rgba(140, 199, 166, 0.14), transparent 70%);
    pointer-events: none;
}

.stApp .st-key-dashboard_hero h1 {
    color: #f2f8f5 !important;
    letter-spacing: -0.03em !important;
}

.stApp .st-key-dashboard_hero [data-testid="stCaptionContainer"],
.stApp .st-key-dashboard_hero [data-testid="stCaptionContainer"] p {
    max-width: min(58rem, 80%);
    color: rgba(226, 236, 231, 0.74) !important;
    font-size: 0.9rem;
    line-height: 1.65;
}

.stApp .st-key-site_hero .site-eyebrow {
    color: var(--page-mint);
    letter-spacing: 0.22em;
    text-wrap: balance;
}

.stApp .st-key-site_hero [data-testid="stMarkdownContainer"] h1.site-hero__title {
    max-width: 52rem;
    color: #f2f8f5;
    text-wrap: balance;
}

.stApp .st-key-dashboard_hero h1 {
    text-wrap: balance;
}

.stApp .st-key-site_hero .site-hero__lead {
    color: rgba(226, 236, 231, 0.76);
}

.stApp .st-key-site_hero [data-testid="stCaptionContainer"],
.stApp .st-key-site_hero [data-testid="stCaptionContainer"] p {
    color: rgba(226, 236, 231, 0.66) !important;
}

/* Korean lines break between words, not inside them (Streamlit sets
   word-break on the text elements themselves, so they are named here). */
.stApp [data-testid="stMain"] :is(
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"]
),
.stApp [data-testid="stMain"] :is(
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"]
) :is(p, li, h1, h2, h3, h4, h5, h6, span, div, a, b, strong) {
    word-break: keep-all;
    overflow-wrap: break-word;
}

.stApp .st-key-site_cta_primary [data-testid="stPageLink"] a,
.stApp .st-key-site_cta_secondary [data-testid="stPageLink"] a {
    min-width: 10.5rem;
    min-height: 2.9rem;
    padding: 0.5rem 1.35rem;
    border-radius: 12px;
    transition: transform 180ms ease, box-shadow 180ms ease, background-color 180ms ease;
}

/* The button labels never clip: the label keeps its full width. */
.stApp :is(.st-key-site_cta_primary, .st-key-site_cta_secondary) [data-testid="stPageLink"] a :is(span, div, p) {
    overflow: visible;
    white-space: nowrap;
}

/* On wider screens the two button columns take the width of their labels
   instead of a fixed fifth of the banner; phones keep full-width buttons. */
@media (min-width: 641px) {
    .stApp .st-key-site_hero [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:has(
        :is(.st-key-site_cta_primary, .st-key-site_cta_secondary)
    ) {
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
    }
}

.stApp .st-key-site_cta_primary [data-testid="stPageLink"] a {
    border-color: var(--page-mint);
    background: var(--page-mint);
}

.stApp .st-key-site_cta_primary [data-testid="stPageLink"] a p {
    color: var(--page-dark);
}

.stApp .st-key-site_cta_secondary [data-testid="stPageLink"] a {
    border-color: rgba(140, 199, 166, 0.55);
    background: rgba(140, 199, 166, 0.06);
}

.stApp .st-key-site_cta_secondary [data-testid="stPageLink"] a p {
    color: #e9f2ee;
}

.stApp .st-key-site_cta_primary [data-testid="stPageLink"] a:hover,
.stApp .st-key-site_cta_secondary [data-testid="stPageLink"] a:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.3);
}

.stApp .st-key-site_cta_secondary [data-testid="stPageLink"] a:hover {
    background: rgba(140, 199, 166, 0.14);
}

.stApp .st-key-site_cta_primary [data-testid="stPageLink"] a:focus-visible,
.stApp .st-key-site_cta_secondary [data-testid="stPageLink"] a:focus-visible {
    outline: 2px solid var(--page-mint);
    outline-offset: 3px;
}

/* Section titles carry a small die, the motif of the logo. */
.stApp [data-testid="stMain"] [data-testid="stHeadingWithActionElements"] > h2,
.stApp .st-key-xai_frame h4 {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: var(--page-ink);
}

.stApp [data-testid="stMarkdownContainer"] h2.site-section-title {
    margin-top: 2.6rem;
}

.stApp [data-testid="stMain"] [data-testid="stHeadingWithActionElements"] > h2::before,
.stApp .st-key-xai_frame h4::before {
    content: "";
    flex: 0 0 auto;
    width: 0.62rem;
    height: 0.62rem;
    border-radius: 2px;
    background: linear-gradient(135deg, var(--page-mint), var(--page-teal));
    box-shadow: 0 0 0 3px rgba(140, 199, 166, 0.2);
}

.stApp .site-section-lead {
    color: var(--page-soft);
    line-height: 1.65;
}

/* Card frames: a hairline that turns from mint to silver and back to teal,
   with inspection marks at two corners, like the frame around the "#" of
   the logo. */
.stApp :is(
    [class*="st-key-site_card"], [class*="st-key-site_scope"],
    [class*="st-key-site_metric"], [class*="st-key-site_admin"],
    [class*="st-key-xai_step_"]
),
.stApp .site-step {
    position: relative;
    border: 1px solid transparent;
    border-radius: 16px;
    background:
        linear-gradient(var(--page-card), var(--page-card)) padding-box,
        linear-gradient(
            135deg,
            rgba(140, 199, 166, 0.95) 0%,
            rgba(208, 214, 219, 0.95) 28%,
            rgba(223, 227, 231, 0.95) 72%,
            rgba(15, 118, 110, 0.5) 100%
        ) border-box;
    box-shadow:
        0 1px 2px rgba(17, 24, 39, 0.04),
        0 12px 28px rgba(17, 24, 39, 0.06);
    transition: transform 180ms ease, box-shadow 180ms ease;
}

.stApp :is(
    [class*="st-key-site_card"], [class*="st-key-site_scope"],
    [class*="st-key-site_metric"], [class*="st-key-site_admin"],
    [class*="st-key-xai_step_"]
)::before,
.stApp .site-step::before {
    --mark: rgba(15, 118, 110, 0.5);
    content: "";
    position: absolute;
    inset: 8px;
    pointer-events: none;
    background:
        linear-gradient(var(--mark), var(--mark)) left top / 12px 1.5px no-repeat,
        linear-gradient(var(--mark), var(--mark)) left top / 1.5px 12px no-repeat,
        linear-gradient(var(--mark), var(--mark)) right bottom / 12px 1.5px no-repeat,
        linear-gradient(var(--mark), var(--mark)) right bottom / 1.5px 12px no-repeat;
}

.stApp :is([class*="st-key-site_card"], [class*="st-key-site_admin"]):hover {
    box-shadow: 0 2px 4px rgba(17, 24, 39, 0.05), 0 18px 36px rgba(17, 24, 39, 0.09);
    transform: translateY(-2px);
}

.stApp :is([class*="st-key-site_card"], [class*="st-key-site_admin"]):hover::before {
    --mark: rgba(15, 118, 110, 0.95);
}

/* Metric tiles elsewhere (diagnosis, admin) get the same hairline; their
   own top accents stay. Inside the site cards the tile stays plain. */
.stApp [data-testid="stMain"] [data-testid="stMetric"] {
    border: 1px solid transparent;
    border-radius: 14px;
    background:
        linear-gradient(#ffffff, #ffffff) padding-box,
        linear-gradient(
            135deg,
            rgba(140, 199, 166, 0.9) 0%,
            rgba(208, 214, 219, 0.95) 30%,
            rgba(223, 227, 231, 0.95) 72%,
            rgba(15, 118, 110, 0.45) 100%
        ) border-box;
}

.stApp [class*="st-key-site_metric"] [data-testid="stMetric"] {
    border: 0;
    background: transparent;
}

.stApp .site-card h3 {
    color: var(--page-ink);
    letter-spacing: -0.02em;
}

.stApp .site-card p,
.stApp .site-card li,
.stApp .site-condition p {
    color: var(--page-body);
    line-height: 1.68;
}

.stApp .site-card li::marker {
    color: var(--page-teal);
}

.stApp .site-condition h4,
.stApp .site-condition b {
    color: var(--page-ink);
}

.stApp .site-condition {
    border-top-color: var(--page-line);
}

.stApp .site-badge {
    border-color: #d4e7de;
    color: #0f5f57;
    background: #eef7f2;
    letter-spacing: 0.02em;
}

.stApp [class*="st-key-site_metric"] [data-testid="stMetricValue"] {
    color: var(--page-ink);
}

.stApp [class*="st-key-site_metric"] [data-testid="stMetricLabel"] p {
    color: var(--page-soft);
}

.stApp .site-source {
    color: #8a9a94;
}

/* Analysis steps: a dark STEP chip like the start-screen index. */
.stApp .site-step {
    padding: 1.1rem 1.15rem 1.15rem;
}

.stApp .site-step__number {
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    color: var(--page-mint);
    background: var(--page-deep);
    font-size: 0.68rem;
    letter-spacing: 0.14em;
}

.stApp .site-step h4 {
    margin-top: 0.55rem;
    color: var(--page-ink);
}

.stApp .site-step p {
    color: var(--page-body);
    line-height: 1.62;
}

.stApp .site-step:not(:last-child)::after {
    color: var(--page-teal);
}

/* The shared SHAP decision frame: the four boxes always share one height. */
.stApp .st-key-xai_frame [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    height: 100%;
}

.stApp .st-key-xai_frame [data-testid="stColumn"] > [data-testid="stVerticalBlock"]
    > [data-testid="stLayoutWrapper"] {
    flex: 1 1 auto;
}

.stApp [class*="st-key-xai_step_"] {
    height: 100%;
}

.stApp [class*="st-key-xai_step_"] strong {
    color: var(--page-ink);
}

.stApp [class*="st-key-xai_step_"] [data-testid="stCaptionContainer"] p {
    color: var(--page-body);
    line-height: 1.6;
}

/* Tabs, expanders and the footer follow the same mint accents. */
.stApp [role="radiogroup"] > button[data-selected="true"] {
    color: var(--page-ink) !important;
}

.stApp [data-testid="stExpander"] {
    border-color: var(--page-line);
    border-radius: 12px;
    background: var(--page-card);
}

/* Charts keep their colours; they just sit on a card like the rest. */
.stApp [data-testid="stMain"] [data-testid="stPlotlyChart"] {
    padding: 0.6rem 0.7rem 0.35rem;
    border: 1px solid var(--page-line);
    border-radius: 14px;
    background: #ffffff;
    box-shadow: 0 6px 18px rgba(20, 42, 34, 0.05);
}

.stApp .st-key-site_project_area [data-testid="stImage"] img {
    border-radius: 12px;
}

.stApp [data-testid="stMain"] :is(h1, h2, h3, h4) {
    color: var(--page-ink);
}

.stApp [data-testid="stExpander"] summary:hover,
.stApp [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
    color: var(--page-teal);
}

.stApp .st-key-site_footer {
    border-top-color: var(--page-line);
}

.stApp .site-footer__brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--page-dark);
    font-size: 0.82rem;
    letter-spacing: 0.2em;
}

.stApp .site-footer__brand::before {
    content: "";
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 2px;
    background: var(--page-accent);
}

@media (max-width: 900px) {
    .stApp .st-key-dashboard_hero {
        --hero-wafer: 190px;
        padding: 1.4rem 1.2rem 1.3rem;
    }

    .stApp .st-key-site_hero {
        --hero-wafer: 260px;
    }

    .stApp .st-key-dashboard_hero [data-testid="stCaptionContainer"],
    .stApp .st-key-dashboard_hero [data-testid="stCaptionContainer"] p {
        max-width: 100%;
    }
}

@media (max-width: 640px) {
    .stApp .st-key-site_hero {
        --hero-wafer: 200px;
        padding: 1.7rem 1.2rem 1.5rem;
    }
}
"""

# The first page of a visit fades in on its own; later page changes are
# animated by NAV_JS.
PAGE_ENTRANCE_CSS = """
.stApp [data-testid="stMainBlockContainer"] {
    animation: shapgpt-page-in-SLUG 320ms ease-out both;
}

@keyframes shapgpt-page-in-SLUG {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: none; }
}

@media (prefers-reduced-motion: reduce) {
    .stApp [data-testid="stMainBlockContainer"] {
        animation: none;
    }
}
"""

# Page transitions. NAV_JS puts these styles in the document head, so they
# outlive the styles of the page being left. The classes on <html> mark the
# phases: out (the page fades away), settle (the next page is arriving), in
# (it rises into place). A dark curtain links the start screen and the pages.
NAV_CSS = """
/* After the first menu click, NAV_JS animates every page change. */
html.shapgpt-nav-used [data-testid="stMainBlockContainer"] {
    animation: none;
}

/* Leaving a page: its content fades and lifts a little, and so does the
   SECOM settings panel in the sidebar (marked by NAV_JS). */
html.shapgpt-nav-out:not(.shapgpt-nav-from-intro) [data-testid="stMainBlockContainer"],
html.shapgpt-nav-out .stApp .st-key-brand_side_panel[data-shapgpt-left] {
    animation: shapgpt-nav-out 220ms cubic-bezier(0.4, 0, 1, 1) both;
}

/* Leaving the start screen: the chosen wafer lights up and the rest fades
   into the dark. */
html.shapgpt-nav-out .shapgpt-intro,
html.shapgpt-nav-out .stApp .st-key-brand_skip {
    animation: shapgpt-intro-out 360ms cubic-bezier(0.4, 0, 1, 1) both;
}

html.shapgpt-nav-out .stApp [class*="st-key-brand_menu_"] {
    animation: shapgpt-wafer-out 300ms cubic-bezier(0.4, 0, 1, 1) both;
}

html[data-shapgpt-pick="1"] .stApp .st-key-brand_menu_1,
html[data-shapgpt-pick="2"] .stApp .st-key-brand_menu_2,
html[data-shapgpt-pick="3"] .stApp .st-key-brand_menu_3,
html[data-shapgpt-pick="4"] .stApp .st-key-brand_menu_4 {
    animation: shapgpt-wafer-pick 420ms cubic-bezier(0.22, 1, 0.36, 1) both;
}

html[data-shapgpt-pick="1"] .stApp .st-key-brand_menu_1 a[data-testid="stPageLink-NavLink"]::before,
html[data-shapgpt-pick="2"] .stApp .st-key-brand_menu_2 a[data-testid="stPageLink-NavLink"]::before,
html[data-shapgpt-pick="3"] .stApp .st-key-brand_menu_3 a[data-testid="stPageLink-NavLink"]::before,
html[data-shapgpt-pick="4"] .stApp .st-key-brand_menu_4 a[data-testid="stPageLink-NavLink"]::before {
    border-color: rgba(140, 199, 166, 0.95);
}

html[data-shapgpt-pick="1"] .stApp .st-key-brand_menu_1 a[data-testid="stPageLink-NavLink"]::after,
html[data-shapgpt-pick="2"] .stApp .st-key-brand_menu_2 a[data-testid="stPageLink-NavLink"]::after,
html[data-shapgpt-pick="3"] .stApp .st-key-brand_menu_3 a[data-testid="stPageLink-NavLink"]::after,
html[data-shapgpt-pick="4"] .stApp .st-key-brand_menu_4 a[data-testid="stPageLink-NavLink"]::after {
    opacity: 1;
}

/* The curtain has the colour of the start screen. */
.shapgpt-curtain {
    position: fixed;
    inset: 0;
    z-index: 1000010;
    pointer-events: none;
    background: #0e1714;
    opacity: 0;
}

.shapgpt-curtain[data-state="close"] {
    animation: shapgpt-curtain-close var(--curtain-ms) cubic-bezier(0.4, 0, 0.6, 1) both;
}

.shapgpt-curtain[data-state="open"] {
    animation: shapgpt-curtain-open var(--curtain-ms) cubic-bezier(0.4, 0, 0.2, 1) both;
}

/* A thin line appears only when the next page takes a while. */
.shapgpt-curtain::after {
    content: "";
    position: absolute;
    top: 50%;
    left: 50%;
    width: 132px;
    height: 2px;
    margin-left: -66px;
    border-radius: 2px;
    background:
        linear-gradient(90deg, transparent, #8cc7a6, transparent) 0 0 / 40% 100% no-repeat,
        rgba(140, 199, 166, 0.16);
    opacity: 0;
    animation:
        shapgpt-curtain-wait 300ms ease 900ms both,
        shapgpt-curtain-sweep 1100ms ease-in-out 900ms infinite;
}

.shapgpt-curtain[data-state="open"]::after {
    display: none;
}

/* Until the next page has replaced the one left, what remains of the old
   page stays hidden and late elements fade in. */
html.shapgpt-nav-settle [data-testid="stMain"] [data-stale="true"],
html.shapgpt-nav-settle .stApp .st-key-brand_side_panel[data-shapgpt-left] {
    display: none !important;
}

html.shapgpt-nav-settle:not(.shapgpt-nav-to-intro) [data-testid="stMain"] [data-stale="false"],
html.shapgpt-nav-settle .stApp .st-key-brand_side_panel:not([data-shapgpt-left]) {
    animation: shapgpt-nav-fade 300ms ease-out both;
}

/* Arriving: the next page rises into place; from the start screen the
   sidebar slides in as the curtain opens. */
html.shapgpt-nav-in [data-testid="stMainBlockContainer"] {
    animation: shapgpt-nav-in 560ms cubic-bezier(0.16, 1, 0.3, 1) both;
}

html.shapgpt-nav-in.shapgpt-nav-from-intro [data-testid="stMainBlockContainer"] {
    animation: shapgpt-nav-in 720ms cubic-bezier(0.16, 1, 0.3, 1) 80ms both;
}

html.shapgpt-nav-in.shapgpt-nav-from-intro [data-testid="stSidebarContent"] {
    animation: shapgpt-rail-in 640ms cubic-bezier(0.16, 1, 0.3, 1) both;
}

@keyframes shapgpt-nav-out {
    to { opacity: 0; transform: translateY(-10px); }
}

@keyframes shapgpt-nav-in {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: none; }
}

@keyframes shapgpt-nav-fade {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes shapgpt-rail-in {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: none; }
}

@keyframes shapgpt-intro-out {
    to { opacity: 0; transform: translateY(-18px) scale(0.98); }
}

@keyframes shapgpt-wafer-out {
    to { opacity: 0; transform: scale(0.9); }
}

@keyframes shapgpt-wafer-pick {
    40% { opacity: 1; transform: translateY(-4px) scale(1.06); }
    to { opacity: 0; transform: translateY(-6px) scale(1.1); }
}

@keyframes shapgpt-curtain-close {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes shapgpt-curtain-open {
    from { opacity: 1; }
    to { opacity: 0; }
}

@keyframes shapgpt-curtain-wait {
    to { opacity: 1; }
}

@keyframes shapgpt-curtain-sweep {
    from { background-position: -60% 0, 0 0; }
    to { background-position: 160% 0, 0 0; }
}

@media (prefers-reduced-motion: reduce) {
    .shapgpt-curtain {
        display: none;
    }
}
"""

# Menu links, sidebar entries, buttons on the pages and the logo still work
# through Streamlit: a click is held for a moment while the current page fades
# out, then the same link is clicked again so Streamlit opens the page. Until
# the next page has arrived (its marker from _style is in place and the old
# elements are gone, or a moment has passed) it stays hidden; then it rises
# into place. With reduced motion pages switch at once. On phones the sidebar
# closes after a choice, because it covers the page.
NAV_JS_TEMPLATE = """
(() => {
    if (window.__shapgptNav) {
        return;
    }
    window.__shapgptNav = true;
    const root = document.documentElement;
    const calm = window.matchMedia("(prefers-reduced-motion: reduce)");
    const phone = window.matchMedia("(max-width: 768px)");
    const sheet = document.createElement("style");
    sheet.id = "shapgpt-nav";
    sheet.textContent = __NAV_CSS__;
    document.head.appendChild(sheet);

    const OUT_MS = 220;
    const OUT_INTRO_MS = 420;
    const TO_INTRO_MS = 300;
    const SETTLE_MS = 350;
    const INTRO_SETTLE_MS = 1400;
    const GIVE_UP_MS = 4000;
    const IN_MS = 820;
    const LINKS = 'a[data-testid="stPageLink-NavLink"], button[data-testid="stLogoLink"]';
    // Old elements still on screen: in the page itself (they hold the next
    // page back) and in the SECOM settings panel (hidden until it is removed).
    const PAGE_LEFTOVERS = '[data-testid="stMain"] [data-stale="true"]';
    const ALL_LEFTOVERS = PAGE_LEFTOVERS + ', .st-key-brand_side_panel [data-stale="true"]';
    let trip = null;
    let turn = 0;

    const leftovers = (selector = PAGE_LEFTOVERS) => document.querySelector(selector) !== null;
    const currentPage = () => {
        for (const marker of document.querySelectorAll(".shapgpt-page")) {
            if (!marker.closest('[data-stale="true"]')) {
                return marker.dataset.page;
            }
        }
        return null;
    };
    const curtain = (state, ms) => {
        let veil = document.querySelector(".shapgpt-curtain");
        if (!veil) {
            veil = document.createElement("div");
            veil.className = "shapgpt-curtain";
            veil.setAttribute("aria-hidden", "true");
            document.body.appendChild(veil);
        }
        veil.style.setProperty("--curtain-ms", `${ms}ms`);
        veil.dataset.state = state;
        return veil;
    };
    const closeSidebar = () => {
        const button = document.querySelector('[data-testid="stSidebarCollapseButton"] button');
        if (button) {
            button.click();
        }
    };

    const leave = (link, toIntro) => {
        const id = ++turn;
        const from = currentPage();
        trip = {id, from, href: link.getAttribute("href"), sent: 0, arrived: 0};
        root.classList.remove("shapgpt-nav-in", "shapgpt-nav-from-intro", "shapgpt-nav-to-intro");
        root.classList.add("shapgpt-nav-used", "shapgpt-nav-out");
        for (const panel of document.querySelectorAll(".st-key-brand_side_panel")) {
            panel.setAttribute("data-shapgpt-left", "");
        }
        let delay = OUT_MS;
        if (from === "intro") {
            root.classList.add("shapgpt-nav-from-intro");
            const wafer = link.closest('[class*="st-key-brand_menu_"]');
            const index = wafer && wafer.className.match(/st-key-brand_menu_(\\d)/);
            if (index) {
                root.dataset.shapgptPick = index[1];
            }
            curtain("close", OUT_INTRO_MS);
            delay = OUT_INTRO_MS;
        } else if (toIntro) {
            root.classList.add("shapgpt-nav-to-intro");
            curtain("close", TO_INTRO_MS);
            delay = TO_INTRO_MS;
        }
        window.setTimeout(() => go(id, link), delay);
    };

    const go = (id, link) => {
        if (!trip || trip.id !== id) {
            return;
        }
        const scroller = document.querySelector('[data-testid="stMain"]');
        if (scroller) {
            scroller.scrollTop = 0;
        }
        root.classList.add("shapgpt-nav-settle");
        trip.sent = performance.now();
        let again = link;
        if (!again.isConnected) {
            again = [...document.querySelectorAll(LINKS)].find((candidate) =>
                candidate.tagName === link.tagName &&
                candidate.getAttribute("href") === trip.href
            );
        }
        if (again) {
            again.dataset.shapgptGo = "1";
            again.click();
            delete again.dataset.shapgptGo;
        }
        window.requestAnimationFrame(() => wait(id));
    };

    const wait = (id) => {
        if (!trip || trip.id !== id) {
            return;
        }
        const now = performance.now();
        const page = currentPage();
        if (!trip.arrived && page !== null && page !== trip.from) {
            trip.arrived = now;
        }
        let ready = false;
        if (trip.arrived && page === "intro") {
            ready = document.querySelector(".st-key-brand_menu_4") !== null &&
                (!leftovers() || now - trip.arrived > INTRO_SETTLE_MS);
        } else if (trip.arrived) {
            ready = !leftovers() || now - trip.arrived > SETTLE_MS;
        }
        if (ready || now - trip.sent > GIVE_UP_MS) {
            enter(id, page);
        } else {
            window.requestAnimationFrame(() => wait(id));
        }
    };

    const enter = (id, page) => {
        const toIntro = page === "intro";
        trip = null;
        root.classList.remove("shapgpt-nav-out");
        root.classList.toggle("shapgpt-nav-to-intro", toIntro);
        if (!toIntro) {
            root.classList.add("shapgpt-nav-in");
        }
        const veil = document.querySelector(".shapgpt-curtain");
        if (veil) {
            const ms = toIntro ? 320 : 640;
            curtain("open", ms);
            window.setTimeout(() => {
                if (veil.dataset.state === "open") {
                    veil.remove();
                }
            }, ms + 40);
        }
        window.setTimeout(() => {
            if (turn === id && !trip) {
                root.classList.remove(
                    "shapgpt-nav-in", "shapgpt-nav-from-intro", "shapgpt-nav-to-intro"
                );
                delete root.dataset.shapgptPick;
            }
        }, IN_MS);
        const since = performance.now();
        const settle = () => {
            if (turn !== id || trip || !root.classList.contains("shapgpt-nav-settle")) {
                return;
            }
            if (!leftovers(ALL_LEFTOVERS) || performance.now() - since > 5000) {
                root.classList.remove("shapgpt-nav-settle");
            } else {
                window.requestAnimationFrame(settle);
            }
        };
        settle();
    };

    // Touching the page ends the settling phase, so a rerun of the page
    // itself never hides its own elements.
    const touched = () => {
        if (!trip) {
            root.classList.remove("shapgpt-nav-settle");
        }
    };
    document.addEventListener("pointerdown", touched, true);
    document.addEventListener("keydown", touched, true);

    document.addEventListener("click", (event) => {
        const link = event.target instanceof Element ? event.target.closest(LINKS) : null;
        if (!link || link.dataset.shapgptGo || event.button !== 0) {
            return;
        }
        if (phone.matches && link.closest('[data-testid="stSidebar"]')) {
            window.setTimeout(closeSidebar, 0);
        }
        const toIntro = link.tagName === "BUTTON";
        if (!toIntro) {
            const url = new URL(link.href, window.location.href);
            if (link.target === "_blank" || url.origin !== window.location.origin ||
                url.pathname === window.location.pathname) {
                return;
            }
        }
        if (calm.matches) {
            return;
        }
        event.preventDefault();
        event.stopImmediatePropagation();
        if (!trip) {
            leave(link, toIntro);
        }
    }, true);
})();
"""
NAV_JS = NAV_JS_TEMPLATE.replace("__NAV_CSS__", json.dumps(NAV_CSS))


# The animation waits at its first frame until the four menus exist and the
# page just left has been removed (a heavy page can block the browser for a
# moment), so it always plays in step; it never waits over 1.5 s. Skip ends
# the animation at once while Streamlit reruns.
INTRO_JS = """
(() => {
    const root = document.documentElement;
    root.classList.remove("shapgpt-intro-instant");
    root.classList.add("shapgpt-intro-wait");
    const since = performance.now();
    const ready = () =>
        document.querySelector(".st-key-brand_menu_4") &&
        !document.querySelector('[data-testid="stMain"] [data-stale="true"]');
    const hold = () => {
        if (ready() || performance.now() - since > 1500) {
            window.requestAnimationFrame(() => root.classList.remove("shapgpt-intro-wait"));
        } else {
            window.requestAnimationFrame(hold);
        }
    };
    window.requestAnimationFrame(hold);
    if (!window.__shapgptSkip) {
        window.__shapgptSkip = true;
        document.addEventListener("click", (event) => {
            if (event.target.closest(".st-key-brand_skip")) {
                root.classList.add("shapgpt-intro-instant");
            }
        }, true);
    }
})();
"""


def intro_logo_html(logo_uri: str, hash_uri: str, mode: str) -> str:
    return (
        f'<div class="shapgpt-intro" data-mode="{mode}" '
        f"style=\"--shapgpt-hash-mask: url('{hash_uri}')\">"
        '<div class="shapgpt-lift">'
        '<div class="shapgpt-logo" role="img" aria-label="SHAPGPT">'
        f'<img class="shapgpt-logo__base" src="{logo_uri}" alt="" draggable="false">'
        f'<img class="shapgpt-logo__hash" src="{hash_uri}" alt="" draggable="false">'
        '<span class="shapgpt-logo__sweep"></span>'
        "</div></div>"
        '<div class="shapgpt-copy">'
        f'<p class="shapgpt-copy__kicker">{INTRO_KICKER}</p>'
        f'<p class="shapgpt-copy__summary">{INTRO_SUMMARY}</p>'
        "</div></div>"
    )


def intro_mode() -> str:
    """Return "seen" for a rerun of the start screen itself (such as Skip), else "play"."""
    return "seen" if st.session_state.get(LAST_PAGE_KEY) == "intro" else "play"


def render_intro(menu: Sequence[MenuItem]) -> None:
    """Start screen: the big logo, one "#" highlight, the English introduction
    and the four wafer menus.

    The animation plays every time the screen opens: first visit, reload, or
    the logo on a page. A rerun of the screen itself (Skip) shows it finished.
    """
    mode = intro_mode()
    logo_uri, hash_uri = logo_layers(str(LOGO_PATH), LOGO_PATH.stat().st_mtime_ns)
    _style(INTRO_CSS + (INTRO_PLAY_CSS if mode == "play" else ""), page="intro")
    st.html(
        intro_logo_html(logo_uri, hash_uri, mode)
        + f"<script>{INTRO_JS}</script><script>{NAV_JS}</script>",
        unsafe_allow_javascript=True,
    )
    with st.container(key="brand_menu"):
        for index, item in enumerate(menu, start=1):
            with st.container(key=f"brand_menu_{index}"):
                st.page_link(item.page, label=item.label)
    if mode == "play":
        st.button("Skip", key="brand_skip", type="tertiary")
    st.session_state[LAST_PAGE_KEY] = "intro"


def render_shell(
    menu: Sequence[MenuItem], *, current: str, extra: Sequence[MenuItem] = ()
) -> None:
    """Dark sidebar navigation for the white content pages.

    The logo (``st.logo``) needs no link of its own: on any page but the
    default one Streamlit makes it a button that opens the default page, which
    is the start screen. Call it before anything else of the page, so the
    styles arrive first.
    """
    active = f'.stApp .st-key-brand_nav_{current} a[data-testid="stPageLink-NavLink"]'
    first_page = LAST_PAGE_KEY not in st.session_state
    _style(
        SHELL_CSS
        + PAGE_CSS
        + f"{active}, {active}:hover {{"
        "background: rgba(140, 199, 166, 0.15);"
        "color: var(--shell-strong);"
        "box-shadow: inset 3px 0 0 var(--shell-mint);}"
        + f'{active} [data-testid="stIconMaterial"] {{color: var(--shell-mint) !important;}}'
        + (PAGE_ENTRANCE_CSS.replace("SLUG", current) if first_page else ""),
        page=current,
    )
    with st.sidebar:
        st.html(
            f'<p class="brand-nav__label">MENU</p><script>{NAV_JS}</script>',
            unsafe_allow_javascript=True,
        )
        with st.container(key="brand_side_nav"):
            for item in menu:
                _side_link(item)
            if extra:
                st.html('<hr class="brand-nav__divider">')
                for item in extra:
                    _side_link(item)
    st.session_state[LAST_PAGE_KEY] = current


def _style(css: str, *, page: str) -> None:
    """Send CSS as markdown, like the app's other styles: it is applied in the
    same render as the elements it styles (style-only ``st.html`` goes to a
    separate container and can land a few frames later). The empty marker
    tells NAV_JS which page has arrived."""
    st.markdown(
        f'<style>{css}</style><span class="shapgpt-page" data-page="{page}"></span>',
        unsafe_allow_html=True,
    )


def _side_link(item: MenuItem) -> None:
    with st.container(key=f"brand_nav_{item.slug}"):
        st.page_link(item.page, label=item.label, icon=item.icon, width="stretch")
