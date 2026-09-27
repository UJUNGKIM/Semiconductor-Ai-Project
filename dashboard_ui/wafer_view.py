"""Display rendering for WM-811K wafer maps and their explanation maps.

Wafer maps hold category codes, not intensities: 0 is outside the wafer, 1 a
normal die and 2 a defective die. Enlarging them for the screen only repeats
each cell as a solid block (nearest neighbour), so no in-between values or
colours can appear and no detail is invented. Images reach Streamlit as
lossless PNG: ``st.image`` would otherwise encode RGB arrays as JPEG and
resample anything wider than 1460 px with a bilinear filter.

Continuous explanation maps (Gradient SHAP, Grad-CAM) use a separate
sequential colour scale with its own legend. They are drawn with the same
die-sized blocks because each value belongs to one die cell.
"""

from __future__ import annotations

from html import escape
from io import BytesIO
import math

import numpy as np
import streamlit as st
from PIL import Image


CATEGORY_CODES = (0, 1, 2)
CATEGORY_LABELS = {0: "웨이퍼 밖", 1: "정상 die", 2: "결함 die"}
CATEGORY_COLORS = {
    0: (217, 223, 230),
    1: (62, 158, 146),
    2: (14, 42, 71),
}
EXPLANATION_STOPS = (
    (0.0, (243, 247, 249)),
    (0.5, (91, 181, 169)),
    (1.0, (11, 37, 69)),
)
DETAIL_MIN_SIDE = 512
THUMBNAIL_MIN_SIDE = 256
# Streamlit shrinks images wider than 1460 px with a bilinear filter.
MAX_IMAGE_SIDE = 1456


def category_codes(wafer: np.ndarray) -> np.ndarray:
    """Return the map as uint8 codes, refusing anything outside 0, 1, 2."""
    array = np.asarray(wafer)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"웨이퍼 맵은 비어 있지 않은 2차원 배열이어야 합니다: {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("웨이퍼 맵 표시는 정수 범주 값만 받습니다.")
    unexpected = sorted(set(np.unique(array).tolist()) - set(CATEGORY_CODES))
    if unexpected:
        raise ValueError(f"웨이퍼 맵에 0, 1, 2 이외의 값이 있습니다: {unexpected}")
    return array.astype(np.uint8, copy=False)


def upscale_nearest(array: np.ndarray, factor: int) -> np.ndarray:
    """Repeat every cell as a ``factor`` x ``factor`` block along both axes."""
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("확대 배율은 1 이상의 정수여야 합니다.")
    return np.repeat(np.repeat(array, int(factor), axis=0), int(factor), axis=1)


def fit_for_display(
    array: np.ndarray, *, min_side: int, max_side: int = MAX_IMAGE_SIDE
) -> tuple[np.ndarray, dict[str, int]]:
    """Scale a cell grid by an integer factor for display, never by resampling.

    The factor makes the shorter side reach ``min_side`` while the longer side
    stays within ``max_side``. A grid already longer than ``max_side`` is
    thinned by keeping every n-th cell, which still never mixes cells.
    """
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        step = math.ceil(longest / max_side)
        return array[::step, ::step], {"factor": 1, "step": step}
    factor = max(1, math.ceil(min_side / min(height, width)))
    factor = max(1, min(factor, max_side // longest))
    return upscale_nearest(array, factor), {"factor": factor, "step": 1}


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def categorical_png(
    wafer: np.ndarray, *, min_side: int = DETAIL_MIN_SIDE
) -> tuple[bytes, dict[str, int]]:
    """Encode a wafer map as a three-colour palette PNG enlarged by blocks."""
    display, scale = fit_for_display(category_codes(wafer), min_side=min_side)
    height, width = display.shape
    image = Image.frombytes("P", (width, height), np.ascontiguousarray(display).tobytes())
    image.putpalette([channel for code in CATEGORY_CODES for channel in CATEGORY_COLORS[code]])
    return _png_bytes(image), {**scale, "width": width, "height": height}


def sequential_rgb(values: np.ndarray) -> np.ndarray:
    """Colour values in [0, 1] with the explanation scale."""
    positions = [position for position, _ in EXPLANATION_STOPS]
    colors = np.array([color for _, color in EXPLANATION_STOPS], dtype=np.float64)
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    channels = [np.interp(clipped, positions, colors[:, index]) for index in range(3)]
    return np.rint(np.stack(channels, axis=-1)).astype(np.uint8)


def explanation_png(
    values: np.ndarray, wafer: np.ndarray, *, min_side: int = DETAIL_MIN_SIDE
) -> tuple[bytes, dict[str, int]]:
    """Encode a 0-1 explanation map; cells outside the wafer stay neutral."""
    codes = category_codes(wafer)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != codes.shape:
        raise ValueError(
            f"설명 지도 {array.shape}와 웨이퍼 맵 {codes.shape}의 크기가 다릅니다."
        )
    if not np.isfinite(array).all():
        raise ValueError("설명 지도에 결측치 또는 무한대가 있습니다.")
    rgb = sequential_rgb(array)
    rgb[codes == 0] = CATEGORY_COLORS[0]
    display, scale = fit_for_display(rgb, min_side=min_side)
    height, width = display.shape[:2]
    image = Image.fromarray(np.ascontiguousarray(display))
    return _png_bytes(image), {**scale, "width": width, "height": height}


def show_png(png: bytes, *, caption: str | None = None) -> None:
    """Show a prepared PNG at its own pixel size without re-encoding it."""
    st.image(png, caption=caption, output_format="PNG", width="content")


def _swatch(color: tuple[int, int, int]) -> str:
    red, green, blue = color
    return (
        '<span class="wafer-legend__swatch" '
        f'style="background: rgb({red}, {green}, {blue})"></span>'
    )


def category_legend_html() -> str:
    items = "".join(
        f'<span class="wafer-legend__item">{_swatch(CATEGORY_COLORS[code])}'
        f"{code} = {escape(CATEGORY_LABELS[code])}</span>"
        for code in CATEGORY_CODES
    )
    return f'<div class="wafer-legend" aria-label="웨이퍼 맵 범례">{items}</div>'


def explanation_scale_html(*, low_label: str, high_label: str) -> str:
    stops = ", ".join(
        f"rgb({color[0]}, {color[1]}, {color[2]}) {position * 100:.0f}%"
        for position, color in EXPLANATION_STOPS
    )
    return (
        '<div class="wafer-legend" aria-label="설명 지도 색 척도">'
        f'<span class="wafer-legend__item">{escape(low_label)}</span>'
        '<span class="explanation-scale" '
        f'style="background: linear-gradient(90deg, {stops})"></span>'
        f'<span class="wafer-legend__item">{escape(high_label)}</span>'
        f'<span class="wafer-legend__item">{_swatch(CATEGORY_COLORS[0])}웨이퍼 밖</span>'
        "</div>"
    )


def scale_note(shape: tuple[int, ...], scale: dict[str, int]) -> str:
    """Describe the display scaling without implying extra resolution."""
    height, width = shape[:2]
    if scale["step"] > 1:
        return (
            f"{height}×{width} 격자가 화면 한도를 넘어 {scale['step']}칸마다 한 칸만 "
            "표시했습니다. 값 섞기 없이 줄였지만 일부 칸은 화면에서 생략됩니다."
        )
    factor = scale["factor"]
    return (
        f"{height}×{width} 격자의 각 칸을 {factor}×{factor} 픽셀 블록으로 복제한 "
        "정수 배율 확대입니다. 보간이나 초해상도를 쓰지 않았으므로 원본보다 정보가 "
        "늘어나지 않습니다."
    )
