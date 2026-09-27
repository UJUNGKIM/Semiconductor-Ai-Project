"""Representative-sample gallery for the WM-811K single-wafer diagnosis.

Every card is built from the stored demo manifest alone: the saved
prediction, calibrated score and review flag were fixed when the samples were
selected. Rendering the gallery therefore never runs the CNN, Gradient SHAP
or OOD scoring; only the one sample the user picks is analysed below it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html import escape

import streamlit as st


CASE_LABELS = {
    "high_confidence_correct": "고신뢰 정답",
    "typical_correct": "전형적 정답",
    "boundary_correct": "경계 근처 정답",
    "low_confidence_correct": "검토 대상 정답",
    "representative_error": "대표 오분류",
}
CARDS_PER_ROW = 4
PICK_BUTTON_LABEL = "이 샘플 분석"


@dataclass(frozen=True)
class SampleCard:
    sample_id: str
    true_class: str
    predicted_class: str
    case: str
    case_label: str
    calibrated_confidence: float
    review_required: bool
    npy_file: str


def class_order(manifest: Mapping[str, object]) -> list[str]:
    """True classes in the order the manifest lists them."""
    return list(dict.fromkeys(str(item["true_class"]) for item in manifest["samples"]))


def cards_for_class(manifest: Mapping[str, object], true_class: str) -> list[SampleCard]:
    """All stored samples of one class, ordered by the manifest's case order."""
    case_order = list(manifest.get("case_display_order") or CASE_LABELS)
    rank = {case: index for index, case in enumerate(case_order)}
    items = [item for item in manifest["samples"] if item["true_class"] == true_class]
    items.sort(key=lambda item: rank.get(item["selection_case"], len(rank)))
    return [
        SampleCard(
            sample_id=str(item["sample_id"]),
            true_class=str(item["true_class"]),
            predicted_class=str(item["predicted_class"]),
            case=str(item["selection_case"]),
            case_label=CASE_LABELS.get(item["selection_case"], str(item["selection_case"])),
            calibrated_confidence=float(item["calibrated_confidence"]),
            review_required=bool(item["review_required"]),
            npy_file=str(item["npy_file"]),
        )
        for item in items
    ]


def gallery_rows(
    cards: Sequence[SampleCard], per_row: int = CARDS_PER_ROW
) -> list[list[SampleCard]]:
    """Split cards into rows of at most ``per_row``; extra cards start a new row."""
    if per_row < 1:
        raise ValueError("한 줄의 카드 수는 1 이상이어야 합니다.")
    return [list(cards[start : start + per_row]) for start in range(0, len(cards), per_row)]


def resolve_selection(cards: Sequence[SampleCard], requested: object) -> str | None:
    """Keep the requested sample if it is in view, otherwise pick the first card."""
    sample_ids = [card.sample_id for card in cards]
    if requested in sample_ids:
        return str(requested)
    return sample_ids[0] if sample_ids else None


def _select_sample(state_key: str, sample_id: str) -> None:
    st.session_state[state_key] = sample_id


def card_html(
    card: SampleCard, *, selected: bool, score_formatter: Callable[[float], str]
) -> str:
    badges = f'<span class="wm-badge">{escape(card.case_label)}</span>'
    if selected:
        badges += '<span class="wm-badge wm-badge--selected">선택됨</span>'
    review = (
        '<dd class="wm-card__review">예 · 전문가 검토</dd>'
        if card.review_required
        else "<dd>아니오 · 자동 분류 가능</dd>"
    )
    return (
        '<div class="wm-card">'
        f'<div class="wm-card__badges">{badges}</div>'
        '<dl class="wm-card__facts">'
        f"<div><dt>실제 클래스</dt><dd>{escape(card.true_class)}</dd></div>"
        f"<div><dt>저장 예측</dt><dd>{escape(card.predicted_class)}</dd></div>"
        "<div><dt>보정 모델 점수</dt>"
        f"<dd>{escape(score_formatter(card.calibrated_confidence))}</dd></div>"
        f"<div><dt>검토 필요 여부</dt>{review}</div>"
        "</dl></div>"
    )


def render_sample_gallery(
    cards: Sequence[SampleCard],
    *,
    state_key: str,
    thumbnail: Callable[[SampleCard], bytes],
    score_formatter: Callable[[float], str],
) -> str | None:
    """Show every card of the class at once and return the selected sample id."""
    selected = resolve_selection(cards, st.session_state.get(state_key))
    st.session_state[state_key] = selected
    with st.container(key="wm_gallery"):
        for row in gallery_rows(cards):
            columns = st.columns(CARDS_PER_ROW, gap="small")
            for column, card in zip(columns, row):
                is_selected = card.sample_id == selected
                state = "selected" if is_selected else "item"
                with column, st.container(border=True, key=f"wm_card_{state}_{card.sample_id}"):
                    st.image(thumbnail(card), output_format="PNG", width="content")
                    st.markdown(
                        card_html(card, selected=is_selected, score_formatter=score_formatter),
                        unsafe_allow_html=True,
                    )
                    st.button(
                        PICK_BUTTON_LABEL,
                        key=f"wm_pick_{card.sample_id}",
                        type="primary" if is_selected else "secondary",
                        on_click=_select_sample,
                        args=(state_key, card.sample_id),
                        width="stretch",
                    )
    return selected
