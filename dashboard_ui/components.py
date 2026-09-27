"""Reusable visual components for the Streamlit dashboards."""

from __future__ import annotations

from html import escape

import streamlit as st


def render_dashboard_header(title: str, description: str) -> None:
    """Render one restrained page heading without decorative copy."""
    with st.container(key="dashboard_hero"):
        st.title(title)
        st.caption(description)


def render_run_context(
    *, source_name: str, rows: int, profile_label: str, ran_at: str
) -> None:
    """Show run metadata as a compact line instead of three more cards."""
    st.markdown(
        f"""
        <div class="run-context" aria-label="현재 진단 실행 정보">
            <span><b>입력</b> {escape(source_name)}</span>
            <span><b>행</b> {rows:,}개</span>
            <span><b>기준</b> {escape(profile_label)}</span>
            <span><b>실행</b> {escape(ran_at)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
