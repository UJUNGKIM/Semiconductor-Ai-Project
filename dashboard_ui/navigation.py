"""Navigation and URL state helpers shared by dashboard modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import streamlit as st


def query_value(name: str, default: str = "") -> str:
    """Return one query parameter as a plain string."""
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return str(value[-1]) if value else default
    return str(value)


def set_query_state(*, module: str, section: str | None = None, row: int | None = None) -> None:
    """Persist the current dashboard location as a shareable deep link."""
    st.query_params["module"] = module
    if section:
        st.query_params["section"] = section
    elif "section" in st.query_params:
        del st.query_params["section"]
    if row is not None:
        st.query_params["row"] = str(row)
    elif section != "explanation" and "row" in st.query_params:
        del st.query_params["row"]


def render_section_navigation(
    groups: Mapping[str, Sequence[tuple[str, str]]],
    *,
    module: str,
    default_section: str,
    key_prefix: str,
) -> str:
    """Render compact two-level navigation and return the selected section slug."""
    label_by_slug = {
        slug: label for items in groups.values() for label, slug in items
    }
    requested = query_value("section", default_section)
    if requested not in label_by_slug:
        requested = default_section

    group_for_slug = {
        slug: group for group, items in groups.items() for _, slug in items
    }
    group_key = f"{key_prefix}_group"
    requested_group = group_for_slug[requested]
    sync_key = f"{key_prefix}_query_section"
    # Widget keys may only be written here, before the widgets below exist in
    # this run. A section requested through the URL (a link or
    # navigate_to_section) is applied now; so is the section shown before
    # leaving the page, because Streamlit drops the keys of unrendered widgets.
    if st.session_state.get(sync_key) != requested or group_key not in st.session_state:
        st.session_state[group_key] = requested_group
        requested_section_key = f"{key_prefix}_{requested_group}"
        st.session_state[requested_section_key] = label_by_slug[requested]
        st.session_state[sync_key] = requested

    group_labels = list(groups)
    selected_group = st.segmented_control(
        "업무 영역",
        group_labels,
        key=group_key,
        label_visibility="collapsed",
        width="stretch",
    ) or requested_group

    items = list(groups[selected_group])
    option_labels = [label for label, _ in items]
    slug_by_label = {label: slug for label, slug in items}
    section_key = f"{key_prefix}_{selected_group}"
    requested_label = label_by_slug.get(requested)
    if section_key not in st.session_state:
        st.session_state[section_key] = (
            requested_label if requested_label in option_labels else option_labels[0]
        )
    selected_label = st.segmented_control(
        "세부 화면",
        option_labels,
        key=section_key,
        label_visibility="collapsed",
        width="stretch",
    ) or option_labels[0]
    selected_slug = slug_by_label[selected_label]
    set_query_state(module=module, section=selected_slug)
    st.session_state[sync_key] = selected_slug
    return selected_slug


def render_flat_navigation(
    items: Sequence[tuple[str, str]], *, key: str, label: str
) -> str:
    """One row of lazily rendered areas, kept in sync with ``?section=``."""
    label_by_slug = {slug: text for text, slug in items}
    slug_by_label = {text: slug for text, slug in items}
    requested = query_value("section", items[0][1])
    if requested not in label_by_slug:
        requested = items[0][1]
    sync_key = f"{key}_query_section"
    # As in render_section_navigation, the widget key is only written before
    # the widget is created in this run.
    if st.session_state.get(sync_key) != requested or key not in st.session_state:
        st.session_state[key] = label_by_slug[requested]
    selected_label = st.segmented_control(
        label,
        list(slug_by_label),
        key=key,
        label_visibility="collapsed",
        width="stretch",
    ) or label_by_slug[requested]
    selected = slug_by_label[selected_label]
    st.query_params["section"] = selected
    st.session_state[sync_key] = selected
    return selected


def navigate_to_section(*, module: str, section: str, row: int | None = None) -> None:
    """Ask the next run to open another section of the current page.

    Streamlit rejects writes to a widget's key once that widget exists in the
    running script, so this only records the destination in the URL.
    render_section_navigation applies it at the start of the next run, before
    it creates the section widgets. Use it from a button's on_click callback,
    which Streamlit runs right before that rerun.
    """
    set_query_state(module=module, section=section, row=row)
