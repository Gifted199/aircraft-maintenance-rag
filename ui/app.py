"""Technician-facing web page for the maintenance manual assistant.

    streamlit run ui/app.py            (API_URL defaults to http://localhost:8000)

Layout only: every decision (what to send, how to read the reply, what to show for each status)
lives in presenter.py, which is tested offline. This page calls the FastAPI service over HTTP and
nothing else: no secrets, no Azure OpenAI, no OpenSearch.
"""

import os

import streamlit as st

from presenter import (
    AIRCRAFT_CHOICES,
    AUTO_DETECT,
    EXAMPLES,
    MAX_QUESTION_CHARS,
    NOTICE,
    View,
    build_view,
    call_api,
    check_health,
    ui_config,
)

CONFIG = ui_config(os.environ)

st.set_page_config(
    page_title="Maintenance manual assistant",
    page_icon=":material/build:",
    layout="centered",
    menu_items={"About": f"Aircraft maintenance manual assistant. {NOTICE}"},
)


@st.cache_data(ttl=15, show_spinner=False)
def cached_health():
    return check_health(CONFIG)


# ------------------------------------------------------------------ state and callbacks

state = st.session_state
state.setdefault("question", "")
state.setdefault("aircraft", AUTO_DETECT)
state.setdefault("pending", False)
state.setdefault("view", None)


def use_example(index: int) -> None:
    example = EXAMPLES[index]
    state.question, state.aircraft, state.pending = example.question, example.aircraft, True


def choose_aircraft(aircraft: str) -> None:
    state.aircraft, state.pending = aircraft, True


def submit() -> None:
    state.pending = True


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.header("Service status")
    health = cached_health()
    icon = {"ok": ":material/check_circle:", "degraded": ":material/warning:",
            "down": ":material/error:"}[health.level]  # fmt: skip
    show = {"ok": st.success, "degraded": st.warning, "down": st.error}[health.level]
    show(f"**{health.label}**  \n{health.detail}", icon=icon)
    if st.button("Check again", icon=":material/refresh:", width="stretch"):
        cached_health.clear()
        st.rerun()

    st.divider()
    st.subheader("How answers are checked")
    st.markdown(
        "- Only the **latest revision** of each task is searched.\n"
        "- Every torque value and part number is checked against the cited task.\n"
        "- Procedures are returned complete and word for word, with WARNINGs and CAUTIONs.\n"
        "- If an answer cannot be verified, the assistant says it does not know."
    )
    st.divider()
    st.caption(NOTICE)

# ------------------------------------------------------------------ question

st.title("Maintenance manual assistant")
st.caption("Ask about torque values, part numbers or procedures for the A-200, B-340 and C-75.")

with st.form("ask", border=False):
    st.selectbox(
        "Aircraft", AIRCRAFT_CHOICES, key="aircraft",
        help="Auto-detect reads the aircraft type from your question, and asks if it is unclear.",
    )  # fmt: skip
    st.text_area(
        "Question", key="question", max_chars=MAX_QUESTION_CHARS, height=100,
        placeholder="e.g. What is the final torque for the main wheel axle nut?",
    )  # fmt: skip
    st.form_submit_button("Ask", type="primary", icon=":material/search:", on_click=submit)

st.caption("Examples")
columns = st.columns(2)
for index, example in enumerate(EXAMPLES):
    columns[index % 2].button(
        f"{example.label}: {example.question}", key=f"example-{index}",
        on_click=use_example, args=(index,), width="stretch",
    )  # fmt: skip

# ------------------------------------------------------------------ ask the API

if state.pending:
    state.pending = False
    if not state.question.strip():
        st.warning("Type a question first.", icon=":material/edit:")
    else:
        with st.spinner("Searching the manuals and checking the answer..."):
            state.view = build_view(call_api(CONFIG, state.question, state.aircraft))
        if state.view.kind == "error":
            cached_health.clear()  # the status panel should reflect what just happened


# ------------------------------------------------------------------ show the result


def render(view: View) -> None:
    st.divider()
    if view.kind == "answered":
        st.subheader(view.title)
        st.markdown(view.message)
        for group in view.alert_groups:  # short answers: the alerts of the steps relied on
            st.markdown(f"**{group.heading}**")
            render_alerts(group.alerts)
        if view.steps:
            st.subheader("Procedure")
            for step in view.steps:
                with st.container(border=True):
                    render_alerts(step.alerts)  # read before doing the step
                    st.markdown(step.markdown)
        if view.citations:
            st.subheader("Sources")
            for c in view.citations:
                with st.container(border=True):
                    st.markdown(f"**{c.heading}**  \n{c.detail}")
                    st.caption("Source document")
                    st.code(c.source_key, language=None)
    elif view.kind == "clarification":
        st.info(f"**{view.title}**  \n{view.message}", icon=":material/help:")
        columns = st.columns(max(len(view.options), 1))
        for column, option in zip(columns, view.options, strict=False):
            column.button(
                f"Ask for the {option}", key=f"clarify-{option}", on_click=choose_aircraft,
                args=(option,), width="stretch",
            )  # fmt: skip
    elif view.kind == "abstained":
        st.info(f"**{view.title}**  \n{view.message}", icon=":material/block:")
    elif view.kind == "rejected":
        st.warning(f"**{view.title}**  \n{view.message}", icon=":material/do_not_disturb_on:")
    else:
        st.error(f"**{view.title}**  \n{view.message}", icon=":material/error:")

    if view.details:
        with st.expander("Retrieval details"):
            for label, value in view.details:
                st.markdown(f"{label}: `{value}`")
    if view.request_id:
        st.caption(f"Request ID: `{view.request_id}` (quote this when reporting a problem)")


def render_alerts(alerts) -> None:
    for alert in alerts:
        if alert.kind == "WARNING":
            st.error(alert.markdown, icon=":material/warning:")
        else:
            st.warning(alert.markdown, icon=":material/error_outline:")


if state.view is not None:
    render(state.view)
