"""Technician-facing chat for the maintenance manual assistant.

    streamlit run ui/app.py            (API_URL defaults to http://localhost:8000)

Layout only: every decision (what to send for a message, clarifications, the remembered aircraft,
how to show each status) lives in presenter.py, which is tested offline. This page calls the
FastAPI service over HTTP and nothing else: no secrets, no Azure OpenAI, no OpenSearch.
"""

import os

import streamlit as st

from presenter import (
    AIRCRAFT_CHOICES,
    AUTO_DETECT,
    CONVERSATION_NOTE,
    EXAMPLES,
    MAX_QUESTION_CHARS,
    NOTICE,
    UI_VERSION,
    Conversation,
    View,
    build_view,
    call_api,
    check_health,
    plan_clarification,
    plan_example,
    plan_message,
    record_exchange,
    record_notice,
    ui_config,
    with_aircraft,
)
from theme import APP_NAME, footer_html, header_html, page_css

CONFIG = ui_config(os.environ)
AVATARS = {"user": ":material/engineering:", "assistant": ":material/menu_book:"}

st.set_page_config(
    page_title=APP_NAME,
    page_icon=":material/flight:",
    layout="centered",
    menu_items={"About": f"{APP_NAME}. {NOTICE}"},
)
st.html(page_css())  # USWDS-aligned styling (theme.py); the colours also live in .streamlit/
st.html(header_html())


@st.cache_data(ttl=15, show_spinner=False)
def cached_health():
    return check_health(CONFIG)


# ------------------------------------------------------------------ state and callbacks

state = st.session_state
state.setdefault("conversation", Conversation())
state.setdefault("queued", None)  # an AskRequest waiting to be sent


def queue(plan, shown: str = "") -> None:
    if plan is None:
        return
    if plan.request is not None:
        state.queued = plan.request
    else:  # a local reply, e.g. "Aircraft set to B-340"
        state.conversation = record_notice(state.conversation, shown, plan)


def on_chat_submit() -> None:
    text = state.chat_input or ""
    queue(plan_message(state.conversation, text), shown=text)


def on_example() -> None:
    index = state.example_pill
    state.example_pill = None
    if index is not None:
        queue(plan_example(EXAMPLES[index]))


def on_clarify(option: str) -> None:
    queue(plan_clarification(state.conversation, option))


def on_aircraft_change() -> None:
    state.conversation = with_aircraft(state.conversation, state.aircraft_select)


def new_conversation() -> None:
    state.conversation, state.queued = Conversation(), None


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    health = cached_health()
    icon = {"ok": ":material/check_circle:", "degraded": ":material/warning:",
            "down": ":material/error:"}[health.level]  # fmt: skip
    show = {"ok": st.success, "degraded": st.warning, "down": st.error}[health.level]
    show(f"**{health.label}**  \n{health.detail}", icon=icon)
    if st.button("Check again", icon=":material/refresh:", width="stretch"):
        cached_health.clear()
        st.rerun()

    st.divider()
    state.aircraft_select = state.conversation.aircraft  # keep the selector in step with the chat
    st.selectbox(
        "Aircraft for this conversation", AIRCRAFT_CHOICES, key="aircraft_select",
        on_change=on_aircraft_change,
        help="Auto-detect reads the aircraft from each question and asks when it is unclear.",
    )  # fmt: skip
    st.button("New conversation", icon=":material/add_comment:", width="stretch",
              on_click=new_conversation)  # fmt: skip
    st.info(CONVERSATION_NOTE, icon=":material/info:")

    st.divider()
    st.caption(
        "Only the latest revision of each task is searched. Every torque value and part number "
        "is checked against the cited task, and procedures are returned complete, word for word."
    )
    st.caption(f"{NOTICE} UI version {UI_VERSION}.")

# ------------------------------------------------------------------ chat

conversation: Conversation = state.conversation
aircraft = conversation.aircraft
st.badge(aircraft if aircraft != AUTO_DETECT else "Aircraft: auto-detect",
         icon=":material/flight:", color="blue" if aircraft != AUTO_DETECT else "gray")  # fmt: skip


def render_alerts(alerts) -> None:
    for alert in alerts:
        if alert.kind == "WARNING":
            st.error(alert.markdown, icon=":material/warning:")
        else:
            st.warning(alert.markdown, icon=":material/error_outline:")


def render(view: View, index: int, is_last: bool) -> None:
    if view.kind == "notice":
        st.markdown(view.message)
        return
    if view.kind == "answered":
        st.markdown(view.message)
        for group in view.alert_groups:  # short answers: the alerts of the steps relied on
            st.markdown(f"**{group.heading}**")
            render_alerts(group.alerts)
        if view.steps:
            st.markdown("**Procedure**")
            for step in view.steps:
                with st.container(border=True):
                    render_alerts(step.alerts)  # read before doing the step
                    st.markdown(step.markdown)
        if view.citations:
            st.markdown("**Sources**")
            for citation in view.citations:
                st.markdown(citation.markdown)
    elif view.kind == "clarification":
        st.info(f"**{view.title}**  \n{view.message}", icon=":material/help:")
        if is_last:
            columns = st.columns(max(len(view.options), 1))
            for column, option in zip(columns, view.options, strict=False):
                column.button(
                    option, key=f"clarify-{index}-{option}", on_click=on_clarify,
                    args=(option,), icon=":material/flight:", width="stretch", type="primary",
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
        st.caption(f"Request ID: `{view.request_id}`")


for index, turn in enumerate(conversation.turns):
    with st.chat_message(turn.role, avatar=AVATARS[turn.role]):
        if turn.role == "user":
            st.text(turn.text)  # the user's words, shown exactly as typed
        else:
            render(turn.view, index, is_last=index == len(conversation.turns) - 1)

if not conversation.turns and state.queued is None:
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown("Ask about torque values, part numbers or procedures for the A-200, B-340 "
                    "and C-75. Try one of these:")  # fmt: skip
        st.pills(
            "Examples", range(len(EXAMPLES)), key="example_pill", on_change=on_example,
            format_func=lambda i: f"{EXAMPLES[i].label}: {EXAMPLES[i].question}",
            label_visibility="collapsed",
        )  # fmt: skip

if state.queued is not None:
    request = state.queued
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.text(request.shown)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Searching the manuals and checking the answer..."):
            view = build_view(call_api(CONFIG, request.question, request.aircraft))
    state.conversation = record_exchange(state.conversation, request, view)
    state.queued = None
    if view.kind == "error":
        cached_health.clear()  # the status panel should reflect what just happened
    st.rerun()

st.chat_input("Ask a maintenance question", key="chat_input", max_chars=MAX_QUESTION_CHARS,
              on_submit=on_chat_submit)  # fmt: skip
st.html(footer_html())
