"""Streamlit UI layer for the healthcare assistant.

This module owns rendering and Streamlit session-state orchestration. Business
logic stays in service modules such as ``agent_graph`` and ``appointment_service``.
"""

import os
import re
from contextlib import nullcontext
from datetime import date, datetime, timedelta

import streamlit as st

from src.agent_formatting import format_entity_details
from src.agent_graph import build_runtime, find_entity_matches, run_agent
from src.appointment_service import (
    available_specialties,
    book_specific_appointment,
    find_doctors_by_specialty,
    load_doctors,
)
from src.chat_utils import extract_problem, is_end_session_request, is_no, is_yes
from src.config import CHAT_HISTORY_LOG_FILE, DISCLAIMER
from src.context_types import CONTEXT_DOCTOR, CONTEXT_GENERIC, CONTEXT_PATIENT, CONTEXT_SYSTEM
from src.logger import append_jsonl, read_jsonl
from src.llm_service import (
    classify_query_context,
    classify_specialty,
    has_openai_key,
    suggest_next_best_actions,
)
from src.record_service import (
    add_patient_history_entry,
    add_user_medical_record,
)
from src.ui_support import (
    extract_uploaded_pdf_text,
    patient_by_id,
    patient_by_name,
    refresh_runtime_data,
    unique_patient_names,
)
from src.ui_tabs import (
    render_agent_trace_tab,
    render_appointments_tab,
    render_evaluation_tab,
    render_patients_tab,
)


def configure_page():
    """Configure Streamlit before any visible UI is rendered."""
    st.set_page_config(
        page_title="Agentic Healthcare Assistant",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def apply_assistant_layout_styles():
    """Inject app-specific layout CSS in one place."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0.35rem;
        }

        [data-testid="stHeader"] {
            height: 0;
        }

        section[data-testid="stSidebar"] {
            font-size: 0.72rem;
        }

        div[data-testid="column"]:has(.chat-panel-marker),
        div[data-testid="column"]:has(.workspace-panel-marker) {
            font-size: 72%;
        }

        div[data-testid="column"]:has(.chat-panel-marker) [data-testid="stChatMessage"],
        div[data-testid="column"]:has(.chat-panel-marker) [data-testid="stMarkdownContainer"],
        div[data-testid="column"]:has(.chat-panel-marker) input,
        div[data-testid="column"]:has(.chat-panel-marker) button,
        div[data-testid="column"]:has(.workspace-panel-marker) [data-testid="stMarkdownContainer"],
        div[data-testid="column"]:has(.workspace-panel-marker) input,
        div[data-testid="column"]:has(.workspace-panel-marker) textarea,
        div[data-testid="column"]:has(.workspace-panel-marker) button,
        div[data-testid="column"]:has(.workspace-panel-marker) label {
            font-size: 0.72rem;
        }

        div[data-testid="column"]:has(.workspace-panel-marker),
        div[data-testid="column"]:has(.workspace-panel-marker) > div {
            position: sticky;
            top: 0.35rem;
            align-self: flex-start;
            max-height: calc(100vh - 0.7rem);
            overflow-y: auto;
            padding-bottom: 0.25rem;
            z-index: 5;
        }

        div[data-testid="column"]:has(.chat-panel-marker) form {
            position: sticky;
            top: 0.35rem;
            z-index: 20;
            background: var(--background-color);
            padding-top: 0.25rem;
            padding-bottom: 0.25rem;
            border-bottom: 1px solid rgba(127, 127, 127, 0.2);
        }

        div[data-testid="column"]:has(.chat-panel-marker) [data-testid="stChatMessage"] {
            margin-bottom: 0.2rem;
            padding-top: 0.25rem;
            padding-bottom: 0.25rem;
        }

        div[data-testid="column"]:has(.chat-panel-marker) [data-testid="stVerticalBlock"],
        div[data-testid="column"]:has(.workspace-panel-marker) [data-testid="stVerticalBlock"] {
            gap: 0.35rem;
        }

        div[data-testid="column"]:has(.chat-panel-marker) .stExpander,
        div[data-testid="column"]:has(.workspace-panel-marker) .stExpander {
            margin-top: 0.15rem;
            margin-bottom: 0.15rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1rem;
        }

        .element-container {
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


PATIENT_CONTEXT_PREFIX = "patient_context_select_"
BOOK_APPOINTMENT_RESULT = {"plan": [{"task": "book_appointment"}]}

WIDGET_STATE_KEYS = {
    "appointment_doctor_selector",
    "appointment_calendar_date",
    "manual_patient_selector",
    "manual_entity_selector",
    "manual_doctor_selector",
}

PATIENT_SESSION_KEYS = [
    "chat_messages",
    "current_patient",
    "current_doctor",
    "current_scope",
    "last_result",
    "guided_availability",
    "guided_specialty",
    "guided_booking",
    "pending_appointment",
    "pending_patient_selection",
    "pending_entity_selection",
    "pending_doctor_selection",
    "appointment_offer",
    "awaiting_close_confirmation",
    "close_session_requested",
    "last_announced_patient_context",
    "next_best_actions",
]


def render_llm_settings_sidebar():
    """Render model configuration in Streamlit's collapsible left sidebar."""
    with st.sidebar:
        st.markdown("### LLM Settings")
        api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model_name = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if model_name:
            os.environ["OPENAI_MODEL"] = model_name


def reset_patient_chat():
    """Clear all patient-scoped Streamlit state while preserving global app settings."""
    for key in list(st.session_state.keys()):
        if key.startswith(PATIENT_CONTEXT_PREFIX) or key in WIDGET_STATE_KEYS:
            st.session_state.pop(key, None)
    for key in PATIENT_SESSION_KEYS:
        st.session_state.pop(key, None)
    st.session_state["patient_context_select_version"] = st.session_state.get("patient_context_select_version", 0) + 1


def request_patient_chat_reset():
    """Defer reset until the next rerun so Streamlit widgets can finish cleanly."""
    st.session_state["reset_patient_chat_requested"] = True


def ensure_chat_state():
    """Initialize the chat state keys used by the assistant tab."""
    if st.session_state.pop("reset_patient_chat_requested", False):
        reset_patient_chat()
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("current_patient", None)
    st.session_state.setdefault("current_doctor", None)
    st.session_state.setdefault("current_scope", CONTEXT_GENERIC)
    st.session_state.setdefault("pending_appointment", None)
    st.session_state.setdefault("pending_patient_selection", None)
    st.session_state.setdefault("pending_doctor_selection", None)
    st.session_state.setdefault("pending_entity_selection", None)
    st.session_state.setdefault("appointment_offer", None)
    st.session_state.setdefault("awaiting_close_confirmation", False)
    st.session_state.setdefault("close_session_requested", False)
    st.session_state.setdefault("patient_context_select_version", 0)
    st.session_state.setdefault("last_announced_patient_context", None)
    st.session_state.setdefault("next_best_actions", [])


def append_chat(role, content):
    """Append one message to the visible conversation transcript."""
    st.session_state["chat_messages"].append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )


def _display_chat_timestamp(value):
    """Format stored ISO timestamps for compact chat display."""
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(value)


def _patient_context_text(patient):
    if not patient:
        return ""
    return "\n".join(
        [
            f"Name: {patient.get('Name', '')}",
            f"Age: {patient.get('Age') or 'Unknown'}",
            f"Gender: {patient.get('Gender') or 'Unknown'}",
            f"Phone: {patient.get('Phone') or 'Unknown'}",
            f"Address: {patient.get('Address') or 'Unknown'}",
            f"Summary: {patient.get('Summary') or ''}",
        ]
    )


def _conversation_text(limit=10):
    messages = st.session_state.get("chat_messages", [])[-limit:]
    lines = []
    for message in messages:
        timestamp = _display_chat_timestamp(message.get("timestamp"))
        prefix = f"{message['role']} ({timestamp})" if timestamp else message["role"]
        lines.append(f"{prefix}: {message['content']}")
    return "\n".join(lines)


def _last_result_summary(result):
    if not result:
        return ""
    tool_logs = "\n".join(
        f"{tool.get('tool')}: {tool.get('message')}" for tool in result.get("tool_logs", [])[-6:]
    )
    plan = ", ".join(step.get("task", "") for step in result.get("plan", []))
    return f"Plan: {plan}\nTools:\n{tool_logs}\nAnswer: {result.get('final_answer', '')[:1200]}"


def set_current_scope(scope, patient=None, doctor=None):
    """Update the visible/sticky conversation scope."""
    st.session_state["current_scope"] = scope
    if scope == CONTEXT_PATIENT:
        st.session_state["current_patient"] = patient or st.session_state.get("current_patient")
        st.session_state["current_doctor"] = None
    elif scope == CONTEXT_DOCTOR:
        st.session_state["current_doctor"] = doctor or st.session_state.get("current_doctor")
        st.session_state["current_patient"] = None


def active_context_decision(prompt):
    """Return a sticky patient/doctor context without rerunning the LLM router."""
    scope = st.session_state.get("current_scope", CONTEXT_GENERIC)
    if scope == CONTEXT_PATIENT and st.session_state.get("current_patient"):
        patient = st.session_state["current_patient"]
        return {
            "scope": CONTEXT_PATIENT,
            "needs_context_selection": False,
            "selected_patient_name": patient["Name"],
            "selected_doctor_name": None,
            "ambiguous_patient_names": [],
            "ambiguous_doctor_names": [],
            "contextual_query": f"For {patient['Name']}, {prompt}",
            "reason": "Patient context is locked for this chat session.",
        }
    if scope == CONTEXT_DOCTOR and st.session_state.get("current_doctor"):
        doctor = st.session_state["current_doctor"]
        return {
            "scope": CONTEXT_DOCTOR,
            "needs_context_selection": False,
            "selected_patient_name": None,
            "selected_doctor_name": doctor["name"],
            "ambiguous_patient_names": [],
            "ambiguous_doctor_names": [],
            "contextual_query": f"For {doctor['name']}, {prompt}",
            "reason": "Doctor context is locked for this chat session.",
        }
    return None


def render_current_context_status():
    scope = st.session_state.get("current_scope", CONTEXT_GENERIC)
    patient = st.session_state.get("current_patient")
    doctor = st.session_state.get("current_doctor")
    if scope == CONTEXT_PATIENT and patient:
        label = f"Patient: {patient['Name']}"
    elif scope == CONTEXT_DOCTOR and doctor:
        label = f"Doctor: {doctor['name']}"
    elif scope == CONTEXT_SYSTEM:
        label = "System-wide"
    else:
        label = "Generic"

    with st.container(border=True):
        st.markdown("##### Current context")
        st.metric("Scope", scope.title())
        st.caption(label)


def refresh_next_best_actions(patient=None, result=None):
    patient = patient or st.session_state.get("current_patient")

    st.session_state["next_best_actions_version"] = (

        st.session_state.get("next_best_actions_version", 0) + 1

    )
    if not patient or not has_openai_key():
        st.session_state["next_best_actions"] = []
        return
    try:
        suggestions = suggest_next_best_actions(
            _patient_context_text(patient),
            _conversation_text(),
            _last_result_summary(result or st.session_state.get("last_result")),
        )
    except Exception as exc:
        st.session_state["next_best_actions"] = []
        st.session_state["next_best_actions_error"] = str(exc)
        return
    st.session_state["next_best_actions_error"] = ""
    st.session_state["next_best_actions"] = [
        action.model_dump()
        for action in suggestions.actions[:3]
        if suggestions.has_suggestions and action.suggested_prompt
    ]


def render_next_best_actions():
    actions = st.session_state.get("next_best_actions") or []
    if not actions:
        return
    
    version = st.session_state.get("next_best_actions_version", 0)
    with st.container(key=f"next_best_actions_container_{version}"):
        with st.expander("Next best actions", expanded=False):
            for index, action in enumerate(actions):
                label = action.get("label") or action.get("suggested_prompt")
                priority = action.get("priority", "medium").title()
                prompt = action.get("suggested_prompt", "")
                text_col, button_col = st.columns([0.72, 0.28], vertical_alignment="center")
                with text_col:
                    st.markdown(f"**{index + 1}. {label}**")
                    st.caption(f"{priority} | {prompt}")
                with button_col:
                    if st.button("Run", key=f"next_best_action_{index}", use_container_width=True):
                        st.session_state["queued_chat_prompt"] = prompt
                        st.rerun()


def _current_context_close_label():
    scope = st.session_state.get("current_scope", CONTEXT_GENERIC)
    current_patient = st.session_state.get("current_patient")
    current_doctor = st.session_state.get("current_doctor")
    if scope == CONTEXT_PATIENT and current_patient:
        return current_patient["Name"]
    if scope == CONTEXT_DOCTOR and current_doctor:
        return current_doctor["name"]
    if scope == CONTEXT_SYSTEM:
        return "System-wide context"
    return "Global context"


def archive_current_chat_session(reason="context_closed"):
    """Persist the current visible chat before resetting the working session."""
    messages = st.session_state.get("chat_messages") or []
    if not messages:
        return
    scope = st.session_state.get("current_scope", CONTEXT_GENERIC)
    context_label = _current_context_close_label() or scope.title()
    append_jsonl(
        CHAT_HISTORY_LOG_FILE,
        {
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "started_at": messages[0].get("timestamp"),
            "scope": scope,
            "context_label": context_label,
            "reason": reason,
            "message_count": len(messages),
            "messages": messages,
        },
    )


def render_close_session_controls():
    context_label = _current_context_close_label()
    if not context_label or not st.session_state.get("chat_messages"):
        return
    if st.session_state.get("awaiting_close_confirmation"):
        st.warning(f"Do you need any more assistance for {context_label} before closing this chat session?")
        close_col, keep_col = st.columns(2)
        if close_col.button("No, close session", type="primary"):
            archive_current_chat_session()
            request_patient_chat_reset()
            st.rerun()
        if keep_col.button("Yes, keep working"):
            st.session_state["awaiting_close_confirmation"] = False
            st.session_state["close_session_requested"] = False
            st.rerun()
    elif st.button("Close current context"):
        st.session_state["awaiting_close_confirmation"] = True
        st.rerun()


def render_trace_diagnostics(result):
    if not result:
        return
    if result.get("appointment_workflow", {}).get("status") == "needs_details":
        st.caption("I am waiting for appointment details in chat before booking.")

    evaluation = result.get("evaluation") or {"score": "N/A"}
    tool_logs = result.get("tool_logs", [])
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Tools", len(tool_logs))
    metric_b.metric("Score", evaluation.get("score", "N/A"))

    with st.expander("Planning breakdown", expanded=False):
        st.table(result.get("plan", []))
    with st.expander("Tool logs", expanded=False):
        st.json(tool_logs)


def render_chat_history_tab():
    """Show chat sessions that were archived when their context was closed."""
    sessions = list(reversed(read_jsonl(CHAT_HISTORY_LOG_FILE, limit=100)))
    st.subheader("Chat History")
    if not sessions:
        st.info("No closed chat sessions yet.")
        return

    for index, session in enumerate(sessions, start=1):
        ended_at = _display_chat_timestamp(session.get("ended_at") or session.get("timestamp"))
        started_at = _display_chat_timestamp(session.get("started_at"))
        context_label = session.get("context_label") or "Unknown context"
        scope = (session.get("scope") or "unknown").title()
        title = f"{ended_at or 'Unknown end time'} | {context_label} | {scope}"
        with st.expander(title, expanded=False):
            if started_at:
                st.caption(f"Started: {started_at}")
            st.caption(f"Ended: {ended_at or 'Unknown'}")
            for message in session.get("messages", []):
                timestamp = _display_chat_timestamp(message.get("timestamp"))
                role = (message.get("role") or "message").title()
                if timestamp:
                    st.markdown(f"**{role}** · {timestamp}")
                else:
                    st.markdown(f"**{role}**")
                st.markdown(message.get("content") or "")


def _patient_summary_for_context(patient):
    if not patient:
        return ""
    return patient.get("Summary") or ""


def _set_current_patient_from_result(result):
    if result.get("patient"):
        matched = patient_by_name(result["patient"]["name"])
        if matched:
            previous_name = (st.session_state.get("current_patient") or {}).get("Name")
            st.session_state["current_patient"] = matched
            return matched if previous_name != matched["Name"] else None
    return None


def announce_patient_context(patient):
    if not patient:
        return
    patient_name = patient["Name"]
    if st.session_state.get("last_announced_patient_context") == patient_name:
        return
    st.session_state["last_announced_patient_context"] = patient_name
    append_chat("assistant", f"Patient context set to {patient_name}.")


def _format_slot_options(options):
    return "\n".join(f"{index}. {_slot_label(slot)}" for index, slot in enumerate(options, start=1))


def determine_query_context(prompt, current_patient):
    """Ask the LLM whether the turn is general, global, patient, or doctor scoped."""
    active_patient = (current_patient or {}).get("Name")
    current_doctor = st.session_state.get("current_doctor")
    active_doctor = (current_doctor or {}).get("name")
    if not has_openai_key():
        return {
            "scope": CONTEXT_GENERIC,
            "needs_context_selection": False,
            "selected_patient_name": None,
            "selected_doctor_name": None,
            "ambiguous_patient_names": [],
            "ambiguous_doctor_names": [],
            "contextual_query": prompt,
            "reason": "OPENAI_API_KEY is not configured; agent will return the key setup message.",
        }
    try:
        doctor_names = [doctor["name"] for doctor in load_doctors()]
        decision = classify_query_context(
            prompt,
            unique_patient_names(),
            doctor_names=doctor_names,
            active_patient=active_patient,
            active_doctor=active_doctor,
            conversation=_conversation_text(),
        )
        return decision.model_dump()
    except Exception as exc:
        return {
            "scope": CONTEXT_GENERIC,
            "needs_context_selection": False,
            "selected_patient_name": None,
            "selected_doctor_name": None,
            "ambiguous_patient_names": [],
            "ambiguous_doctor_names": [],
            "contextual_query": prompt,
            "reason": f"LLM context classification failed, so the agent will handle the query directly: {exc}",
        }


def _canonical_patient_name(name):
    """Resolve exact or unique partial patient references to the stored full name."""
    if not name:
        return None
    exact = patient_by_name(name)
    if exact:
        return exact["Name"]

    name_l = name.lower().strip()
    matches = [
        stored_name
        for stored_name in unique_patient_names()
        if name_l in stored_name.lower() or any(part == name_l for part in stored_name.lower().split())
    ]
    return matches[0] if len(matches) == 1 else None


def _patient_names_from_prompt(prompt):
    """Return stored patient names directly referenced by the prompt."""
    prompt_l = prompt.lower()
    matches = []
    for stored_name in unique_patient_names():
        name_l = stored_name.lower()
        name_parts = [part for part in name_l.split() if len(part) > 2]
        if name_l in prompt_l or any(re.search(rf"\b{re.escape(part)}\b", prompt_l) for part in name_parts):
            matches.append(stored_name)
    return matches


def _unique_canonical_patient_names(names):
    canonical = []
    for name in names or []:
        stored_name = _canonical_patient_name(name)
        if stored_name and stored_name not in canonical:
            canonical.append(stored_name)
    return canonical


def normalize_patient_context_decision(prompt, decision):
    """Collapse LLM patient-name variants before deciding whether a selector is needed."""
    if decision.get("scope") != CONTEXT_PATIENT:
        return decision

    selected_name = _canonical_patient_name(decision.get("selected_patient_name"))
    if not selected_name:
        ambiguous_names = _unique_canonical_patient_names(decision.get("ambiguous_patient_names"))
        if len(ambiguous_names) == 1:
            selected_name = ambiguous_names[0]
        else:
            prompt_matches = _patient_names_from_prompt(prompt)
            if len(prompt_matches) == 1:
                selected_name = prompt_matches[0]

    if selected_name:
        normalized = dict(decision)
        normalized["needs_context_selection"] = False
        normalized["selected_patient_name"] = selected_name
        normalized["ambiguous_patient_names"] = []
        if selected_name.lower() not in (normalized.get("contextual_query") or "").lower():
            normalized["contextual_query"] = f"For {selected_name}, {prompt}"
        return normalized

    normalized_candidates = _unique_canonical_patient_names(decision.get("ambiguous_patient_names"))
    if normalized_candidates:
        normalized = dict(decision)
        normalized["ambiguous_patient_names"] = normalized_candidates
        return normalized
    return decision


def entity_matches_for_prompt(prompt):
    store, _ = build_runtime()
    return find_entity_matches(prompt, store)


def unique_doctor_names():
    names = []
    for doctor in load_doctors():
        name = doctor.get("name")
        if name and name not in names:
            names.append(name)
    return names


def _normalize_person_name_text(name):
    """Normalize names for matching while preserving stored display names elsewhere."""
    return re.sub(r"\bdr\.?\b", "", (name or "").lower()).strip()


def doctor_by_name(name):
    if not name:
        return None
    name_l = name.lower()
    normalized_query = _normalize_person_name_text(name)
    for doctor in load_doctors():
        if doctor.get("name", "").lower() == name_l:
            return doctor
    for doctor in load_doctors():
        doctor_name = doctor.get("name", "").lower()
        normalized_doctor_name = _normalize_person_name_text(doctor_name)
        if (
            name_l in doctor_name
            or doctor_name in name_l
            or (normalized_query and normalized_query in normalized_doctor_name)
            or (normalized_query and normalized_doctor_name in normalized_query)
        ):
            return doctor
    return None


def _canonical_doctor_name(name):
    """Resolve exact or unique partial doctor references to the stored full name."""
    if not name:
        return None
    exact = doctor_by_name(name)
    if exact:
        return exact["name"]

    normalized_query = _normalize_person_name_text(name)
    matches = []
    for stored_name in unique_doctor_names():
        normalized_stored = _normalize_person_name_text(stored_name)
        stored_parts = [part for part in normalized_stored.split() if len(part) > 2]
        if (
            normalized_query in normalized_stored
            or normalized_stored in normalized_query
            or any(part == normalized_query for part in stored_parts)
        ):
            matches.append(stored_name)
    return matches[0] if len(matches) == 1 else None


def _doctor_names_from_prompt(prompt):
    """Return stored doctor names directly referenced by the prompt."""
    prompt_l = _normalize_person_name_text(prompt)
    matches = []
    for stored_name in unique_doctor_names():
        normalized_stored = _normalize_person_name_text(stored_name)
        name_parts = [part for part in normalized_stored.split() if len(part) > 2]
        if normalized_stored in prompt_l or any(re.search(rf"\b{re.escape(part)}\b", prompt_l) for part in name_parts):
            matches.append(stored_name)
    return matches


def _unique_canonical_doctor_names(names):
    canonical = []
    for name in names or []:
        stored_name = _canonical_doctor_name(name)
        if stored_name and stored_name not in canonical:
            canonical.append(stored_name)
    return canonical


def normalize_doctor_context_decision(prompt, decision):
    """Collapse LLM doctor-name variants before deciding whether a selector is needed."""
    if decision.get("scope") != CONTEXT_DOCTOR:
        return decision

    selected_name = _canonical_doctor_name(decision.get("selected_doctor_name"))
    if not selected_name:
        ambiguous_names = _unique_canonical_doctor_names(decision.get("ambiguous_doctor_names"))
        if len(ambiguous_names) == 1:
            selected_name = ambiguous_names[0]
        else:
            prompt_matches = _doctor_names_from_prompt(prompt)
            if len(prompt_matches) == 1:
                selected_name = prompt_matches[0]

    if selected_name:
        normalized = dict(decision)
        normalized["needs_context_selection"] = False
        normalized["selected_doctor_name"] = selected_name
        normalized["ambiguous_doctor_names"] = []
        if selected_name.lower() not in (normalized.get("contextual_query") or "").lower():
            normalized["contextual_query"] = f"For {selected_name}, {prompt}"
        return normalized

    normalized_candidates = _unique_canonical_doctor_names(decision.get("ambiguous_doctor_names"))
    if normalized_candidates:
        normalized = dict(decision)
        normalized["ambiguous_doctor_names"] = normalized_candidates
        return normalized
    return decision


def entity_label(entity):
    if entity["type"] == "doctor":
        return f"Doctor: {entity['name']} ({entity['specialty']}, {entity['location']})"
    return f"Patient: {entity['name']} ({entity.get('source') or 'unknown source'}, age {entity.get('age') or 'unknown'})"


def entity_detail_text(entity):
    return format_entity_details(entity)


def _continue_with_selected_entity(label):
    pending = st.session_state.get("pending_entity_selection")
    if not pending:
        return
    entity = pending["label_to_entity"].get(label)
    if not entity:
        return
    st.session_state["pending_entity_selection"] = None
    if entity["type"] == "patient":
        selected_patient = patient_by_id(entity.get("patient_id")) or patient_by_name(entity["name"])
        previous_name = (st.session_state.get("current_patient") or {}).get("Name")
        if selected_patient:
            set_current_scope(CONTEXT_PATIENT, patient=selected_patient)
            if previous_name != selected_patient["Name"]:
                announce_patient_context(selected_patient)
            prompt = pending["query"]
            contextual_prompt = f"For {selected_patient['Name']}, {prompt}"
            with st.spinner("Continuing with selected context..."):
                result = run_agent(
                    contextual_prompt,
                    context_scope=CONTEXT_PATIENT,
                    selected_patient_name=selected_patient["Name"],
                    conversation=_conversation_text(),
                )
            newly_detected_patient = _set_current_patient_from_result(result)
            if newly_detected_patient:
                announce_patient_context(newly_detected_patient)
            st.session_state["last_result"] = result
            append_chat("assistant", result["final_answer"])
            update_followup_state(result, prompt, selected_patient["Name"])
            return
    if entity["type"] == "doctor":
        prompt = pending["query"]
        contextual_prompt = f"For {entity['name']}, {prompt}"
        with st.spinner("Continuing with selected context..."):
            result = run_agent(
                contextual_prompt,
                context_scope=CONTEXT_DOCTOR,
                selected_doctor_name=entity["name"],
                conversation=_conversation_text(),
            )
        st.session_state["last_result"] = result
        append_chat("assistant", result["final_answer"])
        update_followup_state(result, prompt)
        return
    append_chat("assistant", entity_detail_text(entity))


def render_entity_selection_prompt():
    pending = st.session_state.get("pending_entity_selection")
    if not pending:
        return
    labels = list(pending["label_to_entity"].keys())
    with st.container(border=True):
        st.markdown("#### Select context")
        st.write(pending["message"])
        selected_label = st.selectbox("Match", labels, key="manual_entity_selector")
        if st.button("Continue with selected context", type="primary"):
            _continue_with_selected_entity(selected_label)
            st.rerun()


def _continue_with_selected_patient(patient_name):
    pending = st.session_state.get("pending_patient_selection")
    selected_patient = patient_by_name(patient_name)
    if not pending or not selected_patient:
        return

    previous_name = (st.session_state.get("current_patient") or {}).get("Name")
    set_current_scope(CONTEXT_PATIENT, patient=selected_patient)
    st.session_state["pending_patient_selection"] = None
    if previous_name != selected_patient["Name"]:
        announce_patient_context(selected_patient)
    prompt = pending["query"]
    contextual_prompt = f"For {selected_patient['Name']}, {prompt}"
    with st.spinner("Continuing with selected patient..."):
        result = run_agent(
            contextual_prompt,
            context_scope=CONTEXT_PATIENT,
            selected_patient_name=selected_patient["Name"],
            conversation=_conversation_text(),
        )
    newly_detected_patient = _set_current_patient_from_result(result)
    if newly_detected_patient:
        announce_patient_context(newly_detected_patient)
    st.session_state["last_result"] = result
    append_chat("assistant", result["final_answer"])
    update_followup_state(result, prompt, selected_patient["Name"])


def render_patient_selection_prompt():
    pending = st.session_state.get("pending_patient_selection")
    if not pending:
        return

    options = pending.get("candidates") or unique_patient_names()
    with st.container(border=True):
        st.markdown("#### Select patient")
        st.write(pending["message"])
        selected_patient = st.selectbox("Patient", options, key="manual_patient_selector")
        if st.button("Continue with selected patient", type="primary"):
            _continue_with_selected_patient(selected_patient)
            st.rerun()


def _continue_with_selected_doctor(doctor_name):
    pending = st.session_state.get("pending_doctor_selection")
    selected_doctor = doctor_by_name(doctor_name)
    if not pending or not selected_doctor:
        return

    set_current_scope(CONTEXT_DOCTOR, doctor=selected_doctor)
    st.session_state["pending_doctor_selection"] = None
    prompt = pending["query"]
    contextual_prompt = f"For {selected_doctor['name']}, {prompt}"
    with st.spinner("Continuing with selected doctor..."):
        result = run_agent(
            contextual_prompt,
            context_scope=CONTEXT_DOCTOR,
            selected_doctor_name=selected_doctor["name"],
            conversation=_conversation_text(),
        )
    st.session_state["last_result"] = result
    append_chat("assistant", result["final_answer"])
    update_followup_state(result, prompt)


def render_doctor_selection_prompt():
    pending = st.session_state.get("pending_doctor_selection")
    if not pending:
        return

    options = pending.get("candidates") or unique_doctor_names()
    with st.container(border=True):
        st.markdown("#### Select doctor")
        st.write(pending["message"])
        selected_doctor = st.selectbox("Doctor", options, key="manual_doctor_selector")
        if st.button("Continue with selected doctor", type="primary"):
            _continue_with_selected_doctor(selected_doctor)
            st.rerun()


def _doctor_label(doctor, suggested_specialty):
    suffix = " - suggested" if doctor["specialty"] == suggested_specialty else ""
    return f"{doctor['name']} ({doctor['specialty']}, {doctor['location']}){suffix}"


def _ordered_doctors(suggested_specialty):
    doctors = load_doctors()
    suggested_ids = {doctor["doctor_id"] for doctor in find_doctors_by_specialty(suggested_specialty)}
    suggested = [doctor for doctor in doctors if doctor["doctor_id"] in suggested_ids]
    others = [doctor for doctor in doctors if doctor["doctor_id"] not in suggested_ids]
    return suggested + others


def _slots_for_doctor(doctor, selected_date):
    selected_date_text = selected_date.isoformat()
    exact = []
    alternates = []
    today_text = date.today().isoformat()
    for slot in sorted(doctor.get("available_slots", [])):
        if slot[:10] < today_text:
            continue
        slot_item = {
            "doctor_id": doctor["doctor_id"],
            "doctor_name": doctor["name"],
            "specialty": doctor["specialty"],
            "location": doctor["location"],
            "slot": slot,
        }
        if slot.startswith(selected_date_text):
            exact.append(slot_item)
        else:
            alternates.append(slot_item)
    return exact, alternates[:5]


def render_appointment_controls():
    """Render the doctor/date controls for the active guided appointment workflow."""
    pending = st.session_state.get("pending_appointment")
    if not pending or pending.get("stage") != "select_doctor_date":
        return

    st.container(border=True).markdown(
        f"**Appointment setup**\n\n"
        f"Problem: {pending['problem']}\n\n"
        f"Suggested specialist: **{pending['specialty']}**"
    )
    doctors = _ordered_doctors(pending["specialty"])
    labels = [_doctor_label(doctor, pending["specialty"]) for doctor in doctors]
    selected_label = st.selectbox("Doctor", labels, key="appointment_doctor_selector")
    selected_doctor = doctors[labels.index(selected_label)]
    selected_date = st.date_input(
        "Preferred date",
        min_value=date.today(),
        value=max(date.today() + timedelta(days=1), date.today()),
        key="appointment_calendar_date",
    )

    if st.button("Check availability", type="primary"):
        exact, alternates = _slots_for_doctor(selected_doctor, selected_date)
        options = exact or alternates
        st.session_state["pending_appointment"] = {
            **pending,
            "stage": "choose_slot",
            "doctor_id": selected_doctor["doctor_id"],
            "doctor_name": selected_doctor["name"],
            "doctor_specialty": selected_doctor["specialty"],
            "requested_date": selected_date.isoformat(),
            "options": options,
            "used_alternates": not bool(exact),
        }
        if not options:
            response = (
                f"{selected_doctor['name']} has no available slots on or near {selected_date.isoformat()}. "
                "Please select another doctor or date."
            )
        elif exact:
            response = (
                f"{selected_doctor['name']} has availability on {selected_date.isoformat()}:\n\n"
                f"{_format_slot_options(options)}\n\nReply with the slot number you want to book."
            )
        else:
            response = (
                f"{selected_doctor['name']} has no slots on {selected_date.isoformat()}, "
                f"but these alternate slots are open:\n\n{_format_slot_options(options)}\n\n"
                "Reply with the slot number you want to book."
            )
        append_chat("assistant", response)
        if pending.get("patient_name"):
            persist_patient_chat_history(
                pending["patient_name"],
                f"Checked availability for {selected_doctor['name']} on {selected_date.isoformat()}",
                response,
                BOOK_APPOINTMENT_RESULT,
            )
            refresh_next_best_actions(
                st.session_state.get("current_patient"),
                {**BOOK_APPOINTMENT_RESULT, "final_answer": response},
            )
        st.rerun()


def _handle_pending_appointment(prompt):
    """Advance an in-progress appointment booking before starting a new agent run."""
    pending = st.session_state.get("pending_appointment")
    if not pending:
        return False

    current_patient = st.session_state.get("current_patient")
    patient_name = pending.get("patient_name") or (current_patient or {}).get("Name", "")

    if pending["stage"] == "collect_details":
        problem = extract_problem(prompt) or pending.get("problem")
        pending["problem"] = problem
        st.session_state["pending_appointment"] = pending
        if not problem:
            missing = ["the problem or symptoms"]
            known = []
            if problem:
                known.append(f"problem: {problem}")
            known_text = f" I have {', '.join(known)}." if known else ""
            response = f"I still need {', '.join(missing)} before I can check availability.{known_text}"
            record_appointment_chat_turn(patient_name, prompt, response, current_patient)
            return True

        specialty_decision = classify_specialty(problem, available_specialties()) if has_openai_key() else None
        specialty = (
            specialty_decision.specialty
            if specialty_decision and specialty_decision.has_sufficient_symptom_context and specialty_decision.specialty
            else (available_specialties()[0] if available_specialties() else "General Physician")
        )
        st.session_state["pending_appointment"] = {
            "stage": "select_doctor_date",
            "patient_name": patient_name,
            "problem": problem,
            "specialty": specialty,
        }
        response = (
            f"Based on the symptoms, I suggest a {specialty}. "
            "Use the doctor and calendar selector below to continue with the suggested doctor type or choose another available doctor."
        )
        record_appointment_chat_turn(patient_name, prompt, response, current_patient)
        return True

    if pending["stage"] == "select_doctor_date":
        response = "Please use the doctor selector and calendar below to check availability."
        record_appointment_chat_turn(patient_name, prompt, response, current_patient)
        return True

    if pending["stage"] == "choose_slot":
        match = re.search(r"\b(\d+)\b", prompt)
        options = pending.get("options", [])
        if not match or not options:
            response = "Please reply with the number of the slot you want to book."
            record_appointment_chat_turn(patient_name, prompt, response, current_patient)
            return True
        selected_index = int(match.group(1)) - 1
        if selected_index < 0 or selected_index >= len(options):
            response = f"Please choose a number between 1 and {len(options)}."
            record_appointment_chat_turn(patient_name, prompt, response, current_patient)
            return True
        slot = options[selected_index]
        booking = book_specific_appointment(patient_name, slot["doctor_id"], slot["slot"])
        st.session_state["guided_booking"] = booking
        st.session_state["pending_appointment"] = None
        if booking.get("success"):
            appointment = booking["appointment"]
            response = (
                f"Booked {appointment['doctor_name']} ({appointment['specialty']}) for "
                f"{appointment['patient_name']} on {appointment['slot']} at {appointment['location']}."
            )
        else:
            response = booking.get("message", "I could not book that appointment.")
        record_appointment_chat_turn(patient_name, prompt, response, current_patient)
        return True

    return False


def update_followup_state(result, prompt, fallback_patient_name=""):
    """Translate agent results into UI follow-up state for appointments and offers."""
    if result.get("appointment_workflow", {}).get("status") == "needs_details":
        workflow = result["appointment_workflow"]
        patient_name = workflow.get("patient_name") or fallback_patient_name
        missing = workflow.get("missing", [])
        if "problem or symptoms" not in missing and workflow.get("specialty"):
            st.session_state["pending_appointment"] = {
                "stage": "select_doctor_date",
                "patient_name": patient_name,
                "problem": workflow.get("problem") or prompt,
                "specialty": workflow["specialty"],
            }
        else:
            st.session_state["pending_appointment"] = {"stage": "collect_details", "patient_name": patient_name}
    elif any(step["task"] == "list_appointments" for step in result.get("plan", [])) and not result.get("appointment_list"):
        patient_name = (result.get("patient") or {}).get("name", "") or fallback_patient_name
        st.session_state["appointment_offer"] = {"patient_name": patient_name}


def persist_patient_chat_history(patient_name, user_message, assistant_message, result=None):
    """Persist patient-scoped turns as lightweight history entries."""
    if not patient_name or not user_message or not assistant_message:
        return
    tasks = [step.get("task", "") for step in (result or {}).get("plan", [])]
    if any("appointment" in task or task == "book_appointment" for task in tasks):
        entry_type = "appointment"
    elif any(task == "retrieve_medical_information" for task in tasks):
        entry_type = "medical_advice"
    elif any(task == "retrieve_patient_reports" for task in tasks):
        entry_type = "report_review"
    elif any(task == "retrieve_patient_context" for task in tasks):
        entry_type = "patient_summary_review"
    else:
        entry_type = "chat_interaction"
    summary = f"{entry_type.replace('_', ' ').title()}: {user_message[:180]}"
    timestamp = datetime.now().isoformat(timespec="seconds")
    raw_text = f"Timestamp: {timestamp}\nUser: {user_message}\nAssistant: {assistant_message}"
    add_patient_history_entry(patient_name, entry_type, summary, raw_text)
    refresh_runtime_data()


def record_appointment_chat_turn(patient_name, user_message, assistant_message, current_patient=None):
    """Keep appointment chat side effects in one place."""
    append_chat("assistant", assistant_message)
    persist_patient_chat_history(patient_name, user_message, assistant_message, BOOK_APPOINTMENT_RESULT)
    refresh_next_best_actions(
        current_patient,
        {**BOOK_APPOINTMENT_RESULT, "final_answer": assistant_message},
    )


def progress_area(progress_container=None):
    return progress_container.container() if progress_container is not None else nullcontext()


def run_chat_turn(prompt, progress_container=None):
    """Handle one complete chat turn, including context selection and tool execution."""
    append_chat("user", prompt)

    if st.session_state.get("awaiting_close_confirmation"):
        scope = st.session_state.get("current_scope", CONTEXT_GENERIC)
        current_patient = st.session_state.get("current_patient")
        current_doctor = st.session_state.get("current_doctor")
        context_label = _current_context_close_label() or "this context"
        if is_no(prompt):
            append_chat("assistant", f"Closing the chat session for {context_label}. Starting fresh.")
            if scope == CONTEXT_PATIENT and current_patient:
                persist_patient_chat_history(
                    current_patient["Name"],
                    prompt,
                    f"Closing the chat session for {context_label}. Starting fresh.",
                )
            st.session_state["next_best_actions"] = []
            archive_current_chat_session()
            request_patient_chat_reset()
            return None
        if is_yes(prompt):
            st.session_state["awaiting_close_confirmation"] = False
            st.session_state["close_session_requested"] = False
            append_chat("assistant", f"Okay, I will keep working with {context_label}. What else can I help with?")
            if scope == CONTEXT_PATIENT and current_patient:
                persist_patient_chat_history(
                    current_patient["Name"],
                    prompt,
                    f"Okay, I will keep working with {context_label}. What else can I help with?",
                )
                refresh_next_best_actions(current_patient)
            return None
        append_chat("assistant", f"Please reply yes if you need more assistance for {context_label}, or no to close this session.")
        return None

    if is_end_session_request(prompt):
        scope = st.session_state.get("current_scope", CONTEXT_GENERIC)
        current_patient = st.session_state.get("current_patient")
        current_doctor = st.session_state.get("current_doctor")
        context_label = _current_context_close_label() or "this context"
        st.session_state["awaiting_close_confirmation"] = True
        st.session_state["close_session_requested"] = True
        response = f"Before I close, do you need any more assistance for {context_label}?"
        append_chat("assistant", response)
        if scope == CONTEXT_PATIENT and current_patient:
            persist_patient_chat_history(current_patient["Name"], prompt, response)
            refresh_next_best_actions(current_patient)
        return None

    appointment_offer = st.session_state.get("appointment_offer")
    if appointment_offer and prompt.strip().lower() in {"yes", "y", "sure", "ok", "okay", "book one", "please book"}:
        patient_name = appointment_offer.get("patient_name", "")
        st.session_state["pending_appointment"] = {
            "stage": "collect_details",
            "patient_name": patient_name,
        }
        st.session_state["appointment_offer"] = None
        response = "Sure. What problem or symptoms should I use? I will suggest a specialist and then show a calendar."
        record_appointment_chat_turn(patient_name, prompt, response, st.session_state.get("current_patient"))
        return None

    if _handle_pending_appointment(prompt):
        return None

    current_patient = st.session_state.get("current_patient")
    sticky_decision = active_context_decision(prompt)
    if sticky_decision:
        context_decision = sticky_decision
    else:
        with st.spinner("Determining conversation context..."):
            context_decision = determine_query_context(prompt, current_patient)
    context_decision = normalize_patient_context_decision(prompt, context_decision)
    context_decision = normalize_doctor_context_decision(prompt, context_decision)
    scope = context_decision.get("scope", CONTEXT_GENERIC)
    conversation = _conversation_text()

    if scope == CONTEXT_PATIENT and context_decision.get("needs_context_selection"):
        candidates = context_decision.get("ambiguous_patient_names") or unique_patient_names()
        message = (
            "I found more than one possible patient reference. Please choose the patient to continue."
            if context_decision.get("ambiguous_patient_names")
            else "I could not identify which patient this is for. Please choose the patient manually."
        )
        st.session_state["pending_patient_selection"] = {
            "query": prompt,
            "candidates": candidates,
            "message": message,
        }
        append_chat("assistant", message)
        return None

    if scope == CONTEXT_DOCTOR and context_decision.get("needs_context_selection"):
        candidates = context_decision.get("ambiguous_doctor_names") or unique_doctor_names()
        message = (
            "I found more than one possible doctor reference. Please choose the doctor to continue."
            if context_decision.get("ambiguous_doctor_names")
            else "I could not identify which doctor this is for. Please choose the doctor manually."
        )
        st.session_state["pending_doctor_selection"] = {
            "query": prompt,
            "candidates": candidates,
            "message": message,
        }
        append_chat("assistant", message)
        return None

    selected_patient_name = context_decision.get("selected_patient_name") if scope == CONTEXT_PATIENT else None
    selected_doctor_name = context_decision.get("selected_doctor_name") if scope == CONTEXT_DOCTOR else None

    if selected_patient_name:
        selected_patient = patient_by_name(selected_patient_name)
        if selected_patient:
            previous_name = (current_patient or {}).get("Name")
            current_patient = selected_patient
            set_current_scope(CONTEXT_PATIENT, patient=selected_patient)
            if previous_name != selected_patient["Name"]:
                announce_patient_context(selected_patient)

    current_doctor = st.session_state.get("current_doctor")
    if selected_doctor_name:
        selected_doctor = doctor_by_name(selected_doctor_name)
        if selected_doctor:
            current_doctor = selected_doctor
            set_current_scope(CONTEXT_DOCTOR, doctor=selected_doctor)

    if scope in {CONTEXT_GENERIC, CONTEXT_SYSTEM}:
        st.session_state["current_scope"] = scope

    if scope == CONTEXT_PATIENT and not current_patient:
        message = "I could not identify which patient this is for. Please choose the patient manually."
        st.session_state["pending_patient_selection"] = {
            "query": prompt,
            "candidates": unique_patient_names(),
            "message": message,
        }
        append_chat("assistant", message)
        return None

    if scope == CONTEXT_DOCTOR and not current_doctor:
        message = "I could not identify which doctor this is for. Please choose the doctor manually."
        st.session_state["pending_doctor_selection"] = {
            "query": prompt,
            "candidates": unique_doctor_names(),
            "message": message,
        }
        append_chat("assistant", message)
        return None

    contextual_prompt = context_decision.get("contextual_query") or prompt
    if (
        scope == CONTEXT_PATIENT
        and current_patient
        and current_patient["Name"].lower() not in contextual_prompt.lower()
    ):
        contextual_prompt = f"For {current_patient['Name']}, {prompt}"
    if (
        scope == CONTEXT_DOCTOR
        and current_doctor
        and current_doctor["name"].lower() not in contextual_prompt.lower()
    ):
        contextual_prompt = f"For {current_doctor['name']}, {prompt}"

    with progress_area(progress_container):
        with st.spinner("Thinking and planning through the selected context and tools for an action..."):
            result = run_agent(
                contextual_prompt,
                context_scope=scope,
                selected_patient_name=(current_patient or {}).get("Name") if scope == CONTEXT_PATIENT else None,
                selected_doctor_name=(current_doctor or {}).get("name") if scope == CONTEXT_DOCTOR else None,
                conversation=conversation,
            )

    newly_detected_patient = _set_current_patient_from_result(result)
    if newly_detected_patient:
        announce_patient_context(newly_detected_patient)

    st.session_state["last_result"] = result
    append_chat("assistant", result["final_answer"])
    history_patient = (st.session_state.get("current_patient") or current_patient) if scope == CONTEXT_PATIENT else None
    if history_patient:
        persist_patient_chat_history(history_patient["Name"], prompt, result["final_answer"], result)
        refresh_next_best_actions(history_patient, result)
    update_followup_state(result, prompt)
    return result


def _slot_label(slot):
    return f"{slot['slot']} - {slot['doctor_name']} ({slot['specialty']}, {slot['location']})"



def render_app():
    """Render the full Streamlit application UI."""
    apply_assistant_layout_styles()
    render_llm_settings_sidebar()

    # Top-level navigation stays here; individual tab bodies delegate to helpers.
    st.title("Agentic Healthcare Assistant")
    st.caption(DISCLAIMER)

    tabs = st.tabs(["Assistant", "Chat History", "Patients", "Appointments", "Agent Trace", "Evaluation"])

    with tabs[0]:
        ensure_chat_state()
        st.subheader("Assistant Chat")

        patient_names = unique_patient_names()
        current_patient = st.session_state.get("current_patient")
        result = st.session_state.get("last_result")
        chat_col, side_col = st.columns([0.64, 0.36], gap="large")

        with side_col:
            st.markdown('<span class="workspace-panel-marker"></span>', unsafe_allow_html=True)
            render_current_context_status()
            st.markdown("#### Session Workspace")

            current_name = current_patient["Name"] if current_patient else "Auto-detect from chat"
            selected = st.selectbox(
                "Patient context",
                ["Auto-detect from chat"] + patient_names,
                index=(["Auto-detect from chat"] + patient_names).index(current_name)
                if current_name in ["Auto-detect from chat"] + patient_names
                and st.session_state["patient_context_select_version"] == 0
                else 0,
                key=f"patient_context_select_{st.session_state['patient_context_select_version']}",
            )
            if selected != "Auto-detect from chat":
                selected_patient = patient_by_name(selected)
                if selected_patient and selected_patient != current_patient:
                    set_current_scope(CONTEXT_PATIENT, patient=selected_patient)
                    current_patient = selected_patient

            with st.container(border=True):
                st.markdown("##### Patient context")
                if current_patient:
                    st.metric("Patient", current_patient["Name"])
                    col_a, col_b = st.columns(2)
                    col_a.metric("Age", current_patient.get("Age") or "Unknown")
                    col_b.metric("Gender", current_patient.get("Gender") or "Unknown")
                    st.caption(
                        f"Phone: {current_patient.get('Phone') or 'Unknown'}\n\n"
                        f"Address: {current_patient.get('Address') or 'Unknown'}"
                    )
                    if current_patient.get("Summary"):
                        with st.expander("Clinical summary", expanded=False):
                            st.write(current_patient["Summary"])
                else:
                    st.info("Select a patient or mention a patient by name in chat.")

            if current_patient:
                with st.expander(f"Add to {current_patient['Name']}'s history", expanded=False):
                    history_type = st.selectbox(
                        "Entry type",
                        ["symptoms", "medical_history", "advice", "treatment", "appointment_note", "general_note"],
                        format_func=lambda value: value.replace("_", " ").title(),
                        key="chat_history_entry_type",
                    )
                    history_note = st.text_area(
                        "History note",
                        placeholder="Add symptoms, advice given, treatment notes, care instructions, or other patient history...",
                        key="chat_history_note",
                    )
                    if st.button("Add note to patient history", type="primary"):
                        if history_note.strip():
                            add_patient_history_entry(
                                current_patient["Name"],
                                history_type,
                                f"{history_type.replace('_', ' ').title()}: {history_note.strip()[:180]}",
                                history_note.strip(),
                                source="assistant_patient_context_note",
                            )
                            refresh_runtime_data()
                            append_chat(
                                "assistant",
                                f"Added this {history_type.replace('_', ' ')} note to {current_patient['Name']}'s patient history.",
                            )
                            refresh_next_best_actions(current_patient)
                            st.rerun()
                        else:
                            st.warning("Please enter a note before adding it to the patient history.")

                    st.markdown("##### Upload medical reports")
                    context_reports = st.file_uploader(
                        "Upload PDF report(s) for current patient",
                        type=["pdf"],
                        accept_multiple_files=True,
                        key="chat_context_report_upload",
                    )
                    if st.button("Import report(s) to patient history"):
                        if not context_reports:
                            st.warning("Please choose at least one PDF report.")
                        else:
                            imported = 0
                            for uploaded_report in context_reports:
                                try:
                                    raw_text = extract_uploaded_pdf_text(uploaded_report)
                                except Exception as exc:
                                    st.warning(f"Could not read {uploaded_report.name}: {exc}")
                                    continue
                                summary = raw_text[:1600] if raw_text else f"Uploaded medical report: {uploaded_report.name}"
                                add_user_medical_record(
                                    current_patient["Name"],
                                    summary,
                                    raw_text,
                                    uploaded_report.name,
                                    metadata={"record_type": "uploaded_medical_record"},
                                )
                                imported += 1
                            if imported:
                                refresh_runtime_data()
                                append_chat(
                                    "assistant",
                                    f"Imported {imported} medical report(s) into {current_patient['Name']}'s patient history.",
                                )
                                refresh_next_best_actions(current_patient)
                                st.rerun()

            if result:
                with st.expander("Run details", expanded=False):
                    render_trace_diagnostics(result)

        with chat_col:
            st.markdown('<span class="chat-panel-marker"></span>', unsafe_allow_html=True)
            prompt = st.chat_input(
                "Ask about any patient, doctor, appointments, records, or medical information...",
                key="assistant_chat_input",
            )
            progress_slot = st.empty()
            queued_prompt = st.session_state.pop("queued_chat_prompt", "")
            submitted_prompt = queued_prompt or prompt
            if submitted_prompt and submitted_prompt.strip():
                run_chat_turn(submitted_prompt.strip(), progress_container=progress_slot)
                st.rerun()

            render_patient_selection_prompt()
            render_doctor_selection_prompt()
            render_entity_selection_prompt()
            render_appointment_controls()
            render_close_session_controls()

            render_next_best_actions()

            if st.session_state["chat_messages"]:
                st.markdown("#### Conversation")
            for message in reversed(st.session_state["chat_messages"]):
                with st.chat_message(message["role"]):
                    timestamp = _display_chat_timestamp(message.get("timestamp"))
                    if timestamp:
                        st.caption(timestamp)
                    st.markdown(message["content"])

    with tabs[1]:
        render_chat_history_tab()

    with tabs[2]:
        render_patients_tab()

    with tabs[3]:
        render_appointments_tab()

    with tabs[4]:
        render_agent_trace_tab()

    with tabs[5]:
        render_evaluation_tab()
