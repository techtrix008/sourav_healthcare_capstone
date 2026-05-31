from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, TypedDict

from .appointment_service import (
    available_specialties,
    book_specific_appointment,
    clear_appointments,
    find_doctor_name_in_text,
    find_slots_for_specialty,
    load_doctors,
    list_all_appointments,
    list_appointments_for_doctor,
    list_appointments_for_patient,
)
from .agent_formatting import format_entity_choices, format_entity_details
from .config import AGENT_LOG_FILE, DISCLAIMER, PDF_FILES, RECORDS_FILE
from .context_builder import doctor_context, patient_context, system_context
from .context_types import CONTEXT_DOCTOR, CONTEXT_GENERIC, CONTEXT_PATIENT, CONTEXT_SYSTEM
from .data_loader import load_excel_records
from .disease_search import search_medical_info
from .evaluator import evaluate_agent_run
from .llm_service import (
    AgentPlan,
    answer_scoped_context_question,
    answer_patient_record_question,
    classify_entity_lookup_intent,
    classify_general_symptom_advice_intent,
    classify_patient_record_question_intent,
    classify_query_context,
    classify_report_intent,
    classify_specialty,
    general_answer_with_llm,
    has_openai_key,
    plan_with_llm,
    triage_safety,
)
from .logger import append_jsonl
from .patient_store import PatientStore
from .pdf_parser import parse_patient_pdfs
from .record_service import load_user_records
from .report_service import get_patient_reports
from .vector_store import SimpleVectorStore

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover
    END = None
    StateGraph = None


OPERATIONAL_APPOINTMENT_TASKS = {
    "list_appointments",
    "list_all_appointments",
    "list_doctor_appointments",
    "list_active_appointment_doctors",
    "list_doctors",
    "book_appointment",
    "clear_appointments",
}

PATIENT_REQUIRED_TASKS = {"retrieve_patient_context", "book_appointment", "list_appointments"}


class AgentState(TypedDict, total=False):
    query: str
    context_scope: str
    selected_patient_name: str
    selected_doctor_name: str
    conversation: str
    scoped_context: str
    plan: list[dict[str, Any]]
    plan_model: AgentPlan
    patient: Any
    patient_summary: str
    appointment: dict[str, Any]
    appointment_list: list[dict[str, Any]]
    active_appointment_doctors: list[dict[str, Any]]
    doctor_list: list[dict[str, Any]]
    entity_matches: list[dict[str, Any]]
    appointment_workflow: dict[str, Any]
    medical_info: dict[str, Any]
    patient_reports: dict[str, Any]
    tool_logs: list[dict[str, Any]]
    final_answer: str


@lru_cache(maxsize=1)
def build_runtime() -> tuple[PatientStore, SimpleVectorStore]:
    """Load all local records once and build the in-memory patient/retrieval stores."""
    records = []
    if RECORDS_FILE.exists():
        records.extend(load_excel_records(RECORDS_FILE))
    records.extend(parse_patient_pdfs(PDF_FILES))
    records.extend(load_user_records())
    store = PatientStore(records)
    vector_store = SimpleVectorStore(records)
    return store, vector_store


def clear_runtime_cache() -> None:
    build_runtime.cache_clear()


def _extract_patient_name(query: str, store: PatientStore) -> str | None:
    query_l = query.lower()
    for patient in store.list_patients():
        if patient.name.lower() in query_l:
            return patient.name

    candidate = re.search(r"(?:for|of|is|summarize)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", query)
    return candidate.group(1).strip() if candidate else None


def plan_tasks(query: str) -> list[dict[str, str]]:
    plan = plan_with_llm(query)
    return [step.model_dump() for step in plan.steps]


def _extract_requested_date(query: str) -> str | None:
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", query)
    return match.group(1) if match else None


def _renumber_steps(plan_model: AgentPlan) -> AgentPlan:
    for index, step in enumerate(plan_model.steps, start=1):
        step.step = index
    return plan_model


def _insert_plan_step_before_final(plan_model: AgentPlan, task: str, reason: str) -> AgentPlan:
    if not plan_model.steps:
        return plan_model
    final_step = next((step for step in plan_model.steps if step.task == "final_response"), None)
    if final_step:
        final_step.step += 1
    step_type = type(plan_model.steps[0])
    plan_model.steps.insert(0, step_type(step=1, task=task, reason=reason))
    return _renumber_steps(plan_model)


def _has_stored_report_intent(query: str) -> bool:
    try:
        report_intent = classify_report_intent(query)
    except Exception:
        return False
    return bool(report_intent and report_intent.asks_for_stored_reports)


def _has_patient_record_question_intent(query: str) -> bool:
    try:
        record_intent = classify_patient_record_question_intent(query)
    except Exception:
        return False
    return bool(record_intent and record_intent.asks_patient_record_question)


def _normalize_report_plan(plan_model: AgentPlan, query: str) -> AgentPlan:
    """Force report/test-result requests onto the stored-report path."""
    has_operational_task = any(step.task in OPERATIONAL_APPOINTMENT_TASKS for step in plan_model.steps)
    already_has_report_task = any(step.task == "retrieve_patient_reports" for step in plan_model.steps)

    if has_operational_task and not already_has_report_task:
        return plan_model
    if not _has_stored_report_intent(query):
        return plan_model
    if not plan_model.steps:
        return plan_model
    for step in plan_model.steps:
        if step.task == "retrieve_medical_information":
            step.task = "retrieve_patient_reports"
            step.reason = "The user asked for reports/test results from stored patient records."
    if not any(step.task == "retrieve_patient_reports" for step in plan_model.steps):
        plan_model = _insert_plan_step_before_final(
            plan_model,
            "retrieve_patient_reports",
            "The user asked for stored reports/test results.",
        )
    plan_model.needs_patient = True
    return _renumber_steps(plan_model)


def _remove_unrequested_report_steps(plan_model: AgentPlan, query: str) -> AgentPlan:
    """Prevent a previous report intent from polluting appointment or doctor queries."""
    has_operational_task = any(step.task in OPERATIONAL_APPOINTMENT_TASKS for step in plan_model.steps)
    has_report_task = any(step.task == "retrieve_patient_reports" for step in plan_model.steps)
    if not has_operational_task or not has_report_task:
        return plan_model
    if _has_stored_report_intent(query):
        return plan_model
    plan_model.steps = [step for step in plan_model.steps if step.task != "retrieve_patient_reports"]
    return _renumber_steps(plan_model)


def _entity_query_terms(query: str) -> list[str]:
    stop_words = {
        "who",
        "is",
        "are",
        "the",
        "a",
        "an",
        "show",
        "tell",
        "me",
        "about",
        "details",
        "detail",
        "for",
        "of",
        "doctor",
        "dr",
        "patient",
    }
    return [token for token in re.findall(r"[a-z]+", query.lower()) if token not in stop_words and len(token) > 2]


def find_entity_matches(query: str, store: PatientStore) -> list[dict[str, Any]]:
    terms = _entity_query_terms(query)
    if not terms:
        return []
    matches = []
    seen = set()
    for patient in store.list_patients():
        name_l = patient.name.lower()
        if any(term in name_l for term in terms):
            key = ("patient", patient.patient_id)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "type": "patient",
                    "patient_id": patient.patient_id,
                    "name": patient.name,
                    "age": patient.age,
                    "gender": patient.gender,
                    "phone": patient.phone,
                    "email": patient.email,
                    "address": patient.address,
                    "source": patient.source,
                    "summary": patient.summary,
                }
            )
    for doctor in load_doctors():
        name_l = doctor.get("name", "").lower()
        if any(term in name_l for term in terms):
            key = ("doctor", doctor.get("doctor_id"))
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                {
                    "type": "doctor",
                    "doctor_id": doctor.get("doctor_id"),
                    "name": doctor.get("name"),
                    "specialty": doctor.get("specialty"),
                    "location": doctor.get("location"),
                    "available_slots": doctor.get("available_slots", []),
                }
            )
    return matches


def _normalize_entity_lookup_plan(plan_model: AgentPlan, query: str, store: PatientStore) -> AgentPlan:
    if _has_stored_report_intent(query):
        return plan_model

    if _has_patient_record_question_intent(query):
        return plan_model

    if any(step.task == "lookup_entity_details" for step in plan_model.steps):
        return plan_model
    try:
        entity_intent = classify_entity_lookup_intent(query)
    except Exception:
        entity_intent = None
    if not entity_intent or not entity_intent.asks_for_entity_lookup or not find_entity_matches(query, store):
        return plan_model
    plan_model = _insert_plan_step_before_final(
        plan_model,
        "lookup_entity_details",
        "The user asked who a named person is; search patients and doctors.",
    )
    plan_model.needs_patient = False
    return plan_model


def _normalize_plan_with_context(plan_model: AgentPlan, query: str, store: PatientStore) -> AgentPlan:
    """Align the LLM plan with context classifiers before tools run."""
    if _has_stored_report_intent(query):
        return plan_model

    if _has_patient_record_question_intent(query):
        step_type = type(plan_model.steps[0]) if plan_model.steps else None
        if step_type:
            plan_model.steps = [
                step_type(
                    step=1,
                    task="retrieve_patient_context",
                    reason="The user asked a targeted question about the patient's stored record/history.",
                ),
                step_type(
                    step=2,
                    task="final_response",
                    reason="Answer only the requested patient-record question.",
                ),
            ]
        plan_model.needs_patient = True
        return plan_model

    try:
        advice_intent = classify_general_symptom_advice_intent(query)
    except Exception:
        advice_intent = None
    if advice_intent and advice_intent.asks_for_general_symptom_advice:
        plan_model.needs_patient = False
        plan_model.patient_name = None
        for step in plan_model.steps:
            if step.task == "retrieve_patient_context":
                step.task = "retrieve_medical_information"
                step.reason = "The user asked for general symptom guidance, not a patient record."
            if step.task in {"list_appointments", "book_appointment", "retrieve_patient_reports"}:
                step.task = "retrieve_medical_information"
                step.reason = "The user asked for general symptom guidance without a patient-specific administrative action."
        task_names = {step.task for step in plan_model.steps}
        step_type = type(plan_model.steps[0]) if plan_model.steps else None
        if step_type and "retrieve_medical_information" not in task_names:
            plan_model.steps.insert(
                0,
                step_type(
                    step=1,
                    task="retrieve_medical_information",
                    reason="Fetch trusted general information relevant to the symptom guidance request.",
                ),
            )
        return _renumber_steps(plan_model)

    try:
        decision = classify_query_context(
            query,
            [patient.name for patient in store.list_patients()],
            doctor_names=[doctor["name"] for doctor in load_doctors()],
        )
    except Exception:
        return plan_model
    if decision.selected_patient_name:
        plan_model.patient_name = decision.selected_patient_name
    if decision.scope != CONTEXT_SYSTEM:
        return plan_model
    for step in plan_model.steps:
        if step.task == "list_appointments":
            step.task = "list_all_appointments"
            step.reason = "The user asked for all appointments across the system."
    plan_model.needs_patient = False
    plan_model.patient_name = None
    return _renumber_steps(plan_model)


def active_appointment_doctors() -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for appointment in list_all_appointments():
        if appointment.get("status", "").lower() not in {"booked", "active", "scheduled"}:
            continue
        doctor_id = appointment.get("doctor_id") or appointment.get("doctor_name")
        if doctor_id not in grouped:
            grouped[doctor_id] = {
                "doctor_id": appointment.get("doctor_id"),
                "doctor_name": appointment.get("doctor_name"),
                "specialty": appointment.get("specialty"),
                "location": appointment.get("location"),
                "appointment_count": 0,
                "patients": [],
                "slots": [],
            }
        grouped[doctor_id]["appointment_count"] += 1
        grouped[doctor_id]["patients"].append(appointment.get("patient_name"))
        grouped[doctor_id]["slots"].append(appointment.get("slot"))
    return list(grouped.values())


def summarize_patient(record_text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", record_text.strip())
    selected = [sentence for sentence in sentences if any(term in sentence.lower() for term in ["diagnosis", "plan", "history", "follow", "medication", "vitals"])]
    summary = " ".join(selected[:5]) or " ".join(sentences[:4])
    return summary[:1200]


def _bulletize_text(text: str) -> list[str]:
    sentences = [sentence.strip(" .") for sentence in re.split(r"(?<=[.!?])\s+", text or "") if sentence.strip()]
    return sentences or [text] if text else []


def _patient_record_context(patient) -> str:
    """Build the grounded context block used by patient-record Q&A prompts."""
    parts = [
        f"Patient: {patient.name}",
        f"Age: {patient.age or 'Unknown'}",
        f"Gender: {patient.gender or 'Unknown'}",
        f"Phone: {patient.phone or 'Unknown'}",
        f"Email: {patient.email or 'Unknown'}",
        f"Address: {patient.address or 'Unknown'}",
        f"Primary summary: {patient.summary or ''}",
    ]
    documents = (patient.metadata or {}).get("source_documents", [])
    for index, document in enumerate(documents, start=1):
        metadata = document.get("metadata") or {}
        raw_text = document.get("raw_text") or ""
        if metadata.get("record_type") == "patient_history_entry" and "Assistant:" in raw_text:
            raw_text = raw_text.split("Assistant:", 1)[0].strip()
        parts.extend(
            [
                "",
                f"Linked record {index}:",
                f"Source: {document.get('source') or 'Unknown'}",
                f"Record type: {metadata.get('record_type') or 'source_record'}",
                f"Entry type: {metadata.get('entry_type') or 'not specified'}",
                f"Captured/uploaded/visit date: {metadata.get('captured_at') or metadata.get('uploaded_at') or metadata.get('visit_date') or 'Unknown'}",
                f"Summary: {document.get('summary') or ''}",
                f"Raw text: {raw_text[:1800]}",
            ]
        )
    return "\n".join(parts).strip()


def _doctor_by_name(name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    name_l = name.lower()
    for doctor in load_doctors():
        if doctor.get("name", "").lower() == name_l:
            return doctor
    for doctor in load_doctors():
        doctor_name = doctor.get("name", "").lower()
        if name_l in doctor_name or doctor_name in name_l:
            return doctor
    return None


def _run_agent_impl(
    query: str,
    plan_model: AgentPlan | None = None,
    *,
    context_scope: str = "",
    selected_patient_name: str | None = None,
    selected_doctor_name: str | None = None,
    conversation: str = "",
) -> dict[str, Any]:
    """Execute a planned turn: normalize plan, call tools, compose answer, and evaluate."""
    store, vector_store = build_runtime()
    selected_doctor = _doctor_by_name(selected_doctor_name)
    scoped_context = ""
    if selected_patient_name:
        selected_patient = store.get_patient_by_name(selected_patient_name)
        if selected_patient:
            scoped_context = patient_context(selected_patient)
    if selected_doctor:
        scoped_context = doctor_context(selected_doctor)
    plan_model = plan_model or plan_with_llm(
        query,
        active_patient=selected_patient_name,
        active_doctor=selected_doctor_name,
        conversation=conversation,
        scoped_context=scoped_context,
    )
    if selected_patient_name:
        plan_model.patient_name = selected_patient_name
        plan_model.needs_patient = True
    if selected_doctor_name:
        plan_model.doctor_name = selected_doctor_name
    if context_scope not in {CONTEXT_GENERIC, CONTEXT_SYSTEM}:
        plan_model = _normalize_plan_with_context(plan_model, query, store)
    plan_model = _normalize_entity_lookup_plan(plan_model, query, store)
    plan_model = _normalize_report_plan(plan_model, query)
    plan_model = _remove_unrequested_report_steps(plan_model, query)
    plan = [step.model_dump() for step in plan_model.steps]
    tool_logs: list[dict[str, Any]] = []

    task_names = {task["task"] for task in plan}
    needs_patient = plan_model.needs_patient or bool(task_names & PATIENT_REQUIRED_TASKS)
    patient = None
    if needs_patient:
        patient_name = plan_model.patient_name or _extract_patient_name(query, store)
        if patient_name:
            patient = store.get_patient_by_name(patient_name)
        if not patient:
            matches = vector_store.search(query, limit=1)
            patient = matches[0][0] if matches else None

    if needs_patient and patient:
        tool_logs.append({"tool": "patient_lookup", "success": True, "message": f"Matched {patient.name}"})
        patient_summary = summarize_patient(patient.searchable_text())
    elif needs_patient:
        tool_logs.append({"tool": "patient_lookup", "success": False, "message": "No matching patient found"})
        patient_summary = "No matching patient record was found in the attached dataset."
    else:
        patient_summary = "No patient lookup was needed for this request."

    appointment = {}
    appointment_workflow = {}
    appointment_list = []
    active_doctors = []
    doctor_list = []
    entity_matches = []
    patient_reports = {}
    if "clear_appointments" in task_names:
        appointment = clear_appointments()
        tool_logs.append({"tool": "clear_appointments", "success": appointment.get("success", False), "message": appointment.get("message", "")})
    if "list_appointments" in task_names:
        if patient:
            appointment_list = list_appointments_for_patient(patient.name)
            tool_logs.append(
                {
                    "tool": "list_appointments",
                    "success": True,
                    "message": f"Found {len(appointment_list)} appointment(s) for {patient.name}",
                }
            )
        else:
            tool_logs.append({"tool": "list_appointments", "success": False, "message": "No patient selected"})
    if "list_all_appointments" in task_names:
        appointment_list = list_all_appointments()
        tool_logs.append(
            {
                "tool": "list_all_appointments",
                "success": True,
                "message": f"Found {len(appointment_list)} appointment(s) across all patients",
            }
        )
    if "list_doctor_appointments" in task_names:
        doctor_name = plan_model.doctor_name or find_doctor_name_in_text(query)
        if doctor_name:
            appointment_list = list_appointments_for_doctor(doctor_name)
            tool_logs.append(
                {
                    "tool": "list_doctor_appointments",
                    "success": True,
                    "message": f"Found {len(appointment_list)} appointment(s) for {doctor_name}",
                }
            )
        else:
            active_doctors = active_appointment_doctors()
            tool_logs.append(
                {
                    "tool": "list_active_appointment_doctors",
                    "success": True,
                    "message": f"Found {len(active_doctors)} doctor(s) with active appointments",
                }
            )
    if "list_active_appointment_doctors" in task_names:
        active_doctors = active_appointment_doctors()
        tool_logs.append(
            {
                "tool": "list_active_appointment_doctors",
                "success": True,
                "message": f"Found {len(active_doctors)} doctor(s) with active appointments",
            }
        )
    if "list_doctors" in task_names:
        doctor_list = load_doctors()
        tool_logs.append(
            {
                "tool": "list_doctors",
                "success": True,
                "message": f"Found {len(doctor_list)} doctor(s) in the system",
            }
        )
    if "lookup_entity_details" in task_names:
        entity_matches = find_entity_matches(query, store)
        tool_logs.append(
            {
                "tool": "lookup_entity_details",
                "success": True,
                "message": f"Found {len(entity_matches)} matching patient/doctor entr{'y' if len(entity_matches) == 1 else 'ies'}",
            }
        )
    if "book_appointment" in task_names:
        requested_date = _extract_requested_date(query)
        try:
            specialty_decision = classify_specialty(query, available_specialties())
        except Exception:
            specialty_decision = None
        has_problem_context = bool(specialty_decision and specialty_decision.has_sufficient_symptom_context)
        specialty = specialty_decision.specialty if specialty_decision and specialty_decision.specialty else ""
        if not patient or not requested_date or not has_problem_context:
            missing = []
            if not patient:
                missing.append("patient")
            if not has_problem_context:
                missing.append("problem or symptoms")
            if not requested_date:
                missing.append("preferred date")
            appointment_workflow = {
                "status": "needs_details",
                "patient_name": patient.name if patient else "",
                "problem": query if has_problem_context else "",
                "specialty": specialty,
                "missing": missing,
                "message": "I need a few more details before booking an appointment.",
            }
            tool_logs.append({"tool": "appointment_intake", "success": True, "message": f"Asked for: {', '.join(missing)}"})
        else:
            availability = find_slots_for_specialty(specialty, requested_date)
            slot = (availability["exact_matches"] or availability["alternate_matches"] or [None])[0]
            if slot:
                appointment = book_specific_appointment(patient.name, slot["doctor_id"], slot["slot"])
            else:
                appointment = {"success": False, "message": f"No available slots for {specialty}."}
            tool_logs.append({"tool": "appointment_booking", "success": appointment.get("success", False), "message": appointment.get("message", "")})

    if "retrieve_patient_reports" in task_names:
        if patient:
            patient_reports = get_patient_reports(store, patient.name)
            tool_logs.append(
                {
                    "tool": "patient_reports",
                    "success": True,
                    "message": f"Found {patient_reports.get('count', 0)} stored report(s) for {patient.name}",
                }
            )
        else:
            tool_logs.append({"tool": "patient_reports", "success": False, "message": "No patient selected"})

    medical_info = {}
    if "retrieve_medical_information" in task_names:
        medical_info = search_medical_info(plan_model.condition_or_topic or query + " " + (patient.summary or "" if patient else ""))
        tool_logs.append({"tool": "medical_information_search", "success": True, "message": f"Matched {medical_info['condition']}"})

    sections = []
    patient_record_only_tasks = {"retrieve_patient_context", "answer_general", "final_response"}
    if patient and "retrieve_patient_context" in task_names and task_names <= patient_record_only_tasks:
            sections.append(answer_patient_record_question(query, patient_context(patient), conversation))
    elif needs_patient and not patient:
        sections.append("I could not find a matching patient in the records.")
    if appointment:
        if "deleted_count" in appointment:
            sections.append(f"Appointments:\n- {appointment.get('message')}")
        elif appointment.get("success"):
            appt = appointment["appointment"]
            sections.append(
                "Appointment:\n"
                f"- {appt['status']} with {appt['doctor_name']} ({appt['specialty']}) "
                f"on {appt['slot']} at {appt['location']}."
            )
        else:
            sections.append(f"Appointment:\n- {appointment.get('message')}")
    if "list_appointments" in task_names:
        if patient and appointment_list:
            rows = [
                f"- {item['slot']} with {item['doctor_name']} ({item['specialty']}) at {item['location']} - {item['status']}"
                for item in appointment_list
            ]
            sections.append("Appointments:\n" + "\n".join(rows))
        elif patient:
            sections.append(f"No appointments are currently booked for {patient.name}. Would you like to book one?")
    if "list_all_appointments" in task_names:
        if appointment_list:
            rows = [
                f"- {item['patient_name']}: {item['slot']} with {item['doctor_name']} ({item['specialty']}) at {item['location']} - {item['status']}"
                for item in appointment_list
            ]
            sections.append("All appointments:\n" + "\n".join(rows))
        else:
            sections.append("No appointments are currently booked for any patient.")
    if "list_doctor_appointments" in task_names:
        doctor_name = plan_model.doctor_name or find_doctor_name_in_text(query)
        if not doctor_name:
            if active_doctors:
                rows = [
                    f"- {doctor['doctor_name']} ({doctor['specialty']}) - {doctor['location']}: "
                    f"{doctor['appointment_count']} active appointment(s)"
                    for doctor in active_doctors
                ]
                sections.append("Doctors with active appointments:\n" + "\n".join(rows))
            else:
                sections.append("No doctors currently have active appointments.")
        elif appointment_list:
            rows = [
                f"- {item['patient_name']}: {item['slot']} ({item['specialty']}) at {item['location']} - {item['status']}"
                for item in appointment_list
            ]
            sections.append(f"Appointments for {doctor_name}:\n" + "\n".join(rows))
        else:
            sections.append(f"No appointments are currently booked for {doctor_name}.")
    if "list_active_appointment_doctors" in task_names:
        if active_doctors:
            rows = [
                f"- {doctor['doctor_name']} ({doctor['specialty']}) - {doctor['location']}: "
                f"{doctor['appointment_count']} active appointment(s)"
                for doctor in active_doctors
            ]
            sections.append("Doctors with active appointments:\n" + "\n".join(rows))
        else:
            sections.append("No doctors currently have active appointments.")
    if "list_doctors" in task_names:
        if doctor_list:
            rows = []
            for doctor in doctor_list:
                slots = doctor.get("available_slots", [])
                slot_text = ", ".join(slots) if slots else "No open slots"
                rows.append(
                    f"- {doctor['name']} ({doctor['specialty']}) - {doctor['location']}. "
                    f"Available slots: {slot_text}"
                )
            sections.append("Doctors in the system:\n" + "\n".join(rows))
        else:
            sections.append("No doctors are currently configured in the system.")
    if "lookup_entity_details" in task_names:
        if len(entity_matches) == 1:
            sections.append(format_entity_details(entity_matches[0]))
        elif len(entity_matches) > 1:
            sections.append(format_entity_choices(entity_matches))
        else:
            sections.append("I could not find a matching patient or doctor in the system.")
    if "retrieve_patient_reports" in task_names:
        if patient and patient_reports.get("reports"):
            rows = []
            for index, report in enumerate(patient_reports["reports"], start=1):
                report_lines = [
                    f"### Report {index}",
                    "",
                    "| Field | Value |",
                    "|---|---|",
                    f"| Source | {report['source']} |",
                    f"| Visit/upload date | {report['visit_date']} |",
                    f"| Diagnosis | {report['diagnosis']} |",
                ]
                for section_name, section_text in report.get("sections", {}).items():
                    if section_text and section_text != report["diagnosis"]:
                        report_lines.extend(["", f"**{section_name}**"])
                        report_lines.extend(f"- {item}" for item in _bulletize_text(section_text))
                rows.append("\n".join(report_lines))
            sections.append(f"## Stored medical reports/test results for {patient.name}\n\n" + "\n\n---\n\n".join(rows))
        elif patient:
            sections.append(
                f"I do not have stored medical reports or test results for {patient.name}. "
                "You can upload a PDF or Excel medical record from the Patient Records tab, and I will keep it in history."
            )
    if appointment_workflow:
        missing = appointment_workflow.get("missing", [])
        if "problem or symptoms" in missing:
            sections.append(
                "I can help book that. What problem or symptoms should I use? "
                "After that, I will suggest a doctor and show a calendar for the preferred date."
            )
        elif "preferred date" in missing and appointment_workflow.get("specialty"):
            sections.append(
                f"Based on the symptoms, I suggest a {appointment_workflow['specialty']}. "
                "Use the doctor and calendar selector below to choose a doctor and preferred date."
            )
        else:
            sections.append("I need a few more details before booking the appointment.")
    if medical_info:
        sources = "\n".join(f"- {source}" for source in medical_info["sources"])
        sections.append(f"Medical information: {medical_info['condition']}\n{medical_info['summary']}\nSources:\n{sources}")
    try:
        safety_triage = triage_safety(query)
    except Exception:
        safety_triage = None
    if safety_triage and safety_triage.needs_urgent_action_guidance:
        sections.append(f"Action: {safety_triage.guidance}")
    if not sections and plan_model.direct_answer:
        sections.append(plan_model.direct_answer)
    if not sections or ("answer_general" in task_names and not (task_names - {"answer_general", "final_response"})):
        context_blocks = [*sections]
        if patient:
            context_blocks.append(patient_context(patient))
        if selected_doctor:
            context_blocks.append(doctor_context(selected_doctor))
        sections = [general_answer_with_llm(query, "\n\n".join(context_blocks), conversation)]
    if patient and sections and not sections[0].startswith("Patient details:"):
        sections[0] = f"[{patient.name}] {sections[0]}"
    result = {
        "query": query,
        "plan": plan,
        "patient": patient.__dict__ if patient else None,
        "appointment": appointment,
        "appointment_list": appointment_list,
        "active_appointment_doctors": active_doctors,
        "doctor_list": doctor_list,
        "entity_matches": entity_matches,
        "appointment_workflow": appointment_workflow,
        "medical_info": medical_info,
        "patient_reports": patient_reports,
        "tool_logs": tool_logs,
        "final_answer": "\n\n".join(sections),
    }
    append_jsonl(AGENT_LOG_FILE, result)
    result["evaluation"] = evaluate_agent_run(query, result)
    return result


def _build_graph():
    """Create the two-node LangGraph pipeline: plan first, then execute tools."""
    if StateGraph is None:
        return None

    def planner_node(state: AgentState) -> AgentState:
        plan = plan_with_llm(
            state["query"],
            active_patient=state.get("selected_patient_name"),
            active_doctor=state.get("selected_doctor_name"),
            conversation=state.get("conversation", ""),
            scoped_context=state.get("scoped_context", ""),
        )
        return {"plan_model": plan, "plan": [step.model_dump() for step in plan.steps]}

    def tools_node(state: AgentState) -> AgentState:
        result = _run_agent_impl(
            state["query"],
            state.get("plan_model"),
            context_scope=state.get("context_scope", ""),
            selected_patient_name=state.get("selected_patient_name"),
            selected_doctor_name=state.get("selected_doctor_name"),
            conversation=state.get("conversation", ""),
        )
        return result

    graph = StateGraph(AgentState)
    graph.add_node("llm_planner", planner_node)
    graph.add_node("tool_executor", tools_node)
    graph.set_entry_point("llm_planner")
    graph.add_edge("llm_planner", "tool_executor")
    graph.add_edge("tool_executor", END)
    return graph.compile()


def _finalize_direct_answer(query: str, plan: list[dict[str, Any]], answer: str, **extra: Any) -> dict[str, Any]:
    result = {
        "query": query,
        "plan": plan,
        "patient": None,
        "appointment": {},
        "appointment_list": [],
        "active_appointment_doctors": [],
        "doctor_list": [],
        "entity_matches": [],
        "appointment_workflow": {},
        "medical_info": {},
        "patient_reports": {},
        "tool_logs": [],
        "final_answer": answer,
        **extra,
    }
    append_jsonl(AGENT_LOG_FILE, result)
    result["evaluation"] = evaluate_agent_run(query, result)
    return result


def run_agent(
    query: str,
    *,
    context_scope: str = "",
    selected_patient_name: str | None = None,
    selected_doctor_name: str | None = None,
    conversation: str = "",
) -> dict[str, Any]:
    """Public entrypoint used by the Streamlit chat UI."""
    if not has_openai_key():
        plan = [{"step": 1, "task": "final_response", "reason": "OpenAI API key is required for LLM-driven planning."}]
        result = {
            "query": query,
            "plan": plan,
            "patient": None,
            "appointment": {},
            "appointment_list": [],
            "active_appointment_doctors": [],
            "doctor_list": [],
            "entity_matches": [],
            "appointment_workflow": {},
            "medical_info": {},
            "patient_reports": {},
            "tool_logs": [{"tool": "llm_planner", "success": False, "message": "OPENAI_API_KEY is not configured"}],
            "final_answer": "I need an OpenAI API key configured before I can answer, because this version uses LLM-driven planning and evaluation for every request.",
        }
        result["evaluation"] = evaluate_agent_run(query, result)
        return result
    try:
        store, _ = build_runtime()
        if context_scope == CONTEXT_GENERIC:
            medical_info = {}
            tool_logs = []
            try:
                advice_intent = classify_general_symptom_advice_intent(query)
            except Exception:
                advice_intent = None
            if advice_intent and advice_intent.asks_for_general_symptom_advice:
                medical_info = search_medical_info(query)
                tool_logs.append(
                    {
                        "tool": "medical_information_search",
                        "success": True,
                        "message": f"Matched {medical_info['condition']}",
                    }
                )
            context = ""
            if medical_info:
                sources = "\n".join(f"- {source}" for source in medical_info["sources"])
                context = f"Medical information: {medical_info['condition']}\n{medical_info['summary']}\nSources:\n{sources}"
            answer = general_answer_with_llm(query, context, conversation)
            result = _finalize_direct_answer(
                query,
                [{"step": 1, "task": "answer_general", "reason": "Generic context selected by LLM router."}],
                answer,
                medical_info=medical_info,
                tool_logs=tool_logs,
            )
            return result

        if context_scope == CONTEXT_SYSTEM:
            context = system_context(store)
            answer = answer_scoped_context_question(query, CONTEXT_SYSTEM, context, conversation)
            return _finalize_direct_answer(
                query,
                [{"step": 1, "task": "answer_general", "reason": "System context selected by LLM router."}],
                answer,
            )

        scoped_context = ""
        if context_scope == CONTEXT_PATIENT and selected_patient_name:
            selected_patient = store.get_patient_by_name(selected_patient_name)
            if selected_patient:
                scoped_context = patient_context(selected_patient)
        if context_scope == CONTEXT_DOCTOR and selected_doctor_name:
            selected_doctor = _doctor_by_name(selected_doctor_name)
            if selected_doctor:
                scoped_context = doctor_context(selected_doctor)

        graph = _build_graph()
        if graph is None:
            return _run_agent_impl(
                query,
                context_scope=context_scope,
                selected_patient_name=selected_patient_name,
                selected_doctor_name=selected_doctor_name,
                conversation=conversation,
            )
        final_state = graph.invoke(
            {
                "query": query,
                "context_scope": context_scope,
                "selected_patient_name": selected_patient_name,
                "selected_doctor_name": selected_doctor_name,
                "conversation": conversation,
                "scoped_context": scoped_context,
            }
        )
        return final_state
    except Exception as exc:
        return {
            "query": query,
            "plan": [{"step": 1, "task": "final_response", "reason": "LLM workflow failed"}],
            "patient": None,
            "appointment": {},
            "appointment_list": [],
            "active_appointment_doctors": [],
            "doctor_list": [],
            "entity_matches": [],
            "appointment_workflow": {},
            "medical_info": {},
            "patient_reports": {},
            "tool_logs": [{"tool": "llm_graph", "success": False, "message": str(exc)}],
            "final_answer": f"I could not produce a reliable LLM response for this request. Reason: {exc}",
            "evaluation": {"score": 0, "critique": str(exc)},
        }
