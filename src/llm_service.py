from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import List, Literal, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


class PlanStep(BaseModel):
    step: int = Field(description="1-based execution order")
    task: Literal[
        "retrieve_patient_context",
        "list_appointments",
        "list_all_appointments",
        "list_doctor_appointments",
        "list_active_appointment_doctors",
        "list_doctors",
        "lookup_entity_details",
        "book_appointment",
        "clear_appointments",
        "retrieve_medical_information",
        "retrieve_patient_reports",
        "answer_general",
        "final_response",
    ]
    reason: str


class AgentPlan(BaseModel):
    needs_patient: bool
    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None
    condition_or_topic: Optional[str] = None
    direct_answer: Optional[str] = None
    steps: List[PlanStep]


class AgentEvaluation(BaseModel):
    relevance: float = Field(ge=0, le=1)
    groundedness: float = Field(ge=0, le=1)
    tool_success: float = Field(ge=0, le=1)
    safety: float = Field(ge=0, le=1)
    critique: str


class ContextDecision(BaseModel):
    scope: Literal["generic", "system", "patient", "doctor"]
    needs_context_selection: bool
    selected_patient_name: Optional[str] = None
    selected_doctor_name: Optional[str] = None
    ambiguous_patient_names: List[str] = Field(default_factory=list)
    ambiguous_doctor_names: List[str] = Field(default_factory=list)
    contextual_query: str
    reason: str


class ReportIntent(BaseModel):
    asks_for_stored_reports: bool
    reason: str


class EntityLookupIntent(BaseModel):
    asks_for_entity_lookup: bool
    reason: str


class GeneralSymptomAdviceIntent(BaseModel):
    asks_for_general_symptom_advice: bool
    reason: str


class PatientRecordQuestionIntent(BaseModel):
    asks_patient_record_question: bool
    reason: str


class SpecialtyDecision(BaseModel):
    has_sufficient_symptom_context: bool
    specialty: Optional[str] = None
    reason: str


class SafetyTriage(BaseModel):
    needs_urgent_action_guidance: bool
    guidance: str
    reason: str


class AgentIntentClassification(BaseModel):
    """Grouped intent flags used by the agent planner/normalizer."""

    report_intent: bool = Field(description="User asks for stored reports, labs, tests, or medical records.")
    patient_record_intent: bool = Field(description="User asks a targeted question about patient clinical history, stored notes, symptoms, diagnosis, medication, or interactions.")
    entity_lookup_intent: bool = Field(description="User asks who/show details for a named system entity.")
    general_symptom_intent: bool = Field(description="User asks general symptom guidance without a patient record request.")
    safety_flag: bool = Field(description="User message may need urgent safety guidance.")
    safety_guidance: str = Field(default="", description="Concise urgent-action guidance when safety_flag is true.")
    confidence: float = Field(ge=0, le=1, description="Overall classification confidence.")
    reason: str


class NextBestAction(BaseModel):
    label: str
    suggested_prompt: str
    rationale: str
    priority: Literal["high", "medium", "low"] = "medium"


class NextBestActions(BaseModel):
    has_suggestions: bool
    actions: List[NextBestAction] = Field(default_factory=list)
    reason: str


def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def get_llm(temperature: float = 0) -> ChatOpenAI:
    """Create the configured OpenAI chat model and fail clearly when no key exists."""
    if not has_openai_key():
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), temperature=temperature)


def plan_with_llm(
    query: str,
    active_patient: Optional[str] = None,
    active_doctor: Optional[str] = None,
    conversation: str = "",
    scoped_context: str = "",
) -> AgentPlan:
    """Return the structured tool plan that drives each LangGraph turn."""
    llm = get_llm().with_structured_output(AgentPlan)
    active_patient_context = active_patient or "none"
    active_doctor_context = active_doctor or "none"
    return llm.invoke(
        [
            (
                "system",
                "You are the planning brain for an agentic healthcare administration assistant. "
                "Classify the user's request into tool steps. Use patient-specific steps only when "
                "the query truly needs a patient. Global appointment queries such as 'all appointments "
                "in the system' do not need a patient. Doctor appointment queries need doctor_name, "
                "not patient_name. Requests to show doctors, providers, physicians, clinicians, or specialists "
                "in the system should use list_doctors, not list_doctor_appointments. Requests for test results, labs, investigations, uploaded reports, "
                "or medical records should use retrieve_patient_reports, not retrieve_medical_information. "
                "Requests for doctors who currently have booked/active appointments should use list_active_appointment_doctors. "
                "Requests like 'who is X', 'show details for X', or 'tell me about X' should use lookup_entity_details "
                "so the agent can check both patient records and the doctor directory. "
                "Use the recent conversation and supplied scoped context to avoid asking for details already provided. "
                "Do not invent missing patient or doctor names. Always include final_response as the last step.",
            ),
            (
                "user",
                f"Active patient context: {active_patient_context}\n"
                f"Active doctor context: {active_doctor_context}\n"
                f"Recent conversation:\n{conversation or 'none'}\n\n"
                f"Scoped context:\n{scoped_context or 'none'}\n\n"
                f"User query: {query}",
            ),
        ]
    )


def classify_query_context(
    query: str,
    patient_names: List[str],
    doctor_names: Optional[List[str]] = None,
    active_patient: Optional[str] = None,
    active_doctor: Optional[str] = None,
    conversation: str = "",
) -> ContextDecision:
    """Classify whether the UI should use general, global, patient, or doctor context."""
    llm = get_llm().with_structured_output(ContextDecision)
    active_patient_context = active_patient or "none"
    active_doctor_context = active_doctor or "none"
    patient_list = ", ".join(patient_names) if patient_names else "none"
    doctor_list = ", ".join(doctor_names or []) if doctor_names else "none"
    return llm.invoke(
        [
            (
                "system",
                "You are the only context router for a healthcare assistant. Use the current chat history "
                "and semantic intent to choose exactly one scope. "
                "scope='generic': questions outside the local healthcare system, including weather, time, greetings, "
                "and general medical education not tied to a stored patient or doctor. "
                "scope='system': questions about all information currently stored in this application, including all "
                "patients, histories, interactions, doctors, appointments, reports, or whole-system counts/summaries. "
                "scope='patient': questions or actions about exactly one stored patient. "
                "scope='doctor': questions or actions about exactly one stored doctor. "
                "Never select or ask for a patient or doctor when scope is generic or system. "
                "For patient scope, use only supplied patient names. For doctor scope, use only supplied doctor names. "
                "When the user refers to the active patient or active doctor through continuation phrasing, keep that context. "
                "If a required patient or doctor is missing or ambiguous, set needs_context_selection=true and return the "
                "best ambiguous names. Do not invent names, facts, appointments, or records.",
            ),
            (
                "user",
                f"Active patient: {active_patient_context}\n"
                f"Active doctor: {active_doctor_context}\n"
                f"Known patients: {patient_list}\n"
                f"Known doctors: {doctor_list}\n"
                f"Recent conversation:\n{conversation or 'none'}\n\n"
                f"User query: {query}",
            ),
        ]
    )


@lru_cache(maxsize=128)
def classify_agent_intents(
    query: str,
    context_scope: str = "",
    selected_patient_name: str = "",
    selected_doctor_name: str = "",
    conversation: str = "",
) -> AgentIntentClassification:
    """Classify all agent-normalization intents in one cached LLM call."""
    llm = get_llm().with_structured_output(AgentIntentClassification)
    return llm.invoke(
        [
            (
                "system",
                "You are a multi-intent classifier for a healthcare administration assistant. "
                "Classify all intent flags in one response. Be strict: report_intent is only for stored "
                "reports, lab results, test results, investigations, or medical records from the local system. "
                "patient_record_intent is for targeted questions about one patient's stored clinical history, symptoms, "
                "diagnosis, medication, notes, or interactions, but not when the user explicitly asks for reports/tests "
                "and not for appointment listing or appointment booking actions. "
                "entity_lookup_intent is for identifying or showing details for a named patient, doctor, provider, "
                "or local system entity. general_symptom_intent is for general health guidance that is not asking "
                "to retrieve a patient's records and not asking to book an appointment. safety_flag is true for "
                "possible emergencies or red flags. Never invent local patient or doctor names.",
            ),
            (
                "user",
                f"Scope: {context_scope or 'unknown'}\n"
                f"Selected patient: {selected_patient_name or 'none'}\n"
                f"Selected doctor: {selected_doctor_name or 'none'}\n"
                f"Recent conversation:\n{conversation or 'none'}\n\n"
                f"Query: {query}",
            ),
        ]
    )


def general_answer_with_llm(query: str, context: str = "", conversation: str = "") -> str:
    llm = get_llm(temperature=0.2)
    now = datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")
    response = llm.invoke(
        [
            (
                "system",
                "You are a helpful healthcare administration assistant. Answer clearly from the supplied "
                "context and recent conversation when relevant. If you lack enough information, say what is missing. Do not diagnose or "
                "replace a clinician. If the user asks for the current time, use the supplied current "
                "system time exactly. Never invent facts.",
            ),
            (
                "user",
                f"Current system time: {now}\n\n"
                f"Recent conversation:\n{conversation or 'none'}\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{query}",
            ),
        ]
    )
    return response.content


def answer_patient_record_question(query: str, patient_context: str, conversation: str = "") -> str:
    """Answer targeted patient-record questions without dumping the full chart."""
    llm = get_llm(temperature=0)
    response = llm.invoke(
        [
            (
                "system",
                "You answer questions about one patient's stored records, appointments, doctor directory, "
                "and current session history. Use only the supplied patient context and recent conversation. Answer only what the user asked for; do not dump "
                "the full patient profile unless explicitly requested. If the user asks for the latest, last, "
                "or most recent item, infer that from timestamps or source order in the supplied context. "
                "If the requested information is not present, say that it is not available in the stored history. "
                "Be concise and preserve clinically relevant wording from the record. Never invent missing facts. Do not diagnose.",
            ),
            (
                "user",
                f"Recent conversation:\n{conversation or 'none'}\n\n"
                f"Patient context and linked records:\n{patient_context}\n\n"
                f"Question:\n{query}",
            ),
        ]
    )
    return response.content


def answer_scoped_context_question(query: str, scope: str, scoped_context: str, conversation: str = "") -> str:
    """Answer from a supplied system, patient, or doctor context without inventing facts."""
    llm = get_llm(temperature=0)
    response = llm.invoke(
        [
            (
                "system",
                "You answer healthcare administration questions using only the supplied scoped context "
                "and recent conversation. If the context does not contain the answer, say that clearly. "
                "Do not invent system records, patient facts, doctor facts, appointments, reports, or prior interactions. "
                "Do not provide diagnosis; keep medical guidance educational and safety-conscious.",
            ),
            (
                "user",
                f"Scope: {scope}\n\n"
                f"Recent conversation:\n{conversation or 'none'}\n\n"
                f"Scoped context:\n{scoped_context}\n\n"
                f"Question:\n{query}",
            ),
        ]
    )
    return response.content


def classify_patient_record_question_intent(query: str) -> PatientRecordQuestionIntent:
    llm = get_llm().with_structured_output(PatientRecordQuestionIntent)
    return llm.invoke(
        [
            (
                "system",
                "Decide whether the user is asking a targeted question about a patient's stored record, "
                "summary, session history, symptoms, advice, treatment notes, appointments, or other patient-specific "
                "history. Return true when the assistant should answer from patient records only and should not "
                "list reports or dump the full profile unless explicitly requested. Return false when the user asks "
                "for test results, lab results, investigations, medical reports, uploaded reports, or all reports; "
                "those should use the stored report retrieval path instead.",
            ),
            ("user", query),
        ]
    )


def classify_report_intent(query: str) -> ReportIntent:
    llm = get_llm().with_structured_output(ReportIntent)
    return llm.invoke(
        [
            (
                "system",
                "Decide whether the user is asking for stored patient reports, uploaded reports, "
                "lab results, test results, investigations, or medical records from the local system. "
                "Return true only for stored-report retrieval. Return false for appointment queries, "
                "doctor queries, patient demographic/profile questions, and general medical education, "
                "even if a patient context exists.",
            ),
            ("user", query),
        ]
    )


def classify_entity_lookup_intent(query: str) -> EntityLookupIntent:
    llm = get_llm().with_structured_output(EntityLookupIntent)
    return llm.invoke(
        [
            (
                "system",
                "Decide whether the user is asking to identify or show details for a named person, "
                "doctor, patient, provider, or entity in the local system.",
            ),
            ("user", query),
        ]
    )


def classify_general_symptom_advice_intent(query: str) -> GeneralSymptomAdviceIntent:
    llm = get_llm().with_structured_output(GeneralSymptomAdviceIntent)
    return llm.invoke(
        [
            (
                "system",
                "Decide whether the user is asking for general health guidance about symptoms, "
                "signs, conditions, precautions, or what to do next, without asking to retrieve "
                "a specific patient's records and without explicitly asking to book an appointment. "
                "Return true for queries like 'I have knee pain, what should I do?' even if no "
                "patient context exists. Return false for patient-record requests or booking requests.",
            ),
            ("user", query),
        ]
    )


def classify_specialty(query: str, specialties: List[str]) -> SpecialtyDecision:
    llm = get_llm().with_structured_output(SpecialtyDecision)
    specialty_list = ", ".join(specialties) if specialties else "General Physician"
    return llm.invoke(
        [
            (
                "system",
                "You route appointment requests to the most relevant available specialty. "
                "Use only one specialty from the supplied list. Determine whether the user gave "
                "enough symptom/problem context to choose a specialty. Do not diagnose.",
            ),
            ("user", f"Available specialties: {specialty_list}\nRequest: {query}"),
        ]
    )


def triage_safety(query: str) -> SafetyTriage:
    llm = get_llm().with_structured_output(SafetyTriage)
    return llm.invoke(
        [
            (
                "system",
                "You are a safety triage classifier for a healthcare admin assistant. "
                "Decide whether the message needs urgent action guidance. Include concise, "
                "non-diagnostic guidance and suggest emergency care when red flags may be present.",
            ),
            ("user", query),
        ]
    )


def suggest_next_best_actions(
    patient_context: str,
    conversation: str,
    last_result_summary: str = "",
) -> NextBestActions:
    """Suggest compact follow-up actions from patient context and recent conversation."""
    llm = get_llm(temperature=0.1).with_structured_output(NextBestActions)
    return llm.invoke(
        [
            (
                "system",
                "You are a patient-session copilot for a healthcare administration assistant. "
                "Review the active patient context, the current chat session, and the latest tool result. "
                "Suggest only useful next best actions the assistant can take now, such as booking an appointment, "
                "checking appointments, uploading/reviewing reports, asking for missing symptoms, retrieving test results, "
                "summarizing history, searching trusted medical information, or adding a note to history. "
                "Do not suggest closing the session. Do not diagnose. If there is no meaningful next action, "
                "return has_suggestions=false. Limit to at most 3 concise actions. Each suggested_prompt should be "
                "a message based out of the existing tools in the systemthe healthcare administration "
                "assistant could send in chat that invokes the tool.",
            ),
            (
                "user",
                f"Patient context:\n{patient_context}\n\n"
                f"Current conversation:\n{conversation}\n\n"
                f"Latest result/tool summary:\n{last_result_summary}",
            ),
        ]
    )


def evaluate_with_llm(query: str, answer: str, tool_summary: str) -> AgentEvaluation:
    llm = get_llm().with_structured_output(AgentEvaluation)
    return llm.invoke(
        [
            (
                "system",
                "Evaluate the assistant response for a healthcare admin task. Score relevance, "
                "groundedness, tool_success, and safety from 0 to 1. Be strict but fair.",
            ),
            ("user", f"Query: {query}\nAnswer: {answer}\nTool summary: {tool_summary}"),
        ]
    )
