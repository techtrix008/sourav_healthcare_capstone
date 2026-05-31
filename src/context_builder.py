from __future__ import annotations

import json
from typing import Any

from .appointment_service import load_appointments, load_doctors
from .patient_store import PatientStore


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, indent=2)


def patient_context(patient) -> str:
    """Build the full context block for one patient, including linked history."""
    documents = (patient.metadata or {}).get("source_documents", [])
    data = {
        "patient": {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "phone": patient.phone,
            "email": patient.email,
            "address": patient.address,
            "summary": patient.summary,
            "source": patient.source,
        },
        "linked_records_and_history": documents,
        "appointments": [
            appointment
            for appointment in load_appointments()
            if appointment.get("patient_name", "").lower() == patient.name.lower()
        ],
        "doctor_directory": load_doctors(),
    }
    return _compact_json(data)


def doctor_context(doctor: dict[str, Any]) -> str:
    """Build the full context block for one doctor, including appointments."""
    doctor_name = doctor.get("name", "")
    doctor_id = doctor.get("doctor_id")
    data = {
        "doctor": doctor,
        "appointments": [
            appointment
            for appointment in load_appointments()
            if appointment.get("doctor_id") == doctor_id
            or appointment.get("doctor_name", "").lower() == doctor_name.lower()
        ],
    }
    return _compact_json(data)


def system_context(store: PatientStore) -> str:
    """Build the system-wide context available to system-scoped answers."""
    data = {
        "patients": [
            {
                "patient_id": patient.patient_id,
                "name": patient.name,
                "age": patient.age,
                "gender": patient.gender,
                "phone": patient.phone,
                "email": patient.email,
                "address": patient.address,
                "summary": patient.summary,
                "source": patient.source,
                "linked_records_and_history": (patient.metadata or {}).get("source_documents", []),
            }
            for patient in store.list_patients()
        ],
        "doctor_directory": load_doctors(),
        "appointments": load_appointments(),
    }
    return _compact_json(data)
