from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import APPOINTMENTS_FILE, DOCTOR_SCHEDULE_FILE


def ensure_schedule(path: Path = DOCTOR_SCHEDULE_FILE) -> None:
    if not path.exists():
        path.write_text("[]", encoding="utf-8")
    if not APPOINTMENTS_FILE.exists():
        APPOINTMENTS_FILE.write_text("[]", encoding="utf-8")


def load_doctors() -> list[dict[str, Any]]:
    ensure_schedule()
    return json.loads(DOCTOR_SCHEDULE_FILE.read_text(encoding="utf-8"))


def save_doctors(doctors: list[dict[str, Any]]) -> None:
    DOCTOR_SCHEDULE_FILE.write_text(json.dumps(doctors, indent=2), encoding="utf-8")


def load_appointments() -> list[dict[str, Any]]:
    ensure_schedule()
    return json.loads(APPOINTMENTS_FILE.read_text(encoding="utf-8"))


def list_appointments_for_patient(patient_name: str) -> list[dict[str, Any]]:
    patient_l = patient_name.lower()
    return [item for item in load_appointments() if item.get("patient_name", "").lower() == patient_l]


def list_all_appointments() -> list[dict[str, Any]]:
    return load_appointments()


def list_appointments_for_doctor(doctor_query: str) -> list[dict[str, Any]]:
    doctor_l = doctor_query.lower().replace("dr.", "").replace("doctor", "").strip()
    matches = []
    for item in load_appointments():
        doctor_name = item.get("doctor_name", "")
        normalized = doctor_name.lower().replace("dr.", "").replace("doctor", "").strip()
        if doctor_l in normalized or normalized in doctor_l:
            matches.append(item)
    return matches


def find_doctor_name_in_text(text: str) -> str | None:
    text_l = text.lower()
    for doctor in load_doctors():
        name = doctor["name"]
        normalized_name = name.lower().replace("dr.", "").replace("doctor", "").strip()
        if name.lower() in text_l or normalized_name in text_l:
            return name
        name_parts = normalized_name.split()
        if any(len(part) > 2 and part in text_l for part in name_parts):
            return name
    return None


def clear_appointments() -> dict[str, Any]:
    appointments = load_appointments()
    deleted_count = len(appointments)
    APPOINTMENTS_FILE.write_text("[]", encoding="utf-8")
    return {
        "success": True,
        "message": f"Deleted {deleted_count} current appointment{'s' if deleted_count != 1 else ''}.",
        "deleted_count": deleted_count,
    }


def find_doctors_by_specialty(specialty: str) -> list[dict[str, Any]]:
    specialty_l = specialty.lower()
    return [doctor for doctor in load_doctors() if specialty_l in doctor["specialty"].lower()]


def available_specialties() -> list[str]:
    specialties = []
    for doctor in load_doctors():
        specialty = doctor.get("specialty")
        if specialty and specialty not in specialties:
            specialties.append(specialty)
    return specialties


def find_slots_for_specialty(specialty: str, preferred_date: str | None = None) -> dict[str, Any]:
    doctors = find_doctors_by_specialty(specialty) or find_doctors_by_specialty("General Physician")
    exact_matches = []
    alternate_matches = []

    for doctor in doctors:
        for slot in doctor.get("available_slots", []):
            slot_item = {
                "doctor_id": doctor["doctor_id"],
                "doctor_name": doctor["name"],
                "specialty": doctor["specialty"],
                "location": doctor["location"],
                "slot": slot,
            }
            if preferred_date and slot.startswith(preferred_date):
                exact_matches.append(slot_item)
            else:
                alternate_matches.append(slot_item)

    return {
        "specialty": specialty,
        "preferred_date": preferred_date,
        "exact_matches": exact_matches,
        "alternate_matches": alternate_matches,
    }


def book_specific_appointment(patient_name: str, doctor_id: str, slot: str) -> dict[str, Any]:
    doctor = next((item for item in load_doctors() if item["doctor_id"] == doctor_id), None)
    if not doctor:
        return {"success": False, "message": "Selected doctor was not found."}
    if slot not in doctor.get("available_slots", []):
        return {"success": False, "message": "Selected slot is no longer available."}

    appointments = load_appointments()
    appointment = {
        "appointment_id": f"A{len(appointments) + 1:04d}",
        "patient_name": patient_name,
        "doctor_id": doctor["doctor_id"],
        "doctor_name": doctor["name"],
        "specialty": doctor["specialty"],
        "location": doctor["location"],
        "slot": slot,
        "status": "Booked",
    }
    appointments.append(appointment)
    APPOINTMENTS_FILE.write_text(json.dumps(appointments, indent=2), encoding="utf-8")
    return {"success": True, "message": "Appointment booked successfully.", "appointment": appointment}


def book_appointment(patient_name: str, specialty: str, preferred_slot: str | None = None) -> dict[str, Any]:
    doctors = find_doctors_by_specialty(specialty) or find_doctors_by_specialty("General Physician")
    if not doctors:
        return {"success": False, "message": f"No doctors available for {specialty}."}

    doctor = doctors[0]
    slots = doctor.get("available_slots", [])
    slot = preferred_slot if preferred_slot in slots else slots[0] if slots else None
    if not slot:
        return {"success": False, "message": f"No open slots for {doctor['name']}."}

    return book_specific_appointment(patient_name, doctor["doctor_id"], slot)
