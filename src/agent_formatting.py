from __future__ import annotations

from typing import Any


def format_entity_details(entity: dict[str, Any]) -> str:
    """Format one patient or doctor match for display in chat responses."""
    if entity["type"] == "doctor":
        slots = entity.get("available_slots", [])
        slot_text = ", ".join(slots) if slots else "No open slots"
        return (
            "Doctor details:\n"
            f"- Name: {entity['name']}\n"
            f"- Specialty: {entity['specialty']}\n"
            f"- Location: {entity['location']}\n"
            f"- Available slots: {slot_text}"
        )

    details = [
        f"Name: {entity['name']}",
        f"Age: {entity.get('age') or 'Unknown'}",
        f"Gender: {entity.get('gender') or 'Unknown'}",
        f"Phone: {entity.get('phone') or 'Unknown'}",
        f"Email: {entity.get('email') or 'Unknown'}",
        f"Address: {entity.get('address') or 'Unknown'}",
        f"Source: {entity.get('source') or 'Unknown'}",
    ]
    if entity.get("summary"):
        details.append(f"Summary: {entity['summary']}")
    return "Patient details:\n" + "\n".join(f"- {item}" for item in details)


def format_entity_choices(matches: list[dict[str, Any]]) -> str:
    """Format ambiguous patient/doctor matches for a clarification response."""
    rows = []
    for entity in matches:
        if entity["type"] == "doctor":
            rows.append(f"- Doctor: {entity['name']} ({entity['specialty']}, {entity['location']})")
        else:
            rows.append(
                f"- Patient: {entity['name']} "
                f"({entity.get('age') or 'age unknown'}, {entity.get('source') or 'source unknown'})"
            )
    return "I found multiple matches. Please specify which one you mean:\n" + "\n".join(rows)
