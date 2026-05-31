from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from .models import PatientRecord


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None


def _age_from_dob_or_text(text: str) -> int | None:
    age = _first_match(r"\b(\d{1,3})[- ]year[- ]old\b", text)
    if age:
        return int(age)
    return None


def _section(text: str, start: str, end: str | None = None) -> str | None:
    if end:
        pattern = rf"{start}\s*:\s*(.*?)(?=\n{end}\s*:)"
    else:
        pattern = rf"{start}\s*:\s*(.*)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else None


def parse_patient_pdf(path: Path) -> PatientRecord:
    text = extract_pdf_text(path)
    name = _first_match(r"Patient:\s*(.+)", text) or text.splitlines()[0].strip()
    dob = _first_match(r"DOB:\s*([0-9/]+)", text)
    gender = _first_match(r"Gender:\s*(.+)", text)
    phone = _first_match(r"Phone:\s*(.+)", text)
    address = _first_match(r"Address:\s*(.+)", text)
    visit_date = _first_match(r"Visit Date:\s*(.+)", text)
    diagnosis = _first_match(r"Diagnosis:\s*(.+)", text)

    subjective = _section(text, "Subjective Notes", "Objective Notes")
    objective = _section(text, "Objective Notes", "Assessment Notes")
    assessment = _section(text, "Assessment Notes", "Plan Notes")
    plan = _section(text, "Plan Notes")

    summary_parts = []
    if diagnosis:
        summary_parts.append(f"Diagnosis: {diagnosis}.")
    if subjective:
        summary_parts.append(f"Subjective: {subjective}")
    if assessment and assessment not in summary_parts:
        summary_parts.append(f"Assessment: {assessment}")
    if plan:
        summary_parts.append(f"Plan: {plan}")

    summary = " ".join(summary_parts).strip() or text[:1200]

    patient_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return PatientRecord(
        patient_id=f"pdf-{patient_id}",
        name=name,
        age=_age_from_dob_or_text(text),
        gender=gender,
        phone=phone,
        address=address,
        summary=summary,
        raw_text=text,
        source=path.name,
        metadata={"dob": dob, "visit_date": visit_date, "diagnosis": diagnosis},
    )


def parse_patient_pdfs(paths: list[Path]) -> list[PatientRecord]:
    records = []
    for path in paths:
        if path.exists():
            records.append(parse_patient_pdf(path))
    return records
