from __future__ import annotations

import re
from typing import Any

from .patient_store import PatientStore


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def report_sections(text: str) -> dict[str, str]:
    label_patterns = {
        "Diagnosis": r"Diagnosis",
        "Subjective": r"Subjective(?:\s+Notes)?",
        "Objective": r"Objective(?:\s+Notes)?",
        "Assessment": r"Assessment(?:\s+Notes)?",
        "Plan": r"Plan(?:\s+Notes)?",
    }
    boundary = "|".join(label_patterns.values())
    sections: dict[str, str] = {}
    for label, label_pattern in label_patterns.items():
        pattern = rf"{label_pattern}\s*:\s*(.*?)(?=\s+(?:{boundary})\s*:|$)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            sections[label] = _compact(match.group(1))
    if not sections and text:
        sections["Findings"] = _compact(text)
    return sections


def _matching_records(store: PatientStore, patient_name: str):
    name_l = patient_name.lower()
    return [
        record
        for record in store.list_patients()
        if record.name.lower() == name_l or name_l in record.name.lower() or record.name.lower() in name_l
    ]


def _source_documents(record) -> list[dict[str, Any]]:
    metadata = record.metadata or {}
    documents = metadata.get("source_documents") or []
    if documents:
        return documents
    return [
        {
            "patient_name": record.name,
            "source": record.source,
            "visit_date": metadata.get("visit_date") or metadata.get("uploaded_at") or "Unknown",
            "diagnosis": metadata.get("diagnosis") or "Not specified",
            "summary": record.summary,
            "raw_text": record.raw_text,
            "metadata": metadata,
        }
    ]


def _report_text(document: dict[str, Any]) -> str:
    parts = []
    metadata = document.get("metadata") or {}
    if metadata.get("visit_date"):
        parts.append(f"Visit date: {metadata['visit_date']}.")
    if metadata.get("diagnosis"):
        parts.append(f"Diagnosis: {metadata['diagnosis']}.")
    summary = document.get("summary")
    raw_text = document.get("raw_text")
    if summary:
        parts.append(summary)
    if raw_text and not summary:
        parts.append(raw_text)
    return _compact(" ".join(parts))


def get_patient_reports(store: PatientStore, patient_name: str) -> dict[str, Any]:
    reports = []
    for record in _matching_records(store, patient_name):
        for document in _source_documents(record):
            text = _report_text(document)
            section_source = document.get("raw_text") or text
            if not text:
                continue
            source = document.get("source") or record.source
            source_l = source.lower()
            metadata = document.get("metadata") or {}
            looks_like_report = "report" in source_l or "pdf" in source_l or metadata.get("record_type") == "uploaded_medical_record"
            if not looks_like_report:
                continue
            reports.append(
                {
                    "patient_name": document.get("name") or record.name,
                    "source": source,
                    "visit_date": metadata.get("visit_date") or metadata.get("uploaded_at") or "Unknown",
                    "diagnosis": metadata.get("diagnosis") or "Not specified",
                    "summary": text[:1600],
                    "sections": report_sections(section_source[:2400]),
                }
            )

    return {
        "patient_name": patient_name,
        "reports": reports,
        "count": len(reports),
    }
