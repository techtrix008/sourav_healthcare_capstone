from __future__ import annotations

import re
from io import BytesIO

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from .agent_graph import build_runtime, clear_runtime_cache


def dataframe_to_excel_bytes(rows) -> bytes:
    """Serialize table-like rows for Streamlit Excel download buttons."""
    output = BytesIO()
    pd.DataFrame(rows).to_excel(output, index=False)
    return output.getvalue()


@st.cache_data(show_spinner=False)
def load_runtime_snapshot():
    """Return cached patient records and retrieval document count for UI rendering."""
    store, vector_store = build_runtime()
    return store.list_patients(), len(vector_store.documents)


def refresh_runtime_data() -> None:
    """Clear both agent runtime and Streamlit data caches after persisted data changes."""
    clear_runtime_cache()
    load_runtime_snapshot.clear()


def patient_rows() -> list[dict]:
    patients, _ = load_runtime_snapshot()
    return [
        {
            "Patient ID": patient.patient_id,
            "Name": patient.name,
            "Age": patient.age,
            "Gender": patient.gender,
            "Phone": patient.phone,
            "Address": patient.address,
            "Source": patient.source,
            "Linked Records": patient.metadata.get("merged_record_count", 1),
            "Summary": patient.summary,
        }
        for patient in patients
    ]


def unique_patient_names() -> list[str]:
    names = []
    for row in patient_rows():
        if row["Name"] not in names:
            names.append(row["Name"])
    return names


def patient_by_name(name: str | None):
    if not name:
        return None
    for row in patient_rows():
        if row["Name"] == name:
            return row
    return None


def patient_by_id(patient_id: str | None):
    if not patient_id:
        return None
    for row in patient_rows():
        if row["Patient ID"] == patient_id:
            return row
    return None


def extract_uploaded_pdf_text(uploaded_file) -> str:
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def patient_name_from_text(text: str) -> str | None:
    match = re.search(r"Patient:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    for name in unique_patient_names():
        if name.lower() in text.lower():
            return name
    return None
