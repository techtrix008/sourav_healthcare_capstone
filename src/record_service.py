from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from datetime import datetime

from .config import USER_RECORDS_FILE
from .models import PatientRecord


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _load_raw(path: Path = USER_RECORDS_FILE) -> list[dict[str, Any]]:
    if not path.exists():
        path.write_text("[]", encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_raw(rows: list[dict[str, Any]], path: Path = USER_RECORDS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def load_user_records(path: Path = USER_RECORDS_FILE) -> list[PatientRecord]:
    records = []
    for row in _load_raw(path):
        records.append(
            PatientRecord(
                patient_id=row["patient_id"],
                name=row["name"],
                age=row.get("age"),
                gender=row.get("gender"),
                phone=row.get("phone"),
                email=row.get("email"),
                address=row.get("address"),
                summary=row.get("summary"),
                raw_text=row.get("raw_text"),
                source="user_records.json",
                metadata=row.get("metadata", {}),
            )
        )
    return records


def add_user_record(
    name: str,
    age: int | None,
    gender: str | None,
    phone: str | None,
    email: str | None,
    address: str | None,
    summary: str,
    path: Path = USER_RECORDS_FILE,
) -> PatientRecord:
    rows = _load_raw(path)
    patient_id = f"user-{_slug(name)}-{len(rows) + 1:04d}"
    row = {
        "patient_id": patient_id,
        "name": name,
        "age": age,
        "gender": gender,
        "phone": phone,
        "email": email,
        "address": address,
        "summary": summary,
        "raw_text": summary,
        "metadata": {"record_type": "manual"},
    }
    rows.append(row)
    _save_raw(rows, path)
    return load_user_records(path)[-1]


def add_user_medical_record(
    name: str,
    summary: str,
    raw_text: str,
    source: str,
    age: int | None = None,
    gender: str | None = None,
    phone: str | None = None,
    email: str | None = None,
    address: str | None = None,
    metadata: dict[str, Any] | None = None,
    path: Path = USER_RECORDS_FILE,
) -> PatientRecord:
    rows = _load_raw(path)
    patient_id = f"user-medical-{_slug(name)}-{len(rows) + 1:04d}"
    record_metadata = {
        "record_type": "uploaded_medical_record",
        "source_file": source,
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    if metadata:
        record_metadata.update(metadata)
    row = {
        "patient_id": patient_id,
        "name": name,
        "age": age,
        "gender": gender,
        "phone": phone,
        "email": email,
        "address": address,
        "summary": summary,
        "raw_text": raw_text,
        "metadata": record_metadata,
    }
    rows.append(row)
    _save_raw(rows, path)
    return load_user_records(path)[-1]


def add_patient_history_entry(
    name: str,
    entry_type: str,
    summary: str,
    raw_text: str,
    source: str = "assistant_chat",
    path: Path = USER_RECORDS_FILE,
) -> PatientRecord:
    return add_user_medical_record(
        name=name,
        summary=summary,
        raw_text=raw_text,
        source=source,
        metadata={
            "record_type": "patient_history_entry",
            "entry_type": entry_type,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        },
        path=path,
    )


def update_user_record_summary(name: str, summary: str, path: Path = USER_RECORDS_FILE) -> bool:
    rows = _load_raw(path)
    for row in rows:
        if row["name"].lower() == name.lower():
            row["summary"] = summary
            row["raw_text"] = summary
            _save_raw(rows, path)
            return True
    return False
