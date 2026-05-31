from __future__ import annotations

from difflib import SequenceMatcher
import re

from .models import PatientRecord


def _digits(value: str | None) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _merge_text(left: str | None, right: str | None) -> str | None:
    parts = []
    for value in (left, right):
        value = (value or "").strip()
        if value and value not in parts:
            parts.append(value)
    return "\n\n".join(parts) or None


def _identity_score(left: PatientRecord, right: PatientRecord) -> float:
    left_phone = _digits(left.phone)
    right_phone = _digits(right.phone)
    if left_phone and right_phone and (left_phone.endswith(right_phone[-7:]) or right_phone.endswith(left_phone[-7:])):
        return 1.0

    left_name = _name_key(left.name)
    right_name = _name_key(right.name)
    name_ratio = SequenceMatcher(None, left_name, right_name).ratio()
    if left_name == right_name:
        name_ratio = 1.0

    demographic_bonus = 0.0
    if left.age and right.age and left.age == right.age:
        demographic_bonus += 0.08
    if left.gender and right.gender and left.gender.lower() == right.gender.lower():
        demographic_bonus += 0.05
    return min(name_ratio + demographic_bonus, 1.0)


def _source_document(record: PatientRecord) -> dict:
    return {
        "patient_id": record.patient_id,
        "name": record.name,
        "age": record.age,
        "gender": record.gender,
        "phone": record.phone,
        "email": record.email,
        "address": record.address,
        "summary": record.summary,
        "raw_text": record.raw_text,
        "source": record.source,
        "metadata": record.metadata or {},
    }


def _merge_record(base: PatientRecord, incoming: PatientRecord) -> None:
    incoming_metadata = incoming.metadata or {}
    is_history_entry = incoming_metadata.get("record_type") == "patient_history_entry"
    base.age = base.age or incoming.age
    base.gender = base.gender or incoming.gender
    base.phone = base.phone or incoming.phone
    base.email = base.email or incoming.email
    base.address = base.address or incoming.address
    if not is_history_entry:
        base.summary = _merge_text(base.summary, incoming.summary)
        base.raw_text = _merge_text(base.raw_text, incoming.raw_text)

    sources = list(base.metadata.get("source_documents", []))
    if not sources:
        sources.append(_source_document(base))
    sources.append(_source_document(incoming))
    base.metadata["source_documents"] = sources
    base.metadata["merged_record_count"] = len(sources)
    if incoming.source and incoming.source not in base.source:
        base.source = f"{base.source}, {incoming.source}"


def deduplicate_patient_records(records: list[PatientRecord]) -> list[PatientRecord]:
    merged: list[PatientRecord] = []
    for record in records:
        match = next((candidate for candidate in merged if _identity_score(candidate, record) >= 0.88), None)
        if match:
            _merge_record(match, record)
        else:
            record.metadata = dict(record.metadata or {})
            record.metadata.setdefault("source_documents", [_source_document(record)])
            record.metadata.setdefault("merged_record_count", 1)
            merged.append(record)
    return merged


class PatientStore:
    def __init__(self, records: list[PatientRecord]):
        self.records = deduplicate_patient_records(records)

    def list_patients(self) -> list[PatientRecord]:
        return self.records

    def add_patient_record(self, record: PatientRecord) -> None:
        self.records.append(record)

    def update_patient_summary(self, patient_id: str, new_summary: str) -> PatientRecord | None:
        patient = self.get_patient_by_id(patient_id)
        if patient:
            patient.summary = new_summary
        return patient

    def get_patient_by_id(self, patient_id: str) -> PatientRecord | None:
        return next((record for record in self.records if record.patient_id == patient_id), None)

    def get_patient_by_phone(self, phone: str) -> PatientRecord | None:
        normalized = "".join(ch for ch in phone if ch.isdigit())
        for record in self.records:
            if record.phone and normalized in "".join(ch for ch in record.phone if ch.isdigit()):
                return record
        return None

    def get_patient_by_name(self, name: str) -> PatientRecord | None:
        name_l = name.lower()
        exact = [record for record in self.records if record.name.lower() == name_l]
        if exact:
            return exact[0]

        candidates = []
        for record in self.records:
            ratio = SequenceMatcher(None, name_l, record.name.lower()).ratio()
            if name_l in record.name.lower() or record.name.lower() in name_l:
                ratio += 0.3
            candidates.append((ratio, record))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1] if candidates and candidates[0][0] >= 0.45 else None

    def search(self, query: str, limit: int = 5) -> list[PatientRecord]:
        query_l = query.lower()
        scored = []
        for record in self.records:
            text = record.searchable_text().lower()
            score = sum(1 for token in query_l.split() if token in text)
            score += SequenceMatcher(None, query_l, record.name.lower()).ratio()
            if score > 0.3:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]
