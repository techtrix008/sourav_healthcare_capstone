from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
VECTOR_DIR = BASE_DIR / "vector_db"

RECORDS_FILE = DATA_DIR / "records.xlsx"
DOCTOR_SCHEDULE_FILE = DATA_DIR / "doctors_schedule.json"
APPOINTMENTS_FILE = DATA_DIR / "appointments.json"
USER_RECORDS_FILE = DATA_DIR / "user_records.json"
AGENT_LOG_FILE = LOG_DIR / "agent_runs.jsonl"
EVALUATION_LOG_FILE = LOG_DIR / "evaluation_results.jsonl"

PDF_FILES = [
    DATA_DIR / "sample_patient.pdf",
    DATA_DIR / "sample_report_anjali.pdf",
    DATA_DIR / "sample_report_david.pdf",
    DATA_DIR / "sample_report_ramesh.pdf",
]

DISCLAIMER = (
    "This assistant supports administrative and educational workflows only. "
    "It does not provide a medical diagnosis or replace a licensed clinician."
)
