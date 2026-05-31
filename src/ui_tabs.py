from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .appointment_service import load_appointments, load_doctors, save_doctors
from .config import AGENT_LOG_FILE, EVALUATION_LOG_FILE
from .logger import read_jsonl
from .record_service import add_user_medical_record, add_user_record, update_user_record_summary
from .ui_support import (
    dataframe_to_excel_bytes,
    extract_uploaded_pdf_text,
    patient_name_from_text,
    patient_rows,
    refresh_runtime_data,
    unique_patient_names,
)


def render_patients_tab() -> None:
    """Render patient management, record upload, and patient Excel import/export."""
    st.subheader("Patient Records")
    rows = patient_rows()
    st.download_button(
        "Export patient data to Excel",
        data=dataframe_to_excel_bytes(rows),
        file_name="patients_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    uploaded_patients = st.file_uploader("Upload patient Excel", type=["xlsx"], key="patient_excel_upload")
    if uploaded_patients:
        uploaded_df = pd.read_excel(uploaded_patients)
        added = 0
        for _, row in uploaded_df.iterrows():
            name = str(row.get("Name") or row.get("name") or "").strip()
            summary = str(
                row.get("Summary")
                or row.get("summary")
                or row.get("Notes")
                or row.get("Report")
                or row.get("Test Results")
                or ""
            ).strip()
            if not name or not summary:
                continue
            raw_text = str(row.get("Raw Text") or row.get("raw_text") or summary).strip()
            source_file = str(row.get("Source") or uploaded_patients.name)
            add_user_medical_record(
                name,
                summary,
                raw_text,
                source_file,
                int(row["Age"]) if "Age" in row and pd.notna(row["Age"]) else None,
                str(row.get("Gender") or "") or None,
                str(row.get("Phone_number") or row.get("Phone") or "") or None,
                str(row.get("Email") or "") or None,
                str(row.get("Address") or "") or None,
            )
            added += 1
        refresh_runtime_data()
        st.success(f"Imported {added} patient/medical record(s).")
        st.rerun()

    st.markdown("#### Upload medical reports")
    report_patient = st.selectbox("Attach report to patient", unique_patient_names(), key="report_patient_selector")
    uploaded_reports = st.file_uploader(
        "Upload medical report PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key="medical_report_pdf_upload",
    )
    if uploaded_reports and st.button("Import uploaded medical reports", type="primary"):
        imported = 0
        for uploaded_report in uploaded_reports:
            try:
                raw_text = extract_uploaded_pdf_text(uploaded_report)
            except Exception as exc:
                st.warning(f"Could not read {uploaded_report.name}: {exc}")
                continue
            detected_name = patient_name_from_text(raw_text)
            patient_name = detected_name or report_patient
            summary = raw_text[:1600] if raw_text else f"Uploaded medical report: {uploaded_report.name}"
            add_user_medical_record(patient_name, summary, raw_text, uploaded_report.name)
            imported += 1
        refresh_runtime_data()
        st.success(f"Imported {imported} medical report(s).")
        st.rerun()

    search = st.text_input("Search patients")
    if search:
        rows = [row for row in rows if search.lower() in json.dumps(row, default=str).lower()]
    st.data_editor(rows, use_container_width=True, hide_index=True, key="patient_data_editor")
    st.caption("Edits to built-in Excel/PDF rows are preview-only. Use Add/Update below to persist manual records.")

    with st.expander("Record details"):
        selected_name = st.selectbox("Select patient", [row["Name"] for row in rows] if rows else [])
        for row in rows:
            if row["Name"] == selected_name:
                st.write(row)
                break

    st.markdown("#### Add or update records")
    with st.form("add_patient_record_form"):
        st.markdown("Add patient")
        add_name = st.text_input("Name")
        add_age = st.number_input("Age", min_value=0, max_value=120, value=0)
        add_gender = st.selectbox("Gender", ["", "Female", "Male", "Other"])
        add_phone = st.text_input("Phone")
        add_email = st.text_input("Email")
        add_address = st.text_input("Address")
        add_summary = st.text_area("Clinical summary / notes")
        add_submitted = st.form_submit_button("Add patient record")
    if add_submitted:
        if not add_name.strip() or not add_summary.strip():
            st.warning("Name and clinical summary are required.")
        else:
            add_user_record(
                add_name.strip(),
                int(add_age) if add_age else None,
                add_gender or None,
                add_phone.strip() or None,
                add_email.strip() or None,
                add_address.strip() or None,
                add_summary.strip(),
            )
            refresh_runtime_data()
            st.success(f"Added record for {add_name.strip()}.")
            st.rerun()

    with st.form("update_patient_record_form"):
        st.markdown("Update manual patient summary")
        update_name = st.selectbox("Patient to update", unique_patient_names(), key="update_patient_name")
        update_summary = st.text_area("New summary")
        update_submitted = st.form_submit_button("Update summary")
    if update_submitted:
        if update_user_record_summary(update_name, update_summary.strip()):
            refresh_runtime_data()
            st.success(f"Updated manual record for {update_name}.")
            st.rerun()
        else:
            st.warning("Only manually added records can be updated here. Add this patient as a manual record first.")


def render_appointments_tab() -> None:
    """Render doctor schedule editing and appointment export views."""
    st.subheader("Doctor Schedule")
    doctors = load_doctors()
    st.download_button(
        "Export doctor schedule to Excel",
        data=dataframe_to_excel_bytes(doctors),
        file_name="doctors_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    uploaded_doctors = st.file_uploader("Upload doctor schedule Excel", type=["xlsx"], key="doctor_excel_upload")
    if uploaded_doctors:
        doctor_df = pd.read_excel(uploaded_doctors)
        imported = []
        for _, row in doctor_df.iterrows():
            if not row.get("name") and not row.get("Name"):
                continue
            slots = row.get("available_slots") or row.get("Available Slots") or ""
            if isinstance(slots, str):
                slots = [slot.strip() for slot in slots.replace(";", ",").split(",") if slot.strip()]
            imported.append(
                {
                    "doctor_id": str(row.get("doctor_id") or row.get("Doctor ID") or f"D{len(imported) + 1:03d}"),
                    "name": str(row.get("name") or row.get("Name")),
                    "specialty": str(row.get("specialty") or row.get("Specialty") or "General Physician"),
                    "location": str(row.get("location") or row.get("Location") or ""),
                    "available_slots": slots,
                }
            )
        if imported:
            save_doctors(imported)
            st.success(f"Imported {len(imported)} doctor(s).")
            st.rerun()
    edited_doctors = st.data_editor(doctors, use_container_width=True, hide_index=True, key="doctor_data_editor")
    if st.button("Save doctor schedule edits"):
        normalized_doctors = []
        doctor_rows = edited_doctors.to_dict("records") if hasattr(edited_doctors, "to_dict") else edited_doctors
        for doctor in doctor_rows:
            slots = doctor.get("available_slots", [])
            if isinstance(slots, str):
                slots = [slot.strip() for slot in slots.replace(";", ",").split(",") if slot.strip()]
            normalized_doctors.append({**doctor, "available_slots": slots})
        save_doctors(normalized_doctors)
        st.success("Doctor schedule saved.")
        st.rerun()

    st.subheader("Booked Appointments")
    appointments = load_appointments()
    st.download_button(
        "Export appointments to Excel",
        data=dataframe_to_excel_bytes(appointments),
        file_name="appointments_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.dataframe(appointments, use_container_width=True, hide_index=True)


def render_agent_trace_tab() -> None:
    st.subheader("Memory and Agent Logs")
    logs = read_jsonl(Path(AGENT_LOG_FILE), limit=50)
    if logs:
        for log in reversed(logs):
            with st.expander(f"{log.get('timestamp')} - {log.get('query', '')[:80]}"):
                st.json(log)
    else:
        st.info("No agent runs logged yet. Run a query from the Assistant tab.")


def render_evaluation_tab() -> None:
    st.subheader("Evaluation Dashboard")
    evaluations = read_jsonl(Path(EVALUATION_LOG_FILE), limit=100)
    if evaluations:
        avg_score = sum(item.get("score", 0) for item in evaluations) / len(evaluations)
        appointment_rate = sum(1 for item in evaluations if item.get("appointment_success")) / len(evaluations)
        patient_rate = sum(1 for item in evaluations if item.get("patient_found")) / len(evaluations)
        col1, col2, col3 = st.columns(3)
        col1.metric("Average score", f"{avg_score:.2f}")
        col2.metric("Patient retrieval rate", f"{patient_rate:.0%}")
        col3.metric("Appointment success rate", f"{appointment_rate:.0%}")
        st.dataframe(evaluations, use_container_width=True, hide_index=True)
    else:
        st.info("No evaluation results yet.")
