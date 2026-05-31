from src.agent_graph import run_agent
from src.appointment_service import book_appointment, clear_appointments, load_appointments


def test_agent_finds_patient_and_books_appointment():
    result = run_agent("Ramesh has hypertension. Book a cardiologist on 2026-05-26 and summarize advice.")
    assert result["patient"]["name"] == "Ramesh Kulkarni"
    assert result["appointment"]["success"] is True
    assert "Appointment" in result["final_answer"]


def test_agent_asks_for_booking_details_before_booking():
    result = run_agent("create appointment for ramesh")
    assert result["patient"]["name"] == "Ramesh Kulkarni"
    assert result["appointment"] == {}
    assert result["appointment_workflow"]["status"] == "needs_details"
    assert "problem or symptoms" in result["appointment_workflow"]["missing"]
    assert "preferred date" in result["appointment_workflow"]["missing"]
    assert not any(log["tool"] == "appointment_booking" for log in result["tool_logs"])


def test_agent_clears_appointments_instead_of_booking():
    book_appointment("Ramesh Kulkarni", "Cardiologist")
    assert load_appointments()

    result = run_agent("delete all current appointments")

    assert result["appointment"]["success"] is True
    assert result["appointment"]["deleted_count"] >= 1
    assert load_appointments() == []
    assert not any(log["tool"] == "appointment_booking" for log in result["tool_logs"])
    assert "Deleted" in result["final_answer"]


def test_agent_lists_no_appointments_before_offering_booking():
    clear_appointments()

    result = run_agent("show all appointments for Ramesh")

    assert result["patient"]["name"] == "Ramesh Kulkarni"
    assert result["appointment_list"] == []
    assert "No appointments are currently booked for Ramesh Kulkarni" in result["final_answer"]
    assert not any(log["tool"] == "appointment_booking" for log in result["tool_logs"])


def test_agent_lists_all_appointments_without_patient_context():
    clear_appointments()
    book_appointment("Ramesh Kulkarni", "Cardiologist")

    result = run_agent("show me all appointments for all patients")

    assert result["patient"] is None
    assert "All appointments:" in result["final_answer"]
    assert "Ramesh Kulkarni" in result["final_answer"]

    system_result = run_agent("show me all appointments in the system")
    assert system_result["patient"] is None
    assert "All appointments:" in system_result["final_answer"]
    assert "Ramesh Kulkarni" in system_result["final_answer"]


def test_agent_lists_appointments_for_doctor_without_patient_context():
    clear_appointments()
    book_appointment("Ramesh Kulkarni", "Cardiologist")

    result = run_agent("show me all appointments for Dr Kavita")

    assert result["patient"] is None
    assert "Appointments for Dr. Kavita Menon" in result["final_answer"]
    assert "Ramesh Kulkarni" in result["final_answer"]
