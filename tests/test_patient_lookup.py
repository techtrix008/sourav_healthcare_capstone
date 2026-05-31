from src.agent_graph import build_runtime


def test_patient_lookup_by_name():
    store, _ = build_runtime()
    patient = store.get_patient_by_name("David Thompson")
    assert patient is not None
    assert "Diabetes" in patient.searchable_text()
