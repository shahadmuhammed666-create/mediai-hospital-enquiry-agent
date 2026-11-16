from memory import MemoryStore


def test_memory_store_saves_summary_and_clears_patient(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.json")

    store.save_summary(
        "patient-1",
        {
            "first_name": "Taylor Morgan",
            "past_symptoms": ["rash"],
            "recommended_specialties": ["Dermatology"],
            "appointments_booked": [],
            "preferences": {"time_of_day": "morning"},
        },
    )
    store.save_summary(
        "patient-1",
        {
            "past_symptoms": ["rash", "itchy skin"],
            "recommended_specialties": ["Dermatology"],
            "appointments_booked": [],
        },
    )

    memory = store.load_memory("patient-1")
    assert memory["first_name"] == "Taylor"
    assert memory["past_symptoms"] == ["rash", "itchy skin"]
    assert memory["recommended_specialties"] == ["Dermatology"]
    assert memory["preferences"] == {"time_of_day": "morning"}

    store.clear_memory("patient-1")
    assert store.load_memory("patient-1")["past_symptoms"] == []
