from scripts.run_debrief_llm_benchmark import evaluate, extract_index_json


CASE = {
    "allowed_segments": ["S000001", "S000002", "S000003"],
    "allowed_categories": ["egitim_ani"],
    "expected_segment": "S000003",
    "expected_event_count": 1,
}


def test_extract_index_json_ignores_runner_text():
    output = 'banner\n{"events":[{"segment_id":"S000003","category":"egitim_ani","title":"Kural"}]}\ntimings'
    assert extract_index_json(output)["events"][0]["segment_id"] == "S000003"


def test_evaluate_accepts_expected_single_event():
    payload = {"events": [{"segment_id": "S000003", "category": "egitim_ani", "title": "Kural"}]}
    assert evaluate(payload, CASE)["passed"] is True


def test_evaluate_rejects_wrong_segment_and_extra_event():
    payload = {"events": [
        {"segment_id": "S000001", "category": "egitim_ani", "title": "Tanıtım"},
        {"segment_id": "S000003", "category": "egitim_ani", "title": "Kural"},
    ]}
    result = evaluate(payload, CASE)
    assert result["passed"] is False
    assert any("olay sayısı" in error for error in result["errors"])


def test_evaluate_accepts_empty_events_when_none_expected():
    case = {**CASE, "expected_segment": None, "expected_event_count": 0}
    assert evaluate({"events": []}, case)["passed"] is True


def test_evaluate_rejects_placeholder_title():
    payload = {"events": [{"segment_id": "S000003", "category": "egitim_ani", "title": "kısa Türkçe başlık"}]}
    result = evaluate(payload, CASE)
    assert result["passed"] is False
    assert any("placeholder" in error for error in result["errors"])
