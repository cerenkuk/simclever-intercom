from types import SimpleNamespace

import pytest

from debrief.indexing import IndexValidationError, validate_index


def segments():
    return [
        SimpleNamespace(id="a", segment_ref="S000001", sequence_no=1, start_ms=10000, end_ms=15000),
        SimpleNamespace(id="b", segment_ref="S000002", sequence_no=2, start_ms=16000, end_ms=22000),
    ]


def valid_payload():
    return {"chapters":[{"start_segment_ref":"S000001","end_segment_ref":"S000002","title":"Giriş","summary":"Temel akış","confidence":.9}],"events":[{"segment_ref":"S000002","category":"iletisim","title":"Açıklama","reason":"Konu açıklanıyor","confidence":.8,"importance":.7}]}


def test_valid_index_uses_segment_time_and_lead_in():
    chapters, events = validate_index(valid_payload(), segments(), {"iletisim"}, 5000)
    assert chapters[0][1].segment_ref == "S000001"
    assert events[0][1].start_ms == 16000
    assert events[0][2] == 11000


def test_rejects_empty_segments_and_negative_lead_in():
    with pytest.raises(IndexValidationError, match="segment bulunamadı"):
        validate_index({"chapters": [], "events": []}, [], {"iletisim"})
    with pytest.raises(IndexValidationError, match="Lead-in negatif"):
        validate_index(valid_payload(), segments(), {"iletisim"}, -1)


def test_rejects_placeholder_title_and_session_overflow():
    payload = valid_payload()
    payload["events"][0]["title"] = "Olay"
    with pytest.raises(IndexValidationError, match="placeholder"):
        validate_index(payload, segments(), {"iletisim"})

    payload = valid_payload()
    with pytest.raises(IndexValidationError, match="oturum süresi dışına"):
        validate_index(payload, segments(), {"iletisim"}, session_duration_ms=20_000)


@pytest.mark.parametrize("mutation, message", [
    (lambda p: p["events"][0].update(segment_ref="S999999"), "var olmayan segment"),
    (lambda p: p["events"][0].update(category="yasak"), "izinsiz kategori"),
    (lambda p: p["events"].append(dict(p["events"][0])), "Mükerrer olay"),
])
def test_rejects_invalid_llm_output(mutation, message):
    payload = valid_payload(); mutation(payload)
    with pytest.raises(IndexValidationError, match=message):
        validate_index(payload, segments(), {"iletisim"}, 5000)
