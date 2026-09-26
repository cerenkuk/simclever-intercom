from pathlib import Path

import pytest

from debrief.llm.local_qwen import DEFAULT_RUBRICS, LocalQwenLLMAdapter
from debrief.llm.gemini import GeminiLLMAdapter


CATEGORIES = ["iletisim", "degerlendirme", "mudahale", "karar", "guvenlik", "ekip_calismasi"]


def segment(number: int, start_ms: int, duration_ms: int = 30_000) -> dict:
    return {
        "segment_ref": f"S{number:06d}", "start_ms": start_ms,
        "end_ms": start_ms + duration_ms, "text": f"Eğitim içeriği {number}",
    }


def event(ref: str, category: str = "mudahale") -> dict:
    return {
        "segment_ref": ref, "category": category, "title": f"Uygulama adımı {ref}",
        "reason": "Eğitim açısından önemlidir", "confidence": 0.8, "importance": 0.8,
    }


def test_configuration_is_generic_and_versioned():
    document = DEFAULT_RUBRICS.read_text(encoding="utf-8")
    assert 'version: "001"' in document
    assert "target_event_interval_seconds" in document
    assert "boyunluk" not in document.casefold()
    assert "servikal" not in document.casefold()


def test_windows_cover_long_recording_with_twenty_second_overlap():
    segments = [segment(number, (number - 1) * 30_000) for number in range(1, 35)]
    windows = LocalQwenLLMAdapter._windows(segments)
    assert len(windows) == 7
    assert windows[0][-1]["end_ms"] >= windows[1][0]["start_ms"]
    assert {item["segment_ref"] for item in windows[0]} & {item["segment_ref"] for item in windows[1]}


def test_cue_selection_limits_input_and_preserves_both_ends():
    window = [segment(number, (number - 1) * 10_000) for number in range(1, 151)]
    selected = LocalQwenLLMAdapter._select_cues(window)
    assert len(selected) == 30
    assert selected[0]["segment_ref"] == "S000001"
    assert selected[-1]["segment_ref"] == "S000150"


def test_window_validation_rejects_invented_segment():
    window = [segment(1, 0), segment(2, 30_000)]
    payload = {
        "window": {"start_segment_ref": "S000001", "end_segment_ref": "S000002", "title": "Giriş", "summary": "Özet"},
        "events": [event("S999999")],
    }
    with pytest.raises(RuntimeError, match="uydurma"):
        LocalQwenLLMAdapter._validate_window(payload, window, CATEGORIES, 1)


def test_window_prompt_does_not_seed_domain_events():
    adapter = LocalQwenLLMAdapter(Path("server"), Path("model"))
    prompt = adapter._window_prompt([segment(1, 0)], CATEGORIES, 1)
    assert "CPR uygulandı" not in prompt
    assert "Shock verildi" not in prompt
    assert "Şok verildi" not in prompt
    assert "Oksijen başlandı" not in prompt
    assert "Mahremiyet sağlandı" not in prompt


def test_final_merge_builds_non_overlapping_chapters_and_deduplicates():
    segments = [segment(number, (number - 1) * 60_000) for number in range(1, 7)]
    candidates = [event("S000002"), event("S000005", "guvenlik")]
    merged = {
        "chapter_starts": [
            {"start_segment_ref": "S000001", "title": "Başlangıç", "summary": "Temel yaklaşım"},
            {"start_segment_ref": "S000004", "title": "Uygulama", "summary": "İleri adımlar"},
            {"start_segment_ref": "S000006", "title": "Kapanış", "summary": "Son kontroller"},
        ],
        "events": candidates,
    }
    payload = LocalQwenLLMAdapter._final_payload(merged, segments, candidates, CATEGORIES, 2)
    assert [item["end_segment_ref"] for item in payload["chapters"]] == ["S000003", "S000005", "S000006"]
    assert [item["segment_ref"] for item in payload["events"]] == ["S000002", "S000005"]


def test_generate_index_uses_chunks_and_code_merge(monkeypatch):
    segments = [segment(number, (number - 1) * 120_000, 20_000) for number in range(1, 10)]
    adapter = LocalQwenLLMAdapter(Path("server"), Path("model"))
    calls = []
    windows = adapter._windows(segments)
    window_count = len(windows)

    def fake_run(prompt, max_tokens=1400, schema=None):
        calls.append(prompt)
        refs = [item["segment_ref"] for item in windows[len(calls) - 1]]
        chosen = list(dict.fromkeys(refs))[:3]
        return {
            "window": {"start_segment_ref": refs[0], "end_segment_ref": refs[-1], "title": "Pencere", "summary": "Özet"},
            "events": [event(ref) for ref in chosen],
        }

    monkeypatch.setattr(adapter, "_run", fake_run)
    result = adapter.generate_index(segments, CATEGORIES)
    assert len(calls) == window_count
    assert len(result.payload["events"]) == 5
    assert result.provider == "llama.cpp-server"
    assert result.version.startswith("chunked-code-merge")


def test_generate_index_resumes_completed_chunks_from_checkpoint(monkeypatch):
    segments = [segment(number, (number - 1) * 120_000, 20_000) for number in range(1, 10)]
    adapter = LocalQwenLLMAdapter(Path("server"), Path("model"))
    windows = adapter._windows(segments)
    checkpoint = {}
    first_calls = []

    adapter.checkpoint_loader = lambda: checkpoint
    adapter.checkpoint_saver = lambda value: (checkpoint.clear(), checkpoint.update(value))

    def interrupted(prompt, max_tokens=1400, schema=None):
        first_calls.append(prompt)
        if len(first_calls) > 2:
            raise RuntimeError("kesinti")
        refs = [item["segment_ref"] for item in windows[len(first_calls) - 1]]
        return {"window":{"start_segment_ref":refs[0],"end_segment_ref":refs[-1],"title":"Pencere","summary":"Özet"},
                "events":[event(ref) for ref in list(dict.fromkeys(refs))[:3]]}

    monkeypatch.setattr(adapter, "_run", interrupted)
    with pytest.raises(RuntimeError):
        adapter.generate_index(segments, CATEGORIES)
    assert checkpoint["completed_windows"] == [1, 2]

    resumed_calls = []
    def resumed(prompt, max_tokens=1400, schema=None):
        resumed_calls.append(prompt)
        window = windows[len(resumed_calls) + 1]
        refs = [item["segment_ref"] for item in window]
        return {"window":{"start_segment_ref":refs[0],"end_segment_ref":refs[-1],"title":"Pencere","summary":"Özet"},
                "events":[event(ref) for ref in list(dict.fromkeys(refs))[:3]]}

    monkeypatch.setattr(adapter, "_run", resumed)
    result = adapter.generate_index(segments, CATEGORIES)
    assert len(resumed_calls) == len(windows) - 2
    assert result.payload["events"]


def test_gemini_adapter_accepts_string_model_name(tmp_path, monkeypatch):
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("version: '001'\ntarget_event_interval_seconds: 180\n", encoding="utf-8")
    adapter = GeminiLLMAdapter("test-key", rubric, "gemini-3.7-flash")
    monkeypatch.setattr(adapter.client, "chat", lambda *args, **kwargs: '''
    {"window":{"start_segment_ref":"S000001","end_segment_ref":"S000002","title":"Eğitim","summary":"Özet"},
    "events":[{"segment_ref":"S000002","category":"degerlendirme","title":"Hasta değerlendirildi","reason":"Önemli","confidence":0.9,"importance":0.9}]}
    ''')
    segments = [
        {"segment_ref":"S000001","start_ms":0,"end_ms":1000,"text":"Giriş"},
        {"segment_ref":"S000002","start_ms":1000,"end_ms":5000,"text":"Hasta değerlendirildi"},
    ]
    result = adapter.generate_index(segments, ["degerlendirme"])
    assert result.provider == "google-gemini-api"
    assert result.model == "gemini-3.7-flash"
    assert len(result.payload["events"]) == 1
