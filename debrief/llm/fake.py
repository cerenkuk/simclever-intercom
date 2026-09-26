from debrief.llm.base import LLMResult


class FakeLLMAdapter:
    def __init__(self, payload: dict | None = None):
        self.payload = payload

    def generate_index(self, segments: list[dict], categories: list[str]) -> LLMResult:
        if self.payload is not None:
            payload = self.payload
        else:
            first, last = segments[0], segments[-1]
            event_segment = segments[min(1, len(segments) - 1)]
            payload = {
                "chapters": [{
                    "start_segment_ref": first["segment_ref"], "end_segment_ref": last["segment_ref"],
                    "title": "Eğitim akışı", "summary": "Kayıttaki temel eğitim akışı.", "confidence": 0.9,
                }],
                "events": [{
                    "segment_ref": event_segment["segment_ref"], "category": categories[0],
                    "title": "Önemli eğitim anı", "reason": "Temel konu açıklanıyor.",
                    "confidence": 0.9, "importance": 0.8,
                }],
            }
        return LLMResult("fake", "deterministic-indexer", "1", payload)
