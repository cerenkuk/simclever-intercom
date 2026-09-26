from pathlib import Path

from debrief.asr.base import ASRResult, ASRSegment


class FakeASRAdapter:
    """Hızlı ve deterministik API test adapter'ı."""

    def __init__(self, segments: tuple[ASRSegment, ...] | None = None):
        self.segments = segments or (
            ASRSegment("S000001", 0, 1500, "Simülasyon başladı.", 0.99),
            ASRSegment("S000002", 1500, 3200, "Öğrenci değerlendirme yaptı.", 0.95),
        )

    def transcribe(self, media_path: Path, language: str = "tr") -> ASRResult:
        return ASRResult(
            engine="fake-asr",
            model="deterministic-test",
            engine_version="1",
            config_version="fake-v1",
            language=language,
            duration_ms=max((segment.end_ms for segment in self.segments), default=0),
            segments=self.segments,
        )

