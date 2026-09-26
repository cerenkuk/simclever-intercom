from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ASRSegment:
    ref: str
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None


@dataclass(frozen=True)
class ASRResult:
    engine: str
    model: str
    engine_version: str
    config_version: str
    language: str
    duration_ms: int
    segments: tuple[ASRSegment, ...]


class ASRAdapter(Protocol):
    def transcribe(self, media_path: Path, language: str = "tr") -> ASRResult: ...

