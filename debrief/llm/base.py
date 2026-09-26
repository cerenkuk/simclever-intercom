from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResult:
    provider: str
    model: str
    version: str
    payload: dict
    processing_ms: int | None = None
    model_load_ms: int | None = None
    chunk_count: int | None = None
    failed_chunk_count: int | None = None
    peak_ram_mb: float | None = None


class LLMAdapter(Protocol):
    def generate_index(self, segments: list[dict], categories: list[str]) -> LLMResult: ...
