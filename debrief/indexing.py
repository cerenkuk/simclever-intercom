from pydantic import BaseModel, Field, ValidationError


class ChapterCandidate(BaseModel):
    start_segment_ref: str
    end_segment_ref: str
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(ge=0, le=1)


class EventCandidate(BaseModel):
    segment_ref: str
    category: str
    title: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=1000)
    confidence: float = Field(ge=0, le=1)
    importance: float = Field(ge=0, le=1)


class IndexCandidate(BaseModel):
    chapters: list[ChapterCandidate]
    events: list[EventCandidate]


class IndexValidationError(ValueError):
    pass


PLACEHOLDER_TITLES = {"başlık", "olay", "event", "chapter", "bölüm", "title", "placeholder"}


def _meaningful(value: str) -> bool:
    return value.strip().casefold() not in PLACEHOLDER_TITLES


def validate_index(
    payload: dict,
    segments: list,
    categories: set[str],
    lead_in_ms: int = 5000,
    session_duration_ms: int | None = None,
):
    if lead_in_ms < 0:
        raise IndexValidationError("Lead-in negatif olamaz")
    if not segments:
        raise IndexValidationError("İndeks için segment bulunamadı")
    try:
        candidate = IndexCandidate.model_validate(payload)
    except ValidationError as exc:
        raise IndexValidationError("LLM çıktısı şemaya uymuyor") from exc
    by_ref = {segment.segment_ref: segment for segment in segments}
    chapters = []
    previous_end = -1
    for item in candidate.chapters:
        if item.start_segment_ref not in by_ref or item.end_segment_ref not in by_ref:
            raise IndexValidationError("Bölüm var olmayan segment referansı içeriyor")
        start, end = by_ref[item.start_segment_ref], by_ref[item.end_segment_ref]
        if not _meaningful(item.title):
            raise IndexValidationError("Bölüm placeholder başlık içeriyor")
        if start.start_ms < 0 or end.end_ms < start.start_ms:
            raise IndexValidationError("Bölüm geçersiz zaman aralığı içeriyor")
        if session_duration_ms is not None and end.end_ms > session_duration_ms:
            raise IndexValidationError("Bölüm oturum süresi dışına taşıyor")
        if start.sequence_no > end.sequence_no or start.start_ms < previous_end:
            raise IndexValidationError("Bölümler sırasız veya örtüşüyor")
        chapters.append((item, start, end))
        previous_end = end.end_ms
    events, seen = [], set()
    for item in candidate.events:
        if item.segment_ref not in by_ref:
            raise IndexValidationError("Olay var olmayan segment referansı içeriyor")
        if item.category not in categories:
            raise IndexValidationError("Olay izinsiz kategori içeriyor")
        if not _meaningful(item.title):
            raise IndexValidationError("Olay placeholder başlık içeriyor")
        key = (item.segment_ref, item.category, item.title.casefold())
        if key in seen:
            raise IndexValidationError("Mükerrer olay bulundu")
        seen.add(key)
        segment = by_ref[item.segment_ref]
        if segment.start_ms < 0 or segment.end_ms < segment.start_ms:
            raise IndexValidationError("Olay geçersiz segment zamanı içeriyor")
        if session_duration_ms is not None and segment.start_ms > session_duration_ms:
            raise IndexValidationError("Olay oturum süresi dışına taşıyor")
        events.append((item, segment, max(0, segment.start_ms - lead_in_ms)))
    return chapters, events
