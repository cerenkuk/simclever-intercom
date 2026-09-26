from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from debrief.models import EventStatus, PrivacyMode, RunStatus, SessionStatus, TrackStatus, TrackType


class SessionCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    scenario: str | None = None
    started_at_utc: datetime | None = None
    timezone_name: str = Field(default="Europe/Istanbul", min_length=1, max_length=64)
    duration_ms: int | None = Field(default=None, ge=0)
    privacy_mode: PrivacyMode = PrivacyMode.LOCAL_ONLY
    external_context_id: str | None = Field(default=None, max_length=255)


class SessionRead(SessionCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: SessionStatus
    created_at: datetime
    updated_at: datetime


class TrackCreate(BaseModel):
    parent_track_id: str | None = None
    type: TrackType
    name: str = Field(min_length=1, max_length=200)
    room: str | None = Field(default=None, max_length=120)
    source_label: str | None = Field(default=None, max_length=120)
    media_path: str | None = None
    original_filename: str | None = Field(default=None, max_length=255)
    mime_type: str | None = Field(default=None, max_length=120)
    container_format: str | None = Field(default=None, max_length=120)
    codec_name: str | None = Field(default=None, max_length=120)
    sample_rate: int | None = Field(default=None, gt=0)
    channel_count: int | None = Field(default=None, gt=0)
    offset_ms: int = 0
    duration_ms: int | None = Field(default=None, ge=0)
    checksum_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")

    @model_validator(mode="after")
    def media_tracks_need_a_path(self):
        if self.type in {TrackType.VIDEO, TrackType.AUDIO} and not self.media_path:
            raise ValueError("Video ve ses kanalları için media_path zorunludur")
        return self


class TrackRead(TrackCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    status: TrackStatus
    created_at: datetime
    updated_at: datetime


class OffsetUpdate(BaseModel):
    offset_ms: int


class EventReviewUpdate(BaseModel):
    status: EventStatus
    reviewed_by: str = Field(default="yerel_kullanici", min_length=1, max_length=120)

    @model_validator(mode="after")
    def decision_must_be_final(self):
        if self.status not in {EventStatus.APPROVED, EventStatus.REJECTED}:
            raise ValueError("Olay yalnızca onaylanabilir veya reddedilebilir")
        return self


class InstructorNoteCreate(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    event_time_ms: int = Field(ge=0)
    instructor_name: str = Field(default="eğitmen", min_length=1, max_length=120)


class ManualBookmarkCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    category: str = Field(default="degerlendirme", min_length=1, max_length=80)
    event_time_ms: int = Field(ge=0)
    description: str = Field(default="", max_length=2000)
    instructor_name: str = Field(default="eğitmen", min_length=1, max_length=120)


class DeviceEventCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    category: str = Field(default="cihaz", min_length=1, max_length=80)
    event_time_ms: int = Field(ge=0)
    description: str = Field(default="", max_length=2000)
    device_name: str = Field(default="Cihaz", min_length=1, max_length=120)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    external_event_id: str | None = Field(default=None, min_length=1, max_length=255)


class SystemEventCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    category: str = Field(default="sistem", min_length=1, max_length=80)
    event_time_ms: int = Field(ge=0)
    description: str = Field(default="", max_length=2000)
    system_name: str = Field(default="SimClever", min_length=1, max_length=120)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    external_event_id: str = Field(min_length=1, max_length=255)


class EventUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=80)
    event_time_ms: int = Field(ge=0)
    description: str = Field(default="", max_length=2000)


class TranscriptSegmentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    track_id: str
    run_id: str
    segment_ref: str
    sequence_no: int
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None
    asr_engine: str
    asr_model: str
    asr_version: str
    language: str


class TranscriptionRead(BaseModel):
    id: str
    source_track_id: str
    output_track: TrackRead
    engine: str
    model: str
    engine_version: str
    config_version: str
    language: str
    duration_ms: int
    status: RunStatus
    segments: list[TranscriptSegmentRead]
