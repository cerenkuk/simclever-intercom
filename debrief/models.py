import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from debrief.db import Base


def new_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PrivacyMode(str, enum.Enum):
    LOCAL_ONLY = "yalnız_yerel"
    CLOUD_ALLOWED = "bulut_izinli"


class SessionStatus(str, enum.Enum):
    DRAFT = "taslak"
    READY = "hazır"
    PROCESSING = "işleniyor"
    COMPLETED = "tamamlandı"
    FAILED = "hatalı"


class TrackType(str, enum.Enum):
    VIDEO = "video"
    AUDIO = "ses"
    TRANSCRIPT = "transkript"
    INDEX = "indeks"
    NOTE = "not"
    DEVICE = "cihaz"
    CHECKLIST = "kontrol_listesi"


class TrackStatus(str, enum.Enum):
    PENDING = "bekliyor"
    READY = "hazır"
    PROCESSING = "işleniyor"
    FAILED = "hatalı"


class RunStatus(str, enum.Enum):
    COMPLETED = "tamamlandı"
    FAILED = "hatalı"


class JobStatus(str, enum.Enum):
    QUEUED = "kuyrukta"
    RUNNING = "çalışıyor"
    CANCEL_REQUESTED = "iptal_istendi"
    CANCELLED = "iptal_edildi"
    COMPLETED = "tamamlandı"
    FAILED = "hatalı"


class EventStatus(str, enum.Enum):
    SUGGESTION = "öneri"
    APPROVED = "onaylı"
    REJECTED = "reddedildi"


class EventSource(str, enum.Enum):
    LLM = "llm"
    INSTRUCTOR = "eğitmen"
    INSTRUCTOR_NOTE = "eğitmen_notu"
    DEVICE = "cihaz"
    SYSTEM = "sistem"


class DebriefSession(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        CheckConstraint("duration_ms IS NULL OR duration_ms >= 0", name="ck_session_duration_nonnegative"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    scenario: Mapped[str | None] = mapped_column(Text)
    started_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    timezone_name: Mapped[str] = mapped_column(String(64), nullable=False, default="Europe/Istanbul")
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    privacy_mode: Mapped[PrivacyMode] = mapped_column(
        Enum(PrivacyMode, native_enum=False), nullable=False, default=PrivacyMode.LOCAL_ONLY
    )
    external_context_id: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, native_enum=False), nullable=False, default=SessionStatus.DRAFT
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    tracks: Mapped[list["Track"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )


class Track(Base):
    __tablename__ = "tracks"
    __table_args__ = (
        CheckConstraint("duration_ms IS NULL OR duration_ms >= 0", name="ck_track_duration_nonnegative"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_track_id: Mapped[str | None] = mapped_column(
        ForeignKey("tracks.id", ondelete="SET NULL"), index=True
    )
    type: Mapped[TrackType] = mapped_column(
        Enum(TrackType, native_enum=False, length=16), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    room: Mapped[str | None] = mapped_column(String(120))
    source_label: Mapped[str | None] = mapped_column(String(120))
    media_path: Mapped[str | None] = mapped_column(Text)
    original_filename: Mapped[str | None] = mapped_column(String(255))
    mime_type: Mapped[str | None] = mapped_column(String(120))
    container_format: Mapped[str | None] = mapped_column(String(120))
    codec_name: Mapped[str | None] = mapped_column(String(120))
    sample_rate: Mapped[int | None] = mapped_column(Integer)
    channel_count: Mapped[int | None] = mapped_column(Integer)
    offset_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[TrackStatus] = mapped_column(
        Enum(TrackStatus, native_enum=False), nullable=False, default=TrackStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )

    session: Mapped[DebriefSession] = relationship(back_populates="tracks")
    parent: Mapped["Track | None"] = relationship(remote_side=[id], back_populates="children")
    children: Mapped[list["Track"]] = relationship(back_populates="parent")
    transcript_segments: Mapped[list["TranscriptSegment"]] = relationship(
        back_populates="track", cascade="all, delete-orphan", passive_deletes=True
    )


class TranscriptionRun(Base):
    __tablename__ = "transcription_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    source_track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    output_track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    engine: Mapped[str] = mapped_column(String(80), nullable=False)
    model: Mapped[str] = mapped_column(String(160), nullable=False)
    engine_version: Mapped[str] = mapped_column(String(80), nullable=False)
    config_version: Mapped[str] = mapped_column(String(80), nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    processing_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus, native_enum=False), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (
        CheckConstraint("sequence_no >= 1", name="ck_segment_sequence_positive"),
        CheckConstraint("start_ms >= 0", name="ck_segment_start_nonnegative"),
        CheckConstraint("end_ms >= start_ms", name="ck_segment_end_after_start"),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_segment_confidence"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("transcription_runs.id", ondelete="CASCADE"), index=True)
    segment_ref: Mapped[str] = mapped_column(String(16), nullable=False)
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    asr_engine: Mapped[str] = mapped_column(String(80), nullable=False)
    asr_model: Mapped[str] = mapped_column(String(160), nullable=False)
    asr_version: Mapped[str] = mapped_column(String(80), nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)

    track: Mapped[Track] = relationship(back_populates="transcript_segments")


class IndexRun(Base):
    __tablename__ = "index_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    source_track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    output_track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    llm_provider: Mapped[str] = mapped_column(String(80), nullable=False)
    llm_model: Mapped[str] = mapped_column(String(160), nullable=False)
    llm_version: Mapped[str] = mapped_column(String(80), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(40), nullable=False)
    category_version: Mapped[str] = mapped_column(String(40), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(40), nullable=False)
    lead_in_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    processing_ms: Mapped[int | None] = mapped_column(Integer)
    model_load_ms: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    failed_chunk_count: Mapped[int | None] = mapped_column(Integer)
    peak_ram_mb: Mapped[float | None] = mapped_column(Float)
    input_fingerprint: Mapped[str | None] = mapped_column(String(64), index=True)
    source_fingerprint: Mapped[str | None] = mapped_column(String(64))
    config_fingerprint: Mapped[str | None] = mapped_column(String(64))
    model_fingerprint: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus, native_enum=False), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False), nullable=False, default=JobStatus.QUEUED
    )
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stage: Mapped[str | None] = mapped_column(String(120))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    checkpoint_json: Mapped[str | None] = mapped_column(Text)
    result_json: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Chapter(Base):
    __tablename__ = "chapters"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    index_run_id: Mapped[str] = mapped_column(ForeignKey("index_runs.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    start_segment_id: Mapped[str] = mapped_column(ForeignKey("transcript_segments.id"), nullable=False)
    end_segment_id: Mapped[str] = mapped_column(ForeignKey("transcript_segments.id"), nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    index_run_id: Mapped[str] = mapped_column(ForeignKey("index_runs.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    source_segment_id: Mapped[str | None] = mapped_column(
        ForeignKey("transcript_segments.id"), nullable=True
    )
    evidence_segment_ref: Mapped[str | None] = mapped_column(String(255))
    evidence_text: Mapped[str | None] = mapped_column(Text)
    evidence_start_ms: Mapped[int | None] = mapped_column(Integer)
    evidence_end_ms: Mapped[int | None] = mapped_column(Integer)
    event_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    playback_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    source: Mapped[EventSource] = mapped_column(
        Enum(EventSource, values_callable=lambda items: [item.value for item in items], native_enum=False),
        nullable=False, default=EventSource.LLM,
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    importance: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[EventStatus] = mapped_column(Enum(EventStatus, native_enum=False), nullable=False)
    reviewed_by: Mapped[str | None] = mapped_column(String(120))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
