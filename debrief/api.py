from datetime import datetime, timezone
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from debrief.config import Settings
from debrief.asr import ASRAdapter
from debrief.llm import LLMAdapter
from debrief.indexing import IndexValidationError, validate_index
import yaml
from debrief.media import MediaError, audio_waveform, destination_for, extract_audio_wav, probe_media, resolve_stored_media, sha256_file, video_filmstrip, video_thumbnail
from debrief.privacy import CloudPrivacyError, require_cloud_allowed
from debrief.models import (
    Chapter, DebriefSession, Event, EventSource, EventStatus, IndexRun, JobStatus,
    ProcessingJob, RunStatus, Track, TrackStatus, TrackType, TranscriptSegment, TranscriptionRun,
)
from debrief.schemas import (
    DeviceEventCreate, EventReviewUpdate, EventUpdate, InstructorNoteCreate, ManualBookmarkCreate, OffsetUpdate,
    SessionCreate, SessionRead, SystemEventCreate, TrackCreate, TrackRead,
    TranscriptSegmentRead, TranscriptionRead,
)


def build_router(get_db, settings: Settings, asr_adapter: ASRAdapter, llm_adapter: LLMAdapter, processing_queue=None) -> APIRouter:
    router = APIRouter(prefix="/api/v1")
    indexing_config = settings.indexing()

    def model_fingerprint() -> str:
        identity = str(getattr(llm_adapter, "model", type(llm_adapter).__name__))
        model_path = getattr(llm_adapter, "model_path", None)
        if model_path:
            path = Path(model_path)
            if path.exists():
                stat = path.stat()
                identity = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def event_document(row: Event, db: Session) -> dict:
        segment = db.get(TranscriptSegment, row.source_segment_id) if row.source_segment_id else None
        track = db.get(Track, row.track_id)
        return {
            "id": row.id, "title": row.title, "description": row.description,
            "category": row.category,
            "event_time_ms": row.event_time_ms, "playback_time_ms": row.playback_time_ms,
            "confidence": row.confidence, "status": row.status.value,
            "source": row.source.value,
            "source_segment_id": row.source_segment_id,
            "source_segment_ref": row.evidence_segment_ref or (segment.segment_ref if segment else None),
            "asr_cue": row.evidence_text or (segment.text if segment else None),
            "evidence_start_ms": row.evidence_start_ms if row.evidence_start_ms is not None else (segment.start_ms if segment else None),
            "evidence_end_ms": row.evidence_end_ms if row.evidence_end_ms is not None else (segment.end_ms if segment else None),
            "track_name": track.name if track else None,
            "reviewed_by": row.reviewed_by,
            "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
        }

    def job_document(row: ProcessingJob) -> dict:
        checkpoint = json.loads(row.checkpoint_json) if row.checkpoint_json else {}
        completed_units = checkpoint.get("completed_chunks", checkpoint.get("completed_windows", []))
        identity = checkpoint.get("adapter_identity", "")
        checkpoint_model = Path(identity.partition(":")[2]).name if identity else None
        return {
            "id": row.id, "session_id": row.session_id, "track_id": row.track_id,
            "kind": row.kind, "status": row.status.value, "progress": row.progress,
            "stage": row.stage, "attempt_count": row.attempt_count,
            "max_attempts": row.max_attempts,
            "resume_available": bool(completed_units),
            "completed_units": len(completed_units),
            "checkpoint_model": checkpoint_model,
            "result": json.loads(row.result_json) if row.result_json else None,
            "error": row.error_message,
            "created_at": row.created_at.isoformat(),
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "heartbeat_at": row.heartbeat_at.isoformat() if row.heartbeat_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
        }

    @router.get("/sessions/{session_id}/events")
    def list_session_events(session_id: str, db: Session = Depends(get_db)):
        if db.get(DebriefSession, session_id) is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        rows = list(db.scalars(
            select(Event)
            .join(Track, Event.track_id == Track.id)
            .where(Track.session_id == session_id)
            .order_by(Event.event_time_ms, Event.id)
        ))
        return [event_document(row, db) for row in rows]

    @router.get("/sessions/{session_id}/timeline")
    def get_session_timeline(session_id: str, db: Session = Depends(get_db)):
        session = db.get(DebriefSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        tracks = list(db.scalars(
            select(Track).where(Track.session_id == session_id).order_by(Track.created_at)
        ))
        track_ids = [track.id for track in tracks]
        transcript_tracks = [track for track in tracks if track.type == TrackType.TRANSCRIPT]
        transcript_ids = [track.id for track in transcript_tracks]
        segments = list(db.scalars(
            select(TranscriptSegment)
            .where(TranscriptSegment.track_id.in_(transcript_ids))
            .order_by(TranscriptSegment.start_ms, TranscriptSegment.sequence_no)
        )) if transcript_ids else []
        index_runs = list(db.scalars(
            select(IndexRun)
            .where(IndexRun.output_track_id.in_(track_ids), IndexRun.status == RunStatus.COMPLETED)
            .order_by(IndexRun.created_at.desc())
        )) if track_ids else []
        latest_output_ids, seen_sources = [], set()
        for run in index_runs:
            if run.source_track_id in seen_sources:
                continue
            seen_sources.add(run.source_track_id)
            latest_output_ids.append(run.output_track_id)
        chapters = list(db.scalars(
            select(Chapter)
            .where(Chapter.track_id.in_(latest_output_ids))
            .order_by(Chapter.start_ms, Chapter.sequence_no)
        )) if latest_output_ids else []
        events = list(db.scalars(
            select(Event)
            .where(Event.track_id.in_(latest_output_ids))
            .order_by(Event.event_time_ms, Event.id)
        )) if latest_output_ids else []
        by_id = {track.id: track for track in tracks}
        duration_candidates = [session.duration_ms or 0]
        duration_candidates.extend(track.duration_ms or 0 for track in tracks)
        duration_candidates.extend(segment.end_ms for segment in segments)
        duration_candidates.extend(chapter.end_ms for chapter in chapters)
        duration_candidates.extend(event.event_time_ms for event in events)
        return {
            "session": {"id": session.id, "title": session.title,
                        "duration_ms": max(duration_candidates, default=0)},
            "media_tracks": [
                {"id": track.id, "type": track.type.value, "name": track.name,
                 "room": track.room, "offset_ms": track.offset_ms,
                 "duration_ms": track.duration_ms, "media_url": f"/api/v1/tracks/{track.id}/media",
                 "thumbnail_url": f"/api/v1/tracks/{track.id}/thumbnail" if track.type == TrackType.VIDEO else None,
                 "filmstrip_url": f"/api/v1/tracks/{track.id}/filmstrip" if track.type == TrackType.VIDEO else None,
                 "waveform_url": f"/api/v1/tracks/{track.id}/waveform" if track.type == TrackType.AUDIO else None}
                for track in tracks
                if track.type in {TrackType.VIDEO, TrackType.AUDIO} and track.media_path
            ],
            "cues": [
                {"id": segment.id, "segment_ref": segment.segment_ref,
                 "start_ms": segment.start_ms, "end_ms": segment.end_ms,
                 "text": segment.text[:240], "track_id": segment.track_id,
                 "track_name": by_id.get(segment.track_id).name if by_id.get(segment.track_id) else None}
                for segment in segments
            ],
            "chapters": [
                {"id": chapter.id, "title": chapter.title, "summary": chapter.summary,
                 "start_ms": chapter.start_ms, "end_ms": chapter.end_ms,
                 "confidence": chapter.confidence, "track_id": chapter.track_id}
                for chapter in chapters
            ],
            "events": [event_document(event, db) for event in events],
        }

    @router.post("/sessions/{session_id}/bookmarks", status_code=status.HTTP_201_CREATED)
    def create_manual_bookmark(
        session_id: str, payload: ManualBookmarkCreate, db: Session = Depends(get_db)
    ):
        session = db.get(DebriefSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        source = Track(
            session_id=session_id, type=TrackType.NOTE,
            name=f"Manuel bookmark · {payload.instructor_name}",
            source_label="manuel_eğitmen", duration_ms=session.duration_ms,
            status=TrackStatus.READY,
        )
        db.add(source); db.flush()
        output = Track(
            session_id=session_id, parent_track_id=source.id, type=TrackType.INDEX,
            name=f"{source.name} · İndeks", source_label="manuel-bookmark-v1",
            duration_ms=session.duration_ms, status=TrackStatus.READY,
        )
        db.add(output); db.flush()
        run = IndexRun(
            source_track_id=source.id, output_track_id=output.id,
            llm_provider="manual", llm_model="none", llm_version="manual-bookmark-v1",
            prompt_version="none", category_version="001", schema_version="index-v1",
            lead_in_ms=indexing_config.lead_in_ms, processing_ms=0, model_load_ms=0,
            config_fingerprint=indexing_config.fingerprint,
            chunk_count=0, failed_chunk_count=0, status=RunStatus.COMPLETED,
        )
        db.add(run); db.flush()
        now = datetime.now(timezone.utc)
        event = Event(
            index_run_id=run.id, track_id=output.id, source_segment_id=None,
            event_time_ms=payload.event_time_ms,
            playback_time_ms=max(0, payload.event_time_ms - indexing_config.lead_in_ms),
            title=payload.title.strip(), description=payload.description.strip(),
            category=payload.category.strip(), source=EventSource.INSTRUCTOR,
            confidence=1.0, importance=1.0, status=EventStatus.APPROVED,
            reviewed_by=payload.instructor_name, reviewed_at=now,
        )
        db.add(event); db.commit(); db.refresh(event)
        return event_document(event, db)

    @router.post("/sessions/{session_id}/device-events", status_code=status.HTTP_201_CREATED)
    def create_device_event(
        session_id: str, payload: DeviceEventCreate, db: Session = Depends(get_db)
    ):
        session = db.get(DebriefSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        device_name = payload.device_name.strip()
        source = db.scalar(select(Track).where(
            Track.session_id == session_id,
            Track.type == TrackType.DEVICE,
            Track.name == device_name,
        ))
        if source is None:
            source = Track(
                session_id=session_id, type=TrackType.DEVICE, name=device_name,
                source_label="cihaz_olayi", duration_ms=session.duration_ms,
                status=TrackStatus.READY,
            )
            db.add(source); db.flush()
        run = db.scalar(select(IndexRun).where(
            IndexRun.source_track_id == source.id,
            IndexRun.llm_provider == "device",
        ).order_by(IndexRun.created_at.desc()))
        if run is None:
            output = Track(
                session_id=session_id, parent_track_id=source.id, type=TrackType.INDEX,
                name=f"{device_name} · Olaylar", source_label="device-event-v1",
                duration_ms=session.duration_ms, status=TrackStatus.READY,
            )
            db.add(output); db.flush()
            run = IndexRun(
                source_track_id=source.id, output_track_id=output.id,
                llm_provider="device", llm_model=device_name, llm_version="device-event-v1",
                prompt_version="none", category_version="001", schema_version="index-v1",
                lead_in_ms=indexing_config.lead_in_ms, processing_ms=0, model_load_ms=0,
                config_fingerprint=indexing_config.fingerprint,
                chunk_count=0, failed_chunk_count=0, status=RunStatus.COMPLETED,
            )
            db.add(run); db.flush()
        else:
            output = db.get(Track, run.output_track_id)
        if payload.external_event_id:
            existing = db.scalar(select(Event).where(
                Event.index_run_id == run.id,
                Event.evidence_segment_ref == payload.external_event_id,
            ))
            if existing is not None:
                return event_document(existing, db)
        evidence = payload.description.strip() or payload.title.strip()
        event = Event(
            index_run_id=run.id, track_id=output.id, source_segment_id=None,
            evidence_segment_ref=payload.external_event_id,
            evidence_text=evidence,
            evidence_start_ms=payload.event_time_ms,
            evidence_end_ms=payload.event_time_ms,
            event_time_ms=payload.event_time_ms,
            playback_time_ms=max(0, payload.event_time_ms - indexing_config.lead_in_ms),
            title=payload.title.strip(), description=payload.description.strip(),
            category=payload.category.strip(), source=EventSource.DEVICE,
            confidence=payload.confidence, importance=1.0,
            status=EventStatus.SUGGESTION,
        )
        db.add(event); db.commit(); db.refresh(event)
        return event_document(event, db)

    @router.post("/sessions/{session_id}/system-events", status_code=status.HTTP_201_CREATED)
    def create_system_event(
        session_id: str, payload: SystemEventCreate, db: Session = Depends(get_db)
    ):
        session = db.get(DebriefSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        system_name = payload.system_name.strip()
        source = db.scalar(select(Track).where(
            Track.session_id == session_id,
            Track.type == TrackType.CHECKLIST,
            Track.name == system_name,
        ))
        if source is None:
            source = Track(
                session_id=session_id, type=TrackType.CHECKLIST, name=system_name,
                source_label="sistem_olayi", duration_ms=session.duration_ms,
                status=TrackStatus.READY,
            )
            db.add(source); db.flush()
        run = db.scalar(select(IndexRun).where(
            IndexRun.source_track_id == source.id,
            IndexRun.llm_provider == "system",
        ).order_by(IndexRun.created_at.desc()))
        if run is None:
            output = Track(
                session_id=session_id, parent_track_id=source.id, type=TrackType.INDEX,
                name=f"{system_name} · Olaylar", source_label="system-event-v1",
                duration_ms=session.duration_ms, status=TrackStatus.READY,
            )
            db.add(output); db.flush()
            run = IndexRun(
                source_track_id=source.id, output_track_id=output.id,
                llm_provider="system", llm_model=system_name, llm_version="system-event-v1",
                prompt_version="none", category_version="001", schema_version="index-v1",
                lead_in_ms=indexing_config.lead_in_ms, processing_ms=0, model_load_ms=0,
                chunk_count=0, failed_chunk_count=0,
                config_fingerprint=indexing_config.fingerprint, status=RunStatus.COMPLETED,
            )
            db.add(run); db.flush()
        else:
            output = db.get(Track, run.output_track_id)
        existing = db.scalar(select(Event).where(
            Event.index_run_id == run.id,
            Event.evidence_segment_ref == payload.external_event_id,
        ))
        if existing is not None:
            return event_document(existing, db)
        evidence = payload.description.strip() or payload.title.strip()
        event = Event(
            index_run_id=run.id, track_id=output.id, source_segment_id=None,
            evidence_segment_ref=payload.external_event_id, evidence_text=evidence,
            evidence_start_ms=payload.event_time_ms, evidence_end_ms=payload.event_time_ms,
            event_time_ms=payload.event_time_ms,
            playback_time_ms=max(0, payload.event_time_ms - indexing_config.lead_in_ms),
            title=payload.title.strip(), description=payload.description.strip(),
            category=payload.category.strip(), source=EventSource.SYSTEM,
            confidence=payload.confidence, importance=1.0, status=EventStatus.SUGGESTION,
        )
        db.add(event); db.commit(); db.refresh(event)
        return event_document(event, db)

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.post("/sessions", response_model=SessionRead, status_code=status.HTTP_201_CREATED)
    def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
        item = DebriefSession(**payload.model_dump())
        db.add(item)
        db.commit()
        db.refresh(item)
        return item

    @router.get("/sessions", response_model=list[SessionRead])
    def list_sessions(db: Session = Depends(get_db)):
        return list(db.scalars(select(DebriefSession).order_by(DebriefSession.created_at.desc())))

    @router.get("/sessions/{session_id}", response_model=SessionRead)
    def get_session(session_id: str, db: Session = Depends(get_db)):
        item = db.get(DebriefSession, session_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        return item

    @router.post("/sessions/{session_id}/instructor-notes", status_code=status.HTTP_201_CREATED)
    def create_instructor_note(
        session_id: str, payload: InstructorNoteCreate, db: Session = Depends(get_db)
    ):
        session = db.get(DebriefSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")

        text = payload.text.strip()
        note_track = Track(
            session_id=session_id, type=TrackType.NOTE,
            name=f"Eğitmen notu · {payload.instructor_name}",
            source_label="yazılı_eğitmen_notu", duration_ms=session.duration_ms,
            status=TrackStatus.READY,
        )
        db.add(note_track); db.flush()
        output = Track(
            session_id=session_id, parent_track_id=note_track.id, type=TrackType.INDEX,
            name=f"{note_track.name} · Bookmark", source_label="eğitmen_notu-v2",
            duration_ms=session.duration_ms, status=TrackStatus.READY,
        )
        db.add(output); db.flush()
        fingerprint = hashlib.sha256(json.dumps({
            "kind": "typed_instructor_note_v2", "session_id": session_id,
            "text": text, "event_time_ms": payload.event_time_ms,
            "instructor_name": payload.instructor_name,
        }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        run = IndexRun(
            source_track_id=note_track.id, output_track_id=output.id,
            llm_provider="deterministic-instructor-note", llm_model="none",
            llm_version="typed-instructor-note-v2", prompt_version="none",
            category_version="001", schema_version="index-v1", lead_in_ms=indexing_config.lead_in_ms,
            processing_ms=0, model_load_ms=0, chunk_count=0, failed_chunk_count=0,
            input_fingerprint=fingerprint, config_fingerprint=indexing_config.fingerprint,
            status=RunStatus.COMPLETED,
        )
        db.add(run); db.flush()
        now = datetime.now(timezone.utc)
        event = Event(
            index_run_id=run.id, track_id=output.id, source_segment_id=None,
            event_time_ms=payload.event_time_ms,
            playback_time_ms=max(0, payload.event_time_ms - indexing_config.lead_in_ms),
            title=text[:200], description=text, category="degerlendirme",
            source=EventSource.INSTRUCTOR_NOTE, confidence=1.0, importance=1.0,
            status=EventStatus.APPROVED, reviewed_by=payload.instructor_name,
            reviewed_at=now,
        )
        db.add(event); db.commit(); db.refresh(event)
        return {"output_track_id": output.id, "event": event_document(event, db)}

    @router.post(
        "/sessions/{session_id}/tracks", response_model=TrackRead, status_code=status.HTTP_201_CREATED
    )
    def create_track(session_id: str, payload: TrackCreate, db: Session = Depends(get_db)):
        if db.get(DebriefSession, session_id) is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        if payload.parent_track_id:
            parent = db.get(Track, payload.parent_track_id)
            if parent is None or parent.session_id != session_id:
                raise HTTPException(status_code=422, detail="Üst kanal aynı oturuma ait olmalıdır")
        item = Track(session_id=session_id, **payload.model_dump())
        db.add(item)
        db.commit()
        db.refresh(item)
        return item

    @router.post(
        "/sessions/{session_id}/media",
        response_model=TrackRead,
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_media(
        session_id: str,
        request: Request,
        x_filename: str = Header(..., min_length=1, max_length=255),
        x_track_name: str = Header(..., min_length=1, max_length=200),
        x_track_type: TrackType = Header(...),
        x_room: str | None = Header(default=None, max_length=120),
        x_source_label: str | None = Header(default=None, max_length=120),
        db: Session = Depends(get_db),
    ):
        if db.get(DebriefSession, session_id) is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        if x_track_type not in {TrackType.VIDEO, TrackType.AUDIO, TrackType.NOTE}:
            raise HTTPException(status_code=422, detail="Bu uç yalnız medya kanalları içindir")

        destination: Path | None = None
        derived_audio: Path | None = None
        try:
            decoded_filename = unquote(x_filename)
            decoded_track_name = unquote(x_track_name)
            destination = destination_for(settings.media_root, session_id, decoded_filename)
            total = 0
            with destination.open("xb") as stream:
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > settings.max_upload_bytes:
                        raise MediaError("Dosya izin verilen yükleme boyutunu aşıyor")
                    stream.write(chunk)
            if total == 0:
                raise MediaError("Boş dosya yüklenemez")

            info = probe_media(destination)
            if x_track_type == TrackType.VIDEO and not info.has_video:
                raise MediaError("Video kanalı için görüntü akışı bulunamadı")
            if x_track_type in {TrackType.AUDIO, TrackType.NOTE} and not info.has_audio:
                raise MediaError("Ses kanalı için ses akışı bulunamadı")

            item = Track(
                session_id=session_id,
                type=x_track_type,
                name=decoded_track_name,
                room=unquote(x_room) if x_room else None,
                source_label=unquote(x_source_label) if x_source_label else None,
                media_path=destination.relative_to(settings.media_root.resolve()).as_posix(),
                original_filename=Path(decoded_filename).name,
                mime_type=request.headers.get("content-type"),
                container_format=info.container_format,
                codec_name=info.codec_name,
                sample_rate=info.sample_rate,
                channel_count=info.channel_count,
                duration_ms=info.duration_ms,
                checksum_sha256=sha256_file(destination),
                status=TrackStatus.READY,
            )
            db.add(item)
            db.flush()
            if x_track_type == TrackType.VIDEO and info.has_audio:
                derived_audio = extract_audio_wav(destination)
                audio_info = probe_media(derived_audio)
                audio_track = Track(
                    session_id=session_id,
                    parent_track_id=item.id,
                    type=TrackType.AUDIO,
                    name=f"{decoded_track_name} · Otomatik ses",
                    room=unquote(x_room) if x_room else None,
                    source_label="ffmpeg-16k-mono",
                    media_path=derived_audio.relative_to(settings.media_root.resolve()).as_posix(),
                    original_filename=f"{Path(decoded_filename).stem}_16k.wav",
                    mime_type="audio/wav",
                    container_format=audio_info.container_format,
                    codec_name=audio_info.codec_name,
                    sample_rate=audio_info.sample_rate,
                    channel_count=audio_info.channel_count,
                    duration_ms=audio_info.duration_ms,
                    checksum_sha256=sha256_file(derived_audio),
                    status=TrackStatus.READY,
                )
                db.add(audio_track)
            db.commit()
            db.refresh(item)
            return item
        except MediaError as exc:
            db.rollback()
            if destination is not None and destination.exists():
                destination.unlink()
            if derived_audio is not None and derived_audio.exists():
                derived_audio.unlink()
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/tracks/{track_id}/media")
    def get_track_media(track_id: str, db: Session = Depends(get_db)):
        item = db.get(Track, track_id)
        if item is None or not item.media_path:
            raise HTTPException(status_code=404, detail="Medya kanalı bulunamadı")
        try:
            path = resolve_stored_media(settings.media_root, item.media_path)
        except MediaError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type=item.mime_type)

    @router.get("/tracks/{track_id}/thumbnail")
    def get_track_thumbnail(
        track_id: str, time_ms: int = 1000, db: Session = Depends(get_db)
    ):
        item = db.get(Track, track_id)
        if item is None or item.type != TrackType.VIDEO or not item.media_path:
            raise HTTPException(status_code=404, detail="Video kanalı bulunamadı")
        try:
            path = resolve_stored_media(settings.media_root, item.media_path)
            safe_time_ms = min(max(0, time_ms), max(0, (item.duration_ms or 0) - 100))
            thumbnail = video_thumbnail(path, safe_time_ms)
        except MediaError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return FileResponse(thumbnail, media_type="image/jpeg")

    @router.get("/tracks/{track_id}/waveform")
    def get_track_waveform(track_id: str, db: Session = Depends(get_db)):
        item = db.get(Track, track_id)
        if item is None or item.type != TrackType.AUDIO or not item.media_path:
            raise HTTPException(status_code=404, detail="Ses kanalı bulunamadı")
        try:
            path = resolve_stored_media(settings.media_root, item.media_path)
            output = audio_waveform(path)
        except MediaError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return FileResponse(output, media_type="image/png")

    @router.get("/tracks/{track_id}/filmstrip")
    def get_track_filmstrip(track_id: str, db: Session = Depends(get_db)):
        item = db.get(Track, track_id)
        if item is None or item.type != TrackType.VIDEO or not item.media_path:
            raise HTTPException(status_code=404, detail="Video kanalı bulunamadı")
        try:
            path = resolve_stored_media(settings.media_root, item.media_path)
            output = video_filmstrip(path, item.duration_ms or 1000)
        except MediaError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return FileResponse(output, media_type="image/jpeg")

    @router.post("/tracks/{track_id}/transcriptions", response_model=TranscriptionRead)
    def transcribe_track(track_id: str, language: str = "tr", db: Session = Depends(get_db)):
        source = db.get(Track, track_id)
        if source is None or source.type not in {TrackType.AUDIO, TrackType.NOTE} or not source.media_path:
            raise HTTPException(status_code=422, detail="Yalnız geçerli ses veya not kanalı transkript edilebilir")
        try:
            media_path = resolve_stored_media(settings.media_root, source.media_path)
            processing_started = time.monotonic()
            result = asr_adapter.transcribe(media_path, language)
            processing_ms = round((time.monotonic() - processing_started) * 1000)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"ASR işlemi başarısız: {type(exc).__name__}") from exc

        output_track = Track(
            session_id=source.session_id,
            parent_track_id=source.id,
            type=TrackType.TRANSCRIPT,
            name=f"{source.name} · ASR İpuçları",
            room=source.room,
            source_label=result.model,
            duration_ms=result.duration_ms,
            status=TrackStatus.READY,
        )
        db.add(output_track)
        db.flush()
        run = TranscriptionRun(
            source_track_id=source.id,
            output_track_id=output_track.id,
            engine=result.engine,
            model=result.model,
            engine_version=result.engine_version,
            config_version=result.config_version,
            language=result.language,
            duration_ms=result.duration_ms,
            processing_ms=processing_ms,
            status=RunStatus.COMPLETED,
        )
        db.add(run)
        db.flush()
        segments = []
        for sequence, segment in enumerate(result.segments, start=1):
            item = TranscriptSegment(
                track_id=output_track.id,
                run_id=run.id,
                segment_ref=segment.ref,
                sequence_no=sequence,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                text=segment.text,
                confidence=segment.confidence,
                asr_engine=result.engine,
                asr_model=result.model,
                asr_version=result.engine_version,
                language=result.language,
            )
            db.add(item)
            segments.append(item)
        db.commit()
        db.refresh(output_track)
        return TranscriptionRead(
            id=run.id,
            source_track_id=source.id,
            output_track=TrackRead.model_validate(output_track),
            engine=run.engine,
            model=run.model,
            engine_version=run.engine_version,
            config_version=run.config_version,
            language=run.language,
            duration_ms=run.duration_ms,
            status=run.status,
            segments=[TranscriptSegmentRead.model_validate(item) for item in segments],
        )

    @router.get("/tracks/{track_id}/segments", response_model=list[TranscriptSegmentRead])
    def list_segments(track_id: str, db: Session = Depends(get_db)):
        track = db.get(Track, track_id)
        if track is None or track.type != TrackType.TRANSCRIPT:
            raise HTTPException(status_code=404, detail="Transkript kanalı bulunamadı")
        query = (
            select(TranscriptSegment)
            .where(TranscriptSegment.track_id == track_id)
            .order_by(TranscriptSegment.sequence_no)
        )
        return list(db.scalars(query))

    @router.post("/tracks/{track_id}/indexes")
    def create_index(track_id: str, db: Session = Depends(get_db)):
        source = db.get(Track, track_id)
        if source is None or source.type != TrackType.TRANSCRIPT:
            raise HTTPException(status_code=422, detail="Yalnız transkript kanalı indekslenebilir")
        parent_media = db.get(Track, source.parent_track_id) if source.parent_track_id else None
        session = db.get(DebriefSession, source.session_id)
        if getattr(llm_adapter, "is_cloud", False) and (
            parent_media is None or parent_media.type != TrackType.NOTE
        ):
            try:
                require_cloud_allowed(session)
            except CloudPrivacyError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
        segments = list(db.scalars(select(TranscriptSegment).where(TranscriptSegment.track_id == track_id).order_by(TranscriptSegment.sequence_no)))
        if not segments:
            raise HTTPException(status_code=422, detail="Transkript segmenti bulunamadı")
        if parent_media is not None and parent_media.type == TrackType.NOTE:
            fingerprint = hashlib.sha256(json.dumps({
                "kind": "instructor_note_v1",
                "segments": [(s.segment_ref, s.start_ms, s.end_ms, s.text) for s in segments],
            }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
            cached = db.scalar(select(IndexRun).where(
                IndexRun.source_track_id == source.id,
                IndexRun.input_fingerprint == fingerprint,
                IndexRun.status == RunStatus.COMPLETED,
            ).order_by(IndexRun.created_at.desc()))
            if cached is not None:
                chapter_rows = list(db.scalars(select(Chapter).where(Chapter.index_run_id == cached.id)))
                event_rows = list(db.scalars(select(Event).where(Event.index_run_id == cached.id).order_by(Event.event_time_ms)))
                return {"index_run_id": cached.id, "output_track_id": cached.output_track_id,
                        "cached": True, "processing_ms": cached.processing_ms,
                        "chapters": [{"title": r.title, "start_ms": r.start_ms, "end_ms": r.end_ms, "summary": r.summary} for r in chapter_rows],
                        "events": [event_document(r, db) for r in event_rows]}
            started = time.monotonic()
            output = Track(
                session_id=source.session_id, parent_track_id=source.id, type=TrackType.INDEX,
                name=f"{source.name} · Eğitmen notu indeksi", source_label="eğitmen_notu-v1",
                duration_ms=source.duration_ms, status=TrackStatus.READY,
            )
            db.add(output); db.flush()
            run = IndexRun(
                source_track_id=source.id, output_track_id=output.id,
                llm_provider="deterministic-instructor-note", llm_model="none",
                llm_version="instructor-note-v1", prompt_version="none",
                category_version="001", schema_version="index-v1", lead_in_ms=indexing_config.lead_in_ms,
                processing_ms=0, model_load_ms=0, chunk_count=1, failed_chunk_count=0,
                input_fingerprint=fingerprint, config_fingerprint=indexing_config.fingerprint,
                status=RunStatus.COMPLETED,
            )
            db.add(run); db.flush()
            first, last = segments[0], segments[-1]
            chapter = Chapter(
                index_run_id=run.id, track_id=output.id,
                start_segment_id=first.id, end_segment_id=last.id,
                start_ms=first.start_ms, end_ms=last.end_ms,
                title="Eğitmen sesli notları", summary="Eğitmenin doğrudan kaydedilmiş notları.",
                confidence=1.0, sequence_no=1,
            )
            db.add(chapter)
            event_rows = []
            reviewed_at = datetime.now(timezone.utc)
            for segment in segments:
                text = segment.text.strip()
                row = Event(
                    index_run_id=run.id, track_id=output.id, source_segment_id=segment.id,
                    evidence_segment_ref=segment.segment_ref, evidence_text=segment.text,
                    evidence_start_ms=segment.start_ms, evidence_end_ms=segment.end_ms,
                    event_time_ms=segment.start_ms, playback_time_ms=max(0, segment.start_ms - indexing_config.lead_in_ms),
                    title=text[:200], description=text, category="degerlendirme",
                    source=EventSource.INSTRUCTOR_NOTE, confidence=1.0, importance=1.0,
                    status=EventStatus.APPROVED, reviewed_by="eğitmen_notu_kaynağı",
                    reviewed_at=reviewed_at,
                )
                db.add(row); event_rows.append(row)
            run.processing_ms = round((time.monotonic() - started) * 1000)
            db.commit()
            return {"index_run_id": run.id, "output_track_id": output.id,
                    "cached": False, "processing_ms": run.processing_ms,
                    "chapters": [{"title": chapter.title, "start_ms": chapter.start_ms,
                                  "end_ms": chapter.end_ms, "summary": chapter.summary}],
                    "events": [event_document(row, db) for row in event_rows]}
        category_path = Path(__file__).resolve().parents[1] / "config" / "debrief" / "categories_v001.yaml"
        category_doc = yaml.safe_load(category_path.read_text(encoding="utf-8"))
        categories = [item["id"] for item in category_doc["categories"]]
        fingerprint_document = {
            "segments": [(s.segment_ref, s.start_ms, s.end_ms, s.text) for s in segments],
            "categories": categories, "prompt_version": "index_v001",
            "indexing_config": indexing_config.fingerprint,
            "adapter": type(llm_adapter).__name__,
            "model": str(getattr(llm_adapter, "model", "unknown")),
        }
        fingerprint = hashlib.sha256(json.dumps(
            fingerprint_document, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        cached = db.scalar(select(IndexRun).where(
            IndexRun.source_track_id == source.id,
            IndexRun.input_fingerprint == fingerprint,
            IndexRun.status == RunStatus.COMPLETED,
        ).order_by(IndexRun.created_at.desc()))
        if cached is not None:
            chapter_rows = list(db.scalars(select(Chapter).where(
                Chapter.index_run_id == cached.id
            ).order_by(Chapter.sequence_no)))
            event_rows = list(db.scalars(select(Event).where(
                Event.index_run_id == cached.id
            ).order_by(Event.event_time_ms)))
            return {"index_run_id": cached.id, "output_track_id": cached.output_track_id,
                    "cached": True, "processing_ms": cached.processing_ms,
                    "chapters": [{"title": r.title, "start_ms": r.start_ms, "end_ms": r.end_ms, "summary": r.summary} for r in chapter_rows],
                    "events": [event_document(r, db) for r in event_rows]}
        try:
            result = llm_adapter.generate_index([{"segment_ref": s.segment_ref, "start_ms": s.start_ms, "end_ms": s.end_ms, "text": s.text} for s in segments], categories)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"İndeks motoru geçici olarak tamamlanamadı: {type(exc).__name__}",
            ) from exc
        try:
            chapters, events = validate_index(
                result.payload, segments, set(categories), indexing_config.lead_in_ms,
                session_duration_ms=source.duration_ms,
            )
        except IndexValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        output = Track(session_id=source.session_id, parent_track_id=source.id, type=TrackType.INDEX, name=f"{source.name} · İndeks", source_label=result.model, duration_ms=source.duration_ms, status=TrackStatus.READY)
        db.add(output); db.flush()
        source_fingerprint = hashlib.sha256(json.dumps(
            fingerprint_document["segments"], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        run = IndexRun(source_track_id=source.id, output_track_id=output.id, llm_provider=result.provider, llm_model=result.model, llm_version=result.version, prompt_version="index_v001", category_version=str(category_doc["version"]), schema_version="index-v1", lead_in_ms=indexing_config.lead_in_ms, status=RunStatus.COMPLETED,
                       processing_ms=result.processing_ms, model_load_ms=result.model_load_ms,
                       chunk_count=result.chunk_count, failed_chunk_count=result.failed_chunk_count,
                       peak_ram_mb=result.peak_ram_mb, input_fingerprint=fingerprint,
                       source_fingerprint=source_fingerprint,
                       config_fingerprint=indexing_config.fingerprint,
                       model_fingerprint=model_fingerprint())
        db.add(run); db.flush()
        chapter_rows, event_rows = [], []
        for number, (item, start, end) in enumerate(chapters, 1):
            row = Chapter(index_run_id=run.id, track_id=output.id, start_segment_id=start.id, end_segment_id=end.id, start_ms=start.start_ms, end_ms=end.end_ms, title=item.title, summary=item.summary, confidence=item.confidence, sequence_no=number)
            db.add(row); chapter_rows.append(row)
        for item, segment, playback_ms in events:
            row = Event(index_run_id=run.id, track_id=output.id, source_segment_id=segment.id,
                        evidence_segment_ref=segment.segment_ref, evidence_text=segment.text,
                        evidence_start_ms=segment.start_ms, evidence_end_ms=segment.end_ms,
                        event_time_ms=segment.start_ms, playback_time_ms=playback_ms,
                        title=item.title, description=item.reason, category=item.category,
                        source=EventSource.LLM, confidence=item.confidence,
                        importance=item.importance, status=EventStatus.SUGGESTION)
            db.add(row); event_rows.append(row)
        db.commit()
        return {"index_run_id": run.id, "output_track_id": output.id, "cached": False,
                "processing_ms": run.processing_ms,
                "chapters": [{"title": r.title, "start_ms": r.start_ms, "end_ms": r.end_ms, "summary": r.summary} for r in chapter_rows], "events": [event_document(r, db) for r in event_rows]}

    @router.post("/tracks/{track_id}/jobs/{kind}", status_code=status.HTTP_202_ACCEPTED)
    def enqueue_processing_job(track_id: str, kind: str, db: Session = Depends(get_db)):
        if processing_queue is None:
            raise HTTPException(status_code=503, detail="Arka plan kuyruğu kullanılamıyor")
        track = db.get(Track, track_id)
        if track is None:
            raise HTTPException(status_code=404, detail="Kanal bulunamadı")
        allowed = {
            "asr": {TrackType.AUDIO, TrackType.NOTE},
            "index": {TrackType.TRANSCRIPT},
            "assets": {TrackType.VIDEO, TrackType.AUDIO},
        }
        if kind not in allowed or track.type not in allowed[kind]:
            raise HTTPException(status_code=422, detail="İş türü bu kanal için geçerli değil")
        existing = db.scalar(select(ProcessingJob).where(
            ProcessingJob.track_id == track_id,
            ProcessingJob.kind == kind,
            ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCEL_REQUESTED]),
        ).order_by(ProcessingJob.created_at.desc()))
        if existing is not None:
            return job_document(existing)
        resumable = db.scalar(select(ProcessingJob).where(
            ProcessingJob.track_id == track_id,
            ProcessingJob.kind == kind,
            ProcessingJob.status == JobStatus.FAILED,
            ProcessingJob.checkpoint_json.is_not(None),
        ).order_by(ProcessingJob.created_at.desc()))
        if resumable is not None:
            resumable.status = JobStatus.QUEUED
            resumable.stage = "Kaydedilmiş chunk üzerinden devam kuyruğunda"
            resumable.attempt_count = 0
            resumable.error_message = None
            resumable.completed_at = None
            db.commit(); db.refresh(resumable)
            processing_queue.submit(resumable.id)
            return job_document(resumable)
        job = ProcessingJob(session_id=track.session_id, track_id=track.id, kind=kind)
        db.add(job); db.commit(); db.refresh(job)

        processing_queue.submit(job.id)
        return job_document(job)

    @router.get("/sessions/{session_id}/jobs")
    def list_processing_jobs(session_id: str, db: Session = Depends(get_db)):
        if db.get(DebriefSession, session_id) is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        rows = db.scalars(select(ProcessingJob).where(
            ProcessingJob.session_id == session_id
        ).order_by(ProcessingJob.created_at.desc()))
        return [job_document(row) for row in rows]

    @router.get("/jobs/{job_id}")
    def get_processing_job(job_id: str, db: Session = Depends(get_db)):
        row = db.get(ProcessingJob, job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="İş bulunamadı")
        return job_document(row)

    @router.post("/jobs/{job_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
    def cancel_processing_job(job_id: str, db: Session = Depends(get_db)):
        row = db.get(ProcessingJob, job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="İş bulunamadı")
        if row.status not in {JobStatus.QUEUED, JobStatus.RUNNING}:
            raise HTTPException(status_code=409, detail="Yalnız kuyruktaki veya çalışan iş iptal edilebilir")
        row.status = JobStatus.CANCEL_REQUESTED
        row.stage = "İptal isteği alındı; güvenli kesme noktası bekleniyor"
        db.commit(); db.refresh(row)
        return job_document(row)

    @router.post("/jobs/{job_id}/retry", status_code=status.HTTP_202_ACCEPTED)
    def retry_processing_job(job_id: str, db: Session = Depends(get_db)):
        row = db.get(ProcessingJob, job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="İş bulunamadı")
        if row.status not in {JobStatus.FAILED, JobStatus.CANCELLED}:
            raise HTTPException(status_code=409, detail="Yalnız başarısız veya iptal edilmiş iş yeniden denenebilir")
        row.status = JobStatus.QUEUED
        row.progress = 0
        row.stage = "Manuel yeniden deneme kuyruğunda"
        row.attempt_count = 0
        row.error_message = None
        row.completed_at = None
        db.commit(); db.refresh(row)
        processing_queue.submit(row.id)
        return job_document(row)

    @router.get("/tracks/{track_id}/events")
    def list_events(track_id: str, db: Session = Depends(get_db)):
        rows = db.scalars(select(Event).where(Event.track_id == track_id).order_by(Event.event_time_ms))
        return [event_document(r, db) for r in rows]

    @router.patch("/events/{event_id}/review")
    def review_event(event_id: str, payload: EventReviewUpdate, db: Session = Depends(get_db)):
        row = db.get(Event, event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Olay bulunamadı")
        row.status = payload.status
        row.reviewed_by = payload.reviewed_by
        row.reviewed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(row)
        return event_document(row, db)

    @router.patch("/events/{event_id}")
    def update_event(event_id: str, payload: EventUpdate, db: Session = Depends(get_db)):
        row = db.get(Event, event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Olay bulunamadı")
        row.title = payload.title.strip()
        row.category = payload.category.strip()
        row.event_time_ms = payload.event_time_ms
        row.playback_time_ms = max(0, payload.event_time_ms - indexing_config.lead_in_ms)
        row.description = payload.description.strip()
        db.commit(); db.refresh(row)
        return event_document(row, db)

    @router.delete("/events/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_event(event_id: str, db: Session = Depends(get_db)):
        row = db.get(Event, event_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Olay bulunamadı")
        db.delete(row); db.commit()

    @router.get("/sessions/{session_id}/tracks", response_model=list[TrackRead])
    def list_tracks(session_id: str, db: Session = Depends(get_db)):
        if db.get(DebriefSession, session_id) is None:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
        query = select(Track).where(Track.session_id == session_id).order_by(Track.created_at)
        return list(db.scalars(query))

    @router.patch("/tracks/{track_id}/offset", response_model=TrackRead)
    def update_track_offset(track_id: str, payload: OffsetUpdate, db: Session = Depends(get_db)):
        item = db.get(Track, track_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Kanal bulunamadı")
        item.offset_ms = payload.offset_ms
        db.commit()
        db.refresh(item)
        return item

    if processing_queue is not None:
        def run_asr_job(track_id, context):
            context.update("ASR hazırlanıyor", 0, 100)
            setattr(asr_adapter, "progress_callback", context.update)
            setattr(asr_adapter, "cancel_check", context.check_cancelled)
            setattr(asr_adapter, "checkpoint_loader", context.load_checkpoint)
            setattr(asr_adapter, "checkpoint_saver", context.save_checkpoint)
            try:
                with processing_queue.session_factory() as worker_db:
                    return transcribe_track(track_id, language="tr", db=worker_db)
            finally:
                for name in ("progress_callback", "cancel_check", "checkpoint_loader", "checkpoint_saver"):
                    if hasattr(asr_adapter, name):
                        delattr(asr_adapter, name)

        def run_index_job(track_id, context):
            context.update("İndeks hazırlanıyor", 0, 100)
            adapter_identity = f"{type(llm_adapter).__name__}:{getattr(llm_adapter, 'model', 'unknown')}"
            saved_checkpoint = context.load_checkpoint()
            saved_identity = saved_checkpoint.get("adapter_identity")
            if saved_identity and saved_identity != adapter_identity:
                raise RuntimeError(
                    "Yarım indeks işi farklı LLM ile sürdürülemez: "
                    f"beklenen {saved_identity}, çalışan {adapter_identity}"
                )
            setattr(llm_adapter, "progress_callback", context.update)
            setattr(llm_adapter, "cancel_check", context.check_cancelled)
            setattr(llm_adapter, "checkpoint_loader", context.load_checkpoint)
            setattr(
                llm_adapter, "checkpoint_saver",
                lambda value: context.save_checkpoint({**value, "adapter_identity": adapter_identity}),
            )
            try:
                with processing_queue.session_factory() as worker_db:
                    return create_index(track_id, db=worker_db)
            finally:
                for name in ("progress_callback", "cancel_check", "checkpoint_loader", "checkpoint_saver"):
                    if hasattr(llm_adapter, name):
                        delattr(llm_adapter, name)

        def run_asset_job(track_id, context):
            context.update("Timeline görseli hazırlanıyor", 10, 100)
            with processing_queue.session_factory() as worker_db:
                worker_track = worker_db.get(Track, track_id)
                if worker_track is None or not worker_track.media_path:
                    raise RuntimeError("Medya kanalı bulunamadı")
                path = resolve_stored_media(settings.media_root, worker_track.media_path)
                context.check_cancelled()
                if worker_track.type == TrackType.VIDEO:
                    output = video_filmstrip(path, worker_track.duration_ms or 1000)
                else:
                    output = audio_waveform(path)
                context.update("Timeline görseli kaydedildi", 95, 100)
                return {"asset": output.name}

        processing_queue.register("asr", run_asr_job)
        processing_queue.register("index", run_index_job)
        processing_queue.register("assets", run_asset_job)
        processing_queue.recover()

    return router
