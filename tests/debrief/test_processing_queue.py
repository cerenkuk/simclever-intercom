import time
import threading

from sqlalchemy import select

from debrief.db import Base, build_engine, build_session_factory
from debrief.jobs import ProcessingQueue
from debrief.models import (
    DebriefSession, JobStatus, ProcessingJob, Track, TrackStatus, TrackType,
)


def wait_for(factory, job_id, statuses, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with factory() as db:
            job = db.get(ProcessingJob, job_id)
            if job.status in statuses:
                return job
        time.sleep(0.01)
    raise AssertionError("Kuyruk işi zamanında beklenen duruma gelmedi")


def queue_fixture(tmp_path):
    engine = build_engine(f"sqlite:///{(tmp_path / 'jobs.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    with factory() as db:
        session = DebriefSession(title="Kuyruk")
        db.add(session); db.flush()
        track = Track(session_id=session.id, type=TrackType.AUDIO, name="Ses", status=TrackStatus.READY)
        db.add(track); db.commit()
        return engine, factory, session.id, track.id


def new_job(factory, session_id, track_id, status=JobStatus.QUEUED):
    with factory() as db:
        job = ProcessingJob(session_id=session_id, track_id=track_id, kind="test", status=status)
        db.add(job); db.commit(); db.refresh(job)
        return job.id


def test_queue_retries_once_and_persists_progress(tmp_path):
    engine, factory, session_id, track_id = queue_fixture(tmp_path)
    queue = ProcessingQueue(factory)
    calls = []

    def handler(_track_id, context):
        calls.append(1)
        context.update("Chunk 1/2", 1, 2)
        if len(calls) == 1:
            raise RuntimeError("geçici")
        return {"ok": True}

    queue.register("test", handler)
    job_id = new_job(factory, session_id, track_id)
    queue.submit(job_id)
    job = wait_for(factory, job_id, {JobStatus.COMPLETED})
    assert job.attempt_count == 2
    assert job.progress == 100
    assert job.stage == "Tamamlandı"
    assert calls == [1, 1]
    queue.stop(); engine.dispose()


def test_queue_recovers_running_job_after_restart(tmp_path):
    engine, factory, session_id, track_id = queue_fixture(tmp_path)
    job_id = new_job(factory, session_id, track_id, JobStatus.RUNNING)
    queue = ProcessingQueue(factory)
    queue.register("test", lambda _track, context: (context.update("Devam", 1, 1), {"resumed": True})[1])
    assert queue.recover() == 1
    job = wait_for(factory, job_id, {JobStatus.COMPLETED})
    assert job.attempt_count == 1
    assert "resumed" in job.result_json
    queue.stop(); engine.dispose()


def test_cancel_requested_job_never_runs_handler(tmp_path):
    engine, factory, session_id, track_id = queue_fixture(tmp_path)
    job_id = new_job(factory, session_id, track_id, JobStatus.CANCEL_REQUESTED)
    queue = ProcessingQueue(factory)
    calls = []
    queue.register("test", lambda *_: calls.append(1))
    queue.submit(job_id)
    job = wait_for(factory, job_id, {JobStatus.CANCELLED})
    assert job.status == JobStatus.CANCELLED
    assert calls == []
    queue.stop(); engine.dispose()


def test_shutdown_returns_running_job_to_resume_queue(tmp_path):
    engine, factory, session_id, track_id = queue_fixture(tmp_path)
    job_id = new_job(factory, session_id, track_id)
    queue = ProcessingQueue(factory)
    entered = threading.Event()

    def handler(_track, context):
        entered.set()
        while True:
            context.check_cancelled()
            time.sleep(0.01)

    queue.register("test", handler)
    queue.submit(job_id)
    assert entered.wait(1)
    queue.stop()
    job = wait_for(factory, job_id, {JobStatus.QUEUED})
    assert "devam" in job.stage
    engine.dispose()
