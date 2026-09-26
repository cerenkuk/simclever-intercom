from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
import threading

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from debrief.models import JobStatus, ProcessingJob


LOGGER = logging.getLogger("uvicorn.error")


class JobCancelled(RuntimeError):
    pass


class JobInterrupted(RuntimeError):
    pass


class JobContext:
    def __init__(self, session_factory, job_id: str, stop_event):
        self.session_factory = session_factory
        self.job_id = job_id
        self.stop_event = stop_event

    def update(self, stage: str, current: int = 0, total: int = 100) -> None:
        if self.stop_event.is_set():
            raise JobInterrupted("Uygulama kapanıyor")
        with self.session_factory() as db:
            job = db.get(ProcessingJob, self.job_id)
            if job is None or job.status == JobStatus.CANCEL_REQUESTED:
                raise JobCancelled("İş kullanıcı tarafından iptal edildi")
            job.stage = stage[:120]
            job.progress = max(0, min(99, round(current * 100 / max(1, total))))
            job.heartbeat_at = datetime.now(timezone.utc)
            db.commit()

    def check_cancelled(self) -> None:
        if self.stop_event.is_set():
            raise JobInterrupted("Uygulama kapanıyor")
        with self.session_factory() as db:
            job = db.get(ProcessingJob, self.job_id)
            if job is None or job.status == JobStatus.CANCEL_REQUESTED:
                raise JobCancelled("İş kullanıcı tarafından iptal edildi")

    def load_checkpoint(self) -> dict:
        with self.session_factory() as db:
            job = db.get(ProcessingJob, self.job_id)
            return json.loads(job.checkpoint_json) if job and job.checkpoint_json else {}

    def save_checkpoint(self, value: dict) -> None:
        with self.session_factory() as db:
            job = db.get(ProcessingJob, self.job_id)
            if job is None or job.status == JobStatus.CANCEL_REQUESTED:
                raise JobCancelled("İş kullanıcı tarafından iptal edildi")
            job.checkpoint_json = json.dumps(value, ensure_ascii=False)
            job.heartbeat_at = datetime.now(timezone.utc)
            db.commit()


class ProcessingQueue:
    """Persistent single-worker CPU queue with recovery, retry and cancellation."""

    def __init__(self, session_factory, workers: int = 1):
        self.session_factory = session_factory
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="debrief-job")
        self.handlers = {}
        self._submitted = set()
        self._lock = threading.Lock()
        self._stopping = False
        self._stop_event = threading.Event()

    def register(self, kind: str, handler) -> None:
        self.handlers[kind] = handler

    def recover(self) -> int:
        try:
            with self.session_factory() as db:
                rows = list(db.scalars(select(ProcessingJob).where(
                    ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
                ).order_by(ProcessingJob.created_at)))
                for job in rows:
                    job.status = JobStatus.QUEUED
                    job.stage = "Uygulama yeniden başlatıldı; iş devam kuyruğuna alındı"
                db.commit()
        except SQLAlchemyError:
            LOGGER.warning("İş kuyruğu recovery atlandı; önce alembic upgrade head çalıştırılmalı")
            return 0
        for job in rows:
            self.submit(job.id)
        return len(rows)

    def submit(self, job_id: str) -> None:
        with self._lock:
            if self._stopping or job_id in self._submitted:
                return
            self._submitted.add(job_id)
        self.executor.submit(self._run, job_id)

    def _run(self, job_id: str) -> None:
        retry = False
        try:
            with self.session_factory() as db:
                job = db.get(ProcessingJob, job_id)
                if job is None:
                    return
                if job.status == JobStatus.CANCEL_REQUESTED:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    return
                handler = self.handlers.get(job.kind)
                if handler is None:
                    raise RuntimeError(f"Kayıtlı işleyici yok: {job.kind}")
                job.status = JobStatus.RUNNING
                job.attempt_count += 1
                job.progress = max(1, job.progress)
                job.stage = "İş başlatıldı"
                job.started_at = job.started_at or datetime.now(timezone.utc)
                job.heartbeat_at = datetime.now(timezone.utc)
                track_id, kind = job.track_id, job.kind
                db.commit()
            context = JobContext(self.session_factory, job_id, self._stop_event)
            result = handler(track_id, context)
            if hasattr(result, "model_dump"):
                result = result.model_dump(mode="json")
            context.check_cancelled()
            with self.session_factory() as db:
                job = db.get(ProcessingJob, job_id)
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.stage = "Tamamlandı"
                job.result_json = json.dumps(result, ensure_ascii=False, default=str)
                job.error_message = None
                job.heartbeat_at = job.completed_at = datetime.now(timezone.utc)
                db.commit()
        except JobCancelled:
            with self.session_factory() as db:
                job = db.get(ProcessingJob, job_id)
                if job:
                    job.status = JobStatus.CANCELLED
                    job.stage = "Kullanıcı tarafından iptal edildi"
                    job.completed_at = datetime.now(timezone.utc)
                    db.commit()
        except JobInterrupted:
            with self.session_factory() as db:
                job = db.get(ProcessingJob, job_id)
                if job:
                    job.status = JobStatus.QUEUED
                    job.stage = "Uygulama kapandı; yeniden açıldığında ara kayıttan devam edecek"
                    job.heartbeat_at = datetime.now(timezone.utc)
                    db.commit()
        except Exception as exc:
            if self._stopping:
                LOGGER.info("Arka plan işi kapanış kesme noktasında durdu: %s", job_id)
            else:
                LOGGER.exception("Arka plan işi başarısız: %s", job_id)
            with self.session_factory() as db:
                job = db.get(ProcessingJob, job_id)
                if job:
                    job.error_message = f"{type(exc).__name__}: {exc}"
                    job.heartbeat_at = datetime.now(timezone.utc)
                    if self._stopping:
                        # Kapanış sırasında adapter/decoder tarafından üretilen
                        # hata gerçek bir iş hatası değildir. Checkpoint korunur.
                        job.status = JobStatus.QUEUED
                        job.stage = "Uygulama kapandı; ara kayıttan devam edecek"
                        job.completed_at = None
                        job.error_message = None
                    elif job.attempt_count < job.max_attempts:
                        job.status = JobStatus.QUEUED
                        job.stage = f"Geçici hata; yeniden denenecek ({job.attempt_count}/{job.max_attempts})"
                        retry = True
                    else:
                        job.status = JobStatus.FAILED
                        job.stage = "Deneme sınırı aşıldı"
                        job.completed_at = datetime.now(timezone.utc)
                    db.commit()
        finally:
            with self._lock:
                self._submitted.discard(job_id)
            if retry:
                self.submit(job_id)

    def stop(self) -> None:
        self._stopping = True
        self._stop_event.set()
        with self.session_factory() as db:
            rows = db.scalars(select(ProcessingJob).where(ProcessingJob.status == JobStatus.RUNNING))
            for job in rows:
                job.status = JobStatus.QUEUED
                job.stage = "Uygulama kapandı; yeniden açıldığında devam edecek"
            db.commit()
        self.executor.shutdown(wait=False, cancel_futures=True)
