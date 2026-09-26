import json
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf


class SessionRecorder:
    """Write raw, reference, and processed mono streams for one session."""

    def __init__(self, output_dir, sample_rate=16000):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_mic_path = self.output_dir / "raw_mic.wav"
        self.reference_spk_path = self.output_dir / "reference_spk.wav"
        self.processed_mic_path = self.output_dir / "processed_mic.wav"
        self.raw_file = None
        self.ref_file = None
        self.proc_file = None

    def start(self):
        common = {
            "mode": "w",
            "samplerate": self.sample_rate,
            "channels": 1,
            "subtype": "PCM_16",
        }
        self.raw_file = sf.SoundFile(self.raw_mic_path, **common)
        self.ref_file = sf.SoundFile(self.reference_spk_path, **common)
        self.proc_file = sf.SoundFile(self.processed_mic_path, **common)

    def write_raw_mic(self, data):
        if self.raw_file:
            self.raw_file.write(data)

    def write_ref_spk(self, data):
        if self.ref_file:
            self.ref_file.write(data)

    def write_processed_mic(self, data):
        if self.proc_file:
            self.proc_file.write(data)

    def stop(self):
        for name in ("raw_file", "ref_file", "proc_file"):
            stream = getattr(self, name)
            if stream:
                stream.close()
                setattr(self, name, None)


class Recorder:
    """Thread-safe raw/reference/processed recorder with per-session sidecar."""

    def __init__(self, cfg, event_bus):
        self.cfg = dict(cfg or {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.folder = Path(self.cfg.get("folder", "recordings"))
        self.event_bus = event_bus
        self._lock = threading.RLock()
        self._recorder = None
        self._sidecar_path = None
        self._meta = None

    @property
    def active(self):
        return self._recorder is not None

    def start(self):
        with self._lock:
            if self.active or not self.enabled:
                return False
            free = shutil.disk_usage(self.folder.parent if self.folder.parent.exists() else Path.cwd()).free
            if free < int(self.cfg.get("minimum_free_bytes", 100 * 1024 * 1024)):
                self.event_bus.emit("error", {"message": "Kayıt için yeterli disk alanı yok."})
                return False
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = self.folder / f"recording_{stamp}"
            self._recorder = SessionRecorder(str(session_dir), int(self.cfg.get("sample_rate", 16000)))
            self._recorder.start()
            self._sidecar_path = session_dir / "recording.json"
            now = datetime.now().astimezone()
            self._meta = {
                "schema_version": "1.0", "recording_id": str(uuid.uuid4()), "status": "recording",
                "started_at_local": now.isoformat(), "started_at_utc": now.astimezone(timezone.utc).isoformat(),
                "sample_rate": int(self.cfg.get("sample_rate", 16000)), "channels": 1,
                "sample_width_bytes": 2, "total_samples": 0, "duration_seconds": 0.0,
                "interruptions": [], "files": {"raw":"raw_mic.wav", "reference":"reference_spk.wav", "processed":"processed_mic.wav"}
            }
            self._write_sidecar()
        self.event_bus.emit("recording_started", {"message": "Ham kayıt başladı.", "folder": str(session_dir)})
        return True

    def write(self, raw=None, reference=None, processed=None):
        with self._lock:
            if not self.active:
                return
            raw_length = 0
            if raw is not None:
                data = np.asarray(raw, dtype=np.float32)
                raw_length = len(data)
                self._recorder.write_raw_mic(data)
                self._meta["total_samples"] += len(data)
                self._meta["duration_seconds"] = self._meta["total_samples"] / self._meta["sample_rate"]
            if reference is not None:
                self._recorder.write_ref_spk(np.asarray(reference, dtype=np.float32))
            if processed is not None:
                self._recorder.write_processed_mic(np.asarray(processed, dtype=np.float32))
            if raw_length and self._meta["total_samples"] % (self._meta["sample_rate"] * 5) < raw_length:
                self._write_sidecar()

    def add_interruption(self, kind, duration_seconds, detail=""):
        with self._lock:
            if not self.active:
                return
            samples = max(0, round(duration_seconds * self._meta["sample_rate"]))
            silence = np.zeros(samples, dtype=np.float32)
            self._recorder.write_raw_mic(silence)
            self._recorder.write_processed_mic(silence)
            self._meta["total_samples"] += samples
            self._meta["duration_seconds"] = self._meta["total_samples"] / self._meta["sample_rate"]
            self._meta["interruptions"].append({"type":kind, "duration_seconds":duration_seconds, "inserted_silence_samples":samples, "detail":detail})
            self._write_sidecar()

    def stop(self):
        with self._lock:
            if not self.active:
                return
            self._recorder.stop()
            self._recorder = None
            now = datetime.now().astimezone()
            self._meta.update({"status":"completed", "ended_at_local":now.isoformat(), "ended_at_utc":now.astimezone(timezone.utc).isoformat()})
            self._write_sidecar()
        self.event_bus.emit("recording_stopped", {"message": "Kayıt güvenli biçimde durduruldu.", "sidecar":str(self._sidecar_path)})

    def _write_sidecar(self):
        if not self._sidecar_path or self._meta is None:
            return
        self._sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._sidecar_path.with_suffix(".json.tmp")
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(self._meta, stream, ensure_ascii=False, indent=2)
        temp.replace(self._sidecar_path)
