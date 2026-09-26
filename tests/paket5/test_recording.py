"""tests/paket5/test_recording.py

Kayıt sistemi (Recorder + SidecarWriter) için birim testleri.
Gerçek ses donanımı gerektirmez; sadece dosya sistemi işlemlerini doğrular.
"""
import json
import time
from pathlib import Path

import pytest

import sys
import pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.recording.sidecar import SidecarWriter


class FakeRecorder:
    """SessionRecorder arayüzünü taklit eden sahte nesne."""

    def __init__(self, base):
        self.raw_mic_path = Path(base) / "raw_mic.wav"
        self.reference_spk_path = Path(base) / "reference_spk.wav"
        self.processed_mic_path = Path(base) / "processed_mic.wav"

    def start(self):
        pass

    def stop(self):
        pass


def test_sidecar_initialize(tmp_path):
    """SidecarWriter başlatıldığında sidecar.json dosyasını oluşturmalı."""
    writer = SidecarWriter(str(tmp_path))
    sidecar = tmp_path / "sidecar.json"
    assert sidecar.is_file(), "sidecar.json oluşturulmamış"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["status"] == "initialized"


def test_sidecar_update(tmp_path):
    """update() çağrıldığında sidecar status'ü 'recording' olmalı."""
    writer = SidecarWriter(str(tmp_path))
    fake = FakeRecorder(tmp_path)
    writer.update(fake)
    data = json.loads((tmp_path / "sidecar.json").read_text(encoding="utf-8"))
    assert data["status"] == "recording"
    assert "paths" in data


def test_sidecar_finalize(tmp_path):
    """finalize() çağrıldığında sidecar status'ü 'finished' olmalı."""
    writer = SidecarWriter(str(tmp_path))
    writer.finalize()
    data = json.loads((tmp_path / "sidecar.json").read_text(encoding="utf-8"))
    assert data["status"] == "finished"


def test_recorder_disabled_does_not_crash(tmp_path):
    """Kayıt devre dışıysa start/stop çağrıları hata vermemeli."""
    from shared.recording.recorder import Recorder
    from shared.events.event_bus import EventBus

    cfg = {"enabled": False, "folder": str(tmp_path / "rec")}
    bus = EventBus()
    rec = Recorder(cfg, bus)
    rec.start()  # enabled=False → hiçbir şey yapmamalı
    rec.stop()
