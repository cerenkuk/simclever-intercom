"""tests/paket5/test_k8_drift_click.py

K8 Referans Ses (Her saniye tık) Kayma ve Süre Bütünlüğü Testi
30 saniyelik referans tık sinyali kaydedilir, diske yazılan WAV dosyası ve sidecar JSON incelenir.
Tıkların örnek indekslerinin ve sürelerinin sıfır kayma (drift < 0.001s) ile korunduğu doğrulanır.
"""
import json
import os
import pathlib
import sys

import numpy as np
import pytest
import soundfile as sf

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.recording.recorder import Recorder
from shared.events.event_bus import EventBus


def test_k8_reference_click_drift(tmp_path):
    """Her 1 saniyede (16000 örnekte) 1 ms tık sinyali kaydedilir ve sıfır kayma olduğu doğrulanır."""
    sample_rate = 16000
    duration_sec = 30
    frame_size = 160
    total_frames = (duration_sec * sample_rate) // frame_size

    rec_folder = tmp_path / "recordings"
    event_bus = EventBus()
    recorder = Recorder(
        {
            "enabled": True,
            "folder": str(rec_folder),
            "sample_rate": sample_rate,
            "sidecar_update_interval_sec": 1,
            "minimum_free_bytes": 10 * 1024 * 1024,
        },
        event_bus=event_bus,
    )

    recorder.start()
    assert recorder.active

    click_positions = []
    current_sample = 0

    # 30 saniye boyunca her saniye başında (16000. örnekte) 16 örneklik tık üret
    for frame_idx in range(total_frames):
        raw_frame = np.zeros(frame_size, dtype=np.float32)
        ref_frame = np.zeros(frame_size, dtype=np.float32)
        proc_frame = np.zeros(frame_size, dtype=np.float32)

        # Her 16000 örnekte bir tık (1 saniyede bir)
        if current_sample % 16000 == 0:
            click_positions.append(current_sample)
            raw_frame[:16] = 0.8  # 1 ms genlikli tık

        recorder.write(raw=raw_frame, reference=ref_frame, processed=proc_frame)
        current_sample += frame_size

    rec_dir = pathlib.Path(recorder._recorder.output_dir)
    recorder.stop()

    # Diskteki kaydı kontrol et
    raw_wav = rec_dir / "raw_mic.wav"
    sidecar_json = rec_dir / "recording.json"

    assert raw_wav.is_file()
    assert sidecar_json.is_file()

    # WAV dosyasını oku ve süresini doğrula
    data, sr = sf.read(str(raw_wav))
    assert sr == sample_rate

    actual_samples = len(data)
    expected_samples = duration_sec * sample_rate
    assert actual_samples == expected_samples

    actual_duration = actual_samples / float(sample_rate)
    assert abs(actual_duration - duration_sec) < 0.001  # < 1 ms fark!

    # Tıkların dosyadaki örnek konumlarını kontrol et
    for click_sample in click_positions:
        # Tık genliği 0.8 olmalı
        assert data[click_sample] > 0.5
        # Tıktan hemen önceki örnek 0 olmalı
        if click_sample > 0:
            assert abs(data[click_sample - 1]) < 0.01

    # Sidecar JSON içeriğini kontrol et
    meta = json.loads(sidecar_json.read_text(encoding="utf-8"))
    assert meta["sample_rate"] == 16000
    assert meta["total_samples"] == expected_samples
    assert abs(meta["duration_seconds"] - duration_sec) < 0.001
    assert "started_at_utc" in meta
    assert "started_at_local" in meta
    assert isinstance(meta.get("interruptions"), list)
