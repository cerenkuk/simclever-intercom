import os
import time
from pathlib import Path

import pytest

import sys, pathlib
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from shared.intercom_engine import IntercomEngine

@pytest.fixture
def config_path(tmp_path):
    # Copy existing teacher config into temp dir
    src = Path(__file__).resolve().parents[2] / "config" / "teacher.json"
    dst = tmp_path / "teacher.json"
    dst.write_text(src.read_text())
    return str(dst)

def test_engine_start_stop(config_path, fake_sounddevice_streams):
    engine = IntercomEngine(config_path)
    engine.start()
    time.sleep(0.1)
    # Engine, _threads listesi dolu ve _running True olmalı
    assert engine._running is True
    assert len(engine._threads) > 0
    assert any(t.is_alive() for t in engine._threads)
    engine.stop()
    assert engine._running is False
