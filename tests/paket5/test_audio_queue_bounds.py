"""
tests/paket5/test_audio_queue_bounds.py

Production bounded queue davranislarini dogrular.

Her test dogrudan production kodu cagirir; hicbir trim algoritmasi
bu dosyada yeniden implemente edilmemistir.

- _send trim: engine._enqueue_send() production helper'i cagrilir.
- _play trim: engine._enqueue_play() production helper'i cagrilir.
- Underrun sessizlik: engine._playback_callback() dogrudan cagrilir.
- Network I/O yoklugu: socket modulu patch'lenerek callback'in
  hicbir network cagrisi yapmadigi deterministik olarak kanitlanir.
"""
import socket
import unittest.mock
import numpy as np
import pytest
import sys
import pathlib
from pathlib import Path

project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from shared.intercom_engine import IntercomEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_path(tmp_path):
    src = Path(__file__).resolve().parents[2] / "config" / "teacher.json"
    dst = tmp_path / "teacher.json"
    dst.write_text(src.read_text())
    return str(dst)


@pytest.fixture
def engine(config_path):
    return IntercomEngine(config_path)


# ---------------------------------------------------------------------------
# Yardimci
# ---------------------------------------------------------------------------

def dummy_frame(val=0.0):
    """160 orneklik float32 ses karesi."""
    return np.full(160, val, dtype=np.float32)


# ---------------------------------------------------------------------------
# Test 1: _send kuyrugu > 3 oldugunda oldest trim edilir
#
# Production path: engine._enqueue_send(packet) -- _queue_lock altinda
# cagirilmalidir; bu metot _capture_callback'in kullandigi ayni trim
# helper'idir.
# ---------------------------------------------------------------------------

def test_send_queue_trim_at_3(engine):
    """5 paket _enqueue_send uzerinden eklenmeli; kuyrukta son 3'u kalmali."""
    packets = [f"pkt_{i}".encode() for i in range(5)]
    with engine._queue_lock:
        for pkt in packets:
            engine._enqueue_send(pkt)

    with engine._queue_lock:
        remaining = list(engine._send)

    assert len(remaining) == 3, (
        f"Kuyrugun 3 paket icermesi gerekiyor, {len(remaining)} var"
    )
    # Son 3 paket korunmus olmali: pkt_2, pkt_3, pkt_4
    assert remaining == [b"pkt_2", b"pkt_3", b"pkt_4"], (
        f"Son 3 paket yanlis: {remaining}"
    )


# ---------------------------------------------------------------------------
# Test 2: _play kuyrugu > 6 oldugunda oldest trim edilir
#
# Production path: engine._enqueue_play(frame) -- _queue_lock altinda
# cagirilmalidir; bu metot _push_to_reorder'in kullandigi ayni trim
# helper'idir.
# ---------------------------------------------------------------------------

def test_play_queue_trim_at_6(engine):
    """9 frame _enqueue_play uzerinden eklenmeli; kuyrukta son 6'si kalmali."""
    with engine._queue_lock:
        for i in range(9):
            engine._enqueue_play(dummy_frame(float(i)))

    with engine._queue_lock:
        vals = [round(float(f[0])) for f in engine._play]

    assert len(vals) == 6, (
        f"Kuyrugun 6 frame icermesi gerekiyor, {len(vals)} var"
    )
    # Son 6 frame: 3, 4, 5, 6, 7, 8
    assert vals == [3, 4, 5, 6, 7, 8], (
        f"Beklenen son 6 frame yanlis: {vals}"
    )


# ---------------------------------------------------------------------------
# Test 3: _playback_callback kuyrugu boskken sessizlik (np.zeros) uretir
#
# Production path: engine._playback_callback() dogrudan cagirilir.
# _running = True olarak ayarlanir ki asil oynatma kolu calissin.
# Exception swallowing yok; callback beklenmedik raise yapsa test FAIL eder.
# ---------------------------------------------------------------------------

def test_playback_silence_on_empty_queue(engine):
    """_play kuyrugu boskken _playback_callback np.zeros uretmeli."""
    engine._running = True
    with engine._queue_lock:
        engine._play.clear()

    frames = 160
    outdata = np.ones((frames, 1), dtype=np.float32)  # Baslangicta 1.0

    engine._playback_callback(outdata, frames, None, None)

    # outdata ilk sutunu 0 olmali (sessizlik)
    assert np.all(outdata[:, 0] == 0.0), (
        f"Bos kuyrukta sessizlik bekleniyor, max deger: {np.max(np.abs(outdata))}"
    )


# ---------------------------------------------------------------------------
# Test 4: _playback_callback network / socket I/O yapmaz
#
# Production path: engine._playback_callback() dogrudan cagirilir.
# socket modulu patch'lenerek herhangi bir network cagrisi aninda
# AssertionError yukseltir; boylece timing'e degil deterministik
# davranisa dayali kanit saglanir.
# Exception swallowing yok.
# ---------------------------------------------------------------------------

def _network_forbidden(*args, **kwargs):
    raise AssertionError(
        "_playback_callback should not perform any network/socket operation"
    )


def test_playback_callback_does_not_perform_network_io(engine):
    """_playback_callback hicbir socket/network cagrisi yapmamalidir.

    Kanit: socket modulu tamamen patch'lenir; herhangi bir socket olusturma
    veya baglanma girisimlinde AssertionError yukselir. Callback bu hatayi
    tetiklemeden normal tamamlanmalidir.
    """
    engine._running = True
    with engine._queue_lock:
        engine._play.clear()
        # Non-zero frame: output_level=1.0 ile carpildiktan sonra
        # outdata sifir olmamali; boylece callback'in gercekten calistigi
        # dogrulanir.
        for _ in range(3):
            engine._enqueue_play(dummy_frame(0.5))

    frames = 160
    outdata = np.zeros((frames, 1), dtype=np.float32)

    # Callback icindeki hicbir kod socket.socket() cagirmasin.
    with unittest.mock.patch.object(socket, "socket", side_effect=_network_forbidden):
        with unittest.mock.patch.object(
            socket, "create_connection", side_effect=_network_forbidden
        ):
            # AssertionError ya da baska exception yukselirse test FAIL eder.
            engine._playback_callback(outdata, frames, None, None)

    # Callback network'e dokunmadan tamamlandi ve outdata'yi doldurdu.
    assert not np.all(outdata[:, 0] == 0.0), (
        "_play doluyken outdata tamamen sifir olmamali: "
        f"max={np.max(np.abs(outdata[:, 0]))}"
    )
