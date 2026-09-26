"""
tests/paket5/test_rtp_reorder.py

Bounded RTP reorder buffer (maks 3 paket / ~30 ms) için production
implementasyon testleri.

KESİNLİKLE algoritma test dosyasına kopyalanmamıştır; tüm testler
production IntercomEngine._push_to_reorder() ve _drain_in_order()
methodlarını çağırır.
"""
from pathlib import Path
import numpy as np
import pytest
import sys
import pathlib

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
# Yardımcı fonksiyon
# ---------------------------------------------------------------------------

def dummy_frame(val=0.0):
    """160 örneklik float32 ses karesi (test için sabit değerli)."""
    return np.full(160, val, dtype=np.float32)


def push_seq(engine, *seq_vals):
    """Birden fazla seq'i sırayla push eder; her birinin float değeri seq_val/1000."""
    for s in seq_vals:
        engine._push_to_reorder(s, dummy_frame(s / 1000.0))


def drain_play_as_seq(engine):
    """_play kuyruğundaki kareleri çıkarıp 'yaklaşık seq numaraları' listesi döner.
    Her kare dummy_frame(seq/1000) olarak üretildiğinden round(val*1000) = seq.
    """
    result = []
    with engine._queue_lock:
        while engine._play:
            frame = engine._play.popleft()
            result.append(round(float(frame[0]) * 1000))
    return result


# ---------------------------------------------------------------------------
# Test 1: Normal sıralı paketler
# ---------------------------------------------------------------------------

def test_ordered_sequence(engine):
    """100, 101, 102 sırayla gelirse aynı sırada playback'e verilmeli."""
    push_seq(engine, 100, 101, 102)
    played = drain_play_as_seq(engine)
    assert played == [100, 101, 102], f"Beklenen [100,101,102], gelen {played}"


# ---------------------------------------------------------------------------
# Test 2: Küçük out-of-order: 100, 102, 101
# ---------------------------------------------------------------------------

def test_small_reorder_101_after_102(engine):
    """100 gelir, 102 gelir, 101 gelir -> playback 100, 101, 102 sırasında olmalı."""
    push_seq(engine, 100)
    played_after_100 = drain_play_as_seq(engine)
    assert played_after_100 == [100], f"100 hemen oynatılmalı, gelen {played_after_100}"

    push_seq(engine, 102)
    played_after_102 = drain_play_as_seq(engine)
    # 102 geldi ama 101 bekleniyor; henüz oynatılmamalı
    assert played_after_102 == [], f"101 gelmeden 102 oynatılmamalı, gelen {played_after_102}"

    push_seq(engine, 101)
    played_after_101 = drain_play_as_seq(engine)
    # 101 geldi -> 101, 102 ardışık drain
    assert played_after_101 == [101, 102], f"Beklenen [101,102], gelen {played_after_101}"


# ---------------------------------------------------------------------------
# Test 3: Duplicate paket
# ---------------------------------------------------------------------------

def test_duplicate_rejected(engine):
    """100, 101, 101, 102 gelirse 101 yalnızca bir kez oynatılmalı."""
    push_seq(engine, 100, 101, 101, 102)
    played = drain_play_as_seq(engine)
    assert played == [100, 101, 102], f"Beklenen [100,101,102], gelen {played}"


# ---------------------------------------------------------------------------
# Test 4: Missing packet sonsuza kadar bloklamaz
# ---------------------------------------------------------------------------

def test_missing_packet_does_not_block_forever(engine):
    """100, 102, 103, 104 gelirse 101 hiç gelmeden window aştığında devam edilmeli."""
    push_seq(engine, 100)
    drain_play_as_seq(engine)  # 100 drainlandi; next_expected = 101

    # 101 eksik; 102, 103, 104 gönder.
    # Window = 3: {102,103,104} dolunca ve 101 buffer'da yok -> oldest=102'ye atla, drain.
    push_seq(engine, 102)   # buffer={102}, len=1 < 3, bekle
    played_after_102 = drain_play_as_seq(engine)
    assert played_after_102 == [], f"Henüz flush olmamali, gelen {played_after_102}"

    push_seq(engine, 103)   # buffer={102,103}, len=2 < 3, bekle
    played_after_103 = drain_play_as_seq(engine)
    assert played_after_103 == [], f"Henüz flush olmamali, gelen {played_after_103}"

    push_seq(engine, 104)   # buffer={102,103,104}, len=3 >= 3 VE 101 yok -> flush
    played_after_104 = drain_play_as_seq(engine)

    # 102, 103, 104 playback'e verilmeli (101 beklenmemeli)
    assert len(played_after_104) >= 1, (
        f"Eksik 101 nedeniyle playback durmamalıydı, gelen {played_after_104}"
    )
    for s in played_after_104:
        assert s in (102, 103, 104), f"Beklenmedik seq {s} oynatıldı"
    assert 101 not in played_after_104, "Hic gelmemis 101 oynatilmamali"


# ---------------------------------------------------------------------------
# Test 5: Stale / çok eski paket reddediliyor
# ---------------------------------------------------------------------------

def test_stale_packet_rejected(engine):
    """Çok eski sequence numaralı paket buffer'a eklenmemeli."""
    push_seq(engine, 100, 101, 102)
    drain_play_as_seq(engine)  # 100,101,102 drainlandi; next_expected = 103

    # 90 göndermek -> stale, reddedilmeli
    push_seq(engine, 90)
    played = drain_play_as_seq(engine)
    assert played == [], f"Stale paket oynatılmamalı, gelen {played}"

    # Normal devam: 103 kabul edilmeli
    push_seq(engine, 103)
    played2 = drain_play_as_seq(engine)
    assert played2 == [103], f"Normal paket oynatılmalı, gelen {played2}"


# ---------------------------------------------------------------------------
# Test 6: 16-bit wraparound: 65534, 65535, 0, 1
# ---------------------------------------------------------------------------

def test_wraparound_65535_to_0(engine):
    """65534, 65535, 0, 1 sıralı gelirse doğru sırada oynatılmalı."""
    push_seq(engine, 65534, 65535, 0, 1)
    played = drain_play_as_seq(engine)
    assert played == [65534, 65535, 0, 1], (
        f"Wraparound sırası korunmalı, gelen {played}"
    )


# ---------------------------------------------------------------------------
# Test 7: Wraparound + reorder: 65535, 1, 0
# ---------------------------------------------------------------------------

def test_wraparound_reorder(engine):
    """65535 gelir, sonra 1 gelir (0 eksik), sonra 0 gelir.
    Beklenen: 65535 anında oynatılır, 0 ve 1 sıralı drainlanır.
    """
    push_seq(engine, 65535)
    played_first = drain_play_as_seq(engine)
    assert played_first == [65535], f"65535 hemen oynatılmalı, gelen {played_first}"

    # 0 bekleniyor; 1 geldi ama 0 eksik -> bekle
    push_seq(engine, 1)
    played_after_1 = drain_play_as_seq(engine)
    assert played_after_1 == [], f"0 gelmeden 1 oynatılmamalı, gelen {played_after_1}"

    # 0 geldi -> 0, 1 ardışık drain
    push_seq(engine, 0)
    played_after_0 = drain_play_as_seq(engine)
    assert played_after_0 == [0, 1], f"Beklenen [0,1], gelen {played_after_0}"


# ---------------------------------------------------------------------------
# Test 8: Reorder buffer maksimum sınırı aşmıyor
# ---------------------------------------------------------------------------

def test_reorder_buffer_bounded(engine):
    """Buffer'da _RX_REORDER_WINDOW'dan fazla paket birikemez."""
    WINDOW = engine._RX_REORDER_WINDOW

    # İlk paket -> next_expected = 100
    push_seq(engine, 100)
    drain_play_as_seq(engine)  # 100 drainlandi

    # 101 eksik bırak; 102, 103, 104 gönder (window = 3 -> 102,103,104)
    push_seq(engine, 102)
    push_seq(engine, 103)
    push_seq(engine, 104)

    with engine._state_lock:
        buf_size = len(engine._rx_reorder)

    assert buf_size <= WINDOW, (
        f"Buffer en fazla {WINDOW} paket tutmalı, şu an {buf_size}"
    )


# ---------------------------------------------------------------------------
# Test 9: Reset / reconnect reorder state'i temizliyor
# ---------------------------------------------------------------------------

def test_reset_clears_reorder_state(engine):
    """reset_rx_sequence() çağrısı reorder buffer ve next_expected'ı temizlemeli."""
    push_seq(engine, 100, 102)  # 101 eksik, buffer'da 102 beklemede; 100 play'e gider
    drain_play_as_seq(engine)   # _play temizle

    # 102 buffer'da bekliyor
    with engine._state_lock:
        assert (engine._rx_reorder or engine._rx_next_expected is not None)

    engine.reset_rx_sequence()

    with engine._state_lock:
        assert engine._rx_next_expected is None, "_rx_next_expected None olmali"
        assert engine._rx_reorder == {}, "_rx_reorder bos olmali"
        assert engine._last_rx_seq is None, "_last_rx_seq None olmali"

    # Reset sonrası yeni paket stream başlayabilmeli
    push_seq(engine, 10)
    played = drain_play_as_seq(engine)
    assert played == [10], f"Reset sonrası yeni paket kabul edilmeli, gelen {played}"


# ---------------------------------------------------------------------------
# Test 10: _seq16_diff static helper (16-bit modular arithmetic)
# ---------------------------------------------------------------------------

def test_seq16_diff_arithmetic(engine):
    """_seq16_diff() doğru 16-bit modular distance hesaplar."""
    assert engine._seq16_diff(0, 1) == 1
    assert engine._seq16_diff(65535, 0) == 1       # wrap-around
    assert engine._seq16_diff(65535, 1) == 2       # wrap-around + 1
    assert engine._seq16_diff(100, 100) == 0       # same
    assert engine._seq16_diff(100, 99) == 65535    # backward (stale)
    assert engine._seq16_diff(0, 32767) == 32767   # max forward
    assert engine._seq16_diff(0, 32768) == 32768   # border (stale/backward)
