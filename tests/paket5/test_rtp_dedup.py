from pathlib import Path
import pytest
import numpy as np

import sys, pathlib
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from shared.intercom_engine import IntercomEngine, _rtp_packet, _rtp_payload

@pytest.fixture
def config_path(tmp_path):
    src = Path(__file__).resolve().parents[2] / "config" / "teacher.json"
    dst = tmp_path / "teacher.json"
    dst.write_text(src.read_text())
    return str(dst)

def create_pcm_rtp_packet(seq_num, ssrc=54321, frame_size=160):
    pcm_data = np.zeros(frame_size, dtype=np.int16).tobytes()
    return _rtp_packet(seq_num, timestamp=1000, ssrc=ssrc, payload=pcm_data)

def process_rtp_packet_with_engine(engine, packet):
    """Parses RTP header and evaluates acceptance via production helper engine._accept_rx_sequence."""
    seq_num_in, pkt_ssrc, _ = _rtp_payload(packet)
    if seq_num_in is None:
        return False
    if pkt_ssrc == engine._ssrc:
        return False
    return engine._accept_rx_sequence(seq_num_in)

def test_rtp_sequence_dedup_accepts_normal_ascending(config_path):
    engine = IntercomEngine(config_path)
    assert engine._last_rx_seq is None

    pkt1 = create_pcm_rtp_packet(10)
    pkt2 = create_pcm_rtp_packet(11)
    pkt3 = create_pcm_rtp_packet(12)

    assert process_rtp_packet_with_engine(engine, pkt1) is True
    assert engine._last_rx_seq == 10

    assert process_rtp_packet_with_engine(engine, pkt2) is True
    assert engine._last_rx_seq == 11

    assert process_rtp_packet_with_engine(engine, pkt3) is True
    assert engine._last_rx_seq == 12

def test_rtp_sequence_dedup_rejects_duplicate(config_path):
    engine = IntercomEngine(config_path)

    pkt1 = create_pcm_rtp_packet(100)
    pkt1_dup = create_pcm_rtp_packet(100)

    assert process_rtp_packet_with_engine(engine, pkt1) is True
    assert engine._last_rx_seq == 100

    # Duplicate sequence must be rejected by production helper
    assert process_rtp_packet_with_engine(engine, pkt1_dup) is False
    assert engine._last_rx_seq == 100

def test_rtp_sequence_dedup_rejects_old_sequence(config_path):
    engine = IntercomEngine(config_path)

    pkt_latest = create_pcm_rtp_packet(105)
    pkt_old = create_pcm_rtp_packet(102)

    assert process_rtp_packet_with_engine(engine, pkt_latest) is True
    assert engine._last_rx_seq == 105

    # Out-of-order / stale sequence must be rejected by production helper
    assert process_rtp_packet_with_engine(engine, pkt_old) is False
    assert engine._last_rx_seq == 105

def test_rtp_sequence_dedup_accepts_wraparound(config_path):
    engine = IntercomEngine(config_path)

    pkt_65535 = create_pcm_rtp_packet(65535)
    pkt_0 = create_pcm_rtp_packet(0)
    pkt_1 = create_pcm_rtp_packet(1)

    assert process_rtp_packet_with_engine(engine, pkt_65535) is True
    assert engine._last_rx_seq == 65535

    # 65535 -> 0 wraparound must be accepted
    assert process_rtp_packet_with_engine(engine, pkt_0) is True
    assert engine._last_rx_seq == 0

    # Duplicate 0 after wraparound must be rejected
    assert process_rtp_packet_with_engine(engine, pkt_0) is False

    # Ascending 1 after wraparound must be accepted
    assert process_rtp_packet_with_engine(engine, pkt_1) is True
    assert engine._last_rx_seq == 1

def test_rtp_sequence_dedup_resets_on_restart_and_reconnect(config_path):
    engine = IntercomEngine(config_path)

    pkt_500 = create_pcm_rtp_packet(500)
    assert process_rtp_packet_with_engine(engine, pkt_500) is True
    assert engine._last_rx_seq == 500

    # Sequence reset call
    engine.reset_rx_sequence()
    assert engine._last_rx_seq is None

    # New stream packet after reset (e.g. seq 10) must be accepted
    pkt_10 = create_pcm_rtp_packet(10)
    assert process_rtp_packet_with_engine(engine, pkt_10) is True
    assert engine._last_rx_seq == 10
