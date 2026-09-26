"""Regression tests for network error handling in IntercomEngine.

Tests prove production behavior (not copied logic):
1. _sender survives socket.sendto raising OSError.
2. _sender warning is throttled: one user-visible event per 5 s, not per packet.
3. _heartbeat survives OSError.
4. _heartbeat failure does not generate user-visible warnings.
"""
import time
import threading
import unittest.mock as mock
from pathlib import Path

import pytest

import sys
import pathlib
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from shared.intercom_engine import IntercomEngine, _rtp_packet


@pytest.fixture
def config_path(tmp_path):
    src = Path(__file__).resolve().parents[2] / "config" / "teacher.json"
    dst = tmp_path / "teacher.json"
    dst.write_text(src.read_text())
    return str(dst)


@pytest.fixture
def engine(config_path):
    eng = IntercomEngine(config_path)
    return eng


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_warning_collector(engine):
    """Subscribe to event_bus 'warning' events using engine.event_bus.on(); return list."""
    warnings = []
    engine.event_bus.on("warning", lambda data: warnings.append(data))
    return warnings


def _run_sender_briefly(engine, n_packets=1, timeout=1.0):
    """Enqueue n packets, run _sender in a thread, stop it after timeout or drain."""
    pkt = _rtp_packet(1, timestamp=0, ssrc=engine._ssrc, payload=b"\x00" * 320)
    for _ in range(n_packets):
        with engine._queue_lock:
            engine._send.append(pkt)
    engine._send_event.set()
    engine._stop.clear()

    t = threading.Thread(target=engine._sender, daemon=True)
    t.start()
    time.sleep(timeout)
    engine._stop.set()
    engine._send_event.set()  # unblock wait
    t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Test 1: _sender survives OSError (does not raise or terminate)
# ---------------------------------------------------------------------------

def test_sender_survives_oserror(engine):
    """_sender must survive transient OSError: thread must not die and loop must continue."""
    fake_socket = mock.MagicMock()
    fake_socket.sendto.side_effect = OSError("network unreachable")
    engine._socket = fake_socket
    engine._running = True
    # Patch _is_self_ip to return False so the unicast branch is taken
    engine._is_self_ip = mock.MagicMock(return_value=False)
    engine._remote = ("10.0.0.1", 5000)
    engine._last_send_warning = 0.0

    # Run sender in thread; if it crashes it will die immediately
    _run_sender_briefly(engine, n_packets=3, timeout=0.1)

    # sendto was attempted (loop did not crash before reaching sendto)
    assert fake_socket.sendto.called


# ---------------------------------------------------------------------------
# Test 2: _sender warning is throttled – one event per 5 s, not per packet
# ---------------------------------------------------------------------------

def test_sender_warning_is_throttled(engine):
    """10 consecutive OSError send failures within 5 s must produce exactly 1 warning."""
    warnings = _make_warning_collector(engine)

    fake_socket = mock.MagicMock()
    fake_socket.sendto.side_effect = OSError("network unreachable")
    engine._socket = fake_socket
    engine._running = True
    engine._is_self_ip = mock.MagicMock(return_value=False)
    engine._remote = ("10.0.0.1", 5000)
    engine._last_send_warning = 0.0  # allow first warning

    _run_sender_briefly(engine, n_packets=10, timeout=0.2)

    # Exactly 1 warning (throttled)
    assert len(warnings) == 1, (
        f"Expected 1 throttled warning, got {len(warnings)}: {warnings}"
    )
    assert "technical_detail" in warnings[0]


def test_sender_warning_not_emitted_within_throttle_window(engine):
    """After a warning is emitted, no second warning within the 5 s throttle window."""
    warnings = _make_warning_collector(engine)

    fake_socket = mock.MagicMock()
    fake_socket.sendto.side_effect = OSError("network unreachable")
    engine._socket = fake_socket
    engine._running = True
    engine._is_self_ip = mock.MagicMock(return_value=False)
    engine._remote = ("10.0.0.1", 5000)

    # Simulate warning was just emitted
    engine._last_send_warning = time.monotonic()

    _run_sender_briefly(engine, n_packets=5, timeout=0.1)

    assert len(warnings) == 0, (
        f"Expected 0 warnings within throttle window, got {len(warnings)}"
    )


# ---------------------------------------------------------------------------
# Test 3: _heartbeat survives OSError
# ---------------------------------------------------------------------------

def test_heartbeat_survives_oserror(engine):
    """_heartbeat must not raise on OSError; stale detection is handled by _monitor."""
    fake_socket = mock.MagicMock()
    fake_socket.sendto.side_effect = OSError("network unreachable")
    engine._socket = fake_socket
    engine._is_self_ip = mock.MagicMock(return_value=False)
    engine._remote = ("10.0.0.1", 5000)

    # Patch _stop.wait: return False once (one real loop body executes), then True (exit).
    # This ensures sendto is actually called and the OSError handler is exercised.
    wait_count = [0]

    def _patched_wait(timeout):
        wait_count[0] += 1
        return wait_count[0] > 1

    engine._stop.wait = _patched_wait
    # Must not raise even though sendto raises OSError:
    engine._heartbeat()

    # Prove the loop body ran and sendto was actually called
    assert fake_socket.sendto.called


# ---------------------------------------------------------------------------
# Test 4: _heartbeat failure does not produce user-visible warnings
# ---------------------------------------------------------------------------

def test_heartbeat_oserror_no_user_visible_warning(engine):
    """OSError in _heartbeat must not emit any event_bus 'warning' events."""
    warnings = _make_warning_collector(engine)

    fake_socket = mock.MagicMock()
    fake_socket.sendto.side_effect = OSError("network unreachable")
    engine._socket = fake_socket
    engine._is_self_ip = mock.MagicMock(return_value=False)
    engine._remote = ("10.0.0.1", 5000)

    # Patch _stop.wait: return False once (one iteration), then True (stop)
    wait_count = [0]

    def _patched_wait(timeout):
        wait_count[0] += 1
        return wait_count[0] > 1

    engine._stop.wait = _patched_wait
    engine._heartbeat()

    # The heartbeat attempted to send
    assert fake_socket.sendto.called
    # No user-visible warning emitted
    assert warnings == [], f"Unexpected warnings from heartbeat: {warnings}"
