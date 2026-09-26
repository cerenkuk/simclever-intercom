"""Regression tests for WASAPI COM resource lifecycle and cleanup in query_wasapi_endpoints().

Tests verify production behavior and resource cleanup:
1. Return contract of query_wasapi_endpoints() and list_audio_devices().
2. Balancing CoInitializeEx and CoUninitialize (S_OK, S_FALSE).
3. RPC_E_CHANGED_MODE handling: COM usable, CoUninitialize NOT called.
4. Negative initialization failure: early return, CoUninitialize NOT called.
5. Cleanup execution on exception during enumeration.
6. Secondary cleanup errors (e.g. CoUninitialize / PropVariantClear / Release failures) do not crash the application.
"""
import sys
import pathlib
import unittest.mock as mock
from pathlib import Path

import pytest

project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from shared.intercom_engine import IntercomEngine, query_wasapi_endpoints


@pytest.fixture
def config_path(tmp_path):
    src = Path(__file__).resolve().parents[2] / "config" / "teacher.json"
    dst = tmp_path / "teacher.json"
    dst.write_text(src.read_text())
    return str(dst)


# ---------------------------------------------------------------------------
# Test 1: Return type and contract of query_wasapi_endpoints
# ---------------------------------------------------------------------------

def test_query_wasapi_endpoints_contract():
    """query_wasapi_endpoints returns a list of strings on Windows, or [] on non-win32."""
    capture_devices = query_wasapi_endpoints(1)
    render_devices = query_wasapi_endpoints(0)

    assert isinstance(capture_devices, list)
    assert isinstance(render_devices, list)
    for name in capture_devices:
        assert isinstance(name, str)
    for name in render_devices:
        assert isinstance(name, str)


def test_list_audio_devices_preserves_contract(config_path, fake_sounddevice_streams):
    """list_audio_devices() must return dictionary with 'inputs' and 'outputs' lists."""
    engine = IntercomEngine(config_path)
    devices = engine.list_audio_devices()

    assert isinstance(devices, dict)
    assert "inputs" in devices
    assert "outputs" in devices
    assert len(devices["inputs"]) >= 1
    assert len(devices["outputs"]) >= 1
    assert devices["inputs"][0] == (None, "Sistem Varsayılanı")
    assert devices["outputs"][0] == (None, "Sistem Varsayılanı")


# ---------------------------------------------------------------------------
# Test 2: COM Initialization and Uninitialization balance (S_OK and S_FALSE)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows only COM tests")
def test_wasapi_init_cleanup_balance_s_ok():
    """When CoInitializeEx returns S_OK (0), CoUninitialize must be called exactly once."""
    import ctypes

    mock_ole32 = mock.MagicMock()
    mock_ole32.CoInitializeEx.return_value = 0  # S_OK
    mock_ole32.CoCreateInstance.return_value = -2147467259  # E_FAIL (fails immediately)

    with mock.patch.object(ctypes.windll, "ole32", mock_ole32):
        result = query_wasapi_endpoints(1)
        assert result == []

    mock_ole32.CoInitializeEx.assert_called_once_with(None, 0)
    mock_ole32.CoUninitialize.assert_called_once()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only COM tests")
def test_wasapi_init_cleanup_balance_s_false():
    """When CoInitializeEx returns S_FALSE (1), CoUninitialize must still be called."""
    import ctypes

    mock_ole32 = mock.MagicMock()
    mock_ole32.CoInitializeEx.return_value = 1  # S_FALSE (already init on thread)
    mock_ole32.CoCreateInstance.return_value = -2147467259  # E_FAIL

    with mock.patch.object(ctypes.windll, "ole32", mock_ole32):
        result = query_wasapi_endpoints(0)
        assert result == []

    mock_ole32.CoInitializeEx.assert_called_once_with(None, 0)
    mock_ole32.CoUninitialize.assert_called_once()


# ---------------------------------------------------------------------------
# Test 3: RPC_E_CHANGED_MODE handling
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows only COM tests")
def test_wasapi_changed_mode_does_not_couninitialize():
    """When CoInitializeEx returns RPC_E_CHANGED_MODE, CoUninitialize MUST NOT be called."""
    import ctypes

    mock_ole32 = mock.MagicMock()
    mock_ole32.CoInitializeEx.return_value = -2147417850  # RPC_E_CHANGED_MODE (0x80010106)
    mock_ole32.CoCreateInstance.return_value = -2147467259  # E_FAIL

    with mock.patch.object(ctypes.windll, "ole32", mock_ole32):
        result = query_wasapi_endpoints(1)
        assert result == []

    mock_ole32.CoInitializeEx.assert_called_once_with(None, 0)
    mock_ole32.CoUninitialize.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Failed initialization (negative HRESULT)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows only COM tests")
def test_wasapi_failed_init_does_not_couninitialize():
    """When CoInitializeEx returns negative error other than RPC_E_CHANGED_MODE, do not uninit."""
    import ctypes

    mock_ole32 = mock.MagicMock()
    mock_ole32.CoInitializeEx.return_value = -2147024882  # E_OUTOFMEMORY

    with mock.patch.object(ctypes.windll, "ole32", mock_ole32):
        result = query_wasapi_endpoints(1)
        assert result == []

    mock_ole32.CoInitializeEx.assert_called_once_with(None, 0)
    mock_ole32.CoCreateInstance.assert_not_called()
    mock_ole32.CoUninitialize.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: Secondary cleanup exceptions do not crash
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows only COM tests")
def test_wasapi_couninitialize_exception_does_not_crash():
    """If CoUninitialize raises an exception, query_wasapi_endpoints must not crash."""
    import ctypes

    mock_ole32 = mock.MagicMock()
    mock_ole32.CoInitializeEx.return_value = 0
    mock_ole32.CoCreateInstance.return_value = -2147467259
    mock_ole32.CoUninitialize.side_effect = OSError("COM uninit error")

    with mock.patch.object(ctypes.windll, "ole32", mock_ole32):
        result = query_wasapi_endpoints(1)
        assert result == []
