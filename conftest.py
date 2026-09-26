"""conftest.py – proje kökünde.

Pytest'in sys.path'ına proje kökünü ekler; böylece 'paket5', 'shared' gibi
paketler doğrudan import edilebilir. Bu dosya pytest tarafından otomatik
yüklenir, herhangi bir fixture veya eklenti gerektirmez.
"""
import sys
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def fake_sounddevice_streams(monkeypatch):
    """Keep engine tests deterministic on CI machines without audio hardware."""
    import sounddevice

    class FakeStream:
        def __init__(self, *args, **kwargs):
            self.active = False

        def start(self):
            self.active = True
            return self

        def stop(self):
            self.active = False

        def close(self):
            self.active = False

    monkeypatch.setattr(sounddevice, "InputStream", FakeStream)
    monkeypatch.setattr(sounddevice, "OutputStream", FakeStream)
