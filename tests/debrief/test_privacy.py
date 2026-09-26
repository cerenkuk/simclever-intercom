from types import SimpleNamespace

import pytest

from debrief.models import PrivacyMode
from debrief.privacy import CloudPrivacyError, require_cloud_allowed


def test_local_only_session_is_blocked_before_cloud_request():
    with pytest.raises(CloudPrivacyError):
        require_cloud_allowed(SimpleNamespace(privacy_mode=PrivacyMode.LOCAL_ONLY))


def test_cloud_allowed_session_passes_privacy_gate():
    require_cloud_allowed(SimpleNamespace(privacy_mode=PrivacyMode.CLOUD_ALLOWED))
