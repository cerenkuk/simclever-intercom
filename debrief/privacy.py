from debrief.models import DebriefSession, PrivacyMode


class CloudPrivacyError(PermissionError):
    pass


def require_cloud_allowed(session: DebriefSession) -> None:
    """Fail closed before any transcript or media is prepared for an external provider."""
    if session.privacy_mode != PrivacyMode.CLOUD_ALLOWED:
        raise CloudPrivacyError("Yalnız yerel oturum bulut sağlayıcısına gönderilemez")
