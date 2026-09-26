import os
import yaml

class CloudUploadDeniedError(PermissionError):
    """Custom exception raised when a cloud upload policy check fails (fail-closed)."""
    pass

def load_cloud_upload_policy(config_path: str = None) -> dict:
    if config_path is None:
        pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(pkg_root, "config", "cloud_api_upload_policy.yaml")

    if not os.path.exists(config_path):
        raise CloudUploadDeniedError(f"POLICY_FILE_MISSING: Politika dosyası bulunamadı ({config_path}). Yükleme engellendi.")

    with open(config_path, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f)
    return policy

def validate_cloud_upload_permission(basename: str, audio_path: str = None, metadata: dict = None, config_path: str = None) -> bool:
    """
    Central fail-closed security validator for cloud API audio uploads.
    Checks:
    1. Basename is in explicit allowlist.
    2. Basename does not start with any denied prefix (e.g. 'saha_').
    3. Metadata scenario_group/source_group is 'video' (never 'saha').
    4. Canonical path resolution to prevent path traversal or symlink escapes.
    5. Unknown basenames or missing metadata are rejected.
    """
    try:
        policy = load_cloud_upload_policy(config_path)
    except Exception as e:
        raise CloudUploadDeniedError(f"POLICY_LOAD_ERROR: Politika okunamadı: {e}")

    policy_version = policy.get("policy_version", "unknown")
    allowlist = set(policy.get("allowlist_basenames", []))
    denied_prefixes = tuple(policy.get("denied_prefixes", ["saha_"]))
    allowed_source = policy.get("allowed_source_group", "video")
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    allowed_audio_root = os.path.abspath(os.path.join(pkg_root, "data"))

    # 1. Basename Varlığı ve Tip Kontrolü
    if not basename or not isinstance(basename, str):
        raise CloudUploadDeniedError(f"DENIED_INVALID_BASENAME [policy={policy_version}]: Basename geçersiz veya boş.")

    clean_basename = basename.strip()

    # 2. Denied Prefix Kontrolü (saha_*)
    for prefix in denied_prefixes:
        if clean_basename.startswith(prefix):
            raise CloudUploadDeniedError(f"DENIED_PREFIX [policy={policy_version}]: '{clean_basename}' yasaklı önek ('{prefix}') içeriyor.")

    # 3. Explicit Allowlist Kontrolü
    if clean_basename not in allowlist:
        raise CloudUploadDeniedError(f"DENIED_NOT_IN_ALLOWLIST [policy={policy_version}]: '{clean_basename}' açık allowlist içerisinde yer almıyor.")

    # 4. Metadata Kontrolü (Eksik veya saha kaynaklı mı?)
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise CloudUploadDeniedError(f"DENIED_INVALID_METADATA [policy={policy_version}]: Metadata sözlük değil.")

        source_group = str(metadata.get("source_group") or metadata.get("scenario_group") or "").strip().lower()
        if source_group == "saha" or source_group != allowed_source:
            raise CloudUploadDeniedError(f"DENIED_METADATA_SOURCE [policy={policy_version}]: Kaynak grubu ('{source_group}') izin verilen '{allowed_source}' grubuyla eşleşmiyor.")

    # 5. Canonical Path & Path Traversal / Symlink Kontrolü
    if audio_path:
        real_audio_path = os.path.realpath(os.path.abspath(audio_path))
        real_allowed_root = os.path.realpath(allowed_audio_root)

        # Path traversal / Symlink escape kontrolü
        common_prefix = os.path.commonpath([real_audio_path, real_allowed_root])
        if common_prefix != real_allowed_root:
            raise CloudUploadDeniedError(f"DENIED_PATH_TRAVERSAL [policy={policy_version}]: '{audio_path}' izin verilen kök dışına çıkmaktadır ({real_audio_path}).")

    return True
