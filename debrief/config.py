from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class IndexingConfig:
    version: str
    lead_in_ms: int
    window_ms: int
    overlap_ms: int
    target_event_interval_min_ms: int
    target_event_interval_max_ms: int
    llm_temperature: float
    fingerprint: str


def load_indexing_config(path: Path) -> IndexingConfig:
    raw = path.read_bytes()
    document = yaml.safe_load(raw.decode("utf-8")) or {}
    interval = document.get("target_event_interval_seconds") or {}
    lead_in_seconds = int(document.get("lead_in_seconds", 5))
    window_seconds = int(document.get("window_seconds", 600))
    overlap_seconds = int(document.get("overlap_seconds", 60))
    minimum = int(interval.get("minimum", 120))
    maximum = int(interval.get("maximum", 240))
    if lead_in_seconds < 0:
        raise ValueError("lead_in_seconds negatif olamaz")
    if window_seconds <= 0:
        raise ValueError("window_seconds pozitif olmalıdır")
    if overlap_seconds < 0 or overlap_seconds >= window_seconds:
        raise ValueError("overlap_seconds pencere süresinden küçük ve negatif olmayan bir değer olmalıdır")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Olay yoğunluğu aralığı geçersiz")
    return IndexingConfig(
        version=str(document.get("version", "unknown")),
        lead_in_ms=lead_in_seconds * 1000,
        window_ms=window_seconds * 1000,
        overlap_ms=overlap_seconds * 1000,
        target_event_interval_min_ms=minimum * 1000,
        target_event_interval_max_ms=maximum * 1000,
        llm_temperature=float(document.get("llm_temperature", 0)),
        fingerprint=hashlib.sha256(raw).hexdigest(),
    )


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DEBRIEF_DATABASE_URL", f"sqlite:///{(PROJECT_ROOT / 'debrief.local.db').as_posix()}"
    )
    media_root: Path = Path(os.getenv("DEBRIEF_MEDIA_ROOT", str(PROJECT_ROOT / "debrief-media")))
    max_upload_bytes: int = 4 * 1024 * 1024 * 1024
    auto_create_schema: bool = False
    app_name: str = "SimClever Debrief"
    default_timezone: str = "Europe/Istanbul"
    asr_profile: str = os.getenv("DEBRIEF_ASR_PROFILE", "quality")
    llm_mode: str = os.getenv("DEBRIEF_LLM_MODE", "fake")
    llm_model_path: Path = Path(os.getenv(
        "DEBRIEF_LLM_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "qwen3.5-2b" / "Qwen3.5-2B-Q4_K_M.gguf"),
    ))
    llm_server_path: Path = Path(os.getenv(
        "DEBRIEF_LLM_SERVER_PATH",
        str(PROJECT_ROOT / "tools" / "llama.cpp" / "b10516" / "llama-server.exe"),
    ))
    llm_server_port: int = int(os.getenv("DEBRIEF_LLM_SERVER_PORT", "8091"))
    gemini_model: str = os.getenv("DEBRIEF_GEMINI_MODEL", "gemini-3.5-flash-lite")
    event_rubric_path: Path = PROJECT_ROOT / "config" / "debrief" / "event_rubrics_v001.yaml"
    indexing_config_path: Path = PROJECT_ROOT / "config" / "debrief" / "indexing_v001.yaml"

    def indexing(self) -> IndexingConfig:
        return load_indexing_config(self.indexing_config_path)

    def gemini_api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "")
