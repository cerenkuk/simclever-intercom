import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path


ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi",
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus",
}


class MediaError(ValueError):
    pass


@dataclass(frozen=True)
class MediaInfo:
    duration_ms: int
    container_format: str | None
    codec_name: str | None
    sample_rate: int | None
    channel_count: int | None
    has_video: bool
    has_audio: bool


def safe_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise MediaError(f"Desteklenmeyen dosya uzantısı: {suffix or '(yok)'}")
    return suffix


def destination_for(media_root: Path, session_id: str, original_filename: str) -> Path:
    suffix = safe_extension(original_filename)
    session_dir = (media_root / session_id).resolve()
    root = media_root.resolve()
    if root not in session_dir.parents:
        raise MediaError("Geçersiz oturum medya yolu")
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir / f"{uuid.uuid4()}{suffix}"


def probe_media(path: Path) -> MediaInfo:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,format_name:stream=codec_type,codec_name,sample_rate,channels",
        "-of", "json", str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
        payload = json.loads(completed.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        raise MediaError("Dosya FFprobe ile okunamadı veya geçerli medya değil") from exc

    streams = payload.get("streams", [])
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    raw_duration = payload.get("format", {}).get("duration")
    if raw_duration is None:
        raise MediaError("Medya süresi belirlenemedi")
    duration_ms = max(0, round(float(raw_duration) * 1000))
    primary = audio or video or {}
    return MediaInfo(
        duration_ms=duration_ms,
        container_format=payload.get("format", {}).get("format_name"),
        codec_name=primary.get("codec_name"),
        sample_rate=int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        channel_count=int(audio["channels"]) if audio and audio.get("channels") else None,
        has_video=video is not None,
        has_audio=audio is not None,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_audio_wav(video_path: Path) -> Path:
    output = video_path.with_name(f"{uuid.uuid4()}.wav")
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as exc:
        if output.exists():
            output.unlink()
        raise MediaError("Videonun sesi 16 kHz WAV biçimine çıkarılamadı") from exc
    if not output.is_file() or output.stat().st_size == 0:
        raise MediaError("Videodan boş ses dosyası üretildi")
    return output


def video_thumbnail(video_path: Path, time_ms: int = 1000) -> Path:
    bucket_ms = max(500, (max(0, time_ms) // 5000) * 5000)
    output = video_path.with_suffix(video_path.suffix + f".thumb.{bucket_ms}.jpg")
    if output.is_file() and output.stat().st_size > 0:
        return output
    command = [
        "ffmpeg", "-v", "error", "-y", "-ss", f"{bucket_ms / 1000:.3f}", "-i", str(video_path),
        "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", str(output),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as exc:
        if output.exists():
            output.unlink()
        raise MediaError("Video önizleme görseli üretilemedi") from exc
    if not output.is_file() or output.stat().st_size == 0:
        raise MediaError("Boş video önizleme görseli üretildi")
    return output


def audio_waveform(audio_path: Path) -> Path:
    output = audio_path.with_suffix(audio_path.suffix + ".waveform-v2.png")
    if output.is_file() and output.stat().st_size > 0:
        return output
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(audio_path),
        "-filter_complex", "showwavespic=s=1600x120:colors=0x13c9dd:draw=full:scale=sqrt",
        "-frames:v", "1", str(output),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
    except (subprocess.SubprocessError, OSError) as exc:
        if output.exists(): output.unlink()
        raise MediaError("Ses waveform görseli üretilemedi") from exc
    if not output.is_file() or output.stat().st_size == 0:
        raise MediaError("Boş waveform görseli üretildi")
    return output


def video_filmstrip(video_path: Path, duration_ms: int) -> Path:
    output = video_path.with_suffix(video_path.suffix + ".filmstrip.jpg")
    if output.is_file() and output.stat().st_size > 0:
        return output
    seconds = max(1.0, duration_ms / 1000)
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(video_path),
        "-vf", f"fps=12/{seconds:.3f},scale=160:90:force_original_aspect_ratio=increase,crop=160:90,tile=12x1",
        "-frames:v", "1", "-q:v", "5", str(output),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
    except (subprocess.SubprocessError, OSError) as exc:
        if output.exists(): output.unlink()
        raise MediaError("Video film şeridi üretilemedi") from exc
    if not output.is_file() or output.stat().st_size == 0:
        raise MediaError("Boş video film şeridi üretildi")
    return output


def resolve_stored_media(media_root: Path, media_path: str) -> Path:
    root = media_root.resolve()
    candidate = (root / media_path).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise MediaError("Medya dosyası bulunamadı")
    return candidate
