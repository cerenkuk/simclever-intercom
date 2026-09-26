import math
import logging
import os
import time
from pathlib import Path

from debrief.asr.base import ASRResult, ASRSegment


LOGGER = logging.getLogger("uvicorn.error")


class Paket3FasterWhisperAdapter:
    """Paket 3 ürün kararının segment koruyan Debrief adapter'ı."""

    PROFILES = {
        "fast": {
            "model_id": "faster-whisper-small-int8", "model_name": "small",
            "beam_size": 1, "config_version": "asr-cue-small-int8-beam1-v1",
        },
        "quality": {
            "model_id": "faster-whisper-large-v3-int8", "model_name": "large-v3",
            "beam_size": 5, "config_version": "paket3-large-v3-none-pad400-v1",
        },
    }
    SAMPLE_RATE = 16_000
    CHUNK_SECONDS = 300
    OVERLAP_SECONDS = 1

    def __init__(self, profile: str = "quality"):
        if profile not in self.PROFILES:
            raise ValueError(f"Bilinmeyen ASR profili: {profile}")
        self.profile = profile
        self.config = self.PROFILES[profile]
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            LOGGER.info("ASR model yükleniyor: %s", self.config["model_name"])
            started_at = time.monotonic()
            cpu_threads = max(1, min(8, (os.cpu_count() or 4) - 1))
            self._model = WhisperModel(
                self.config["model_name"], device="cpu", compute_type="int8",
                cpu_threads=cpu_threads, num_workers=1,
            )
            LOGGER.info("ASR model hazır: %.2f sn", time.monotonic() - started_at)
        return self._model

    def transcribe(self, media_path: Path, language: str = "tr") -> ASRResult:
        import faster_whisper
        from faster_whisper.audio import decode_audio

        started_at = time.monotonic()
        audio = decode_audio(str(media_path), sampling_rate=self.SAMPLE_RATE)
        duration_seconds = len(audio) / self.SAMPLE_RATE
        chunk_count = max(1, math.ceil(duration_seconds / self.CHUNK_SECONDS))
        progress = getattr(self, "progress_callback", None)
        cancel = getattr(self, "cancel_check", None)
        if progress:
            progress("ASR sesi çözümlendi; model hazırlanıyor", 0, chunk_count)
        model = self._get_model()
        checkpoint_loader = getattr(self, "checkpoint_loader", None)
        checkpoint_saver = getattr(self, "checkpoint_saver", None)
        checkpoint = checkpoint_loader() if checkpoint_loader else {}
        if checkpoint.get("kind") == "asr" and checkpoint.get("media_path") == str(media_path):
            output = [ASRSegment(**item) for item in checkpoint.get("segments", [])]
            completed_chunks = set(checkpoint.get("completed_chunks", []))
        else:
            output, completed_chunks = [], set()
        for chunk_number in range(chunk_count):
            if cancel:
                cancel()
            if chunk_number in completed_chunks:
                if progress:
                    progress(f"ASR chunk {chunk_number + 1}/{chunk_count} ara kayıttan alındı", chunk_number + 1, chunk_count)
                continue
            if progress:
                progress(f"ASR chunk {chunk_number + 1}/{chunk_count} işleniyor", chunk_number, chunk_count)
            nominal_start = chunk_number * self.CHUNK_SECONDS
            decode_start = max(0, nominal_start - (self.OVERLAP_SECONDS if chunk_number else 0))
            decode_end = min(duration_seconds, (chunk_number + 1) * self.CHUNK_SECONDS)
            chunk = audio[round(decode_start * self.SAMPLE_RATE):round(decode_end * self.SAMPLE_RATE)]
            chunk_started_at = time.monotonic()
            LOGGER.info("ASR chunk başladı: %s/%s", chunk_number + 1, chunk_count)
            generator, _ = model.transcribe(
                chunk, language=language, temperature=0.0,
                beam_size=self.config["beam_size"], vad_filter=True,
                vad_parameters={"speech_pad_ms": 400},
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
            for segment in generator:
                absolute_start = decode_start + segment.start
                absolute_end = decode_start + segment.end
                if chunk_number and (absolute_start + absolute_end) / 2 < nominal_start:
                    continue
                text = segment.text.strip()
                if not text:
                    continue
                raw_confidence = math.exp(segment.avg_logprob) if segment.avg_logprob is not None else None
                confidence = min(1.0, max(0.0, raw_confidence)) if raw_confidence is not None else None
                output.append(
                    ASRSegment(
                        ref=f"S{len(output) + 1:06d}",
                        start_ms=round(absolute_start * 1000),
                        end_ms=round(absolute_end * 1000),
                        text=text, confidence=confidence,
                    )
                )
            completed_chunks.add(chunk_number)
            if checkpoint_saver:
                checkpoint_saver({
                    "kind": "asr", "media_path": str(media_path),
                    "completed_chunks": sorted(completed_chunks),
                    "segments": [
                        {"ref": item.ref, "start_ms": item.start_ms, "end_ms": item.end_ms,
                         "text": item.text, "confidence": item.confidence}
                        for item in output
                    ],
                })
            # Tamamlanma mesajından önce ara kayıt kalıcı hale gelmelidir.
            LOGGER.info(
                "ASR chunk tamamlandı: %s/%s, %.2f sn",
                chunk_number + 1, chunk_count, time.monotonic() - chunk_started_at,
            )
            if progress:
                progress(f"ASR chunk {chunk_number + 1}/{chunk_count} tamamlandı", chunk_number + 1, chunk_count)
        LOGGER.info(
            "ASR tamamlandı: %s profil, %s chunk, %s cue, %.2f sn",
            self.profile, chunk_count, len(output), time.monotonic() - started_at,
        )
        return ASRResult(
            engine="faster-whisper",
            model=self.config["model_id"],
            engine_version=getattr(faster_whisper, "__version__", "unknown"),
            config_version=self.config["config_version"],
            language=language,
            duration_ms=round(duration_seconds * 1000),
            segments=tuple(output),
        )
