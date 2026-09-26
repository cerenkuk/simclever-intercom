import json
import logging
import re
import time
from pathlib import Path

import yaml

from debrief.llm.base import LLMResult
from debrief.llm.llama_server import LlamaServerClient


LOGGER = logging.getLogger("uvicorn.error")


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUBRICS = ROOT / "config" / "debrief" / "event_rubrics_v001.yaml"
PLACEHOLDERS = {"kısa başlık", "başlık", "olaya göre başlık", "..."}


def _extract(output: str) -> dict:
    decoder = json.JSONDecoder()
    values = []
    for match in re.finditer(r"\{", output):
        try:
            value, _ = decoder.raw_decode(output[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("events"), list):
            values.append(value)
    if not values:
        raise RuntimeError("Qwen geçerli events JSON'u üretmedi")
    return values[-1]


class LocalQwenLLMAdapter:
    WINDOW_MS = 180_000
    OVERLAP_MS = 20_000

    def __init__(
        self, server: Path, model: Path, rubric_path: Path = DEFAULT_RUBRICS,
        threads: int = 8, port: int = 8091,
    ):
        self.model = Path(model)
        self.rubric_path = Path(rubric_path)
        self.client = LlamaServerClient(server, model, port=port, threads=threads, context=8192)

    def _config(self) -> dict:
        return yaml.safe_load(self.rubric_path.read_text(encoding="utf-8"))

    def _run(self, prompt: str, max_tokens: int = 1400, schema: dict | None = None) -> dict:
        return _extract(self.client.chat(prompt, max_tokens=max_tokens, schema=schema))

    @staticmethod
    def _event_schema(refs: list[str], categories: list[str]) -> dict:
        return {
            "type": "object", "additionalProperties": False,
            "properties": {
                "segment_ref": {"type": "string", "enum": refs},
                "category": {"type": "string", "enum": categories},
                "title": {"type": "string", "minLength": 3, "maxLength": 160},
                "reason": {"type": "string", "minLength": 3, "maxLength": 240},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "importance": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["segment_ref", "category", "title", "reason", "confidence", "importance"],
        }

    @classmethod
    def _window_schema(cls, cues: list[dict], categories: list[str], desired: int) -> dict:
        refs = [item["segment_ref"] for item in cues]
        return {
            "type": "object", "additionalProperties": False,
            "properties": {
                "window": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "start_segment_ref": {"type": "string", "enum": refs},
                        "end_segment_ref": {"type": "string", "enum": refs},
                        "title": {"type": "string"}, "summary": {"type": "string"},
                    },
                    "required": ["start_segment_ref", "end_segment_ref", "title", "summary"],
                },
                "events": {"type": "array", "minItems": desired, "maxItems": desired,
                           "items": cls._event_schema(refs, categories)},
            },
            "required": ["window", "events"],
        }

    @classmethod
    def _merge_schema(cls, segments: list[dict], candidates: list[dict], categories: list[str], target: int) -> dict:
        event_refs = sorted({item["segment_ref"] for item in candidates})
        chapter_refs = [item["segment_ref"] for item in segments]
        chapter = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "start_segment_ref": {"type": "string", "enum": chapter_refs},
                "title": {"type": "string"}, "summary": {"type": "string"},
            },
            "required": ["start_segment_ref", "title", "summary"],
        }
        return {
            "type": "object", "additionalProperties": False,
            "properties": {
                "chapter_starts": {"type": "array", "minItems": 3, "maxItems": 5, "items": chapter},
                "events": {"type": "array", "minItems": target, "maxItems": target,
                           "items": cls._event_schema(event_refs, categories)},
            },
            "required": ["chapter_starts", "events"],
        }

    @classmethod
    def _windows(cls, segments: list[dict]) -> list[list[dict]]:
        last_end = max(item["end_ms"] for item in segments)
        step = cls.WINDOW_MS - cls.OVERLAP_MS
        windows = []
        start = 0
        while start < last_end:
            window = [
                item for item in segments
                if item["start_ms"] < start + cls.WINDOW_MS and item["end_ms"] >= start
            ]
            if window:
                windows.append(window)
            start += step
        return windows

    @staticmethod
    def _transcript(segments: list[dict]) -> str:
        return "\n".join(
            f'{item["segment_ref"]} {item["start_ms"]/1000:.2f}-{item["end_ms"]/1000:.2f}: {item["text"]}'
            for item in segments
        )

    @staticmethod
    def _select_cues(window: list[dict], limit: int = 30) -> list[dict]:
        """Keep broad time coverage plus information-rich cues for CPU inference."""
        if len(window) <= limit:
            return window
        coverage_count = limit // 2
        coverage_indexes = {
            round(number * (len(window) - 1) / (coverage_count - 1))
            for number in range(coverage_count)
        }
        remaining = limit - len(coverage_indexes)
        rich_indexes = sorted(
            (index for index in range(len(window)) if index not in coverage_indexes),
            key=lambda index: (-len(window[index]["text"].split()), index),
        )[:remaining]
        return [window[index] for index in sorted(coverage_indexes | set(rich_indexes))]

    @staticmethod
    def _title_tokens(title: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-zçğıöşü0-9]+", title.casefold())
            if len(token) > 2
        }

    @classmethod
    def _deduplicate_candidates(cls, candidates: list[dict]) -> list[dict]:
        result = []
        for candidate in sorted(candidates, key=lambda item: (-float(item["importance"]), item["start_ms"])):
            tokens = cls._title_tokens(candidate["title"])
            duplicate = False
            for existing in result:
                other = cls._title_tokens(existing["title"])
                similarity = len(tokens & other) / max(1, len(tokens | other))
                if candidate["segment_ref"] == existing["segment_ref"] or similarity >= 0.75:
                    duplicate = True
                    break
            if not duplicate:
                result.append(candidate)
        return sorted(result, key=lambda item: item["start_ms"])

    @staticmethod
    def _select_spread(candidates: list[dict], target: int) -> list[dict]:
        if len(candidates) <= target:
            return candidates
        selected = []
        groups = {}
        for item in candidates:
            groups.setdefault(item["window_number"], []).append(item)
        for number in sorted(groups):
            selected.append(max(groups[number], key=lambda item: float(item["importance"])))
        if len(selected) > target:
            indexes = [round(number * (len(selected) - 1) / (target - 1)) for number in range(target)] if target > 1 else [0]
            selected = [selected[index] for index in indexes]
        elif len(selected) < target:
            used = {item["segment_ref"] for item in selected}
            remaining = sorted(
                (item for item in candidates if item["segment_ref"] not in used),
                key=lambda item: -float(item["importance"]),
            )
            selected.extend(remaining[:target - len(selected)])
        return sorted(selected, key=lambda item: item["start_ms"])

    @staticmethod
    def _deterministic_chapters(segments: list[dict], outlines: list[dict]) -> list[dict]:
        count = min(5, max(3, len(outlines)))
        indexes = [round(number * (len(outlines) - 1) / (count - 1)) for number in range(count)]
        chosen = [outlines[index] for index in indexes]
        sequence = {item["segment_ref"]: number for number, item in enumerate(segments)}
        starts = [item["start_segment_ref"] for item in chosen]
        chapters = []
        for number, outline in enumerate(chosen):
            start_ref = starts[number]
            next_index = sequence[starts[number + 1]] if number + 1 < len(starts) else len(segments)
            chapters.append({
                "start_segment_ref": start_ref,
                "end_segment_ref": segments[next_index - 1]["segment_ref"],
                "title": outline["title"], "summary": outline["summary"], "confidence": 0.8,
            })
        return chapters

    def _window_prompt(self, window: list[dict], categories: list[str], desired: int) -> str:
        first, last = window[0]["segment_ref"], window[-1]["segment_ref"]
        return f"""Bir eğitim kaydının {first}-{last} aralığındaki zaman damgalı ASR cue'larını incele.
ASR kusurlu olabilir; sayı/doz iddia etme ve cue metnini başlık olarak aynen kopyalama.
Yalnız geçerli JSON üret; Markdown ve düşünme metni yazma.
Bu pencereden TAM OLARAK {desired} farklı, kaba ama açık eğitim olayı seç.
Olay başlığı seçilen ASR cue'nun kendi içeriğindeki eylemi açıklasın. Prompttaki biçim ve alan adlarını olay içeriği olarak kopyalama.
Selamlama, kişi adı, soru parçası, konu başlığı ve tek başına anlamsız cümleyi olay seçme.
segment_ref yalnız verilen kimliklerden, category yalnız [{', '.join(categories)}] listesinden olmalıdır.
Pencere için ayrıca kısa ve içerik tabanlı bir title ile summary üret.
Şema:
{{"window":{{"start_segment_ref":"{first}","end_segment_ref":"{last}","title":"İçerik başlığı","summary":"Kısa özet"}},"events":[{{"segment_ref":"S000001","category":"iletisim","title":"Açık olay adı","reason":"Neden önemli","confidence":0.8,"importance":0.8}}]}}

ASR cue'ları:
{self._transcript(window)}"""

    @staticmethod
    def _validate_window(payload: dict, window: list[dict], categories: list[str], desired: int) -> tuple[dict, list[dict]]:
        refs = {item["segment_ref"] for item in window}
        outline = payload.get("window")
        events = payload.get("events")
        if not isinstance(outline, dict) or not isinstance(events, list) or len(events) < desired:
            raise RuntimeError("Pencere çıktısı eksik")
        if outline.get("start_segment_ref") not in refs or outline.get("end_segment_ref") not in refs:
            raise RuntimeError("Pencere özeti uydurma segment içeriyor")
        valid = []
        for item in events[:desired]:
            if item.get("segment_ref") not in refs or item.get("category") not in categories:
                raise RuntimeError("Pencere olayı uydurma segment veya izinsiz kategori içeriyor")
            title = str(item.get("title", "")).strip()
            if not title or title.casefold() in PLACEHOLDERS:
                raise RuntimeError("Pencere olayı geçersiz başlık içeriyor")
            if not all(key in item for key in ("reason", "confidence", "importance")):
                raise RuntimeError("Pencere olayı zorunlu alanları içermiyor")
            valid.append(item)
        return outline, valid

    def _merge_prompt(
        self, segments: list[dict], outlines: list[dict], candidates: list[dict],
        categories: list[str], target: int,
    ) -> str:
        candidate_lines = "\n".join(
            f'{item["segment_ref"]} | {item["category"]} | {item["title"]} | {item["reason"]}'
            for item in candidates
        )
        outline_lines = "\n".join(
            f'{item["start_segment_ref"]}-{item["end_segment_ref"]} | {item["title"]} | {item["summary"]}'
            for item in outlines
        )
        sample_step = max(1, len(segments) // 18)
        boundary_segments = segments[::sample_step]
        if boundary_segments[-1]["segment_ref"] != segments[-1]["segment_ref"]:
            boundary_segments.append(segments[-1])
        boundary_lines = "\n".join(
            f'{item["segment_ref"]}: {item["text"]}' for item in boundary_segments
        )
        allowed_events = ", ".join(sorted({item["segment_ref"] for item in candidates}))
        allowed_starts = ", ".join(item["segment_ref"] for item in boundary_segments)
        return f"""Pencere sonuçlarını bütün kayıt için birleştir.
Yalnız geçerli JSON üret; Markdown ve düşünme yazma.
Örtüşme kaynaklı aynı/komşu olayları tekilleştir ve TAM OLARAK {target} önemli bookmark seç.
Olay segment_ref'i yalnız [{allowed_events}], category yalnız [{', '.join(categories)}] olabilir.
Başlık açık eylemi anlatsın; ASR cue kopyası, kişi adı, soru parçası, sayı/doz veya placeholder olmasın.
3-5 anlamlı chapter üret. İlk chapter S000001 ile başlamalı. Chapter start_segment_ref yalnız [{allowed_starts}] olabilir.
Çıktı:
{{"chapter_starts":[{{"start_segment_ref":"S000001","title":"İçerik başlığı","summary":"Kısa özet"}}],"events":[{{"segment_ref":"S000001","category":"iletisim","title":"Açık olay adı","reason":"Neden önemli","confidence":0.8,"importance":0.8}}]}}

Pencere özetleri:
{outline_lines}

Bookmark adayları:
{candidate_lines}

Kayıt akışı için seyrek ASR cue'ları:
{boundary_lines}"""

    @staticmethod
    def _final_payload(
        merged: dict, segments: list[dict], candidates: list[dict],
        categories: list[str], target: int,
    ) -> dict:
        by_ref = {item["segment_ref"]: item for item in segments}
        sequence = {item["segment_ref"]: number for number, item in enumerate(segments)}
        candidate_refs = {item["segment_ref"] for item in candidates}
        events = merged.get("events")
        starts = merged.get("chapter_starts")
        if not isinstance(events, list) or len(events) != target:
            raise RuntimeError("Birleştirme hedef olay sayısını üretmedi")
        if not isinstance(starts, list) or not 3 <= len(starts) <= 5:
            raise RuntimeError("Birleştirme 3-5 chapter başlangıcı üretmedi")
        seen = set()
        final_events = []
        for item in events:
            ref, title = item.get("segment_ref"), str(item.get("title", "")).strip()
            if ref not in candidate_refs or ref in seen or item.get("category") not in categories:
                raise RuntimeError("Birleştirme geçersiz veya mükerrer olay içeriyor")
            if not title or title.casefold() in PLACEHOLDERS:
                raise RuntimeError("Birleştirme geçersiz başlık içeriyor")
            if not all(key in item for key in ("reason", "confidence", "importance")):
                raise RuntimeError("Birleştirme olay alanları eksik")
            seen.add(ref)
            final_events.append(item)
        start_refs = [item.get("start_segment_ref") for item in starts]
        if start_refs[0] != segments[0]["segment_ref"] or any(ref not in by_ref for ref in start_refs):
            raise RuntimeError("Chapter başlangıçları kayıt kapsamına uymuyor")
        if start_refs != sorted(start_refs, key=lambda ref: sequence[ref]) or len(set(start_refs)) != len(start_refs):
            raise RuntimeError("Chapter başlangıçları sırasız veya mükerrer")
        chapters = []
        for number, item in enumerate(starts):
            start_ref = item["start_segment_ref"]
            next_index = sequence[start_refs[number + 1]] if number + 1 < len(starts) else len(segments)
            end_ref = segments[next_index - 1]["segment_ref"]
            title, summary = str(item.get("title", "")).strip(), str(item.get("summary", "")).strip()
            if not title or not summary:
                raise RuntimeError("Chapter başlık veya özeti boş")
            chapters.append({
                "start_segment_ref": start_ref, "end_segment_ref": end_ref,
                "title": title, "summary": summary, "confidence": 0.8,
            })
        final_events.sort(key=lambda item: sequence[item["segment_ref"]])
        return {"chapters": chapters, "events": final_events}

    def generate_index(self, segments: list[dict], categories: list[str]) -> LLMResult:
        if not segments:
            raise RuntimeError("İndekslenecek ASR cue bulunamadı")
        config = self._config()
        index_started_at = time.monotonic()
        outlines, candidates, failures = [], [], 0
        by_ref = {item["segment_ref"]: item for item in segments}
        windows = self._windows(segments)
        progress = getattr(self, "progress_callback", None)
        cancel = getattr(self, "cancel_check", None)
        checkpoint_loader = getattr(self, "checkpoint_loader", None)
        checkpoint_saver = getattr(self, "checkpoint_saver", None)
        signature = f"{segments[0]['segment_ref']}:{segments[-1]['segment_ref']}:{len(segments)}"
        checkpoint = checkpoint_loader() if checkpoint_loader else {}
        if checkpoint.get("kind") == "index" and checkpoint.get("signature") == signature:
            outlines = checkpoint.get("outlines", [])
            candidates = checkpoint.get("candidates", [])
            completed_windows = set(checkpoint.get("completed_windows", []))
        else:
            completed_windows = set()
        for window_number, window in enumerate(windows, 1):
            if cancel:
                cancel()
            if window_number in completed_windows:
                if progress:
                    progress(f"İndeks chunk {window_number}/{len(windows)} ara kayıttan alındı", window_number, len(windows) + 1)
                continue
            if progress:
                progress(f"İndeks chunk {window_number}/{len(windows)} işleniyor", window_number - 1, len(windows) + 1)
            chunk_started_at = time.monotonic()
            LOGGER.info("İndeks chunk başladı: %s/%s", window_number, len(windows))
            cues = self._select_cues(window)
            duration = window[-1]["end_ms"] - window[0]["start_ms"]
            desired = 2 if duration >= 120_000 else 1
            try:
                payload = self._run(
                    self._window_prompt(cues, categories, desired),
                    max_tokens=260 + desired * 100,
                )
                outline, events = self._validate_window(payload, cues, categories, desired)
            except Exception:
                failures += 1
                LOGGER.exception("İndeks chunk başarısız: %s/%s", window_number, len(windows))
                continue
            outlines.append(outline)
            for event in events:
                source = by_ref[event["segment_ref"]]
                candidates.append({
                    **event, "window_number": window_number,
                    "start_ms": source["start_ms"],
                })
            LOGGER.info(
                "İndeks chunk tamamlandı: %s/%s, %.2f sn",
                window_number, len(windows), time.monotonic() - chunk_started_at,
            )
            if progress:
                progress(f"İndeks chunk {window_number}/{len(windows)} tamamlandı", window_number, len(windows) + 1)
            completed_windows.add(window_number)
            if checkpoint_saver:
                checkpoint_saver({
                    "kind": "index", "signature": signature,
                    "completed_windows": sorted(completed_windows),
                    "outlines": outlines, "candidates": candidates,
                })
        duration_ms = segments[-1]["end_ms"] - segments[0]["start_ms"]
        target = max(1, round(duration_ms / (config["target_event_interval_seconds"] * 1000)))
        if len(outlines) < min(3, len(windows)):
            raise RuntimeError(f"{len(windows)} chunk'ın {failures} tanesi başarısız; güvenilir indeks üretilemedi")
        unique = self._deduplicate_candidates(candidates)
        if len(unique) < min(target, 2):
            raise RuntimeError("Chunk'lar yeterli benzersiz olay üretmedi")
        selected = self._select_spread(unique, min(target, len(unique)))
        events = [
            {key: item[key] for key in ("segment_ref", "category", "title", "reason", "confidence", "importance")}
            for item in selected
        ]
        payload = {"chapters": self._deterministic_chapters(segments, outlines), "events": events}
        if cancel:
            cancel()
        if progress:
            progress("İndeks sonuçları birleştiriliyor", len(windows), len(windows) + 1)
        LOGGER.info(
            "İndeks tamamlandı: %s chunk, %s olay, %s hata, %.2f sn",
            len(windows), len(events), failures, time.monotonic() - index_started_at,
        )
        model_name = self.model.stem if isinstance(self.model, Path) else str(self.model)
        return LLMResult(
            "llama.cpp-server", model_name, f"chunked-code-merge-002-{config['version']}", payload,
            processing_ms=round((time.monotonic() - index_started_at) * 1000),
            model_load_ms=self.client.last_model_load_ms,
            chunk_count=len(windows), failed_chunk_count=failures,
        )
