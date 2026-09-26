import json
import time
import urllib.error
import urllib.request

from debrief.llm.base import LLMResult
from debrief.llm.local_qwen import LocalQwenLLMAdapter, _extract


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    last_model_load_ms = 0

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite"):
        if not api_key.strip():
            raise GeminiError("GEMINI_API_KEY tanımlı değil")
        self.api_key = api_key
        self.model = model

    def chat(self, prompt: str, max_tokens: int = 1200, schema: dict | None = None) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        generation = {"maxOutputTokens": max_tokens, "responseMimeType": "application/json"}
        if self.model.startswith("gemini-3"):
            generation["thinkingConfig"] = {"thinkingLevel": "low"}
        else:
            generation["temperature"] = 0
        if schema:
            generation["responseSchema"] = schema
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": generation}
        request = urllib.request.Request(
            url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
            method="POST",
        )
        retryable = {429, 500, 502, 503, 504}
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    document = json.loads(response.read().decode("utf-8"))
                return document["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as exc:
                if exc.code in retryable and attempt < 3:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise GeminiError(f"Gemini API HTTP {exc.code} döndürdü") from exc
            except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise GeminiError("Gemini API beklenen yanıtı döndürmedi") from exc
        raise GeminiError("Gemini API tekrar denemelerinden sonra yanıt vermedi")


class GeminiLLMAdapter(LocalQwenLLMAdapter):
    is_cloud = True

    def __init__(self, api_key: str, rubric_path, model: str = "gemini-3.5-flash-lite"):
        self.model = model
        self.rubric_path = rubric_path
        self.client = GeminiClient(api_key, model)

    def _run(self, prompt: str, max_tokens: int = 1400, schema: dict | None = None) -> dict:
        return _extract(self.client.chat(prompt, max_tokens=max_tokens, schema=schema))

    def generate_index(self, segments: list[dict], categories: list[str]) -> LLMResult:
        started = time.monotonic()
        result = super().generate_index(segments, categories)
        return LLMResult(
            provider="google-gemini-api", model=self.model,
            version=f"cloud-{result.version}", payload=result.payload,
            processing_ms=round((time.monotonic() - started) * 1000), model_load_ms=0,
            chunk_count=result.chunk_count, failed_chunk_count=result.failed_chunk_count,
        )
