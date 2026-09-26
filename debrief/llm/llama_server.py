import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


LOGGER = logging.getLogger("uvicorn.error")


class LlamaServerError(RuntimeError):
    pass


class LlamaServerClient:
    """Starts one local llama-server process and reuses the loaded model."""

    def __init__(
        self, server: Path, model: Path, host: str = "127.0.0.1", port: int = 8091,
        threads: int = 8, context: int = 8192,
    ):
        self.server = Path(server)
        self.model = Path(model)
        self.base_url = f"http://{host}:{port}"
        self.host = host
        self.port = port
        self.threads = threads
        self.context = context
        self._process = None
        self._lock = threading.Lock()
        self.last_model_load_ms = 0

    def _health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=1) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def ensure_started(self, timeout_seconds: int = 120) -> None:
        if self._health():
            return
        with self._lock:
            if self._health():
                return
            if not self.server.is_file() or not self.model.is_file():
                raise LlamaServerError("llama-server veya yerel model dosyası bulunamadı")
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self._process = subprocess.Popen(
                [
                    str(self.server), "-m", str(self.model), "--host", self.host,
                    "--port", str(self.port), "-c", str(self.context), "-t", str(self.threads),
                    "--no-webui", "--reasoning", "off", "--reasoning-format", "deepseek",
                ],
                cwd=self.server.parents[3], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            started_at = time.monotonic()
            LOGGER.info("LLM model yükleniyor: %s", self.model.name)
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise LlamaServerError("llama-server model yüklenirken kapandı")
                if self._health():
                    self.last_model_load_ms = round((time.monotonic() - started_at) * 1000)
                    LOGGER.info("LLM model hazır: %.2f sn", self.last_model_load_ms / 1000)
                    return
                time.sleep(0.5)
            self.stop()
            raise LlamaServerError("llama-server başlatma zaman aşımı")

    def chat(self, prompt: str, max_tokens: int = 1200, schema: dict | None = None) -> str:
        self.ensure_started()
        started_at = time.monotonic()
        request_body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "seed": 42, "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object", **({"schema": schema} if schema else {})},
        }
        payload = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                document = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise LlamaServerError("llama-server isteği başarısız") from exc
        try:
            content = document["choices"][0]["message"]["content"]
            LOGGER.info("LLM çıkarımı tamamlandı: %.2f sn", time.monotonic() - started_at)
            return content
        except (KeyError, IndexError, TypeError) as exc:
            raise LlamaServerError("llama-server beklenen yanıt biçimini döndürmedi") from exc

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
