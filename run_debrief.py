import os
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("DEBRIEF_LLM_MODE", "qwen-local")
os.environ.setdefault(
    "DEBRIEF_LLM_MODEL_PATH",
    str(ROOT / "models" / "qwen3.5-4b" / "Qwen_Qwen3.5-4B-Q4_K_M.gguf"),
)
os.environ.setdefault(
    "DEBRIEF_LLM_SERVER_PATH",
    str(ROOT / "tools" / "llama.cpp" / "b10516" / "llama-server.exe"),
)


if __name__ == "__main__":
    uvicorn.run("debrief.main:app", host="127.0.0.1", port=8000, reload=False)

