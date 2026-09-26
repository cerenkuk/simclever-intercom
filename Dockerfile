FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && rm -rf /var/lib/apt/lists/*
COPY requirements-debrief.txt ./
RUN pip install --no-cache-dir -r requirements-debrief.txt
COPY . .
RUN chmod +x /app/tools/llama.cpp/llama-server 2>/dev/null || true
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn debrief.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
