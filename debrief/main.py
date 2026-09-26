import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from debrief.api import build_router
from debrief.asr import ASRAdapter, Paket3FasterWhisperAdapter
from debrief.llm import FakeLLMAdapter, GeminiLLMAdapter, LLMAdapter, LocalQwenLLMAdapter
from debrief.config import Settings
from debrief.db import Base, build_engine, build_session_factory, session_dependency
from debrief.jobs import ProcessingQueue


LOGGER = logging.getLogger("uvicorn.error")


def install_windows_media_disconnect_filter() -> None:
    """Ignore the harmless Proactor traceback produced by aborted HTTP Range requests."""
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()

    def handler(active_loop, context):
        error = context.get("exception")
        callback = repr(context.get("handle", ""))
        if (
            isinstance(error, ConnectionResetError)
            and getattr(error, "winerror", None) == 10054
            and "_ProactorBasePipeTransport._call_connection_lost" in callback
        ):
            LOGGER.debug("Tarayıcı medya Range bağlantısını kapattı: WinError 10054")
            return
        if previous:
            previous(active_loop, context)
        else:
            active_loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


def create_app(settings: Settings | None = None, asr_adapter: ASRAdapter | None = None, llm_adapter: LLMAdapter | None = None) -> FastAPI:
    settings = settings or Settings()
    engine = build_engine(settings.database_url)
    session_factory = build_session_factory(engine)

    if settings.auto_create_schema:
        Base.metadata.create_all(engine)

    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.state.engine = engine
    app.state.session_factory = session_factory
    processing_queue = ProcessingQueue(session_factory, workers=1)
    app.state.processing_queue = processing_queue
    app.router.add_event_handler("startup", install_windows_media_disconnect_filter)
    if llm_adapter is None:
        if settings.llm_mode == "qwen-local":
            llm_adapter = LocalQwenLLMAdapter(
                settings.llm_server_path, settings.llm_model_path,
                settings.event_rubric_path, port=settings.llm_server_port,
            )
        elif settings.llm_mode == "gemini-cloud":
            llm_adapter = GeminiLLMAdapter(
                settings.gemini_api_key(), settings.event_rubric_path, settings.gemini_model
            )
        else:
            llm_adapter = FakeLLMAdapter()
    app.include_router(
        build_router(
            session_dependency(session_factory), settings,
            asr_adapter or Paket3FasterWhisperAdapter(settings.asr_profile), llm_adapter,
            processing_queue,
        )
    )
    app.router.add_event_handler("shutdown", processing_queue.stop)

    stop_client = getattr(getattr(llm_adapter, "client", None), "stop", None)
    if callable(stop_client):
        app.router.add_event_handler("shutdown", stop_client)

    @app.get("/", include_in_schema=False)
    def web_app():
        return FileResponse(Path(__file__).resolve().parent / "web" / "index.html")

    @app.get("/phase2", include_in_schema=False)
    def phase2_web_app():
        return FileResponse(Path(__file__).resolve().parent / "web" / "phase2.html")

    @app.get("/brand/simclever-logo.png", include_in_schema=False)
    def brand_logo():
        return FileResponse(Path(__file__).resolve().parent / "web" / "assets" / "simclever-logo.png")

    return app


app = create_app()
