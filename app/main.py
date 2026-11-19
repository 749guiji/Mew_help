from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import admin as admin_routes
from app.api.routes import chat as chat_routes
from app.api.routes import extract as extract_routes
from app.api.routes import jobs as jobs_routes
from app.api.routes import kb as kb_routes
from app.core.llm import get_llm
from app.services.chat_service import ChatService
from app.services.extraction_service import ExtractionService, build_extraction_chain

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(chat_service=None, extraction_service=None) -> FastAPI:
    app = FastAPI()

    if chat_service is None:
        chat_service = ChatService()
    if extraction_service is None:
        extraction_service = ExtractionService(build_extraction_chain(get_llm()))

    app.state.chat_service = chat_service
    app.state.extraction_service = extraction_service

    app.include_router(admin_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(extract_routes.router)
    app.include_router(jobs_routes.router)
    app.include_router(kb_routes.router)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


app = create_app()
