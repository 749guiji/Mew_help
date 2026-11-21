from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import OperationalError

from app.api.routes import admin as admin_routes
from app.api.routes import chat as chat_routes
from app.api.routes import extract as extract_routes
from app.api.routes import jobs as jobs_routes
from app.api.routes import kb as kb_routes
from app.api.routes import actions as actions_routes
from app.api.routes import conversations as conversations_routes
from app.api.routes import feedback as feedback_routes
from app.api.routes import orders as orders_routes
from app.api.routes import review as review_routes
from app.api.routes import tickets as tickets_routes
from app.core.llm import get_llm
from app.services.chat_service import ChatService
from app.services.extraction_service import ExtractionService, build_extraction_chain

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(chat_service=None, extraction_service=None) -> FastAPI:
    import logging

    from app.core.budget import compute_budget
    from app.core.config import get_settings

    app = FastAPI()

    # ch08:数据库不可用(如 MySQL 未起)时回结构化错误而非裸 500
    @app.exception_handler(OperationalError)
    async def _db_unavailable(request, exc):
        return JSONResponse(
            status_code=503, content={"detail": "数据库暂时不可用,请稍后再试。"}
        )

    # ch07 启动自检:历史预算连一轮稳态都装不下就报警
    b = compute_budget(get_settings())
    if b.history < b.per_round_steady:
        logging.getLogger("budget").warning(
            "上下文预算不足:历史预算 %d < 一轮稳态 %d,滑窗将归零", b.history, b.per_round_steady
        )

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
    app.include_router(actions_routes.router)
    app.include_router(conversations_routes.router)
    app.include_router(feedback_routes.router)
    app.include_router(orders_routes.router)
    app.include_router(review_routes.router)
    app.include_router(tickets_routes.router)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


app = create_app()
