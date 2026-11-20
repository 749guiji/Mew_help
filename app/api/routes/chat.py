import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest

router = APIRouter()


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    svc = request.app.state.chat_service

    async def gen():
        async for event in svc.stream_chat(payload.session_id, payload.message):
            yield f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
