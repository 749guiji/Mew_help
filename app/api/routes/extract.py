from fastapi import APIRouter, Request

from app.schemas.chat import ExtractRequest

router = APIRouter()


@router.post("/extract")
async def extract(payload: ExtractRequest, request: Request):
    return await request.app.state.extraction_service.extract(payload.text)
