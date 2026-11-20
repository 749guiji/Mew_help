from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    message: str
    user_id: str | None = None
    conversation_id: int | None = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be blank")
        return v


class ExtractRequest(BaseModel):
    text: str
