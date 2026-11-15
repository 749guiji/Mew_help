from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.prompts.system import EXTRACTION_SYSTEM_PROMPT
from app.schemas.extraction import AfterSalesExtraction


def build_extraction_chain(llm) -> Runnable:
    return ChatPromptTemplate.from_messages(
        [("system", EXTRACTION_SYSTEM_PROMPT), ("human", "{input}")]
    ) | llm.with_structured_output(AfterSalesExtraction, method="function_calling")


class ExtractionService:
    def __init__(self, extract_chain: Runnable | None = None):
        self._chain = extract_chain

    async def extract(self, text: str) -> AfterSalesExtraction:
        return await self._chain.ainvoke({"input": text})
