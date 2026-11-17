"""Historical-conversation QA mining: LLM extracts QA pairs, then dedupe + ingest.

The second offline pipeline turns raw customer-service conversations into
``knowledge_chunks``: each conversation batch is passed through a structured
``with_structured_output(..., method="function_calling")`` chain that returns
user-question / agent-answer pairs. Every extracted pair is written to
``qa_extraction_staging`` first, the whole batch is deduplicated by a normalized
question key, and the kept pairs become ``content_type='mined'`` chunks that are
then vectorized with the shared ``insert_chunks`` / ``vectorize_pending`` path.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.db.models import QaExtractionStaging
from app.kb.chunker import Chunk
from app.kb.ingest import insert_chunks, vectorize_pending
from app.prompts.system import QA_EXTRACTION_SYSTEM_PROMPT


class QAPair(BaseModel):
    question: str
    answer: str


class QAPairs(BaseModel):
    pairs: list[QAPair]


@dataclass
class QA:
    question: str
    answer: str


_PUNCT_RE = re.compile(r"[^\w\s]+")


def build_qa_extraction_chain(llm) -> Runnable:
    """Build the structured QA-extraction chain (Ollama needs explicit FC)."""
    return ChatPromptTemplate.from_messages(
        [("system", QA_EXTRACTION_SYSTEM_PROMPT), ("human", "{input}")]
    ) | llm.with_structured_output(QAPairs, method="function_calling")


def extract_qa(chain, text: str) -> list[QA]:
    """Run the chain on ``text`` and return its extracted QA pairs."""
    result = chain.invoke({"input": text})
    pairs = getattr(result, "pairs", None) or []
    return [QA(question=p.question, answer=p.answer) for p in pairs]


def normalize(q: str) -> str:
    """Dedupe key: strip whitespace, lowercase, remove all punctuation."""
    return _PUNCT_RE.sub("", q.strip().lower())


def dedupe(qas: list[QA]) -> list[QA]:
    """Keep the first QA per normalized question, preserving input order."""
    seen: set[str] = set()
    out: list[QA] = []
    for qa in qas:
        key = normalize(qa.question)
        if key in seen:
            continue
        seen.add(key)
        out.append(qa)
    return out


def _read_conversations(source_path: str) -> list[str]:
    """Read a jsonl of conversations; each line is a JSON object with ``text``.

    Lines that are not JSON (e.g. a plain conversation per line) are used as-is.
    """
    texts: list[str] = []
    for line in Path(source_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            texts.append(line)
            continue
        if isinstance(obj, dict) and "text" in obj:
            texts.append(str(obj["text"]))
        else:
            texts.append(line)
    return texts


def mine_conversations(
    session,
    store,
    embedder,
    chain,
    source_path: str,
    batch_size: int = 10,
) -> int:
    """Extract QA pairs from ``source_path``, dedupe, then ingest the kept ones.

    Returns the number of kept pairs written as ``knowledge_chunks`` rows.
    """
    texts = _read_conversations(source_path)

    staged: list[tuple[QaExtractionStaging, QA]] = []
    for start in range(0, len(texts), batch_size):
        batch_no = str(uuid.uuid4())
        for i, text in enumerate(texts[start : start + batch_size]):
            source_ref = f"{source_path}:{start + i}"
            for qa in extract_qa(chain, text):
                row = QaExtractionStaging(
                    batch_no=batch_no,
                    source_ref=source_ref,
                    question=qa.question,
                    answer=qa.answer,
                    status="extracted",
                )
                session.add(row)
                staged.append((row, qa))

    kept = dedupe([qa for _, qa in staged])

    seen: set[str] = set()
    for row, qa in staged:
        key = normalize(qa.question)
        if key in seen:
            row.status = "discarded"
        else:
            seen.add(key)
            row.status = "kept"
    session.commit()

    chunks = [
        Chunk(
            category="历史问答",
            questions=qa.question,
            answer=qa.answer,
            section_path=None,
            content_type="mined",
            is_key_clause=False,
        )
        for qa in kept
    ]
    if chunks:
        insert_chunks(session, chunks)
        vectorize_pending(session, store, embedder)
    return len(chunks)
