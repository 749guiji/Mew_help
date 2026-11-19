"""Single source of truth for the knowledge material list.

Both the ``/kb`` admin page and the offline ``ingest_docs`` CLI read from here,
so the list of materials (which files, what ``content_type``) never diverges
between "what the page shows" and "what the build actually ingests".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

KNOWLEDGE_DIR = Path("knowledge")


@dataclass
class KbMaterial:
    name: str
    path: Path
    content_type: str
    chars: int


def content_type_from_path(path: Path) -> str:
    """Derive ``content_type`` from a file name (same rule as the ch03 spec).

    ``faq`` for FAQ files, ``policy`` for policy/return files, otherwise
    ``manual``.
    """
    name = path.name.lower()
    if "faq" in name:
        return "faq"
    if "policy" in name or "退货" in name:
        return "policy"
    return "manual"


def list_kb_materials(directory: Path = KNOWLEDGE_DIR) -> list[KbMaterial]:
    """Return the knowledge materials under ``directory``, sorted by name."""
    out: list[KbMaterial] = []
    for path in sorted(directory.glob("*.md")):
        out.append(
            KbMaterial(
                name=path.name,
                path=path,
                content_type=content_type_from_path(path),
                chars=len(path.read_text(encoding="utf-8")),
            )
        )
    return out


def preview_kb(directory: Path = KNOWLEDGE_DIR) -> dict:
    """Chunk every material without writing anything (dry-run), returning counts."""
    from app.core.config import get_settings
    from app.kb.chunker import chunk_markdown

    settings = get_settings()
    files = chunks = key_clauses = 0
    for material in list_kb_materials(directory):
        files += 1
        cs = chunk_markdown(
            material.path.read_text(encoding="utf-8"),
            material.content_type,
            max_chars=settings.chunk_max_chars,
            overlap=settings.chunk_overlap_chars,
        )
        chunks += len(cs)
        key_clauses += sum(1 for c in cs if c.is_key_clause)
    return {"files": files, "chunks": chunks, "key_clauses": key_clauses}
