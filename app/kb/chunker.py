"""Structure-aware Markdown chunker (pure functions, no DB / LLM / vector store).

Turns Markdown text into a list of :class:`Chunk` for the offline knowledge-base
pipeline. The rules implemented here follow the ch03 spec (design decisions
6.2 / 6.3):

- The text is split into sections by ATX headings (``#`` .. ``######``). Each
  section carries the path of its ancestor headings; the current section title
  is kept separately.
- ``category`` is the ancestor heading path (``"退货政策 > 运费说明"``), or
  ``fallback_category`` when the section has no parent heading.
- ``section_path`` is the full path *including* the current heading, or ``None``
  for headingless input (used only for provenance).
- For ``content_type="faq"`` the section body is parsed into question/answer
  pairs. Convention (decided here, per SDD ruling R5): a line ``Q: ...`` or
  ``问: ...`` starts a question, the following ``A: ...`` / ``答: ...`` lines
  (which may span several lines) form its answer. ``questions`` gets the real
  question text, ``answer`` gets the answer text. Long answers are split with
  the same ``max_chars``/``overlap`` rules (each piece keeps the same question).
  An FAQ section whose body contains no ``Q:``/``A:`` marker yields no chunks.
- For ``content_type in ("policy", "manual")`` ``questions`` gets the current
  section title (``""`` when there is no heading) and ``answer`` gets the body.
- A section body shorter than or equal to ``max_chars`` is emitted as one chunk;
  longer bodies are split recursively: paragraphs (``\\n\\n``) first, then
  sentences (keeping the terminating ``。！？.!?``) inside a paragraph that still
  exceeds ``max_chars``, then packed greedily with ``overlap`` characters of
  overlap whose boundary is cut back to the nearest sentence end. Text with no
  sentence punctuation is hard-split by width so it always terminates.
- Markdown pipe tables (consecutive lines starting with ``|``, with a header row
  plus at least one data row) are split row-by-row; every chunk re-includes the
  header row and is prefixed with the nearest heading above the table.
- ``is_key_clause`` is a simple heuristic: ``True`` when the answer hits any of
  ``必须 | 不得 | 一律 | 免责 | 责任``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_END = "。！？.!?"
_SENTENCE_RE = re.compile(rf"[^{_SENTENCE_END}]+[{_SENTENCE_END}]*")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_QUESTION_RE = re.compile(r"^\s*(?:Q|问)\s*[:：]\s*(.*)$")
_ANSWER_RE = re.compile(r"^\s*(?:A|答)\s*[:：]\s*(.*)$")

_KEY_CLAUSE_WORDS = ("必须", "不得", "一律", "免责", "责任")


@dataclass
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str | None
    content_type: str
    is_key_clause: bool


def _is_key_clause(text: str) -> bool:
    return any(word in text for word in _KEY_CLAUSE_WORDS)


def _split_sentences(text: str) -> list[str]:
    """Split into sentence units, keeping the terminating punctuation attached."""
    return [m for m in _SENTENCE_RE.findall(text) if m.strip()]


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """Width-based split for text without sentence punctuation (always terminates)."""
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    step = max_chars - overlap
    if step <= 0:
        step = max_chars
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks


def _overlap_tail(text: str, overlap: int) -> str:
    """Return a sentence-aligned suffix of ``text``, at most ``overlap`` chars.

    The overlap boundary is cut forward to the first sentence-ending mark so we
    never resume in the middle of a sentence.
    """
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    for idx, ch in enumerate(tail):
        if ch in _SENTENCE_END:
            return tail[idx + 1 :]
    return tail


def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines (``\\n\\n``) into stripped, non-empty paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _pack_units(units: list[tuple[str, bool]], max_chars: int, overlap: int) -> list[str]:
    """Greedily pack paragraph/sentence units into chunks of at most ``max_chars``.

    ``units`` are ``(text, starts_new_paragraph)`` pairs; ``starts_new_paragraph``
    decides whether a ``\\n\\n`` separator is inserted before the unit so that
    paragraph boundaries survive inside a chunk.
    """
    chunks: list[str] = []
    current = ""

    for text, starts_new_paragraph in units:
        sep = "\n\n" if (starts_new_paragraph and current) else ""
        if not current:
            current = text
        elif len(current) + len(sep) + len(text) <= max_chars:
            current += sep + text
        else:
            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            current = text if not tail or len(tail) + len(text) > max_chars else tail + text

    if current:
        chunks.append(current)
    return chunks


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Spec: split by paragraph (``\n\n``) first; only fall back to sentence
    # packing inside a paragraph that still exceeds ``max_chars``. Text with no
    # sentence punctuation is hard-split by width so it always terminates.
    units: list[tuple[str, bool]] = []
    for paragraph in _split_paragraphs(text):
        if len(paragraph) <= max_chars:
            units.append((paragraph, True))
        else:
            for idx, sentence in enumerate(_split_sentences(paragraph)):
                if len(sentence) > max_chars:
                    for sub_idx, sub in enumerate(_hard_split(sentence, max_chars, overlap)):
                        units.append((sub, idx == 0 and sub_idx == 0))
                else:
                    units.append((sentence, idx == 0))

    return _pack_units(units, max_chars, overlap)


def _parse_qa(text: str) -> list[tuple[str, str]]:
    """Parse ``Q:``/``A:`` (or ``问:``/``答:``) pairs out of an FAQ section body."""
    pairs: list[tuple[str, str]] = []
    question: str | None = None
    answer_lines: list[str] | None = None

    def flush() -> None:
        nonlocal question, answer_lines
        if question is not None and answer_lines is not None:
            answer = "\n".join(answer_lines).strip()
            if question and answer:
                pairs.append((question, answer))
        question = None
        answer_lines = None

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        q_match = _QUESTION_RE.match(raw_line)
        a_match = _ANSWER_RE.match(raw_line)
        if q_match:
            flush()
            question = q_match.group(1).strip()
            answer_lines = []
        elif a_match:
            if question is None:
                continue
            if answer_lines is None:
                answer_lines = []
            answer_lines.append(a_match.group(1).strip())
        elif answer_lines is not None:
            # Continuation of a multi-line answer.
            answer_lines.append(line)

    flush()
    return pairs


def _split_table(lines: list[str], max_chars: int, heading: str | None) -> list[str]:
    """Split a pipe-table run row-by-row, re-including the header in every chunk."""
    if len(lines) < 2:
        return []
    header = lines[0].strip()
    prefix = f"{heading}\n" if heading else ""
    chunks: list[str] = []
    current_rows: list[str] = []

    for row in lines[1:]:
        row = row.strip()
        candidate_rows = current_rows + [row]
        candidate = header + "\n" + "\n".join(candidate_rows)
        if current_rows and len(prefix + candidate) > max_chars:
            chunks.append(prefix + header + "\n" + "\n".join(current_rows))
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        chunks.append(prefix + header + "\n" + "\n".join(current_rows))
    return chunks


def _chunk_section_body(
    body: str,
    *,
    max_chars: int,
    overlap: int,
    heading: str | None,
) -> list[str]:
    """Split a non-FAQ section body: pipe tables row-by-row, other text by length."""
    chunks: list[str] = []
    text_lines: list[str] = []
    table_lines: list[str] = []

    def flush_text() -> None:
        nonlocal text_lines
        if text_lines:
            chunks.extend(_split_long_text("\n".join(text_lines), max_chars, overlap))
            text_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        if len(table_lines) >= 2:
            chunks.extend(_split_table(table_lines, max_chars, heading))
        else:
            # A lone ``| ... |`` line is not a usable table; treat it as text.
            text_lines.extend(table_lines)
        table_lines = []

    for raw_line in body.split("\n"):
        if raw_line.strip().startswith("|"):
            flush_text()
            table_lines.append(raw_line)
        else:
            if table_lines:
                flush_table()
            text_lines.append(raw_line)

    flush_table()
    flush_text()
    return chunks


def chunk_markdown(
    text: str,
    content_type: str,
    *,
    max_chars: int = 800,
    overlap: int = 100,
    fallback_category: str = "未分类",
) -> list[Chunk]:
    if not text:
        return []

    chunks: list[Chunk] = []
    # Stack of open sections: (level, title, body_lines). A "root" entry with
    # level 0 / title None holds content that appears before any heading.
    stack: list[list] = [[0, None, []]]

    def finalize(section: list) -> None:
        _, title, body_lines = section
        body = "\n".join(body_lines).strip()
        if not body:
            return  # skip empty sections (incl. headings with only sub-headings)

        ancestors = [s[1] for s in stack[1:] if s[1] is not None]
        category = ">".join(ancestors) if ancestors else fallback_category
        section_path = ">".join(ancestors + [title]) if title else None

        if content_type == "faq":
            for question, answer in _parse_qa(body):
                for piece in _split_long_text(answer, max_chars, overlap):
                    chunks.append(
                        Chunk(
                            category=category,
                            questions=question,
                            answer=piece,
                            section_path=section_path,
                            content_type=content_type,
                            is_key_clause=_is_key_clause(piece),
                        )
                    )
            return

        questions = title or ""
        for answer in _chunk_section_body(
            body, max_chars=max_chars, overlap=overlap, heading=title
        ):
            chunks.append(
                Chunk(
                    category=category,
                    questions=questions,
                    answer=answer,
                    section_path=section_path,
                    content_type=content_type,
                    is_key_clause=_is_key_clause(answer),
                )
            )

    for raw_line in text.split("\n"):
        match = _HEADING_RE.match(raw_line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                finalize(stack.pop())
            stack.append([level, title, []])
        else:
            stack[-1][2].append(raw_line)

    while stack:
        finalize(stack.pop())

    return chunks
