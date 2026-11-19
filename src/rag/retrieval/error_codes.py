from __future__ import annotations

import re

from src.rag.models import RetrievedChunk, RetrievalFilter


_CODE_LOOKUP = re.compile(
    r"(?:(?:what (?:is|does)|explain|define|meaning of)\s+(?:the\s+)?"
    r"|(?:什么是|请?解释(?:一下)?)\s*)?"
    r"(?:error(?:\s+code)?\s+|错误码[:：]?\s*)?"
    r"`?(?P<code>[a-z][a-z0-9]*(?:[-_][a-z0-9]+)+)`?"
    r"(?:\s+(?:mean|stand for)|\s*(?:(?:是)?(?:什么|啥)(?:意思|含义)?|的含义))?"
    r"\s*[?？!！。.]?",
    re.I,
)


def lookup_error_code(query: str) -> str | None:
    lookup = _CODE_LOOKUP.fullmatch(query.strip())
    if lookup is None:
        return None
    code = lookup["code"]
    if not any(char.isdigit() for char in code) and not code.upper().startswith(("E_", "ERR_", "ERROR_")):
        return None
    return code


def find_error_code_definition(
    code: str | None, candidates: list[RetrievedChunk], filters: RetrievalFilter | None = None,
) -> RetrievedChunk | None:
    """Select one scoped hit that defines the requested error code."""
    if code is None:
        return None
    exact = rf"(?<![a-z0-9_-]){re.escape(code)}(?![a-z0-9_-])"
    definition = re.compile(
        rf"{exact}`?(?:[ \t]+(?:means|indicates|stands for|is (?:emitted|raised|returned|"
        rf"triggered) (?:when|if|for|by|because))\b[ \t]+\S"
        rf"|[ \t]*(?:表示|意味着|代表)[ \t]*[^\s。！？])"
        rf"|^[ \t]*(?:[-*#]+[ \t]+)?(?:error code[ \t]+|错误码[ \t]*)?"
        rf"`?{exact}`?[ \t]*[:：–—][ \t]*\S",
        re.I | re.M,
    )
    return next(
        (
            chunk for chunk in candidates
            if (filters is None or filters.matches(chunk)) and definition.search(chunk.text)
        ),
        None,
    )


def reserve_definition(
    ranked: list[RetrievedChunk], definition: RetrievedChunk | None, limit: int,
) -> list[RetrievedChunk]:
    if definition is None:
        return ranked[:limit]
    # Keep the fused/reranked metadata when the same chunk survived that stage.
    key = (definition.doc_id, definition.chunk_id)
    rest = []
    for chunk in ranked:
        if (chunk.doc_id, chunk.chunk_id) == key:
            definition = chunk
        else:
            rest.append(chunk)
    return [definition, *rest[:limit - 1]]
