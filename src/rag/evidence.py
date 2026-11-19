"""Check explicit fact requests before returning local extractive answers."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class EvidenceText:
    text: str
    product_version: str | None = None


_CLAUSE = re.compile(r"\band\s+(?=(?:at\s+)?(?:what|which|how|when|why|who)\b)|[?？\n]+", re.I)
_SENTENCE = re.compile(r"(?<=[.!?;])\s+|[。！？；\n]+")
_TERM = re.compile(r"[a-z0-9]+(?:[_.-][a-z0-9]+)*", re.I)
_QUESTION_WORDS = frozenset(
    "a about after an and are as at be before can could current did do does for from had has have "
    "how in is it may must of on or should that the this to was were what when where which "
    "who why will with would".split()
)
_VERSION = re.compile(r"(?<![a-z0-9_.])v?(\d+\.\d+(?:\.\d+)*)(?![a-z0-9_]|\.\d)", re.I)
_DEFAULT = re.compile(r"\bdefault\b|默认", re.I)
_TIMEOUT = re.compile(r"\btime[ -]?outs?\b|超时", re.I)
_DEFAULT_TIMEOUT_QUESTION = re.compile(
    r"^\s*(?:what|which)\b|\bhow\s+(?:long|many)\b|多少|多久|多长", re.I,
)
_DURATION = re.compile(
    r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"[\s-]*(?:milliseconds?|ms|seconds?|secs?|minutes?|mins?|hours?|days?|weeks?)\b"
    r"|[零一二两三四五六七八九十百\d]+(?:\.\d+)?\s*(?:毫秒|秒|分钟|小时|天|周)",
    re.I,
)
_LOOKUPS = (
    (
        re.compile(r"\b(?:what|which)\b.*\bdate\b|哪天|几月几日|什么日期|日期.{0,10}多少", re.I),
        re.compile(
            r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"
            r"|\b(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December)\s+\d{1,2}\b"
            r"|(?:\d{4}年)?\d{1,2}月\d{1,2}[日号]",
            re.I,
        ),
    ),
    (
        re.compile(r"\b(?:what|which)\b.*\b(?:shell |cli )?command\b|什么命令|哪条命令|命令是什么", re.I),
        re.compile(
            r"(?m)^\s*\$\s+[a-zA-Z]\S*[^\n]*"
            r"|(?i:command|run|execute|命令|执行)[:：]?\s*`[a-zA-Z][^\s`]*(?: [^`\n]+)?`"
            r"|\x60\x60\x60(?:bash|shell|sh|console)\s*\n[^\x60]+",
        ),
    ),
    (
        re.compile(r"\btable name\b|数据库表名|表名.{0,8}(?:什么|哪|是)", re.I),
        re.compile(
            r"\bcreate\s+table\s+(?:if\s+not\s+exists\s+)?[\w.\"\x60]+"
            r"|\btable(?:\s+name)?(?:\s+is|\s*[:=])\s*`?[A-Za-z_][\w.]*"
            r"|\btable\s+`[A-Za-z_][\w.]*`|表名(?:是|为|[:：])\s*`?[A-Za-z_][\w.]*",
            re.I,
        ),
    ),
    (
        re.compile(r"\bhow long\b|\bhow many (?:milli)?seconds\b|"
                   r"\bhow many (?:minutes|hours|days|weeks)\b|多久|多长时间|多少(?:毫秒|秒|分钟|小时|天)", re.I),
        _DURATION,
    ),
)


def _terms(text: str) -> set[str]:
    return {
        term.removesuffix("s") if term.isalpha() and len(term) >= 4 else term
        for term in _TERM.findall(text.casefold())
        if term not in _QUESTION_WORDS
    }


def _versions(text: str) -> set[str]:
    return set(_VERSION.findall(_DURATION.sub(" ", text)))


def _supporting_excerpt(
    evidence: EvidenceText, value: re.Pattern[str], *,
    default_timeout: bool, versions: set[str],
) -> str | None:
    version = (evidence.product_version or "").strip().casefold().removeprefix("v")
    if default_timeout and versions and version and version not in versions:
        return None
    if not default_timeout:
        match = value.search(evidence.text)
        if match is None:
            return None
        if match.end() <= 280:
            return evidence.text
        return evidence.text[max(0, match.start() - 100):match.end() + 100]

    document_versions = {version} if version else _versions(evidence.text)
    for sentence in _SENTENCE.split(evidence.text):
        if not (_DEFAULT.search(sentence) and _TIMEOUT.search(sentence) and value.search(sentence)):
            continue
        matching_versions = _versions(sentence) or document_versions
        if not versions or versions <= matching_versions:
            return sentence
    return None


def select_local_evidence(
    question: str, candidates: list[EvidenceText], *, limit: int,
) -> list[tuple[int, str]]:
    """Prefer explicit values for typed lookups and refuse when a required value is absent."""
    # Ignore isolated matches such as 'rate' in unrelated finance and telemetry questions.
    terms = _terms(question)
    candidate_terms = set().union(*(_terms(candidate.text) for candidate in candidates))
    if len(terms) >= 3 and len(terms & candidate_terms) < 2:
        return []

    selected: dict[int, str] = {}
    open_ended = False
    for clause in filter(str.strip, _CLAUSE.split(question)):
        default_timeout = bool(
            _DEFAULT.search(clause) and _TIMEOUT.search(clause) and _DEFAULT_TIMEOUT_QUESTION.search(clause)
        )
        lookup = _DURATION if default_timeout else next(
            (value for intent, value in _LOOKUPS if intent.search(clause)), None,
        )
        if lookup is None:
            open_ended = True
            continue

        versions = _versions(clause) if default_timeout else set()
        for index, candidate in enumerate(candidates):
            excerpt = _supporting_excerpt(
                candidate, lookup, default_timeout=default_timeout, versions=versions,
            )
            if excerpt is not None:
                if index in selected and excerpt not in selected[index]:
                    selected[index] += "\n" + excerpt
                else:
                    selected.setdefault(index, excerpt)
                break
        else:
            return []

    if len(selected) > limit:
        return []
    if open_ended or not selected:
        for index, evidence in enumerate(candidates):
            if len(selected) >= limit:
                break
            selected.setdefault(index, evidence.text)
    return list(selected.items())
