"""Citation extraction and validation.

Required citation format:  ``[Source: HR-001, Chunk: HR-001-CH02]``

A citation is **valid** only if its chunk ID is one of the chunks actually given
to the LLM for this question *and* its policy ID matches that chunk's policy.
Citation-looking text that does not meet both conditions is reported as invalid,
so an answer cannot pass validation by containing arbitrary bracketed text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieval import RetrievedChunk

STRICT_CITATION = re.compile(
    r"\[Source:\s*(?P<policy_id>[A-Z]{2,5}-\d{3})\s*,\s*Chunk:\s*(?P<chunk_id>[A-Z]{2,5}-\d{3}-CH\d{2})\s*\]"
)
# Anything that *looks* like an attempt at a citation, used to catch malformed ones.
LOOSE_CITATION = re.compile(r"\[\s*(?:Source|Chunk)\s*:[^\]]*\]", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Citation:
    policy_id: str
    chunk_id: str

    def __str__(self) -> str:
        return f"[Source: {self.policy_id}, Chunk: {self.chunk_id}]"


@dataclass
class CitationReport:
    """Result of validating the citations in an answer."""

    valid: list[Citation] = field(default_factory=list)
    invalid: list[tuple[Citation, str]] = field(default_factory=list)  # (citation, reason)
    malformed: list[str] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)

    @property
    def has_citations(self) -> bool:
        return bool(self.valid or self.invalid)

    @property
    def all_valid(self) -> bool:
        """True only if there is at least one citation and none are invalid or malformed."""
        return bool(self.valid) and not self.invalid and not self.malformed

    @property
    def cited_policy_ids(self) -> set[str]:
        return {c.policy_id for c in self.valid}


def extract_citations(answer: str) -> list[Citation]:
    """Return unique strict-format citations in order of first appearance."""
    seen: dict[tuple[str, str], Citation] = {}
    for m in STRICT_CITATION.finditer(answer):
        key = (m.group("policy_id"), m.group("chunk_id"))
        seen.setdefault(key, Citation(*key))
    return list(seen.values())


def _sentences(text: str) -> list[str]:
    # Split after sentence punctuation or after a citation, keeping citations with their sentence.
    parts = re.split(r"(?<=[.!?])\s+(?!\[Source)|(?<=\])\s+(?=[A-Z])", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def validate_citations(answer: str, context_chunks: list[RetrievedChunk]) -> CitationReport:
    """Validate every citation in ``answer`` against the chunks that were given to the LLM."""
    allowed = {c.chunk_id: c.policy_id for c in context_chunks}
    report = CitationReport()

    for citation in extract_citations(answer):
        if citation.chunk_id not in allowed:
            report.invalid.append((citation, "chunk was not in the retrieved context"))
        elif allowed[citation.chunk_id] != citation.policy_id:
            report.invalid.append(
                (citation, f"chunk belongs to {allowed[citation.chunk_id]}, not {citation.policy_id}")
            )
        else:
            report.valid.append(citation)

    for m in LOOSE_CITATION.finditer(answer):
        if not STRICT_CITATION.fullmatch(m.group(0)):
            report.malformed.append(m.group(0))

    # Informational: sentences carrying no citation at all.
    for sentence in _sentences(answer):
        if len(sentence) > 25 and not LOOSE_CITATION.search(sentence):
            report.uncited_sentences.append(sentence)
    return report
