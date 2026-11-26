"""Document loading and chunking.

Each policy file is Markdown with a small front-matter header, e.g.:

    ---
    policy_id: HR-001
    title: Employee Leave Policy
    ...
    ---

Chunking is paragraph-aware: paragraphs are packed into chunks of at most
``chunk_size`` characters, and each new chunk starts with the last
``chunk_overlap`` characters (word-aligned) of the previous chunk so that
facts spanning a boundary are not lost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_FIELDS = ("policy_id", "title", "department")


@dataclass(frozen=True)
class PolicyDocument:
    """A loaded policy document and its metadata."""

    policy_id: str
    title: str
    department: str
    source_file: str
    text: str
    extra_metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A chunk of a policy document, ready to be embedded."""

    chunk_id: str
    text: str
    metadata: dict[str, str | int]

    @property
    def embedding_text(self) -> str:
        """Text that is embedded: the policy title is prepended to give each chunk context."""
        return f"{self.metadata['title']} ({self.metadata['policy_id']})\n{self.text}"


def parse_front_matter(raw: str, source: str = "<string>") -> tuple[dict[str, str], str]:
    """Split a document into its ``key: value`` front matter and its body."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"{source}: missing front-matter header delimited by '---'")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"{source}: invalid front-matter line {line!r}")
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    missing = [f for f in REQUIRED_FIELDS if not meta.get(f)]
    if missing:
        raise ValueError(f"{source}: missing required metadata fields {missing}")
    return meta, match.group(2).strip()


def load_documents(policy_dir: Path) -> list[PolicyDocument]:
    """Load every ``*.md`` policy file in ``policy_dir`` (sorted for reproducibility)."""
    policy_dir = Path(policy_dir)
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"Policy directory not found: {policy_dir}")
    files = sorted(policy_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md policy files found in {policy_dir}")

    documents: list[PolicyDocument] = []
    seen_ids: set[str] = set()
    for path in files:
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"), source=path.name)
        policy_id = meta.pop("policy_id")
        if policy_id in seen_ids:
            raise ValueError(f"Duplicate policy_id {policy_id} in {path.name}")
        seen_ids.add(policy_id)
        documents.append(
            PolicyDocument(
                policy_id=policy_id,
                title=meta.pop("title"),
                department=meta.pop("department"),
                source_file=path.name,
                text=body,
                extra_metadata=meta,
            )
        )
    return documents


def _split_long_block(block: str, chunk_size: int) -> list[str]:
    """Split a paragraph longer than ``chunk_size`` at sentence, then word, boundaries."""
    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", block):
        while len(sentence) > chunk_size:  # pathological: a single very long sentence
            cut = sentence.rfind(" ", 0, chunk_size)
            cut = cut if cut > 0 else chunk_size
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return [p for p in pieces if p]


def _overlap_tail(text: str, overlap: int) -> str:
    """Return roughly the last ``overlap`` characters of ``text``.

    The tail starts at a paragraph or sentence boundary when one leaves at least a
    third of the overlap, otherwise at a word boundary.
    """
    if overlap <= 0:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    boundary = max(tail.find("\n\n"), tail.find(". "))
    if boundary != -1 and len(tail) - boundary >= overlap // 3:
        return tail[boundary:].lstrip(". \n")
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, str]]:
    """Chunk ``text`` and return ``(chunk_text, sections)`` pairs.

    ``sections`` lists (``; ``-separated) the ``##`` sections whose new,
    non-overlap content appears in the chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    # Turn the document into (block, section) units, splitting any oversized block.
    # A heading is never a unit on its own: it is joined to the paragraph that
    # follows it, so a chunk can never end with a dangling heading.
    units: list[tuple[str, str]] = []
    section = "Introduction"
    pending_heading = ""
    for block in (b.strip() for b in re.split(r"\n\s*\n", text)):
        if not block:
            continue
        heading = re.match(r"^#{1,6}\s+(.*)$", block)
        if heading:
            if block.startswith("## "):
                section = heading.group(1).strip()
            pending_heading = f"{pending_heading}\n\n{block}".strip()
            continue
        pieces = _split_long_block(block, chunk_size - chunk_overlap - len(pending_heading) - 2)
        if pending_heading:
            pieces[0] = f"{pending_heading}\n\n{pieces[0]}"
            pending_heading = ""
        units.extend((piece, section) for piece in pieces)
    if pending_heading:
        units.append((pending_heading, section))

    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    current_sections: list[str] = []
    prefix = ""  # overlap text carried from the previous chunk

    def flush() -> None:
        nonlocal current, current_sections, prefix
        body = "\n\n".join(current)
        full = f"{prefix}\n\n{body}".strip() if prefix else body
        chunks.append((full, "; ".join(current_sections)))
        prefix = _overlap_tail(body, chunk_overlap)
        current, current_sections = [], []

    for piece, piece_section in units:
        candidate_len = len(prefix) + 2 + len("\n\n".join(current + [piece]))
        if current and candidate_len > chunk_size:
            flush()
        current.append(piece)
        if piece_section not in current_sections:
            current_sections.append(piece_section)
    if current:
        flush()
    return chunks


def chunk_document(doc: PolicyDocument, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk one document, attaching full metadata and a unique ID to every chunk."""
    chunks = []
    for index, (text, section) in enumerate(chunk_text(doc.text, chunk_size, chunk_overlap), start=1):
        chunk_id = f"{doc.policy_id}-CH{index:02d}"
        metadata: dict[str, str | int] = {
            "chunk_id": chunk_id,
            "policy_id": doc.policy_id,
            "title": doc.title,
            "department": doc.department,
            "source_file": doc.source_file,
            "section": section,
            "chunk_index": index,
            "char_count": len(text),
        }
        for key in ("version", "effective_date", "classification"):
            if key in doc.extra_metadata:
                metadata[key] = doc.extra_metadata[key]
        chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=metadata))
    return chunks


def chunk_documents(docs: list[PolicyDocument], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk all documents and verify chunk IDs are globally unique."""
    all_chunks = [c for d in docs for c in chunk_document(d, chunk_size, chunk_overlap)]
    ids = [c.chunk_id for c in all_chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs generated")
    return all_chunks
