"""
Lightweight in-memory RAG for per-client hospital FAQ knowledge bases.

Each client's knowledge base is a single text file (see
client_config.csv's `knowledge_base_file` column, exposed as
state["templates"]["_knowledge_base_file"]). At first use, the file is
split into overlapping chunks and each chunk is embedded once via
OpenAI's embeddings API; the (chunk_text, embedding) pairs are cached in
memory, keyed by the file's path + mtime - so an edited file is picked
up automatically without a restart, but an unchanged file is never
re-embedded on every turn (which would be slow and wasteful).

Deliberately dependency-light: no vector database, no numpy - each
client's knowledge base is a single small-to-medium document, not a
large corpus, so a linear cosine-similarity scan in pure Python is
entirely fast enough at this scale. If a clinic's knowledge base ever
grows to genuinely large size (many documents, thousands of chunks),
this module would need to move to a real vector store instead - not a
concern at the current scale.

GENERIC BY DESIGN: nothing here is specific to any one clinic - the
file path is the only per-client input, exactly like doctors_base_url.
Adding a new clinic's knowledge base is just adding a new text file and
pointing client_config.csv's knowledge_base_file column at it.
"""

import logging
import math
import os
import re
from typing import Optional

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger("rag")

_EMBEDDING_MODEL_NAME = "text-embedding-3-small"
_embeddings_model: Optional[OpenAIEmbeddings] = None

CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150
DEFAULT_TOP_K = 4

# Cache: file_path -> (mtime, [(chunk_text, embedding_vector), ...])
_CACHE: dict = {}


def _get_embeddings_model() -> OpenAIEmbeddings:
    """Lazily construct the embeddings client - avoids requiring
    OPENAI_API_KEY at import time (e.g. for tests that never touch RAG)."""

    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OpenAIEmbeddings(model=_EMBEDDING_MODEL_NAME)
    return _embeddings_model


def _chunk_text(text: str) -> list:
    """Split text into chunks along paragraph/blank-line boundaries
    where possible (keeps related sentences together), falling back to
    a hard character-count split with overlap for any single paragraph
    longer than the chunk size on its own."""

    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 1 <= CHUNK_SIZE_CHARS:
            current = (current + "\n" + para).strip()
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(para) <= CHUNK_SIZE_CHARS:
            current = para
        else:
            step = CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
            for i in range(0, len(para), step):
                chunks.append(para[i:i + CHUNK_SIZE_CHARS])

    if current:
        chunks.append(current)

    return chunks


def _load_and_embed(file_path: str) -> list:
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    chunks = _chunk_text(text)
    if not chunks:
        return []

    logger.info("rag: embedding %d chunk(s) for %s", len(chunks), file_path)

    try:
        vectors = _get_embeddings_model().embed_documents(chunks)
    except Exception:
        logger.exception("rag: failed to embed knowledge base chunks for %s", file_path)
        return []

    return list(zip(chunks, vectors))


def _get_cached_chunks(file_path: str) -> list:
    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        logger.warning("rag: knowledge base file not found: %s", file_path)
        return []

    cached = _CACHE.get(file_path)
    if cached and cached[0] == mtime:
        return cached[1]

    chunks_with_vectors = _load_and_embed(file_path)
    _CACHE[file_path] = (mtime, chunks_with_vectors)
    return chunks_with_vectors


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def list_services(file_path: str, language: str = "ar") -> list:
    """Return the knowledge base's own list of top-level SERVICES, in
    document order.

    WHY THIS EXISTS: "what services do you offer?" is a
    table-of-contents question, and semantic search is the wrong tool
    for it. `search()` returns the passages most similar to the query,
    which for this question means a handful of DETAIL paragraphs -
    confirmed in production, the answer came back as a bulleted mix of
    inpatient amenities (garden, gym, art therapy area, isolation
    rooms) while four of the clinic's six actual services were never
    mentioned at all. Amenities are not services, and a partial list
    misrepresents what the clinic offers.

    The knowledge base is already structured for this: a numbered
    top-level section whose heading names services ("4. الخدمات |
    Services"), with each service as a numbered subheading beneath it
    ("4.1 خدمة الطوارئ النفسية | Psychiatric Emergency Service"). This
    reads those subheadings directly, so the list is always complete,
    always in the clinic's own wording, and always in the clinic's own
    order.

    Returns [] when the file is missing or has no recognizable services
    section - the caller then falls back to normal semantic search.
    """

    if not file_path:
        return []

    try:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        logger.warning("rag: knowledge base file not found for service listing: %s", file_path)
        return []

    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    section_number = None
    for line in lines:
        match = re.match(r"^(\d+)\.\s+(.*)$", line)
        if not match:
            continue
        heading = match.group(2)
        # Heading must NAME services - not merely mention the word in a
        # sentence, so keep this to short heading-like lines.
        if len(heading) <= 80 and re.search(r"الخدمات|services", heading, re.IGNORECASE):
            section_number = match.group(1)
            break

    if section_number is None:
        logger.info("rag: no services section heading found in %s", file_path)
        return []

    services = []
    for line in lines:
        match = re.match(rf"^{section_number}\.(\d+)\s+(.+)$", line)
        if not match:
            continue

        heading = match.group(2).strip()
        arabic_part, _, english_part = heading.partition("|")
        arabic_part = arabic_part.strip()
        english_part = english_part.strip()

        name = (english_part or arabic_part) if language == "en" else (arabic_part or english_part)
        if name:
            services.append(name)

    logger.info("rag: %d service(s) read from the services section of %s", len(services), file_path)

    return services


def search(file_path: str, query: str, top_k: int = DEFAULT_TOP_K) -> list:
    """Return the top_k most relevant chunks (plain strings, most
    relevant first) for `query` from the knowledge base at `file_path`.
    Returns [] if the file is missing/empty, or the query/chunks
    couldn't be embedded (e.g. a transient API error)."""

    if not file_path:
        return []

    chunks_with_vectors = _get_cached_chunks(file_path)
    if not chunks_with_vectors:
        return []

    try:
        query_vector = _get_embeddings_model().embed_query(query)
    except Exception:
        logger.exception("rag: failed to embed query %r", query)
        return []

    scored = [
        (chunk, _cosine_similarity(query_vector, vector))
        for chunk, vector in chunks_with_vectors
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [chunk for chunk, _score in scored[:top_k]]
