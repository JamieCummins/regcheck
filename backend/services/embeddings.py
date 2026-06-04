from __future__ import annotations

import os
import pickle
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import nltk
except ModuleNotFoundError:  # pragma: no cover - graceful fallback
    nltk = None
else:  # pragma: no cover
    nltk.data.path.append("./nltk_data")

try:  # pragma: no cover - optional dependency
    import tiktoken
except ModuleNotFoundError:  # pragma: no cover - graceful fallback
    tiktoken = None

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?:\n{2,}|(?<=[.!?])\s+)")


@dataclass(slots=True)
class TextChunk:
    text: str
    start: int
    end: int


def _fallback_sentence_split(text: str) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_PATTERN.split(cleaned) if part.strip()]


@lru_cache(maxsize=1)
def _ensure_nltk_sentence_tokenizer() -> None:
    if nltk is None:  # pragma: no cover - optional dependency
        return

    data_dir = Path("./nltk_data")
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    data_path = str(data_dir)
    if data_path not in nltk.data.path:
        nltk.data.path.append(data_path)

    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.download(resource, download_dir=data_path, quiet=True)
        except Exception:
            continue


def _get_tokenizer(model_name: str = "text-embedding-3-large"):
    if tiktoken is None:
        return _FallbackTokenizer()
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception as exc:  # pragma: no cover - environment/cache dependent
        logger.warning("Falling back to approximate tokenization", extra={"model": model_name}, exc_info=exc)
        return _FallbackTokenizer()


class _FallbackTokenizer:
    """Small offline tokenizer approximation used when tiktoken data is unavailable."""

    def encode(self, text: str) -> list[str]:
        return re.findall(r"\S+", text or "")

    def decode(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)


def extract_chunks_tokens(
    text: str,
    max_chunk_tokens: int = 300,
    encoding_name: str = "text-embedding-3-large",
) -> list[str]:
    """Token-based chunking that keeps chunk boundaries on sentence edges when possible."""
    return [
        chunk.text
        for chunk in extract_chunks_tokens_with_spans(
            text,
            max_chunk_tokens=max_chunk_tokens,
            encoding_name=encoding_name,
        )
    ]


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _fallback_sentence_spans(text: str) -> list[TextChunk]:
    cleaned = text or ""
    if not cleaned.strip():
        return []

    spans: list[TextChunk] = []
    start = 0
    for match in _SENTENCE_SPLIT_PATTERN.finditer(cleaned):
        end = match.start()
        trimmed_start, trimmed_end = _trim_span(cleaned, start, end)
        if trimmed_start < trimmed_end:
            spans.append(TextChunk(cleaned[trimmed_start:trimmed_end], trimmed_start, trimmed_end))
        start = match.end()

    trimmed_start, trimmed_end = _trim_span(cleaned, start, len(cleaned))
    if trimmed_start < trimmed_end:
        spans.append(TextChunk(cleaned[trimmed_start:trimmed_end], trimmed_start, trimmed_end))
    return spans


def _split_oversize_span(
    span: TextChunk,
    *,
    tokenizer,
    max_chunk_tokens: int,
) -> list[TextChunk]:
    words = list(re.finditer(r"\S+", span.text))
    if not words:
        return []

    chunks: list[TextChunk] = []
    current_start: int | None = None
    current_end: int | None = None
    current_text_parts: list[str] = []
    current_tokens = 0

    for word in words:
        word_text = word.group(0)
        try:
            word_tokens = len(tokenizer.encode(word_text))
        except Exception:
            word_tokens = max(1, len(word_text.split()))
        word_tokens = max(1, word_tokens)

        if current_text_parts and current_tokens + word_tokens > max_chunk_tokens:
            assert current_start is not None and current_end is not None
            text = span.text[current_start - span.start : current_end - span.start].strip()
            if text:
                chunks.append(TextChunk(text, current_start, current_end))
            current_start = None
            current_end = None
            current_text_parts = []
            current_tokens = 0

        if current_start is None:
            current_start = span.start + word.start()
        current_end = span.start + word.end()
        current_text_parts.append(word_text)
        current_tokens += word_tokens

    if current_text_parts and current_start is not None and current_end is not None:
        text = span.text[current_start - span.start : current_end - span.start].strip()
        if text:
            chunks.append(TextChunk(text, current_start, current_end))
    return chunks


def extract_chunks_tokens_with_spans(
    text: str,
    max_chunk_tokens: int = 300,
    encoding_name: str = "text-embedding-3-large",
) -> list[TextChunk]:
    """Token-based chunking with character spans into the original text."""
    tokenizer = _get_tokenizer(encoding_name)
    sentence_spans = _fallback_sentence_spans(text)
    chunks: list[TextChunk] = []
    current: list[TextChunk] = []
    current_tokens = 0

    for sentence in sentence_spans:
        sent_tokens = tokenizer.encode(sentence.text)
        sent_len = len(sent_tokens)

        # If a single sentence exceeds the limit, break it into token-sized slices so no chunk
        # sent to the embeddings API is over the max context.
        if sent_len > max_chunk_tokens:
            if current:
                text_start = current[0].start
                text_end = current[-1].end
                joined = (text or "")[text_start:text_end].strip()
                if joined:
                    chunks.append(TextChunk(joined, text_start, text_end))
                current = []
                current_tokens = 0
            chunks.extend(
                _split_oversize_span(
                    sentence,
                    tokenizer=tokenizer,
                    max_chunk_tokens=max_chunk_tokens,
                )
            )
            continue

        if current_tokens + sent_len <= max_chunk_tokens:
            current.append(sentence)
            current_tokens += sent_len
        else:
            if current:
                text_start = current[0].start
                text_end = current[-1].end
                joined = (text or "")[text_start:text_end].strip()
                if joined:
                    chunks.append(TextChunk(joined, text_start, text_end))
            current = [sentence]
            current_tokens = sent_len

    if current:
        text_start = current[0].start
        text_end = current[-1].end
        joined = (text or "")[text_start:text_end].strip()
        if joined:
            chunks.append(TextChunk(joined, text_start, text_end))

    # Remove empties
    return [c for c in chunks if c.text]


def openai_embed_segments(segments: Sequence[str], model: str = "text-embedding-3-large") -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    max_batch = 2048
    embeddings: list[list[float]] = []
    for start in range(0, len(segments), max_batch):
        batch = segments[start : start + max_batch]
        response = client.embeddings.create(input=list(batch), model=model)
        embeddings.extend(d.embedding for d in response.data)
    return np.asarray(embeddings, dtype=np.float32)


def openai_embed_text(
    text: str,
    model: str = "text-embedding-3-large",
    *,
    max_chunk_tokens: int = 300,
) -> tuple[list[str], np.ndarray]:
    segments = extract_chunks_tokens(text, max_chunk_tokens=max_chunk_tokens, encoding_name=model)
    embeddings = openai_embed_segments(segments, model=model)
    return list(segments), embeddings


def save_embeddings(segments: Sequence[str], embeddings: np.ndarray, path: str) -> None:
    with open(path, "wb") as handle:
        pickle.dump(
            {
                "segments": list(segments),
                "embeddings": np.asarray(embeddings, dtype=np.float32),
            },
            handle,
        )


def load_embeddings(path: str) -> tuple[list[str], np.ndarray]:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return list(data["segments"]), np.asarray(data["embeddings"], dtype=np.float32)


def _coerce_embeddings_matrix(embeddings: np.ndarray, segment_count: int) -> np.ndarray:
    """Ensure embeddings are a 2D float32 matrix with shape (n_segments, dim)."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(0, 0)
    if arr.ndim == 1:
        if arr.size == 0:
            return arr.reshape(0, 0)
        if segment_count <= 1:
            return arr.reshape(1, -1)
        if arr.size % segment_count == 0:
            return arr.reshape(segment_count, -1)
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def retrieve_relevant_chunks(
    query_embedding: np.ndarray,
    corpus: "EmbeddingCorpus",
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[tuple[str, str, float]]:
    if top_k is None and threshold is None:
        raise ValueError("At least one of 'top_k' or 'threshold' must be specified.")

    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
    qvec = query_embedding.astype(np.float32, copy=False).reshape(-1)
    qnorm = float(np.linalg.norm(qvec)) or 1.0

    embeddings = corpus.embeddings
    if embeddings.size == 0:
        return []
    # cosine similarity: (E·q) / (||E|| * ||q||)
    sims = (embeddings @ qvec) / (corpus.norms * qnorm)

    if threshold is not None:
        idx = np.flatnonzero(sims >= threshold)
    else:
        idx = np.arange(sims.shape[0])

    if idx.size == 0:
        return []

    if top_k is not None and idx.size > top_k:
        # argpartition yields top_k indices in arbitrary order; then sort by score desc.
        local = np.argpartition(sims[idx], -top_k)[-top_k:]
        idx = idx[local]
    # Sort by similarity descending
    idx = idx[np.argsort(sims[idx])[::-1]]

    return [
        (corpus.chunk_ids[int(i)], corpus.segments[int(i)], float(sims[int(i)]))
        for i in idx
    ]


def get_top_k_segments_openai(
    segments: Sequence[str],
    embeddings: np.ndarray,
    query: str,
    k: int,
    model: str = "text-embedding-3-large",
) -> list[str]:
    from numpy.linalg import norm

    query_embedding = openai_embed_segments([query], model=model)[0]
    scores = np.array(
        [np.dot(embedding, query_embedding) / (norm(embedding) * norm(query_embedding)) for embedding in embeddings]
    )
    top_indices = np.argsort(-scores)[:k]
    return [segments[index] for index in top_indices]


def get_embedding(text: str, model: str = "text-embedding-3-large"):
    return openai_embed_segments([text], model=model)[0]


def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    if tiktoken is None:
        raise RuntimeError("tiktoken is required for token counting")
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


@dataclass(slots=True)
class EmbeddingCorpus:
    segments: list[str]
    embeddings: np.ndarray
    chunk_ids: list[str]
    norms: np.ndarray
    metadata: list[dict] = field(default_factory=list)


def build_corpus(
    text: str,
    model: str = "text-embedding-3-large",
    embeddings_path: str | None = None,
    chunk_prefix: str | None = None,
    max_segments: int | None = None,
    max_chunk_tokens: int = 300,
) -> EmbeddingCorpus:
    segments: list[str]
    embeddings: np.ndarray
    if embeddings_path and os.path.exists(embeddings_path):
        segments, embeddings = load_embeddings(embeddings_path)
    else:
        segments, embeddings = openai_embed_text(text, model=model, max_chunk_tokens=max_chunk_tokens)

    embeddings = _coerce_embeddings_matrix(embeddings, len(segments))
    if embeddings.shape[0] != len(segments):
        min_rows = min(len(segments), int(embeddings.shape[0]))
        if min_rows == 0:
            segments = []
            embeddings = embeddings[:0]
        else:
            logger.warning(
                "Embedding corpus row mismatch; truncating",
                extra={"segment_count": len(segments), "embedding_rows": int(embeddings.shape[0]), "keep": min_rows},
            )
            segments = list(segments[:min_rows])
            embeddings = embeddings[:min_rows]

    if max_segments is not None and max_segments > 0 and len(segments) > max_segments:
        segments = segments[:max_segments]
        embeddings = embeddings[: max_segments]
    if embeddings_path and not os.path.exists(embeddings_path):
        save_embeddings(segments, embeddings, embeddings_path)

    chunk_ids = [
        f"{(chunk_prefix or 'CHUNK').upper()}_{i+1:04d}" for i in range(len(segments))
    ]
    if embeddings.size == 0:
        norms = np.empty((0,), dtype=np.float32)
    else:
        # Final defensive reshape to avoid AxisError if earlier coercion missed anything.
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        elif embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        try:
            norms = np.linalg.norm(embeddings, axis=1).astype(np.float32, copy=False)
        except np.AxisError:
            embeddings = np.atleast_2d(embeddings)
            norms = np.linalg.norm(embeddings, axis=1).astype(np.float32, copy=False)
        norms[norms == 0] = 1.0
    return EmbeddingCorpus(
        segments=list(segments),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        chunk_ids=list(chunk_ids),
        norms=norms,
        metadata=[],
    )


def build_corpus_from_segments(
    segments: Sequence[str],
    *,
    model: str = "text-embedding-3-large",
    chunk_prefix: str | None = None,
    max_segments: int | None = None,
    metadata: Sequence[dict] | None = None,
) -> EmbeddingCorpus:
    """Build an embedding corpus from pre-chunked text and optional metadata."""
    limited_segments = list(segments)
    limited_metadata = list(metadata or [])
    if max_segments is not None and max_segments > 0 and len(limited_segments) > max_segments:
        limited_segments = limited_segments[:max_segments]
        limited_metadata = limited_metadata[:max_segments]

    if limited_segments:
        embeddings = openai_embed_segments(limited_segments, model=model)
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)
    embeddings = _coerce_embeddings_matrix(embeddings, len(limited_segments))
    if embeddings.size == 0:
        norms = np.empty((0,), dtype=np.float32)
    else:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1).astype(np.float32, copy=False)
        norms[norms == 0] = 1.0

    chunk_ids = [
        f"{(chunk_prefix or 'CHUNK').upper()}_{i+1:04d}" for i in range(len(limited_segments))
    ]
    if len(limited_metadata) < len(limited_segments):
        limited_metadata.extend({} for _ in range(len(limited_segments) - len(limited_metadata)))

    return EmbeddingCorpus(
        segments=list(limited_segments),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        chunk_ids=chunk_ids,
        norms=norms,
        metadata=list(limited_metadata[: len(limited_segments)]),
    )
