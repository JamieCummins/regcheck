from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

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
        raise RuntimeError("tiktoken is required for token-based chunking")
    return tiktoken.encoding_for_model(model_name)


def extract_chunks_tokens(
    text: str,
    max_chunk_tokens: int = 300,
    encoding_name: str = "text-embedding-3-large",
) -> list[str]:
    """Token-based chunking that keeps chunk boundaries on sentence edges when possible."""
    tokenizer = _get_tokenizer(encoding_name)
    sentences: list[str]
    if nltk is None:
        sentences = _fallback_sentence_split(text)
    else:
        from nltk.tokenize import sent_tokenize

        try:
            sentences = sent_tokenize(text)
        except LookupError:
            _ensure_nltk_sentence_tokenizer()
            try:
                sentences = sent_tokenize(text)
            except Exception:
                sentences = _fallback_sentence_split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = tokenizer.encode(sentence)
        sent_len = len(sent_tokens)

        # Keep sentences intact even if they exceed the target token count; better to have a large
        # chunk than to truncate mid-sentence.
        if sent_len > max_chunk_tokens:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_tokens = 0
            chunks.append(sentence.strip())
            continue

        if current_tokens + sent_len <= max_chunk_tokens:
            current.append(sentence)
            current_tokens += sent_len
        else:
            if current:
                chunks.append(" ".join(current).strip())
            current = [sentence]
            current_tokens = sent_len

    if current:
        chunks.append(" ".join(current).strip())

    # Remove empties
    return [c for c in chunks if c]


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

    if max_segments is not None and max_segments > 0 and len(segments) > max_segments:
        segments = segments[:max_segments]
        embeddings = embeddings[: max_segments]
    if embeddings_path and not os.path.exists(embeddings_path):
        save_embeddings(segments, embeddings, embeddings_path)

    chunk_ids = [
        f"{(chunk_prefix or 'CHUNK').upper()}_{i+1:04d}" for i in range(len(segments))
    ]
    norms = np.linalg.norm(embeddings, axis=1).astype(np.float32, copy=False)
    norms[norms == 0] = 1.0
    return EmbeddingCorpus(
        segments=list(segments),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        chunk_ids=list(chunk_ids),
        norms=norms,
    )
