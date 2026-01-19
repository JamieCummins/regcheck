from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - graceful fallback
    pd = None


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
    if nltk is None:
        raise RuntimeError("nltk is required for sentence tokenization.")
    tokenizer = _get_tokenizer(encoding_name)
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
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


def openai_embed_text(text: str, model: str = "text-embedding-3-large") -> tuple[list[str], np.ndarray]:
    segments = extract_chunks_tokens(text, encoding_name=model)
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


def _segments_to_df(
    segments: Sequence[str],
    embeddings: np.ndarray,
    chunk_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    if pd is None:
        raise RuntimeError("pandas is required to create embedding DataFrames")
    ids = (
        list(chunk_ids)
        if chunk_ids is not None
        else [f"CHUNK_{i+1:04d}" for i in range(len(segments))]
    )
    return pd.DataFrame({"chunk_id": ids, "chunk": list(segments), "embedding": list(embeddings)})


def retrieve_relevant_chunks(
    prompt: str,
    df: pd.DataFrame,
    top_k: int | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    if top_k is None and threshold is None:
        raise ValueError("At least one of 'top_k' or 'threshold' must be specified.")

    if pd is None:
        raise RuntimeError("pandas is required for similarity retrieval")

    topic_embedding = get_embedding(prompt)
    similarities = cosine_similarity([topic_embedding], np.vstack(df["embedding"]))[0]
    df = df.copy()
    df["similarity"] = similarities
    df_sorted = df.sort_values(by="similarity", ascending=False)

    if threshold is not None:
        df_sorted = df_sorted[df_sorted["similarity"] >= threshold]
    if top_k is not None:
        df_sorted = df_sorted.head(top_k)
    return df_sorted[["chunk_id", "chunk", "similarity"]]


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
    df: Any


def build_corpus(
    text: str,
    model: str = "text-embedding-3-large",
    embeddings_path: str | None = None,
    chunk_prefix: str | None = None,
) -> EmbeddingCorpus:
    segments: list[str]
    embeddings: np.ndarray
    if embeddings_path and os.path.exists(embeddings_path):
        segments, embeddings = load_embeddings(embeddings_path)
    else:
        segments, embeddings = openai_embed_text(text, model=model)
        if embeddings_path:
            save_embeddings(segments, embeddings, embeddings_path)
    chunk_ids = [
        f"{(chunk_prefix or 'CHUNK').upper()}_{i+1:04d}" for i in range(len(segments))
    ]
    df = _segments_to_df(segments, embeddings, chunk_ids=chunk_ids)
    return EmbeddingCorpus(segments=list(segments), embeddings=embeddings, df=df)
