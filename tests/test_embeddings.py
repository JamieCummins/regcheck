import os
import pickle

import numpy as np
import pytest

# Existing tests for disk-cache coercion rely on the OpenAI path loading cached files.
os.environ.setdefault("OPENAI_API_KEY", "test")

from backend.services.embeddings import (
    build_corpus,
    save_embeddings,
    tfidf_embed_query,
    tfidf_embed_text,
)


# ---------------------------------------------------------------------------
# Disk-cache coercion tests (OpenAI path — load pre-saved embeddings from disk)
# ---------------------------------------------------------------------------

def test_build_corpus_handles_empty_embeddings(tmp_path):
    path = tmp_path / "empty.pkl"
    save_embeddings([], np.asarray([], dtype=np.float32), str(path))

    corpus = build_corpus("", embeddings_path=str(path))
    assert corpus.segments == []
    assert corpus.embeddings.ndim == 2
    assert corpus.embeddings.shape[0] == 0
    assert corpus.norms.shape == (0,)


def test_build_corpus_coerces_1d_embedding_vector(tmp_path):
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"segments": ["only"], "embeddings": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)}, handle)

    corpus = build_corpus("", embeddings_path=str(path))
    assert corpus.segments == ["only"]
    assert corpus.embeddings.shape == (1, 3)
    assert corpus.norms.shape == (1,)
    assert np.isclose(float(corpus.norms[0]), np.sqrt(14.0))


def test_build_corpus_axis_error_guard(monkeypatch, tmp_path):
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"segments": ["seg1"], "embeddings": np.asarray([5.0, 6.0], dtype=np.float32)}, handle)

    from backend import services as svc_mod  # noqa: F401  # ensures module path importable
    import backend.services.embeddings as emb_mod

    def bad_coerce(embeddings, segment_count):
        return np.asarray([5.0, 6.0], dtype=np.float32)

    monkeypatch.setattr(emb_mod, "_coerce_embeddings_matrix", bad_coerce)

    corpus = build_corpus("", embeddings_path=str(path))
    assert corpus.embeddings.shape[0] == 1
    assert corpus.embeddings.shape[1] >= 1
    assert corpus.norms.shape == (1,)


# ---------------------------------------------------------------------------
# TF-IDF fallback tests (no API key)
# ---------------------------------------------------------------------------

def test_tfidf_corpus_vectorizer_stored(monkeypatch):
    import backend.services.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "_openai_key_available", lambda: False)

    corpus = build_corpus("The primary outcome was measured at baseline and follow-up.")
    assert corpus.vectorizer is not None
    assert hasattr(corpus.vectorizer, "vocabulary_")
    assert len(corpus.vectorizer.vocabulary_) > 0


def test_tfidf_corpus_embedding_shape(monkeypatch):
    import backend.services.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "_openai_key_available", lambda: False)

    text = "Participants were randomised to treatment or control. Primary outcome was pain at 12 weeks."
    corpus = build_corpus(text)
    assert corpus.embeddings.ndim == 2
    assert corpus.embeddings.shape[0] == len(corpus.segments)
    assert corpus.embeddings.shape[1] > 0
    assert corpus.norms.shape == (len(corpus.segments),)


def test_tfidf_query_embedding_compatible_with_corpus(monkeypatch):
    import backend.services.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "_openai_key_available", lambda: False)

    text = "Participants were randomised to treatment or control. Primary outcome was pain at 12 weeks."
    corpus = build_corpus(text)

    query_vec = tfidf_embed_query("primary outcome pain", corpus.vectorizer)
    assert query_vec.ndim == 1
    assert query_vec.shape[0] == corpus.embeddings.shape[1]


def test_tfidf_retrieval_returns_relevant_chunk(monkeypatch):
    from backend.services.embeddings import retrieve_relevant_chunks
    import backend.services.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "_openai_key_available", lambda: False)

    text = (
        "The primary outcome was reduction in pain score. "
        "Secondary outcomes included quality of life. "
        "Adverse events were recorded throughout the trial."
    )
    corpus = build_corpus(text)
    query_vec = tfidf_embed_query("primary outcome pain score", corpus.vectorizer)
    results = retrieve_relevant_chunks(query_vec, corpus, top_k=1)

    assert len(results) == 1
    _chunk_id, chunk_text, score = results[0]
    assert "pain" in chunk_text.lower() or "outcome" in chunk_text.lower()
    assert score >= 0.0


def test_tfidf_embed_text_empty_input_does_not_crash():
    segments, embeddings, vectorizer = tfidf_embed_text("")
    assert isinstance(segments, list)
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == len(segments)


def test_tfidf_embed_query_empty_vocab_returns_zero_vector():
    from sklearn.feature_extraction.text import TfidfVectorizer
    empty_vectorizer = TfidfVectorizer()
    vec = tfidf_embed_query("anything", empty_vectorizer)
    assert vec.ndim == 1
    assert vec.shape == (1,)
    assert float(vec[0]) == 0.0


def test_tfidf_corpus_does_not_write_disk_cache(monkeypatch, tmp_path):
    import backend.services.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "_openai_key_available", lambda: False)

    path = tmp_path / "should_not_exist.pkl"
    build_corpus("Some real text for the corpus.", embeddings_path=str(path))
    assert not path.exists(), "TF-IDF mode must not write to disk cache"