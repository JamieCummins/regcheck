import pickle

import numpy as np

import backend.services.embeddings as embeddings
from backend.services.embeddings import (
    build_corpus,
    build_corpus_from_segments,
    extract_chunks_tokens_with_spans,
    save_embeddings,
)


def _capture_openai_kwargs(monkeypatch):
    captured = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    embeddings._embed_client.cache_clear()
    return captured


def test_embed_client_routes_to_configured_endpoint(monkeypatch):
    # EMBEDDINGS_BASE_URL/_API_KEY let retrieval embeddings run on a local
    # OpenAI-compatible provider (e.g. GPUStack) instead of hosted OpenAI.
    captured = _capture_openai_kwargs(monkeypatch)
    monkeypatch.setenv("EMBEDDINGS_BASE_URL", "https://gpustack.unibe.ch/v1")
    monkeypatch.setenv("EMBEDDINGS_API_KEY", "gpustack_test")
    try:
        embeddings._embed_client()
        assert captured["base_url"] == "https://gpustack.unibe.ch/v1"
        assert captured["api_key"] == "gpustack_test"
    finally:
        embeddings._embed_client.cache_clear()


def test_embed_client_defaults_to_hosted_openai(monkeypatch):
    captured = _capture_openai_kwargs(monkeypatch)
    monkeypatch.delenv("EMBEDDINGS_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    try:
        embeddings._embed_client()
        assert captured["base_url"] is None  # hosted OpenAI default
        assert captured["api_key"] == "sk-openai"
    finally:
        embeddings._embed_client.cache_clear()


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
        # return a 1D array to simulate a mis-shaped legacy embedding
        return np.asarray([5.0, 6.0], dtype=np.float32)

    monkeypatch.setattr(emb_mod, "_coerce_embeddings_matrix", bad_coerce)

    corpus = build_corpus("", embeddings_path=str(path))
    assert corpus.embeddings.shape[0] == 1
    assert corpus.embeddings.shape[1] >= 1
    assert corpus.norms.shape == (1,)


def test_extract_chunks_tokens_with_spans_preserves_source_offsets():
    text = "First sentence. Second sentence with evidence."
    chunks = extract_chunks_tokens_with_spans(text, max_chunk_tokens=100)

    assert chunks
    first = chunks[0]
    assert text[first.start:first.end] == first.text


def test_build_corpus_from_segments_keeps_metadata(monkeypatch):
    import backend.services.embeddings as emb_mod

    def fake_embed(segments, model="text-embedding-3-large"):
        return np.ones((len(segments), 3), dtype=np.float32)

    monkeypatch.setattr(emb_mod, "openai_embed_segments", fake_embed)

    corpus = build_corpus_from_segments(
        ["alpha", "beta"],
        chunk_prefix="PAPER",
        metadata=[{"source_id": "paper"}, {"source_id": "paper"}],
    )

    assert corpus.chunk_ids == ["PAPER_0001", "PAPER_0002"]
    assert corpus.metadata[0]["source_id"] == "paper"
    assert corpus.embeddings.shape == (2, 3)


def test_boundary_aware_chunking_splits_at_headings():
    # A methods fact (exclusion/sample size) must not share a chunk with adjacent results.
    text = (
        "Method\n"
        "Participants were recruited via Prolific. Eight participants were excluded for "
        "failing the attention check, leading to a final sample size of N = 192.\n\n"
        "## Results\nThe effect was significant, t(190) = 2.30, p = .011.\n"
    )
    chunks = extract_chunks_tokens_with_spans(text, max_chunk_tokens=200)
    excl = [c.text for c in chunks if "N = 192" in c.text]
    assert excl, "exclusion fact should be present in some chunk"
    assert not any("t(190)" in t for t in excl), "exclusion fact must be split from results stats"
    # Character spans must map back to the source text.
    for c in chunks:
        assert c.text.strip() == text[c.start:c.end].strip()


def test_chunking_falls_back_without_headings():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = extract_chunks_tokens_with_spans(text, max_chunk_tokens=200)
    assert len(chunks) == 1  # no headings -> single block -> prior behaviour
