import os
from types import SimpleNamespace

import pytest

import backend.cli as cli

_ENV_KEYS = [
    "EMBEDDINGS_MODEL",
    "EMBEDDINGS_BASE_URL",
    "EMBEDDINGS_API_KEY",
    "GPUSTACK_API_KEY",
    "GPUSTACK_BASE_URL",
]


@pytest.fixture(autouse=True)
def _isolate_runtime_env():
    # `_apply_runtime_env` mutates os.environ directly (not via monkeypatch), so
    # snapshot the relevant keys, clear them for a known-empty start, and restore.
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _args(**kw):
    base = dict(client="openai", embedding_model=None, parser_choice="pymupdf")
    base.update(kw)
    return SimpleNamespace(**base)


def test_gpustack_routes_embeddings_to_campus_endpoint():
    os.environ["GPUSTACK_API_KEY"] = "gpustack_test"
    cli._apply_runtime_env(_args(client="gpustack"))
    # No --embedding-model given → default to GPUStack's embedding model, and point
    # the embeddings endpoint + key at GPUStack so retrieval stays local too.
    assert os.environ["EMBEDDINGS_MODEL"] == "qwen3-embedding-0.6b"
    assert os.environ["EMBEDDINGS_BASE_URL"] == "https://gpustack.unibe.ch/v1"
    assert os.environ["EMBEDDINGS_API_KEY"] == "gpustack_test"


def test_explicit_embedding_model_wins():
    os.environ["GPUSTACK_API_KEY"] = "gpustack_test"
    cli._apply_runtime_env(_args(client="gpustack", embedding_model="my-embed"))
    assert os.environ["EMBEDDINGS_MODEL"] == "my-embed"


def test_existing_embeddings_endpoint_is_not_overridden():
    os.environ["GPUSTACK_API_KEY"] = "gpustack_test"
    os.environ["EMBEDDINGS_BASE_URL"] = "https://my.local/v1"
    cli._apply_runtime_env(_args(client="gpustack"))
    assert os.environ["EMBEDDINGS_BASE_URL"] == "https://my.local/v1"  # respected


def test_non_gpustack_leaves_endpoint_unset():
    cli._apply_runtime_env(_args(client="openai", embedding_model="text-embedding-3-large"))
    assert os.environ["EMBEDDINGS_MODEL"] == "text-embedding-3-large"
    assert "EMBEDDINGS_BASE_URL" not in os.environ


def test_cli_parser_accepts_gpustack_and_embedding_model():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "general",
            "--preregistration", "p.pdf",
            "--paper", "a.pdf",
            "--dimensions-csv", "d.csv",
            "--client", "gpustack",
            "--embedding-model", "qwen3-embedding-0.6b",
            "--parser-choice", "pymupdf",
        ]
    )
    assert args.client == "gpustack"
    assert args.embedding_model == "qwen3-embedding-0.6b"
