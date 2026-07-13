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


# --- harmonisation with the API/app: dimension presets, parser default, OSF source ---

def test_dimension_set_resolves_backend_preset():
    args = SimpleNamespace(dimensions_csv=None, dimension_set="psychology")
    dims = cli._resolve_dimensions_arg(args, require=True)
    assert len(dims) == 9
    assert dims[0]["dimension"] == "Hypotheses"


def test_dimension_set_and_csv_are_mutually_exclusive():
    args = SimpleNamespace(dimensions_csv="x.csv", dimension_set="psychology")
    with pytest.raises(ValueError, match="only one"):
        cli._resolve_dimensions_arg(args, require=True)


def test_dimensions_required_for_general_but_optional_elsewhere():
    args = SimpleNamespace(dimensions_csv=None, dimension_set=None)
    with pytest.raises(ValueError, match="Provide dimensions"):
        cli._resolve_dimensions_arg(args, require=True)
    assert cli._resolve_dimensions_arg(args, require=False) is None


def test_general_prereg_requires_exactly_one_source():
    with pytest.raises(ValueError, match="preregistration"):
        cli._resolve_general_prereg(SimpleNamespace(osf_url=None, preregistration=None))
    with pytest.raises(ValueError, match="only one"):
        cli._resolve_general_prereg(
            SimpleNamespace(osf_url="https://osf.io/abc12", preregistration="p.pdf")
        )
    path, ext = cli._resolve_general_prereg(
        SimpleNamespace(osf_url=None, preregistration="/tmp/p.docx")
    )
    assert path == "/tmp/p.docx" and ext == ".docx"


def test_general_parser_default_matches_app_and_accepts_dimension_set():
    args = cli.build_parser().parse_args(
        ["general", "--paper", "a.pdf", "--dimension-set", "psychology"]
    )
    assert args.parser_choice == "pymupdf"  # harmonised with web/API default
    assert args.dimension_set == "psychology"
