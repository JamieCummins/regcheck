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


def test_additional_preregistration_flag_is_repeatable():
    args = cli.build_parser().parse_args(
        [
            "general",
            "--preregistration", "p.pdf",
            "--paper", "a.pdf",
            "--dimension-set", "psychology",
            "--additional-preregistration", "plan.docx",
            "--additional-preregistration", "materials.xml",
        ]
    )
    assert args.additional_preregistration == ["plan.docx", "materials.xml"]


def test_resolve_general_prereg_merges_additional_materials(tmp_path):
    main = tmp_path / "prereg.txt"
    main.write_text("Main preregistration body.", encoding="utf-8")
    extra = tmp_path / "analysis_plan.txt"
    extra.write_text("Additional analysis plan.", encoding="utf-8")

    path, ext = cli._resolve_general_prereg(
        SimpleNamespace(
            osf_url=None,
            preregistration=str(main),
            additional_preregistration=[str(extra)],
        )
    )
    assert ext == ".txt"
    combined = open(path, encoding="utf-8").read()
    # Same labelled-separator shape as the web app's multi-file upload.
    assert "===== Document 1: prereg.txt =====" in combined
    assert "===== Document 2: analysis_plan.txt =====" in combined
    assert "Main preregistration body." in combined
    assert "Additional analysis plan." in combined


def test_resolve_general_prereg_missing_additional_material_fails(tmp_path):
    main = tmp_path / "prereg.txt"
    main.write_text("Main.", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="material not found"):
        cli._resolve_general_prereg(
            SimpleNamespace(
                osf_url=None,
                preregistration=str(main),
                additional_preregistration=[str(tmp_path / "nope.pdf")],
            )
        )


def test_batch_row_namespace_splits_additional_preregistration():
    row = {
        "preregistration": "p.pdf",
        "paper": "a.pdf",
        "additional_preregistration": "plan.docx; materials.xml",
    }
    ns = cli._batch_row_namespace(
        SimpleNamespace(
            client="openai",
            parser_choice="pymupdf",
            append_previous_output=False,
            reasoning_effort="medium",
            embedding_model=None,
        ),
        row,
    )
    assert ns.additional_preregistration == ["plan.docx", "materials.xml"]


def test_supplement_flag_is_repeatable_and_documented(capsys):
    args = cli.build_parser().parse_args(
        [
            "general",
            "--preregistration", "p.pdf",
            "--paper", "a.pdf",
            "--dimension-set", "psychology",
            "--supplement", "s1.pdf",
            "--supplement", "s2.docx",
        ]
    )
    assert args.supplement == ["s1.pdf", "s2.docx"]
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["general", "--help"])
    assert "--supplement PATH" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_supplement_paths_are_merged_into_the_paper(tmp_path, monkeypatch):
    import numpy as np

    import backend.services.comparisons as comparisons
    import backend.services.embeddings as emb

    monkeypatch.setattr(
        emb, "openai_embed_segments",
        lambda s, model=None, **k: np.ones((max(1, len(list(s))), 8), dtype=np.float32),
    )
    seen: list[dict] = []

    def fake_runner(p, pa, c, d, **kw):
        seen.append({"paper": pa, **kw})
        return comparisons.ComparisonResult(items=[comparisons.ComparisonItem(
            dimension=d, deviation_judgement="no", deviation_information="ok")])

    monkeypatch.setattr(comparisons, "run_comparison", fake_runner)
    prereg = tmp_path / "p.txt"; prereg.write_text("Pairwise deletion for all analyses.")
    paper = tmp_path / "paper.txt"; paper.write_text("Results. See the Supplementary Materials for models.")
    s1 = tmp_path / "supp_a.txt"; s1.write_text("Table S1: listwise deletion in MPLUS.")
    s2 = tmp_path / "supp_b.txt"; s2.write_text("Figure S2 notes.")
    args = cli.build_parser().parse_args([
        "general", "--preregistration", str(prereg), "--paper", str(paper),
        "--dimension-set", "psychology", "--client", "claude",
        "--supplement", str(s1), "--supplement", str(s2),
    ])
    payload = await cli._run_general(args)
    merged = seen[0]["paper"]
    assert "See the Supplementary Materials" in merged
    assert "===== Supplementary material 1: supp_a.txt =====" in merged
    assert "listwise deletion in MPLUS" in merged
    assert merged.index("supp_a.txt") < merged.index("===== Supplementary material 2: supp_b.txt =====")
    # The paper still references a supplement, but it was supplied: no judge note.
    assert payload["paper_references_supplement"] is True
    assert all(kw["paper_references_supplement"] is False for kw in seen)


def test_batch_row_namespace_splits_supplements():
    ns = cli._batch_row_namespace(
        SimpleNamespace(
            client="openai", parser_choice="pymupdf", append_previous_output=False,
            reasoning_effort="medium", embedding_model=None,
        ),
        {"preregistration": "p.pdf", "paper": "a.pdf", "supplement": "s1.pdf; s2.pdf"},
    )
    assert ns.supplement == ["s1.pdf", "s2.pdf"]
