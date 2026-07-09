"""Command-line entrypoint to run RegCheck comparisons without the frontend.

Usage examples:
  python -m backend.cli general --preregistration /path/prereg.pdf --paper /path/paper.pdf --dimensions-csv dims.csv --client openai
  python -m backend.cli clinical --registration-id NCT0000 --paper /path/paper.pdf --dimensions-csv dims.csv --client openai
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from backend.services.llm import DEFAULT_GPUSTACK_BASE_URL

from backend.services.comparisons import (
    animals_trial_comparison,
    clinical_trial_comparison,
    general_preregistration_comparison,
)
from backend.services.dimensions import discipline_keys, get_discipline_dimensions

logger = logging.getLogger("backend.cli")

# The web app's built-in discipline presets, surfaced to the CLI via --dimension-set
# so the same named dimension sets are available everywhere.
DISCIPLINE_KEYS = discipline_keys()


def _load_dimensions_from_csv(csv_path: Path) -> list[dict[str, str]]:
    """Read dimensions from a CSV with 'dimension' and optional 'definition' columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dimensions CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(
                "Dimensions CSV is missing headers. Expected at least a 'dimension' column."
            )
        dimensions: list[dict[str, str]] = []
        for row in reader:
            name = (
                (row.get("dimension") or row.get("name") or row.get("Dimension") or row.get("Name") or "")
                .strip()
            )
            definition = (row.get("definition") or row.get("Definition") or "").strip()
            if name:
                dimensions.append({"dimension": name, "definition": definition})
        if not dimensions:
            raise ValueError(
                "No dimensions found in CSV. Ensure there is a 'dimension' column with values."
            )
        return dimensions


def _normalized_suffix(path: str) -> str:
    suffix = Path(path).suffix
    if not suffix:
        raise ValueError(f"Cannot determine file extension for {path}.")
    return suffix.lower()


def _resolve_dimensions_arg(args, *, require: bool) -> list[dict[str, str]] | None:
    """Resolve dimensions from --dimension-set (a built-in discipline preset, same
    as the web app) or --dimensions-csv. At most one may be given."""
    csv_path = getattr(args, "dimensions_csv", None)
    set_key = getattr(args, "dimension_set", None)
    if csv_path and set_key:
        raise ValueError("Use only one of --dimensions-csv or --dimension-set.")
    if set_key:
        dims = get_discipline_dimensions(set_key)
        if not dims:
            raise ValueError(f"Unknown dimension set: {set_key}")
        logger.info("Using '%s' discipline preset (%d dimensions)", set_key, len(dims))
        return dims
    if csv_path:
        logger.info("Loading dimensions from CSV: %s", csv_path)
        return _load_dimensions_from_csv(Path(csv_path))
    if require:
        raise ValueError("Provide dimensions via --dimensions-csv or --dimension-set.")
    return None


def _resolve_general_prereg(args) -> tuple[str, str]:
    """Resolve the preregistration source: an OSF link (--osf-url, resolved here the
    same way the worker does) or a local file (--preregistration). Exactly one."""
    osf_url = getattr(args, "osf_url", None)
    prereg = getattr(args, "preregistration", None)
    if osf_url and prereg:
        raise ValueError("Use only one of --preregistration or --osf-url.")
    if osf_url:
        from backend.services import osf

        dest = tempfile.mkdtemp(prefix="regcheck-osf-")
        logger.info("Resolving OSF preregistration: %s", osf_url)
        path, ext = osf.fetch_osf_preregistration(osf_url, dest_dir=dest)
        return path, ext
    if not prereg:
        raise ValueError("Provide a preregistration via --preregistration or --osf-url.")
    return prereg, _normalized_suffix(prereg)


def _maybe_write_html_report(args, payload: dict, evidence_out: dict | None) -> None:
    """Write a self-contained, browsable HTML report when --report-html is set."""
    report_path = getattr(args, "report_html", None)
    if not report_path:
        return
    from backend.services.report_html import write_report_html

    evidence = evidence_out or {}
    title = getattr(args, "report_title", None) or f"RegCheck · {Path(args.paper).stem}"
    items = payload.get("items", []) if isinstance(payload, dict) else []
    meta = f"Model: {args.client} · parser: {args.parser_choice} · {len(items)} dimensions"
    write_report_html(
        report_path,
        title=title,
        items=items,
        manifest=evidence.get("manifest"),
        render_data=evidence.get("render_data"),
        meta=meta,
    )
    logger.info("Wrote browsable HTML report to %s", report_path)


async def _run_general(args) -> dict:
    dimensions = _resolve_dimensions_arg(args, require=True)
    prereg_path, prereg_ext = _resolve_general_prereg(args)
    logger.info("Running general preregistration comparison (dimensions=%d)", len(dimensions))
    evidence_out: dict | None = {} if getattr(args, "report_html", None) else None
    result = await general_preregistration_comparison(
        prereg_path,
        prereg_ext,
        args.paper,
        _normalized_suffix(args.paper),
        args.client,
        args.parser_choice,
        selected_dimensions=dimensions,
        append_previous_output=args.append_previous_output,
        reasoning_effort=args.reasoning_effort,
        multiple_experiments=args.multiple_experiments,
        experiment_number=args.experiment_number,
        experiment_text=args.experiment_text,
        evidence_out=evidence_out,
        num_voters=_resolve_num_voters(args),
        isolation_cache_dir=_resolve_isolation_cache_dir(args),
        isolation_passes=args.isolation_passes,
    )
    logger.info("Completed general comparison.")
    payload = result.model_dump()
    _maybe_write_html_report(args, payload, evidence_out)
    return payload


async def _run_clinical(args) -> dict:
    dimensions = _resolve_dimensions_arg(args, require=False)
    logger.info(
        "Running clinical trial comparison for %s (dimensions=%s)",
        args.registration_id,
        "custom" if dimensions else "default",
    )
    result = await clinical_trial_comparison(
        args.registration_id,
        args.paper,
        _normalized_suffix(args.paper),
        args.client,
        parser_choice=args.parser_choice,
        selected_dimensions=dimensions,
        append_previous_output=args.append_previous_output,
        reasoning_effort=args.reasoning_effort,
        num_voters=_resolve_num_voters(args),
    )
    logger.info("Completed clinical comparison.")
    return result.model_dump()


async def _run_animals(args) -> dict:
    if not args.registration_csv:
        raise ValueError(
            "Animal trial comparisons currently require --registration-csv until API support is available."
        )
    dimensions = _resolve_dimensions_arg(args, require=False)
    logger.info(
        "Running animals (PCT) comparison for %s using CSV %s (dimensions=%s)",
        args.registration_id,
        args.registration_csv,
        "custom" if dimensions else "default",
    )
    result = await animals_trial_comparison(
        args.registration_id,
        args.paper,
        _normalized_suffix(args.paper),
        args.client,
        registration_csv_path=args.registration_csv,
        parser_choice=args.parser_choice,
        selected_dimensions=dimensions,
        append_previous_output=args.append_previous_output,
        reasoning_effort=args.reasoning_effort,
        num_voters=_resolve_num_voters(args),
    )
    logger.info("Completed animals comparison.")
    return result.model_dump()


def _write_output(payload: dict, output_path: str | None, output_format: str) -> None:
    """Write results to stdout or file in the requested format."""
    if output_format == "json":
        text = json.dumps(payload, indent=2)
        if output_path:
            Path(output_path).write_text(text, encoding="utf-8")
        else:
            print(text)
        return

    # CSV output (default)
    items = payload.get("items", []) if isinstance(payload, dict) else []
    fieldnames = [
        "dimension",
        "paper_content_quotes",
        "paper_content_summary",
        "registration_content_quotes",
        "registration_content_summary",
        "deviation_judgement",
        "deviation_information",
        "unlocated_in_paper",
        "unlocated_in_registration",
    ]
    rows: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row: dict[str, str] = {}
        for key in fieldnames:
            value = item.get(key, "")
            row[key] = "" if value is None else str(value)
        rows.append(row)

    if output_path:
        path = Path(output_path)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_isolation_cache_dir(args) -> str | None:
    """Resolve where to cache multi-study isolation extractions (general flow only).

    Default-on for the CLI so repeated local runs of the same paper reuse the identical
    isolated text (making the pipeline reproducible run-to-run); ``--no-isolation-cache``
    disables it. This is CLI-only — the API/worker path never passes a cache dir, so it
    always re-extracts fresh per job (no cross-job/cross-user reuse).
    """
    if getattr(args, "no_isolation_cache", False):
        return None
    explicit = getattr(args, "isolation_cache_dir", None)
    if explicit:
        return explicit
    return str(Path.home() / ".cache" / "regcheck" / "isolation")


def _resolve_num_voters(args) -> int:
    """Resolve the per-dimension vote count from the judgement-method flags.

    'single-judge' (or unset) → 1 judgement per dimension (unchanged behaviour).
    'consensus-vote' → --number-of-voters independent judgements per dimension.
    """
    method = getattr(args, "judgement_method", "single-judge")
    if method != "consensus-vote":
        return 1
    n = int(getattr(args, "number_of_voters", 5) or 5)
    if n < 1:
        raise ValueError("--number-of-voters must be a positive integer.")
    return n


def _add_judgement_args(p: argparse.ArgumentParser) -> None:
    """Optional within-study output normalisation, shared by every subcommand."""
    p.add_argument(
        "--judgement-method",
        default="single-judge",
        choices=["single-judge", "consensus-vote"],
        help=(
            "How each dimension's verdict is decided. 'single-judge' (default) is one "
            "model judgement per dimension. 'consensus-vote' judges each dimension "
            "--number-of-voters times on the SAME retrieved evidence and reports the "
            "plurality verdict (recall-biased tiebreak), with a consensus note appended "
            "to the rationale. The report format is unchanged."
        ),
    )
    p.add_argument(
        "--number-of-voters",
        type=int,
        default=5,
        help=(
            "Number of independent judgements per dimension when "
            "--judgement-method consensus-vote (default 5; ignored for single-judge)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RegCheck backend comparisons from the command line."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    general = subparsers.add_parser(
        "general", help="Compare a preregistration document to a paper."
    )
    general.add_argument(
        "--preregistration",
        help="Path to the preregistration file (.pdf/.docx/.txt/.html). Or use --osf-url.",
    )
    general.add_argument(
        "--osf-url",
        help="OSF link to a registration or file to use as the preregistration (alternative to --preregistration; mirrors the web app's OSF source).",
    )
    general.add_argument(
        "--paper",
        required=True,
        help="Path to the published paper file (.pdf/.docx/.txt/.html).",
    )
    general.add_argument(
        "--dimensions-csv",
        help="CSV file with columns 'dimension' and optional 'definition'. Or use --dimension-set.",
    )
    general.add_argument(
        "--dimension-set",
        choices=DISCIPLINE_KEYS,
        help=(
            "Use one of the web app's built-in discipline presets instead of a CSV "
            f"({', '.join(DISCIPLINE_KEYS)})."
        ),
    )
    general.add_argument(
        "--client",
        default="openai",
        choices=["openai", "deepseek", "qwen", "claude", "gpustack"],
        help="LLM provider to use ('gpustack' = Uni Bern GPUStack; requires the Bern network).",
    )
    general.add_argument(
        "--embedding-model",
        help=(
            "Embeddings model for retrieval (default: text-embedding-3-large; "
            "use qwen3-embedding-0.6b with --client gpustack)."
        ),
    )
    general.add_argument(
        "--parser-choice",
        default="pymupdf",
        choices=["grobid", "dpt2", "pymupdf", "external"],
        help="PDF parser to extract paper text (default pymupdf, matching the web app/API).",
    )
    general.add_argument(
        "--append-previous-output",
        action="store_true",
        help="Append previous dimension responses into later prompts.",
    )
    general.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning setting for OpenAI models (ignored by other providers).",
    )
    general.add_argument(
        "--multiple-experiments",
        action="store_true",
        help="Indicate the paper has multiple experiments and you want to isolate one experiment's text.",
    )
    general.add_argument(
        "--experiment-number",
        help="Identifier for the relevant experiment (e.g., 2, 2B).",
    )
    general.add_argument(
        "--experiment-text",
        help="Optional freeform note about the relevant experiment to include in the extraction prompt.",
    )
    general.add_argument(
        "--isolation-cache-dir",
        help=(
            "Directory for caching multi-study isolation extractions so repeated local runs "
            "of the same paper reuse identical isolated text — makes the pipeline reproducible "
            "run-to-run (default: ~/.cache/regcheck/isolation). Only used with --multiple-experiments."
        ),
    )
    general.add_argument(
        "--no-isolation-cache",
        action="store_true",
        help="Disable the multi-study isolation cache (always re-extract the study text).",
    )
    general.add_argument(
        "--isolation-passes",
        type=int,
        default=1,
        help=(
            "Run the multi-study isolation N times in parallel and UNION the extracted "
            "spans (default 1). The single pass varies ~20%% on deviation-relevant content "
            "(exclusions, analyses, sample tables); N>=3 recovers a span unless every pass "
            "drops it. Only used with --multiple-experiments."
        ),
    )
    general.add_argument(
        "--output",
        help="Optional path to write JSON results. Defaults to stdout.",
    )
    general.add_argument(
        "--output-format",
        default="csv",
        choices=["csv", "json"],
        help="Output format (csv or json). Defaults to csv.",
    )
    general.add_argument(
        "--report-html",
        help=(
            "Also write a self-contained, browsable HTML report (Overview + Evidence "
            "panels with in-document quote tracing) to this path; open it in any browser."
        ),
    )
    general.add_argument(
        "--report-title",
        help="Title shown in the --report-html report (defaults to 'RegCheck · <paper filename>').",
    )

    _add_judgement_args(general)

    clinical = subparsers.add_parser(
        "clinical", help="Compare a clinical trial registration (by ID) to a paper."
    )
    clinical.add_argument(
        "--registration-id",
        required=True,
        help="Clinical trial registration identifier (e.g., NCT number).",
    )
    clinical.add_argument(
        "--paper",
        required=True,
        help="Path to the published paper file (.pdf or .docx).",
    )
    clinical.add_argument(
        "--dimensions-csv",
        help="Optional CSV with 'dimension' and 'definition' columns to override defaults.",
    )
    clinical.add_argument(
        "--dimension-set",
        choices=DISCIPLINE_KEYS,
        help=f"Use a built-in discipline preset instead of defaults ({', '.join(DISCIPLINE_KEYS)}).",
    )
    clinical.add_argument(
        "--client",
        default="openai",
        choices=["openai", "deepseek", "qwen", "claude", "gpustack"],
        help="LLM provider to use ('gpustack' = Uni Bern GPUStack; requires the Bern network).",
    )
    clinical.add_argument(
        "--embedding-model",
        help=(
            "Embeddings model for retrieval (default: text-embedding-3-large; "
            "use qwen3-embedding-0.6b with --client gpustack)."
        ),
    )
    clinical.add_argument(
        "--parser-choice",
        default="pymupdf",
        choices=["grobid", "dpt2", "pymupdf", "external"],
        help="PDF parser to extract paper text (default pymupdf, matching the web app/API).",
    )
    clinical.add_argument(
        "--append-previous-output",
        action="store_true",
        help="Append previous dimension responses into later prompts.",
    )
    clinical.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning setting for OpenAI models (ignored by other providers).",
    )
    clinical.add_argument(
        "--output",
        help="Optional path to write JSON results. Defaults to stdout.",
    )
    clinical.add_argument(
        "--output-format",
        default="csv",
        choices=["csv", "json"],
        help="Output format (csv or json). Defaults to csv.",
    )

    _add_judgement_args(clinical)

    animals = subparsers.add_parser(
        "animals", help="Compare a preclinical (PCT) registration to a paper."
    )
    animals.add_argument(
        "--registration-id",
        required=True,
        help="PreclinicalTrials.eu identifier (e.g., PCTE0000405).",
    )
    animals.add_argument(
        "--registration-csv",
        help="CSV export containing a pct_id column. Required until API integration is available.",
    )
    animals.add_argument(
        "--paper",
        required=True,
        help="Path to the published paper file (.pdf or .docx).",
    )
    animals.add_argument(
        "--dimensions-csv",
        help="Optional CSV with 'dimension' and 'definition' columns to override defaults.",
    )
    animals.add_argument(
        "--dimension-set",
        choices=DISCIPLINE_KEYS,
        help=f"Use a built-in discipline preset instead of defaults ({', '.join(DISCIPLINE_KEYS)}).",
    )
    animals.add_argument(
        "--client",
        default="openai",
        choices=["openai", "deepseek", "qwen", "claude", "gpustack"],
        help="LLM provider to use ('gpustack' = Uni Bern GPUStack; requires the Bern network).",
    )
    animals.add_argument(
        "--embedding-model",
        help=(
            "Embeddings model for retrieval (default: text-embedding-3-large; "
            "use qwen3-embedding-0.6b with --client gpustack)."
        ),
    )
    animals.add_argument(
        "--parser-choice",
        default="pymupdf",
        choices=["grobid", "dpt2", "pymupdf", "external"],
        help="PDF parser to extract paper text (default pymupdf, matching the web app/API).",
    )
    animals.add_argument(
        "--append-previous-output",
        action="store_true",
        help="Append previous dimension responses into later prompts.",
    )
    animals.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning setting for OpenAI models (ignored by other providers).",
    )
    animals.add_argument(
        "--output",
        help="Optional path to write JSON results. Defaults to stdout.",
    )
    animals.add_argument(
        "--output-format",
        default="csv",
        choices=["csv", "json"],
        help="Output format (csv or json). Defaults to csv.",
    )
    _add_judgement_args(animals)

    return parser


def _apply_runtime_env(args) -> None:
    """Translate CLI provider/embedding selections into the env vars the service
    layer reads, so a single ``--client gpustack`` run keeps BOTH the LLM and the
    retrieval embeddings on GPUStack without extra setup.

    Precedence is preserved: anything already exported (e.g. in ``.env``) wins over
    these conveniences.
    """
    client = getattr(args, "client", None)

    # Embeddings model: an explicit --embedding-model wins; otherwise, when running
    # against GPUStack, default to its embedding model rather than an OpenAI-only one.
    embedding_model = getattr(args, "embedding_model", None)
    if not embedding_model and client == "gpustack":
        embedding_model = "qwen3-embedding-0.6b"
    if embedding_model:
        os.environ["EMBEDDINGS_MODEL"] = embedding_model

    # Point embeddings at GPUStack too when using it for chat, unless the operator
    # has already aimed them somewhere explicitly.
    if client == "gpustack":
        base_url = (os.environ.get("GPUSTACK_BASE_URL") or DEFAULT_GPUSTACK_BASE_URL).strip()
        os.environ.setdefault("EMBEDDINGS_BASE_URL", base_url)
        gpustack_key = (os.environ.get("GPUSTACK_API_KEY") or "").strip()
        if gpustack_key:
            os.environ.setdefault("EMBEDDINGS_API_KEY", gpustack_key)
        if getattr(args, "parser_choice", None) in {"grobid", "dpt2", "external"}:
            print(
                f"Note: --parser-choice {args.parser_choice} sends the PDF to a remote parser. "
                "For a fully-local GPUStack pipeline use --parser-choice pymupdf "
                "(or a self-hosted GROBID via GROBID_URL).",
                file=sys.stderr,
            )


def main(argv: Iterable[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_runtime_env(args)
    try:
        if args.command == "general":
            payload = asyncio.run(_run_general(args))
        elif args.command == "clinical":
            payload = asyncio.run(_run_clinical(args))
        elif args.command == "animals":
            payload = asyncio.run(_run_animals(args))
        else:
            parser.error("Unknown command")
            return
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    _write_output(payload, getattr(args, "output", None), getattr(args, "output_format", "csv"))


if __name__ == "__main__":
    main()
