#!/usr/bin/env python3
"""
Batch runner for clinical trial comparisons.

Reads a CSV containing paper paths and ClinicalTrials.gov IDs, then runs
`python -m backend.cli clinical ...` for each row, writing separate outputs.

Example CSV (headers are configurable):
registration_id,paper
NCT00000000,/path/to/paper1.pdf
NCT00000001,relative/path/to/paper2.docx
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable


_PRINT_LOCK = threading.Lock()


def _locked_print(*args, **kwargs) -> None:
    # Avoid interleaved output when running multiple subprocesses concurrently.
    with _PRINT_LOCK:
        print(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch clinical comparisons from a CSV.")
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV with at least two columns: ClinicalTrials.gov ID and paper path.",
    )
    parser.add_argument(
        "--paper-column",
        default="paper",
        help="CSV column name containing the paper path (.pdf or .docx). Default: paper.",
    )
    parser.add_argument(
        "--id-column",
        default="registration_id",
        help="CSV column name containing the ClinicalTrials.gov identifier. Default: registration_id.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-row outputs will be written.",
    )
    parser.add_argument(
        "--dimensions-csv",
        help="Optional dimensions CSV to override defaults for every run.",
    )
    parser.add_argument(
        "--client",
        default="openai",
        choices=["openai", "deepseek", "groq"],
        help="LLM provider to use.",
    )
    parser.add_argument(
        "--parser-choice",
        default="grobid",
        choices=["grobid", "dpt2"],
        help="PDF parser to extract paper text.",
    )
    parser.add_argument(
        "--append-previous-output",
        action="store_true",
        help="Append previous dimension responses into later prompts.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning setting for OpenAI models (ignored by other providers).",
    )
    parser.add_argument(
        "--output-format",
        default="csv",
        choices=["csv", "json"],
        help="Output format for each run.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=10,
        help="Number of concurrent papers to process at once. Default: 10.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def _load_rows(
    csv_path: Path, paper_column: str, id_column: str
) -> Iterable[tuple[int, str, Path]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV is missing headers.")
        if paper_column not in reader.fieldnames:
            raise ValueError(f"CSV missing required column: {paper_column}")
        if id_column not in reader.fieldnames:
            raise ValueError(f"CSV missing required column: {id_column}")

        for idx, row in enumerate(reader, start=1):
            trial_id = (row.get(id_column) or "").strip()
            paper_raw = (row.get(paper_column) or "").strip()
            if not trial_id or not paper_raw:
                print(f"[skip] Row {idx}: missing id or paper path", file=sys.stderr)
                continue
            paper_path_raw = Path(paper_raw).expanduser()
            candidates: list[Path] = []
            if paper_path_raw.is_absolute():
                candidates.append(paper_path_raw)
            else:
                # Try relative to the CSV first (original behavior), then CWD (handy if paths are written from project root).
                candidates.append((csv_path.parent / paper_path_raw).resolve())
                candidates.append((Path.cwd() / paper_path_raw).resolve())

            paper_path = candidates[0]
            for candidate in candidates:
                if candidate.exists():
                    paper_path = candidate
                    break
            yield idx, trial_id, paper_path


def _run_one_row(
    *,
    idx: int,
    trial_id: str,
    paper_path: Path,
    output_dir: Path,
    output_format: str,
    client: str,
    parser_choice: str,
    reasoning_effort: str,
    append_previous_output: bool,
    dimensions_csv: Path | None,
) -> tuple[int, Path, int]:
    """Run one clinical comparison row. Returns (row_index, output_path, exit_code)."""
    if not paper_path.exists():
        _locked_print(
            f"[error] Row {idx}: paper not found -> {paper_path}",
            file=sys.stderr,
            flush=True,
        )
        return idx, output_dir / f"{idx:03d}_{trial_id}.{output_format}", 1

    # Include paper filename in output for easier tracing of results
    output_name = f"{idx:03d}_{paper_path.stem}__{trial_id}.{output_format}"
    output_path = output_dir / output_name

    cmd = [
        sys.executable,
        "-m",
        "backend.cli",
        "clinical",
        "--registration-id",
        trial_id,
        "--paper",
        str(paper_path),
        "--client",
        client,
        "--parser-choice",
        parser_choice,
        "--reasoning-effort",
        reasoning_effort,
        "--output-format",
        output_format,
        "--output",
        str(output_path),
    ]
    if append_previous_output:
        cmd.append("--append-previous-output")
    if dimensions_csv:
        cmd.extend(["--dimensions-csv", str(dimensions_csv)])

    _locked_print(f"[run] Row {idx}: {trial_id} -> {output_path}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        _locked_print(
            f"[error] Row {idx}: command failed with exit code {exc.returncode}",
            file=sys.stderr,
            flush=True,
        )
        return idx, output_path, exc.returncode
    return idx, output_path, 0


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dimensions_csv = None
    if args.dimensions_csv:
        dimensions_csv = Path(args.dimensions_csv).expanduser().resolve()
        if not dimensions_csv.exists():
            raise SystemExit(f"Dimensions CSV not found: {dimensions_csv}")

    rows = list(_load_rows(csv_path, args.paper_column, args.id_column))
    if not rows:
        raise SystemExit("No valid rows found in CSV.")

    if args.dry_run:
        for idx, trial_id, paper_path in rows:
            # Match the actual output naming so dry-runs show the true destination.
            output_name = f"{idx:03d}_{paper_path.stem}__{trial_id}.{args.output_format}"
            output_path = output_dir / output_name
            cmd = [
                sys.executable,
                "-m",
                "backend.cli",
                "clinical",
                "--registration-id",
                trial_id,
                "--paper",
                str(paper_path),
                "--client",
                args.client,
                "--parser-choice",
                args.parser_choice,
                "--reasoning-effort",
                args.reasoning_effort,
                "--output-format",
                args.output_format,
                "--output",
                str(output_path),
            ]
            if args.append_previous_output:
                cmd.append("--append-previous-output")
            if dimensions_csv:
                cmd.extend(["--dimensions-csv", str(dimensions_csv)])
            _locked_print(f"[dry-run] Row {idx}: {trial_id} -> {output_path}")
            _locked_print(" ".join(cmd))
        _locked_print("Done.")
        return

    jobs = max(1, int(args.jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                _run_one_row,
                idx=idx,
                trial_id=trial_id,
                paper_path=paper_path,
                output_dir=output_dir,
                output_format=args.output_format,
                client=args.client,
                parser_choice=args.parser_choice,
                reasoning_effort=args.reasoning_effort,
                append_previous_output=args.append_previous_output,
                dimensions_csv=dimensions_csv,
            )
            for idx, trial_id, paper_path in rows
        ]
        for future in concurrent.futures.as_completed(futures):
            # Exceptions are already printed per-row where possible, but still surface unexpected ones.
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - best-effort logging for batch runs
                _locked_print(f"[error] Unexpected failure: {exc}", file=sys.stderr, flush=True)

    _locked_print("Done.")


if __name__ == "__main__":
    main()
