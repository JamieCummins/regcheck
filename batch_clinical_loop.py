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
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable


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

    for idx, trial_id, paper_path in rows:
        if not paper_path.exists():
            print(f"[error] Row {idx}: paper not found -> {paper_path}", file=sys.stderr)
            continue

        output_name = f"{idx:03d}_{trial_id}.{args.output_format}"
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

        print(f"[run] Row {idx}: {trial_id} -> {output_path}")
        if args.dry_run:
            print(" ".join(cmd))
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[error] Row {idx}: command failed with exit code {exc.returncode}", file=sys.stderr)
            continue

    print("Done.")


if __name__ == "__main__":
    main()
