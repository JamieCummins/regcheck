"""Run the frozen verdict-doctrine probe suite against the live pipeline.

Each probe is a minimal synthetic registration/paper pair that isolates ONE
doctrine rule (see manifest.json); a correct judge returns the expected verdict,
so any diff flags a regression in prompt or pipeline behaviour.

Usage:
    .venv/bin/python benchmarks/run_probes.py [--client openai] [--out PATH]

Requires the provider key in .env. Results are written as JSON; commit the
snapshot for a release under benchmarks/results/ (e.g. v1.0.0-probes.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from backend.services.comparisons import _normalize_verdict, run_comparison  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", default="openai")
    parser.add_argument("--out", default=None, help="Write results JSON here (default: stdout only)")
    parser.add_argument("--only", default=None, help="Run a single probe by name")
    args = parser.parse_args()

    manifest = json.loads((BENCH_DIR / "manifest.json").read_text(encoding="utf-8"))
    results = []
    failures = 0
    for probe in manifest["probes"]:
        name = probe["name"]
        if args.only and name != args.only:
            continue
        probe_dir = BENCH_DIR / "probes" / name
        registration = (probe_dir / "registration.txt").read_text(encoding="utf-8")
        paper = (probe_dir / "paper.txt").read_text(encoding="utf-8")
        outcome = run_comparison(
            registration,
            paper,
            args.client,
            probe["dimension"],
            dimension_definition=probe["dimension_definition"],
        )
        item = outcome.items[0]
        got = _normalize_verdict(item.deviation_judgement)
        ok = got == probe["expected"]
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL':4}  {name:34} expected={probe['expected']:7} got={got:7} ({probe['rule']})")
        results.append(
            {
                "name": name,
                "rule": probe["rule"],
                "dimension": probe["dimension"],
                "expected": probe["expected"],
                "got": got,
                "pass": ok,
                "raw_judgement": item.deviation_judgement,
                "rationale": item.deviation_information,
            }
        )

    summary = {
        "client": args.client,
        "total": len(results),
        "passed": sum(1 for r in results if r["pass"]),
        "results": results,
    }
    print(f"\n{summary['passed']}/{summary['total']} probes passed")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
