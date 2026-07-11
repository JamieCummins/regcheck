# RegCheck verdict benchmark

Frozen regression benchmark for the comparison engine's verdict behaviour.
Two layers:

## 1. Synthetic probe suite (`probes/` + `manifest.json`)

Fifteen minimal registration/paper pairs, each isolating exactly one rule of
the verdict doctrine (literal deviations, specification-as-stated bounds,
additions vs elaboration, licensing clauses, disclosure-irrelevance,
substance-over-surface incl. epistemic status, corroboration gaps, mutual
silence, categorical inference). The expected verdict per probe is recorded in
`manifest.json`.

Run (needs the provider key in `.env`; costs a handful of LLM calls):

```
.venv/bin/python benchmarks/run_probes.py --out benchmarks/results/<version>-probes.json
```

**Policy:** run this after ANY change to the master prompt, verdict schema, or
retrieval defaults. A newly failing probe means the change moved doctrine
behaviour — either fix the change or (if the doctrine itself was deliberately
revised) update the probe + manifest in the same commit, with the reasoning in
the commit message. Judgements are stochastic at the margin; re-run a failing
probe once before treating it as real.

## 2. Real-pair snapshot (three ground-truth pairs)

Three real preregistration↔paper comparisons whose source documents live in
`test_materials/` (gitignored — published PDFs stay out of the repo; ask Jamie
for the folder). Recorded verdicts per dimension live in
`results/<version>-pairs.json`.

```
.venv/bin/python -m backend.cli general \
  --preregistration test_materials/preregistration_meijer_etal.docx \
  --paper test_materials/paper_meijer_etal.pdf \
  --dimension-set psychology --output-format json --output /tmp/meijer.json

.venv/bin/python -m backend.cli general \
  --preregistration test_materials/preregistration_nevejans_experiment2.docx \
  --paper test_materials/paper_nevejans.pdf \
  --dimension-set psychology --multiple-experiments --experiment-number 2 \
  --output-format json --output /tmp/nevejans.json

.venv/bin/python -m backend.cli general \
  --preregistration test_materials/registration_cummins_etal_study2.docx \
  --paper test_materials/paper_cummins_etal.pdf \
  --dimension-set psychology --multiple-experiments --experiment-number 2 \
  --output-format json --output /tmp/cummins.json
```

**Policy:** re-run before a release that touches prompts/retrieval and diff the
per-dimension verdicts against the last committed snapshot. Verdict flips must
be individually adjudicated (they may be corrections — record which).

Snapshots record the judging model and settings; comparisons across different
models are not meaningful.
