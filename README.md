# RegCheck

AI-assisted comparison tool for preregistrations/clinical trial registrations/preclinical (animals) registrations and published papers. FastAPI serves the web UI and HTTP API; a CLI entrypoint enables backend-only runs with CSV-defined dimensions. Redis is used for task state when running via the web app.

Status: beta (under active development).

## About this fork

This fork extends upstream RegCheck with three additions:

1. **Local inference via Ollama:** run comparisons entirely offline against a locally hosted LLM, with no OpenAI/Groq/DeepSeek API key required. `ollama` is selectable as a provider in the CLI, the single-comparison web forms, and the batch flow. See [Using Ollama](#using-ollama-local-inference).
2. **Text embedding via Ollama or TF-IDF:** create embeddings with no OpenAI API key required, automatically using Ollama or TF-IDF depending on available resources.
3. **Batch clinical comparison in the web GUI:** upload many paper PDFs at once; the app auto-extracts each NCT ID, fetches the registration from ClinicalTrials.gov, runs every comparison with live progress, and lets you download all results as a single ZIP. See [Batch clinical comparison](#batch-clinical-comparison-web-gui).

> These features were first prototyped in a separate draft repo (`regcheck-0.2.0-beta`), where batch processing was a standalone CLI script (`batch_clinical.py`). In this repo the batch feature is integrated into the web GUI as FastAPI routes with live progress tracking, and Ollama support is wired through the shared comparison service so it works across the CLI and web flows alike.

## Contents
- `app.py` / `backend/`: FastAPI app, routes, services (comparisons, embeddings, parsing).
- `templates/` + `static/`: Frontend pages and assets.
- `uploads/`: Runtime uploads directory (created at runtime; ignored by git).
- `backend/worker.py`: Background worker that pulls comparison jobs from Redis.
- `nltk_data/`: Not committed; downloaded locally via NLTK.
- `test_materials/`: CSV example inputs (PDF/DOCX samples intentionally excluded).
- `backend/cli.py`: Headless CLI for running comparisons without the UI.
- `backend/routes/batch.py` + `templates/batch.html`, `templates/batch_progress.html`: Batch clinical comparison (web GUI).

## Prerequisites
- Python 3.12+ (virtualenv recommended)
- Redis (local or remote) for the web flow; CLI can run without Redis.
- API keys as needed: `OPENAI_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY` (set whichever provider you use). Not required when using Ollama.
- Optional: GROBID/DPT2 settings if using those parsers.
- Optional: [Ollama](https://ollama.com) for fully local inference (no API key needed).

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Download required NLTK data (sentence tokenizer):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```
Copy `.env.example` to `.env` (optional), then set environment variables:
```bash
cp .env.example .env
```
Environment variables:
```
REDIS_URL=redis://localhost:6379/0               # preferred (HEROKU_REDIS_OLIVE_URL also supported)
SESSION_SECRET=your-session-secret
LOG_LEVEL=INFO                                   # optional
OPENAI_API_KEY=...
GROQ_API_KEY=...
DEEPSEEK_API_KEY=...

# Optional model overrides
OPENAI_MODEL=gpt-5
OPENAI_COMPARISON_MODEL=gpt-5
OPENAI_EXPERIMENT_MODEL=gpt-5
OPENAI_EXPERIMENT_REASONING_EFFORT=medium        # low | medium | high
GROQ_MODEL=llama-3.3-70b-versatile
DEEPSEEK_MODEL=deepseek-reasoner

# Ollama (local inference; no API key required)
OLLAMA_MODEL=llama3.2                             # any model available in your Ollama instance
OLLAMA_BASE_URL=http://localhost:11434/v1         # default; the /v1 suffix is required
OLLAMA_EMBEDDING_MODEL=nomic-embed-text-v2-moe    # default; any embedding-capable model in your Ollama instance

# Optional parser overrides
GROBID_URL=https://lfoppiano-grobid.hf.space/api/processFulltextDocument
DPT_API_KEY=...
DPT_URL=https://api.va.eu-west-1.landing.ai/v1/ade/parse
PDF_PARSER_FALLBACKS=dpt2,pymupdf   # ordered fallbacks when the primary parser fails (set blank to disable)

STATIC_DIR=static            # optional override
TEMPLATES_DIR=templates      # optional override
UPLOAD_DIR=uploads           # optional override

# Optional: S3-backed upload storage (recommended for multi-dyno deployments)
S3_BUCKET=your-bucket-name
AWS_REGION=us-east-1
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or other AWS auth) must be set for the runtime.

# Optional: resource caps for scaling
MAX_CONCURRENT_COMPARISON_TASKS=6                # per worker process
MAX_EMBEDDING_SEGMENTS=1200                      # cap segments per document
MAX_UPLOAD_BYTES=20971520                        # max upload size (bytes)
WEB_CONCURRENCY=2                                # gunicorn workers (web dyno)
WEB_TIMEOUT=120                                  # gunicorn timeout (seconds)
TASK_TTL_SECONDS=259200                          # expire task metadata after 3 days
MAX_QUEUE_LENGTH=200                             # max queued+in-flight jobs before returning 503
```
Heroku deployments must set `SESSION_SECRET` (the app will refuse to boot on dynos without it to avoid session resets).

## Using Ollama (local inference)

Ollama runs comparisons entirely on your machine, which requires no API key, and no data leaves the host (unless you use one of Ollama's cloud models). It is exposed as the `ollama` provider everywhere a provider can be chosen: the CLI (`--client ollama`), the web comparison forms, and the batch flow.
The embedding of text can also be done locally with Ollama, which is more reliable than the TF-IDF approach and still works without an API key. Set `OLLAMA_EMBEDDING_MODEL` to a suitable embedding-capable model in your Ollama instance.

**1. Install Ollama:** See [ollama.com/download](https://ollama.com/download) (`brew install ollama` on macOS, `curl -fsSL https://ollama.com/install.sh | sh` on Linux, installer on Windows).

**2. Pull a model:** For the comparisons, choose a model that follows structured JSON instructions well. For the embeddings, choose a specialized model (such as `nomic-embed-text-v2-moe` or `embeddinggemma`).
```bash
ollama pull llama3.2        # default; small and fast
ollama pull gpt-oss         # stronger JSON compliance
ollama pull nomic-embed-text-v2-moe   # specialized model for text embedding with multilingual support
```

**3. Start the server** (macOS/Linux; on Windows the tray app starts it automatically):
```bash
ollama serve   # http://localhost:11434
```

**4. Configure `.env`** (both values default to those shown, so this is only needed to override them):
```
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text-v2-moe
```
> `OLLAMA_BASE_URL` must include the `/v1` suffix — the OpenAI-compatible endpoint lives at `/v1/chat/completions`. Without it, you get 404 errors.

**5. Select Ollama.** In the web UI pick **Ollama (local)** from the provider dropdown; from the CLI pass `--client ollama`.

> Local inference is much slower than cloud providers (roughly 1–5 minutes per dimension, depending on hardware and model size). Smaller models occasionally emit malformed JSON; switch to a larger model if that happens.

## Running the web app
```bash
uvicorn backend.main:create_app --factory --reload
# or: uvicorn app:app --reload
```
Then open http://localhost:8000 for the UI. FastAPI routes:
- `GET /compare` (unified registration-to-paper flow)
- `POST /compare`
- `POST /general_preregistration`
- `POST /clinical_trials`
- `POST /animals_trials` (requires a `pct_id` and CSV upload until API integration is available)
- `GET /task_status/{task_id}`
- `GET /result/{task_id}`
- `GET /batch`, `POST /batch`, `GET /batch/{batch_id}`, `GET /batch_json/{batch_id}`, `GET /batch_download/{batch_id}` (batch clinical comparison — see below)

## Batch clinical comparison (web GUI)

A **Batch Clinical Comparison** page (linked from the navbar) processes many clinical-trial papers in one go. Open `/batch`, upload multiple PDFs, choose a provider and parser, optionally pick dimensions, and submit.

For each uploaded PDF the app:
1. Scans the text for an `NCT########` ID with PyMuPDF; the first match wins. Papers with no NCT ID are skipped.
2. Fetches the matching registration from ClinicalTrials.gov and runs the standard clinical-trial comparison.
3. Tracks per-paper state (`PENDING → RUNNING → SUCCESS / SKIPPED / FAILED`) in Redis.

Papers are processed sequentially in an async background task, so the upload returns immediately and redirects to a live progress page that polls until the batch is `COMPLETE`. When finished, **Download results** streams a single ZIP containing one CSV per successful paper, named `{NCT_ID}_{paper}.csv`.

Routes:
- `GET /batch` — upload form
- `POST /batch` — accept PDFs, queue the batch, redirect to progress
- `GET /batch/{batch_id}` — live progress page
- `GET /batch_json/{batch_id}` — JSON status (polled by the progress page)
- `GET /batch_download/{batch_id}` — ZIP of all successful result CSVs

> Requires Redis (used for batch and per-paper state). Batch currently covers the clinical-trials flow only. Processing runs inside the web process via an asyncio background task — it does not use the separate Redis `worker` dyno.

## CLI: backend-only comparisons
The CLI reads dimensions from a CSV (`dimension,definition` columns). Example file: `test_materials/dimensions_example.csv`.

General preregistration vs paper:
```bash
python -m backend.cli general \
  --preregistration /path/prereg.pdf \
  --paper /path/paper.pdf \
  --dimensions-csv test_materials/dimensions_example.csv \
  --client openai \
  --parser-choice grobid \
  --append-previous-output \
  --reasoning-effort medium \
  --output-format csv \
  --output result.csv
```

Clinical trial (by registration ID) vs paper:
```bash
python -m backend.cli clinical \
  --registration-id NCT0000 \
  --paper /path/paper.pdf \
  --client openai \
  --parser-choice grobid \
  --output-format csv \
  --output result.csv
```
To override default dimensions, add `--dimensions-csv custom_dimensions.csv`.

Animals (PCT) trial vs paper (CSV required until API is available):
```bash
python -m backend.cli animals \
  --registration-id PCTE0000405 \
  --registration-csv /path/preclinical_export.csv \
  --paper /path/paper.pdf \
  --client openai \
  --parser-choice grobid \
  --append-previous-output \
  --reasoning-effort medium \
  --dimensions-csv custom_dimensions.csv \
  --output-format csv \
  --output result.csv
```
`--client` accepts `openai`, `groq`, `deepseek`, or `ollama` (see [Using Ollama](#using-ollama-local-inference)). If `--output` is omitted, results print to stdout. `--output-format` accepts `csv` (default) or `json`. `--append-previous-output` passes prior dimension responses into later prompts.

## Dimensions CSV format
CSV headers: `dimension,definition`. Additional columns are ignored. Blank dimension names are skipped. Definitions are optional but recommended to tighten prompts.

## Testing
```bash
pytest
```

## Notes
- Default comparison concurrency is now 6 per worker process; tune `MAX_CONCURRENT_COMPARISON_TASKS` and dyno sizing based on memory headroom and provider rate limits.
- Web flow uses Redis for progress tracking; the CLI calls comparison services directly and works without Redis.
- On Heroku, use a separate `worker` dyno to process comparisons from the Redis queue; the web dyno enqueues jobs.
- For multi-dyno deployments (web + worker), configure `S3_BUCKET` so workers can fetch uploaded files reliably. When S3 is configured, uploads are deleted from S3 after each job completes.
- Supported LLM providers: `openai`, `groq`, `deepseek`, `ollama`. Set the corresponding API key for cloud providers; Ollama runs locally and needs no key. `reasoning_effort` applies only to OpenAI models.
- PDF parser choice: `grobid` or `dpt2`; `.docx` files are supported via `python-docx` reader.

## License
GNU Affero General Public License v3.0 (see `LICENSE`).
