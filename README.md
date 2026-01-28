# RegCheck

AI-assisted comparison tool for preregistrations/clinical trial registrations/preclinical (animals) registrations and published papers. FastAPI serves the web UI and HTTP API; a CLI entrypoint enables backend-only runs with CSV-defined dimensions. Redis is used for task state when running via the web app.

Status: beta (under active development).

## Contents
- `app.py` / `backend/`: FastAPI app, routes, services (comparisons, embeddings, parsing).
- `templates/` + `static/`: Frontend pages and assets.
- `uploads/`: Runtime uploads directory (created at runtime; ignored by git).
- `backend/worker.py`: Background worker that pulls comparison jobs from Redis.
- `nltk_data/`: Not committed; downloaded locally via NLTK.
- `test_materials/`: CSV example inputs (PDF/DOCX samples intentionally excluded).
- `backend/cli.py`: Headless CLI for running comparisons without the UI.

## Prerequisites
- Python 3.12+ (virtualenv recommended)
- Redis (local or remote) for the web flow; CLI can run without Redis.
- API keys as needed: `OPENAI_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY` (set whichever provider you use).
- Optional: GROBID/DPT2 settings if using those parsers.

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

# Optional parser overrides
GROBID_URL=https://kermitt2-grobid.hf.space/api/processFulltextDocument
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
If `--output` is omitted, results print to stdout. `--output-format` accepts `csv` (default) or `json`. `--append-previous-output` passes prior dimension responses into later prompts.

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
- Supported LLM providers: `openai`, `groq`, `deepseek` (set corresponding API key). `reasoning_effort` applies only to OpenAI models.
- PDF parser choice: `grobid` or `dpt2`; `.docx` files are supported via `python-docx` reader.

## License
GNU Affero General Public License v3.0 (see `LICENSE`).
