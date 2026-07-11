# Staging soak checklist (pre-v1.0.0 deploy)

Run this on **regcheck-staging** before promoting a release that touches the
worker, auth, or deletion paths. Each drill targets a specific hardening
mechanism; the "expect" line is the pass criterion.

## 0. Config-var gate (before the first deploy of this release)

The TLS and parser defaults changed in v1.0.0 — set these **before** deploying
or Redis/Postgres connections fail at boot:

```
heroku config:set REDIS_TLS_INSECURE=1 DATABASE_SSL=relaxed -a <app>
# Only if scanned-PDF OCR should keep working on the hosted app:
heroku config:set SCANNED_PDF_FALLBACK=dpt2 -a <app>
```

Also confirm `DATABASE_URL` is present (production now refuses to boot onto
the SQLite fallback) and `SESSION_SECRET` is set.

## 1. Readiness

- `curl https://<staging>/ready` → HTTP 200, `redis`/`database`/`worker` all
  `ok`. Scale the worker to 0 (`heroku ps:scale worker=0`), wait ~60s, re-curl.
  **Expect:** 503 with `worker: error: no live worker heartbeat`; `/health`
  still 200. Scale back to 1 and confirm `/ready` recovers.

## 2. Overlapping deploys (worker lease safety)

- Start a long comparison (big PDF, many dimensions). While it runs,
  `heroku ps:restart worker` (simulates a deploy overlap: old + new worker
  briefly coexist).
  **Expect:** the job is NOT stolen mid-run by the new worker (per-worker
  processing lists are only reclaimed after a heartbeat lapse, ~30s). The job
  either completes on the old dyno or is recovered once — never runs twice
  concurrently. Check logs for `_recover_orphaned_processing` claims.

## 3. Forced worker crash (orphan recovery)

- Start a comparison, then `heroku ps:kill worker.1` mid-run.
  **Expect:** after the heartbeat TTL lapses, the restarted worker reclaims
  the orphaned job from `comparison:processing:<dead-id>` and the report
  completes (or fails cleanly) — it does not sit PENDING forever.

## 4. Delete during run (tombstones)

- Start a comparison, and while IN_PROGRESS delete the report (UI or API).
  **Expect:** the run stops mattering — on completion the worker sees the
  `deleted:{task_id}` tombstone and removes everything (Redis hash, artifacts,
  S3 uploads, DB row). The result page 404s/denies; nothing reappears.

## 5. Duplicate regenerate (idempotency)

- On a completed report, fire two regenerates back-to-back (double-click, or
  two curl calls within a second).
  **Expect:** one 202/redirect and one 409 (`regen:lock`); only ONE new job is
  queued. A regenerate on a PENDING/IN_PROGRESS report → 409 no-op. A
  regenerate on a deleted report → 409 tombstone refusal.

## 6. Cross-account spot checks (Phase-1 auth)

- Browser A signed in as owner, browser B anonymous with a stale session:
  B must not be able to rename/delete/regenerate A's report, and a private
  report link opened by B redirects to login (fails closed even if the DB row
  is missing).

## 7. Parser + API surface

- Upload a scanned (image-only) PDF with the default parser.
  **Expect:** with `SCANNED_PDF_FALLBACK=dpt2` set, the report completes and
  "View settings" shows the DPT-2 fallback; without it, a clear error tells
  the user to pick the OCR parser — the file is never sent externally.
- `https://<staging>/docs` renders (self-hosted Swagger under CSP — check the
  browser console for CSP violations), `/openapi.json` reports version 1.0.0,
  and `POST /api/v1/compare` with a valid key returns 202.

## 8. Watch for a day

- Keep `/ready` on the uptime monitor. Skim `heroku logs` for: repeated
  orphan-recovery claims (would indicate heartbeat flapping), 500s from the
  global handler, and CSP violation reports if `CSP_REPORT_ONLY=1` is set.
