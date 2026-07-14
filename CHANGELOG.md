# Changelog

Notable changes to RegCheck. The application version lives in
`APP_VERSION` in `backend/main.py` and is what `/openapi.json` reports.


## 1.0.0 — 2026-07

First stable release. Highlights relative to the public beta:

### Comparison engine
- Rewritten verdict semantics: literal, severity-orthogonal deviations;
  specification-as-stated bounds; strict verdict precedence
  (deviation > insufficient evidence > consistent, mutual silence is never
  consistency); registered-element anchoring; CONSORT-based rules for
  unregistered additions; substance-over-surface judging (claim strength and
  epistemic status count as substance).
- Verification loop: elements the judge cannot locate trigger a targeted
  full-document search, then a re-judgement with the recovered excerpts.
- Quotes-as-IDs prompting with server-side quote reconstruction; evidence
  cards now show the excerpts the judge actually cited (falling back to all
  retrieved excerpts).
- Optional consensus voting (independent strands, plurality verdict with a
  recall-biased tiebreak).
- Registered Reports mode (Stage 1 vs Stage 2), including a deterministic
  carried-forward-text check; political-science and preclinical dimension
  presets; a dedicated preclinical comparison context for animal registrations.

### Platform
- Accounts (Google + ORCID), report ownership, naming, public/private
  visibility with allow-lists, and a keyed public API under `/api/v1`.
- Document intake: PDF (local PyMuPDF by default), DOCX, TXT, HTML uploads;
  ClinicalTrials.gov and OSF registration fetch; multi-file combining.
- Report viewer with three-tier quote tracing, offline HTML export, and a
  post-run settings disclosure (including the parser actually used).

### Hardening (v1.0.0 release gate)
- Ownership can no longer be asserted from a stale browser session; report
  visibility fails closed when metadata is incomplete.
- Deletion tombstones, regenerate idempotency, worker leases + heartbeats
  (deploy-overlap safe); `/ready` readiness probe.
- External parsing services are opt-in only; documents are never sent to an
  external OCR service by default, and any fallback parser is disclosed.
- Streamed, size-capped OSF downloads; aggregate upload and dimension caps.
- TLS verification on by default for Redis and Postgres (explicit opt-outs
  for managed providers with self-signed certificates).
- Dependency refresh to a pip-audit-clean set; self-hosted Swagger UI under
  the site CSP; CI (tests, lint, audit, migrations) with a hash-locked
  lockfile.
- Post-submit page hardened against content blockers: moved from
  `/survey/{id}` to `/next-steps/{id}` (old URL 301s; old form posts still
  accepted) and all `survey-*` selectors renamed — annoyance filter lists
  target survey-named URLs and elements, and this page is the only path from
  a submitted wizard to the report. A "View your report" link outside the
  hideable card (plus an auto-redirect when the report is ready but the card
  is hidden) guarantees the page is never a dead end. Survey analytics,
  Redis keys, and the account-offer step are unchanged.
- Wizard resilience against content blockers: a per-step guard detects
  controls force-hidden by extension-injected CSS and restores them with
  inline `!important` styles; if the wizard script itself is blocked (or
  recovery fails), a fallback notice tells the user a browser extension is
  interfering instead of leaving a silently dead page.
- Frame-bust cross-site iframe embeds: requests carrying `Sec-Fetch-Dest:
  iframe` + `Sec-Fetch-Site: cross-site` get a tiny page that navigates the
  top window to the canonical site (`CANONICAL_BASE_URL`, default
  `https://regcheck.app`), preserving path and query. Rescues browsers still
  rendering the cached pre-migration registrar masking page, whose embedded
  frame the CSP otherwise correctly blanks.
- Registration-quality evaluation (single-document completeness assessment)
  moved to its own product, PreCheck, maintained in a separate codebase. The
  Tools menu now links to the PreCheck site (`PRECHECK_URL`, default
  `https://precheck.app`). Existing quality reports remain viewable.
- Live cost tracking: per-run token/cost estimates shown on reports and in the
  CLI; chain-of-thought capture for providers that expose it (CSV exports
  only); `python -m backend.cli batch` for manifest-driven batch runs.
