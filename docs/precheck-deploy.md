# PreCheck deployment (APP_MODE=precheck)

PreCheck is the registration-quality tool served as its own site from THIS
codebase. One repo, two deployments: `APP_MODE` selects the brand and the
route surface; everything else (engine, hardening, viewer) is shared, so
fixes land on both sites.

## What the mode changes

- **Brand**: "PreCheck" wordmark, its own landing page, og/meta tags, page
  titles, and privacy-policy naming (`brand_name` template global).
- **Routes**: the comparison suite 404s (`/compare`, `/clinical_trials`,
  `/general_preregistration`, `/animals_trials`, `/demo`, `/faq`, `/api*`,
  `/docs`, `/openapi.json`, `/coming-soon`, `/jobs` — see
  `_REGCHECK_ONLY_PREFIXES` in backend/main.py). Served surface:
  landing, `/evaluate_registration`, report viewer/status, accounts
  (login/profile/dashboard/sharing), survey, contact/team/privacy,
  `/health` + `/ready`, static.
- **Nothing else**: worker, CLI, retention, tombstones, leases — identical.

## Provisioning a PreCheck app (mirrors the RegCheck checklist)

1. `heroku create precheck-<env>` — same Procfile (web + worker dynos).
2. Add-ons: its OWN `heroku-postgresql` and Redis. **Never share RegCheck's.**
3. Config vars (before first deploy — same gates as RegCheck):
   ```
   APP_MODE=precheck
   REDIS_TLS_INSECURE=1
   DATABASE_SSL=relaxed
   SESSION_SECRET=<fresh secret, not RegCheck's>
   OPENAI_API_KEY=… (+ other provider keys as offered)
   SCANNED_PDF_FALLBACK=dpt2        # only if hosted OCR is wanted
   ```
   Plus S3: its own bucket + credentials (do not reuse RegCheck's bucket).
4. **OAuth clients are per-domain**: register a NEW Google OAuth client and a
   NEW ORCID client with redirect URIs on the PreCheck domain, and set
   `GOOGLE_CLIENT_ID/SECRET`, `ORCID_CLIENT_ID/SECRET`,
   `OAUTH_REDIRECT_BASE_URL=https://<precheck-domain>`.
5. Domain + DNS + ACM cert via `heroku domains:add`.
6. `git push` the same commit RegCheck runs (or pin a branch per app).
7. Smoke: `/ready` 200 with a live worker; `/` shows PreCheck; `/compare`
   404s; run one evaluation end-to-end.

## Release discipline

- A deploy of either app ships the shared engine — after engine changes,
  deploy BOTH apps (data/config stay fully isolated per app).
- The report-persistence rule applies per site: PreCheck reports live in
  PreCheck's Redis/Postgres/S3 and must survive its deploys.
- Before a public PreCheck launch: give the privacy policy a dedicated pass
  (the shared text is brand-parametrized and accurate, but examples still
  mention papers/comparisons), and finalise the quality prompt + criteria
  wording (sign-off pending).

## Local development

`.claude/launch.json` has a `precheck` config (port 8079) that runs the same
app with `APP_MODE=precheck`; tests live in tests/test_precheck_mode.py.
