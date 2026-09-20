# RegCheck Redis data schema (for the analytics dashboard)

Derived from the code only (no production data was read). Sources:
`backend/routes/comparisons.py`, `backend/routes/survey.py`, `backend/routes/reports.py`,
`backend/services/comparisons.py`, `backend/services/reports.py`,
`backend/services/report_artifacts.py`, `backend/services/cost_tracking.py`,
`backend/worker.py`, `backend/db/models.py`.

## 0. Read this first

1. **User accounts are not in Redis.** Accounts, OAuth identities, API keys, report
   ownership and shares live in **Postgres** (`users`, `oauth_identities`, `api_keys`,
   `reports`, `report_shares`). Sessions are signed cookies, and the rate limiter is
   in-process memory. Neither touches Redis. Every account metric is therefore
   "N/A in Redis" and needs Postgres (section 4).
2. **The task hash key is the bare UUID**, `<task_id>`, not `comparison:task:<task_id>`.
   There is no prefix and no index of task ids, so task hashes can only be discovered
   by `SCAN` + `TYPE == hash` + UUID-shaped key, or via `survey:task_ids`, or via
   Postgres `reports.task_id`.
3. **The task hash carries no timestamp.** No `created_at`, `started_at` or
   `finished_at` is ever written. The only date in Redis is `survey:<task_id>.submitted_at`.
4. **Most of Redis is ephemeral.** Anonymous task hashes expire after 7 days
   (`ANONYMOUS_TASK_TTL_SECONDS`), failed ones after 3 days (`TASK_TTL_SECONDS`).
   Only signed-in users' reports are persisted (`PERSIST`), and they disappear when the
   owner deletes them. `survey:*` keys have **no TTL**, so they are the only durable
   usage log, but they are removed when the matching report is explicitly deleted.
   A dashboard that wants history for anonymous runs (cost in particular) has to
   snapshot on a schedule shorter than 7 days.

Metric codes used below:
**M1** usage per day · **M2** new accounts per day · **M3** cumulative accounts ·
**M4** domain (research area) distribution · **M5** user demographics ·
**M6** reports per user account · **M7** costs per day · **–** operational only.

## 1. Redis storage schema

### 1.1 Key inventory

| Key pattern | Type | TTL | Purpose |
|---|---|---|---|
| `<task_id>` | Hash | anon: 7 d; signed-in: none; FAILURE: reset to 3 d | One comparison run / report: state, settings, results, cost |
| `survey:<task_id>` | Hash | none | Post-submission survey answers (or profile copy) for that run |
| `survey:task_ids` | Set | none | Index of every `task_id` that has a `survey:<task_id>` hash |
| `comparison:queue` | List | none | Pending jobs (JSON job payloads, `RPUSH` / `BRPOPLPUSH`) |
| `comparison:processing:<worker_id>` | List | none | Jobs in flight on one worker |
| `comparison:processing` | List | none | Legacy in-flight list (pre per-worker lists); still read for backpressure/recovery |
| `comparison:deadletter` | List | none | Jobs that exceeded the retry limit or were unparseable |
| `worker:heartbeat:<worker_id>` | String | 30 s | Unix time of the worker's last loop |
| `upload:<task_id>:{paper,prereg,csv}` | String | ≥ 7 d | gzip+base64 upload blob; only when S3 is not configured |
| `report:<task_id>:manifest` | String (JSON) | inherits report retention | Evidence manifest (sources, chunks, artifact pointers) |
| `report:<task_id>:source:<source_id>:raw` | String | same | gzip+base64 original document |
| `report:<task_id>:source:<source_id>:render` | String | same | gzip+base64 rendered text/layout for quote tracing |
| `report:s3_artifact_cleanup` | Sorted set | none | member = S3 key, score = unix expiry time |
| `deleted:<task_id>` | String `"1"` | 3 h | Deletion tombstone so an in-flight worker doesn't resurrect a deleted report |
| `regen:lock:<task_id>` | String `"1"` | 120 s | NX lock, one regeneration at a time |

All values are strings (`decode_responses=True`); "int" below means a decimal string.

### 1.2 `<task_id>` hash (one per run)

`task_id` is a UUID4 string generated at submission; it is also the report URL id
(`/result/<task_id>`) and the Postgres `reports.task_id` primary key.

| Field | Type | Meaning | Metrics |
|---|---|---|---|
| `state` | `PENDING` \| `IN_PROGRESS` \| `SUCCESS` \| `FAILURE` (`UNKNOWN` / `FORBIDDEN` are API-response-only, never stored) | Job lifecycle | M1 (filter completed vs failed) |
| `status` | string | Human-readable progress or error message | – |
| `comparison_type` | `clinical_trials` \| `general_preregistration` \| `registered_report` \| `animals_trials` | Which tool/mode was run | M1 (breakdown), M4 (weak proxy for field) |
| `total_dimensions` | int | Number of dimensions requested | M1 (volume), M7 (cost driver) |
| `processed_dimensions` | int | Dimensions finished so far | – |
| `dimensions` | JSON array of strings | Dimension names | – |
| `result_json` | JSON `{items: [...], cost: {...}}` | Report content plus cost snapshot; see 1.3 | M7 |
| `settings_json` | JSON object | `comparison_type`, `client` (LLM provider/model choice), `parser_choice`, `reasoning_effort`, `append_previous_output` (bool), `multiple_experiments` (bool), `experiment_number`, `dimensions` | M1 / M7 breakdowns by model and effort |
| `parser_used` | string | Parser actually used (may differ after scanned-PDF fallback) | – |
| `title` | string | Auto-generated, user-renamable report title | – (contains paper filename; treat as sensitive) |
| `visibility` | `public` \| `private` (anonymous = `public`; legacy hashes may still hold `unlisted` / `restricted`, both now meaning `private`) | Sharing level | – |
| `owner_id` | string: `users.id` UUID, or `""` for anonymous | Owning account | M6; anonymous-vs-signed-in split for M1 |
| `retention` | `"persist"` or seconds as digit string | Retention policy; evidence artifacts inherit it | – (tells you whether the key will expire) |
| `regen_job` | JSON job payload | Verbatim job for "Regenerate" (file paths, upload/S3 keys, OSF URL, registration id) | – |
| `evidence_status` | `pending` \| `ready` \| error state | Evidence bundle status | – |
| `evidence_error` | string | Evidence failure message | – |
| `evidence_storage` | `redis` (or S3 variant) | Where artifacts are stored | – |
| `evidence_source_count`, `evidence_chunk_count`, `evidence_artifact_count`, `evidence_artifact_bytes` | int | Evidence bundle size stats | – (storage monitoring) |
| `multi_study_isolation`, `multi_study_isolation_error` | string / JSON | Multi-study isolation outcome | – |
| `rr_integrity` | JSON | Registered Reports carried-forward text analysis | – |

### 1.3 `result_json` contents

`items[]`: one object per dimension: `dimension`, `chain_of_thought`,
`paper_content_quotes`, `paper_content_summary`, `registration_content_quotes`,
`registration_content_summary`, `deviation_judgement`, `deviation_information`,
`unlocated_in_paper`, `unlocated_in_registration`. Report content, not usage data.
(`deviation_judgement` could feed a verdict-distribution chart if one is ever wanted.)

`cost` (absent on reports created before cost tracking was added):

| Field | Type | Meaning | Metrics |
|---|---|---|---|
| `input_tokens`, `output_tokens` | int | LLM tokens for the run | M7 |
| `embedding_tokens` | int | Embedding tokens | M7 |
| `llm_calls` | int | Number of LLM completions | M7 |
| `llm_usd`, `embedding_usd`, `total_usd` | float, 4 dp | **Estimated** USD from a prefix-matched price table (`COST_PRICING_JSON` overrides) | M7 |
| `estimate_complete` | bool | `false` when a model had no price row, so `total_usd` is an undercount | M7 (data-quality flag) |
| `models` | array of strings | Model ids used | M7 breakdown |

Regenerating a report resets `result_json`, so the earlier run's cost is overwritten
and lost. Costs are estimates, not billing data.

### 1.4 `survey:<task_id>` hash

Written when the user lands on or submits the `/next-steps/<task_id>` page. UI only:
runs submitted via `/api/v1` never get a survey hash.

| Field | Type | Meaning | Metrics |
|---|---|---|---|
| `submitted_at` | ISO-8601 UTC string | When the survey was recorded, seconds to minutes after job submission. **The only timestamp in Redis.** | M1, M7 (date axis), time axis for M4/M5 |
| `research_field` | `Psychology` \| `Medicine` \| `Economics` \| `Animal Research` \| `Other` \| `""` | Scientific field | M4 |
| `academic_position` | `Undergrad` \| `Master's` \| `PhD` \| `Postdoc` \| `Professor` \| `Non-academic` \| `""` | Career stage | M5 |
| `use_case` | `Author of paper` \| `Reviewer of paper` \| `Journal editor of paper` \| `Reader of paper` \| `Other` \| `""` | Why they use RegCheck | M5 |
| `skipped` | `"0"` \| `"1"` | `"1"` = user pressed skip; answer fields are then empty | M4/M5 (exclude from denominators), M1 (still counts as a run) |
| `from_profile` | `"1"` or absent | Answers were copied from a signed-in user's profile rather than typed | M1 split signed-in vs anonymous; M4/M5 de-duplication caveat |
| `comparison_type` | same enum as 1.2, or absent | Copied from the task hash; **anonymous form path only** (absent when `from_profile=1`) | M1 breakdown |

`survey:task_ids` (Set of `task_id`): the enumeration entry point for all of the above → M1, M4, M5, M7.

### 1.5 Relationships

```
survey:task_ids ──member──▶ task_id ──┬─▶ survey:<task_id>            (1:0..1, no TTL)
                                      ├─▶ <task_id>                   (hash; may have expired)
                                      ├─▶ report:<task_id>:manifest ─▶ report:<task_id>:source:<source_id>:{raw,render}
                                      ├─▶ upload:<task_id>:{paper,prereg,csv}
                                      ├─▶ deleted:<task_id>, regen:lock:<task_id>
                                      └─▶ Postgres reports.task_id    (signed-in runs only)

<task_id>.owner_id ─▶ Postgres users.id
comparison:queue / processing / deadletter items are JSON with a "task_id" field
```

Deleting a report removes `<task_id>`, `survey:<task_id>`, its `survey:task_ids`
membership, uploads and artifacts, so that run vanishes from every Redis-derived metric.

**Privacy note.** `survey:<task_id>` deliberately stores no `user_id` ("survey answers
are unlinked from the account", `survey.py`). Joining it to `<task_id>.owner_id` re-links
answers to accounts. The dashboard should aggregate without performing that join, or
that design decision should be revisited explicitly.

## 2. Statistic → required data

| Statistic | From Redis | Verdict |
|---|---|---|
| **Usage per day** | `SMEMBERS survey:task_ids` → `survey:<id>.submitted_at` bucketed by date; optional breakdown via `survey.comparison_type`, `from_profile`, and (while it still exists) `<id>.state` / `settings_json` | **Partial.** Counts UI runs that reached the next-steps page. Misses API runs, runs whose report was later deleted, and any run where the redirect never loaded. No submission timestamp exists on the task hash itself. |
| **New accounts per day** | none | **N/A - Data Not Present in REDIS.** Postgres `users.created_at`. |
| **User accounts (cumulative)** | none | **N/A - Data Not Present in REDIS.** Cumulative sum of Postgres `users.created_at`. |
| **Distribution of domains** | `survey:<id>.research_field` where `skipped="0"` and non-empty | **Available, per run, not per person.** A signed-in user with 20 reports contributes 20 identical `from_profile=1` rows. For a per-user distribution use Postgres `users.research_field`. |
| **Demographics of users** | `survey:<id>.academic_position`, `survey:<id>.use_case` (same filters) | **Available, per run**, same caveat. Only these two attributes exist; no country, gender, age or institution is collected anywhere. Per-user: Postgres `users.academic_position`, `users.use_case`. |
| **Reports per user account (over time)** | `<task_id>.owner_id` grouped and counted (requires `SCAN` for task hashes) | **Partial and undated.** Gives the current distribution for signed-in, non-deleted reports, but Redis has no date for it. Use Postgres `reports.owner_id` + `reports.created_at` (indexed), which is the intended source. |
| **Costs per day** | `<id>.result_json` → `cost.total_usd` (plus `llm_usd`, `embedding_usd`, tokens, `models`, `estimate_complete`), dated by `survey:<id>.submitted_at` or, for signed-in runs, Postgres `reports.created_at` | **Partial.** Estimated USD only; anonymous hashes expire after 7 days (failed after 3) so the crawler must snapshot; regeneration overwrites earlier cost; API runs have no Redis date; pre-cost-tracking reports have no `cost`. |

## 3. Data → statistic

Given per field in the **Metrics** column of tables 1.2, 1.3 and 1.4. Summary by metric:

| Metric | Contributing Redis fields |
|---|---|
| M1 usage/day | `survey:task_ids`; `survey.submitted_at`, `.skipped`, `.from_profile`, `.comparison_type`; task `state`, `comparison_type`, `total_dimensions`, `settings_json`, `owner_id` |
| M2 new accounts/day | none |
| M3 cumulative accounts | none |
| M4 domains | `survey.research_field` (+ `.skipped`, `.from_profile`, `.submitted_at`); task `comparison_type` as weak proxy |
| M5 demographics | `survey.academic_position`, `survey.use_case` (+ `.skipped`, `.from_profile`, `.submitted_at`) |
| M6 reports/user | task `owner_id` (undated) |
| M7 costs/day | task `result_json.cost.*`, `settings_json`, `total_dimensions`; `survey.submitted_at` for the date |

Every key in 1.1 not listed here (queues, heartbeats, uploads, artifacts, tombstones,
locks) is operational and contributes to none of the seven metrics. Queue lengths and
`comparison:deadletter` would support a separate health panel (live depth, failures).

## 4. Postgres fields the dashboard needs

| Table.column | Metrics |
|---|---|
| `users.id`, `users.created_at` | M2, M3 |
| `users.research_field` | M4 (per user) |
| `users.academic_position`, `users.use_case` | M5 (per user) |
| `reports.task_id`, `.owner_id`, `.created_at` | M6, M1 (signed-in runs incl. API), date join for M7 |
| `reports.comparison_type`, `.source` (`ui` \| `api`) | M1 breakdown |
| `oauth_identities.provider` (`google` \| `orcid`) | M5 (sign-in method) |
| `api_keys.request_count`, `.last_used_at` | API usage |

## 5. Gaps worth closing in code

Cheap additions that would make M1 and M7 exact rather than proxied:

- Write `created_at` (and `finished_at` on SUCCESS/FAILURE) to the task hash.
- Append one compact record per finished run to a durable, non-expiring log
  (e.g. a Redis stream `analytics:runs` or a Postgres table) with date, `comparison_type`,
  `source`, signed-in flag, `state`, model, and the `cost` snapshot, so anonymous costs
  survive the 7-day TTL and regenerations don't overwrite history.

## 6. Crawling safely

Read-only commands only (`SCAN`, `TYPE`, `HGETALL`/`HMGET`, `SMEMBERS`/`SSCAN`, `TTL`,
`LLEN`). Use `SCAN` rather than `KEYS`. Prefer `HMGET` of the needed fields over
`HGETALL`, since `result_json` and `regen_job` are large. Do not read `upload:*` or
`report:*:source:*` values (users' unpublished manuscripts).
