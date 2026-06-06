# RegCheck API Handoff For INSPECT-AI

The RegCheck API is deployed and working.

## Base URL

```text
https://preregpt-8584b32c9141.herokuapp.com
```

## Authentication

Use server-to-server authentication only. Do not call this API from browser-side JavaScript.

Send the shared token with every request:

```http
Authorization: Bearer <REGCHECK_API_TOKEN>
```

The token should be shared separately through a secure channel.

## Ethics-Only PyMuPDF Check

There is no separate endpoint for the ethics-only version. Use the standard comparison endpoint with:

- `parser_choice=pymupdf`
- `dimensions` set to exactly the three ethics dimensions below

This makes RegCheck run only:

- ethics committee
- ethics approval number
- ethics approval date

It does not run the other default checks.

### Path

```http
POST /api/v1/comparisons
Content-Type: multipart/form-data
```

### ClinicalTrials.gov Example

```bash
curl -X POST https://preregpt-8584b32c9141.herokuapp.com/api/v1/comparisons \
  -H "Authorization: Bearer <REGCHECK_API_TOKEN>" \
  -F "paper=@paper.pdf" \
  -F "registration_id=NCT01234567" \
  -F "parser_choice=pymupdf" \
  -F 'dimensions=[
    {
      "dimension": "Ethical approval: Committee",
      "definition": "The official name of the ethics committee, IRB, REC, REB, or other body that approved the study."
    },
    {
      "dimension": "Ethical approval: Number",
      "definition": "The ethics approval identifier, approval number, protocol code, IRB number, REC reference, or equivalent approval reference."
    },
    {
      "dimension": "Ethics approval: Date",
      "definition": "The date on which ethics approval was granted, including approval dates for relevant protocol amendments where reported."
    }
  ]'
```

### Registration File Example

```bash
curl -X POST https://preregpt-8584b32c9141.herokuapp.com/api/v1/comparisons \
  -H "Authorization: Bearer <REGCHECK_API_TOKEN>" \
  -F "paper=@paper.pdf" \
  -F "registration_file=@registration-or-preregistration.pdf" \
  -F "parser_choice=pymupdf" \
  -F 'dimensions=[
    {
      "dimension": "Ethical approval: Committee",
      "definition": "The official name of the ethics committee, IRB, REC, REB, or other body that approved the study."
    },
    {
      "dimension": "Ethical approval: Number",
      "definition": "The ethics approval identifier, approval number, protocol code, IRB number, REC reference, or equivalent approval reference."
    },
    {
      "dimension": "Ethics approval: Date",
      "definition": "The date on which ethics approval was granted, including approval dates for relevant protocol amendments where reported."
    }
  ]'
```

## Standard Comparison Endpoint

```http
POST /api/v1/comparisons
Content-Type: multipart/form-data
```

Required every time:

- `paper`

Then provide exactly one of:

- `registration_id`
- `registration_file`

Optional fields:

- `dimensions`: JSON array string. If omitted, RegCheck uses backend defaults.
- `parser_choice`: defaults to `grobid`. Use `pymupdf` for the ethics-only request above.
- `client`: defaults to `openai`
- `reasoning_effort`: defaults to `medium`
- `append_previous_output`: defaults to `yes`
- `multiple_experiments`: defaults to `no`
- `experiment_number`: only used when `multiple_experiments` is `yes`

## Queued Response

Successful `POST` requests return quickly with HTTP `202`:

```json
{
  "task_id": "abc-123",
  "state": "queued",
  "status": "Task queued",
  "status_url": "/api/v1/comparisons/abc-123"
}
```

## Poll A Comparison

```http
GET /api/v1/comparisons/{task_id}
```

```bash
curl https://preregpt-8584b32c9141.herokuapp.com/api/v1/comparisons/<task_id> \
  -H "Authorization: Bearer <REGCHECK_API_TOKEN>"
```

Possible states:

- `queued`
- `in_progress`
- `success`
- `failure`

## Running Response

```json
{
  "task_id": "abc-123",
  "state": "in_progress",
  "status": "Processed 1/3: Ethical approval: Committee",
  "processed_dimensions": 1,
  "total_dimensions": 3,
  "result": {
    "items": []
  }
}
```

## Final Response

```json
{
  "task_id": "abc-123",
  "state": "success",
  "status": "Report complete",
  "processed_dimensions": 3,
  "total_dimensions": 3,
  "result": {
    "items": [
      {
        "dimension": "Ethical approval: Committee",
        "paper_content_quotes": "...",
        "paper_content_summary": "...",
        "registration_content_quotes": "...",
        "registration_content_summary": "...",
        "deviation_judgement": "yes",
        "deviation_information": "..."
      }
    ]
  }
}
```

## Validation Errors

Errors use this shape:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message."
  }
}
```

Common codes:

- `MISSING_API_AUTH`
- `INVALID_API_AUTH`
- `MISSING_PAPER`
- `MISSING_REGISTRATION_INPUT`
- `AMBIGUOUS_REGISTRATION_INPUT`
- `INVALID_REGISTRATION_ID`
- `INVALID_DIMENSIONS`
- `TASK_NOT_FOUND`

## Notes

- The ethics-only request is just a dimensions override. INSPECT-AI should store and reuse that three-item `dimensions` array when it wants the ethics-only workflow.
- `parser_choice=pymupdf` makes PyMuPDF the primary parser for the uploaded paper PDF.
- If `dimensions` is omitted, RegCheck runs the full default dimension set for the selected comparison mode.
