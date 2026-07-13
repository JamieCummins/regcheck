# Security Policy

## Reporting a vulnerability

Email **jamie.cummins@unibe.ch** with a description of the issue, steps to
reproduce, and the impact you believe it has. Please do not open a public
GitHub issue for security reports, and please do not test against other
people's reports or accounts on the hosted service.

You can expect an acknowledgement within a few working days. Once a fix is
released we are happy to credit you (or keep you anonymous — your choice).

## Supported versions

Only the latest released version (the code running at regcheck.app and the
current `main` branch) receives security fixes.

## Scope notes

- Reports created without an account are intentionally viewable by anyone who
  holds the report link ("public by link"); this is documented behaviour, not
  a vulnerability. Private reports must not be readable without a matching
  signed-in account — report anything that violates that.
- API keys are bearer credentials (`rc_live_…`). Anyone holding a key can act
  as its owner; keys can be revoked from the profile page.
- Uploaded documents are processed by the model provider selected at run time
  (and OpenAI for retrieval embeddings), as described in the privacy policy.
