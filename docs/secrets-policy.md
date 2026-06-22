# Secrets policy

mat holds **two classes** of credentials:

| Class | Where it lives | Exposed to clients? |
|-------|----------------|---------------------|
| Gateway key | `MAT_GATEWAY_KEY` env or config | Yes — clients send `Authorization: Bearer …` |
| Downstream provider keys | Named in connector `auth_env` fields | **Never** |

## Rules

1. **Never log secret values.** Not in traces, not in debug output, not in error
   messages. Log the `auth_env` *name* if needed for diagnosis, never the key.
2. **Never commit secrets.** Connector files reference env var names only
   (`auth_env: OPENROUTER_API_KEY`), not values.
3. **Redact in traces.** The trace logger runs all string fields through a redactor
   that strips values matching `Bearer …`, `sk-…`, and any env var listed in loaded
   connectors' `auth_env` fields.
4. **Config files in repos are templates.** Use `.env` (gitignored) or your OS secret
   store for real keys.

## Implementation checklist

Every code path that touches outbound HTTP or writes a trace must:

- [ ] Read downstream keys from `os.environ[connector.auth_env]` at call time
- [ ] Pass keys only to the HTTP client headers, not into log context
- [ ] Route trace payloads through `traces.redact.redact_secrets()` before persistence

## Rotation

If a key is accidentally logged or committed: rotate it at the provider immediately.
mat's trace files are local JSONL by default — treat `traces/` like sensitive data if
your downstream keys were ever mishandled.

## Reporting

If you find a code path that logs or persists a secret value, open a security issue or
email the maintainer directly. Do not paste the secret in the report.
