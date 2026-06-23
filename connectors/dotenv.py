"""Load local environment variables from .env.

mat is configured via environment variables (API keys, pool paths, etc.).
This helper makes local development reproducible by loading a `.env` file
when present, without requiring shell export hygiene.
"""

from __future__ import annotations


def load_env() -> bool:
    """Load `.env` (if present). Returns True if loaded, else False."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except Exception:
        return False
    return bool(load_dotenv(override=False))

