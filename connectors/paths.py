"""Pool directory resolution — examples vs installed connectors."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "connectors" / "examples"
INSTALLED_REPO_DIR = REPO_ROOT / "connectors" / "installed"
USER_POOL_DIR = Path.home() / ".config" / "mat" / "connectors"
AA_CACHE_PATH = Path.home() / ".config" / "mat" / "cache" / "aa_models.json"


def user_config_dir() -> Path:
    return Path.home() / ".config" / "mat"


def default_pool_dir() -> Path:
    """Installed pool used by mat-serve, mat-benchmark, mat-train-live."""
    if env := os.environ.get("MAT_POOL_DIR"):
        return Path(env).expanduser()
    if USER_POOL_DIR.exists() and any(USER_POOL_DIR.glob("*.yaml")):
        return USER_POOL_DIR
    if INSTALLED_REPO_DIR.exists() and any(INSTALLED_REPO_DIR.glob("*.yaml")):
        return INSTALLED_REPO_DIR
    return USER_POOL_DIR


def ensure_user_pool_dir() -> Path:
    USER_POOL_DIR.mkdir(parents=True, exist_ok=True)
    return USER_POOL_DIR


def is_example_connector(path: Path) -> bool:
    try:
        return EXAMPLES_DIR in path.resolve().parents
    except OSError:
        return False
