"""Central configuration loader.

Every script and notebook reads its parameters from config/settings.yaml
through this module, so a value is only ever defined in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = REPO_ROOT / "config" / "settings.yaml"


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Load settings.yaml into a plain dict.

    Args:
        path: optional explicit path; defaults to config/settings.yaml
              at the repository root.
    """
    settings_path = Path(path) if path else DEFAULT_SETTINGS
    with open(settings_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(settings: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested key with dotted syntax, e.g. get(s, "model.lora.r")."""
    node: Any = settings
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
