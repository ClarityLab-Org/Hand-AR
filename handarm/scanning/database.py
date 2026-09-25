"""Database persistence for model search cache."""

import json
import os
from pathlib import Path
from typing import Dict, Any

from handarm.config import PROJECT_ROOT

# Always resolve relative to project root, not CWD (Bug #5 fix)
INDEX_FILE: str = str(PROJECT_ROOT / "database.json")


def save_index(data: Dict[str, Any]) -> None:
    """Atomically saves search index data to database.json.
    Uses write-to-temp-then-rename pattern to prevent corruption.
    """
    temp_file = INDEX_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_file, INDEX_FILE)


def load_index(path: str | None = None) -> Dict[str, Any]:
    """Loads the search database index from database.json.
    Returns empty dict if file doesn't exist or is corrupted.
    """
    path = path or INDEX_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print("Corrupted or empty database. Resetting...")
        return {}
