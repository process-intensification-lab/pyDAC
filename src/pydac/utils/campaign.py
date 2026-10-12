"""Atomic JSON writes for simulation and campaign records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = False,
) -> None:
    """Atomically write a JSON-serializable campaign payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=sort_keys)
    temporary_path.replace(path)
