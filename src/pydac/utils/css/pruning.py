"""Remove intermediate cycle folders of a finished CSS run."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


def prune_cycle_directories(results_base_folder: Path, keep_cycle_idx: int) -> list[str]:
    """Remove intermediate cycle folders while preserving the selected cycle."""
    if keep_cycle_idx < 1:
        raise ValueError("`keep_cycle_idx` must be at least 1.")
    if not results_base_folder.is_dir():
        raise FileNotFoundError(f"Results folder does not exist: {results_base_folder}")

    removed = []
    for path in results_base_folder.iterdir():
        match = re.fullmatch(r"cycle_(\d+)", path.name)
        if not path.is_dir() or match is None:
            continue
        if int(match.group(1)) == keep_cycle_idx:
            continue
        shutil.rmtree(path)
        removed.append(path.name)
    return sorted(removed)
