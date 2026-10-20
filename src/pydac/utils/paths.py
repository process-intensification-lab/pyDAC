"""
Paths as they are shown in terminal messages.

A run prints where it writes (run folder, ``[Saved]`` figures, reports). Inside
the working folder the path is printed relative to it, so the messages stay
short and do not carry the machine's folder layout; elsewhere it is printed
absolute. Only the text changes: the files are written where they always were.
"""

from __future__ import annotations

import os
from pathlib import Path


def display_path(path: str | os.PathLike) -> str:
    """Return ``path`` relative to the working folder if it lies inside it, else absolute.

    A message never fails: if the working folder is gone (deleted while the run
    went on), the path is printed as given.
    """
    try:
        absolute = Path(os.path.abspath(path))
        return str(absolute.relative_to(Path.cwd()))
    except ValueError:
        return str(absolute)
    except OSError:
        return os.fspath(path)
