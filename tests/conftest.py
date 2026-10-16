"""Shared pytest path setup for the flat pyDAC module layout."""

from __future__ import annotations

import os
from pathlib import Path

# Tests never show figures. Without a backend, matplotlib probes $DISPLAY on its
# first pyplot import, which can hang for minutes on an unreachable X server.
# A backend set by the user still wins.
os.environ.setdefault("MPLBACKEND", "Agg")
