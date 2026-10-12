"""Single-source access to the reference profile.

The packaged default is composed from ``inputs.defaults``. ``PYDAC_PROFILE``
remains an advanced hook for selecting an alternate complete JSON profile.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : stage 1 of 5 — the reference profile every case inherits from.
# Owns     : composing the packaged default or reading explicit `PYDAC_PROFILE`.
# Not here : what the values mean (schema.py), how they merge (loader.py).
# Entry    : reference_profile()  ·  read_json_document()
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydac.inputs.defaults.profile import default_profile

from pydac.config.errors import ConfigurationError


PROFILE_PATH_VARIABLE = "PYDAC_PROFILE"


def profile_path() -> Path:
    """Return the alternate complete profile selected through ``PYDAC_PROFILE``."""
    selected = os.environ.get(PROFILE_PATH_VARIABLE)
    if selected:
        return Path(selected)
    raise ConfigurationError("PYDAC_PROFILE is not set; defaults come from inputs.defaults.")


def reject_nonstandard_constant(value: str) -> None:
    """Reject ``NaN``/``Infinity``: configurations must stay standard JSON."""
    raise ConfigurationError(f"Non-standard JSON constant {value!r} is not allowed.")


def read_json_document(path: str | Path) -> dict[str, Any]:
    """Parse a JSON object from ``path``, rejecting non-standard constants."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            document = json.load(handle, parse_constant=reject_nonstandard_constant)
    except OSError as exc:
        raise ConfigurationError(f"Could not read configuration {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigurationError(f"{source} must be a JSON object.")
    return document


@lru_cache(maxsize=2)
def _cached_reference_profile(path: Path) -> dict[str, Any]:
    """Parse a profile once per path; callers always receive a copy."""
    return read_json_document(path)


def reference_profile() -> dict[str, Any]:
    """Return an independent copy of the reference profile in use."""
    if not os.environ.get(PROFILE_PATH_VARIABLE):
        return default_profile()
    return deepcopy(_cached_reference_profile(profile_path()))


__all__ = [
    "PROFILE_PATH_VARIABLE",
    "profile_path",
    "read_json_document",
    "reference_profile",
    "reject_nonstandard_constant",
]
