"""Named architecture presets for development-time configuration.

The presets are input conveniences only.  They expand to the same canonical
``simulation.bed`` mapping used by the typed configuration, so resolved JSON
documents and case fingerprints never contain an alias.
"""

from __future__ import annotations

from typing import Any

from pydac.inputs.defaults.column import BED_PROFILE_ALIASES


def resolve_bed_profile(alias: str) -> dict[str, Any]:
    """Return a fresh canonical bed mapping for one public preset alias."""
    if not isinstance(alias, str):
        raise TypeError("`bed` preset must be a string alias.")

    name = alias.strip().lower()
    if name not in BED_PROFILE_ALIASES:
        choices = ", ".join(BED_PROFILE_ALIASES)
        raise ValueError(
            f"Unknown bed preset {alias!r}; choose one of {choices}."
        )

    from pydac.core.bed_architectures import resolve_bed_config

    return resolve_bed_config()


__all__ = ["resolve_bed_profile"]
