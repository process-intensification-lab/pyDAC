"""Operational result-persistence policies shared by solver workflows."""

from __future__ import annotations


RUN_FULL = "run_full"
MOO_COMPACT = "moo_compact"
PERSISTENCE_POLICIES = (RUN_FULL, MOO_COMPACT)


def normalize_persistence_policy(value: str | None) -> str:
    """Return one canonical policy, defaulting ordinary runs to ``run_full``."""
    if value is None:
        return RUN_FULL
    if not isinstance(value, str):
        raise TypeError("`persistence_policy` must be a string or None.")
    normalized = value.strip().lower()
    if normalized not in PERSISTENCE_POLICIES:
        supported = ", ".join(PERSISTENCE_POLICIES)
        raise ValueError(
            f"Unsupported persistence policy {value!r}; expected one of: {supported}."
        )
    return normalized


__all__ = [
    "MOO_COMPACT",
    "PERSISTENCE_POLICIES",
    "RUN_FULL",
    "normalize_persistence_policy",
]
