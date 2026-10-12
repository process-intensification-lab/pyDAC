"""What an explicit ``null`` means, path by path.

An omitted field always inherits the reference profile. An explicit ``null``
is a different statement — the user is saying something about that field — so
each path that accepts one names the outcome it asks for. The policy is a
table rather than control flow, because a user must be able to read, print,
and test it; ``config.loader`` only applies what this module decides.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : stages 2-3 — consulted whenever a document or a setting says
#            `null`. An OMITTED field never reaches here; it just inherits.
# Owns     : the four policies and the path table below.
# Not here : the canonical value itself — the packed-bed design is delegated
#            to `bed_architectures`, which owns architecture validation. Every
#            other default is a plain value in the profile.
# Entry    : null_policy()  ·  resolve_null()
# Guide    : docs/manual/01_inputs.md §5
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping

from pydac.config.errors import ConfigurationError
from pydac.config.schema import default_bed_mapping
from pydac.config.steps import NULLABLE_STEP_FIELDS


INHERIT = "inherit"  # same as omitting the field
CANONICAL = "canonical"  # supplied by the module that owns the value
DERIVED = "derived"  # computed by the derivation stage from other fields
DISABLED = "disabled"  # kept as None; the feature is switched off

NULL_POLICY_BY_PATH = {
    ("simulation", "bed"): CANONICAL,
    ("simulation", "geometry", "length_m"): DERIVED,
    ("simulation", "geometry", "radius_m"): DERIVED,
    ("simulation", "geometry", "aspect_ratio"): DERIVED,
    ("simulation", "geometry", "volume_m3"): DERIVED,
    ("simulation", "wall", "overall_htc_override_W_m2K"): DISABLED,
    ("simulation", "wall", "bed_wall_htc_override_W_m2K"): DISABLED,
    ("simulation", "bed", "material"): DISABLED,
    ("execution", "resume_from_cycle"): DISABLED,
    ("execution", "cycle_steps"): DISABLED,
    ("execution", "termination_mode"): DISABLED,
    ("execution", "css_probe_step"): DISABLED,
    ("execution", "extra_tag"): DISABLED,
}

_STEPS_PREFIX = ("simulation", "cycle", "steps")


@lru_cache(maxsize=1)
def canonical_values() -> Mapping[tuple[str, ...], Any]:
    """Values a `CANONICAL` directive delegates to the module owning them.

    Built lazily so importing ``config`` never requires ``bed_architectures``
    to be importable, and cached so the resolver runs once per process.
    """
    return MappingProxyType({("simulation", "bed"): default_bed_mapping()})


def null_policy(path: tuple[str, ...]) -> str | None:
    """Return the ``null`` policy for a document path, or ``None`` if forbidden.

    Step fields accept ``null`` exactly where their typed record admits
    ``None`` (see ``config.steps.NULLABLE_STEP_FIELDS``) and always inherit.
    """
    if path in NULL_POLICY_BY_PATH:
        return NULL_POLICY_BY_PATH[path]
    if len(path) == 6 and path[:3] == _STEPS_PREFIX:
        group, field_name = path[4], path[5]
        if field_name in NULLABLE_STEP_FIELDS.get(group, frozenset()):
            return INHERIT
    return None


def resolve_null(path: tuple[str, ...], base_value: Any) -> Any:
    """Turn an explicit ``null`` at ``path`` into the value its policy names."""
    policy = null_policy(path)
    if policy is None:
        raise ConfigurationError(f"`null` is not allowed for `{'.'.join(path) or '<root>'}`.")
    if policy == CANONICAL:
        return deepcopy(canonical_values()[path])
    if policy == INHERIT:
        return deepcopy(base_value)
    return None  # DERIVED and DISABLED both leave the next stage a None.


__all__ = [
    "CANONICAL",
    "DERIVED",
    "DISABLED",
    "INHERIT",
    "NULL_POLICY_BY_PATH",
    "canonical_values",
    "null_policy",
    "resolve_null",
]
