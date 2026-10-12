"""Canonical path-based settings for experiment configuration documents.

One vocabulary serves three spellings: a nested runner block, a dotted runner
mapping, and repeated ``--set path=value`` items from the terminal. They are
normalized here to the same canonical paths, which is what makes the VS Code
route and the terminal route produce identical documents.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : stage 3 — everything a runner block or `--set` states.
# Owns     : the canonical path grammar (`simulation.` and `cycle.` prefixes
#            are optional; `cycle.steps.X` and `cycle.X` both mean `steps.X`)
#            and OPAQUE_MAPPING_PATHS, the mappings the user owns wholesale.
# Not here : whether a path exists in the document — the loader checks that.
# Entry    : resolve_settings()  ·  setting_path_parts()
# Guide    : docs/manual/01_inputs.md §3.1, §3.2
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from pydac.config.errors import SettingsError
from pydac.inputs.defaults.steps import INITIAL_SOLID_LOADINGS_FIELD, cycle_defaults


# Mappings whose keys the *user* owns: they travel as one setting instead of
# being flattened, and replace the profile block wholesale instead of merging
# into it. An override block only means what it states.
OPAQUE_MAPPING_PATHS = frozenset(
    {
        "bed.design",
        "bed.material",
        "adsorption.equilibrium_systems",
        "adsorption.kinetics",
        "sorbent.heats_of_adsorption_J_mol",
        "sorbent.heat_capacity",
        "sorbent.fixed_adsorbates",
        "execution.step_options",
        "execution.cycle_options",
    }
    | {
        f"steps.{step}.composition.{INITIAL_SOLID_LOADINGS_FIELD}"
        for step in cycle_defaults()["steps"]
    }
)


def _parse_cli_value(raw_value: str) -> Any:
    """Parse a JSON scalar, falling back to an unquoted string."""
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise SettingsError("A `--set` value must not be empty.")
    try:
        return json.loads(raw_value, parse_constant=_reject_nonstandard_constant)
    except json.JSONDecodeError:
        return raw_value


def _reject_nonstandard_constant(value: str) -> None:
    raise SettingsError(f"Non-standard JSON constant {value!r} is not allowed.")


def _canonical_path(path: str) -> str:
    if not isinstance(path, str):
        raise SettingsError("Setting paths must be strings.")
    parts = tuple(part.strip() for part in path.split("."))
    if not parts or any(not part for part in parts):
        raise SettingsError(f"Invalid setting path {path!r}.")

    if parts[0] == "simulation":
        parts = parts[1:]
    if parts[:2] == ("cycle", "steps"):
        parts = ("steps",) + parts[2:]
    elif parts[:1] == ("cycle",):
        parts = ("steps",) + parts[1:]
    if not parts:
        raise SettingsError(f"Invalid setting path {path!r}.")
    return ".".join(parts)


def _flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def add(path: str, value: Any) -> None:
        if path in flattened:
            raise SettingsError(f"Setting path {path!r} was supplied twice.")
        flattened[path] = deepcopy(value)

    for raw_key, value in mapping.items():
        if not isinstance(raw_key, str):
            raise SettingsError("Setting paths must be strings.")

        # Accept a complete nested runner block as well as the documented
        # public paths. The wrapper is structural, not part of the canonical
        # setting vocabulary.
        if not prefix and raw_key == "simulation" and isinstance(value, Mapping):
            for path, nested_value in _flatten_mapping(value).items():
                add(path, nested_value)
            continue
        if not prefix and raw_key == "cycle" and isinstance(value, Mapping):
            for path, nested_value in _flatten_mapping(value).items():
                add(path, nested_value)
            continue
        key = f"{prefix}.{raw_key}" if prefix else raw_key
        canonical = _canonical_path(key)

        if canonical in OPAQUE_MAPPING_PATHS:
            add(canonical, value)
        elif isinstance(value, Mapping):
            for path, nested_value in _flatten_mapping(value, canonical).items():
                add(path, nested_value)
        else:
            add(canonical, value)
    return flattened


def resolve_settings(
    settings: Mapping[str, Any] | Iterable[str] | None,
) -> dict[str, Any]:
    """Normalize runner mappings or CLI ``path=value`` items.

    Nested runner mappings and dotted CLI-style mappings resolve to the same
    canonical paths. Values are copied so callers can safely reuse their
    settings block after loading an experiment.
    """
    if settings is None:
        return {}
    if isinstance(settings, Mapping):
        return _flatten_mapping(settings)

    resolved: dict[str, Any] = {}
    if isinstance(settings, str):
        settings = (settings,)
    for expression in settings:
        if not isinstance(expression, str) or "=" not in expression:
            raise SettingsError(
                "CLI settings must use the form `path=value`, "
                f"received {expression!r}."
            )
        path, raw_value = expression.split("=", 1)
        canonical = _canonical_path(path.strip())
        if canonical in resolved:
            raise SettingsError(f"Setting path {canonical!r} was supplied twice.")
        resolved[canonical] = _parse_cli_value(raw_value.strip())
    return resolved


def setting_path_parts(path: str) -> tuple[str, ...]:
    """Return canonical dotted path components for loader validation."""
    return tuple(_canonical_path(path).split("."))


__all__ = ["OPAQUE_MAPPING_PATHS", "SettingsError", "resolve_settings", "setting_path_parts"]
