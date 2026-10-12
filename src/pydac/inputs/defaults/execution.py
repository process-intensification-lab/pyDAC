"""Solve, CSS, output, and plotting behavior defaults."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydac.inputs.defaults._rules import Rule, nonempty_string, positive_integer, positive_number


def execution_defaults() -> dict[str, Any]:
    """Return the default execution behavior for one resolved experiment."""
    return {
        "solve": True,
        "plot_profiles": False,
        "plot_final_cycle_profiles": False,
        "keep_only_final_cycle": False,
        "print_to_terminal": True,
        "print_cycle_build": True,
        "output_root": "simulation_results",
        "max_cycles": 50,
        "css_tolerance": 0.005,
        "resume_from_cycle": None,
        "use_sparsity": True,
        "compute_numerical_sparsity": False,
        "cvode_sparsity_mode": "pde_only",
        "cycle_steps": None,
        "termination_mode": None,
        "step_options": {},
        "cycle_options": {},
        "css_probe_step": None,
        "require_css": True,
        "extra_tag": None,
        "plot_format": "pdf",
        "plot_cyclic_profiles": True,
        "generate_kpi_outputs": True,
    }


# The tag starts the run/campaign folder name: one portable folder name, short enough
# that the whole leaf (tag, grid, bed, hash) stays well under the 255-byte limit.
_FOLDER_TAG = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,63}")


def _folder_tag(value: Any, name: str) -> Any:
    """Check (not change: the fingerprint keeps the text) that the tag, with its
    spaces as "_" as in the folder name, is one portable folder name."""
    if not isinstance(value, str):
        return value  # left to the type check
    if not _FOLDER_TAG.fullmatch(value.strip().replace(" ", "_")):
        raise ValueError(
            f"`{name}` starts the folder name: use at most 64 letters, digits, '_', '-' or "
            f"'.' (spaces become '_'), not starting with '.'; got {value!r}."
        )
    return value


def _canonical_step(token: Any, name: str) -> str:
    from pydac.utils.tvsa import normalize_step_token

    try:
        return normalize_step_token(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{name}`: {exc}") from exc


def _termination_mode(value: Any, name: str) -> str:
    from pydac.utils.tvsa import normalize_termination_mode

    return normalize_termination_mode(value, option_name=name, allow_none=False)


def _step_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence of step tokens.")
    steps = tuple(_canonical_step(token, name) for token in value)
    if not steps:
        raise ValueError(f"`{name}` must name at least one step.")
    return steps


def _step_options(value: Any, name: str) -> dict[str, dict[str, Any]]:
    from pydac.utils.run import validate_step_options_structure
    from pydac.utils.tvsa import normalize_step_options

    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping of step token to options.")
    validate_step_options_structure(dict(value))
    return {
        step: dict(options)
        for step, options in normalize_step_options(value).items()
    }


def _cycle_options(value: Any, name: str) -> dict[int, dict[str, Any]]:
    from pydac.utils.css import validate_cycle_options_structure

    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping of cycle index to overrides.")
    indexed = {}
    for raw_index, overrides in value.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"`{name}` keys must be integer cycle indices.") from exc
        if not isinstance(overrides, Mapping):
            raise TypeError(f"`{name}[{index}]` must be a mapping of overrides.")
        resolved = dict(overrides)
        if "step_options" in resolved:
            resolved["step_options"] = _step_options(
                resolved["step_options"], f"{name}[{index}].step_options"
            )
        indexed[index] = resolved
    validate_cycle_options_structure(indexed)
    return indexed


def _lowercase(value: Any, _name: str) -> Any:
    return value.lower() if isinstance(value, str) else value


def execution_rules():
    """Return stronger execution constraints and canonicalizers."""
    return {
        ("execution", "output_root"): nonempty_string(),
        ("execution", "max_cycles"): positive_integer(),
        ("execution", "css_tolerance"): positive_number(),
        ("execution", "resume_from_cycle"): positive_integer(nullable=True),
        ("execution", "cvode_sparsity_mode"): Rule(
            kind="string",
            choices=("pde_only", "full"),
            normalize=_lowercase,
        ),
        ("execution", "cycle_steps"): Rule(
            kind="array",
            nullable=True,
            min_items=1,
            item_kind="string",
            normalize=_step_sequence,
        ),
        ("execution", "termination_mode"): Rule(
            kind="string",
            nullable=True,
            choices=("time", "event"),
            normalize=_termination_mode,
        ),
        ("execution", "step_options"): Rule(
            kind="object", opaque=True, normalize=_step_options
        ),
        ("execution", "cycle_options"): Rule(
            kind="object", opaque=True, normalize=_cycle_options
        ),
        ("execution", "css_probe_step"): Rule(
            kind="string", nullable=True, min_length=1, normalize=_canonical_step
        ),
        ("execution", "extra_tag"): nonempty_string(nullable=True, normalize=_folder_tag),
        ("execution", "plot_format"): nonempty_string(),
    }


__all__ = ["execution_defaults", "execution_rules"]
