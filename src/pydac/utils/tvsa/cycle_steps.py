"""TVSA standalone and cyclic step-sequence builders."""

from __future__ import annotations

from pydac.utils.tvsa.steps import normalize_step_token


# The vanilla TVSA cycle: wall pre-heating followed directly by steam-assisted
# desorption. A steam purge is an optional extra stage, not part of the default.
DEFAULT_STEP_MODE_SEQUENCE = (
    "startup", "blow", "heat", "s_des", "cool", "press", "ads"
)
DEFAULT_CYCLIC_SEQUENCE = (
    "ads", "blow", "heat", "s_des", "cool", "press"
)


def build_cycle_steps(cycle_steps=None):
    """Build a standalone sequence with mandatory startup first."""
    tokens = DEFAULT_STEP_MODE_SEQUENCE if cycle_steps is None else list(cycle_steps)
    resolved = [normalize_step_token(token) for token in tokens]
    return ["startup"] + [step for step in resolved if step != "startup"]


def build_repeating_cycle_steps(cycle_steps=None):
    """Build and validate the repeating sequence used for CSS cycles."""
    tokens = DEFAULT_CYCLIC_SEQUENCE if cycle_steps is None else list(cycle_steps)
    if not tokens:
        raise ValueError("Cyclic `cycle_steps` must contain at least one explicit step.")

    resolved = [normalize_step_token(token) for token in tokens]
    if "startup" in resolved:
        raise ValueError(
            "Cyclic `cycle_steps` must describe the repeating cycle only. "
            "Do not include `startup`; cycle 1 replaces the leading adsorption "
            "step with startup automatically."
        )
    if resolved[0] != "ads":
        raise ValueError(
            "Cyclic `cycle_steps` must start with `ads`/`adsorption` so cycle 1 "
            "can replace that first adsorption step with startup."
        )
    return resolved


def build_cyclic_cycle_steps(cycle_idx, cycle_steps=None):
    """Build the sequence executed by one concrete CSS cycle."""
    resolved = build_repeating_cycle_steps(cycle_steps=cycle_steps)
    return ["startup"] + resolved[1:] if cycle_idx == 1 else resolved


__all__ = [
    "DEFAULT_CYCLIC_SEQUENCE",
    "DEFAULT_STEP_MODE_SEQUENCE",
    "build_cycle_steps",
    "build_cyclic_cycle_steps",
    "build_repeating_cycle_steps",
]
