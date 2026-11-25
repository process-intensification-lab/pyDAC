"""Grid and CVODE solver defaults for a reference pyDAC experiment."""

from __future__ import annotations

from typing import Any

from pydac.inputs.defaults._rules import Rule, positive_integer, positive_number
from pydac.utils.output_schedule import SCHEDULE_DEFAULTS


def grid_defaults() -> dict[str, Any]:
    """Return the spatial grid and output time-grid defaults."""
    return {
        "nz": 50,
        "nr": 25,
        "nt": 1000,
        "nr_plot_2d": 50,
        "output_schedule": dict(SCHEDULE_DEFAULTS),
    }


def solver_defaults() -> dict[str, Any]:
    """Return CVODE and integrated-scalar tolerance defaults."""
    return {
        # Selected after the tau=8 radial-hydraulics tolerance ladder.
        "rtol": 1e-8,
        "atol": 1e-10,
        "atol_scalar_mass": 1e-8,
        "atol_scalar_energy": 1e-6,
    }


def solver_rules():
    """Return stronger constraints not implied by the numeric defaults."""
    rules = {
        ("simulation", "grid", name): positive_integer()
        for name in grid_defaults() if name != "output_schedule"
    }
    for name in SCHEDULE_DEFAULTS:
        rules[("simulation", "grid", "output_schedule", name)] = (
            Rule(kind="integer", minimum=2, nullable=True) if name == "dense_start_points"
            else positive_number(nullable=True))
    rules.update(
        {
            ("simulation", "solver", name): positive_number()
            for name in solver_defaults()
        }
    )
    return rules


__all__ = [
    "grid_defaults",
    "solver_defaults",
    "solver_rules",
]
