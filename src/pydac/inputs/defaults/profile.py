"""Composer for the complete editable pyDAC reference profile."""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping

from pydac.inputs.defaults._rules import Rule

from pydac.inputs.defaults import adsorption, column, execution, mixture, particle, reference, solver, steps


def default_profile() -> dict[str, Any]:
    """Return one complete, fresh reference experiment document."""
    column_sections = column.column_defaults()
    return {
        "schema_version": 8,
        "simulation": {
            "grid": solver.grid_defaults(),
            "solver": solver.solver_defaults(),
            "bed": column_sections["bed"],
            "geometry": column_sections["geometry"],
            "wall": column_sections["wall"],
            "sorbent": particle.sorbent_defaults(),
            "mixture": mixture.mixture_defaults(),
            "adsorption": adsorption.adsorption_defaults(),
            "reference": reference.reference_defaults(),
            "cycle": steps.cycle_defaults(),
        },
        "execution": execution.execution_defaults(),
    }


@lru_cache(maxsize=1)
def default_rules() -> Mapping[tuple[str, ...], Rule]:
    """Compose stronger path rules colocated with the domain defaults."""
    rules = {}
    for domain_rules in (
        solver.solver_rules,
        column.column_rules,
        particle.sorbent_rules,
        mixture.mixture_rules,
        adsorption.adsorption_rules,
        reference.reference_rules,
        steps.step_rules,
        execution.execution_rules,
    ):
        declared = domain_rules()
        overlap = set(rules) & set(declared)
        if overlap:
            raise RuntimeError(f"Default rules are declared twice: {sorted(overlap)}")
        rules.update(declared)
    return MappingProxyType(rules)


__all__ = ["default_profile", "default_rules"]
