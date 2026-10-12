"""Operational gas-composition domain guard for long-running solves."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True, slots=True)
class CompositionDomainGuard:
    """Permissive terminal guard for a negative gas closure species.

    The closure concentration may undershoot zero by the larger of an
    absolute and a local ``c_tot``-relative allowance.  This is an operational
    failure guard only: it never clips, projects, or modifies the model state.
    """

    enabled: bool = True
    absolute_allowance_mol_m3: float = 2.0
    relative_allowance: float = 0.10

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("Composition-domain guard `enabled` must be bool.")
        values = (
            self.absolute_allowance_mol_m3,
            self.relative_allowance,
        )
        if any(
            isinstance(value, bool)
            or not isfinite(float(value))
            or float(value) < 0.0
            for value in values
        ):
            raise ValueError(
                "Composition-domain allowances must be finite and non-negative."
            )
        if self.enabled and not any(float(value) > 0.0 for value in values):
            raise ValueError(
                "An enabled composition-domain guard needs a positive allowance."
            )

    def as_dict(self) -> dict[str, bool | float]:
        return asdict(self)


def _closure_fields(y, *, model):
    layout = model.state_layout
    c_tot = (
        np.asarray(y)[layout.field("c_tot").state_slice].reshape(model.Nz, model.Nr)
        * float(model.c_tot_ref)
    )
    independent_sum = np.zeros_like(c_tot)
    independent = {}
    for species in layout.independent_gas_species:
        concentration = (
            np.asarray(y)[layout.field(f"c_{species}").state_slice]
            .reshape(model.Nz, model.Nr)
            * float(model.c_ref_by_species[species])
        )
        independent[species] = concentration
        independent_sum += concentration
    return c_tot, independent_sum, independent


def closure_domain_diagnostic(y, *, model, guard: CompositionDomainGuard) -> dict:
    """Return compact dimensional details at the least-valid closure cell."""
    c_tot, independent_sum, independent = _closure_fields(y, model=model)
    closure = c_tot - independent_sum
    allowance = np.maximum(
        float(guard.absolute_allowance_mol_m3),
        float(guard.relative_allowance) * c_tot,
    )
    margin = closure + allowance
    index = tuple(
        int(value)
        for value in np.unravel_index(np.nanargmin(margin), margin.shape)
    )
    z_index, r_index = index
    return {
        "closure_species": str(model.state_layout.closure_species),
        "index": [z_index, r_index],
        "z_m": float(model.z_c[z_index] * model.L_b),
        "r_m": float(model.r_c[r_index] * model.R_b),
        "c_tot_mol_m3": float(c_tot[index]),
        "independent_concentrations_mol_m3": {
            species: float(values[index]) for species, values in independent.items()
        },
        "independent_sum_mol_m3": float(independent_sum[index]),
        "closure_concentration_mol_m3": float(closure[index]),
        "allowed_undershoot_mol_m3": float(allowance[index]),
        "closure_margin_mol_m3": float(margin[index]),
    }


def build_composition_domain_event_spec(model, guard: CompositionDomainGuard):
    """Build a terminal failure event without modifying the physical RHS."""
    if not isinstance(guard, CompositionDomainGuard):
        raise TypeError("`guard` must be a CompositionDomainGuard.")
    if not guard.enabled:
        return None

    def metric(y):
        return closure_domain_diagnostic(y, model=model, guard=guard)[
            "closure_margin_mol_m3"
        ]

    return {
        "name": "composition_domain_violation",
        "metric_fn": metric,
        "diagnostic_fn": lambda y: closure_domain_diagnostic(
            y, model=model, guard=guard
        ),
        "threshold_nd": 0.0,
        "threshold_dim": 0.0,
        "direction": -1,
        "label": "Gas closure exceeded the permitted negative undershoot",
        "metric_label": "minimum closure-domain margin",
        "unit": "mol/m3",
        "ref": 1.0,
        "is_failure": True,
        "failure_reason": "state_domain_violation",
        "failure_status": "state_domain_violation",
        "failure_message": (
            "Gas closure concentration exceeded the configured permissive "
            "negative allowance."
        ),
        "metadata": guard.as_dict(),
    }


__all__ = [
    "CompositionDomainGuard",
    "build_composition_domain_event_spec",
    "closure_domain_diagnostic",
]
