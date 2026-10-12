"""Read persisted outlet observations without reconstructing hydraulic models."""

import numpy as np


def outlet_net_species_flows(solution):
    """Net per-species molar flow at the outlet face [mol/s], summed over rings.

    Signed: rings flowing inward subtract, as the bed balance sees them.
    Return None for older artifacts without exact face histories.
    """
    outlet = solution.get("outlet_face")
    if outlet is None:
        return None
    return {
        species: np.sum(np.asarray(values, dtype=float), axis=0)
        for species, values in outlet["species_molar_flow_mol_s"].items()
    }


def outlet_sensible_enthalpy_flow(solution):
    """Net sensible enthalpy flow at the outlet face [W], summed over rings.

    Relative to the model's own 298.15 K datum, so it is negative for a stream
    leaving below that datum. The sign carries no information about flow
    direction; the molar flow does.
    """
    outlet = solution.get("outlet_face")
    if outlet is None or "sensible_enthalpy_flow_W" not in outlet:
        return None
    return np.sum(np.asarray(outlet["sensible_enthalpy_flow_W"], dtype=float), axis=0)


def outlet_mole_fractions(solution):
    """Net molar-flow mixing cup; undefined for zero or net inward flow.

    Return None for older artifacts without exact face histories. Callers own
    and identify their legacy fallback rather than silently labelling it exact.
    """
    flows = outlet_net_species_flows(solution)
    if flows is None:
        return None
    total = sum(flows.values())
    return {
        species: np.divide(flow, total, out=np.full_like(total, np.nan), where=total > 0.0)
        for species, flow in flows.items()
    }
