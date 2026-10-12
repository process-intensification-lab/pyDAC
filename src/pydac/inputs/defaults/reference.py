"""Reference state defaults used for nondimensionalization."""

from __future__ import annotations

from typing import Any

from pydac.inputs.defaults._rules import fraction, positive_number


def reference_defaults() -> dict[str, Any]:
    """Return the reference thermodynamic and flow state."""
    return {
        "temperature_K": 298.15,
        "pressure_Pa": 101325.0,
        "relative_humidity": 0.5,
        "co2_mole_fraction": 4e-4,
        "velocity_m_s": 0.1,
    }


def reference_rules():
    rules = {
        ("simulation", "reference", name): positive_number()
        for name in ("temperature_K", "pressure_Pa", "velocity_m_s")
    }
    for name in ("relative_humidity", "co2_mole_fraction"):
        rules[("simulation", "reference", name)] = fraction()
    return rules


__all__ = ["reference_defaults", "reference_rules"]
