"""Named equilibrium-system and kinetic-model defaults."""

from __future__ import annotations

from typing import Any

from pydac.inputs.defaults._rules import Rule


def adsorption_defaults() -> dict[str, Any]:
    """Return the current wet DAC model declaration using registry names."""
    return {
        "equilibrium_systems": {
            "TothGABWADST": {
                "datasets": {"Toth": "Cleaned-Toth"},
                "parameters": {},
            }
        },
        "kinetics": {
            "CO2": {
                "model": "LDF",
                "parameters": {"k_LDF_s_inv": 0.003},
            },
            "H2O": {
                "model": "LDF",
                "parameters": {"k_LDF_s_inv": 8.3e-4},
            },
        },
    }


def adsorption_rules():
    equilibrium_selection = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "datasets": {
                "type": "object",
                "additionalProperties": {"type": "string", "minLength": 1},
            },
            "parameters": {
                "type": "object",
                "additionalProperties": True,
            },
        },
        "required": ["datasets", "parameters"],
    }
    kinetic_selection = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "model": {"type": "string", "minLength": 1},
            "parameters": {
                "type": "object",
                "additionalProperties": True,
            },
        },
        "required": ["model", "parameters"],
    }
    base = ("simulation", "adsorption")
    return {
        base + ("equilibrium_systems",): Rule(
            kind="object",
            opaque=True,
            schema={
                "type": "object",
                "minProperties": 1,
                "additionalProperties": equilibrium_selection,
            },
            description=(
                "Equilibrium systems keyed by explicit registry model name."
            ),
        ),
        base + ("kinetics",): Rule(
            kind="object",
            opaque=True,
            schema={
                "type": "object",
                "minProperties": 1,
                "additionalProperties": kinetic_selection,
            },
            description="One kinetic model per adsorbate, keyed by species name.",
        ),
    }


__all__ = ["adsorption_defaults", "adsorption_rules"]
