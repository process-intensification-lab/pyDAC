"""Sorbent and particle defaults for the reference material."""

from __future__ import annotations

from typing import Any

from pydac.inputs.defaults._rules import Rule, fraction, positive_number, species_keys


def sorbent_defaults() -> dict[str, Any]:
    """Return Lewatit/sorbent particle properties in SI units.

    The heats of adsorption are a sorbent property keyed by species: they must
    cover `mixture.adsorbates`; a species the mixture does not adsorb is ignored. The solid heat capacity names one of the
    implemented models (`Low2023`, `constant`) with its exact parameters.
    `fixed_adsorbates` are non-exchanging loadings that only add heat capacity.
    """
    return {
        "conductivity_W_mK": 0.43,
        "density_kg_m3": 744.0,
        "particle_radius_m": 2.5e-4,
        "porosity": 0.338,
        "pore_radius_m": 2.85e-8,
        "heats_of_adsorption_J_mol": {"CO2": -81000.0, "H2O": -46000.0},
        "heat_capacity": {
            "model": "Low2023",
            "parameters": {
                "a_J_K_per_kg": -3.23e7,
                "b_J_per_kgK2": 2.27,
                "c_J_per_kgK": -994.0,
            },
        },
        "fixed_adsorbates": {},
    }


def sorbent_rules():
    """Return physical constraints stronger than the inferred float type."""
    rules = {
        ("simulation", "sorbent", name): positive_number()
        for name in (
            "conductivity_W_mK",
            "density_kg_m3",
            "particle_radius_m",
            "pore_radius_m",
        )
    }
    base = ("simulation", "sorbent")
    rules[base + ("porosity",)] = fraction(strict=True)
    rules[base + ("heats_of_adsorption_J_mol",)] = Rule(
        kind="object",
        opaque=True,
        normalize=species_keys,
        schema={
            "type": "object",
            "minProperties": 1,
            # The sign is checked at load for the adsorbates only: a heat for a
            # species that does not adsorb is ignored, whatever its value.
            "additionalProperties": {"type": "number"},
        },
        description=(
            "Heat of adsorption [J/mol] of each adsorbate, keyed by species name "
            "(any case: co2 = CO2); "
            "negative (exothermic); every species of mixture.adsorbates needs one, "
            "others are ignored."
        ),
    )
    rules[base + ("heat_capacity",)] = Rule(
        kind="object",
        opaque=True,
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "model": {"type": "string", "minLength": 1},
                "parameters": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
            },
            "required": ["model", "parameters"],
        },
        description=(
            "Solid heat capacity: `Low2023` (a_J_K_per_kg, b_J_per_kgK2, "
            "c_J_per_kgK; evaluated at the step feed temperature) or `constant` "
            "(value_J_kgK)."
        ),
    )
    rules[base + ("fixed_adsorbates",)] = Rule(
        kind="object",
        opaque=True,
        normalize=species_keys,
        schema={
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "loading_mol_per_kg": {"type": "number", "minimum": 0},
                    "heat_capacity_J_molK": {"type": "number", "exclusiveMinimum": 0},
                },
                "required": ["loading_mol_per_kg", "heat_capacity_J_molK"],
            },
        },
        description=(
            "Non-exchanging loadings keyed by species (any case: n2 = N2): they add heat capacity "
            "to the solid and take no part in the mass balance."
        ),
    )
    return rules


__all__ = [
    "sorbent_defaults",
    "sorbent_rules",
]
