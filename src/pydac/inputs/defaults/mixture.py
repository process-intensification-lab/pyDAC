"""Named gas-mixture defaults used by the generated state layout."""

from __future__ import annotations

from typing import Any

from pydac.inputs.defaults._rules import Rule, nonempty_string


def mixture_defaults() -> dict[str, Any]:
    """Return the current DAC mixture using physical species names."""
    return {
        "species": ["CO2", "H2O", "O2", "N2"],
        "closure_species": "N2",
        "adsorbates": ["CO2", "H2O"],
    }


def mixture_rules():
    base = ("simulation", "mixture")
    return {
        base + ("species",): Rule(
            kind="array",
            min_items=2,
            item_kind="string",
            schema={
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 2,
                "uniqueItems": True,
            },
            description=(
                "Ordered gas species. Molecular constants come from the species catalog."
            ),
        ),
        base + ("closure_species",): nonempty_string(
            description="Species reconstructed by concentration closure."
        ),
        base + ("adsorbates",): Rule(
            kind="array",
            min_items=1,
            item_kind="string",
            schema={
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "uniqueItems": True,
            },
            description="Ordered adsorbing subset of the gas mixture.",
        ),
    }


__all__ = ["mixture_defaults", "mixture_rules"]
