"""Named adsorption-kinetics contracts and explicit registry.

Kinetic laws consume current and equilibrium loadings supplied by the caller;
they do not choose or call an equilibrium implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from pydac.utils.model_registry import KineticRegistryEntry


# Public name-based interface. Equilibrium is calculated before this call.
class KineticLaw:
    """Named-species kinetic contract returning physical loading rates."""

    species: tuple[str, ...] = ()

    def loading_rates_mol_per_kg_s(
        self,
        *,
        partial_pressures_Pa: Mapping[str, Any],
        temperature_K: Any,
        current_loadings_mol_per_kg: Mapping[str, Any],
        equilibrium_loadings_mol_per_kg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return physical loading rates in mol/kg/s by species name."""
        raise NotImplementedError


class Kinetics(KineticLaw):
    """Compatibility name retained for existing imports."""


class LDFRate(Kinetics):
    """Linear-driving-force law with the existing low-pressure damping."""

    def __init__(
        self,
        *,
        component: str,
        k_LDF: float,
    ) -> None:
        self.component = _species_name(component)
        self.k_LDF = float(k_LDF)
        if not np.isfinite(self.k_LDF) or self.k_LDF <= 0:
            raise ValueError(f"k_LDF for {self.component} must be > 0.")
        print(f"Selected {self.component} rate equation: LDF rate")

    @property
    def species(self) -> tuple[str, ...]:
        return (self.component,)

    def loading_rates_mol_per_kg_s(
        self,
        *,
        partial_pressures_Pa: Mapping[str, Any],
        temperature_K: Any,
        current_loadings_mol_per_kg: Mapping[str, Any],
        equilibrium_loadings_mol_per_kg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del temperature_K
        pressure = np.maximum(
            _required_value(
                partial_pressures_Pa, self.component, "partial pressure"
            ),
            1e-32,
        )
        current = np.maximum(
            _required_value(
                current_loadings_mol_per_kg, self.component, "current loading"
            ),
            1e-32,
        )
        equilibrium = np.clip(
            _required_value(
                equilibrium_loadings_mol_per_kg,
                self.component,
                "equilibrium loading",
            ),
            1e-32,
            None,
        )
        pressure_ratio = pressure / (1.0 + pressure)
        return {
            self.component: self.k_LDF
            * pressure_ratio
            * (equilibrium - current)
        }

def _species_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Species names must be non-empty strings.")
    return value.strip().upper()


def _build_ldf(
    species: str,
    parameters: Mapping[str, Any],
) -> KineticLaw:
    values = dict(parameters)
    unknown = sorted(set(values) - {"k_LDF_s_inv"})
    if unknown:
        raise ValueError(f"Unknown parameters for 'LDF': {unknown}.")
    if "k_LDF_s_inv" not in values:
        raise ValueError("LDF requires parameter 'k_LDF_s_inv'.")
    return LDFRate(
        component=species,
        k_LDF=values["k_LDF_s_inv"],
    )


# Every supported kinetic model is visible in this one mapping.
KINETIC_REGISTRY: Mapping[str, KineticRegistryEntry] = MappingProxyType(
    {
        "LDF": KineticRegistryEntry(
            build=_build_ldf,
            supported_species=None,
            required_parameters=("k_LDF_s_inv",),
            positive_parameters=("k_LDF_s_inv",),
        )
    }
)


# Laws registered at run time with pydac.register_kinetics: this Python session only.
_REGISTERED_KINETICS: dict[str, KineticRegistryEntry] = {}


def kinetic_registry() -> Mapping[str, KineticRegistryEntry]:
    """The built-in kinetic laws plus those registered in this session."""
    return MappingProxyType({**KINETIC_REGISTRY, **_REGISTERED_KINETICS})


def _required_value(values: Mapping[str, Any], species: str, label: str) -> Any:
    if not isinstance(values, Mapping):
        raise TypeError(f"Named {label}s must be supplied as a mapping.")
    try:
        return values[species]
    except KeyError as exc:
        raise ValueError(f"Missing {label} for species {species!r}.") from exc


__all__ = [
    "KINETIC_REGISTRY",
    "kinetic_registry",
    "KineticLaw",
    "Kinetics",
    "LDFRate",
]
