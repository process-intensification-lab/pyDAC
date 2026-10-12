"""Immutable molecular data and gas-property correlations.

The catalog owns invariant species data used by the numerical model.  Values
that depend on the evaluation state, such as ``cp(T)``, are represented by
frozen correlation coefficients and evaluated through an explicit method;
they are not stored as mutable runtime values.

Only species with existing pyDAC v3 data are registered here.  Adding a new
species is intentionally local: define one ``SpeciesProperties`` record and
add it to ``SPECIES_CATALOG`` together with focused catalog tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping


def _finite(value: Real, field_name: str) -> float:
    """Validate and normalize one scalar catalog value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{field_name}` must be a real number.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"`{field_name}` must be finite.")
    return result


def _positive(value: Real, field_name: str) -> float:
    result = _finite(value, field_name)
    if result <= 0.0:
        raise ValueError(f"`{field_name}` must be greater than zero.")
    return result


@dataclass(frozen=True, slots=True)
class ShomateHeatCapacity:
    """Frozen coefficients for a molar Shomate heat-capacity correlation.

    The correlation uses ``t = T_K / 1000`` and returns ``J/mol/K``:

    ``cp = A + B*t + C*t**2 + D*t**3 + E/t**2``.
    """

    a_J_per_molK: float
    b_J_per_molK: float
    c_J_per_molK: float
    d_J_per_molK: float
    e_J_per_molK: float

    def __post_init__(self) -> None:
        for field_name in (
            "a_J_per_molK",
            "b_J_per_molK",
            "c_J_per_molK",
            "d_J_per_molK",
            "e_J_per_molK",
        ):
            _finite(getattr(self, field_name), field_name)

    def evaluate_J_per_molK(self, temperature_K: Real) -> float:
        """Evaluate the correlation at a positive absolute temperature."""
        temperature = _positive(temperature_K, "temperature_K")
        t = temperature / 1000.0
        return (
            self.a_J_per_molK
            + self.b_J_per_molK * t
            + self.c_J_per_molK * t**2
            + self.d_J_per_molK * t**3
            + self.e_J_per_molK / t**2
        )


@dataclass(frozen=True, slots=True)
class SpeciesProperties:
    """Immutable molecular constants and pure-gas property data.

    All dimensional field names state their units explicitly.  The thermal
    conductivity is the existing reference-state value used by v3; it is not
    presented as a temperature-independent universal property.
    """

    name: str
    molecular_weight_kg_per_mol: float
    lennard_jones_sigma_m: float
    lennard_jones_epsilon_over_k_K: float
    fuller_diffusion_volume_m3_per_mol: float
    thermal_conductivity_reference_W_per_mK: float
    thermal_conductivity_reference_temperature_K: float
    heat_capacity_shomate: ShomateHeatCapacity

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("`name` must be a non-empty species name.")
        for field_name in (
            "molecular_weight_kg_per_mol",
            "lennard_jones_sigma_m",
            "lennard_jones_epsilon_over_k_K",
            "fuller_diffusion_volume_m3_per_mol",
            "thermal_conductivity_reference_W_per_mK",
            "thermal_conductivity_reference_temperature_K",
        ):
            _positive(getattr(self, field_name), field_name)
        if not isinstance(self.heat_capacity_shomate, ShomateHeatCapacity):
            raise TypeError("`heat_capacity_shomate` must be ShomateHeatCapacity.")


SPECIES_CATALOG: Mapping[str, SpeciesProperties] = MappingProxyType(
    {
        "CO2": SpeciesProperties(
            name="CO2",
            molecular_weight_kg_per_mol=44.009 / 1000.0,
            lennard_jones_sigma_m=3.941e-10,
            lennard_jones_epsilon_over_k_K=195.2,
            fuller_diffusion_volume_m3_per_mol=26.9e-6,
            thermal_conductivity_reference_W_per_mK=0.0166,
            thermal_conductivity_reference_temperature_K=298.0,
            heat_capacity_shomate=ShomateHeatCapacity(
                a_J_per_molK=24.99735,
                b_J_per_molK=55.18696,
                c_J_per_molK=-33.69137,
                d_J_per_molK=7.948387,
                e_J_per_molK=-0.136638,
            ),
        ),
        "H2O": SpeciesProperties(
            name="H2O",
            molecular_weight_kg_per_mol=18.015 / 1000.0,
            lennard_jones_sigma_m=2.641e-10,
            lennard_jones_epsilon_over_k_K=809.1,
            fuller_diffusion_volume_m3_per_mol=13.1e-6,
            thermal_conductivity_reference_W_per_mK=0.02457,
            thermal_conductivity_reference_temperature_K=398.0,
            heat_capacity_shomate=ShomateHeatCapacity(
                a_J_per_molK=30.09200,
                b_J_per_molK=6.832514,
                c_J_per_molK=6.793435,
                d_J_per_molK=-2.534480,
                e_J_per_molK=0.082139,
            ),
        ),
        "O2": SpeciesProperties(
            name="O2",
            molecular_weight_kg_per_mol=31.999 / 1000.0,
            lennard_jones_sigma_m=3.467e-10,
            lennard_jones_epsilon_over_k_K=106.7,
            fuller_diffusion_volume_m3_per_mol=16.3e-6,
            thermal_conductivity_reference_W_per_mK=0.026,
            thermal_conductivity_reference_temperature_K=298.0,
            heat_capacity_shomate=ShomateHeatCapacity(
                a_J_per_molK=31.32230,
                b_J_per_molK=-20.23531,
                c_J_per_molK=57.86644,
                d_J_per_molK=-36.50616,
                e_J_per_molK=-0.007374,
            ),
        ),
        "N2": SpeciesProperties(
            name="N2",
            molecular_weight_kg_per_mol=28.014 / 1000.0,
            lennard_jones_sigma_m=3.798e-10,
            lennard_jones_epsilon_over_k_K=71.4,
            fuller_diffusion_volume_m3_per_mol=18.5e-6,
            thermal_conductivity_reference_W_per_mK=0.02583,
            thermal_conductivity_reference_temperature_K=298.0,
            heat_capacity_shomate=ShomateHeatCapacity(
                a_J_per_molK=28.98641,
                b_J_per_molK=1.853978,
                c_J_per_molK=-9.647459,
                d_J_per_molK=16.63537,
                e_J_per_molK=0.000117,
            ),
        ),
    }
)


def available_species() -> tuple[str, ...]:
    """Return registered species names in deterministic order."""
    return tuple(sorted(SPECIES_CATALOG))


def get_species_properties(name: str) -> SpeciesProperties:
    """Return a registered species record with a clear unknown-name error."""
    if not isinstance(name, str):
        raise TypeError("Species name must be a string.")
    normalized = name.strip().upper()
    try:
        return SPECIES_CATALOG[normalized]
    except KeyError as exc:
        names = ", ".join(available_species())
        raise ValueError(
            f"Unknown species {name!r}. Available catalog species: {names}."
        ) from exc


__all__ = [
    "SPECIES_CATALOG",
    "ShomateHeatCapacity",
    "SpeciesProperties",
    "available_species",
    "get_species_properties",
]
