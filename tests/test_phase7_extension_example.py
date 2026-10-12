"""Tiny registry extension exercised by the engineer extension recipe."""

from __future__ import annotations

from pydac.core.isotherm_models import EQUILIBRIUM_REGISTRY, EquilibriumSystem
from pydac.core.kinetic_models import KINETIC_REGISTRY, KineticLaw
from pydac.utils.flow_transforms import MW_H2O
from pydac.utils.model_registry import (
    EquilibriumChoice,
    EquilibriumRegistryEntry,
    KineticChoice,
    KineticRegistryEntry,
    build_equilibrium_system,
    build_kinetic_laws,
)
from pydac.core.species_catalog import get_species_properties


class ConstantEquilibrium(EquilibriumSystem):
    """Small test model returning one named equilibrium loading."""

    species = ("CO2",)

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def equilibrium_loadings(
        self,
        *,
        partial_pressures_Pa,
        temperature_K,
        current_loadings_mol_per_kg,
    ):
        del partial_pressures_Pa, temperature_K, current_loadings_mol_per_kg
        return {"CO2": self.value}


class DifferenceLaw(KineticLaw):
    """Small test model returning equilibrium minus current loading."""

    species = ("CO2",)

    def loading_rates_mol_per_kg_s(
        self,
        *,
        partial_pressures_Pa,
        temperature_K,
        current_loadings_mol_per_kg,
        equilibrium_loadings_mol_per_kg,
    ):
        del partial_pressures_Pa, temperature_K
        return {
            "CO2": equilibrium_loadings_mol_per_kg["CO2"]
            - current_loadings_mol_per_kg["CO2"]
        }


def test_tiny_named_registry_extension_requires_no_factory_edit():
    equilibrium_registry = dict(EQUILIBRIUM_REGISTRY)
    equilibrium_registry["Constant"] = EquilibriumRegistryEntry(
        adsorbates=("CO2",),
        required_gas_species=("CO2",),
        required_current_loadings=(),
        build=lambda datasets, parameters: ConstantEquilibrium(parameters["value"]),
        required_parameters=("value",),
    )
    equilibrium = build_equilibrium_system(
        mixture_species=("CO2", "N2"),
        adsorbates=("CO2",),
        choices=(
            EquilibriumChoice("Constant", parameters={"value": 2.5}),
        ),
        registry=equilibrium_registry,
    )

    kinetic_registry = dict(KINETIC_REGISTRY)
    kinetic_registry["Difference"] = KineticRegistryEntry(
        build=lambda species, parameters: DifferenceLaw(),
        supported_species=("CO2",),
    )
    kinetics = build_kinetic_laws(
        adsorbates=("CO2",),
        choices=(KineticChoice("CO2", "Difference"),),
        equilibrium_system=equilibrium,
        registry=kinetic_registry,
    )

    loadings = equilibrium.equilibrium_loadings(
        partial_pressures_Pa={"CO2": 1.0},
        temperature_K=300.0,
        current_loadings_mol_per_kg={"CO2": 0.5},
    )
    rates = kinetics["CO2"].loading_rates_mol_per_kg_s(
        partial_pressures_Pa={"CO2": 1.0},
        temperature_K=300.0,
        current_loadings_mol_per_kg={"CO2": 0.5},
        equilibrium_loadings_mol_per_kg=loadings,
    )

    assert loadings == {"CO2": 2.5}
    assert rates == {"CO2": 2.0}


def test_steam_flow_transform_uses_the_catalog_water_weight():
    assert MW_H2O == get_species_properties("H2O").molecular_weight_kg_per_mol
