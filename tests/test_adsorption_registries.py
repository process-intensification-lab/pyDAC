from __future__ import annotations

import numpy as np
import pytest

from pydac.core.column_model_factory import ColumnModelFactory
from pydac.config import ConfigurationError, load_experiment
from pydac.config.cycle import CycleConfig
from pydac.core.isotherm_models import (
    EQUILIBRIUM_REGISTRY,
    EquilibriumSystem,
)
from pydac.core.kinetic_models import (
    KINETIC_REGISTRY,
    KineticLaw,
    LDFRate,
)
from pydac.utils.model_registry import (
    EquilibriumChoice,
    EquilibriumRegistryEntry,
    KineticChoice,
    KineticRegistryEntry,
    build_equilibrium_system,
    build_kinetic_laws,
)


MIXTURE = ("CO2", "H2O", "O2", "N2")
ADSORBATES = ("CO2", "H2O")


def build_current_equilibrium(*, wet: bool):
    selections = (
        (EquilibriumChoice("TothGABWADST"),)
        if wet
        else (EquilibriumChoice("Toth"), EquilibriumChoice("GAB"))
    )
    return build_equilibrium_system(
        mixture_species=MIXTURE,
        adsorbates=ADSORBATES,
        choices=selections,
        registry=EQUILIBRIUM_REGISTRY,
    )


def test_factory_creates_current_models_through_named_registries():
    experiment = load_experiment("tests/cases/v3_smoke_startup.json")
    bundle = ColumnModelFactory(CycleConfig.from_experiment(experiment)).components

    assert bundle.equilibrium_system.model_names == ("TothGABWADST",)
    assert bundle.equilibrium_system.species == ADSORBATES
    assert tuple(bundle.kinetics) == ADSORBATES
    assert bundle.equilibrium_model("co2") is bundle.equilibrium_system.component_models["CO2"]
    assert bundle.kinetic_law("h2o") is bundle.kinetics["H2O"]
    assert all(isinstance(law, LDFRate) for law in bundle.kinetics.values())


def test_factory_builds_named_independent_equilibrium_systems():
    experiment = load_experiment(
        "tests/cases/v3_smoke_startup.json",
        settings={
            "adsorption.equilibrium_systems": {
                "Toth": {
                    "datasets": {"Toth": "Cleaned-Toth"},
                    "parameters": {},
                },
                "GAB": {"datasets": {}, "parameters": {}},
            }
        },
    )
    bundle = ColumnModelFactory(CycleConfig.from_experiment(experiment)).components

    assert bundle.equilibrium_system.model_names == ("Toth", "GAB")
    assert bundle.equilibrium_system.species == ADSORBATES


def test_loader_rejects_unknown_kinetic_model_before_factory_construction():
    kinetics = {
        "CO2": {"model": "UnknownRate", "parameters": {}},
        "H2O": {"model": "LDF", "parameters": {"k_LDF_s_inv": 8.3e-4}},
    }
    with pytest.raises(ConfigurationError, match="Unknown model 'UnknownRate'"):
        load_experiment(
            "tests/cases/v3_smoke_startup.json",
            settings={"adsorption.kinetics": kinetics},
        )


def test_independent_and_coupled_systems_share_one_named_call():
    pressures = {
        "CO2": np.array([40.0, 400.0]),
        "H2O": np.array([500.0, 1000.0]),
    }
    temperature = np.array([298.15, 308.15])

    independent = build_current_equilibrium(wet=False)
    independent_result = independent.equilibrium_loadings(
        partial_pressures_Pa=pressures,
        temperature_K=temperature,
        current_loadings_mol_per_kg={},
    )
    np.testing.assert_array_equal(
        independent_result["CO2"],
        independent.component_models["CO2"].dry_isotherm(
            pressures["CO2"], temperature
        )[0],
    )
    np.testing.assert_array_equal(
        independent_result["H2O"],
        independent.component_models["H2O"].isotherm(
            pressures["H2O"], temperature
        )[0],
    )

    current = {"CO2": np.zeros(2), "H2O": np.array([0.2, 1.0])}
    coupled = build_current_equilibrium(wet=True)
    coupled_result = coupled.equilibrium_loadings(
        partial_pressures_Pa=pressures,
        temperature_K=temperature,
        current_loadings_mol_per_kg=current,
    )
    assert tuple(coupled_result) == ADSORBATES
    np.testing.assert_array_equal(
        coupled_result["CO2"],
        coupled.component_models["CO2"].co_isotherm_wadst(
            pressures["CO2"], temperature, current["H2O"]
        )[0],
    )
    np.testing.assert_array_equal(
        coupled_result["H2O"],
        coupled.component_models["H2O"].isotherm(
            pressures["H2O"], temperature
        )[0],
    )


def test_wet_system_requires_named_current_water_loading():
    coupled = build_current_equilibrium(wet=True)
    with pytest.raises(ValueError, match="current loading.*H2O"):
        coupled.equilibrium_loadings(
            partial_pressures_Pa={"CO2": 40.0, "H2O": 500.0},
            temperature_K=298.15,
            current_loadings_mol_per_kg={"CO2": 0.1},
        )


def test_ldf_contract_consumes_precomputed_named_loadings():
    law = LDFRate(component="CO2", k_LDF=0.003)
    pressure = np.array([0.0, 40.0])
    current = np.array([0.1, 0.2])
    equilibrium = np.array([0.4, 0.5])

    rates = law.loading_rates_mol_per_kg_s(
        partial_pressures_Pa={"CO2": pressure},
        temperature_K=298.15,
        current_loadings_mol_per_kg={"CO2": current},
        equilibrium_loadings_mol_per_kg={"CO2": equilibrium},
    )
    safe_pressure = np.maximum(pressure, 1e-32)
    expected = 0.003 * safe_pressure / (1.0 + safe_pressure) * (
        equilibrium - current
    )
    np.testing.assert_array_equal(rates["CO2"], expected)


def test_ldf_has_no_equilibrium_or_dimensionless_rhs_adapter():
    law = LDFRate(component="CO2", k_LDF=0.003)
    assert not hasattr(law, "thermo")
    assert not hasattr(law, "ads_rate")


@pytest.mark.parametrize(
    ("selections", "match"),
    [
        (
            (EquilibriumChoice("Toth"), EquilibriumChoice("Toth")),
            "exactly once",
        ),
        ((EquilibriumChoice("Toth"),), "exactly once"),
        (
            (EquilibriumChoice("GAB"), EquilibriumChoice("Toth")),
            "declared order",
        ),
    ],
)
def test_equilibrium_builder_rejects_duplicate_or_missing_coverage(
    selections, match
):
    with pytest.raises(ValueError, match=match):
        build_equilibrium_system(
            mixture_species=MIXTURE,
            adsorbates=ADSORBATES,
            choices=selections,
            registry=EQUILIBRIUM_REGISTRY,
        )


def test_registry_requirements_fail_before_model_use():
    with pytest.raises(ValueError, match="absent from the mixture.*H2O"):
        build_equilibrium_system(
            mixture_species=("CO2", "N2"),
            adsorbates=ADSORBATES,
            choices=(EquilibriumChoice("TothGABWADST"),),
            registry=EQUILIBRIUM_REGISTRY,
        )

    equilibrium = build_current_equilibrium(wet=False)
    with pytest.raises(ValueError, match="Kinetic models.*exactly once"):
        build_kinetic_laws(
            adsorbates=ADSORBATES,
            choices=(
                KineticChoice("CO2", "LDF", {"k_LDF_s_inv": 0.003}),
            ),
            equilibrium_system=equilibrium,
            registry=KINETIC_REGISTRY,
        )


class ConstantEquilibrium(EquilibriumSystem):
    def __init__(self, value: float) -> None:
        self.value = float(value)

    @property
    def species(self) -> tuple[str, ...]:
        return ("CO2",)

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
    @property
    def species(self) -> tuple[str, ...]:
        return ("CO2",)

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


def test_test_only_models_are_added_by_registry_entry_without_factory_edits():
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
        choices=(EquilibriumChoice("Constant", {"value": 2.5}),),
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

    equilibrium_values = equilibrium.equilibrium_loadings(
        partial_pressures_Pa={"CO2": 1.0},
        temperature_K=300.0,
        current_loadings_mol_per_kg={"CO2": 0.5},
    )
    rates = kinetics["CO2"].loading_rates_mol_per_kg_s(
        partial_pressures_Pa={"CO2": 1.0},
        temperature_K=300.0,
        current_loadings_mol_per_kg={"CO2": 0.5},
        equilibrium_loadings_mol_per_kg=equilibrium_values,
    )
    assert equilibrium_values == {"CO2": 2.5}
    assert rates == {"CO2": 2.0}


def test_builtin_registries_and_selection_parameters_are_immutable():
    with pytest.raises(TypeError):
        EQUILIBRIUM_REGISTRY["new"] = EQUILIBRIUM_REGISTRY["Toth"]
    with pytest.raises(TypeError):
        KINETIC_REGISTRY["new"] = KINETIC_REGISTRY["LDF"]

    selection = EquilibriumChoice(
        "Toth", datasets={"Toth": "Cleaned-Toth"}
    )
    with pytest.raises(TypeError):
        selection.datasets["Toth"] = "other"
