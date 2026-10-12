from __future__ import annotations

import json

import numpy as np
import pytest

from pydac.core.column_model_factory import ColumnModelFactory
from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig


LEGACY_CASE = "tests/cases/v3_smoke_startup.json"


def _build_startup(*, source=LEGACY_CASE, settings=None):
    config = load_experiment(source, settings=settings)
    cycle = CycleConfig.from_experiment(config)
    return ColumnModelFactory(cycle).build_column_model(
        step="startup",
        solver_options=cycle.solver_for_step(
            use_sparsity=False,
            compute_numerical_sparsity=False,
        ),
    )


def _warm_state(model):
    return {
        "Nz": model.Nz,
        "Nr": model.Nr,
        "ci": model.ci.copy(),
        "qi": model.qi.copy(),
        "c_tot": model.c_tot[..., 0].copy(),
        "T": model.T[..., 0].copy(),
        "T_wall": model.Tw[:, 0, 0].copy(),
    }


@pytest.mark.parametrize("nr", [1, 2])
def test_default_initialization_uses_the_generated_layout_in_1d_and_2d(nr):
    model = _build_startup(settings={"grid.nz": 4, "grid.nr": nr})

    assert model.state_layout.mixture.species == ("CO2", "H2O", "O2", "N2")
    assert model.state_layout.closure_species == "N2"
    assert model.state_layout.independent_gas_species == ("CO2", "H2O", "O2")
    assert model.state_layout.adsorbates == ("CO2", "H2O")
    assert model.ci.shape == (4, nr, 3)
    assert model.qi.shape == (4, nr, 2)
    assert model.T.shape == (4, nr, 1)
    assert model.Tw.shape == (4, 1, 1)

    np.testing.assert_array_equal(
        model.MW,
        [model.MW_CO2, model.MW_H2O, model.MW_O2, model.MW_N2],
    )
    np.testing.assert_array_equal(
        model.c_ref,
        [model.c_CO2_ref, model.c_H2O_ref, model.c_O2_ref, model.c_N2_ref],
    )
    np.testing.assert_array_equal(
        model.q_ref,
        [model.q_eq_CO2_ref, model.q_eq_H2O_ref],
    )


def test_non_nitrogen_closure_is_reconstructed_by_name():
    model = _build_startup(
        settings={
            "grid.nz": 4,
            "grid.nr": 1,
            "mixture.closure_species": "O2",
        }
    )

    assert model.closure_species == "O2"
    assert model.independent_species == ("CO2", "H2O", "N2")
    assert tuple(field.species for field in model.state_layout.fields[:3]) == (
        "CO2",
        "H2O",
        "N2",
    )

    c_total = model.c_tot[..., 0]
    c_o2_from_closure = c_total - np.sum(model.ci, axis=-1)
    expected_o2 = model.y_initial_by_species["O2"] * model.c_tot_initial
    np.testing.assert_allclose(c_o2_from_closure, expected_o2, rtol=0.0, atol=1e-14)

    expected_fluxes = np.take(
        model.flux_in_ci_full,
        model.independent_gas_indices,
        axis=-1,
    )
    np.testing.assert_array_equal(model.flux_in_ci, expected_fluxes)
    assert model.runtime_properties["state_layout"]["mixture"]["closure"] == "O2"


def test_co2_n2_model_initializes_through_the_same_property_path():
    model = _build_startup(
        source=None,
        settings={
            "grid.nz": 4,
            "grid.nr": 1,
            "mixture": {
                "species": ["CO2", "N2"],
                "closure_species": "N2",
                "adsorbates": ["CO2"],
            },
            "adsorption": {
                "equilibrium_systems": {
                    "Toth": {
                        "datasets": {"Toth": "Cleaned-Toth"},
                        "parameters": {},
                    }
                },
                "kinetics": {
                    "CO2": {
                        "model": "LDF",
                        "parameters": {"k_LDF_s_inv": 0.003},
                    }
                },
            },
        },
    )

    assert model.gas_species == ("CO2", "N2")
    assert model.independent_species == ("CO2",)
    assert model.adsorbates == ("CO2",)
    assert model.ci.shape == (4, 1, 1)
    assert model.qi.shape == (4, 1, 1)
    assert model.MW.shape == (2,)
    assert model.D_binary.shape == (2, 2)
    assert model.D_m.shape == (2,)
    assert model.c_ref.shape == (2,)
    assert model.q_ref.shape == (1,)
    assert not hasattr(model, "MW_H2O")
    np.testing.assert_allclose(sum(model.y_feed_by_species.values()), 1.0)
    np.testing.assert_allclose(sum(model.y_initial_by_species.values()), 1.0)


def test_warm_start_shapes_are_validated_after_layout_generation():
    config = load_experiment(
        LEGACY_CASE,
        settings={"grid.nz": 4, "grid.nr": 2},
    )
    cycle = CycleConfig.from_experiment(config)
    factory = ColumnModelFactory(cycle)
    options = cycle.solver_for_step(
        use_sparsity=False,
        compute_numerical_sparsity=False,
    )
    startup = factory.build_column_model(step="startup", solver_options=options)
    adsorption = factory.build_column_model(
        step="ads",
        initial_state=_warm_state(startup),
        solver_options=options,
    )

    assert adsorption.ci.shape == (4, 2, 3)
    assert adsorption.qi.shape == (4, 2, 2)
    assert adsorption.state_layout.to_dict()["grid"] == {"nz": 4, "nr": 2}
    np.testing.assert_allclose(sum(adsorption.y_initial_by_species.values()), 1.0)


def test_every_current_cycle_step_builds_the_same_structured_property_contract():
    config = load_experiment(LEGACY_CASE)
    cycle = CycleConfig.from_experiment(config)
    factory = ColumnModelFactory(cycle)
    options = cycle.solver_for_step(
        use_sparsity=False,
        compute_numerical_sparsity=False,
    )
    startup = factory.build_column_model(step="startup", solver_options=options)
    warm_state = _warm_state(startup)

    for step in cycle.steps.names:
        model = factory.build_column_model(
            step=step,
            initial_state=None if step == "startup" else warm_state,
            solver_options=options,
        )
        assert model.runtime_properties["step"] == step
        assert np.isfinite(model.rho_g_mix)
        json.dumps(model.runtime_properties, allow_nan=False)


def test_runtime_property_summary_is_complete_and_strict_json():
    model = _build_startup(settings={"grid.nz": 4, "grid.nr": 1})
    report = model.runtime_properties

    json.dumps(report, allow_nan=False)
    assert report["evaluation"]["gas_composition_source"] == "feed"
    assert report["evaluation"]["dispersion_composition_source"] == "feed"
    assert set(report["species"]) == set(model.gas_species)
    assert len(report["binary_diffusivity_m2_s"]) == 6
    assert set(report["mixture"]) == {
        "density_kg_m3",
        "viscosity_Pa_s",
        "heat_capacity_J_molK",
        "thermal_conductivity_W_mK",
    }
    assert set(report["dimensionless"]) == {
        "reynolds_number",
        "prandtl_number",
    }
    assert set(report["inlet"]["species_mass_flow_kg_s"]) == set(model.gas_species)
    assert report["bed_architecture"]["resolved"]["step_property_context"]

    for species in model.gas_species:
        evaluated = report["species"][species]
        assert set(evaluated) == {
            "heat_capacity_J_molK",
            "viscosity_Pa_s",
            "mixture_diffusivity_m2_s",
            "schmidt_number",
            "sherwood_number",
        }

    assert report["mixture"]["density_kg_m3"] == model.rho_g_mix
    assert report["mixture"]["viscosity_Pa_s"] == model.mu_g_mix
    assert report["mixture"]["heat_capacity_J_molK"] == model.cp_g_mix
    assert (
        report["mixture"]["thermal_conductivity_W_mK"]
        == model.lambda_g_mix
    )
