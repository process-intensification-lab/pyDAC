from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from pydac.core.column_model_factory import ColumnModelFactory
from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig


CASE = "tests/cases/v3_smoke_startup.json"
CO2_N2_SETTINGS = {
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
}


def _build_model(settings):
    config = load_experiment(CASE, settings=settings)
    cycle = CycleConfig.from_experiment(config)
    with contextlib.redirect_stdout(io.StringIO()):
        return ColumnModelFactory(cycle).build_column_model(
            step="startup",
            solver_options=cycle.solver_for_step(
                use_sparsity=False,
                compute_numerical_sparsity=False,
            ),
        )


def _initial_state(model):
    state = np.zeros(model.state_layout.state_size)
    state[: model.state_layout.pde_size] = model.get_initial_state()
    return state


@pytest.mark.parametrize(
    "settings",
    [
        {"grid.nz": 4, "grid.nr": 2},
        {
            "grid.nz": 4,
            "grid.nr": 2,
            "mixture.closure_species": "O2",
        },
        CO2_N2_SETTINGS,
    ],
    ids=("default_n2_closure", "o2_closure", "co2_n2"),
)
def test_extra_energy_source_matches_full_species_accumulation(settings):
    model = _build_model(settings)
    state = _initial_state(model)
    temperature_slice = model.state_field_by_name["T_bed"].state_slice
    state[temperature_slice] = model.T_enthalpy_ref_nd + 0.2
    derivative = model.pde_system_kpis(0.0, state).copy()

    dc_independent_dtau = np.column_stack(
        [
            derivative[field.state_slice]
            for field in model.gas_state_fields
        ]
    )
    dq_dtau = np.column_stack(
        [
            derivative[field.state_slice]
            for field in model.loading_state_fields
        ]
    )
    dc_tot_dtau = derivative[
        model.state_field_by_name["c_tot"].state_slice
    ]

    closure = model.gas_index[model.closure_species]
    independent = model.independent_gas_indices
    dc_closure_dtau = (
        model.c_tot_ref * dc_tot_dtau
        - dc_independent_dtau @ model.c_ref[independent]
    ) / model.c_ref[closure]

    dc_all_dtau = np.empty((model.Nz * model.Nr, model.Nc_gas))
    dc_all_dtau[:, independent] = dc_independent_dtau
    dc_all_dtau[:, closure] = dc_closure_dtau
    gas_accumulation = dc_all_dtau @ model.acc_g
    adsorbed_accumulation = dq_dtau @ model.acc_ads
    theta_bed = state[temperature_slice] - model.T_enthalpy_ref_nd
    expected_source = -theta_bed * (
        gas_accumulation + adsorbed_accumulation
    )

    np.testing.assert_allclose(
        model._dc_closure_dtau_buf,
        dc_closure_dtau,
        rtol=2e-15,
        atol=1e-68,
    )
    np.testing.assert_allclose(
        model._source_extra_buf,
        expected_source,
        rtol=2e-15,
        atol=1e-13,
    )


def test_fixed_and_pressure_driven_inlet_enthalpy_fluxes_use_298K_datum():
    model = _build_model(
        {
            "grid.nz": 4,
            "grid.nr": 2,
            "steps.startup.thermal.feed_temperature_K": 373.15,
        }
    )
    theta_feed = (
        model.T_feed - model.T_enthalpy_ref_K
    ) / model.T_ref
    feed_sensible_energy = np.sum(model.c_feed * model.cp_g) * theta_feed
    expected_fixed_flux = feed_sensible_energy * (
        model.u_inlet / model.u_ref
    )
    np.testing.assert_allclose(
        model.flux_in_T,
        expected_fixed_flux,
        rtol=2e-15,
        atol=0.0,
    )

    model.u_ax_single[0] = 0.25
    model.update_inlet_fluxes()
    expected_pressure_flux = feed_sensible_energy * 0.25
    np.testing.assert_allclose(
        model.flux_in_T,
        expected_pressure_flux,
        rtol=2e-15,
        atol=0.0,
    )


def test_298K_feed_has_zero_sensible_enthalpy_flux():
    model = _build_model({"grid.nz": 4, "grid.nr": 1})

    assert model.T_feed == model.T_enthalpy_ref_K
    np.testing.assert_array_equal(model.flux_in_T, 0.0)


def test_closure_species_requires_a_positive_reference_scale():
    settings = dict(CO2_N2_SETTINGS)
    settings["reference.co2_mole_fraction"] = 1.0

    with pytest.raises(
        ValueError,
        match="including the closure species",
    ):
        _build_model(settings)
