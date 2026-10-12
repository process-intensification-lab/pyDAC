from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from pydac.core.column_model_factory import ColumnModelFactory
from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.core.solver import Solver
from pydac.core.sparsity_patterns.analytical_sparsity import (
    generate_sparsity_pattern,
    plot_sparsity,
)
from pydac.utils.solver.events import max_bed_species_mole_fraction
from pydac.utils.solver.jacobian import finite_difference_jacobian
from pydac.utils.solver.sparsity import (
    prepare_cvode_solver_sparsity,
    try_get_solver_sparsity,
)


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


def build_model(*, settings=None, step="startup", initial_state=None):
    config = load_experiment(CASE, settings=settings)
    cycle = CycleConfig.from_experiment(config)
    model = ColumnModelFactory(cycle).build_column_model(
        step=step,
        initial_state=initial_state,
        solver_options=cycle.solver_for_step(
            use_sparsity=False,
            compute_numerical_sparsity=False,
        ),
    )
    return model


def complete_initial_state(model):
    state = np.zeros(model.state_layout.state_size)
    state[: model.state_layout.pde_size] = model.get_initial_state()
    return state


@pytest.mark.parametrize(
    "settings",
    [
        {"grid.nz": 4, "grid.nr": 1},
        {"grid.nz": 4, "grid.nr": 2},
        CO2_N2_SETTINGS,
        {"grid.nz": 4, "grid.nr": 2, "mixture.closure_species": "O2"},
    ],
)
def test_rhs_shape_order_and_values_follow_generated_layout(settings):
    model = build_model(settings=settings)
    state = complete_initial_state(model)
    derivative = model.pde_system_kpis(0.0, state).copy()

    assert state.shape == derivative.shape == (model.state_layout.state_size,)
    assert np.all(np.isfinite(derivative))
    assert model.state_layout.unpack(derivative).keys() == {
        field.name for field in model.state_layout.fields
    }
    assert tuple(field.name for field in model.scalar_state_fields) == (
        *(f"m_{species}" for species in model.adsorbates),
        "E_compression",
        "E_vacuum",
        "E_jacket",
    )


def test_species_balance_diagnostic_is_opt_in_json_safe_and_rhs_neutral():
    model = build_model(settings={"grid.nz": 4, "grid.nr": 2})
    state = complete_initial_state(model)
    baseline = model.pde_system_kpis(0.0, state).copy()

    records = []
    model.set_species_balance_diagnostic_callback(records.append)
    observed = model.pde_system_kpis(0.0, state).copy()

    np.testing.assert_array_equal(observed, baseline)
    assert len(records) == 1
    record = records[0]
    json.dumps(record)
    assert record["schema_version"] == 1
    assert record["closure_species"] == model.closure_species
    assert record["all_state_values_finite"] is True
    assert record["negative_closure_cell_count"] == 0
    assert record["minimum_closure_concentration_mol_m3"] > 0.0
    assert record["decomposition_max_abs_error_mol_m3_s"] < 1e-10
    assert set(
        record["minimum_closure_cell"]["closure_derivative_mol_m3_s"]
    ) == {
        "convection",
        "axial_dispersion",
        "radial_dispersion",
        "adsorption",
        "total",
    }

    model.set_species_balance_diagnostic_callback(None)
    model.pde_system_kpis(0.0, state)
    assert len(records) == 1


def test_species_balance_diagnostic_reports_coupled_closure_violation():
    model = build_model(settings={"grid.nz": 4, "grid.nr": 2})
    state = complete_initial_state(model)
    c_tot_field = model.state_field_by_name["c_tot"]
    co2_field = model.state_field_by_name["c_CO2"]
    c_tot_dim = state[c_tot_field.start] * model.c_tot_ref
    state[co2_field.start] = 1.1 * c_tot_dim / model.c_ref_by_species["CO2"]

    records = []
    model.set_species_balance_diagnostic_callback(records.append)
    with np.errstate(all="ignore"):
        model.pde_system_kpis(0.0, state)

    record = records[0]
    assert record["negative_closure_cell_count"] >= 1
    assert record["minimum_closure_concentration_mol_m3"] < 0.0
    assert record["maximum_independent_sum_over_total"] > 1.0
    assert record["minimum_closure_cell"]["index"] == [0, 0]


def test_solver_constraints_tolerances_and_scales_follow_layout(tmp_path):
    model = build_model(settings=CO2_N2_SETTINGS)
    solver = Solver(model, base_folder=str(tmp_path))
    initial = solver._initial_state()
    indices, constraint_types = solver._solver_constraints()
    tolerances = solver._absolute_tolerances(initial)

    expected = {
        index
        for field in model.state_layout.fields
        if field.constraint == "nonnegative"
        for index in range(field.start, field.stop)
    }
    assert set(indices) == expected
    assert constraint_types == [1] * len(indices)
    gas = model.state_layout.gas_field("CO2")
    mass = model.state_layout.field("m_CO2")
    energy = model.state_layout.field("E_jacket")
    np.testing.assert_array_equal(tolerances[gas.state_slice], 1e-12)
    np.testing.assert_array_equal(tolerances[mass.state_slice], model.atol_scalar_mass)
    np.testing.assert_array_equal(tolerances[energy.state_slice], model.atol_scalar_energy)
    assert solver.state_scale_by_field["c_CO2"] == model.c_ref_by_species["CO2"]


def test_warm_start_rejects_a_different_layout_signature():
    startup = build_model(settings={"grid.nz": 4, "grid.nr": 1})
    warm = {
        "Nz": startup.Nz,
        "Nr": startup.Nr,
        "ci": startup.ci.copy(),
        "qi": startup.qi.copy(),
        "c_tot": startup.c_tot[..., 0].copy(),
        "T": startup.T[..., 0].copy(),
        "T_wall": startup.Tw[:, 0, 0].copy(),
        "layout_signature": "0" * 64,
    }
    with pytest.raises(ValueError, match="different state layout"):
        build_model(
            settings={"grid.nz": 4, "grid.nr": 1},
            step="ads",
            initial_state=warm,
        )


def test_generic_solver_export_and_final_fields_round_trip(tmp_path):
    model = build_model(settings=CO2_N2_SETTINGS)
    solver = Solver(model, base_folder=str(tmp_path))
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_cvode()
        solver.save_results()
    final = solver.get_final_fields()

    assert solver.sol["extra"]["state_layout"]["signature"] == model.state_layout.signature
    assert final["layout_signature"] == model.state_layout.signature
    assert final["ci"].shape == (model.Nz, model.Nr, 1)
    assert final["qi"].shape == (model.Nz, model.Nr, 1)
    assert solver.sol["y_2d"].shape[0] == model.state_layout.n_2d_fields
    assert solver.sol["y_scalars"].shape[0] == model.state_layout.n_scalar_states


@pytest.mark.parametrize("nr", [1, 2])
def test_default_solver_smoke_succeeds_in_1d_and_2d(nr, tmp_path):
    model = build_model(settings={"grid.nz": 4, "grid.nr": nr})
    # A Solver writes its provenance under base_folder; never into the checkout.
    solver = Solver(model, base_folder=str(tmp_path))
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_cvode()

    assert solver.sol["cvode"]["success"] is True
    assert solver.sol["termination"]["reason"] == "final_time"
    assert solver.sol["y"].shape[0] == model.state_layout.state_size
    assert np.all(np.isfinite(solver.sol["y"]))


def test_species_event_reconstructs_a_nonstandard_closure():
    model = build_model(
        settings={"grid.nz": 4, "grid.nr": 2, "mixture.closure_species": "O2"}
    )
    state = complete_initial_state(model)
    oxygen = max_bed_species_mole_fraction(
        state,
        layout=model.state_layout,
        species="O2",
        reference_concentrations=model.c_ref_by_species,
        total_reference_concentration=model.c_tot_ref,
    )
    np.testing.assert_allclose(oxygen, model.y_initial_by_species["O2"])


@pytest.mark.parametrize(
    "settings",
    [
        {"grid.nz": 4, "grid.nr": 1},
        {"grid.nz": 4, "grid.nr": 2},
        CO2_N2_SETTINGS,
    ],
    ids=("default_dac_1d", "default_dac_2d", "co2_n2"),
)
def test_analytical_sparsity_is_layout_sized_in_memory_and_conservative(
    tmp_path, settings
):
    model = build_model(settings=settings)
    layout = model.state_layout
    pattern = generate_sparsity_pattern(layout)
    assert pattern.shape == (layout.state_size, layout.state_size)

    # The solver gets the same mask, generated in memory, as a private copy.
    solver_mask, source, origin = try_get_solver_sparsity(step="startup", layout=layout)
    assert (source, origin) == ("analytical", "generated in memory")
    assert np.array_equal(solver_mask.indptr, pattern.indptr)
    assert np.array_equal(solver_mask.indices, pattern.indices)
    solver_mask.data[:] = False
    again, _, _ = try_get_solver_sparsity(step="ads", layout=layout)
    assert again.data.all() and (again != pattern).nnz == 0

    # The drawing tool writes only where it is told.
    written = plot_sparsity(pattern, layout, tmp_path / "plots" / "mask.pdf")
    assert written == tmp_path / "plots" / "mask.pdf" and written.stat().st_size > 0
    assert sorted(tmp_path.rglob("*")) == [tmp_path / "plots", written]

    pruned, mode, info = prepare_cvode_solver_sparsity(
        pattern,
        layout=layout,
        state_size=layout.state_size,
        sparsity_mode="pde_only",
    )
    assert mode == "pde_only"
    assert info["non_kpi_size"] == layout.pde_size
    assert pruned[layout.pde_size :, :].nnz == 0

    initial = complete_initial_state(model)
    numerical = finite_difference_jacobian(
        lambda values: model.pde_system_kpis(0.0, values).copy(),
        initial,
        eps=1e-7,
    )
    numerical_mask = np.abs(numerical) > 1e-9
    missing = numerical_mask & ~pattern.toarray()
    assert not np.any(missing), np.argwhere(missing)


def test_numerical_jacobian_diagnostic_writes_into_the_step_folder(tmp_path):
    from pydac.core.sparsity_patterns import numerical_sparsity

    config = load_experiment(CASE, settings={"grid.nz": 4, "grid.nr": 2})
    cycle = CycleConfig.from_experiment(config)
    model = ColumnModelFactory(cycle).build_column_model(
        step="startup",
        solver_options=cycle.solver_for_step(
            use_sparsity=True,
            compute_numerical_sparsity=True,
        ),
    )
    package = Path(numerical_sparsity.__file__).resolve().parents[2]

    def package_files():
        return {p for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts}

    before = package_files()
    with contextlib.redirect_stdout(io.StringIO()):
        Solver(model, base_folder=str(tmp_path)).solve_cvode()
    assert package_files() == before

    snapshots = sorted((tmp_path / "startup" / "numerical_jacobian").glob("*.npy"))
    assert snapshots, "no Jacobian snapshot written into the step folder"

    with contextlib.redirect_stdout(io.StringIO()):
        summary = numerical_sparsity.aggregate_numerical_sparsity(
            tmp_path / "startup", model.state_layout
        )
    assert summary["snapshots"] == len(snapshots)
    assert summary["missing_from_analytical"] == 0
    assert (tmp_path / "startup" / "numerical_jacobian" / "sparsity_union_bool.npy").exists()
    assert Path(summary["plot"]).parent == tmp_path / "startup" / "numerical_jacobian"
