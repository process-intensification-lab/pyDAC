from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from pydac.core.column_model_factory import ColumnModelFactory
from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.post_processing.context.runtime_store import RuntimeReportStore
from pydac.post_processing.context.simulation_context import SimulationContext
from pydac.post_processing.kpi.cycle_kpis import CycleKPIs
from pydac.post_processing.kpi.plotting import KPIPlotting
from pydac.post_processing.metrics.cycle_data import CycleData
from pydac.post_processing.metrics.cycle_metrics import CycleMetrics
from pydac.post_processing.profiles.cyclic_plotting import CyclicSteadyStateProfiles
from pydac.post_processing.profiles.step_post_processing import StepPostProcessor
from pydac.core.solver import Solver
from pydac.core.state_layout import StateLayout, current_dac_state_layout
from pydac.utils.css.profiles import extract_final_fields, final_profiles
from pydac.utils.postprocessing.runtime_report import (
    merge_runtime_reports,
    runtime_report_from_model,
    validate_runtime_report,
)
from pydac.utils.postprocessing.saved_solution import (
    SAVED_SOLUTION_SCHEMA_KEY,
    SAVED_SOLUTION_SCHEMA_VERSION,
    saved_solution_view,
)


CASE = "tests/cases/v3_smoke_startup.json"
GENERIC_SETTINGS = {
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


def _build_model(*, nr=1, settings=None, step="startup", initial_state=None):
    overrides = {"grid.nr": nr}
    if settings:
        overrides.update(settings)
    config = load_experiment(CASE, settings=overrides)
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


def _warm_state(model):
    return {
        "Nz": model.Nz,
        "Nr": model.Nr,
        "ci": model.ci.copy(),
        "qi": model.qi.copy(),
        "c_tot": model.c_tot[..., 0].copy(),
        "T": model.T[..., 0].copy(),
        "T_wall": model.Tw[:, 0, 0].copy(),
        "layout_signature": model.state_layout.signature,
    }


def _solve(model, folder):
    solver = Solver(model, base_folder=str(folder))
    solver.solve_cvode()
    solver.save_results()
    return solver, StepPostProcessor(model, solver)


@pytest.fixture(scope="module")
def dac_1d(tmp_path_factory):
    return _solve(_build_model(nr=1), tmp_path_factory.mktemp("phase6_dac_1d"))


@pytest.fixture(scope="module")
def dac_2d(tmp_path_factory):
    return _solve(_build_model(nr=2), tmp_path_factory.mktemp("phase6_dac_2d"))


@pytest.fixture(scope="module")
def generic_1d(tmp_path_factory):
    return _solve(
        _build_model(nr=1, settings=GENERIC_SETTINGS),
        tmp_path_factory.mktemp("phase6_generic_1d"),
    )


def test_state_layout_round_trip_rebuilds_offsets_and_rejects_tampering():
    layout = current_dac_state_layout(4, 2)
    assert StateLayout.from_dict(layout.to_dict()) == layout

    tampered = deepcopy(layout.to_dict())
    tampered["fields"][0]["start"] = 1
    with pytest.raises(ValueError, match="metadata differs"):
        StateLayout.from_dict(tampered)

    tampered = deepcopy(layout.to_dict())
    tampered["signature"] = "not-the-generated-signature"
    with pytest.raises(ValueError, match="signature does not match"):
        StateLayout.from_dict(tampered)


def test_saved_solution_uses_persisted_names_and_only_narrow_legacy_fallback(dac_2d):
    solver, _ = dac_2d
    view = saved_solution_view(solver.sol)
    assert view.compatibility_mode == "versioned_canonical"
    assert solver.sol[SAVED_SOLUTION_SCHEMA_KEY] == SAVED_SOLUTION_SCHEMA_VERSION
    assert "y" not in solver.sol
    assert tuple(view.fields) == tuple(field.name for field in view.layout.fields)
    np.testing.assert_array_equal(view.field("q_H2O"), solver.sol["y_2d"][4])

    previous_v3 = deepcopy(solver.sol)
    previous_v3.pop(SAVED_SOLUTION_SCHEMA_KEY)
    assert saved_solution_view(previous_v3).compatibility_mode == "persisted_layout"

    legacy = deepcopy(solver.sol)
    legacy.pop(SAVED_SOLUTION_SCHEMA_KEY)
    legacy["extra"].pop("state_layout")
    legacy_view = saved_solution_view(legacy)
    assert legacy_view.compatibility_mode == "legacy_current_dac"
    assert legacy_view.layout.signature == view.layout.signature

    ambiguous = deepcopy(legacy)
    ambiguous["y_2d"] = ambiguous["y_2d"][:-1]
    with pytest.raises(ValueError, match="exact historical current-DAC"):
        saved_solution_view(ambiguous)

    unsupported = deepcopy(solver.sol)
    unsupported[SAVED_SOLUTION_SCHEMA_KEY] = SAVED_SOLUTION_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="Unsupported saved-solution schema"):
        saved_solution_view(unsupported)

    duplicated = deepcopy(solver.sol)
    duplicated["y"] = np.empty(
        (solver.state_layout.state_size, solver.sol["t"].size)
    )
    with pytest.raises(ValueError, match="must not contain the duplicate `y`"):
        saved_solution_view(duplicated)


def test_prepared_storage_groups_share_the_original_packed_allocation(tmp_path):
    model = _build_model(nr=2)
    solver = Solver(model, base_folder=str(tmp_path))
    solver.solve_cvode()
    packed_history = solver.sol["y"]
    assert packed_history.flags.c_contiguous
    expected = {}
    for field in model.state_layout.fields:
        values = packed_history[field.state_slice].reshape(
            (*field.shape, model.Nt)
        ).copy()
        values *= solver.state_scale_by_field[field.name]
        if field.kind in ("gas concentration", "total concentration"):
            np.maximum(values, 0.0, out=values)
        expected[field.name] = values

    prepared = solver.prepare_results()

    assert "y" not in prepared
    assert np.shares_memory(packed_history, prepared["y_2d"])
    assert np.shares_memory(packed_history, prepared["y_1d"])
    assert np.shares_memory(packed_history, prepared["y_scalars"])
    assert prepared["y_2d"].flags.c_contiguous
    assert prepared["y_1d"].flags.c_contiguous
    assert prepared["y_scalars"].flags.c_contiguous
    assert solver.prepare_results() is prepared

    view = saved_solution_view(prepared)
    for field in model.state_layout.fields:
        np.testing.assert_array_equal(view.field(field.name), expected[field.name])


def test_step_postprocessor_prefers_live_results_and_keeps_disk_fallback(
    tmp_path, monkeypatch
):
    model = _build_model(nr=1)
    solver = Solver(model, base_folder=str(tmp_path))
    solver.solve_cvode()
    solver.save_results()

    def unexpected_load_results(*args, **kwargs):
        raise AssertionError("live post-processing must not reload sol.pkl")

    monkeypatch.setattr(solver, "load_results", unexpected_load_results)
    live_processor = StepPostProcessor(model, solver)
    assert live_processor.sol is solver.sol

    reload_solver = Solver(model, base_folder=str(tmp_path))
    disk_processor = StepPostProcessor(model, reload_solver)
    assert disk_processor.sol is reload_solver.sol
    assert disk_processor.sol is not solver.sol
    assert "y" not in disk_processor.sol
    np.testing.assert_array_equal(
        disk_processor.sol["y_2d"], solver.sol["y_2d"]
    )


def test_runtime_report_is_strict_complete_and_uses_spatial_references():
    model = _build_model(nr=1, settings=GENERIC_SETTINGS)
    report = runtime_report_from_model(model)

    json.dumps(report, allow_nan=False)
    assert report["schema_version"] == 1
    assert tuple(report["species_catalog"]) == ("CO2", "N2")
    assert report["models"]["equilibrium_systems"][0]["implementation"]
    assert report["models"]["kinetics"]["CO2"]["implementation"] == "LDFRate"
    assert report["spatial_values_policy"]["mode"] == "external_solution_reference"

    startup = report["steps"]["startup"]
    assert startup["evaluation_state"]["temperature"] == {
        "value": model.T_feed,
        "unit": "K",
    }
    assert tuple(
        startup["properties"]["per_species"]["mixture_diffusivity"]["values"]
    ) == ("CO2", "N2")
    assert startup["properties"]["pairwise"]["binary_diffusivity"]["pairs"] == [
        {
            "species": ["CO2", "N2"],
            "value": model.runtime_properties["binary_diffusivity_m2_s"]["CO2-N2"],
        }
    ]
    assert startup["spatial_values"]["artifact"] == "startup/sol.pkl"
    assert "values" not in startup["spatial_values"]


def test_runtime_report_merge_is_ordered_idempotent_and_conflict_strict(tmp_path):
    startup_model = _build_model(nr=1, settings=GENERIC_SETTINGS)
    ads_model = _build_model(
        nr=1,
        settings=GENERIC_SETTINGS,
        step="ads",
        initial_state=_warm_state(startup_model),
    )
    startup = runtime_report_from_model(startup_model)
    adsorption = runtime_report_from_model(ads_model)
    merged = merge_runtime_reports(startup, adsorption)

    assert tuple(merged["steps"]) == ("startup", "ads")
    assert merge_runtime_reports(merged, adsorption) == merged

    store = RuntimeReportStore(tmp_path)
    store.save(startup)
    store.save(adsorption)
    assert store.load() == merged

    conflict = deepcopy(adsorption)
    conflict["steps"]["ads"]["evaluation_state"]["temperature"]["value"] += 1.0
    validate_runtime_report(conflict)
    with pytest.raises(ValueError, match="completed step 'ads' differs"):
        merge_runtime_reports(merged, conflict)


def test_simulation_context_schema_expands_from_declared_species_and_merges():
    startup_model = _build_model(nr=1, settings=GENERIC_SETTINGS)
    ads_model = _build_model(
        nr=1,
        settings=GENERIC_SETTINGS,
        step="ads",
        initial_state=_warm_state(startup_model),
    )
    startup = SimulationContext.from_model(startup_model)
    adsorption = SimulationContext.from_model(ads_model)
    merged = startup.merged_with(adsorption)

    assert merged.species == ("CO2", "N2")
    assert merged.steps == ("startup", "ads")
    dispersion_ids = tuple(
        parameter_id
        for parameter_id in merged.required_step_parameter_ids
        if parameter_id.startswith("bed.dispersion.")
    )
    assert dispersion_ids == (
        "bed.dispersion.axial.co2",
        "bed.dispersion.axial.n2",
        "bed.dispersion.radial.co2",
        "bed.dispersion.radial.n2",
    )


@pytest.mark.parametrize(
    ("fixture_name", "expected"),
    [
        # Re-pinned when the mixed cup became the stream's own composition
        # (molar-flow ratio at the outlet face, enthalpy-balance temperature).
        # The old values weighted last-cell-centre mole fractions by mass flux,
        # while the flash in the same duty calculation already read the face, so
        # the two halves of the condenser disagreed by these amounts.
        # Re-pinned again (~1e-7) when the molecular term was dropped from the
        # radial conductivity (Dixon Eq. 6, lambda_rad = lambda_0 at no flow).
        (
            "dac_1d",
            {
                "y_CO2": 1.3812699538571547e-08,
                "y_H2O": 0.004037740218480605,
                "y_O2": 0.21061004831966235,
                "y_N2": 0.7853521976491574,
                "T_out": 298.18076280961526,
            },
        ),
        (
            "dac_2d",
            {
                "y_CO2": 1.3812678972563044e-08,
                "y_H2O": 0.004037736849440062,
                "y_O2": 0.21061004903213562,
                "y_N2": 0.7853522003057454,
                "T_out": 298.18075453885757,
            },
        ),
    ],
)
def test_current_dac_postprocessing_regression_matches_energy_model(
    fixture_name, expected, request
):
    _, processor = request.getfixturevalue(fixture_name)
    final = {name: values[-1] for name, values in processor.get_mixed_cups().items()}
    assert final == pytest.approx(expected, rel=1e-10, abs=1e-12)
    assert len(processor.compute_step_metrics().records) == 31
    assert tuple(processor.get_mass_balance_summary()) == (
        "CO2",
        "H2O",
        "O2",
        "N2",
    )


def test_generic_solution_profiles_metrics_css_and_dac_guards(generic_1d):
    solver, processor = generic_1d
    assert processor.gas_species == ("CO2", "N2")
    assert processor.adsorbates == ("CO2",)
    assert tuple(processor.get_mixed_cups()) == ("y_CO2", "y_N2", "T_out")
    assert tuple(processor.get_mass_balance_summary()) == ("CO2", "N2")

    records = processor.compute_step_metrics().records
    assert {record.species for record in records} == {"CO2", "N2", "system"}
    processor.plotter.plot_all_profiles(save=False, show=False)
    with pytest.raises(ValueError, match="legacy get_masses tuple requires"):
        processor.get_masses()

    solution_path = Path(solver.base_folder) / "startup" / "sol.pkl"
    final = extract_final_fields(solution_path)
    assert final["ci"].shape == (4, 1, 1)
    assert final["qi"].shape == (4, 1, 1)
    assert tuple(final_profiles(solution_path)) == (
        "c_CO2",
        "q_CO2",
        "c_tot",
        "T_bed",
        "T_wall",
    )

    cyclic = CyclicSteadyStateProfiles.__new__(CyclicSteadyStateProfiles)
    cyclic.layout_signature = None
    cyclic.gas_species = ()
    cyclic.adsorbates = ()
    cyclic.field_specs = dict(cyclic.FIELD_SPECS)
    cyclic.default_plot_fields = tuple(cyclic.DEFAULT_PLOT_FIELDS)
    cyclic.is_1d = None
    cyclic._mixed_cup_zero_den_warned = False
    outlet = cyclic._extract_outlet_profiles(solver.sol, "startup")
    assert {"y_CO2", "y_N2", "ppm_CO2", "ppm_N2", "q_CO2"} <= set(outlet)
    assert cyclic.default_plot_fields == (
        "y_CO2",
        "y_N2",
        "q_CO2",
        "T_bed",
        "T_wall",
        "pressure",
    )

    cycle_metrics = CycleMetrics.from_step_metrics(
        "cycle_1", {"startup": processor.compute_step_metrics()}
    )
    kpis = CycleKPIs(
        CycleData(cycle_metrics, SimulationContext.from_model(processor.pb))
    )
    assert tuple(kpis.mass_change_table(basis="state")) == ("CO2", "N2")
    KPIPlotting(kpis).plot_css_mass_changes(
        basis="state", save=False, show=False
    )
    # Purity needs CO2 only (17b); this startup-only cycle simply has no product step.
    with pytest.raises(ValueError, match="collects product in no step"):
        kpis.purity()
    assert np.isnan(kpis.h2o_to_co2_ratio())
    assert kpis.steam_mass_used() == 0.0


def test_generic_outlet_plots_stay_filled_when_no_stream_leaves(generic_1d, monkeypatch):
    # A closed step (cool) has no outgoing stream: the mixed-cup traces are NaN.
    # The generic outlet figures must still show the bed and gas at the outlet.
    import matplotlib.pyplot as plt

    _, processor = generic_1d
    cups = processor.get_mixed_cups()
    closed = {key: np.full_like(value, np.nan) for key, value in cups.items()}
    monkeypatch.setattr(processor, "_mixed_cups_cache", closed)
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    before = set(plt.get_fignums())
    try:
        processor.plotter.plot_generic_outlet_profiles(save=False, show=False)
        figures = [plt.figure(n) for n in sorted(set(plt.get_fignums()) - before)]
        assert len(figures) == 2
        for figure in figures:
            for axis in figure.axes:
                traces = [
                    line.get_ydata()
                    for line in axis.get_lines()
                    if len(line.get_xdata()) > 2
                ]
                assert any(np.isfinite(y).any() for y in traces), axis.get_title()
    finally:
        monkeypatch.undo()
        for n in set(plt.get_fignums()) - before:
            plt.close(n)


def test_css_plots_of_each_reported_cycle_get_their_own_folder(tmp_path, monkeypatch):
    # Reporting cycle 2 after cycle 3 in the same format must not overwrite cycle 3's figures.
    (tmp_path / "2D_mode").mkdir()
    monkeypatch.setattr(CyclicSteadyStateProfiles, "_load_css_cycle", lambda self: None)
    folders = [
        Path(CyclicSteadyStateProfiles(base_folder=str(tmp_path), css_cycle_idx=cycle).plot_folder)
        for cycle in (3, 2)
    ]
    assert folders == [
        tmp_path / "2D_mode" / "plots_css" / "cycle_3",
        tmp_path / "2D_mode" / "plots_css" / "cycle_2",
    ]
    assert all(folder.is_dir() for folder in folders)


@pytest.mark.parametrize("n_points", [1, 4])
def test_pareto_report_survives_a_variable_without_spread(tmp_path, n_points):
    # A small front can share one value of a variable (crossover copies it), or be
    # one point: the KDE of the mode has no spread to work with and used to crash
    # the report after the campaign, before pareto_front.csv was written.
    from pydac.workflows.orchestrator_NSGA2 import run_pareto_postprocessing

    (tmp_path / "pareto_bounds.csv").write_text(
        "variable,lower_bound,upper_bound,space\n"
        "t_ads,10800.0,43200.0,optimizer_velocity\n"
        "t_heat,3600.0,10800.0,optimizer_velocity\n"
        "P_vac,19000.0,21000.0,optimizer_velocity\n"
    )
    results = tmp_path / "results"
    results.mkdir()
    X = np.column_stack([
        np.linspace(20000.0, 40000.0, n_points),
        np.full(n_points, 5440.0),
        np.linspace(19500.0, 20500.0, n_points),
    ])
    F = np.column_stack([np.linspace(36.0, 51.0, n_points), -np.linspace(35.0, 37.0, n_points)])
    np.savez(results / "last_pareto_front.npz", X=X, F=F)
    np.savez(results / "last_population.npz", X=np.vstack([X, X + 100.0]), F=np.vstack([F, F + 1.0]))

    run_pareto_postprocessing(str(results))

    front = np.genfromtxt(results / "pareto_front.csv", delimiter=",", names=True)
    assert np.atleast_1d(front["t_heat"]).tolist() == [5440.0] * n_points
    modes = np.genfromtxt(results / "decision_variable_modes.csv", delimiter=",", names=True,
                          dtype=None, encoding="utf-8")
    t_heat = modes[modes["Variable"] == "t_heat"]
    assert t_heat["Mode_unscaled"][0] == pytest.approx(5440.0)
    assert (results / "figures").is_dir()
