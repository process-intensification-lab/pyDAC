"""Contracts for trajectory-free NSGA-II step persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.core.TVSA_cycle import TVSA_Cycle
from pydac.core.solver import Solver
from pydac.post_processing.kpi.cycle_kpis import CycleKPIs
from pydac.post_processing.metrics.cycle_data_loader import CycleDataLoader
from pydac.utils.css.cycle import load_resume_prev_fields, load_resume_prev_profile
from pydac.utils.css.runner import evaluate_css_after_cycle
from pydac.utils.postprocessing.runtime_report import runtime_report_from_model
from pydac.utils.postprocessing.terminal_state import (
    TERMINAL_STATE_FILENAME,
    TERMINAL_STATE_SCHEMA_VERSION,
    load_terminal_state,
    terminal_final_fields,
    terminal_profiles,
)
from pydac.utils.solver.persistence import MOO_COMPACT, RUN_FULL, normalize_persistence_policy
from pydac.workflows.orchestrator_TVSA_cycle_css import run_css_simulation


CASE = "tests/cases/v3_smoke_startup.json"


def _cycle_config():
    return CycleConfig.from_experiment(load_experiment(CASE))


def _model():
    cycle = _cycle_config()
    return cycle, ColumnModelFactory(cycle).build_column_model(
        step="startup",
        initial_state=None,
        solver_options=cycle.solver_for_step(
            use_sparsity=False,
            compute_numerical_sparsity=False,
        ),
    )


def _run_startup(folder: Path, policy: str):
    runner = TVSA_Cycle(
        _cycle_config(),
        base_folder=str(folder),
        base_folder_is_spatial_root=True,
        print_to_terminal=False,
        persistence_policy=policy,
    )
    final_fields, final_time = runner._run_step(
        "startup",
        solve=True,
        use_sparsity=False,
    )
    return runner, final_fields, final_time


def test_persistence_policy_is_explicit_and_run_full_is_default():
    assert normalize_persistence_policy(None) == RUN_FULL
    assert normalize_persistence_policy(" MOO_COMPACT ") == MOO_COMPACT
    with pytest.raises(ValueError, match="Unsupported persistence policy"):
        normalize_persistence_policy("endpoint_only")


def test_compact_terminal_round_trip_and_release(tmp_path):
    _, model = _model()
    solver = Solver(model, base_folder=str(tmp_path))
    solver.solve_cvode()
    solver.prepare_results()
    expected_fields = solver.get_final_fields()
    expected_profile = solver.get_final_profile()
    for name in ("c_tot", "T", "T_wall"):
        assert not np.shares_memory(expected_fields[name], solver.sol["y_2d"])
        assert not np.shares_memory(expected_fields[name], solver.sol["y_1d"])

    terminal_path = solver.save_terminal_results()
    assert Path(terminal_path).name == TERMINAL_STATE_FILENAME
    assert Path(terminal_path).is_file()
    assert not (tmp_path / "startup" / "sol.pkl").exists()

    terminal = load_terminal_state(terminal_path)
    assert terminal["terminal_state_schema_version"] == TERMINAL_STATE_SCHEMA_VERSION
    assert terminal["persistence_policy"] == MOO_COMPACT
    reconstructed = terminal_final_fields(terminal)
    reconstructed_profile = terminal_profiles(terminal)
    for name in ("ci", "qi", "c_tot", "T", "T_wall"):
        np.testing.assert_array_equal(reconstructed[name], expected_fields[name])
    for name in expected_profile:
        np.testing.assert_array_equal(reconstructed_profile[name], expected_profile[name])

    solver.release_results()
    assert solver.sol["results_released"] is True
    assert not ({"y", "y_2d", "y_1d", "y_scalars"} & set(solver.sol))


def test_run_full_and_moo_compact_have_exact_step_outputs(tmp_path):
    full_runner, full_fields, full_time = _run_startup(tmp_path / "full", RUN_FULL)
    compact_runner, compact_fields, compact_time = _run_startup(
        tmp_path / "compact", MOO_COMPACT
    )

    assert compact_time == full_time
    for name in ("ci", "qi", "c_tot", "T", "T_wall"):
        np.testing.assert_array_equal(compact_fields[name], full_fields[name])
    for name, full_values in full_runner.final_profile_for_step("startup").items():
        np.testing.assert_array_equal(
            compact_runner.final_profile_for_step("startup")[name],
            full_values,
        )

    full_kpis = pd.read_pickle(tmp_path / "full/startup/kpis.pkl")
    compact_kpis = pd.read_pickle(tmp_path / "compact/startup/kpis.pkl")
    pd.testing.assert_frame_equal(compact_kpis, full_kpis, check_exact=True)

    assert (tmp_path / "full/startup/sol.pkl").is_file()
    assert not (tmp_path / "full/startup/terminal.pkl").exists()
    assert (tmp_path / "compact/startup/terminal.pkl").is_file()
    assert not list((tmp_path / "compact").rglob("sol.pkl"))
    assert compact_runner._last_step_run_record["solver"] == full_runner._last_step_run_record[
        "solver"
    ]

    compact_runtime = json.loads(
        (tmp_path / "compact/resolved_runtime.json").read_text(encoding="utf-8")
    )
    assert compact_runtime["provenance"]["spatial_solution_pattern"] == (
        "<step>/terminal.pkl"
    )
    assert compact_runtime["steps"]["startup"]["spatial_values"]["artifact"] == (
        "startup/terminal.pkl"
    )


def test_compact_css_uses_live_profile_and_terminal_resume(tmp_path, monkeypatch):
    cycle_folder = tmp_path / "cycle_1"
    runner, expected_fields, _ = _run_startup(cycle_folder, MOO_COMPACT)
    expected_profile = runner.final_profile_for_step("startup")

    def no_full_solution_load(*args, **kwargs):
        raise AssertionError("compact CSS must not load sol.pkl")

    monkeypatch.setattr('pydac.utils.css.runner.final_profiles', no_full_solution_load)
    check = evaluate_css_after_cycle(
        tvsa_cycle=runner,
        cycle_idx=1,
        step_records=[{"step": "startup"}],
        probe_step="startup",
        prev_prof=None,
        css_tol=0.005,
        cycle_run_config={
            "requested_cycle_plots": False,
            "force_live_cycle_plots": False,
            "run_kwargs": {"termination_mode": "time"},
            "step_options": {},
        },
        results_base_folder=str(tmp_path),
        cycle_steps=["startup"],
        print_to_terminal=False,
    )
    assert check.reached is False
    for name, values in expected_profile.items():
        np.testing.assert_array_equal(check.prev_prof[name], values)

    resumed_fields = load_resume_prev_fields(
        runner,
        str(tmp_path),
        previous_cycle_idx=1,
        previous_executed_steps=["startup"],
    )
    resumed_profile = load_resume_prev_profile(
        runner,
        str(tmp_path),
        previous_cycle_idx=1,
        previous_executed_steps=["startup"],
        probe_step="startup",
    )
    for name in ("ci", "qi", "c_tot", "T", "T_wall"):
        np.testing.assert_array_equal(resumed_fields[name], expected_fields[name])
    for name, values in expected_profile.items():
        np.testing.assert_array_equal(resumed_profile[name], values)


def test_compact_runtime_report_keeps_full_report_backward_compatible():
    _, model = _model()
    full = runtime_report_from_model(model)
    compact = runtime_report_from_model(model, persistence_policy=MOO_COMPACT)

    assert full["provenance"]["spatial_solution_pattern"] == "<step>/sol.pkl"
    assert compact["provenance"]["spatial_solution_pattern"] == (
        "<step>/terminal.pkl"
    )
    assert compact["spatial_values_policy"]["mode"] == (
        "external_compact_terminal"
    )


@pytest.mark.parametrize("nr", [1, 2], ids=["1d", "2d"])
def test_full_and_compact_css_are_strictly_equivalent(tmp_path, nr):
    from pydac.utils.nsga2.evaluation import extract_candidate_kpis

    experiment = load_experiment(
        CASE,
        settings={
            "execution.cycle_steps": [
                "ads",
                "blow",
                "heat",
                "s_des",
                "press",
                "cool",
            ],
            "execution.max_cycles": 2,
            "execution.css_tolerance": 1e9,
            "execution.termination_mode": "time",
            "execution.plot_profiles": False,
            "execution.plot_final_cycle_profiles": False,
            "execution.print_to_terminal": False,
            "execution.print_cycle_build": False,
            "grid.nr": nr,
        },
    )
    full = run_css_simulation(
        experiment,
        base_folder=str(tmp_path / "full_css"),
        retain_cycles=2,
        persistence_policy=RUN_FULL,
    )
    compact = run_css_simulation(
        experiment,
        base_folder=str(tmp_path / "compact_css"),
        retain_cycles=2,
        persistence_policy=MOO_COMPACT,
    )

    assert (compact["css_reached"], compact["cycles_completed"], compact["css_cycle_idx"]) == (
        full["css_reached"],
        full["cycles_completed"],
        full["css_cycle_idx"],
    )
    assert compact["css_history"] == full["css_history"]
    assert [record["solver"] for record in compact["last_step_records"]] == [
        record["solver"] for record in full["last_step_records"]
    ]
    assert [record["termination"] for record in compact["last_step_records"]] == [
        record["termination"] for record in full["last_step_records"]
    ]

    steps = ("ads", "blow", "heat", "s_des", "press", "cool")
    full_root = Path(full["results_base_folder"])
    compact_root = Path(compact["results_base_folder"])
    for step in steps:
        full_step = pd.read_pickle(full_root / f"cycle_2/{step}/kpis.pkl")
        compact_step = pd.read_pickle(compact_root / f"cycle_2/{step}/kpis.pkl")
        pd.testing.assert_frame_equal(compact_step, full_step, check_exact=True)

    full_kpis = extract_candidate_kpis(
        CycleKPIs(CycleDataLoader(full_root, cycle_index=2).load())
    )
    compact_kpis = extract_candidate_kpis(
        CycleKPIs(CycleDataLoader(compact_root, cycle_index=2).load())
    )
    assert compact_kpis == full_kpis
    assert len(list(compact_root.rglob("terminal.pkl"))) == 12
    assert not list(compact_root.rglob("sol.pkl"))

    # A loaded cycle carries its source attribution when reports are saved elsewhere.
    from pydac.post_processing.kpi.reporting import KPIReporting
    from pydac.utils.provenance import read_record
    import shutil
    loaded = CycleDataLoader(compact_root, cycle_index=2).load()
    shutil.move(compact_root, tmp_path / "relocated_compact")
    report_folder = tmp_path / "external_reports"
    reporting = KPIReporting(CycleKPIs(loaded), output_folder=report_folder)
    reporting.save_css_summary_report(T_L=299.15)
    report = read_record(report_folder / "cycle_2_report_provenance.json")
    assert report["sources"] == loaded.provenance
    assert report["execution"]["T_L"] == 299.15
    reporting.save_css_summary_report(T_L=300.15)
    refreshed = read_record(report_folder / "cycle_2_report_provenance.json")
    assert refreshed["record_id"] != report["record_id"]
    assert refreshed["sources"] == report["sources"]
    assert read_record(report_folder / "provenance" / f"{report['record_id']}.json") == report
