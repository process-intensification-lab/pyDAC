"""NSGA-II guardrails for unsuccessful final CVODE solves."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import pytest

from pydac.utils.solver.outcome import (
    FinalSolverFailure,
    compact_solver_outcome,
    require_final_solver_success,
)
from pydac.utils.solver.composition_domain import CompositionDomainGuard


def _failed_solution():
    return {
        "solver": "scikit",
        "cvode": {
            "success": False,
            "status": -4,
            "message": "corrector convergence failed",
            "nfev": 37,
            "linsolver": "sparse",
        },
        "t": [0.0, 18.25],
    }


def test_final_solver_failure_summary_is_compact_and_complete():
    with pytest.raises(FinalSolverFailure) as caught:
        require_final_solver_success(_failed_solution(), step="heat")

    assert caught.value.summary == {
        "step": "heat",
        "status": -4,
        "message": "corrector convergence failed",
        "nfev": 37,
        "effective_solver": "sparse",
        "final_time_s": 18.25,
    }


def test_successful_final_solve_is_accepted():
    solution = _failed_solution()
    solution["cvode"].update(
        success=True,
        status=0,
        message="success",
        linsolver="sparse",
    )

    require_final_solver_success(solution, step="heat")


def test_missing_final_success_metadata_fails_closed():
    with pytest.raises(FinalSolverFailure) as caught:
        require_final_solver_success(
            {"solver": "scikit", "cvode": {}, "t": []},
            step="ads",
        )

    assert caught.value.summary["status"] == "missing_metadata"
    assert caught.value.summary["final_time_s"] is None


def test_initial_event_skip_records_that_no_linear_solver_ran():
    outcome = compact_solver_outcome(
        {
            "solver": "scikit",
            "cvode": {
                "success": True,
                "status": 2,
                "message": "Initial terminal event already satisfied.",
                "nfev": 0,
            },
            "termination": {"status": "initial_event"},
            "t": [0.0, 1e-9],
        },
        step="cool",
    )

    assert outcome["effective_solver"] == "not_run"


@pytest.mark.parametrize(
    ("persistence_policy", "expected_events"),
    [
        ("run_full", ["solve", "save"]),
        ("moo_compact", ["solve"]),
    ],
)
def test_failed_step_persistence_is_bounded_and_never_postprocessed_or_propagated(
    monkeypatch,
    tmp_path,
    persistence_policy,
    expected_events,
):
    # Import through the CSS entry point, which installs the repository's src
    # folder exactly as production does.
    from pydac.workflows import orchestrator_TVSA_cycle_css  # noqa: F401
    import pydac.core.TVSA_cycle as cycle_module

    events = []

    class FakeSolver:
        def __init__(self, *args, **kwargs):
            self.sol = None

        def solve_cvode(self):
            events.append("solve")
            self.sol = _failed_solution()

        def save_results(self):
            events.append("save")

        def save_terminal_results(self):
            events.append("terminal-save")

        def get_final_fields(self):
            events.append("propagate")
            return {"must_not": "happen"}

    class FakeContext:
        @staticmethod
        def from_model(model):
            return object()

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def exists(self):
            return False

        def validate_compatible(self, value):
            raise AssertionError("A new store should not need validation.")

        def from_model(self, model):
            return object()

        def save(self, value):
            events.append("context-save")

    class FakeConfig:
        grid = object()

        @staticmethod
        def solver_for_step(**kwargs):
            return object()

    factory = SimpleNamespace(
        config=FakeConfig(),
        build_column_model=lambda **kwargs: SimpleNamespace(termination_mode="time"),
    )
    cycle = object.__new__(cycle_module.TVSA_Cycle)
    cycle.column_factory = factory
    cycle.base_folder = str(tmp_path)
    cycle.base_folder_is_spatial_root = True
    cycle.print_to_terminal = False
    cycle.cyclic_mode = True
    cycle.cycle_idx = 1
    cycle.last_cycle_idx = 1
    cycle.persistence_policy = persistence_policy

    monkeypatch.setattr(cycle_module, "Solver", FakeSolver)
    monkeypatch.setattr(cycle_module, "SimulationContext", FakeContext)
    monkeypatch.setattr(cycle_module, "SimulationContextStore", FakeStore)
    monkeypatch.setattr(cycle_module, "RuntimeReportStore", FakeStore)
    monkeypatch.setattr(
        cycle_module,
        "runtime_report_from_model",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        cycle_module,
        "StepPostProcessor",
        lambda *args, **kwargs: events.append("postprocess"),
    )
    monkeypatch.setattr(
        cycle_module,
        "_resolve_spatial_base_folder_from_solver_settings",
        lambda base_folder, **kwargs: base_folder,
    )
    monkeypatch.setattr(
        cycle_module,
        "_describe_termination_setup_from_model",
        lambda model: {"mode": "time", "display": "fixed time"},
    )
    monkeypatch.setattr(
        cycle_module,
        "log_to",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    with pytest.raises(FinalSolverFailure):
        cycle._run_step("heat", solve=True)

    assert events == expected_events


def test_state_domain_failure_is_rejected_without_saving_large_trajectory(
    monkeypatch,
    tmp_path,
):
    from pydac.workflows import orchestrator_TVSA_cycle_css  # noqa: F401
    import pydac.core.TVSA_cycle as cycle_module

    events = []
    domain_solution = {
        "solver": "scikit",
        "cvode": {
            "success": False,
            "status": "state_domain_violation",
            "message": "closure allowance exceeded",
            "nfev": 3993,
            "linsolver": "sparse",
        },
        # Raw nondimensional time; the termination record is dimensional.
        "t": [0.0, 2.0],
        "termination": {
            "reason": "state_domain_violation",
            "success": False,
            "status": "state_domain_violation",
            "message": "closure allowance exceeded",
            "t_final_s": 2232.3,
            "event": {
                "t_event_s": 2232.3,
                "metadata": {
                    "absolute_allowance_mol_m3": 2.0,
                    "relative_allowance": 0.1,
                    "enabled": True,
                },
                "diagnostic": {"closure_concentration_mol_m3": -2.54},
            },
        },
    }

    class FakeSolver:
        def __init__(self, *args, **kwargs):
            self.sol = None

        def solve_cvode(self):
            events.append("solve")
            self.sol = domain_solution

        def save_results(self):
            events.append("save")

    class FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def exists(self):
            return False

        def from_model(self, model):
            return object()

    class FakeConfig:
        grid = object()

        @staticmethod
        def solver_for_step(**kwargs):
            return object()

    cycle = object.__new__(cycle_module.TVSA_Cycle)
    cycle.column_factory = SimpleNamespace(
        config=FakeConfig(),
        build_column_model=lambda **kwargs: SimpleNamespace(termination_mode="time"),
    )
    cycle.base_folder = str(tmp_path)
    cycle.base_folder_is_spatial_root = True
    cycle.print_to_terminal = False
    cycle.cyclic_mode = True
    cycle.cycle_idx = 1
    cycle.last_cycle_idx = 1
    cycle.composition_domain_guard = CompositionDomainGuard()
    cycle.persistence_policy = "moo_compact"

    monkeypatch.setattr(cycle_module, "Solver", FakeSolver)
    monkeypatch.setattr(
        cycle_module,
        "SimulationContext",
        SimpleNamespace(from_model=lambda model: object()),
    )
    monkeypatch.setattr(cycle_module, "SimulationContextStore", FakeStore)
    monkeypatch.setattr(cycle_module, "RuntimeReportStore", FakeStore)
    monkeypatch.setattr(
        cycle_module,
        "runtime_report_from_model",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        cycle_module,
        "_resolve_spatial_base_folder_from_solver_settings",
        lambda base_folder, **kwargs: base_folder,
    )
    monkeypatch.setattr(
        cycle_module,
        "_describe_termination_setup_from_model",
        lambda model: {"mode": "time", "display": "fixed time"},
    )
    monkeypatch.setattr(
        cycle_module,
        "log_to",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    with pytest.raises(FinalSolverFailure) as caught:
        cycle._run_step("s_des", solve=True)

    assert events == ["solve"]
    assert caught.value.summary["final_time_s"] == pytest.approx(2232.3)
    assert caught.value.summary["failure_kind"] == "state_domain_violation"


def test_nsga_candidate_keeps_penalties_and_persists_solver_failure(
    monkeypatch,
    tmp_path,
):
    from pydac.workflows import orchestrator_NSGA2 as nsga2

    require_summary = {
        "step": "heat",
        "status": -4,
        "message": "corrector convergence failed",
        "nfev": 37,
        "effective_solver": "sparse",
        "final_time_s": 18.25,
    }
    failure = FinalSolverFailure(require_summary)
    run_config = SimpleNamespace(
        is_steam_assisted=True,
        tau_ar=0.5,
        bed_volume=1e-3,
        base_folder=str(tmp_path),
        experiment=object(),
        require_css=True,
        constraints=nsga2.NSGA2Constraints(),
        var_names=("t_ads", "t_heat", "t_des", "P_vac", "v_feed", "v_steam"),
        composition_domain_guard=CompositionDomainGuard(),
    )
    eval_folder = tmp_path / "candidate"

    monkeypatch.setattr(
        nsga2,
        "decision_vector_to_model_inputs",
        lambda *args, **kwargs: ({"t_ads": 900.0}, {"v_feed": 0.01}),
    )
    monkeypatch.setattr(
        nsga2,
        "_format_candidate_summary",
        lambda *args, **kwargs: "candidate",
    )
    monkeypatch.setattr(nsga2, "operating_step_settings", lambda settings: {})
    from pydac.config import load_experiment as real_load_experiment
    provenance_case = real_load_experiment("tests/cases/v3_smoke_startup.json")
    monkeypatch.setattr(nsga2, "load_experiment", lambda *args, **kwargs: provenance_case)
    monkeypatch.setattr(nsga2, "write_resolved_config", lambda *args, **kwargs: None)

    def fail_css(**kwargs):
        assert kwargs["retain_cycles"] == 2
        assert kwargs["composition_domain_guard"] == CompositionDomainGuard()
        assert kwargs["persistence_policy"] == "moo_compact"
        raise failure

    monkeypatch.setattr(nsga2, "run_css_simulation", fail_css)
    monkeypatch.setattr(
        nsga2,
        "CycleDataLoader",
        lambda *args, **kwargs: pytest.fail("KPI loading followed a failed solve"),
    )

    result = nsga2.evaluate_pydac_candidate(
        [900.0, 60.0, 300.0, 3000.0, 0.01, 0.001],
        run_config=run_config,
        eval_folder=str(eval_folder),
    )

    assert result["failed"] is True
    assert result["eq_work"] == nsga2._FAILED_OBJECTIVE
    assert result["neg_productivity"] == nsga2._FAILED_OBJECTIVE
    assert result["oxygen_ppm"] == nsga2._FAILED_CONSTRAINT
    assert result["solver_failure"] == require_summary
    persisted = json.loads(
        (eval_folder / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["solver_failure"] == require_summary

    problem = object.__new__(nsga2.TVSAOptimizationProblem)
    problem.run_config = SimpleNamespace(
        require_css=True,
        is_steam_assisted=True,
        constraints=nsga2.NSGA2Constraints(),
    )
    monkeypatch.setattr(
        nsga2,
        "evaluate_pydac_candidate",
        lambda *args, **kwargs: result,
    )
    out = {}
    problem._evaluate([0.0] * 6, out)
    assert out["G"][0] == nsga2._FAILED_CONSTRAINT
