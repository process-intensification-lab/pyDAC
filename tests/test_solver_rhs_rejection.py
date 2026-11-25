"""S1: a ValueError at a non-physical Newton trial state is rejected (NaN ydot), not raised.

Contract (core/solver.py, Solver.pde_callback): only ValueError, only at a non-physical
state; a ValueError at a physical state is raised unchanged; at most
MAX_RHS_REJECTIONS per solve, then a ValueError chained to the original; the count is
in sol["cvode"] only when > 0; a successful solve whose saved t, y or event arrays are not
finite fails, with or without rejections, also on the initial-event path.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.core.solver import Solver


CASE = "tests/cases/v3_smoke_startup.json"


def _solver(tmp_path, nt=None):
    settings = {"grid.nz": 6}
    if nt is not None:
        settings["grid.nt"] = nt
    cycle = CycleConfig.from_experiment(load_experiment(CASE, settings=settings))
    with contextlib.redirect_stdout(io.StringIO()):
        model = ColumnModelFactory(cycle).build_column_model(
            step="startup", initial_state=None,
            solver_options=cycle.solver_for_step(use_sparsity=False, compute_numerical_sparsity=False),
        )
        return Solver(model, base_folder=str(tmp_path))


def _state(solver):
    y = np.zeros(solver.state_layout.state_size)
    y[: solver.state_layout.pde_size] = solver.get_initial_state()
    return y


def _slice(solver, name):
    return solver.state_layout.field(name).state_slice


def test_the_physical_state_check(tmp_path):
    solver = _solver(tmp_path)
    y = _state(solver)
    assert solver._state_is_physical(y)
    for name in ("c_tot", "T_bed", "T_wall"):
        bad = y.copy()
        bad[_slice(solver, name)][0] = 0.0
        assert not solver._state_is_physical(bad), name
    for value in (np.nan, np.inf):
        bad = y.copy()
        bad[-1] = value                       # a scalar state counts too
        assert not solver._state_is_physical(bad)
    gas = [f for f in solver.state_layout.fields if f.kind == "gas concentration"]
    bad = y.copy()
    bad[gas[0].state_slice][0] = -1e-3
    assert not solver._state_is_physical(bad)
    # The closure species is what remains of c_tot: an overfull independent set is non-physical.
    # The comparison is dimensional, so it holds whatever the field scales are.
    bad = y.copy()
    c_tot = bad[_slice(solver, "c_tot")][0] * solver.state_scale_by_field["c_tot"]
    bad[gas[0].state_slice][0] = 2.0 * c_tot / solver.state_scale_by_field[gas[0].name]
    assert not solver._state_is_physical(bad)


def test_a_value_error_at_a_physical_state_is_raised_unchanged(tmp_path):
    solver = _solver(tmp_path)
    error = ValueError("genuine")

    def failing(t, y, out):
        raise error

    solver.pde_system = failing
    y = _state(solver)
    with pytest.raises(ValueError) as caught:
        solver.pde_callback(0.0, y, np.zeros_like(y))
    assert caught.value is error
    assert solver.rhs_rejections == 0


def test_a_non_physical_state_is_rejected_up_to_the_cap_then_raised_chained(tmp_path):
    solver = _solver(tmp_path)
    error = ValueError("drag law")

    def failing(t, y, out):
        raise error

    solver.pde_system = failing
    y = _state(solver)
    y[_slice(solver, "c_tot")][0] = -1.0
    ydot = np.zeros_like(y)
    for k in range(solver.MAX_RHS_REJECTIONS):
        solver.pde_callback(0.5, y, ydot)
        assert np.all(np.isnan(ydot))
    assert solver.rhs_rejections == solver.MAX_RHS_REJECTIONS
    assert solver.first_rhs_rejection == (0.5 * solver.tau_hyd, "ValueError: drag law")
    with pytest.raises(ValueError, match="cap of one solve") as caught:
        solver.pde_callback(0.6, y, ydot)
    assert caught.value.__cause__ is error


def test_an_untouched_solve_saves_no_rejection_keys(tmp_path):
    solver = _solver(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_cvode()
    assert solver.sol["cvode"]["success"]
    assert "rhs_rejections" not in solver.sol["cvode"]
    assert "first_rhs_rejection" not in solver.sol["cvode"]


def test_cvode_recovers_from_transient_rejections_and_reports_them(tmp_path):
    """Installed CVODE: a few rejected trial evaluations, then a normal solve."""
    reference = _solver(tmp_path / "ref")
    with contextlib.redirect_stdout(io.StringIO()):
        reference.solve_cvode()

    solver = _solver(tmp_path / "s1")
    original = solver.pde_system
    calls = {"n": 0, "injected": False}

    def flaky(t, y, out):
        calls["n"] += 1
        calls["injected"] = calls["n"] in (40, 41, 42)
        if calls["injected"]:
            raise ValueError("injected")
        return original(t, y, out=out)

    solver.pde_system = flaky
    physical = solver._state_is_physical
    solver._state_is_physical = lambda y: physical(y) and not calls["injected"]
    solver.rhs_rejections = 7                 # stale count from an earlier solve: reset at entry
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_cvode()
    cvode = solver.sol["cvode"]
    assert cvode["success"]
    # The NaN derivatives also reach CVODE's difference-quotient Jacobian, so the next
    # trial states are not finite either: about 25 rejections per incident (measured
    # 2026-10-24 on the dense and sparse paths), all well inside the cap.
    assert 3 <= cvode["rhs_rejections"] < 100
    assert cvode["first_rhs_rejection"][1] == "ValueError: injected"
    assert np.all(np.isfinite(solver.sol["y"]))
    # The retried steps converge to the same solution, within the solver tolerances (measured 5e-5).
    np.testing.assert_allclose(solver.sol["y"][:, -1], reference.sol["y"][:, -1], rtol=1e-3, atol=1e-8)


def _fake_result(t_eval, y0, *, nan_in="y"):
    from types import SimpleNamespace

    t = np.asarray(t_eval, dtype=float)
    values = np.tile(np.asarray(y0, dtype=float)[:, None], (1, len(t)))
    t_events = y_events = i_events = None
    if nan_in == "y":
        values[0, -1] = np.nan
    elif nan_in == "y_events":
        t_events, y_events, i_events = np.array([t[-1]]), values[:, -1:].T.copy(), np.array([[0]])
        y_events[0, 0] = np.nan
    return SimpleNamespace(success=True, status=0, message="ok", t=t, y=values.T, nfev=1, njev=0,
                           t_events=t_events, y_events=y_events, i_events=i_events)


def test_a_reported_success_with_a_non_finite_output_fails_after_rejections(tmp_path, monkeypatch):
    import pydac.core.solver as solver_module

    class FakeCVODE:
        """Rejects one trial state, then claims success with a NaN trajectory."""

        def __init__(self, callback, **options):
            self.callback = callback

        def solve(self, t_eval, y0):
            y = np.array(y0, dtype=float)
            y[0] = -1.0
            self.callback(0.0, y, np.zeros_like(y))
            return _fake_result(t_eval, y0)

    solver = _solver(tmp_path)

    def failing(t, y, out):
        raise ValueError("trial")

    solver.pde_system = failing
    monkeypatch.setattr(solver_module, "CVODE", FakeCVODE)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_cvode()
    cvode = solver.sol["cvode"]
    assert cvode["rhs_rejections"] == 1
    assert cvode["success"] is False
    assert "not finite" in cvode["message"]
    assert solver.sol["termination"]["success"] is False
    assert solver.sol["termination"]["reason"] == "nonfinite_output"


@pytest.mark.parametrize("nan_in", ["y", "y_events"])
def test_a_reported_success_with_a_non_finite_output_fails_without_rejections(
        tmp_path, monkeypatch, nan_in):
    import pydac.core.solver as solver_module

    class FakeCVODE:
        """Claims success with a NaN in the trajectory or in the event states; no rejection."""

        def __init__(self, callback, **options):
            pass

        def solve(self, t_eval, y0):
            return _fake_result(t_eval, y0, nan_in=nan_in)

    solver = _solver(tmp_path)
    monkeypatch.setattr(solver_module, "CVODE", FakeCVODE)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_cvode()
    cvode = solver.sol["cvode"]
    assert "rhs_rejections" not in cvode
    assert cvode["success"] is False and cvode["status"] == "nonfinite_output"
    assert nan_in in cvode["message"]
    assert solver.sol["termination"]["success"] is False


def _initially_satisfied_event(solver):
    """A step-termination event already met by every state (metric 0 below threshold 1)."""
    solver.event_specs = [{
        "name": "step_termination", "label": "test", "metric_label": "p", "unit": "Pa",
        "threshold_dim": 1.0, "threshold_nd": 1.0, "direction": -1, "ref": 1.0,
        "metric_fn": lambda y: 0.0,
    }]


def test_the_initial_event_path_resets_the_count_and_refuses_a_non_finite_state(tmp_path):
    solver = _solver(tmp_path / "finite")
    _initially_satisfied_event(solver)
    solver.rhs_rejections, solver.first_rhs_rejection = 27, (1.0, "stale")
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_cvode()
    assert solver.sol["termination"]["status"] == "initial_event"
    assert solver.sol["cvode"]["success"] is True
    assert (solver.rhs_rejections, solver.first_rhs_rejection) == (0, None)

    solver = _solver(tmp_path / "nan")
    _initially_satisfied_event(solver)
    initial_state = solver._initial_state

    def with_nan():
        y0 = initial_state()
        y0[-1] = np.nan                       # an otherwise unused scalar
        return y0

    solver._initial_state = with_nan
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_cvode()
    assert solver.sol["cvode"]["success"] is False
    assert solver.sol["cvode"]["status"] == "nonfinite_output"
    assert solver.sol["termination"]["success"] is False
