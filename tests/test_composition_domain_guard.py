"""Permissive, RHS-neutral gas-closure failure guard."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pydac.core.solver import Solver
from pydac.utils.solver.composition_domain import (
    CompositionDomainGuard,
    build_composition_domain_event_spec,
    closure_domain_diagnostic,
)
from pydac.utils.solver.events import build_termination_summary
from pydac.utils.solver.outcome import FinalSolverFailure, require_final_solver_success


class _Field:
    def __init__(self, start):
        self.state_slice = slice(start, start + 1)


class _Layout:
    closure_species = "N2"
    independent_gas_species = ("CO2", "H2O", "O2")

    def field(self, name):
        return {
            "c_CO2": _Field(0),
            "c_H2O": _Field(1),
            "c_O2": _Field(2),
            "c_tot": _Field(3),
        }[name]


def _model():
    return SimpleNamespace(
        state_layout=_Layout(),
        Nz=1,
        Nr=1,
        c_tot_ref=10.0,
        c_ref_by_species={"CO2": 10.0, "H2O": 10.0, "O2": 10.0},
        z_c=np.array([0.5]),
        r_c=np.array([0.5]),
        L_b=2.0,
        R_b=0.1,
    )


def test_guard_allows_small_negative_closure_but_rejects_large_undershoot():
    guard = CompositionDomainGuard(
        absolute_allowance_mol_m3=1.0,
        relative_allowance=0.05,
    )
    model = _model()

    accepted = closure_domain_diagnostic(
        np.array([0.60, 0.49, 0.0, 1.0]), model=model, guard=guard
    )
    rejected = closure_domain_diagnostic(
        np.array([0.60, 0.51, 0.0, 1.0]), model=model, guard=guard
    )

    assert accepted["closure_concentration_mol_m3"] == pytest.approx(-0.9)
    assert accepted["closure_margin_mol_m3"] == pytest.approx(0.1)
    assert rejected["closure_concentration_mol_m3"] == pytest.approx(-1.1)
    assert rejected["closure_margin_mol_m3"] == pytest.approx(-0.1)


def test_relative_allowance_wins_at_high_total_concentration():
    guard = CompositionDomainGuard(
        absolute_allowance_mol_m3=1.0,
        relative_allowance=0.05,
    )
    diagnostic = closure_domain_diagnostic(
        np.array([2.3, 2.0, 0.0, 4.0]), model=_model(), guard=guard
    )

    assert diagnostic["closure_concentration_mol_m3"] == pytest.approx(-3.0)
    assert diagnostic["allowed_undershoot_mol_m3"] == pytest.approx(2.0)
    assert diagnostic["closure_margin_mol_m3"] == pytest.approx(-1.0)


def test_domain_event_becomes_a_failed_termination_with_compact_diagnostic():
    guard = CompositionDomainGuard(
        absolute_allowance_mol_m3=1.0,
        relative_allowance=0.05,
    )
    spec = build_composition_domain_event_spec(_model(), guard)
    state = np.array([0.60, 0.51, 0.0, 1.0])
    raw = SimpleNamespace(
        success=True,
        status=0,
        message="event reached",
        t=np.array([0.0, 1.9]),
        t_events=np.array([2.0]),
        y_events=np.array([state]),
        i_events=np.array([[-1]], dtype=np.int32),
    )

    summary = build_termination_summary(
        raw,
        [spec],
        tau_hyd=10.0,
        configured_mode="time",
    )

    assert summary["success"] is False
    assert summary["reason"] == "state_domain_violation"
    assert summary["status"] == "state_domain_violation"
    assert summary["event"]["t_event_s"] == pytest.approx(20.0)
    assert summary["event"]["metadata"] == guard.as_dict()
    assert summary["event"]["diagnostic"][
        "closure_concentration_mol_m3"
    ] == pytest.approx(-1.1)


def test_initial_domain_violation_fails_before_cvode_and_reaches_existing_gate():
    guard = CompositionDomainGuard(
        absolute_allowance_mol_m3=1.0,
        relative_allowance=0.05,
    )
    spec = build_composition_domain_event_spec(_model(), guard)
    state = np.array([0.60, 0.51, 0.0, 1.0])
    solver = object.__new__(Solver)
    solver.event_specs = [spec]
    solver.tau_hyd = 10.0
    solver.termination_mode = "time"

    assert solver._handle_initial_event(state) is True
    with pytest.raises(FinalSolverFailure) as caught:
        require_final_solver_success(solver.sol, step="s_des")

    summary = caught.value.summary
    assert summary["status"] == "state_domain_violation"
    assert summary["failure_kind"] == "state_domain_violation"
    assert summary["state_domain"]["guard"] == guard.as_dict()


def test_initial_domain_failure_precedes_an_already_satisfied_normal_event():
    guard = CompositionDomainGuard(
        absolute_allowance_mol_m3=1.0,
        relative_allowance=0.05,
    )
    domain_spec = build_composition_domain_event_spec(_model(), guard)
    normal_spec = {
        "name": "step_termination",
        "metric_fn": lambda y: -1.0,
        "threshold_nd": 0.0,
        "threshold_dim": 0.0,
        "direction": -1,
        "label": "ordinary event",
        "metric_label": "ordinary metric",
        "unit": "fraction",
        "ref": 1.0,
    }
    solver = object.__new__(Solver)
    solver.event_specs = [normal_spec, domain_spec]
    solver.tau_hyd = 10.0
    solver.termination_mode = "event"

    assert solver._handle_initial_event(np.array([0.60, 0.51, 0.0, 1.0]))
    assert solver.sol["termination"]["reason"] == "state_domain_violation"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"absolute_allowance_mol_m3": -1.0},
        {"relative_allowance": -0.01},
        {
            "absolute_allowance_mol_m3": 0.0,
            "relative_allowance": 0.0,
        },
    ],
)
def test_invalid_enabled_guards_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CompositionDomainGuard(**kwargs)
