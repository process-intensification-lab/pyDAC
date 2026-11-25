"""The vacuum pump's standby power P_idle (1 - g) in the solver's energy.

Documented in docs/manual/06_post_processing.md §8.2. With the mode "off"
(the default) nothing changes; a declared mode adds exactly P_idle (1 - g) to the
vacuum power and nothing else.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.core.outlet_reporting import outlet_face_history
from pydac.core.solver import Solver
from pydac.core.vacuum_manifold import VacuumManifold
from tests.neutral_packed_bed import neutral_profile
from tests.test_profile_inventory_outlet import _prepare


def _vacuum_power(model, state):
    """Return the vacuum power [W] the RHS integrates, and the whole RHS."""
    rhs = model.pde_system_kpis(0, state).copy()
    field = model.state_layout.field("E_vacuum")
    return rhs[field.state_slice].item() * model.state_scale_by_field[field.name] / model.tau_hyd, rhs


def test_a_declared_mode_adds_p_idle_times_one_minus_g_and_changes_nothing_else(tmp_path, monkeypatch):
    model, _, history = _prepare(tmp_path, nr=1, pump=True)
    state = history[:, 0].copy()
    base, rhs_off = _vacuum_power(model, state)
    c_tot = state[model.state_field_by_name["c_tot"].state_slice]
    T_bed = state[model.state_field_by_name["T_bed"].state_slice]
    g = model.vacuum_manifold.throughput_fraction((c_tot * T_bed).reshape(model.Nz, model.Nr)[-1, 0])
    assert 0.0 < g < 1.0
    from dataclasses import replace
    monkeypatch.setattr(model, "vacuum_manifold", replace(model.vacuum_manifold, p_idle_W=3000.0))
    idle, rhs_idle = _vacuum_power(model, state)
    assert idle - base == pytest.approx(3000.0 * (1.0 - g), rel=1e-12)
    field = model.state_layout.field("E_vacuum").state_slice
    mask = np.ones(rhs_off.size, dtype=bool)
    mask[field] = False
    np.testing.assert_array_equal(rhs_idle[mask], rhs_off[mask])     # only the energy moves


def _closed_blow(mode, power, tmp_path, t_end=30.0):
    """A blow held below its setpoint: the control stays closed (g = 0) for the whole step."""
    profile = neutral_profile()
    profile["simulation"]["grid"].update(nz=6, nr=1)
    steps = profile["simulation"]["cycle"]["steps"]
    steps["startup"]["composition"]["co2_initial_mole_fraction"] = 0.1
    steps["blow"]["flow"].update(vacuum_outlet_model="vacuum_pump", vacuum_pump_speed_m3_s=0.04,
                                 vacuum_pump_idle_mode=mode, vacuum_pump_idle_power_W=power)
    steps["blow"]["pressure"]["outlet_pressure_Pa"] = 2.0e5        # above the bed: closed
    steps["blow"]["termination"]["mode"] = "time"
    steps["blow"]["horizon"].update(nominal_time_s=t_end, event_max_time_s=t_end)
    cycle = CycleConfig.from_experiment(load_experiment(default_profile=profile))
    factory = ColumnModelFactory(cycle)
    options = cycle.solver_for_step(use_sparsity=False)
    with contextlib.redirect_stdout(io.StringIO()):
        start = factory.build_column_model(step="startup", solver_options=options)
        warm = {"Nz": start.Nz, "Nr": start.Nr, "ci": start.ci.copy(), "qi": start.qi.copy(),
                "c_tot": start.c_tot[..., 0].copy(), "T": start.T[..., 0].copy(),
                "T_wall": start.Tw[:, 0, 0].copy(), "layout_signature": start.state_layout.signature}
        model = factory.build_column_model(step="blow", initial_state=warm, solver_options=options)
        solver = Solver(model, base_folder=str(tmp_path))
        solver.solve_cvode()
    assert solver.sol["cvode"]["success"]
    outlet = outlet_face_history(model, solver.sol["y"])
    assert np.all(outlet["velocity_m_s"] == 0.0)                                  # closed throughout
    field = model.state_layout.field("E_vacuum")
    energy = solver.sol["y"][field.state_slice].ravel() * model.state_scale_by_field[field.name]
    return np.asarray(solver.sol["t"], dtype=float) * model.tau_hyd, energy   # raw time is t / tau_hyd


def test_a_closed_step_charges_p_idle_for_its_whole_duration(tmp_path):
    t, energy = _closed_blow("fixed_speed", 2500.0, tmp_path / "idle")
    assert t[-1] == pytest.approx(30.0)
    np.testing.assert_allclose(energy, 2500.0 * t, rtol=1e-6, atol=1e-6)


def test_a_closed_step_in_the_off_mode_charges_nothing(tmp_path):
    t, energy = _closed_blow(None, None, tmp_path / "off")
    assert t[-1] == pytest.approx(30.0)
    assert np.all(energy == 0.0)


def _condenser_history(law):
    """A saved-history stand-in for update_vacuum_pump_energy: the last cell crosses the band.

    The pressure falls from above P_set + delta to P_set, so g runs through the whole
    smoothstep; condenser vapour and inerts are small and constant.
    """
    import pandas as pd
    from types import SimpleNamespace

    from pydac.post_processing.profiles.step_post_processing import StepPostProcessor

    p_ref, p_set = 1e5, 1e4
    time = np.linspace(0.0, 600.0, 601)
    p_last = p_set * (1.0 + 2e-3 * np.exp(-time / 100.0))          # x from 2 to ~0
    model = SimpleNamespace(cp_g_mix=29.1, p_amb=101325.0, p_ref=p_ref, vacuum_manifold=law)
    history = SimpleNamespace(
        pb=model, time=time, A_rings=np.array([0.05]), p_tot=p_last.reshape(1, 1, -1),
        gas_species=("CO2", "H2O", "N2"), n_N2_out=np.full(time.size, 1e-5),
        condenser_results_df=pd.DataFrame({"nv": np.full(time.size, 2e-5)}),
    )
    history._inert_species = lambda: StepPostProcessor._inert_species(history)
    return StepPostProcessor.update_vacuum_pump_energy(history), time, p_last / p_ref


def test_the_condenser_adjusted_energy_adds_the_same_standby_term():
    from dataclasses import replace

    from scipy.integrate import simpson

    off = VacuumManifold(u_cap=1.0, p_set=0.1, p_ult=0.0, delta=1e-4)
    e_off, time, p = _condenser_history(off)
    e_idle, _, _ = _condenser_history(replace(off, p_idle_W=3000.0))
    x = np.clip((p - 0.1) / 1e-4, 0.0, 1.0)                          # independent smoothstep
    g = x * x * (3.0 - 2.0 * x)
    assert 0.0 < g.min() < 1e-3 and g.max() == 1.0                    # the band is crossed
    expected = simpson(3000.0 * (1.0 - g), x=time) / 1e6
    assert e_idle - e_off == pytest.approx(expected, rel=1e-12)
    assert e_off > 0.0


def test_the_off_mode_retains_zero_with_the_clamped_expansion_bracket(tmp_path, monkeypatch):
    """A closed outlet above ambient charges zero after the schema-11 expansion clamp."""
    from dataclasses import replace

    model, _, history = _prepare(tmp_path, nr=1, pump=True)       # last cell ~101333 Pa > p_amb
    closed = replace(model.vacuum_manifold, p_set=101340.0 / model.p_ref)
    monkeypatch.setattr(model, "vacuum_manifold", closed)
    power, _ = _vacuum_power(model, history[:, 0].copy())
    assert power == 0.0 and not np.signbit(power)                  # no expansion credit


def _maruyama(p_Pa):
    """Maruyama et al. (2020) Eq. 10, the pressure in atm."""
    s = 19.55 * np.asarray(p_Pa, dtype=float) / 101325.0
    return 0.8 * s / (1.0 + s)


def test_the_solver_vacuum_power_uses_maruyama_with_the_pressure_in_atm(tmp_path):
    from scipy.constants import gas_constant as R

    model, _, history = _prepare(tmp_path, nr=1, pump=True)
    state = history[:, 0].copy()
    power, _ = _vacuum_power(model, state)
    c_tot = state[model.state_field_by_name["c_tot"].state_slice] * model.c_tot_ref
    T_bed = state[model.state_field_by_name["T_bed"].state_slice] * model.T_ref
    p = (c_tot * R * T_bed).reshape(model.Nz, model.Nr)[-1, 0]
    q_out = model.vacuum_manifold.outlet_velocity(p / model.p_ref) * model.u_ref * model.A_rings[0]
    gamma = model.cp_g_mix / (model.cp_g_mix - R)
    k = (gamma - 1.0) / gamma
    expected = p * max(q_out, 0.) / (k * _maruyama(p)) * max((model.p_amb / p) ** k - 1.0, 0.)
    assert power == pytest.approx(expected, rel=1e-9)


def test_the_condenser_adjusted_energy_uses_maruyama_with_the_pressure_in_atm():
    from scipy.constants import gas_constant as R
    from scipy.integrate import simpson

    off = VacuumManifold(u_cap=1.0, p_set=0.1, p_ult=0.0, delta=1e-4)
    energy, time, p_nd = _condenser_history(off)
    p = p_nd * 1e5
    gamma = 29.1 / (29.1 - R)
    k = (gamma - 1.0) / gamma
    q = 3e-5 * R * 278.15 / p                                        # vapour + inerts at T_cond
    power = p * q / (k * _maruyama(p)) * ((101325.0 / p) ** k - 1.0)
    assert energy == pytest.approx(simpson(power, x=time) / 1e6, rel=1e-12)
