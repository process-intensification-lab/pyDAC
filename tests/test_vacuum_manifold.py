"""Vacuum pump with pressure control at the setpoint: the 1D outlet law.

The law is documented in docs/manual/05_cycle_steps.md §2.7. Sections: T7 the law,
T5 configuration, schema migration and command-line refusals.
"""

from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np
import pytest

from pydac.config import ConfigurationError, load_experiment
from pydac.config.migrations import (
    CURRENT_SCHEMA_VERSION,
    PRE_VACUUM_OUTLET_SCHEMA_VERSION,
    V8_VACUUM_OUTLET_FIELDS,
    upgrade_experiment_document,
)
from pydac.core.vacuum_manifold import (
    DEFAULT_SETPOINT_BAND_FRACTION,
    VACUUM_PUMP_STEPS,
    VacuumManifold,
)
from pydac.inputs.case_io import case_fingerprint, load_case, validate_case_document


CASE = "tests/cases/v3_smoke_startup.json"
PUMP = {
    "steps.heat.flow.vacuum_outlet_model": "vacuum_pump",
    "steps.heat.flow.vacuum_pump_speed_m3_s": 0.04,
}


def without_vacuum_outlet_fields(document):
    """Drop the schema-9 fields from every step flow: the shape of schema 8 and earlier."""
    from pydac.config.migrations import V9_FEED_BLOWER_FIELDS
    document["simulation"]["grid"].pop("output_schedule", None)
    for step in document["simulation"]["cycle"]["steps"].values():
        for field in (*V8_VACUUM_OUTLET_FIELDS, *V9_FEED_BLOWER_FIELDS, "ads_outlet_overpressure_Pa"):
            step["flow"].pop(field, None)
    return document


# --- T7: the law ------------------------------------------------------------

LAW = VacuumManifold(u_cap=2.0, p_set=0.05, p_ult=0.01, delta=0.05 * 1e-3)


def _x(x):
    return LAW.p_set + x * LAW.delta


def test_the_throttle_is_closed_below_and_fully_open_above_the_band():
    p = np.array([-1.0, 0.0, _x(-0.1), _x(0.0)])
    assert np.array_equal(LAW.outlet_velocity(p), np.zeros(4))
    assert LAW.throughput_fraction(_x(1.0)) == 1.0
    assert LAW.throughput_fraction(_x(7.0)) == 1.0


def test_a_non_positive_trial_pressure_returns_zero_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert LAW.outlet_velocity(np.array([0.0, -3.0])).tolist() == [0.0, 0.0]
        assert LAW.outlet_velocity(0.0) == 0.0


@pytest.mark.parametrize("x", [0.25, 0.5, 0.75])
def test_inside_the_band_the_law_is_the_smoothstep_times_the_pump_curve(x):
    p = _x(x)
    expected = LAW.u_cap * (3 * x**2 - 2 * x**3) * (1 - LAW.p_ult / p)
    np.testing.assert_allclose(LAW.outlet_velocity(p), expected, rtol=1e-12, atol=1e-14)


def test_far_above_the_band_the_pump_curve_alone_sets_the_flow():
    p = np.array([0.2, 1.0])
    np.testing.assert_allclose(
        LAW.outlet_velocity(p), LAW.u_cap * (1 - LAW.p_ult / p), rtol=1e-12, atol=1e-14
    )


def test_the_law_is_c1_monotone_and_bounded_by_the_pump_speed():
    p = np.linspace(LAW.p_set - 2 * LAW.delta, 4 * LAW.p_set, 20001)
    u = LAW.outlet_velocity(p)
    assert (np.diff(u) >= 0).all()
    assert (u >= 0).all() and (u <= LAW.u_cap).all()
    # First derivative continuous at both band edges, by one-sided differences.
    h = LAW.delta * 1e-6
    for edge in (_x(0.0), _x(1.0)):
        left = (LAW.outlet_velocity(edge) - LAW.outlet_velocity(edge - h)) / h
        right = (LAW.outlet_velocity(edge + h) - LAW.outlet_velocity(edge)) / h
        # The one-sided slopes differ by O(h / delta) of the band slope u_cap / delta.
        assert abs(right - left) <= 1e-4 * LAW.u_cap / LAW.delta


def test_an_ideal_pump_and_the_suction_pressure_identity():
    ideal = VacuumManifold(u_cap=1.0, p_set=0.05, p_ult=0.0, delta=0.05e-3)
    assert ideal.outlet_velocity(0.3) == 1.0
    p = np.array([_x(0.3), _x(0.8), 0.3])
    g = LAW.throughput_fraction(p)
    p_s = LAW.suction_pressure(p)
    # Same molar throughput at equal temperature: S(p_s) p_s = g S(p) p.
    np.testing.assert_allclose(p_s - LAW.p_ult, g * (p - LAW.p_ult), rtol=1e-12)
    assert LAW.suction_pressure(_x(-1.0)) == LAW.p_ult


# --- T5: configuration ------------------------------------------------------------


def test_the_reference_case_uses_the_valve_everywhere():
    for step in load_experiment(CASE).simulation.cycle.steps:
        assert step.config.flow.vacuum_outlet_model == "valve"
        assert step.config.flow.vacuum_pump_speed_m3_s is None


@pytest.mark.parametrize("step", sorted(VACUUM_PUMP_STEPS))
def test_the_pump_loads_on_a_vacuum_connected_step(step):
    config = load_experiment(CASE, settings={
        f"steps.{step}.flow.vacuum_outlet_model": "vacuum_pump",
        f"steps.{step}.flow.vacuum_pump_speed_m3_s": 0.04,
    })
    flows = {s.name: s.config.flow for s in config.simulation.cycle.steps}
    assert flows[step].vacuum_outlet_model == "vacuum_pump"
    assert flows[step].vacuum_pump_speed_m3_s == 0.04


@pytest.mark.parametrize(("settings", "message"), [
    ({"steps.ads.flow.vacuum_outlet_model": "vacuum_pump",
      "steps.ads.flow.vacuum_pump_speed_m3_s": 0.04}, "applies only to the vacuum-connected"),
    ({"steps.heat.flow.vacuum_outlet_model": "vacuum_pump"}, "vacuum_pump_speed_m3_s"),
    ({**PUMP, "steps.heat.flow.vacuum_pump_ultimate_pressure_Pa": 1e6},
     "must be below the manifold setpoint"),
    ({"steps.heat.flow.vacuum_pump_speed_m3_s": 0.04}, "read only with"),
    ({"steps.heat.flow.vacuum_setpoint_band_fraction": 1e-3}, "read only with"),
    ({**PUMP, "grid.nr": 3}, "only on a 1D grid"),
    ({**PUMP, "steps.heat.flow.vacuum_pump_speed_m3_s": 0.0}, "vacuum_pump_speed_m3_s"),
    ({**PUMP, "steps.heat.flow.vacuum_setpoint_band_fraction": 0.5}, "band_fraction"),
    ({**PUMP, "steps.heat.flow.vacuum_setpoint_band_fraction": 0.0}, "band_fraction"),
    # A band below 1e-6 is not resolvable next to the setpoint (5e-324 gave NaN at p_set).
    ({**PUMP, "steps.heat.flow.vacuum_setpoint_band_fraction": 1e-7}, "band_fraction"),
    ({"steps.heat.flow.vacuum_outlet_model": "pump"}, "vacuum_outlet_model"),
    ({"steps.heat.flow.vacuum_pump_idle_mode": "fixed_speed"}, "read only with"),
    ({"steps.heat.flow.vacuum_pump_idle_power_W": 3000.0}, "read only with"),
    ({**PUMP, "steps.heat.flow.vacuum_pump_idle_mode": "idle"}, "vacuum_pump_idle_mode"),
    ({**PUMP, "steps.heat.flow.vacuum_pump_idle_power_W": 3000.0}, "draws no standby power"),
    ({**PUMP, "steps.heat.flow.vacuum_pump_idle_mode": "off",
      "steps.heat.flow.vacuum_pump_idle_power_W": 3000.0}, "draws no standby power"),
    ({**PUMP, "steps.heat.flow.vacuum_pump_idle_mode": "fixed_speed"}, "at the ultimate pressure"),
    ({**PUMP, "steps.heat.flow.vacuum_pump_idle_mode": "minimum_speed"}, "at minimum speed"),
    ({**PUMP, "steps.heat.flow.vacuum_pump_idle_mode": "fixed_speed",
      "steps.heat.flow.vacuum_pump_idle_power_W": 0.0}, "vacuum_pump_idle_power_W"),
])
def test_a_declaration_the_model_would_not_read_is_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        load_experiment(CASE, settings=settings)


def test_the_builder_repeats_the_check():
    from pydac.core.vacuum_manifold import build_vacuum_manifold
    from types import SimpleNamespace

    model = SimpleNamespace(
        vacuum_outlet_model="vacuum_pump", step="heat", is_1d=False, alpha_bd=np.inf,
        vacuum_pump_speed_m3_s=0.04, vacuum_pump_ultimate_pressure_Pa=None,
        vacuum_setpoint_band_fraction=None, p_out=5000.0, p_ref=1e5, inlet_area=0.1,
        u_ref=0.1, vacuum_pump_idle_mode=None, vacuum_pump_idle_power_W=None,
    )
    with pytest.raises(ValueError, match="only on a 1D grid"):
        build_vacuum_manifold(model)
    model.is_1d = True
    law = build_vacuum_manifold(model)
    assert law == VacuumManifold(
        u_cap=0.04 / (0.1 * 0.1), p_set=0.05, p_ult=0.0,
        delta=DEFAULT_SETPOINT_BAND_FRACTION * 0.05,
    )
    # The pump does not read the valve conductance: the reference blow keeps its 0.01.
    model.alpha_bd = 0.01
    assert build_vacuum_manifold(model) == law
    model.vacuum_outlet_model = "valve"
    model.vacuum_pump_speed_m3_s = None
    assert build_vacuum_manifold(model) is None


@pytest.mark.parametrize(("change", "message"), [
    ({"vacuum_outlet_model": "valve"}, "read only with"),          # a pump field on the valve
    ({"step": "ads"}, "applies only to the vacuum-connected"),
    ({"vacuum_pump_speed_m3_s": 0.0}, "vacuum_pump_speed_m3_s"),
    ({"vacuum_pump_speed_m3_s": None}, "vacuum_pump_speed_m3_s"),
    ({"vacuum_pump_speed_m3_s": float("nan")}, "finite number"),
    ({"vacuum_pump_speed_m3_s": True}, "finite number"),
    ({"vacuum_pump_ultimate_pressure_Pa": -1.0}, ">= 0"),
    ({"vacuum_pump_ultimate_pressure_Pa": 5000.0}, "below the manifold setpoint"),
    ({"vacuum_setpoint_band_fraction": -1e-3}, "band_fraction"),
    ({"vacuum_setpoint_band_fraction": float("inf")}, "finite number"),
    ({"vacuum_setpoint_band_fraction": 5e-324}, "band_fraction"),
    ({"vacuum_setpoint_band_fraction": 9.9e-7}, "band_fraction"),
    ({"p_out": 0.0}, "positive pressure"),
    ({"vacuum_outlet_model": "pump"}, "vacuum_outlet_model"),
    ({"vacuum_pump_idle_mode": "idle"}, "vacuum_pump_idle_mode"),
    ({"vacuum_pump_idle_power_W": 3000.0}, "draws no standby power"),
    ({"vacuum_pump_idle_mode": "fixed_speed"}, "positive power"),
    ({"vacuum_pump_idle_mode": "minimum_speed", "vacuum_pump_idle_power_W": -1.0}, "positive power"),
    ({"vacuum_pump_idle_mode": "fixed_speed", "vacuum_pump_idle_power_W": float("nan")},
     "finite number"),
    ({"vacuum_outlet_model": "valve", "vacuum_pump_speed_m3_s": None,
      "vacuum_pump_idle_mode": "off"}, "read only with"),
])
def test_the_builder_refuses_what_validation_refuses(change, message):
    from types import SimpleNamespace

    from pydac.core.vacuum_manifold import build_vacuum_manifold

    model = SimpleNamespace(
        vacuum_outlet_model="vacuum_pump", step="heat", is_1d=True, alpha_bd=np.inf,
        vacuum_pump_speed_m3_s=0.04, vacuum_pump_ultimate_pressure_Pa=None,
        vacuum_setpoint_band_fraction=None, p_out=5000.0, p_ref=1e5, inlet_area=0.1,
        u_ref=0.1, vacuum_pump_idle_mode=None, vacuum_pump_idle_power_W=None,
    )
    for name, value in change.items():
        setattr(model, name, value)
    with pytest.raises(ValueError, match=message):
        build_vacuum_manifold(model)


def _blow_model(**settings):
    """A 1D blow model warm-started from the startup state (above P_set: the pump is open)."""
    import contextlib
    import io

    from pydac.config.cycle import CycleConfig
    from pydac.core.column_model_factory import ColumnModelFactory

    config = load_experiment(CASE, settings={"grid.nz": 6, **settings})
    cycle = CycleConfig.from_experiment(config)
    factory = ColumnModelFactory(cycle)
    options = cycle.solver_for_step(use_sparsity=False)
    with contextlib.redirect_stdout(io.StringIO()):
        seed = factory.build_column_model(step="startup", solver_options=options)
        state = {
            "ci": np.asarray(seed.ci).reshape(seed.Nz, seed.Nr, seed.Nc_ci).copy(),
            "qi": np.asarray(seed.qi).reshape(seed.Nz, seed.Nr, seed.Nc_solid).copy(),
            "c_tot": np.asarray(seed.c_tot).reshape(seed.Nz, seed.Nr).copy(),
            "T": np.asarray(seed.T).reshape(seed.Nz, seed.Nr).copy(),
            "T_wall": np.asarray(seed.Tw).reshape(seed.Nz).copy(),
            "Nz": seed.Nz, "Nr": seed.Nr, "layout_signature": seed.state_layout.signature,
        }
        return factory.build_column_model(step="blow", initial_state=state, solver_options=options)


def _rhs(model):
    y = np.zeros(model.state_layout.state_size)
    y[: model.state_layout.pde_size] = model.get_initial_state()
    return model.pde_system_kpis(0.0, y).copy()


def test_the_pump_ignores_any_blowdown_conductance_including_zero():
    pump = {
        "steps.blow.flow.vacuum_outlet_model": "vacuum_pump",
        "steps.blow.flow.vacuum_pump_speed_m3_s": 0.04,
        "steps.blow.pressure.outlet_pressure_Pa": 10000.0,
    }
    reference = _rhs(_blow_model(**pump))       # the reference blow's 0.01 1/m
    assert np.all(np.isfinite(reference))
    for conductance in (0.0, 1e-12, 1e9):
        model = _blow_model(**pump, **{"steps.blow.flow.blowdown_valve_conductance": conductance})
        assert model.bd_fact == 0.0
        np.testing.assert_array_equal(_rhs(model), reference)
    # The valve still reads it.
    valve = _blow_model(**{"steps.blow.flow.blowdown_valve_conductance": 0.5})
    assert valve.bd_fact == pytest.approx(1 / (0.5 * valve.L_b))


BLOW_PUMP = {
    # The smoke case runs only startup, in time mode; these tests need blow, in event mode.
    "execution.cycle_steps": ["startup", "blow"],
    "execution.termination_mode": "event",
    "steps.blow.flow.vacuum_outlet_model": "vacuum_pump",
    "steps.blow.flow.vacuum_pump_speed_m3_s": 0.04,
    "steps.blow.pressure.outlet_pressure_Pa": 10000.0,
}


def _band_warnings(settings):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_experiment(CASE, settings=settings)
    return [str(w.message) for w in caught if "throttle band" in str(w.message)]


@pytest.mark.parametrize(("settings", "expected"), [
    # P_set + 0.1 Pa sits inside the 10 Pa band: only the plateau or the cap can end the step.
    ({**BLOW_PUMP, "steps.blow.termination.threshold_offset": 0.1}, 1),
    # P_set + 0.5 % (50 Pa = 5 delta) is above the band.
    ({**BLOW_PUMP, "steps.blow.termination.threshold_offset": 50.0}, 0),
    # The default 25 Pa offset is inside the band once delta = 1e-3 P_set exceeds it.
    ({**BLOW_PUMP, "steps.blow.pressure.outlet_pressure_Pa": 50000.0}, 1),
    ({**BLOW_PUMP, "steps.blow.termination.threshold_offset": 0.1,
      "execution.termination_mode": "time"}, 0),
    ({"execution.termination_mode": "event", "execution.cycle_steps": ["startup", "blow"],
      "steps.blow.termination.threshold_offset": 0.1}, 0),   # the valve
    # Band edge: P_set + delta = 10010 Pa is still in the band, 10010.5 Pa is above it.
    ({**BLOW_PUMP, "steps.blow.termination.threshold_offset": 10.0}, 1),
    ({**BLOW_PUMP, "steps.blow.termination.threshold_offset": 10.5}, 0),
    # Another pressure metric.
    ({**BLOW_PUMP, "steps.blow.termination.metric": "min_internal_pressure",
      "steps.blow.termination.threshold_offset": 0.1}, 1),
    # A rising target is not a descent target.
    ({**BLOW_PUMP, "steps.blow.termination.direction": "above",
      "steps.blow.termination.threshold_offset": 0.1}, 0),
    # Not executed: the step is declared but not in the run.
    ({**BLOW_PUMP, "execution.cycle_steps": ["startup"], "steps.blow.termination.threshold_offset": 0.1}, 0),
    # The per-step override decides, in both directions.
    ({**BLOW_PUMP, "execution.termination_mode": "time", "steps.blow.termination.threshold_offset": 0.1,
      "execution.step_options": {"blow": {"termination_mode": "event"}}}, 1),
    ({**BLOW_PUMP, "steps.blow.termination.threshold_offset": 0.1,
      "execution.step_options": {"blow": {"termination_mode": "time"}}}, 0),
    # A CSS cycle that runs blow in event mode is enough.
    ({**BLOW_PUMP, "execution.termination_mode": "time", "steps.blow.termination.threshold_offset": 0.1,
      "execution.cycle_steps": ["ads", "blow"],
      "execution.cycle_options": {"2": {"step_options": {"blow": {"termination_mode": "event"}}}}}, 1),
    # A cycle-level mode in any accepted spelling is the mode the runner executes.
    *[({**BLOW_PUMP, "execution.termination_mode": "time", "steps.blow.termination.threshold_offset": 0.1,
        "execution.cycle_steps": ["ads", "blow"],
        "execution.cycle_options": {"1": {"termination_mode": mode}}}, 1)
      for mode in ("event", "EVENT", " event ")],
    # A cycle no CSS run reaches does not count: beyond max_cycles, or before resume_from_cycle.
    ({**BLOW_PUMP, "execution.termination_mode": "time", "steps.blow.termination.threshold_offset": 0.1,
      "execution.cycle_steps": ["ads", "blow"], "execution.max_cycles": 1,
      "execution.cycle_options": {"999": {"termination_mode": "event"}}}, 0),
    ({**BLOW_PUMP, "execution.termination_mode": "time", "steps.blow.termination.threshold_offset": 0.1,
      "execution.cycle_steps": ["ads", "blow"], "execution.resume_from_cycle": 3,
      "execution.cycle_options": {"2": {"termination_mode": "event"}}}, 0),
])
def test_a_pump_target_inside_the_throttle_band_is_warned(settings, expected):
    assert len(_band_warnings(settings)) == expected


def test_a_cycle_termination_mode_is_stored_canonical():
    case = load_experiment(CASE, settings={"execution.cycle_options": {"2": {"termination_mode": " EVENT "}}})
    assert case.execution.cycle_options[2]["termination_mode"] == "event"


def test_the_band_warning_reads_p_feed_and_explicit_values_as_thresholds():
    from types import SimpleNamespace

    from pydac.core.vacuum_manifold import target_inside_band_warning

    termination = SimpleNamespace(metric="max_internal_pressure", direction="below",
                                  threshold_source="p_feed", threshold_value=None, threshold_offset=0.0)
    kwargs = dict(path="p", termination=termination, setpoint_Pa=10000.0, band_fraction=None)
    assert target_inside_band_warning(**kwargs, feed_pressure_Pa=10010.0) is not None
    assert target_inside_band_warning(**kwargs, feed_pressure_Pa=20000.0) is None
    # An explicit value (an explicit null source inherits, so this path is not reachable by --set).
    termination.threshold_source, termination.threshold_value, termination.threshold_offset = None, 10005.0, 0.0
    assert target_inside_band_warning(**kwargs, feed_pressure_Pa=20000.0) is not None
    termination.threshold_value = 10011.0
    assert target_inside_band_warning(**kwargs, feed_pressure_Pa=20000.0) is None


# --- T5: schema 9 migration -------------------------------------------------------


def _complete():
    return validate_case_document(load_case(CASE))


def test_a_complete_v8_case_upgrades_to_the_valve_and_keeps_its_v8_fields():
    v8 = without_vacuum_outlet_fields(_complete())
    v8["schema_version"] = PRE_VACUUM_OUTLET_SCHEMA_VERSION
    startup = v8["simulation"]["cycle"]["steps"]["startup"]["composition"]
    startup["initial_solid_loadings_mol_per_kg"] = {"CO2": 0.25, "H2O": 1.5}
    upgraded = validate_case_document(deepcopy(v8))
    assert upgraded["schema_version"] == CURRENT_SCHEMA_VERSION
    for step in upgraded["simulation"]["cycle"]["steps"].values():
        assert {k: step["flow"][k] for k in V8_VACUUM_OUTLET_FIELDS} == {
            "vacuum_outlet_model": "valve",
            "vacuum_pump_speed_m3_s": None,
            "vacuum_pump_ultimate_pressure_Pa": None,
            "vacuum_setpoint_band_fraction": None,
            "vacuum_pump_idle_mode": None,
            "vacuum_pump_idle_power_W": None,
        }
    loadings = upgraded["simulation"]["cycle"]["steps"]["startup"]["composition"]
    assert loadings["initial_solid_loadings_mol_per_kg"] == {"CO2": 0.25, "H2O": 1.5}


def test_a_complete_v8_case_is_the_same_case_as_its_v9_form():
    v8 = without_vacuum_outlet_fields(_complete())
    v8["schema_version"] = PRE_VACUUM_OUTLET_SCHEMA_VERSION
    assert validate_case_document(deepcopy(v8)) == _complete()
    assert case_fingerprint(deepcopy(v8)) == case_fingerprint(load_case(CASE))


@pytest.mark.parametrize("field", list(V8_VACUUM_OUTLET_FIELDS))
@pytest.mark.parametrize("version", range(1, PRE_VACUUM_OUTLET_SCHEMA_VERSION + 1))
def test_each_v9_field_is_refused_in_every_earlier_schema(field, version):
    document = {"schema_version": version, "simulation": {"cycle": {"steps": {
        "heat": {"flow": {field: None}}}}}}
    with pytest.raises(ConfigurationError, match="is new in pyDAC 3.2.0; set `schema_version` to 11"):
        upgrade_experiment_document(document)


def test_a_v8_partial_preset_states_only_what_it_states():
    preset = {"schema_version": 8, "simulation": {"cycle": {"steps": {
        "heat": {"flow": {"volumetric_flow_m3_s": 0.0}},
        "blow": {"pressure": {"outlet_pressure_Pa": 5000.0}}}}}}
    upgraded = upgrade_experiment_document(preset)
    steps = upgraded["simulation"]["cycle"]["steps"]
    assert steps["heat"]["flow"]["vacuum_outlet_model"] == "valve"
    assert "flow" not in steps["blow"]
    assert load_experiment(preset) == load_experiment(None, settings={
        "steps.heat.flow.volumetric_flow_m3_s": 0.0,
        "steps.blow.pressure.outlet_pressure_Pa": 5000.0,
    })


# --- T5: the same refusals at the command line, before any run is planned ----------------

@pytest.mark.parametrize(("sets", "message"), [
    (["steps.heat.flow.vacuum_pump_speed_m3_s=0.04"], "read only with"),
    (["steps.ads.flow.vacuum_outlet_model=vacuum_pump", "steps.ads.flow.vacuum_pump_speed_m3_s=0.04"],
     "applies only to the vacuum-connected"),
    (["steps.heat.flow.vacuum_outlet_model=vacuum_pump"], "vacuum_pump_speed_m3_s"),
    (["steps.heat.flow.vacuum_outlet_model=vacuum_pump", "steps.heat.flow.vacuum_pump_speed_m3_s=0.04",
      "grid.nr=3"], "only on a 1D grid"),
    (["steps.heat.flow.vacuum_outlet_model=vacuum_pump", "steps.heat.flow.vacuum_pump_speed_m3_s=0.04",
      "steps.heat.flow.vacuum_pump_ultimate_pressure_Pa=1e6"], "below the manifold setpoint"),
])
@pytest.mark.parametrize("command", ["validate", "steps"])
def test_the_command_line_refuses_a_bad_pump_declaration(command, sets, message, capsys):
    from pydac.cli import main

    argv = [command, "--config", CASE]
    for item in sets:
        argv += ["--set", item]
    if command == "steps":
        argv.append("--dry-run")
    assert main(argv) == 2
    assert message in capsys.readouterr().err


def test_an_explicit_null_pump_field_inherits_the_preset_value(tmp_path):
    """Explicit null on a step field means inherit (config/nulls.py), not reset."""
    import json

    flow = {"vacuum_outlet_model": "vacuum_pump", "vacuum_pump_speed_m3_s": 0.04,
            "vacuum_pump_ultimate_pressure_Pa": 100.0, "vacuum_setpoint_band_fraction": 1e-2}
    preset = {"schema_version": CURRENT_SCHEMA_VERSION,
              "simulation": {"grid": {"nr": 1}, "cycle": {"steps": {"heat": {"flow": flow}}}}}
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(preset))
    flow = {s.name: s.config.flow for s in load_experiment(str(path), settings={
        "steps.heat.flow.vacuum_pump_speed_m3_s": None,
        "steps.heat.flow.vacuum_pump_ultimate_pressure_Pa": None,
        "steps.heat.flow.vacuum_setpoint_band_fraction": None,
    }).simulation.cycle.steps}["heat"]
    assert (flow.vacuum_pump_speed_m3_s, flow.vacuum_pump_ultimate_pressure_Pa,
            flow.vacuum_setpoint_band_fraction) == (0.04, 100.0, 1e-2)


# --- Standby power ----------------------------------------------------------------


def test_the_idle_power_is_zero_in_the_off_mode_and_weighted_by_one_minus_g():
    p = np.array([_x(-1.0), _x(0.0), _x(0.25), _x(0.5), _x(1.0), _x(3.0)])
    assert LAW.p_idle_W == 0.0 and LAW.idle_power(p) == 0.0          # exactly 0.0, not an array
    assert type(LAW.idle_power(p)) is float                           # a Python float, no NumPy scalar
    idle = VacuumManifold(u_cap=2.0, p_set=0.05, p_ult=0.01, delta=0.05 * 1e-3, p_idle_W=2500.0)
    np.testing.assert_allclose(idle.idle_power(p), 2500.0 * (1.0 - LAW.throughput_fraction(p)),
                               rtol=0, atol=1e-12)
    assert idle.idle_power(_x(-1.0)) == 2500.0 and idle.idle_power(_x(1.0)) == 0.0


@pytest.mark.parametrize("mode", ["minimum_speed", "fixed_speed"])
def test_the_builder_reads_the_idle_power_of_a_declared_mode(mode):
    from types import SimpleNamespace

    from pydac.core.vacuum_manifold import build_vacuum_manifold

    model = SimpleNamespace(
        vacuum_outlet_model="vacuum_pump", step="heat", is_1d=True, alpha_bd=np.inf,
        vacuum_pump_speed_m3_s=0.04, vacuum_pump_ultimate_pressure_Pa=None,
        vacuum_setpoint_band_fraction=None, p_out=5000.0, p_ref=1e5, inlet_area=0.1,
        u_ref=0.1, vacuum_pump_idle_mode=mode, vacuum_pump_idle_power_W=3000.0,
    )
    assert build_vacuum_manifold(model).p_idle_W == 3000.0
    model.vacuum_pump_idle_mode, model.vacuum_pump_idle_power_W = None, None
    assert build_vacuum_manifold(model).p_idle_W == 0.0


def test_a_declared_idle_mode_loads_case_insensitive():
    config = load_experiment(CASE, settings={
        **PUMP, "steps.heat.flow.vacuum_pump_idle_mode": "Fixed_Speed",
        "steps.heat.flow.vacuum_pump_idle_power_W": 3600.0,
    })
    flow = {s.name: s.config.flow for s in config.simulation.cycle.steps}["heat"]
    assert (flow.vacuum_pump_idle_mode, flow.vacuum_pump_idle_power_W) == ("fixed_speed", 3600.0)
