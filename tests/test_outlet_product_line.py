"""Per-step outlet switches: where a step's outlet goes, and what the KPIs read back.

`steps.<step>.outlet` decides whether a step's outlet reaches the product line
(`collect_product`), whether a condenser stands before the vacuum pump (`condenser`)
and the temperature it cools to. The defaults preserve the historical product-line
routing and condenser settings; these tests cover the non-default lines, the
refusals and the schema-v3 upgrade.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pydac
from pydac.config import ConfigurationError, load_experiment
from pydac.config.cycle import CycleConfig
from pydac.config.migrations import (
    CURRENT_SCHEMA_VERSION,
    PRE_OUTLET_SCHEMA_VERSION,
    PRE_VALVE_RENAME_SCHEMA_VERSION,
    PRE_WALL_RENAME_SCHEMA_VERSION,
    V4_VALVE_FIELD_RENAMES,
    V5_WALL_FIELD_RENAMES,
    upgrade_experiment_document,
)
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.core.solver import Solver
from pydac.inputs.case_io import validate_case_document
from pydac.inputs.defaults import default_profile
from pydac.post_processing.kpi.cycle_kpis import CycleKPIs
from pydac.post_processing.metrics.cycle_data import CycleData
from pydac.post_processing.metrics.cycle_metrics import (
    CYCLE_METRIC_COLUMNS,
    CycleMetrics,
)
from pydac.post_processing.profiles.step_post_processing import StepPostProcessor
from pydac.utils.css.runner import validate_css_case
from pydac.utils.postprocessing.normalization import STEP_KEY_TO_CYCLE_STEP
import pydac.utils.postprocessing.context_schema as context_schema
from pydac.post_processing.context.simulation_context import SimulationContext
from tests.test_energy_ledger_gate import _context_row
from tests.test_sorbent_declarations import _with_v6_sorbent_fields


CASE = "tests/cases/v3_smoke_startup.json"
COLLECTED = {"s_purge", "w_des", "s_des"}


def _outlet(collect, condenser, temperature=278.15):
    return SimpleNamespace(
        collect_product=collect,
        condenser=condenser,
        condenser_temperature_K=temperature,
    )


def _ids(metrics, metric_id):
    return {
        r.species: r.value for r in metrics.records if r.definition.metric_id == metric_id
    }


# --- configuration ------------------------------------------------------------------


def test_defaults_collect_through_a_condenser_exactly_the_desorption_steps():
    config = load_experiment(CASE)
    for step in config.simulation.cycle.steps:
        outlet = step.config.outlet
        assert outlet.collect_product is (step.name in COLLECTED)
        assert outlet.condenser is (step.name in COLLECTED)
        assert outlet.condenser_temperature_K == 278.15


@pytest.mark.parametrize("step", ["startup", "ads", "cool", "press"])
def test_a_step_without_a_product_outlet_cannot_collect(step):
    with pytest.raises(ConfigurationError, match="collect_product` must be false"):
        load_experiment(CASE, settings={f"steps.{step}.outlet.collect_product": True})


def test_a_condenser_needs_a_collected_step():
    with pytest.raises(ConfigurationError, match="requires `outlet.collect_product`"):
        load_experiment(CASE, settings={"steps.s_des.outlet.collect_product": False})
    vented = load_experiment(
        CASE,
        settings={
            "steps.s_des.outlet.collect_product": False,
            "steps.s_des.outlet.condenser": False,
        },
    )
    steps = {step.name: step.config for step in vented.simulation.cycle.steps}
    assert not steps["s_des"].outlet.collect_product


def test_a_css_cycle_that_collects_nowhere_is_refused_before_it_runs():
    settings = {
        "execution.cycle_steps": ["ads", "blow", "heat", "s_des", "cool", "press"],
        "steps.s_des.outlet.collect_product": False,
        "steps.s_des.outlet.condenser": False,
    }
    with pytest.raises(ValueError, match="collects product"):
        validate_css_case(load_experiment(CASE, settings=settings))
    # The same steps with a collected preheating are a valid cycle.
    settings["steps.heat.outlet.collect_product"] = True
    assert validate_css_case(load_experiment(CASE, settings=settings))


def _with_v5_wall_names(document):
    """Give the wall override the name schema v5 (and every earlier one) used."""
    wall = document["simulation"]["wall"]
    for old, new in V5_WALL_FIELD_RENAMES.items():
        wall[old] = wall.pop(new)
    return _with_v6_sorbent_fields(document)


def _with_v4_valve_names(document):
    """Give every step the valve-conductance names schema v4 used (and the v5 wall name)."""
    for step in document["simulation"]["cycle"]["steps"].values():
        for old, new in V4_VALVE_FIELD_RENAMES.items():
            step["flow"][old] = step["flow"].pop(new)
    return _with_v5_wall_names(document)


def _v3_document():
    document = _with_v4_valve_names(deepcopy(default_profile()))
    document["schema_version"] = PRE_OUTLET_SCHEMA_VERSION
    for step in document["simulation"]["cycle"]["steps"].values():
        del step["outlet"]
    del document["simulation"]["geometry"]["specified_by"]
    return document


def test_a_schema_v3_case_upgrades_to_the_lines_it_computed_with():
    upgraded = upgrade_experiment_document(_v3_document())
    assert upgraded["schema_version"] == CURRENT_SCHEMA_VERSION
    assert upgraded == default_profile()


def test_a_schema_v3_case_with_an_outlet_group_is_refused():
    document = _v3_document()
    document["simulation"]["cycle"]["steps"]["s_des"]["outlet"] = {}
    with pytest.raises(ConfigurationError, match="exists only since case schema 4"):
        upgrade_experiment_document(document)


def test_a_schema_v4_case_gets_the_unit_free_valve_names_with_the_same_values():
    document = _with_v4_valve_names(deepcopy(default_profile()))
    document["schema_version"] = PRE_VALVE_RENAME_SCHEMA_VERSION
    document["simulation"]["cycle"]["steps"]["blow"]["flow"][
        "blowdown_valve_conductance_m_s"
    ] = 0.02
    upgraded = upgrade_experiment_document(document)
    assert upgraded["schema_version"] == CURRENT_SCHEMA_VERSION
    assert upgraded["simulation"]["cycle"]["steps"]["blow"]["flow"][
        "blowdown_valve_conductance"
    ] == 0.02
    complete = validate_case_document(load_experiment(CASE))
    v4 = _with_v4_valve_names(deepcopy(complete))
    v4["schema_version"] = PRE_VALVE_RENAME_SCHEMA_VERSION
    assert validate_case_document(v4) == complete


def test_valve_names_must_match_the_schema_version():
    v4 = deepcopy(default_profile())
    v4["schema_version"] = PRE_VALVE_RENAME_SCHEMA_VERSION
    with pytest.raises(ConfigurationError, match="exists only since case schema 5"):
        upgrade_experiment_document(v4)
    with pytest.raises(ConfigurationError, match="blowdown_valve_conductance_m_s"):
        load_experiment(CASE, settings={"steps.blow.flow.blowdown_valve_conductance_m_s": 0.02})


def test_a_schema_v5_case_gets_the_overall_wall_override_name_with_the_same_value():
    document = _with_v5_wall_names(deepcopy(default_profile()))
    document["schema_version"] = PRE_WALL_RENAME_SCHEMA_VERSION
    document["simulation"]["wall"]["wall_htc_override_W_m2K"] = 12.5
    upgraded = upgrade_experiment_document(document)
    assert upgraded["schema_version"] == CURRENT_SCHEMA_VERSION
    assert upgraded["simulation"]["wall"]["overall_htc_override_W_m2K"] == 12.5
    assert "wall_htc_override_W_m2K" not in upgraded["simulation"]["wall"]
    complete = validate_case_document(load_experiment(CASE))
    v5 = _with_v5_wall_names(deepcopy(complete))
    v5["schema_version"] = PRE_WALL_RENAME_SCHEMA_VERSION
    assert validate_case_document(v5) == complete


def test_wall_override_name_must_match_the_schema_version():
    v5 = deepcopy(default_profile())
    v5["schema_version"] = PRE_WALL_RENAME_SCHEMA_VERSION
    with pytest.raises(ConfigurationError, match="exists only since case schema 6"):
        upgrade_experiment_document(v5)
    with pytest.raises(ConfigurationError, match="wall_htc_override_W_m2K"):
        load_experiment(CASE, settings={"wall.wall_htc_override_W_m2K": 12.5})


def test_a_complete_schema_v3_case_still_loads_with_the_same_lines():
    complete = validate_case_document(load_experiment(CASE))
    assert complete["schema_version"] == CURRENT_SCHEMA_VERSION
    v3 = _with_v4_valve_names(deepcopy(complete))
    for step in v3["simulation"]["cycle"]["steps"].values():
        del step["outlet"]
    del v3["simulation"]["geometry"]["specified_by"]
    v3["schema_version"] = PRE_OUTLET_SCHEMA_VERSION
    assert validate_case_document(v3) == complete


# --- step post-processing -----------------------------------------------------------


def _solved(step, folder, initial_state=None):
    config = load_experiment(CASE)
    cycle = CycleConfig.from_experiment(config)
    model = ColumnModelFactory(cycle).build_column_model(
        step=step,
        initial_state=initial_state,
        solver_options=cycle.solver_for_step(
            use_sparsity=False, compute_numerical_sparsity=False
        ),
    )
    solver = Solver(model, base_folder=str(folder))
    solver.solve_cvode()
    solver.save_results()
    return model, StepPostProcessor(model, solver)


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


@pytest.fixture(scope="module")
def s_des(tmp_path_factory):
    """A steam desorption that starts from the end of a (short) startup."""
    folder = tmp_path_factory.mktemp("outlet_s_des")
    startup, _ = _solved("startup", folder)
    _, processor = _solved("s_des", folder, initial_state=_warm_state(startup))
    return processor


def _metrics_with(processor, monkeypatch, outlet):
    monkeypatch.setattr(processor, "_outlet_settings", lambda: outlet)
    return processor.compute_step_metrics()


def test_the_default_line_records_condenser_and_vacuum_separator(s_des):
    metrics = s_des.compute_step_metrics()
    assert set(_ids(metrics, "mass.after.condenser")) == {"CO2", "H2O", "O2", "N2"}
    assert set(_ids(metrics, "mass.after.vacuum")) == {"CO2", "H2O", "O2", "N2"}
    assert "system" in _ids(metrics, "energy.condenser")
    assert not _ids(metrics, "mass.vented")


def test_a_vented_steam_step_records_its_outlet_and_still_counts_its_steam(
    s_des, monkeypatch
):
    metrics = _metrics_with(s_des, monkeypatch, _outlet(False, False))
    assert not _ids(metrics, "mass.after.vacuum")
    assert not _ids(metrics, "mass.after.condenser")
    assert not _ids(metrics, "energy.condenser")
    vented = _ids(metrics, "mass.vented")
    change = {}
    for r in metrics.records:
        # The first `mass.change` of a species is the one the vent is computed from.
        if r.definition.metric_id == "mass.change":
            change.setdefault(r.species, r.value)
    steam = _ids(metrics, "steam.mass")["H2O"]
    assert steam > 0.0
    assert vented["CO2"] == -change["CO2"]
    # The injected steam leaves through the vent too.
    assert vented["H2O"] == pytest.approx(-change["H2O"] + steam, rel=1e-12)
    # No condenser: the pump moves the whole outlet stream, the solver's own figure.
    assert _ids(metrics, "energy.vacuum")["system"] == s_des.get_energies()[1]


def test_a_collected_step_without_a_condenser_flashes_the_bed_outlet(
    s_des, monkeypatch
):
    metrics = _metrics_with(s_des, monkeypatch, _outlet(True, False))
    assert not _ids(metrics, "mass.after.condenser")
    assert not _ids(metrics, "energy.condenser")
    after_vacuum = _ids(metrics, "mass.after.vacuum")
    assert set(after_vacuum) == {"CO2", "H2O", "O2", "N2"}
    assert _ids(metrics, "energy.vacuum")["system"] == s_des.get_energies()[1]
    # The flash can only remove: the product never exceeds what left the bed.
    from scipy.integrate import simpson

    outlet_co2 = s_des.pb.MW_CO2 * simpson(s_des.n_CO2_out, x=s_des.time)
    outlet_h2o = s_des.pb.MW_H2O * simpson(s_des.n_H2O_out, x=s_des.time)
    assert 0.0 <= after_vacuum["CO2"] <= outlet_co2 * (1 + 1e-12)
    assert 0.0 <= after_vacuum["H2O"] <= outlet_h2o * (1 + 1e-12)


def test_the_condenser_temperature_reaches_the_condenser(s_des, monkeypatch):
    cold = _metrics_with(s_des, monkeypatch, _outlet(True, True, 278.15))
    warm = _metrics_with(s_des, monkeypatch, _outlet(True, True, 288.15))
    assert _ids(warm, "energy.condenser")["system"] != _ids(cold, "energy.condenser")[
        "system"
    ]
    # Warmer condenser, more water stays in the vapour that reaches the pump.
    assert (
        _ids(warm, "mass.after.condenser")["H2O"]
        > _ids(cold, "mass.after.condenser")["H2O"]
    )


def test_a_condenser_above_the_cooling_water_limit_leaves_nothing_to_a_chiller(
    s_des, monkeypatch
):
    metrics = _metrics_with(s_des, monkeypatch, _outlet(True, True, 320.0))
    total = _ids(metrics, "energy.condenser")["system"]
    above_ambient = _ids(metrics, "energy.condenser.ambient")["system"]
    assert above_ambient == pytest.approx(total, rel=1e-12)


def test_a_dry_stream_leaves_the_vacuum_flash_as_vapour(s_des, monkeypatch):
    """Above its dew point the two-phase flash would return a number anyway."""
    s_des._get_flowrates()
    n = np.full(s_des.Nt, 1e-3)
    monkeypatch.setattr(s_des, "_get_flowrates", lambda: None)
    monkeypatch.setattr(s_des, "_condenser_stream_mask", lambda: np.ones(s_des.Nt, bool))
    monkeypatch.setattr(s_des, "n_CO2_out", n * 0.99, raising=False)
    monkeypatch.setattr(s_des, "n_H2O_out", n * 0.01, raising=False)
    from pydac.core.thermodynamics_separators import Component, FlashSeparator

    separator = FlashSeparator(
        Component("CO2", 304.2, 73.74), Component("H2O", 647.096, 220.64)
    )
    n_feed, results = s_des._raw_outlet_vacuum_flash(separator, 298.15, 101325.0)
    np.testing.assert_allclose(n_feed, n)
    for res in results:
        assert res["nv"] == pytest.approx(1e-3)
        assert res["nl"] == 0.0
        assert res["y_CO2"] == pytest.approx(0.99)


# --- cycle KPIs ---------------------------------------------------------------------


KPI_STEPS = ("ads", "heat", "s_purge", "s_des")
KPI_SPECIES = ("CO2", "H2O")


def _context():
    static = {
        **context_schema.COMMON_STATIC_BY_ID,
        **context_schema.ARCHITECTURE_STATIC_BY_ID["packed_bed"],
    }
    rows = [_context_row(d, context_schema.STATIC_STEP) for d in static.values()]
    for step_key in KPI_STEPS:
        for definition in context_schema.step_definitions_by_id(KPI_SPECIES).values():
            rows.append(_context_row(definition, step_key))
    return SimulationContext(
        pd.DataFrame(rows, columns=list(context_schema.SIMULATION_CONTEXT_COLUMNS))
    )


def _row(step_key, metric_id, value, species="system", unit="kg", basis=None, steps=KPI_STEPS):
    return {
        "Cycle": "cycle_1",
        "Step Index": steps.index(step_key),
        "Step Key": step_key,
        "Step": STEP_KEY_TO_CYCLE_STEP[step_key],
        "Species": species,
        "Metric ID": metric_id,
        "Value": value,
        "Unit": unit,
        "Basis": basis,
    }


def _kpis(*, s_purge_collected):
    rows = []
    for step_key in KPI_STEPS:
        for species in KPI_SPECIES:
            rows.append(_row(step_key, "mass.final", 1.0, species, basis="state"))
    rows += [
        _row("heat", "mass.vented", 0.01, "CO2"),
        _row("heat", "mass.vented", 0.0, "H2O"),
        _row("s_purge", "steam.mass", 0.2, "H2O"),
        _row("s_des", "steam.mass", 0.3, "H2O"),
        _row("s_des", "mass.after.vacuum", 0.5, "CO2"),
        _row("s_des", "mass.after.vacuum", 0.05, "H2O"),
    ]
    if s_purge_collected:
        rows += [
            _row("s_purge", "mass.after.vacuum", 0.1, "CO2"),
            _row("s_purge", "mass.after.vacuum", 0.01, "H2O"),
        ]
    else:
        rows += [
            _row("s_purge", "mass.vented", 0.1, "CO2"),
            _row("s_purge", "mass.vented", 0.4, "H2O"),
        ]
    metrics = CycleMetrics(pd.DataFrame(rows, columns=list(CYCLE_METRIC_COLUMNS)))
    return CycleKPIs(CycleData(metrics, _context()))


def test_product_steps_are_the_steps_that_recorded_a_product():
    assert _kpis(s_purge_collected=True).product_step_keys() == ("s_purge", "s_des")
    vented = _kpis(s_purge_collected=False)
    assert vented.product_step_keys() == ("s_des",)
    assert vented.recovered_mass("CO2") == pytest.approx(0.5)
    assert vented.vented_mass_by_step("CO2") == {"heat": 0.01, "s_purge": 0.1}


def test_steam_counts_in_every_step_even_when_vented():
    vented = _kpis(s_purge_collected=False)
    assert vented.steam_mass_used() == pytest.approx(0.5)
    assert vented.steam_to_co2_ratio() == pytest.approx(0.5 / 0.5)


def test_the_regeneration_ends_at_the_last_desorption_step_whatever_its_outlet():
    for collected in (True, False):
        assert _kpis(s_purge_collected=collected).last_desorption_step_key() == "s_des"


def test_a_result_without_product_records_falls_back_to_the_desorption_steps():
    """Results written before the switches existed recorded no vent on a desorption step."""
    rows = [_row("s_des", "mass.final", 1.0, "CO2", basis="state", steps=("s_des",))]
    metrics = CycleMetrics(pd.DataFrame(rows, columns=list(CYCLE_METRIC_COLUMNS)))
    legacy = CycleKPIs(CycleData(metrics, _context()))
    assert legacy.product_step_keys() == ("s_des",)


def test_a_cycle_that_vents_every_desorption_step_has_no_product():
    rows = [
        _row("s_des", "mass.final", 1.0, "CO2", basis="state", steps=("s_des",)),
        _row("s_des", "mass.vented", 0.4, "CO2", steps=("s_des",)),
    ]
    metrics = CycleMetrics(pd.DataFrame(rows, columns=list(CYCLE_METRIC_COLUMNS)))
    with pytest.raises(ValueError, match="collects product in no step"):
        CycleKPIs(CycleData(metrics, _context())).product_step_keys()


def test_every_binary_flash_passes_a_dry_sample_through_and_stays_inside_its_feed():
    """Q17a-1: the condenser and both vacuum flashes share one dew-point gate."""
    from pydac.core.thermodynamics_separators import Component, FlashSeparator

    separator = FlashSeparator(
        Component("CO2", 304.2, 73.74), Component("H2O", 647.096, 220.64)
    )
    # 3 % water at 20 kPa is 600 Pa of vapour, below p_sat(278.15 K) = 872 Pa: dry.
    dry = StepPostProcessor._binary_flash(separator, 2.0, 0.97, 278.15, 2e4)
    assert dry["nv"] == 2.0 and dry["nl"] == 0.0 and dry["y_CO2"] == 0.97
    # Steam-rich feed: a real split, inside the feed.
    wet = StepPostProcessor._binary_flash(separator, 2.0, 0.2, 278.15, 2e4)
    assert 0.0 < wet["nv"] < 2.0
    assert wet["nl"] == pytest.approx(2.0 - wet["nv"])


def test_a_condenser_step_records_the_temperature_it_delivers(s_des, monkeypatch):
    default = s_des.compute_step_metrics()
    assert _ids(default, "temperature.condenser") == {"system": 278.15}
    warm = _metrics_with(s_des, monkeypatch, _outlet(True, True, 288.15))
    assert _ids(warm, "temperature.condenser") == {"system": 288.15}
    vented = _metrics_with(s_des, monkeypatch, _outlet(False, False))
    assert not _ids(vented, "temperature.condenser")


def _priced(T_condenser):
    steps = ("s_des",)
    rows = [
        _row("s_des", "mass.final", 1.0, "CO2", basis="state", steps=steps),
        _row("s_des", "mass.after.vacuum", 0.5, "CO2", steps=steps),
        _row("s_des", "energy.condenser", -20.0, unit="MJ", steps=steps),
        _row("s_des", "energy.condenser.ambient", -15.0, unit="MJ", steps=steps),
    ]
    if T_condenser is not None:
        rows.append(_row("s_des", "temperature.condenser", T_condenser, unit="K", steps=steps))
    metrics = CycleMetrics(pd.DataFrame(rows, columns=list(CYCLE_METRIC_COLUMNS)))
    return CycleKPIs(CycleData(metrics, _context()))


def test_each_chiller_is_priced_at_its_step_condenser_temperature():
    """Q17a-2: the chiller delivers the step's condenser temperature, not a cycle constant."""
    from dataclasses import replace

    from pydac.post_processing.kpi.cycle_kpis import EquivalentWorkConditions

    def charged(kpis):
        table = kpis.energy_breakdown(equivalent=True, absolute=True)
        return float(table["Condenser"].iloc[0])

    base = EquivalentWorkConditions()
    warm = replace(base, T_condensation=288.15)
    # 5 MJ go to the chiller (20 total, 15 to cooling water).
    assert charged(_priced(288.15)) == pytest.approx(5.0 / CycleKPIs.chiller_cop(warm))
    # A warmer condenser needs a cheaper chiller.
    assert charged(_priced(288.15)) < charged(_priced(278.15))
    # A result without the record keeps the conditions' temperature (278.15 K).
    assert charged(_priced(None)) == pytest.approx(5.0 / CycleKPIs.chiller_cop(base))
    assert charged(_priced(None)) == charged(_priced(278.15))
    assert _priced(288.15).condenser_pricing() == {
        "s_des": (288.15, CycleKPIs.chiller_cop(warm))
    }


# --- mixtures without H2O, O2 or CO2 (17b) ------------------------------------------

DRY = {
    "mixture": {"species": ["CO2", "N2"], "closure_species": "N2", "adsorbates": ["CO2"]},
    "adsorption": {
        "equilibrium_systems": {"Toth": {"datasets": {"Toth": "Cleaned-Toth"}, "parameters": {}}},
        "kinetics": {"CO2": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.003}}},
    },
    "execution.cycle_steps": ["ads", "blow", "heat", "w_des", "cool", "press"],
    "execution.max_cycles": 2,
    "execution.require_css": False,
    "execution.output_root": None,
}


def test_a_dry_cycle_collects_without_condenser_and_reports_its_kpis(tmp_path, capsys):
    settings = {**DRY, "execution.output_root": str(tmp_path)}
    result = pydac.run_css(load_experiment(CASE, settings=settings))
    out = capsys.readouterr().out
    assert "KPI figure 'product composition' skipped: the mixture has no H2O, O2." in out

    kpis = pydac.CycleKPIs(pydac.CycleDataLoader(result.folder).load())
    assert kpis.product_step_keys() == ("w_des",)
    # Nothing condenses: the product line has no condenser stage.
    assert kpis.data.value(step="w_des", species="CO2", metric_id="mass.after.condenser",
                           default=None) is None
    assert kpis.energy_value("w_des", "energy.condenser") == 0.0
    assert kpis.recovered_mass("CO2") > 0.0
    purity = kpis.purity()
    assert set(purity["masses_kg"]) == {"CO2", "N2"}
    assert 0.0 < purity["purity_wet"] <= 1.0
    assert np.isnan(purity["oxygen_ppm"]) and not np.isnan(purity["nitrogen_ppm"])
    assert np.isnan(kpis.h2o_to_co2_ratio()) and kpis.steam_mass_used() == 0.0
    assert kpis.preheating_temperatures()["steam_condensation_K"] is None


def test_a_run_refuses_steam_steps_without_h2o_and_product_without_co2():
    dry = load_experiment(CASE, settings={**{k: v for k, v in DRY.items()
                                              if not k.startswith("execution.")},
                                          "execution.cycle_steps": ["startup", "s_des"]})
    with pytest.raises(ConfigurationError, match=r"\['s_des'\] inject steam"):
        pydac.run_steps(dry)
    no_co2 = load_experiment(CASE, settings={
        "mixture": {"species": ["H2O", "N2"], "closure_species": "N2", "adsorbates": ["H2O"]},
        "adsorption": {
            "equilibrium_systems": {"GAB": {"datasets": {}, "parameters": {}}},
            "kinetics": {"H2O": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.00083}}},
        },
        "execution.cycle_steps": ["startup", "blow", "heat", "w_des"],
    })
    with pytest.raises(ConfigurationError, match=r"\['w_des'\] collect CO2 product"):
        pydac.run_steps(no_co2)


def test_the_steam_condensation_constraint_needs_h2o(tmp_path):
    case = load_experiment(CASE, settings={**DRY, "execution.output_root": str(tmp_path)})
    with pytest.raises(ConfigurationError, match="has no H2O"):
        pydac.run_nsga2(case, {"t_ads": [900.0, 1800.0]}, steam_condensation=True)
    with pytest.raises(ConfigurationError, match="has no O2"):
        pydac.run_nsga2(case, {"t_ads": [900.0, 1800.0]}, oxygen_ppm_max=40.0)
    assert not any(tmp_path.iterdir())


def test_the_steam_condensation_constraint_needs_steam_assisted_desorption(tmp_path):
    humid = load_experiment(CASE, settings={"execution.output_root": str(tmp_path)})
    assert "H2O" in humid.simulation.mixture.species
    # A wall-heated cycle, and a vacuum cycle whose blowdown collects the product.
    for steps, extra in ((["ads", "blow", "heat", "w_des", "cool", "press"], {}),
                         (["ads", "blow", "press"], {"steps.blow.outlet.collect_product": True})):
        case = load_experiment(CASE, settings={"execution.output_root": str(tmp_path),
                                               "execution.cycle_steps": steps, **extra})
        with pytest.raises(ConfigurationError, match="has no s_des step"):
            pydac.run_nsga2(case, {"t_ads": [900.0, 1800.0]}, steam_condensation=True)
    assert not any(tmp_path.iterdir())
