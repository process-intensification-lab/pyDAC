"""Case validation refuses stop rules the solver would reject or never read."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pydac.config import ConfigurationError, load_experiment
from pydac.config.steps import STEP_NAMES, STEP_ONLY_TERMINATION_FIELDS
from pydac.utils.css.runner import validate_css_case
from pydac.utils.run.validation import validate_step_case
from pydac.utils.solver.events import build_event_specs
from pydac.utils.tvsa.termination import EVENT_METRIC_REQUIREMENTS, EVENT_METRICS

CASE = "tests/cases/v3_smoke_startup.json"
STEPS = "simulation.cycle.steps"

DRY = {
    "mixture": {"species": ["CO2", "N2"], "closure_species": "N2", "adsorbates": ["CO2"]},
    "adsorption": {
        "equilibrium_systems": {"Toth": {"datasets": {"Toth": "Cleaned-Toth"}, "parameters": {}}},
        "kinetics": {"CO2": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.003}}},
    },
}


def _event_specs(metric, *, species=("CO2", "H2O", "O2", "N2"), adsorbates=("CO2", "H2O")):
    layout = SimpleNamespace(mixture=SimpleNamespace(species=species), adsorbates=adsorbates)
    return build_event_specs(
        model=SimpleNamespace(), step="ads", layout=layout, A_rings=None,
        sorbent_weights=None, p_ref=1.0, T_ref=1.0, reference_concentrations=None,
        total_reference_concentration=1.0, termination_mode="event",
        termination_metric=metric, termination_direction="above",
        termination_threshold_value=1.0,
    )


def test_the_validated_metric_names_are_the_ones_event_assembly_builds():
    for metric in EVENT_METRICS:
        assert len(_event_specs(metric)) == 1
    with pytest.raises(ValueError, match="Unsupported termination metric"):
        _event_specs("not_a_runtime_metric")
    for metric, requirement in EVENT_METRIC_REQUIREMENTS.items():
        if requirement is None:
            continue
        kind, needed = requirement
        missing = {
            "species": {"species": ("CO2", "H2O", "N2")},
            "adsorbate": {"adsorbates": ("H2O",)},
        }[kind]
        assert needed not in next(iter(missing.values()))
        with pytest.raises(ValueError, match="Unsupported termination metric"):
            _event_specs(metric, **missing)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("ads.termination.metric", "not_a_runtime_metric", "metric` must be one of"),
        ("blow.termination.threshold_source", "T_wall", "threshold_source` must be one of"),
        ("ads.termination.threshold_source", "p_out", "exactly one of `threshold_value`"),
        ("blow.termination.threshold_value", 1000.0, "exactly one of `threshold_value`"),
    ],
)
def test_validate_refuses_unknown_names_and_a_double_threshold(path, value, message):
    with pytest.raises(ConfigurationError, match=message):
        load_experiment(CASE, settings={f"{STEPS}.{path}": value})


@pytest.mark.parametrize(
    ("field", "step"),
    [
        (field, step)
        for field, steps in STEP_ONLY_TERMINATION_FIELDS.items()
        for step in STEP_NAMES
        if step not in steps
    ],
)
def test_validate_refuses_a_termination_field_the_step_never_reads(field, step):
    value = True if field.startswith("stop_on_") else 1.0
    with pytest.raises(ConfigurationError, match=f"`termination.{field}` acts only on"):
        load_experiment(CASE, settings={f"{STEPS}.{step}.termination.{field}": value})


def test_the_shipped_stop_rules_still_validate():
    case = load_experiment(CASE)
    termination = {step.name: step.config.termination for step in case.simulation.cycle.steps}
    assert termination["cool"].stop_on_low_pressure
    assert termination["blow"].stop_on_pressure_plateau
    assert termination["press"].stop_on_pressure_plateau


def test_preflight_refuses_an_event_metric_the_mixture_cannot_evaluate():
    # w_purge stops on the O2 mole fraction in event mode by default; the smoke
    # case forces time mode globally, so the override is lifted here.
    dry_steps = {
        **DRY,
        "execution.cycle_steps": ["startup", "blow", "heat", "w_purge"],
        "execution.termination_mode": None,
    }
    case = load_experiment(CASE, settings=dry_steps)
    with pytest.raises(ValueError, match=r"`w_purge` runs in event mode on "
                                         r"`max_bed_oxygen_mole_fraction`, which needs O2"):
        validate_step_case(case)

    # In time mode the metric is never evaluated, whichever override sets it.
    for settings in (
        {"execution.termination_mode": "time"},
        {"execution.step_options": {"w_purge": {"termination_mode": "time"}}},
        {f"{STEPS}.w_purge.termination.mode": "time"},
    ):
        validate_step_case(load_experiment(CASE, settings={**dry_steps, **settings}))


def test_preflight_sees_an_event_override_set_for_one_cycle_only():
    settings = {
        **DRY,
        "execution.cycle_steps": ["ads", "blow", "heat", "w_purge", "w_des", "cool", "press"],
        "execution.termination_mode": "time",
    }
    validate_css_case(load_experiment(CASE, settings=settings))
    case = load_experiment(CASE, settings={
        **settings, "execution.cycle_options": {"2": {"termination_mode": "event"}},
    })
    with pytest.raises(ValueError, match="`w_purge` runs in event mode"):
        validate_css_case(case)
