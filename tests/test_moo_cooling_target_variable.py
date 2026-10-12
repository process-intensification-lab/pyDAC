"""The seventh decision variable: the cooldown target temperature.

The `cool` step ends when the bed is actually cool, not when a clock says so.
Making its *threshold* the decision variable lets the optimizer choose how
cold — bounded below the oxidation limit, so every candidate is admissible by
construction — while the step duration still adapts to the column shape.

The load-bearing property tested here is that the change is **additive**: a
campaign that does not ask for the variable produces exactly the vector, the
settings and the bounds it produced before the variable existed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymoo")  # NSGA-II requires pymoo.

from pydac.config import operating_step_settings
from pydac.config.operating import OPERATING_SETTING_TARGETS
from pydac.utils.nsga2.decisions import (
    BASE_VAR_NAMES,
    COOLING_TARGET_VAR_NAME,
    STEAM_VAR_NAME,
    decision_vector_to_model_inputs,
    resolve_var_names,
    validate_var_names,
)


SIX = (7200.0, 1800.0, 3600.0, 20000.0, 1.0e-3, 5.0e-3)
COOLING_TARGET_K = 310.0


def test_the_knob_writes_the_cool_step_threshold_and_nothing_else():
    targets = OPERATING_SETTING_TARGETS["cooling_target_temperature_K"]
    assert targets == (("cool", "termination", "threshold_value"),)

    settings = operating_step_settings(
        {"cooling_target_temperature_K": COOLING_TARGET_K}
    )
    assert settings == {
        "steps.cool.termination.threshold_value": COOLING_TARGET_K
    }


def test_the_knob_reaches_the_model_through_the_ordinary_loader():
    """A threshold is an ordinary step field, so nothing special carries it."""
    from pydac.config import load_experiment

    experiment = load_experiment(
        None,
        settings=operating_step_settings(
            {"cooling_target_temperature_K": COOLING_TARGET_K}
        ),
    )
    steps = {step.name: step.config for step in experiment.simulation.cycle.steps}
    assert steps["cool"].termination.threshold_value == COOLING_TARGET_K
    assert steps["cool"].termination.metric == "max_bed_temperature"
    assert steps["cool"].termination.direction == "below"


@pytest.mark.parametrize("is_steam_assisted", [False, True])
def test_the_existing_vectors_are_unchanged(is_steam_assisted):
    expected = BASE_VAR_NAMES + ((STEAM_VAR_NAME,) if is_steam_assisted else ())
    assert resolve_var_names(is_steam_assisted) == expected


@pytest.mark.parametrize("is_steam_assisted", [False, True])
def test_the_optional_variable_is_appended_never_interleaved(is_steam_assisted):
    without = resolve_var_names(is_steam_assisted)
    with_target = resolve_var_names(
        is_steam_assisted, optimize_cooling_target=True
    )
    assert with_target == (*without, COOLING_TARGET_VAR_NAME)


def test_a_campaign_that_does_not_ask_gets_the_settings_it_always_got():
    settings, _ = decision_vector_to_model_inputs(
        SIX, is_steam_assisted=True, tau_ar=0.5, bed_volume=1.0e-3
    )
    assert "cooling_target_temperature_K" not in settings

    declared, _ = decision_vector_to_model_inputs(
        SIX,
        is_steam_assisted=True,
        tau_ar=0.5,
        bed_volume=1.0e-3,
        var_names=resolve_var_names(True),
    )
    assert declared == settings


def test_the_seventh_variable_maps_to_the_cool_threshold():
    settings, _ = decision_vector_to_model_inputs(
        (*SIX, COOLING_TARGET_K),
        is_steam_assisted=True,
        tau_ar=0.5,
        bed_volume=1.0e-3,
        var_names=resolve_var_names(True, optimize_cooling_target=True),
    )
    assert settings["cooling_target_temperature_K"] == COOLING_TARGET_K

    six_settings, _ = decision_vector_to_model_inputs(
        SIX, is_steam_assisted=True, tau_ar=0.5, bed_volume=1.0e-3
    )
    assert {
        key: value
        for key, value in settings.items()
        if key != "cooling_target_temperature_K"
    } == six_settings


def test_a_vector_of_the_wrong_length_is_refused_against_its_own_declaration():
    with pytest.raises(ValueError, match="expected 7"):
        decision_vector_to_model_inputs(
            SIX,
            is_steam_assisted=True,
            tau_ar=0.5,
            bed_volume=1.0e-3,
            var_names=resolve_var_names(True, optimize_cooling_target=True),
        )


def test_an_unknown_or_reordered_vector_is_refused():
    with pytest.raises(ValueError, match="Unsupported decision variables"):
        validate_var_names(("t_ads", "T_cool_target"), is_steam_assisted=True)
    reordered = (COOLING_TARGET_VAR_NAME, *resolve_var_names(True))
    with pytest.raises(ValueError, match="Unsupported decision variables"):
        validate_var_names(reordered, is_steam_assisted=True)
