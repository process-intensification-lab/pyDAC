"""Startup initial solid loadings (case schema 8).

`steps.startup.composition.initial_solid_loadings_mol_per_kg` sets a uniform
initial solid instead of equilibrating it with the initial gas. It is all or
nothing (every adsorbate, no other species), finite and non-negative, startup
only, and public through the case, the settings and the CLI. A schema-7 case is
upgraded to null, the equilibrium start it computed with.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from pydac.config import ConfigurationError, load_experiment
from pydac.config.cycle import CycleConfig
from pydac.config.steps import STEP_NAMES
from pydac.config.migrations import (
    CURRENT_SCHEMA_VERSION,
    PRE_INITIAL_LOADING_SCHEMA_VERSION,
    upgrade_experiment_document,
)
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.inputs.case_io import case_fingerprint, load_case, validate_case_document
from pydac.inputs.defaults import default_profile
from pydac.inputs.defaults.steps import INITIAL_SOLID_LOADINGS_FIELD


CASE = "tests/cases/v3_smoke_startup.json"
SETTING = f"steps.startup.composition.{INITIAL_SOLID_LOADINGS_FIELD}"
LOADINGS = {"CO2": 0.5, "H2O": 1.25}


def without_initial_solid_loadings(document):
    """Drop the field from every step: the shape of schema 7 and every earlier one."""
    for step in document["simulation"]["cycle"]["steps"].values():
        del step["composition"][INITIAL_SOLID_LOADINGS_FIELD]
    return document


def _without_the_field(document):
    """Return the schema-7 shape of a schema-8 document."""
    document["schema_version"] = PRE_INITIAL_LOADING_SCHEMA_VERSION
    return without_initial_solid_loadings(document)


def _startup_solid(settings):
    experiment = load_experiment(CASE, settings=settings)
    model = ColumnModelFactory(CycleConfig.from_experiment(experiment)).build_column_model(
        step="startup"
    )
    model.set_ic()
    return model


def test_a_declared_loading_is_the_uniform_initial_solid():
    model = _startup_solid({SETTING: LOADINGS})
    for species, index in model.adsorbate_index.items():
        assert (model.qi[..., index] == LOADINGS[species]).all()


def test_null_keeps_the_equilibrium_start():
    model = _startup_solid({})
    assert model.initial_solid_loadings_mol_per_kg is None
    equilibrium = model._equilibrium_loadings_at_state(
        model.y_initial_by_species, model.T_initial, model.p_initial
    )
    for species, index in model.adsorbate_index.items():
        assert model.qi[..., index] == pytest.approx(equilibrium[species] + 1e-32)


@pytest.mark.parametrize(
    "settings",
    [
        {SETTING: LOADINGS},
        {"steps": {"startup": {"composition": {INITIAL_SOLID_LOADINGS_FIELD: LOADINGS}}}},
        [f'{SETTING}={{"CO2": 0.5, "H2O": 1.25}}'],
    ],
    ids=["dotted", "nested", "cli"],
)
def test_every_settings_route_gives_the_same_case(settings):
    config = load_experiment(CASE, settings=settings)
    startup = config.simulation.cycle.steps[0]
    assert startup.name == "startup"
    assert dict(getattr(startup.config.composition, INITIAL_SOLID_LOADINGS_FIELD)) == LOADINGS
    reference = load_experiment(CASE, settings={SETTING: LOADINGS})
    assert case_fingerprint(config) == case_fingerprint(reference)


def test_a_setting_replaces_the_mapping_instead_of_merging_it():
    base = validate_case_document(load_experiment(CASE, settings={SETTING: LOADINGS}))
    config = load_experiment(base, settings={SETTING: {"CO2": 0.0, "H2O": 0.0}})
    loadings = getattr(config.simulation.cycle.steps[0].config.composition,
                       INITIAL_SOLID_LOADINGS_FIELD)
    assert dict(loadings) == {"CO2": 0.0, "H2O": 0.0}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"CO2": 0.5}, "cover the declared adsorbates"),
        ({"CO2": 0.5, "H2O": 1.0, "N2": 0.1}, "cover the declared adsorbates"),
        ({"CO2": -0.1, "H2O": 1.0}, "finite and non-negative"),
        ({"CO2": True, "H2O": 1.0}, "must be a number"),
        ({"CO2": "0.5", "H2O": 1.0}, "must be a number"),
        ({"CO2": 10**400, "H2O": 1.0}, "outside the floating-point range"),
        (0.5, "must be a mapping|JSON object"),
    ],
)
def test_an_invalid_loading_is_refused_at_load(value, message):
    with pytest.raises(ConfigurationError, match=message):
        load_experiment(CASE, settings={SETTING: value})
    complete = validate_case_document(load_experiment(CASE))
    complete["simulation"]["cycle"]["steps"]["startup"]["composition"][
        INITIAL_SOLID_LOADINGS_FIELD
    ] = value
    with pytest.raises(ConfigurationError, match=message):
        load_case(complete)


def test_another_step_refuses_the_field():
    with pytest.raises(ConfigurationError, match="acts only on startup"):
        load_experiment(
            CASE, settings={f"steps.ads.composition.{INITIAL_SOLID_LOADINGS_FIELD}": LOADINGS}
        )


def test_a_schema_v7_case_gets_null_the_equilibrium_start_it_computed_with():
    v7 = _without_the_field(deepcopy(default_profile()))
    assert upgrade_experiment_document(v7) == default_profile()

    complete = validate_case_document(load_experiment(CASE))
    assert complete["schema_version"] == CURRENT_SCHEMA_VERSION
    assert validate_case_document(_without_the_field(deepcopy(complete))) == complete


def test_the_field_needs_schema_8():
    v7 = _without_the_field(deepcopy(default_profile()))
    v7["simulation"]["cycle"]["steps"]["startup"]["composition"][
        INITIAL_SOLID_LOADINGS_FIELD
    ] = LOADINGS
    with pytest.raises(ConfigurationError, match="exists only since case schema 8"):
        upgrade_experiment_document(v7)


def test_the_cli_validates_the_field(capsys):
    from pydac.cli import main

    good = f'{SETTING}={{"CO2": 0.5, "H2O": 1.25}}'
    bad = f'{SETTING}={{"CO2": 0.5}}'
    assert main(["validate", "--set", good, "--validate"]) == 0
    assert main(["validate", "--set", bad, "--validate"]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_the_json_schema_accepts_the_field():
    jsonschema = pytest.importorskip("jsonschema")
    from pydac.config.schema_generation import generate_json_schema

    config = load_experiment(None, settings={SETTING: LOADINGS})
    jsonschema.Draft202012Validator(generate_json_schema()).validate(
        validate_case_document(config)
    )


# --- Sol schema-8 audit (S8-001..003) ---------------------------------------


@pytest.mark.parametrize("version", range(1, CURRENT_SCHEMA_VERSION))
def test_an_older_frozen_case_with_the_field_names_schema_8(version):
    complete = validate_case_document(load_experiment(CASE))
    complete["schema_version"] = version
    for route in (validate_case_document, load_case):
        with pytest.raises(ConfigurationError, match="exists only since case schema 8"):
            route(deepcopy(complete))


def test_species_names_do_not_depend_on_case():
    lower = _startup_solid({SETTING: {"co2": 0.5, "h2o": 1.25}})
    for species, index in lower.adsorbate_index.items():
        assert (lower.qi[..., index] == LOADINGS[species]).all()


@pytest.mark.parametrize("value", [{"CO2": 0.5, "co2": 0.7, "H2O": 1.25},
                                   {"CO2": 0.5, " co2 ": 0.5, "H2O": 1.25}])
def test_two_keys_for_one_species_are_refused(value, capsys):
    from pydac.cli import main

    with pytest.raises(ConfigurationError, match="names CO2 twice"):
        load_experiment(CASE, settings={SETTING: value})
    complete = validate_case_document(load_experiment(CASE))
    complete["simulation"]["cycle"]["steps"]["startup"]["composition"][
        INITIAL_SOLID_LOADINGS_FIELD] = value
    with pytest.raises(ConfigurationError, match="names CO2 twice"):
        load_case(complete)
    import json
    assert main(["validate", "--set", f"{SETTING}={json.dumps(value)}", "--validate"]) == 2
    assert "names CO2 twice" in capsys.readouterr().err


def test_two_keys_for_one_fixed_adsorbate_are_refused():
    from pydac.utils.column.validation import resolve_fixed_adsorbates

    entry = {"loading_mol_per_kg": 0.1, "heat_capacity_J_molK": 29.0}
    assert set(resolve_fixed_adsorbates({"n2": entry})) == {"N2"}
    with pytest.raises(ValueError, match="names N2 twice"):
        resolve_fixed_adsorbates({"N2": entry, "n2": entry})


def test_a_null_setting_keeps_the_mapping_of_the_preset():
    preset = validate_case_document(load_experiment(CASE, settings={SETTING: LOADINGS}))
    config = load_experiment(preset, settings={SETTING: None})
    loadings = getattr(config.simulation.cycle.steps[0].config.composition,
                       INITIAL_SOLID_LOADINGS_FIELD)
    assert dict(loadings) == LOADINGS


def test_an_exported_case_loads_with_the_same_mapping_and_fingerprint(tmp_path):
    from pydac import export_case

    config = load_experiment(CASE, settings={SETTING: LOADINGS})
    path = export_case(config, tmp_path / "case.json")
    again = load_case(path)
    assert dict(getattr(again.simulation.cycle.steps[0].config.composition,
                        INITIAL_SOLID_LOADINGS_FIELD)) == LOADINGS
    assert case_fingerprint(again) == case_fingerprint(config)


def test_a_declared_solid_leaves_the_startup_gas_unchanged():
    import numpy as np

    mapped, null = _startup_solid({SETTING: LOADINGS}), _startup_solid({})
    for name in ("ci", "c_tot", "T", "Tw"):
        assert np.array_equal(getattr(mapped, name), getattr(null, name)), name
    assert mapped.p_initial == null.p_initial


@pytest.mark.parametrize("step", [name for name in STEP_NAMES if name != "startup"])
def test_every_other_step_refuses_the_field(step):
    with pytest.raises(ConfigurationError, match="acts only on startup"):
        load_experiment(
            CASE, settings={f"steps.{step}.composition.{INITIAL_SOLID_LOADINGS_FIELD}": LOADINGS}
        )


def test_an_nsga2_candidate_keeps_the_declared_startup_solid():
    from pydac.workflows.orchestrator_NSGA2 import candidate_settings

    base = load_experiment(CASE, settings={SETTING: LOADINGS})
    settings = candidate_settings({})
    assert not any(INITIAL_SOLID_LOADINGS_FIELD in key for key in settings)
    candidate = load_experiment(base, settings=settings)
    assert dict(getattr(candidate.simulation.cycle.steps[0].config.composition,
                        INITIAL_SOLID_LOADINGS_FIELD)) == LOADINGS
