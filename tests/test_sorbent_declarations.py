"""Public sorbent declarations (case schema 7): per-adsorbate heats, solid cp, fixed loadings.

The heats of adsorption are one species-keyed sorbent property that must cover
`mixture.adsorbates` (other species are ignored); the solid cp names an implemented model with its
exact parameters; fixed loadings only add heat capacity. All three are public
(JSON, settings, API), are checked by `--validate`/`--dry-run` before a run, and a
schema-6 case is upgraded to the values it computed with.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from pydac.config import ConfigurationError, load_experiment
from pydac.config.migrations import (
    CURRENT_SCHEMA_VERSION,
    PRE_SORBENT_MAPPING_SCHEMA_VERSION,
    V6_HEAT_FIELDS,
    V6_HEATS_OF_ADSORPTION_J_MOL,
    V6_SORBENT_DECLARATIONS,
    V7_SORBENT_FIELDS,
    upgrade_experiment_document,
)
from pydac.inputs.case_io import validate_case_document
from pydac.inputs.defaults import default_profile
from pydac.utils.column.validation import (
    SOLID_HEAT_CAPACITY_MODELS,
    resolve_heats_of_adsorption,
    resolve_solid_heat_capacity,
)
from tests.test_initial_solid_loadings import without_initial_solid_loadings


CASE = "tests/cases/v3_smoke_startup.json"
DRY_CO2_N2 = {
    "mixture": {"species": ["CO2", "N2"], "closure_species": "N2", "adsorbates": ["CO2", "N2"]},
    "kinetics": {
        species: {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.003}}
        for species in ("CO2", "N2")
    },
    "equilibrium": {
        "CompetitiveSips": {
            "datasets": {"CompetitiveSips": "synthetic-test"},
            "parameters": {
                "component_parameters": {
                    species: {
                        "qinf_ref_mol_per_kg": 2.0,
                        "qinf_temperature_coefficient": 0.0,
                        "affinity_prefactor_inverse_pressure_unit": 0.1,
                        "affinity_heat_J_per_mol": 1000.0,
                        "heterogeneity_ref": 1.0,
                        "heterogeneity_temperature_coefficient": 0.0,
                    }
                    for species in ("CO2", "N2")
                },
                "reference_temperature_K": 300.0,
                "pressure_unit_Pa": 100000.0,
            },
        }
    },
}


def _with_v6_sorbent_fields(document):
    """Give the sorbent the per-species heat fields schema v6 (and every earlier one) used.

    Schema v6 also had no startup initial solid loadings (added in v8).
    """
    without_initial_solid_loadings(document)
    sorbent = document["simulation"]["sorbent"]
    heats = sorbent.pop("heats_of_adsorption_J_mol")
    for old, species in V6_HEAT_FIELDS.items():
        sorbent[old] = heats[species]
    for name in V6_SORBENT_DECLARATIONS:
        del sorbent[name]
    return document


def _dry_settings(heats):
    """Settings for a dry CO2/N2 case with competitive Sips, reached through the public path."""
    return {
        "mixture": DRY_CO2_N2["mixture"],
        "adsorption.equilibrium_systems": DRY_CO2_N2["equilibrium"],
        "adsorption.kinetics": DRY_CO2_N2["kinetics"],
        "sorbent.heats_of_adsorption_J_mol": heats,
    }


def test_a_schema_v6_case_gets_the_heats_mapping_and_the_cp_it_computed_with():
    document = _with_v6_sorbent_fields(deepcopy(default_profile()))
    document["schema_version"] = PRE_SORBENT_MAPPING_SCHEMA_VERSION
    assert upgrade_experiment_document(document) == default_profile()

    document["simulation"]["sorbent"]["heat_of_adsorption_co2_J_mol"] = -70000.0
    upgraded = upgrade_experiment_document(document)
    assert upgraded["schema_version"] == CURRENT_SCHEMA_VERSION
    assert upgraded["simulation"]["sorbent"]["heats_of_adsorption_J_mol"] == {
        "CO2": -70000.0,
        "H2O": V6_HEATS_OF_ADSORPTION_J_MOL["H2O"],
    }

    complete = validate_case_document(load_experiment(CASE))
    v6 = _with_v6_sorbent_fields(deepcopy(complete))
    v6["schema_version"] = PRE_SORBENT_MAPPING_SCHEMA_VERSION
    assert validate_case_document(v6) == complete


def test_a_schema_v6_overlay_stating_one_heat_keeps_the_other_reference_value():
    overlay = {
        "schema_version": PRE_SORBENT_MAPPING_SCHEMA_VERSION,
        "simulation": {"sorbent": {"heat_of_adsorption_h2o_J_mol": -50000.0}},
    }
    config = load_experiment(overlay)
    assert dict(config.simulation.sorbent.heats_of_adsorption_J_mol) == {
        "CO2": V6_HEATS_OF_ADSORPTION_J_MOL["CO2"],
        "H2O": -50000.0,
    }


@pytest.mark.parametrize("name", V7_SORBENT_FIELDS)
def test_sorbent_field_names_must_match_the_schema_version(name):
    v6 = _with_v6_sorbent_fields(deepcopy(default_profile()))
    v6["schema_version"] = PRE_SORBENT_MAPPING_SCHEMA_VERSION
    v6["simulation"]["sorbent"][name] = {}
    with pytest.raises(ConfigurationError, match="exists only since case schema 7"):
        upgrade_experiment_document(v6)


def test_the_old_per_species_heat_fields_are_refused_in_a_current_document():
    with pytest.raises(ConfigurationError, match="heat_of_adsorption_co2_J_mol"):
        load_experiment(CASE, settings={"sorbent.heat_of_adsorption_co2_J_mol": -70000.0})


def test_an_n2_heat_of_adsorption_is_reachable_through_the_public_settings():
    config = load_experiment(CASE, settings=_dry_settings({"CO2": -20000.0, "N2": -10000.0}))
    assert dict(config.simulation.sorbent.heats_of_adsorption_J_mol) == {
        "CO2": -20000.0,
        "N2": -10000.0,
    }
    complete = validate_case_document(config)
    assert complete["simulation"]["sorbent"]["heats_of_adsorption_J_mol"] == {
        "CO2": -20000.0,
        "N2": -10000.0,
    }


@pytest.mark.parametrize(
    ("heats", "message"),
    [
        ({"CO2": -20000.0}, r"missing declared adsorbates \['N2'\]"),
        ({"CO2": -20000.0, "N2": 10000.0}, "finite and negative"),
    ],
)
def test_heats_that_do_not_match_the_adsorbates_are_refused_before_a_run(heats, message):
    # Before schema 7 a missing N2 heat passed --dry-run and failed at the model build.
    with pytest.raises(ConfigurationError, match=message):
        load_experiment(CASE, settings=_dry_settings(heats))


def test_the_solid_cp_declaration_is_public_and_checked_before_a_run():
    constant = {"model": "constant", "parameters": {"value_J_kgK": 1550.0}}
    config = load_experiment(CASE, settings={"sorbent.heat_capacity": constant})
    assert dict(config.simulation.sorbent.heat_capacity) == constant
    for declaration, message in (
        ({"model": "Shomate", "parameters": {}}, "available models are Low2023 and constant"),
        ({"model": "constant", "parameters": {}}, "requires only `value_J_kgK`"),
        ({"model": "constant", "parameters": {"value_J_kgK": -1.0}}, "finite and positive"),
    ):
        with pytest.raises(ConfigurationError, match=message):
            load_experiment(CASE, settings={"sorbent.heat_capacity": declaration})


def test_fixed_adsorbates_are_public_and_checked_before_a_run():
    fixed = {"H2O": {"loading_mol_per_kg": 0.15, "heat_capacity_J_molK": 75.3}}
    config = load_experiment(CASE, settings={"sorbent.fixed_adsorbates": fixed})
    assert dict(config.simulation.sorbent.fixed_adsorbates) == fixed
    with pytest.raises(ConfigurationError, match="Invalid fixed-adsorbate properties"):
        load_experiment(
            CASE, settings={"sorbent.fixed_adsorbates": {"H2O": {"loading_mol_per_kg": 0.15}}}
        )


def test_the_solid_cp_table_lists_what_the_resolver_evaluates():
    assert set(SOLID_HEAT_CAPACITY_MODELS) == {"low2023", "constant"}
    low2023 = default_profile()["simulation"]["sorbent"]["heat_capacity"]
    a, b, c = (low2023["parameters"][name] for name in ("a_J_K_per_kg", "b_J_per_kgK2", "c_J_per_kgK"))
    T = 298.15
    assert resolve_solid_heat_capacity(low2023, T) == a * T**(-2) + b * T - c
    assert resolve_solid_heat_capacity(
        {"model": "Constant", "parameters": {"value_J_kgK": 1550.0}}, T
    ) == 1550.0


def test_heats_follow_the_adsorbate_order_and_ignore_species_not_adsorbed():
    assert list(resolve_heats_of_adsorption({"N2": -1.0, "CO2": -2.0}, ("CO2", "N2"))) == [
        "CO2",
        "N2",
    ]
    # A dry case keeps the sorbent's H2O heat from the profile: it is not used.
    assert resolve_heats_of_adsorption({"CO2": -2.0, "H2O": -1.0}, ("CO2",)) == {"CO2": -2.0}


def test_the_three_opaque_registries_list_the_same_mappings():
    # An opaque mapping is declared in three places (manual 01 §10.1); they must agree.
    from pydac.config.settings import OPAQUE_MAPPING_PATHS
    from pydac.inputs.case_io import _OPAQUE_PATHS
    from pydac.inputs.defaults import default_rules

    def setting(path):
        # A document path as a canonical setting: no `simulation.`, and `steps.X` for a step.
        parts = path[1:] if path[0] == "simulation" else path
        return ".".join(parts[1:] if parts[0] == "cycle" else parts)

    rules = {path for path, rule in default_rules().items() if rule.opaque}
    assert rules == set(_OPAQUE_PATHS)
    assert {setting(path) for path in rules} == set(OPAQUE_MAPPING_PATHS)


def test_a_current_heat_mapping_replaces_the_default_one_wholesale():
    # Unlike the v6 one-scalar overlay, a current mapping does not inherit the H2O heat.
    for source, settings in (
        (CASE, {"sorbent.heats_of_adsorption_J_mol": {"CO2": -70000.0}}),
        ({"simulation": {"sorbent": {"heats_of_adsorption_J_mol": {"CO2": -70000.0}}}}, None),
    ):
        with pytest.raises(ConfigurationError, match=r"missing declared adsorbates \['H2O'\]"):
            load_experiment(source, settings=settings)


def test_a_solid_cp_negative_only_at_a_later_step_is_refused_naming_that_step(capsys):
    from pydac.cli import main

    declaration = {
        "model": "Low2023",
        "parameters": {"a_J_K_per_kg": 0.0, "b_J_per_kgK2": -1.0, "c_J_per_kgK": -350.0},
    }
    assert resolve_solid_heat_capacity(declaration, 298.15) > 0
    setting = "sorbent.heat_capacity=" + json.dumps(declaration)
    assert main(["validate", "--set", setting, "--validate"]) == 2
    assert main(["steps", "--set", setting, "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "step 'heat'" in err and "373.15 K" in err and "Traceback" not in err


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ({"sorbent.heat_capacity": {"model": "constant", "parameters": {"value_J_kgK": True}}},
         "must be a number"),
        ({"sorbent.heat_capacity": {"model": "constant", "parameters": {"value_J_kgK": "1550"}}},
         "must be a number"),
        ({"sorbent.heat_capacity": {"model": "constant", "parameters": {"value_J_kgK": 10**400}}},
         "outside the floating-point range"),
        ({"sorbent.heats_of_adsorption_J_mol": {"CO2": -(10**400), "H2O": -46000.0}},
         "outside the floating-point range"),
        ({"sorbent.fixed_adsorbates": {"H2O": {"loading_mol_per_kg": "0.15",
                                                "heat_capacity_J_molK": 75.3}}},
         "must be a number"),
    ],
)
def test_sorbent_numbers_must_be_json_numbers_in_range(setting, message):
    with pytest.raises(ConfigurationError, match=message):
        load_experiment(CASE, settings=setting)
    complete = validate_case_document(load_experiment(CASE))
    path, value = next(iter(setting.items()))
    complete["simulation"]["sorbent"][path.split(".")[1]] = value
    with pytest.raises(ConfigurationError, match=message):
        validate_case_document(complete)


def test_an_overflowing_number_exits_2_without_a_traceback(capsys):
    from pydac.cli import main

    setting = "sorbent.heat_capacity=" + json.dumps(
        {"model": "constant", "parameters": {"value_J_kgK": 10**400}}
    )
    assert main(["validate", "--set", setting, "--validate"]) == 2
    assert "Traceback" not in capsys.readouterr().err


def test_the_json_schema_accepts_a_heat_the_run_ignores():
    jsonschema = pytest.importorskip("jsonschema")
    from pydac.config.schema_generation import generate_json_schema

    config = load_experiment(
        None, settings={"sorbent.heats_of_adsorption_J_mol": {"CO2": -81000.0, "H2O": -46000.0,
                                                              "N2": 10.0}}
    )
    jsonschema.Draft202012Validator(generate_json_schema()).validate(validate_case_document(config))


# --- Species keys do not depend on case (Mattia, 2026-10-08) ------------------

HEATS = "sorbent.heats_of_adsorption_J_mol"
LOADINGS_SETTING = "steps.startup.composition.initial_solid_loadings_mol_per_kg"
LDF = {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.003}}
FIXED = {"loading_mol_per_kg": 0.1, "heat_capacity_J_molK": 29.0}


def test_lowercase_species_keys_give_the_same_case_and_fingerprint():
    from pydac.inputs.case_io import case_fingerprint, load_case

    upper = load_experiment(CASE, settings={
        HEATS: {"CO2": -81000.0, "H2O": -46000.0},
        LOADINGS_SETTING: {"CO2": 0.5, "H2O": 1.25},
        "sorbent.fixed_adsorbates": {"N2": FIXED},
    })
    lower = load_experiment(CASE, settings={
        HEATS: {"co2": -81000.0, " H2o ": -46000.0},
        LOADINGS_SETTING: {"co2": 0.5, "h2o": 1.25},
        "sorbent.fixed_adsorbates": {"n2": FIXED},
    })
    assert case_fingerprint(lower) == case_fingerprint(upper)
    document = validate_case_document(lower)
    assert list(document["simulation"]["sorbent"]["heats_of_adsorption_J_mol"]) == ["CO2", "H2O"]
    assert list(document["simulation"]["sorbent"]["fixed_adsorbates"]) == ["N2"]
    # A frozen case written by hand with lower-case keys is the same case too.
    frozen = validate_case_document(upper)
    frozen["simulation"]["sorbent"]["heats_of_adsorption_J_mol"] = {"co2": -81000.0,
                                                                     "h2o": -46000.0}
    assert case_fingerprint(load_case(frozen)) == case_fingerprint(upper)


@pytest.mark.parametrize(
    ("settings", "species"),
    [
        ({HEATS: {"CO2": -1.0, "co2": -2.0, "H2O": -3.0}}, "CO2"),
        ({"sorbent.fixed_adsorbates": {"N2": FIXED, "n2 ": FIXED}}, "N2"),
        ({"adsorption.kinetics": {"CO2": LDF, "co2": LDF, "H2O": LDF}}, "CO2"),
        ({LOADINGS_SETTING: {"CO2": 0.5, "Co2": 0.5, "H2O": 1.0}}, "CO2"),
    ],
    ids=["heats", "fixed_adsorbates", "kinetics", "initial_loadings"],
)
def test_two_keys_for_one_species_are_refused_in_every_species_mapping(settings, species):
    with pytest.raises(ConfigurationError, match=f"names {species} twice"):
        load_experiment(CASE, settings=settings)
