"""Every supported case schema upgrades to the current case, and only its own names.

A complete current case is written back in the shape of each older schema; it must
upgrade to the same document, fingerprint and typed case. A field name a later
schema introduced is refused in every schema before it.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from pydac.config import ConfigurationError, load_experiment
from pydac.config.migrations import (
    CURRENT_SCHEMA_VERSION,
    PRE_SORBENT_MAPPING_SCHEMA_VERSION,
    V4_VALVE_FIELD_RENAMES,
    V5_WALL_FIELD_RENAMES,
    V6_HEATS_OF_ADSORPTION_J_MOL,
)
from pydac.inputs.case_io import case_fingerprint, load_case, validate_case_document
from tests.test_initial_solid_loadings import without_initial_solid_loadings
from tests.test_sorbent_declarations import _with_v6_sorbent_fields


LEGACY_CASE = "tests/cases/v3_smoke_startup.json"
OLDER_VERSIONS = range(1, CURRENT_SCHEMA_VERSION)


def _complete():
    return validate_case_document(load_case(LEGACY_CASE))


def _written_as(version):
    """The current reference case in the shape of a complete schema-`version` case."""
    document = _complete()
    simulation = document["simulation"]
    steps = simulation["cycle"]["steps"].values()
    if version == 7:
        without_initial_solid_loadings(document)
    if version <= 6:
        _with_v6_sorbent_fields(document)  # also drops the v8 startup loadings
    if version <= 5:
        for old, new in V5_WALL_FIELD_RENAMES.items():
            simulation["wall"][old] = simulation["wall"].pop(new)
    if version <= 4:
        for step in steps:
            for old, new in V4_VALVE_FIELD_RENAMES.items():
                step["flow"][old] = step["flow"].pop(new)
    if version <= 3:
        for step in steps:
            del step["outlet"]
        del simulation["geometry"]["specified_by"]
    if version == 2:
        simulation["solver"]["krylov_min_states"] = 2500
        for step in steps:
            step["solver"] = {"linear_solver": "direct"}
    if version == 1:
        del simulation["mixture"]
        with open(LEGACY_CASE, encoding="utf-8") as handle:
            simulation["adsorption"] = json.load(handle)["simulation"]["adsorption"]
    document["schema_version"] = version
    return document


@pytest.mark.parametrize("version", OLDER_VERSIONS)
def test_a_complete_older_case_upgrades_to_the_same_case(version):
    current = load_case(LEGACY_CASE)
    assert validate_case_document(_written_as(version)) == _complete()
    assert case_fingerprint(_written_as(version)) == case_fingerprint(current)
    assert load_case(_written_as(version)) == current


def _with_a_name_since(document, since):
    simulation = document["simulation"]
    if since == 8:
        startup = simulation["cycle"]["steps"]["startup"]["composition"]
        startup["initial_solid_loadings_mol_per_kg"] = {"CO2": 0.0, "H2O": 0.0}
    elif since == 7:
        simulation["sorbent"]["heats_of_adsorption_J_mol"] = dict(V6_HEATS_OF_ADSORPTION_J_MOL)
    elif since == 6:
        simulation["wall"]["overall_htc_override_W_m2K"] = None
    elif since == 5:
        simulation["cycle"]["steps"]["blow"]["flow"]["blowdown_valve_conductance"] = None
    elif since == 4:
        simulation["cycle"]["steps"]["ads"]["outlet"] = {"collect_product": False}
    return document


@pytest.mark.parametrize(
    ("since", "version"),
    [(since, version) for since in (4, 5, 6, 7, 8) for version in range(1, since)],
)
def test_a_name_from_a_later_schema_is_refused_in_every_earlier_one(since, version):
    document = _with_a_name_since(_written_as(version), since)
    with pytest.raises(ConfigurationError):
        validate_case_document(document)


def test_the_v1_adsorption_fields_translate_to_these_blocks():
    """An oracle independent of the migration: the matrix above derives every
    expected case through it, so a wrong translation would move both sides."""
    with open(LEGACY_CASE, encoding="utf-8") as handle:
        legacy = json.load(handle)["simulation"]["adsorption"]
    assert legacy["k_ldf_co2"] == 0.003 and legacy["k_ldf_h2o"] == 0.00083
    assert _complete()["simulation"]["adsorption"] == {
        "equilibrium_systems": {
            "TothGABWADST": {"datasets": {"Toth": "Cleaned-Toth"}, "parameters": {}},
        },
        "kinetics": {
            "CO2": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.003}},
            "H2O": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.00083}},
        },
    }


@pytest.mark.parametrize("species", ["CO2", "H2O"])
def test_a_schema_v6_overlay_stating_one_heat_keeps_the_other_reference_value(species):
    field = f"heat_of_adsorption_{species.lower()}_J_mol"
    config = load_experiment({
        "schema_version": PRE_SORBENT_MAPPING_SCHEMA_VERSION,
        "simulation": {"sorbent": {field: -50000.0}},
    })
    assert dict(config.simulation.sorbent.heats_of_adsorption_J_mol) == {
        **V6_HEATS_OF_ADSORPTION_J_MOL, species: -50000.0,
    }


def test_an_early_older_case_missing_a_later_field_says_how_to_update_it(tmp_path):
    """Early builds wrote some schemas before all their fields existed: such a case is
    refused with the way to bring it up to date, and that way works."""
    document = _written_as(1)
    del document["simulation"]["wall"]["bed_wall_htc_override_W_m2K"]
    with pytest.raises(ConfigurationError, match="missing field.*load it as a preset"):
        validate_case_document(document)

    import pydac

    path = tmp_path / "early_v1.json"
    path.write_text(json.dumps(document))
    updated = tmp_path / "updated.json"
    pydac.export_case(pydac.load_case(path), updated)
    assert json.loads(updated.read_text())["schema_version"] == CURRENT_SCHEMA_VERSION
    assert pydac.load_case(updated, exact=True) == pydac.load_case(path)
