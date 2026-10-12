from __future__ import annotations

import json
from pathlib import Path

import pytest

from pydac.core.column_model_factory import ColumnModelFactory
from pydac.config import (
    ConfigurationError,
    experiment_to_document,
    load_experiment,
    operating_step_settings,
)
from pydac.config.cycle import CycleConfig
from pydac.config.migrations import (
    CURRENT_SCHEMA_VERSION,
    V4_VALVE_FIELD_RENAMES,
    V5_WALL_FIELD_RENAMES,
)
from pydac.config.input_help import settings_catalog
from pydac.config.schema_generation import PACKAGED_SCHEMA_PATH, render_json_schema
from tests.test_sorbent_declarations import _with_v6_sorbent_fields
from pydac.inputs.defaults.steps import cycle_defaults
from pydac.inputs.case_io import case_fingerprint, load_case


LEGACY_CASE = "tests/cases/v3_smoke_startup.json"


@pytest.mark.parametrize("relative_root", [False, True])
def test_local_optimizer_launcher_honors_output_root_and_workers(monkeypatch, tmp_path, relative_root):
    from runners import run_NSGA2 as runner

    monkeypatch.chdir(tmp_path)
    output_root = Path("chosen output") if relative_root else tmp_path / "chosen output"
    bounds_path = tmp_path / "bounds.json"
    bounds_path.write_text(json.dumps({
        "t_ads": [900, 1800], "t_heat": [300, 600], "t_des": [300, 900],
        "P_vac": [5000, 10000], "v_feed": [0.01, 0.1], "v_steam": [0.001, 0.01],
    }))
    calls = []
    from pydac.workflows import orchestrator_NSGA2

    monkeypatch.setattr(orchestrator_NSGA2, "run_nsga2_optimization",
                        lambda config, **kwargs: calls.append((config, kwargs)))

    assert runner.main([
        "--bounds", str(bounds_path), "--workers", "2", "--pop-size", "4",
        "--generations", "2", "--set", f"execution.output_root={output_root}",
    ]) == 0

    assert len(calls) == 1
    config, options = calls[0]
    folder = Path(config.base_folder)
    assert folder.parent == output_root / "nsga2"
    assert options["n_procs"] == 2
    saved = json.loads((folder / "resolved_config.json").read_text())
    assert saved["execution"]["output_root"] == str(output_root)
    assert not (tmp_path / "simulation_results").exists()


def test_every_reference_cycle_step_uses_external_htc_1000():
    external_htc_by_step = {
        name: step["thermal"]["external_htc_W_m2K"]
        for name, step in cycle_defaults()["steps"].items()
    }

    assert set(external_htc_by_step.values()) == {1000.0}, external_htc_by_step


def test_optimized_desorption_time_updates_both_event_horizons():
    duration_s = 21_600.0
    config = load_experiment(
        LEGACY_CASE,
        settings=operating_step_settings({"desorption_time_s": duration_s}),
    )
    steps = {
        step.name: step.config for step in config.simulation.cycle.steps
    }

    for step_name in ("s_des", "w_des"):
        horizon = steps[step_name].horizon
        assert horizon.nominal_time_s == pytest.approx(duration_s)
        assert horizon.event_max_time_s == pytest.approx(duration_s)


def _breakthrough_settings():
    return {
        "mixture": {
            "species": ["CO2", "N2"],
            "closure_species": "N2",
            "adsorbates": ["CO2"],
        },
        "adsorption": {
            "equilibrium_systems": {
                "Toth": {
                    "datasets": {"Toth": "Cleaned-Toth"},
                    "parameters": {},
                }
            },
            "kinetics": {
                "CO2": {
                    "model": "LDF",
                    "parameters": {"k_LDF_s_inv": 0.003},
                }
            },
        },
    }


def test_schema_v1_case_upgrades_to_named_current_declaration():
    config = load_experiment(LEGACY_CASE)
    document = experiment_to_document(config)

    assert config.schema_version == CURRENT_SCHEMA_VERSION
    assert document["simulation"]["mixture"] == {
        "species": ["CO2", "H2O", "O2", "N2"],
        "closure_species": "N2",
        "adsorbates": ["CO2", "H2O"],
    }
    assert document["simulation"]["adsorption"] == {
        "equilibrium_systems": {
            "TothGABWADST": {
                "datasets": {"Toth": "Cleaned-Toth"},
                "parameters": {},
            }
        },
        "kinetics": {
            "CO2": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.003}},
            "H2O": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.00083}},
        },
    }


# Derived case values (e.g. the steam volumetric flow) differ in the last digits across
# numerical environments, so this fingerprint baseline holds only in the qualification lock.
@pytest.mark.qualification
def test_legacy_frozen_case_and_resolved_current_document_have_one_fingerprint():
    legacy = load_case(LEGACY_CASE)
    assert legacy.schema_version == CURRENT_SCHEMA_VERSION
    assert case_fingerprint(LEGACY_CASE) == case_fingerprint(legacy)


def _schema_v2_case(linear_solver):
    """A complete schema-v2 case: the current one plus the solver choice v3 dropped,
    minus the outlet group v4 added, with the valve names v5, the wall name v6 and the
    sorbent fields v7 renamed."""
    document = experiment_to_document(load_case(LEGACY_CASE))
    document["schema_version"] = 2
    document["simulation"]["solver"]["krylov_min_states"] = 2500
    for step in document["simulation"]["cycle"]["steps"].values():
        step["solver"] = {"linear_solver": linear_solver}
        del step["outlet"]
        for old, new in V4_VALVE_FIELD_RENAMES.items():
            step["flow"][old] = step["flow"].pop(new)
    del document["simulation"]["geometry"]["specified_by"]
    for old, new in V5_WALL_FIELD_RENAMES.items():
        document["simulation"]["wall"][old] = document["simulation"]["wall"].pop(new)
    return _with_v6_sorbent_fields(document)


@pytest.mark.parametrize("linear_solver", ["direct", "auto", "AUTO"])
def test_schema_v2_direct_or_auto_case_upgrades_to_the_same_current_case(linear_solver):
    current = load_case(LEGACY_CASE)
    upgraded = load_case(_schema_v2_case(linear_solver))

    assert upgraded.schema_version == CURRENT_SCHEMA_VERSION
    assert experiment_to_document(upgraded) == experiment_to_document(current)
    assert case_fingerprint(_schema_v2_case(linear_solver)) == case_fingerprint(current)


def test_schema_v2_case_asking_for_gmres_is_refused():
    document = _schema_v2_case("direct")
    document["simulation"]["cycle"]["steps"]["heat"]["solver"]["linear_solver"] = "gmres"

    with pytest.raises(ConfigurationError, match="steps.heat.solver.linear_solver.*GMRES was removed"):
        load_case(document)


def test_schema_v2_solver_fields_are_unknown_in_the_current_schema():
    document = experiment_to_document(load_case(LEGACY_CASE))
    document["simulation"]["cycle"]["steps"]["heat"]["solver"] = {"linear_solver": "direct"}

    with pytest.raises(ConfigurationError, match="solver"):
        load_case(document)


def test_schema_v1_dry_selection_upgrades_to_two_independent_systems():
    legacy = json.loads(Path(LEGACY_CASE).read_text(encoding="utf-8"))
    legacy["simulation"]["adsorption"]["co_adsorption"] = False

    config = load_experiment(legacy)
    assert tuple(config.simulation.adsorption.equilibrium_systems) == (
        "Toth",
        "GAB",
    )
    bundle = ColumnModelFactory(CycleConfig.from_experiment(config)).components
    assert bundle.equilibrium_system.model_names == ("Toth", "GAB")


def test_generic_breakthrough_declaration_round_trips_exactly_and_builds_models():
    config = load_experiment(settings=_breakthrough_settings())
    document = experiment_to_document(config)

    assert experiment_to_document(load_experiment(document)) == document
    assert experiment_to_document(load_case(document)) == document

    bundle = ColumnModelFactory(CycleConfig.from_experiment(config)).components
    assert bundle.equilibrium_system.species == ("CO2",)
    assert tuple(bundle.kinetics) == ("CO2",)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        (
            {
                **_breakthrough_settings(),
                "mixture": {
                    "species": ["CO2", "Xe", "N2"],
                    "closure_species": "N2",
                    "adsorbates": ["CO2"],
                },
            },
            "Unknown species 'XE'",
        ),
        (
            {"mixture.closure_species": "Ar"},
            "Closure species 'AR' is absent",
        ),
        (
            {
                "adsorption.equilibrium_systems": {
                    "Toth": {
                        "datasets": {"Toth": "Cleaned-Toth"},
                        "parameters": {},
                    }
                }
            },
            "Equilibrium models must cover every adsorbate exactly once",
        ),
        (
            {
                "adsorption.kinetics": {
                    "CO2": {
                        "model": "LDF",
                        "parameters": {"k_LDF_s_inv": 0.003},
                    }
                }
            },
            "Kinetic models must cover every adsorbate exactly once",
        ),
        (
            {
                "adsorption.kinetics": {
                    "CO2": {"model": "NoSuchModel", "parameters": {}},
                    "H2O": {
                        "model": "LDF",
                        "parameters": {"k_LDF_s_inv": 0.00083},
                    },
                }
            },
            "Unknown model 'NoSuchModel'",
        ),
        (
            {
                "adsorption.kinetics.CO2.parameters.k_LDF_s_inv": -0.003,
            },
            "k_LDF_s_inv.*greater than zero",
        ),
    ],
)
def test_invalid_physical_declarations_fail_during_loading(settings, match):
    with pytest.raises(ConfigurationError, match=match):
        load_experiment(settings=settings)


def test_schema_id_matches_release_repository_and_version():
    import tomllib

    from pydac.utils.version import __version__

    metadata = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    repository = metadata["project"]["urls"]["Repository"]
    expected = (
        repository.replace("https://github.com/", "https://raw.githubusercontent.com/")
        + f"/v{__version__}/src/pydac/config/json_schema/experiment.schema.json"
    )
    assert json.loads(render_json_schema())["$id"] == expected
    assert json.loads(PACKAGED_SCHEMA_PATH.read_text(encoding="utf-8"))["$id"] == expected


def test_generated_schema_and_show_settings_expose_named_configuration():
    packaged = json.loads(PACKAGED_SCHEMA_PATH.read_text(encoding="utf-8"))
    generated = json.loads(render_json_schema())
    assert packaged == generated

    simulation = generated["$defs"]["simulation"]["properties"]
    assert set(simulation["mixture"]["properties"]) == {
        "species",
        "closure_species",
        "adsorbates",
    }
    assert set(simulation["adsorption"]["properties"]) == {
        "equilibrium_systems",
        "kinetics",
    }

    catalog = settings_catalog()
    assert "mixture.closure_species" in catalog
    assert "adsorption.equilibrium_systems" in catalog
    assert "k_LDF_s_inv" in catalog


# --------------------------------------------------------------------------
# Column size: one input pair, the other computed
# --------------------------------------------------------------------------

def _geometry(settings=None, source=None):
    return dict(load_experiment(source, settings=settings).simulation.geometry.items())


def test_the_column_size_is_given_by_volume_and_aspect_ratio_or_by_length_and_radius():
    import math

    default = _geometry()
    assert default["specified_by"] == "volume_aspect_ratio"
    assert default["length_m"] == pytest.approx(2 * default["aspect_ratio"] * default["radius_m"])

    by_size = _geometry({"geometry.specified_by": "length_radius",
                         "geometry.length_m": 0.1, "geometry.radius_m": 0.05})
    assert by_size["aspect_ratio"] == pytest.approx(1.0)
    assert by_size["volume_m3"] == pytest.approx(math.pi * 0.05**2 * 0.1)


@pytest.mark.parametrize("settings, message", [
    ({"geometry.length_m": 1.0}, "is computed from"),
    ({"geometry.specified_by": "length_radius", "geometry.length_m": 0.1}, "needs"),
    ({"geometry.specified_by": "length_radius", "geometry.length_m": 0.1,
      "geometry.radius_m": 0.05, "geometry.volume_m3": 1.0}, "is computed from"),
])
def test_a_case_gives_the_column_size_one_way_only(settings, message):
    with pytest.raises(ConfigurationError, match=message):
        load_experiment(None, settings=settings)


def test_a_setting_on_a_resolved_case_recomputes_the_other_pair():
    """A frozen case stores all four values; changing an input must not trip on them."""
    resolved = load_experiment(None)
    varied = dict(load_experiment(resolved, settings={"geometry.aspect_ratio": 2.0})
                  .simulation.geometry.items())
    assert varied["length_m"] == pytest.approx(4 * varied["radius_m"])
    switched = dict(load_experiment(resolved, settings={
        "geometry.specified_by": "length_radius", "geometry.length_m": 0.2,
        "geometry.radius_m": 0.05}).simulation.geometry.items())
    assert switched["aspect_ratio"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Steam-fed steps: the volumetric flow follows the steam mass flow
# --------------------------------------------------------------------------

def _steam_flow(experiment, step="s_des"):
    steps = {item.name: item.config for item in experiment.simulation.cycle.steps}
    return steps[step].flow.volumetric_flow_m3_s


def test_a_written_steam_volumetric_flow_must_equal_the_derived_one():
    derived = _steam_flow(load_experiment(None))
    same = load_experiment(None, settings={"steps.s_des.flow.volumetric_flow_m3_s": derived})
    assert _steam_flow(same) == derived
    with pytest.raises(ConfigurationError, match="is computed from the step's"):
        load_experiment(None, settings={"steps.s_des.flow.volumetric_flow_m3_s": 1.0})


def test_a_hand_edited_frozen_case_with_a_different_steam_flow_is_refused():
    document = experiment_to_document(load_experiment(None))
    document["simulation"]["cycle"]["steps"]["s_des"]["flow"]["volumetric_flow_m3_s"] = 0.5
    with pytest.raises(ConfigurationError, match="is computed from the step's"):
        load_case(document)
    with pytest.raises(ConfigurationError, match="is computed from the step's"):
        load_experiment(document)


@pytest.mark.parametrize("setting", [
    {"steps.s_des.flow.steam_mass_flow_kg_s": 0.002},
    {"steps.s_des.pressure.feed_pressure_Pa": 10000.0},
    {"steps.s_des.thermal.feed_temperature_K": 393.15},
    operating_step_settings({"steam_mass_flow_kg_s": 0.002, "low_pressure_Pa": 10000.0}),
])
def test_a_setting_on_a_resolved_case_recomputes_the_steam_flow(setting):
    """A frozen case stores the derived flow; changing one of its inputs must not trip on it."""
    resolved = load_experiment(None)
    varied = load_experiment(resolved, settings=setting)
    assert _steam_flow(varied) != pytest.approx(_steam_flow(resolved))
