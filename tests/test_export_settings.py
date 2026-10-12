"""`pydac.export_settings`: settings written as a preset file.

The file states only what the settings (and an optional base preset) state, and
loading it alone gives the case the settings give: same fingerprint. Where a
setting means something a preset cannot say field by field (a field of an
opaque block, an inheriting null over a base, a computed field a setting made
stale), the export writes what the setting meant. Nothing is written for
invalid settings or over an existing file.
"""

from __future__ import annotations

import json
import math

import pytest

import pydac
from pydac.cli import main
from pydac.config.errors import ConfigurationError
from pydac.inputs.case_io import case_fingerprint


TWEAKS = {
    "grid.nz": 7,
    "mixture.adsorbates": ["CO2", "H2O"],
    "sorbent.heat_capacity": {"model": "constant", "parameters": {"value_J_kgK": 1100.0}},
    "steps.w_des.thermal.jacket_temperature_K": 400.0,
    "execution.extra_tag": "exported",
}


def _same_case(path, base=None, settings=None) -> bool:
    return case_fingerprint(pydac.load_case(path)) == case_fingerprint(
        pydac.load_case(base, settings=settings)
    )


def test_the_file_states_only_the_settings_and_loads_the_same_case(tmp_path):
    target = tmp_path / "sub" / "preset.json"
    assert pydac.export_settings(TWEAKS, target) == target
    assert json.loads(target.read_text()) == {
        "schema_version": 8,
        "simulation": {
            "grid": {"nz": 7},
            "mixture": {"adsorbates": ["CO2", "H2O"]},
            "sorbent": {"heat_capacity": TWEAKS["sorbent.heat_capacity"]},
            "cycle": {"steps": {"w_des": {"thermal": {"jacket_temperature_K": 400.0}}}},
        },
        "execution": {"extra_tag": "exported"},
    }
    assert _same_case(target, settings=TWEAKS)
    # A list of plain values stays on one line, as in the hand-written presets.
    assert '"adsorbates": ["CO2", "H2O"]' in target.read_text()


def test_every_settings_spelling_gives_the_same_file(tmp_path):
    nested = {"simulation": {"grid": {"nz": 7}}, "cycle": {"steps": {"w_des": {"thermal": {
        "jacket_temperature_K": 400.0}}}}}
    cli = ["grid.nz=7", "cycle.steps.w_des.thermal.jacket_temperature_K=400.0"]
    texts = {
        pydac.export_settings(settings, tmp_path / f"{index}.json").read_text()
        for index, settings in enumerate((nested, cli))
    }
    assert len(texts) == 1


def test_invalid_settings_and_an_existing_file_are_refused_before_writing(tmp_path):
    target = tmp_path / "preset.json"
    with pytest.raises(ConfigurationError, match="Unknown setting path"):
        pydac.export_settings({"grid.nzz": 7}, target)
    with pytest.raises(ConfigurationError, match="Unknown cycle step"):
        pydac.export_settings({"steps.dry.horizon.nominal_time_s": 1.0}, target)
    with pytest.raises(ConfigurationError):
        pydac.export_settings({"mixture.adsorbates": ["CO2", "N2"]}, target)
    assert not target.exists()
    pydac.export_settings({"grid.nz": 7}, target)
    with pytest.raises(ConfigurationError, match="overwrite=True"):
        pydac.export_settings({"grid.nz": 9}, target)
    assert json.loads(target.read_text())["simulation"]["grid"] == {"nz": 7}
    pydac.export_settings({"grid.nz": 9}, target, overwrite=True)
    assert json.loads(target.read_text())["simulation"]["grid"] == {"nz": 9}


def test_a_base_preset_is_kept_under_the_settings(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"schema_version": 8, "simulation": {
        "grid": {"nz": 5, "nr": 3},
        "cycle": {"steps": {"startup": {"thermal": {"jacket_temperature_K": 330.0}}}},
    }}))
    settings = {"grid.nz": 7, "steps.startup.thermal.jacket_temperature_K": None}
    target = pydac.export_settings(settings, tmp_path / "preset.json", base=base)
    written = json.loads(target.read_text())["simulation"]
    assert written["grid"] == {"nz": 7, "nr": 3}
    # A null setting inherits what it sits on: here the base value, not the default.
    assert written["cycle"]["steps"]["startup"]["thermal"]["jacket_temperature_K"] == 330.0
    assert _same_case(target, base, settings)


def test_a_disabling_null_is_written_as_null(tmp_path):
    base = {"execution": {"extra_tag": "tagged"}}
    target = pydac.export_settings({"execution.extra_tag": None}, tmp_path / "p.json", base=base)
    assert json.loads(target.read_text())["execution"] == {"extra_tag": None}
    assert pydac.load_case(target).execution.extra_tag is None


def test_one_field_of_a_block_a_preset_gives_whole_writes_the_whole_block(tmp_path):
    # A preset gives the bed (and its opaque `design`) as one block; the setting
    # changed one field of it.
    settings = {"bed.design.eps_b": 0.4}
    target = pydac.export_settings(settings, tmp_path / "p.json")
    bed = json.loads(target.read_text())["simulation"]["bed"]
    assert bed["design"]["eps_b"] == 0.4 and {"architecture", "material"} <= set(bed)
    assert _same_case(target, settings=settings)
    settings = {"sorbent.heat_capacity.parameters.b_J_per_kgK2": 2.5}
    target = pydac.export_settings(settings, tmp_path / "q.json")
    heat_capacity = json.loads(target.read_text())["simulation"]["sorbent"]["heat_capacity"]
    assert heat_capacity["model"] == "Low2023"
    assert heat_capacity["parameters"]["b_J_per_kgK2"] == 2.5
    assert len(heat_capacity["parameters"]) == 3
    assert _same_case(target, settings=settings)


def test_a_bed_alias_in_the_settings_or_the_base(tmp_path):
    settings = {"bed": "packed_bed_default"}
    target = pydac.export_settings(settings, tmp_path / "a.json")
    assert _same_case(target, settings=settings)
    base = {"simulation": {"bed": "packed_bed_default"}}
    target = pydac.export_settings({"bed.design.eps_b": 0.4}, tmp_path / "b.json", base=base)
    assert _same_case(target, base, {"bed.design.eps_b": 0.4})


def test_a_computed_geometry_field_the_settings_make_stale_is_left_out(tmp_path):
    geometry = pydac.load_case().simulation.geometry
    base = {"simulation": {"geometry": {
        "specified_by": "length_radius", "length_m": geometry.length_m,
        "radius_m": geometry.radius_m, "volume_m3": geometry.volume_m3,
    }}}
    settings = {"geometry.length_m": 2 * geometry.length_m}
    target = pydac.export_settings(settings, tmp_path / "p.json", base=base)
    written = json.loads(target.read_text())["simulation"]["geometry"]
    assert "volume_m3" not in written
    assert math.isclose(pydac.load_case(target).simulation.geometry.volume_m3,
                        2 * geometry.volume_m3)
    assert _same_case(target, base, settings)


def test_a_steam_flow_the_settings_make_stale_is_left_out(tmp_path):
    frozen = pydac.export_case(pydac.load_case(), tmp_path / "defaults.json")
    flow = json.loads(frozen.read_text())["simulation"]["cycle"]["steps"]["s_des"]["flow"]
    base = {"simulation": {"cycle": {"steps": {"s_des": {"flow": {
        "steam_mass_flow_kg_s": flow["steam_mass_flow_kg_s"],
        "volumetric_flow_m3_s": flow["volumetric_flow_m3_s"],
    }}}}}}
    settings = {"steps.s_des.flow.steam_mass_flow_kg_s": 2 * flow["steam_mass_flow_kg_s"]}
    target = pydac.export_settings(settings, tmp_path / "p.json", base=base)
    written = json.loads(target.read_text())["simulation"]["cycle"]["steps"]["s_des"]["flow"]
    assert "volumetric_flow_m3_s" not in written
    assert _same_case(target, base, settings)


def test_cli_export_preset(tmp_path, capsys):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"simulation": {"grid": {"nr": 3}}}))
    target = tmp_path / "preset.json"
    command = ["validate", "--config", str(base), "--set", "grid.nz=7",
               "--export-preset", str(target)]
    assert main(command) == 0
    assert f"Exported preset: {target}" in capsys.readouterr().out
    assert json.loads(target.read_text())["simulation"]["grid"] == {"nr": 3, "nz": 7}
    assert target.read_text() == pydac.export_settings(
        ["grid.nz=7"], tmp_path / "python.json", base=base).read_text()
    assert main(command) == 2
    assert "--overwrite-preset" in capsys.readouterr().err
    assert main(command + ["--overwrite-preset"]) == 0
    assert main(["validate", "--overwrite-preset"]) == 2
    assert "--overwrite-preset requires --export-preset" in capsys.readouterr().err
    frozen = pydac.export_case(pydac.load_case(), tmp_path / "frozen.json")
    assert main(["validate", "--case", str(frozen), "--export-preset",
                 str(tmp_path / "x.json")]) == 2
    assert "a --case is already complete" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


def test_a_preset_that_would_load_another_case_is_not_written(tmp_path, monkeypatch):
    # Guard: if the preset ever lost a setting's meaning, refuse instead of writing it.
    import pydac.config.loader as loader

    monkeypatch.setattr(loader, "settings_to_preset", lambda settings, base=None: {})
    target = tmp_path / "p.json"
    with pytest.raises(ConfigurationError, match="cannot be written as a preset"):
        pydac.export_settings({"grid.nz": 7}, target)
    assert not target.exists()


# --------------------------------------------------------------------------
# Regressions: integer keys, steam-free flows, terminal = Python, no partial writes
# --------------------------------------------------------------------------

def test_es001_integer_mapping_keys_give_a_file_that_loads(tmp_path):
    settings = {"execution.cycle_options": {1: {}}}
    target = pydac.export_settings(settings, tmp_path / "p.json")
    assert _same_case(target, settings=settings)


@pytest.mark.parametrize("step", ["startup", "ads", "press"])
def test_es003_a_stated_flow_of_a_step_without_steam_is_kept(tmp_path, step):
    base = {"simulation": {"cycle": {"steps": {step: {"flow": {
        "volumetric_flow_m3_s": 0.002}}}}}}
    settings = {f"steps.{step}.thermal.feed_temperature_K": 400.0}
    target = pydac.export_settings(settings, tmp_path / "p.json", base=base)
    written = json.loads(target.read_text())["simulation"]["cycle"]["steps"][step]["flow"]
    assert written == {"volumetric_flow_m3_s": 0.002}
    assert _same_case(target, base, settings)


@pytest.mark.parametrize("step", ["s_des", "s_purge"])
def test_es003_a_steam_step_switched_off_keeps_its_stated_flow(tmp_path, step):
    frozen = pydac.export_case(pydac.load_case(), tmp_path / "defaults.json")
    flow = json.loads(frozen.read_text())["simulation"]["cycle"]["steps"][step]["flow"]
    base = {"simulation": {"cycle": {"steps": {step: {"flow": flow}}}}}
    settings = {f"steps.{step}.flow.steam_mass_flow_kg_s": 0}
    target = pydac.export_settings(settings, tmp_path / "p.json", base=base)
    assert _same_case(target, base, settings)


@pytest.mark.parametrize("settings", [
    ["steps.s_des.flow.steam_mass_flow_kg_s=0"],
    ["steps.s_purge.flow.steam_mass_flow_kg_s=0"],
    ["geometry.specified_by=length_radius", "geometry.length_m=0.2", "geometry.radius_m=0.05"],
])
def test_es002_one_terminal_call_writes_one_case(tmp_path, capsys, settings):
    # The terminal builds the case as load_case does, so every file of one call
    # (frozen case, resolved config, preset) is the same case as Python's.
    files = {name: tmp_path / f"{name}.json" for name in ("case", "resolved", "preset")}
    assert main(["validate", *[token for item in settings for token in ("--set", item)],
                 "--export-case", str(files["case"]),
                 "--write-resolved", str(files["resolved"]),
                 "--export-preset", str(files["preset"])]) == 0
    python = case_fingerprint(pydac.load_case(settings=settings))
    assert case_fingerprint(pydac.load_case(files["case"], exact=True)) == python
    assert case_fingerprint(pydac.load_case(files["resolved"])) == python
    assert case_fingerprint(pydac.load_case(files["preset"])) == python


def test_es004_a_refused_call_writes_no_file(tmp_path, capsys):
    full, preset = tmp_path / "full.json", tmp_path / "preset.json"
    # An orphan --overwrite-preset is refused before --export-case writes.
    assert main(["validate", "--overwrite-preset", "--export-case", str(full)]) == 2
    assert not full.exists()
    # An existing preset target is refused before --export-case writes.
    preset.write_text("{}")
    assert main(["validate", "--export-case", str(full), "--export-preset", str(preset)]) == 2
    assert not full.exists() and preset.read_text() == "{}"
    # And an existing case target before the preset is written.
    full.write_text("{}")
    preset.unlink()
    assert main(["validate", "--export-case", str(full), "--export-preset", str(preset)]) == 2
    assert not preset.exists() and full.read_text() == "{}"
    capsys.readouterr()
