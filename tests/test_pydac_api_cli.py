"""The public `pydac` layer: Python API, `pydac` command, outcomes and run folder rules."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pydac
from pydac import api
from pydac.cli import main
from pydac.config.errors import ConfigurationError, RunRefused
from pydac.inputs.case_io import case_fingerprint
from pydac.utils.run.resume import resolve_resume

REPO = Path(__file__).resolve().parents[1]
CASE = str(REPO / "tests" / "cases" / "v3_smoke_startup.json")
CSS_STEPS = '["ads","blow","heat","s_des","press","cool"]'
QUIET_CSS = [
    f"execution.cycle_steps={CSS_STEPS}",
    "execution.termination_mode=time",
    "execution.plot_cyclic_profiles=false",
    "execution.generate_kpi_outputs=false",
]
# The smoke case stops startup at 2 s; an optimized t_ads needs room.
NSGA2_CASE = {
    "execution.cycle_steps": ["ads", "blow", "heat", "s_des", "press", "cool"],
    "steps.startup.horizon.event_max_time_s": 43200,
}
BOUNDS = {
    "t_ads": [900.0, 1800.0], "t_heat": [300.0, 600.0], "t_des": [300.0, 900.0],
    "P_vac": [5000.0, 10000.0], "v_feed": [0.01, 0.1], "v_steam": [0.001, 0.01],
}


def _set(settings):
    return [token for setting in settings for token in ("--set", setting)]


def _printed_folders(out: str, label="Run folder") -> set[Path]:
    return {Path(line.split(": ", 1)[1]) for line in out.splitlines() if line.startswith(label + ": ")}


def _hashes(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob("*")) if path.is_file()}


# --------------------------------------------------------------------------
# Package surface
# --------------------------------------------------------------------------

def test_public_names_and_light_import():
    for name in ("load_case", "export_case", "export_settings", "run_steps", "run_css",
                 "run_nsga2", "resume", "report", "RunResult", "ConfigurationError", "RunRefused",
                 "ExperimentConfig", "CycleDataLoader", "CycleKPIs", "__version__"):
        assert name in pydac.__all__ and getattr(pydac, name) is not None
    assert issubclass(pydac.RunRefused, pydac.ConfigurationError)
    assert issubclass(pydac.ConfigurationError, ValueError)
    # The optional NSGA-II dependency and the plotting stack load only on use.
    probe = ("import sys, pydac, pydac.cli; pydac.load_case; pydac.cli.build_parser(); "
             "print(sorted(m for m in ('pymoo', 'matplotlib') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_version_command():
    out = subprocess.run([sys.executable, "-m", "pydac.cli", "--version"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == f"pydac {pydac.__version__}"


def test_run_functions_take_only_a_loaded_case(tmp_path):
    for run in (pydac.run_steps, pydac.run_css):
        with pytest.raises(ConfigurationError, match=r"pydac.load_case\(path\)"):
            run(CASE)
    with pytest.raises(ConfigurationError, match=r"pydac.load_case\(path\)"):
        pydac.run_nsga2(CASE, BOUNDS)
    with pytest.raises(ConfigurationError, match="accepts no settings"):
        pydac.load_case(CASE, exact=True, settings={"grid.nz": 3})
    with pytest.raises(ConfigurationError, match="give its path"):
        pydac.load_case(exact=True)
    case = pydac.load_case(CASE, settings=["grid.nz=6"])
    assert case.simulation.grid.nz == 6
    assert pydac.load_case(CASE, settings={"grid": {"nz": 6}}) == case
    target = tmp_path / "case.json"
    assert pydac.export_case(case, target) == target
    with pytest.raises(ConfigurationError, match="Refusing to overwrite"):
        pydac.export_case(case, target)
    assert pydac.load_case(target, exact=True) == case


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def test_validate_exit_codes_and_export(tmp_path, capsys):
    target = tmp_path / "case.json"
    assert main(["validate", "--config", CASE, "--export-case", str(target)]) == 0
    assert target.is_file()
    assert main(["validate", "--config", CASE, "--export-case", str(target)]) == 2
    assert "Configuration error: Refusing to overwrite" in capsys.readouterr().err
    assert main(["validate", "--config", CASE, "--set", "grid.nz=7",
                 "--export-case", str(target), "--overwrite-case"]) == 0
    assert main(["validate", "--case", str(target)]) == 0
    assert main(["validate", "--case", str(target), "--set", "grid.nz=3"]) == 2
    captured = capsys.readouterr()
    assert "Configuration error: Frozen-case mode" in captured.err
    assert "Configuration error" not in captured.out
    assert main(["validate", "--config", str(tmp_path / "missing.json")]) == 2


# --------------------------------------------------------------------------
# Step runs: same run folder from the CLI and Python; occupied-folder behavior
# --------------------------------------------------------------------------

def test_same_step_case_from_cli_and_python(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    case_path = tmp_path / "case.json"
    assert main(["validate", "--config", CASE, "--export-case", str(case_path)]) == 0
    capsys.readouterr()

    assert main(["steps", "--case", str(case_path)]) == 0
    printed = _printed_folders(capsys.readouterr().out)
    assert len(printed) == 1
    run = printed.pop()
    assert run.is_absolute() and run.parent == tmp_path / "simulation_results" / "step_mode"
    saved = json.loads((run / "1D_mode" / "case_metadata.json").read_text())
    case = pydac.load_case(case_path, exact=True)
    assert saved["case_fingerprint"] == case_fingerprint(case)
    assert run.name.endswith(case_fingerprint(case)[:8])

    # The same case from Python names the same folder, which is now occupied.
    with pytest.raises(RunRefused, match="already holds a run"):
        pydac.run_steps(case)
    result = pydac.run_steps(case, overwrite=True)
    assert result == pydac.RunResult(folder=run, results_folder=run / "1D_mode", workflow="steps")
    assert _printed_folders(capsys.readouterr().out) == {run}


def test_occupied_folder_refusal_and_overwrite_scope(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    case = pydac.load_case(CASE)
    run = Path(api.run_folder(case, "steps")).absolute()
    (run / "1D_mode").mkdir(parents=True)
    (run / "1D_mode" / "old.txt").write_text("old run")
    sibling = run.parent / "other_run"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("keep")

    assert main(["steps", "--config", CASE]) == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("Refused: ") and "--overwrite" in captured.err
    assert main(["steps", "--config", CASE, "--dry-run"]) == 2
    assert main(["steps", "--config", CASE, "--dry-run", "--overwrite"]) == 0
    assert f"--overwrite would delete {run}" in capsys.readouterr().out
    assert (run / "1D_mode" / "old.txt").is_file()
    assert main(["steps", "--config", CASE, "--overwrite", "--export-case", "x.json"]) == 2
    assert not (tmp_path / "x.json").exists()

    assert main(["steps", "--config", CASE, "--overwrite"]) == 0
    assert not (run / "1D_mode" / "old.txt").exists()
    assert (run / "1D_mode" / "startup" / "sol.pkl").is_file()
    assert (sibling / "keep.txt").read_text() == "keep"


def test_step_resume_refuses_stale_downstream_reuse(tmp_path):
    """Reused steps must form a leading block of the executed sequence."""
    from pydac.config.loader import write_resolved_config
    from pydac.utils.provenance import start_activity

    config = pydac.load_case(CASE, settings={"execution.cycle_steps": ["ads", "blow", "heat"]})
    results = tmp_path / "saved" / "1D_mode"
    results.mkdir(parents=True)
    write_resolved_config(config, results / "resolved_config.json")
    start_activity(results, "run", config=config, execution={"workflow": "steps"})
    for step in ("startup", "ads", "blow", "heat"):
        (results / step).mkdir()
        (results / step / "sol.pkl").write_bytes(b"saved")

    with pytest.raises(ConfigurationError, match="cannot reuse heat after re-solving blow"):
        resolve_resume(tmp_path / "saved", workflow="steps", settings=[
            "execution.solve=false", 'execution.step_options={"blow": {"solve": true}}'])
    plan = resolve_resume(tmp_path / "saved", workflow="steps", settings=[
        'execution.step_options={"startup": {"solve": false}, "ads": {"solve": false}}'])
    assert "reusing saved startup, ads, solving blow, heat" in plan.summary


# --------------------------------------------------------------------------
# CSS runs: exit outcomes, resume and report
# --------------------------------------------------------------------------

def test_css_exit_codes_and_python_outcome(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    strict = QUIET_CSS + ["execution.max_cycles=2", "execution.css_tolerance=1e-12",
                          "execution.extra_tag=strict"]
    assert main(["css", "--config", CASE, *_set(strict)]) == 3
    captured = capsys.readouterr()
    assert "CSS not reached after cycle 2 (execution.require_css=true)" in captured.err
    assert "(CSS not reached: last completed cycle)" in captured.out
    run = _printed_folders(captured.out).pop()

    # Not reaching CSS is an outcome in Python, not an exception.
    capped = pydac.load_case(CASE, settings=strict)
    result = pydac.run_css(capped, overwrite=True)
    assert result.folder == run and result.workflow == "css"
    assert result.css_reached is False and result.cycle == 2
    assert result.results_folder == run / "1D_mode"
    lenient = strict[:-1] + ["execution.require_css=false", "execution.extra_tag=capped"]
    assert main(["css", "--config", CASE, *_set(lenient)]) == 0
    reached = QUIET_CSS + ["execution.max_cycles=2", "execution.css_tolerance=1e6",
                           "execution.extra_tag=reached"]
    assert main(["css", "--config", CASE, *_set(reached)]) == 0

    # Continue the strict run to cycle 3 in place: still not reached, exit 3 again.
    assert main(["css", "--resume", str(run), "--set", "execution.max_cycles=3"]) == 3
    resumed = pydac.load_case(run / "1D_mode" / "resolved_config.json", exact=True)
    assert resumed.execution.resume_from_cycle == 3
    # A converged run is refused with its own type.
    reached_run = Path(api.run_folder(pydac.load_case(CASE, settings=reached), "css")).absolute()
    with pytest.raises(RunRefused, match="already reached cyclic steady state"):
        pydac.resume(reached_run, settings={"execution.max_cycles": 4})
    with pytest.raises(ConfigurationError, match="workers applies only to NSGA-II"):
        pydac.resume(run, workers=2)


def test_css_report_regenerates_without_touching_the_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    settings = QUIET_CSS + ["execution.max_cycles=2", "execution.css_tolerance=1e6"]
    result = pydac.run_css(pydac.load_case(CASE, settings=settings))
    assert result.css_reached is True and result.cycle == 2
    results = result.results_folder
    assert not (results / "kpis" / "cycle_2_css_kpi_summary.json").exists()
    before = _hashes(results)
    capsys.readouterr()

    # The spatial folder is accepted and normalized to the run folder.
    report = pydac.report(results, fmt="png")
    assert report == result
    after = _hashes(results)
    assert {name: digest for name, digest in after.items() if name in before} == before
    added = set(after) - set(before)
    assert any(name.startswith("kpis/") and name.endswith(".png") for name in added)
    assert "kpis/cycle_2_css_kpi_summary.json" in added

    # Cycle 1 has startup in place of adsorption: profiles and O2 audit, no KPIs.
    assert main(["css", "--report", str(result.folder), "--cycle", "1"]) == 0
    assert "KPIs skipped: cycle 1 has no adsorption step" in capsys.readouterr().out
    assert not (results / "kpis" / "cycle_1_css_kpi_summary.json").exists()
    assert main(["css", "--report", str(result.folder), "--cycle", "7"]) == 2
    assert "available: 1, 2" in capsys.readouterr().err
    for extra in (["--dry-run"], ["--set", "grid.nz=3"], ["--resume", str(result.folder)],
                  ["--overwrite"]):
        assert main(["css", "--report", str(result.folder), *extra]) == 2
    assert main(["nsga2", "--report", str(result.folder)]) == 2
    assert "report it with pydac css --report" in capsys.readouterr().err
    assert main(["css", "--cycle", "2"]) == 2


def test_css_strict_single_cycle_exits_3_with_default_outputs(tmp_path, monkeypatch, capsys):
    """CI smoke 'strict': cycle 1 cannot reach CSS; KPIs on by default."""
    monkeypatch.chdir(tmp_path)
    strict = [f"execution.cycle_steps={CSS_STEPS}", "execution.max_cycles=1",
              "execution.require_css=true", "execution.plot_cyclic_profiles=false"]
    assert pydac.load_case(CASE, settings=strict).execution.generate_kpi_outputs is True
    assert main(["css", "--config", CASE, *_set(strict)]) == 3
    captured = capsys.readouterr()
    assert "KPIs skipped: cycle 1" in captured.out
    assert "CSS not reached after cycle 1" in captured.err


def test_report_refuses_a_step_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = pydac.run_steps(pydac.load_case(CASE))
    with pytest.raises(ConfigurationError, match="step run, which has no report"):
        pydac.report(result.folder)
    with pytest.raises(ConfigurationError, match="is not a saved pyDAC run"):
        pydac.report(tmp_path)


def test_css_reporting_failure_after_solve_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    solved = []

    def fake_solve(case, folder):
        solved.append(folder)
        return api.RunResult(Path(folder).absolute(), Path(folder).absolute() / "1D_mode",
                             "css", css_reached=True, cycle=1), f"{folder}/1D_mode"

    def broken_outputs(*args, **kwargs):
        raise RuntimeError("plotting broke")

    monkeypatch.setattr(api, "_solve_css", fake_solve)
    monkeypatch.setattr(api, "_css_outputs", broken_outputs)
    assert main(["css", "--config", CASE, *_set(QUIET_CSS)]) == 1
    err = capsys.readouterr().err
    assert solved and "Run completed; reporting failed: plotting broke" in err
    assert "pydac css --report" in err


# --------------------------------------------------------------------------
# nsga2
# --------------------------------------------------------------------------

def test_nsga2_cli_names_refuses_and_continues(tmp_path, monkeypatch, capsys):
    from pydac.workflows import orchestrator_NSGA2 as nsga2

    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(nsga2, "run_nsga2_optimization",
                        lambda config, **kwargs: calls.append((config, kwargs)))
    bounds = tmp_path / "bounds.json"
    bounds.write_text(json.dumps(BOUNDS))
    base = ["nsga2", "--config", CASE, *_set(f"{k}={json.dumps(v)}" for k, v in NSGA2_CASE.items()),
            "--bounds", str(bounds), "--pop-size", "4", "--generations", "2"]

    assert main([*base, "--dry-run"]) == 0
    assert not calls and not (tmp_path / "simulation_results").exists()
    assert main([*base, "--workers", "2"]) == 0
    config, options = calls[0]
    campaign = Path(config.base_folder).absolute()
    assert _printed_folders(capsys.readouterr().out, "Campaign folder") == {campaign}
    assert campaign.parent == tmp_path / "simulation_results" / "nsga2"
    assert options["n_procs"] == 2 and (campaign / "resolved_config.json").is_file()

    # The same campaign from Python lands in the same folder.
    case = pydac.load_case(CASE, settings=NSGA2_CASE)
    result = pydac.run_nsga2(case, BOUNDS, pop_size=4, generations=2)
    assert result.folder == campaign and result.results_folder == campaign / "results"
    assert calls[1][0].base_folder == config.base_folder

    # Committed generations: a fresh start is refused, exit 2 on stderr.
    identity = nsga2.nsga2_campaign_identity(
        config, pop_size=4, n_generations=2, seed=1,
        termination_config=options["termination_config"])
    (campaign / "results").mkdir(parents=True)
    (campaign / "results" / "checkpoint_state.json").write_text(json.dumps(
        {"campaign_identity": identity, "completed_generations": 1}))
    assert main(base) == 2
    assert capsys.readouterr().err.startswith("Refused: ")
    with pytest.raises(RunRefused, match="continue it with --resume"):
        pydac.run_nsga2(case, BOUNDS, pop_size=4, generations=2)
    assert len(calls) == 2

    for bad in ({"t_ads": [1.0, 0.0]}, dict(BOUNDS, bogus=[0.0, 1.0])):
        bounds.write_text(json.dumps(bad))
        assert main(base) == 2
    assert "inactive decision variables" in capsys.readouterr().err
    assert main([*base[:-4], "--ftol", "0.1"]) == 2

    # The fake checkpoint fails the real continuation preflight: exit 2, dry run too.
    assert main(["nsga2", "--resume", str(campaign), "--dry-run"]) == 2
    assert "Unsupported checkpoint schema" in capsys.readouterr().err
    resumed = []
    monkeypatch.setattr(nsga2, "preflight_nsga2_resume", lambda folder: json.loads(
        (Path(folder) / "results" / "checkpoint_state.json").read_text()))
    monkeypatch.setattr(nsga2, "resume_nsga2_campaign",
                        lambda folder, **kwargs: resumed.append((folder, kwargs)))
    assert main(["nsga2", "--resume", str(campaign), "--dry-run"]) == 0
    assert "after generation 1 of 2" in capsys.readouterr().out
    assert main(["nsga2", "--resume", str(campaign), "--workers", "3"]) == 0
    assert resumed == [(str(campaign), {"n_procs": 3})]
    for extra in (["--seed", "2"], ["--set", "grid.nz=5"], ["--config", CASE]):
        assert main(["nsga2", "--resume", str(campaign), *extra]) == 2
    assert pydac.resume(campaign, workers=2).workflow == "nsga2"
    with pytest.raises(ConfigurationError, match="settings are not accepted"):
        pydac.resume(campaign, settings={"grid.nz": 5})
    assert main(["nsga2", "--report", str(campaign)]) == 2
    assert "no saved Pareto front" in capsys.readouterr().err
    assert main(["nsga2", "--report", str(campaign / "results")]) == 2
    assert "give the campaign folder" in capsys.readouterr().err


def test_nsga2_campaign_folder_starts_with_the_extra_tag(tmp_path, monkeypatch, capsys):
    """execution.extra_tag prefixes a campaign folder as it prefixes a run folder."""
    from pydac.utils.run import current_grid_tag

    monkeypatch.chdir(tmp_path)
    bounds = tmp_path / "bounds.json"
    bounds.write_text(json.dumps(BOUNDS))
    options = dict(pop_size=4, generations=2)
    tagged = pydac.load_case(CASE, settings={**NSGA2_CASE, "execution.extra_tag": "trial 7"})
    plain = pydac.load_case(CASE, settings=NSGA2_CASE)
    grid = current_grid_tag(plain.simulation.grid)

    folder = Path(api._plan_campaign(
        tagged, bounds, workers=1, seed=1, purity_min=None, oxygen_ppm_max=None,
        eq_work_max=None, steam_condensation=False, termination="fixed_generations",
        ftol=None, period=None, n_skip=None, max_evaluations=None, **options).folder)
    assert folder.parent == Path("simulation_results") / "nsga2"
    assert folder.name.startswith(f"trial_7_{grid}_packed_bed_")
    # The CLI dry run prints the same folder.
    command = ["nsga2", "--config", CASE, *_set(f"{k}={json.dumps(v)}" for k, v in NSGA2_CASE.items()),
               "--set", "execution.extra_tag=trial 7", "--bounds", str(bounds),
               "--pop-size", "4", "--generations", "2", "--dry-run"]
    assert main(command) == 0
    assert _printed_folders(capsys.readouterr().out, "Campaign folder") == {folder.absolute()}
    # Without a tag the leaf starts with the grid; the tag changes the hash (it is part of the case).
    untagged = Path(api._plan_campaign(
        plain, bounds, workers=1, seed=1, purity_min=None, oxygen_ppm_max=None,
        eq_work_max=None, steam_condensation=False, termination="fixed_generations",
        ftol=None, period=None, n_skip=None, max_evaluations=None, **options).folder)
    assert untagged.name.startswith(f"{grid}_packed_bed_")
    assert untagged.name[-8:] != folder.name[-8:]
    assert not (tmp_path / "simulation_results").exists()


def test_extra_tag_is_one_portable_folder_name(tmp_path, monkeypatch, capsys):
    """The tag starts a run/campaign folder name: a path or a name too long for the
    filesystem is refused when the case loads, before any folder (Sol CT-003/004)."""
    monkeypatch.chdir(tmp_path)
    for tag in ("tutorial_05", "trial 7", "v1.2-a", "x" * 64):
        assert pydac.load_case(CASE, settings={"execution.extra_tag": tag}).execution.extra_tag == tag
    for tag in ("../../escape", "a/b", "a\\b", "..", ".hidden", "x" * 65, "café", "a:b"):
        with pytest.raises(ConfigurationError, match="starts the folder name"):
            pydac.load_case(CASE, settings={"execution.extra_tag": tag})
    # The CLI refuses it too (exit 2), with nothing written.
    assert main(["css", "--config", CASE, "--set", "execution.extra_tag=../../escape", "--dry-run"]) == 2
    assert "starts the folder name" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_nsga2_constraint_thresholds_name_the_same_campaign_from_python_and_cli(
        tmp_path, monkeypatch, capsys):
    """The CLI parses thresholds as floats; integers from Python must not move the folder."""
    from pydac.workflows import orchestrator_NSGA2 as nsga2

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nsga2, "run_nsga2_optimization", lambda config, **kwargs: None)
    bounds = tmp_path / "bounds.json"
    bounds.write_text(json.dumps(BOUNDS))
    command = ["nsga2", "--config", CASE, *_set(f"{k}={json.dumps(v)}" for k, v in NSGA2_CASE.items()),
               "--bounds", str(bounds), "--pop-size", "4", "--generations", "2",
               "--purity-min", "0.5", "--oxygen-ppm-max", "1000", "--eq-work-max", "60", "--dry-run"]
    assert main(command) == 0
    (cli_folder,) = _printed_folders(capsys.readouterr().out, "Campaign folder")
    case = pydac.load_case(CASE, settings=NSGA2_CASE)
    result = pydac.run_nsga2(case, BOUNDS, pop_size=4, generations=2,
                             purity_min=0.5, oxygen_ppm_max=1000, eq_work_max=60)
    assert result.folder == cli_folder
    for bad in (True, "60", float("nan")):
        with pytest.raises(ConfigurationError, match="`eq_work_max` must be a finite number"):
            pydac.run_nsga2(case, BOUNDS, pop_size=4, generations=2, eq_work_max=bad)


def test_nsga2_search_space_preflight_refuses_before_writing(tmp_path, monkeypatch, capsys):
    """A startup horizon shorter than the t_ads bounds fails every candidate."""
    monkeypatch.chdir(tmp_path)
    bounds = tmp_path / "bounds.json"
    bounds.write_text(json.dumps(BOUNDS))
    command = ["nsga2", "--config", CASE, "--set", f"execution.cycle_steps={CSS_STEPS}",
               "--bounds", str(bounds), "--pop-size", "4", "--generations", "2"]
    for extra in (["--dry-run"], []):
        assert main([*command, *extra]) == 2
        err = capsys.readouterr().err
        assert "Candidates at the lower bounds (t_ads=900" in err
        assert "step `startup`: `event_max_time_s` cannot be shorter than nominal time" in err
    assert not (tmp_path / "simulation_results").exists()
    # Only the upper corner exceeds the startup limit: refused too.
    within = pydac.load_case(CASE, settings={**NSGA2_CASE,
                                             "steps.startup.horizon.event_max_time_s": 1200})
    with pytest.raises(ConfigurationError, match="upper bounds"):
        pydac.run_nsga2(within, BOUNDS, pop_size=4, generations=2)


def test_nsga2_first_generation_all_failed_stops(tmp_path, monkeypatch):
    """Every candidate of generation 1 fails: the campaign stops, nothing is committed."""
    from pydac.utils.nsga2.logging import AllCandidatesFailed
    from pydac.workflows import orchestrator_NSGA2 as nsga2

    monkeypatch.chdir(tmp_path)

    def broken_solve(**kwargs):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(nsga2, "run_css_simulation", broken_solve)
    case = pydac.load_case(CASE, settings=NSGA2_CASE)
    with pytest.raises(AllCandidatesFailed) as error:
        pydac.run_nsga2(case, BOUNDS, pop_size=4, generations=3)
    message = str(error.value)
    assert "Every candidate of generation 1 failed (4 of 4)" in message
    assert "Candidates evaluated 4, failed 4." in message
    assert "4 x RuntimeError: solver exploded" in message

    campaign = next((tmp_path / "simulation_results" / "nsga2").iterdir())
    assert not (campaign / "results" / "checkpoint_state.json").exists()
    assert (campaign / "results" / "candidate_records" / "generation_0001.jsonl").is_file()
    # Nothing committed: the same campaign may start again once the cause is fixed.
    run_config = nsga2.NSGA2RunConfig(base_folder=str(campaign), bounds=BOUNDS, experiment=case)
    identity = nsga2.nsga2_campaign_identity(
        run_config, pop_size=4, n_generations=3, seed=1,
        termination_config=nsga2.NSGA2TerminationConfig(mode="fixed_generations"))
    nsga2.guard_campaign_folder(campaign, identity, continuation=False)


def test_campaign_evaluation_summary_counts_and_groups_errors(tmp_path):
    from pydac.utils.nsga2.evaluation import (
        campaign_evaluation_summary,
        format_evaluation_summary,
    )

    folder = tmp_path / "results" / "candidate_records"
    folder.mkdir(parents=True)
    records = [{"failure": None}, {"failure": {"error": "A: x"}}, {"failure": {"error": "B: y"}},
               {"failure": {"error": "A: x"}}]
    (folder / "generation_0001.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records[:2]) + "\n")
    (folder / "generation_0002.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records[2:]) + "\n")
    summary = campaign_evaluation_summary(tmp_path)
    assert summary == {"evaluated": 4, "failed": 3, "errors": {"A: x": 2, "B: y": 1}}
    assert format_evaluation_summary(summary, limit=1) == (
        "Candidates evaluated 4, failed 3.\n  2 x A: x\n  ... and 1 other distinct errors")
    assert campaign_evaluation_summary(tmp_path / "empty")["evaluated"] == 0


# --------------------------------------------------------------------------
# Checkout launchers: runners pass their constants; pydac never reads them
# --------------------------------------------------------------------------

def test_runner_constants_reach_pydac_explicitly(tmp_path, monkeypatch, capsys):
    from runners import run_steps_simulation as runner

    monkeypatch.chdir(tmp_path)
    # Precedence: defaults < CONFIG_PATH < DEVELOPMENT_SETTINGS < --set.
    monkeypatch.setattr(runner, "CONFIG_PATH", CASE)
    monkeypatch.setattr(runner, "DEVELOPMENT_SETTINGS", {"grid": {"nz": 7}, "execution.extra_tag": "dev"})
    assert runner.main(["--print-resolved"]) == 0
    document = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert document["simulation"]["grid"]["nz"] == 7 and document["execution"]["extra_tag"] == "dev"
    assert runner.main(["--print-resolved", "--set", "grid.nz=9"]) == 0
    assert json.loads(capsys.readouterr().out.split("\n", 1)[1])["simulation"]["grid"]["nz"] == 9

    # A fresh run refuses its occupied folder unless OVERWRITE (= --overwrite) is set.
    assert runner.main([]) == 0
    run = _printed_folders(capsys.readouterr().out).pop()
    assert run.name.startswith("dev_packed_bed_")
    assert runner.main([]) == 2
    assert capsys.readouterr().err.startswith("Refused: ")
    monkeypatch.setattr(runner, "OVERWRITE", True)
    assert runner.main([]) == 0
    assert f"Overwrite: deleted {run}" in capsys.readouterr().out

    # CASE_PATH runs a frozen case exactly; it cannot be combined with --case, and
    # non-empty DEVELOPMENT_SETTINGS are refused like --set.
    case_path = tmp_path / "case.json"
    pydac.export_case(pydac.load_case(CASE), case_path)
    monkeypatch.setattr(runner, "CASE_PATH", str(case_path))
    monkeypatch.setattr(runner, "CONFIG_PATH", None)
    assert runner.main(["--dry-run"]) == 2
    assert "does not accept the runner's DEVELOPMENT_SETTINGS" in capsys.readouterr().err
    monkeypatch.setattr(runner, "DEVELOPMENT_SETTINGS", {})
    assert runner.main(["--dry-run"]) == 0
    assert f"Mode: frozen case {case_path}" in capsys.readouterr().out
    assert runner.main(["--case", str(case_path), "--dry-run"]) == 2
    assert "CASE_PATH cannot be combined with --case" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Environment warning
# --------------------------------------------------------------------------

def test_environment_warning_outside_a_dedicated_environment(tmp_path, monkeypatch, capsys):
    from pydac.cli import environment_warning

    system = tmp_path / "usr"
    system.mkdir()
    assert "outside a dedicated environment" in environment_warning(system, system, {})
    assert environment_warning(system, system, {"PYDAC_NO_ENV_WARNING": "1"}) is None
    assert environment_warning(tmp_path / "venv", system, {}) is None  # venv
    conda_env = tmp_path / "miniconda" / "envs" / "pydac"
    (conda_env / "conda-meta").mkdir(parents=True)
    assert environment_warning(conda_env, conda_env, {}) is None
    conda_base = tmp_path / "miniconda"
    (conda_base / "condabin").mkdir()
    (conda_base / "conda-meta").mkdir()
    assert "outside a dedicated environment" in environment_warning(conda_base, conda_base, {})

    # The command warns on stderr and still runs; silenced by the variable.
    monkeypatch.setattr(sys, "base_prefix", sys.prefix)
    monkeypatch.delenv("PYDAC_NO_ENV_WARNING", raising=False)
    if (Path(sys.prefix) / "conda-meta").is_dir():
        pytest.skip("the test interpreter is a conda environment")
    assert main(["validate", "--config", CASE]) == 0
    assert capsys.readouterr().err.startswith("Warning: pydac is running outside")
    monkeypatch.setenv("PYDAC_NO_ENV_WARNING", "1")
    assert main(["validate", "--config", CASE]) == 0
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------
# Validation catches the wall-channeling grid refusal and frozen-case export refusal before solving.
# --------------------------------------------------------------------------

def test_validate_refuses_a_1d_channeling_case_like_the_model_build(capsys):
    channeling = 'bed.design={"eps_b":0.4,"flow_model":"wall-channeling-reduced"}'
    assert main(["validate", "--config", CASE, "--set", "grid.nr=1", "--set", channeling,
                 "--validate"]) == 2
    assert "needs Nr >= 2" in capsys.readouterr().err


def test_export_case_refusal_names_the_overwrite_flag(tmp_path, capsys):
    target = tmp_path / "case.json"
    assert main(["validate", "--config", CASE, "--export-case", str(target)]) == 0
    assert main(["validate", "--config", CASE, "--export-case", str(target)]) == 2
    assert "--overwrite-case" in capsys.readouterr().err
    assert main(["validate", "--config", CASE, "--export-case", str(target),
                 "--overwrite-case"]) == 0


def test_css_refuses_run_wide_step_options_it_would_ignore(capsys):
    """CSS step options live per cycle, in execution.cycle_options."""
    run_wide = 'execution.step_options={"cool":{"plot_profiles":true}}'
    assert main(["css", "--config", CASE, "--set", run_wide, "--dry-run"]) == 2
    assert "execution.cycle_options" in capsys.readouterr().err
    per_cycle = 'execution.cycle_options={"2":{"step_options":{"cool":{"plot_profiles":true}}}}'
    assert main(["css", "--config", CASE, "--set", 'execution.cycle_steps=["ads","blow","heat","s_des","cool","press"]',
                 "--set", per_cycle, "--dry-run"]) == 0


def test_diagnostic_solver_modes_warn_before_the_run(tmp_path, monkeypatch):
    """The diagnostic solver modes are valid but never silent."""
    import warnings

    from pydac import cli

    monkeypatch.chdir(tmp_path)
    with pytest.warns(UserWarning) as record:
        assert main(["steps", "--config", CASE, "--set",
                     "execution.compute_numerical_sparsity=true", "--set",
                     "execution.cvode_sparsity_mode=full", "--dry-run"]) == 0
    messages = [str(item.message) for item in record]
    assert any("`execution.compute_numerical_sparsity=true` is a slow" in m for m in messages)
    assert any("`execution.cvode_sparsity_mode=full` is intended only" in m for m in messages)
    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        assert main(["steps", "--config", CASE, "--set", "execution.use_sparsity=true",
                     "--dry-run"]) == 0
    assert not [item for item in quiet if "sparsity" in str(item.message)]
    with pytest.warns(UserWarning, match="use_sparsity=false"):
        api._preflight(api._validator(api.STEPS),
                       pydac.load_case(CASE, settings={"execution.use_sparsity": False}))
    # In the terminal, pyDAC's own warnings are one plain line.
    assert cli._format_warning("slow", UserWarning, api.__file__, 1) == "Warning: slow\n"


def test_pydac_leaves_the_matplotlib_backend_to_the_user():
    """Importing pyDAC's plotting must not switch a notebook to Agg: plt.show() would show nothing."""
    modules = ["pydac.post_processing.kpi.plotting", "pydac.post_processing.profiles.step_plotting",
               "pydac.post_processing.profiles.cyclic_plotting",
               "pydac.post_processing.optimization.nsga2_post_processing"]
    code = ("import importlib, matplotlib\n"
            + "".join(f"importlib.import_module({name!r})\n" for name in modules)
            + "print(matplotlib.get_backend())")
    done = subprocess.run([sys.executable, "-c", code], env={**os.environ, "MPLBACKEND": "svg"},
                          capture_output=True, text=True, check=True)
    assert done.stdout.strip().lower() == "svg"


def test_a_run_keeps_notebook_figures_showing(tmp_path):
    """matplotlib-inline turns interactive mode on when its backend is first loaded. If a run
    draws the session's first figure, that load must not happen inside pyDAC's ioff(), or ioff
    undoes it on exit and no later notebook figure shows. A stand-in backend does the same."""
    (tmp_path / "inline_like_backend.py").write_text(
        "import matplotlib\n"
        "from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas\n"
        "matplotlib.interactive(True)\n")
    code = ("import matplotlib, matplotlib.pyplot as plt\n"
            "from pydac.api import _figures_stay_on_disk\n"
            "_figures_stay_on_disk(lambda: plt.close(plt.figure()))()\n"
            "print(matplotlib.is_interactive())")
    env = {**os.environ, "MPLBACKEND": "module://inline_like_backend",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(tmp_path), os.environ.get("PYTHONPATH")]))}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          check=True)
    assert done.stdout.strip() == "True"


def test_cli_draws_on_agg_unless_the_user_chose_a_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MPLBACKEND", raising=False)
    assert main(["validate", "--config", CASE]) == 0
    assert os.environ["MPLBACKEND"] == "Agg"
    monkeypatch.setenv("MPLBACKEND", "svg")
    assert main(["validate", "--config", CASE]) == 0
    assert os.environ["MPLBACKEND"] == "svg"


@pytest.mark.parametrize("nr", [1, 3])
def test_a_run_leaves_no_figure_open(tmp_path, monkeypatch, nr):
    """A notebook shows every figure still open at the end of a cell: pyDAC saves its
    figures and must leave none open, or a run would flood the notebook."""
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    plt.close("all")
    case = pydac.load_case(CASE, settings={"grid.nr": nr, "execution.plot_profiles": True,
                                           "execution.plot_format": "png"})
    result = pydac.run_steps(case)
    assert list(result.results_folder.rglob("*.png"))
    assert plt.get_fignums() == []


PLOTTED = {"grid.nr": 3, "execution.plot_profiles": True, "execution.plot_format": "png"}
HEAT_MAP = {"T_bed": {"times": [1, 10, 30, 60, 90, 150], "unit": "min"}}


def test_snapshot_times_are_checked_before_a_run():
    plotted = pydac.load_case(CASE, settings=PLOTTED)
    checked = api._snapshot_times(plotted, api.STEPS, {"Startup": {"T_bed": {"times": [0, 2]}}})
    assert checked == {"startup": {"T_bed": {"times": (0.0, 2.0), "unit": "s"}}}
    assert api._snapshot_times(plotted, api.STEPS, None) is None
    bad = [
        ({}, "must be a mapping"),
        ({"heat": HEAT_MAP}, "is not in this run"),
        ({"startup": {"T_wall": {"times": [1]}}}, "no step plot has this field"),
        ({"startup": {"T_bed": [1, 2]}}, '{"times": [...], "unit": "min"}'),
        ({"startup": {"T_bed": {"times": [1], "unit": "minutes"}}}, "unit must be one of"),
        ({"startup": {"T_bed": {"times": list(range(7))}}}, "1 to 6 numbers"),
        ({"startup": {"T_bed": {"times": [-1]}}}, "1 to 6 numbers"),
        ({"startup": {"T_bed": {"times": [True]}}}, "1 to 6 numbers"),
        # Too large for a float, or infinite once converted to seconds: refused, not crashed.
        ({"startup": {"T_bed": {"times": [10**400]}}}, "1 to 6 numbers"),
        ({"startup": {"T_bed": {"times": [1e308], "unit": "h"}}}, "1 to 6 numbers"),
        ({"startup": {"T_bed": {"times": [float("nan")]}}}, "1 to 6 numbers"),
        ({"startup": {"T_bed": {"times": [5, 1, 5]}}}, "a time is repeated"),
        ({"startup": {"T_bed": {"times": [1]}}, "STARTUP": {"T_bed": {"times": [2]}}},
         "given twice"),
    ]
    for value, message in bad:
        with pytest.raises(ConfigurationError) as caught:
            api._snapshot_times(plotted, api.STEPS, value)
        assert message in str(caught.value), (value, str(caught.value))
    # Without step plots the times would change nothing: refused, not ignored.
    quiet = pydac.load_case(CASE)
    with pytest.raises(ConfigurationError, match="draws none"):
        api._snapshot_times(quiet, api.STEPS, {"startup": {"T_bed": {"times": [1]}}})
    # A CSS run checks the steps of its cycle.
    cyclic = pydac.load_case(CASE, settings={**PLOTTED, "execution.plot_profiles": False,
                                             "execution.plot_final_cycle_profiles": True,
                                             "execution.cycle_steps": None})
    assert "heat" in api._snapshot_times(cyclic, api.CSS, {"heat": HEAT_MAP})
    # ... but the final-cycle replay never draws startup (it runs in cycle 1 only).
    with pytest.raises(ConfigurationError, match="draws no plots of step 'startup'"):
        api._snapshot_times(cyclic, api.CSS, {"startup": {"T_bed": {"times": [1]}}})


def test_snapshot_times_follow_the_plot_switches_of_each_step():
    """A step gets map times only if the run draws its plots: step_options override
    plot_profiles; a CSS run also counts the plot switches of cycle_options."""
    cases = [
        # (workflow, settings, steps drawn)
        (api.STEPS, {"execution.step_options": {"startup": {"plot_profiles": False}}}, []),
        (api.STEPS, {"execution.plot_profiles": False,
                     "execution.step_options": {"startup": {"plot_profiles": True}}}, ["startup"]),
        (api.CSS, {"execution.plot_profiles": False, "execution.cycle_steps": None,
                   "execution.cycle_options": {"2": {"plot_profiles": True}}},
         ["ads", "blow", "heat", "s_des", "cool", "press"]),
        (api.CSS, {"execution.plot_profiles": False, "execution.cycle_steps": None,
                   "execution.cycle_options": {"1": {"step_options": {"heat": {"plot_profiles": True}}}}},
         ["heat"]),
    ]
    for workflow, settings, drawn in cases:
        case = pydac.load_case(CASE, settings={**PLOTTED, **settings})
        assert api._drawn_steps(case, workflow) == drawn, settings
    # Built-in sequence: step_options turn on one step, the run checks the others too.
    steps_run = pydac.load_case(CASE, settings={**PLOTTED, "execution.plot_profiles": False,
                                                "execution.cycle_steps": ["startup", "heat"],
                                                "execution.step_options": {"heat": {"plot_profiles": True}}})
    assert "heat" in api._snapshot_times(steps_run, api.STEPS, {"heat": HEAT_MAP})
    with pytest.raises(ConfigurationError, match="draws no plots of step 'startup'"):
        api._snapshot_times(steps_run, api.STEPS, {"startup": {"T_bed": {"times": [1]}}})
    # Plots turned on only through cycle_options are accepted (refused before ST-002).
    css_run = pydac.load_case(CASE, settings={**PLOTTED, "execution.plot_profiles": False,
                                              "execution.cycle_steps": None,
                                              "execution.cycle_options": {"2": {"plot_profiles": True}}})
    assert "heat" in api._snapshot_times(css_run, api.CSS, {"heat": HEAT_MAP})


def test_a_run_draws_its_maps_at_the_snapshot_times(tmp_path, monkeypatch):
    """The times reach the 2D plots of the run, and pyDAC's own choice comes back after it."""
    from pydac.post_processing.profiles.step_plotting import ProfilesPlotter
    from pydac.utils.plot_helpers import SNAPSHOT_TIME_OVERRIDES

    monkeypatch.chdir(tmp_path)
    before = json.dumps(SNAPSHOT_TIME_OVERRIDES, sort_keys=True)
    seen = []
    schedule = ProfilesPlotter._get_snapshot_schedule

    def spy(self, field_name, n_total, snapshot_times=None, time_unit=None, **options):
        if field_name == "T_bed":
            seen.append(SNAPSHOT_TIME_OVERRIDES.get(self.pb.step, {}).get("T_bed"))
        return schedule(self, field_name, n_total, snapshot_times, time_unit, **options)

    monkeypatch.setattr(ProfilesPlotter, "_get_snapshot_schedule", spy)
    times = {"startup": {"T_bed": {"times": [0, 1, 2], "unit": "s"}}}
    pydac.run_steps(pydac.load_case(CASE, settings=PLOTTED), snapshot_times=times)
    assert seen and all(entry == {"times": (0.0, 1.0, 2.0), "unit": "s"} for entry in seen)
    assert json.dumps(SNAPSHOT_TIME_OVERRIDES, sort_keys=True) == before

    # Restored also when the run fails.
    def fail(case, folder):
        raise RuntimeError("solver stopped")

    monkeypatch.setattr(api, "_solve_steps", fail)
    with pytest.raises(RuntimeError):
        pydac.run_steps(pydac.load_case(CASE, settings=PLOTTED), snapshot_times=times,
                        overwrite=True)
    assert json.dumps(SNAPSHOT_TIME_OVERRIDES, sort_keys=True) == before


@pytest.mark.parametrize("nr", [1, 3])
def test_snapshot_times_reach_the_1d_profiles_too(tmp_path, monkeypatch, nr):
    """A field's times are the same in its 2D map and in its 1D profile panels; a panel
    without times (here u_z, and y_CO2 of this run) keeps the evenly spaced default."""
    from pydac.post_processing.profiles.step_plotting import ProfilesPlotter

    monkeypatch.chdir(tmp_path)
    drawn = {}
    profile = ProfilesPlotter._profile_schedule
    schedule = ProfilesPlotter._get_snapshot_schedule

    def profile_spy(self, field_name, n_times=10):
        pairs = profile(self, field_name, n_times)
        drawn.setdefault((self.pb.step, "profile", field_name), [label for _, label in pairs])
        return pairs

    def map_spy(self, field_name, n_total, snapshot_times=None, time_unit=None, **options):
        indices, labels, tag = schedule(self, field_name, n_total, snapshot_times, time_unit, **options)
        if options.get("report", True):
            drawn.setdefault((self.pb.step, "map", field_name), list(labels))
        return indices, labels, tag

    monkeypatch.setattr(ProfilesPlotter, "_profile_schedule", profile_spy)
    monkeypatch.setattr(ProfilesPlotter, "_get_snapshot_schedule", map_spy)
    times = {"startup": {"T_bed": {"times": [0, 1, 2], "unit": "s"}}}
    pydac.run_steps(pydac.load_case(CASE, settings=PLOTTED | {"grid.nr": nr}), snapshot_times=times)

    expected = ["t = 0.0 s", "t = 1.0 s", "t = 2.0 s"]
    assert drawn[("startup", "map", "T_bed")] == expected
    assert drawn[("startup", "profile", "T_bed")] == expected
    # Evenly spaced default (up to 10 saved times; this short run saves 5).
    default = drawn[("startup", "profile", None)]
    assert drawn[("startup", "profile", "y_CO2")] == default != expected


def test_snapshot_times_on_a_single_saved_state_give_one_panel():
    """Times that all fall on the only saved state draw one panel, in the map and in the profiles."""
    from types import SimpleNamespace

    import numpy as np

    from pydac.post_processing.profiles.step_plotting import ProfilesPlotter
    from pydac.utils.plot_helpers import SNAPSHOT_TIME_OVERRIDES

    plotter = ProfilesPlotter.__new__(ProfilesPlotter)
    plotter.pp = SimpleNamespace(time=np.array([0.0]), Nt=1, pb=SimpleNamespace(step="heat"))
    saved = SNAPSHOT_TIME_OVERRIDES.get("heat")
    SNAPSHOT_TIME_OVERRIDES["heat"] = {"T_bed": {"times": (0.0, 1.0, 2.0), "unit": "s"}}
    try:
        indices, labels, _ = ProfilesPlotter._get_snapshot_schedule(plotter, "T_bed", 1)
        assert indices == [0] and len(labels) == 1
        assert ProfilesPlotter._profile_schedule(plotter, "T_bed") == [(0, labels[0])]
    finally:
        if saved is None:
            SNAPSHOT_TIME_OVERRIDES.pop("heat", None)
        else:
            SNAPSHOT_TIME_OVERRIDES["heat"] = saved


def test_cli_snapshot_times(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    plotted = [arg for path, value in PLOTTED.items() for arg in ("--set", f"{path}={value}".lower())]
    times = json.dumps({"startup": {"T_bed": {"times": [0, 1, 2]}}})
    assert main(["steps", "--config", CASE, *plotted, "--snapshot-times", times, "--dry-run"]) == 0
    assert main(["steps", "--config", CASE, *plotted, "--snapshot-times", "{heat", "--dry-run"]) == 2
    assert "--snapshot-times is not valid JSON" in capsys.readouterr().err
    huge = '{"startup": {"T_bed": {"times": [1' + "0" * 400 + "]}}}"
    assert main(["steps", "--config", CASE, *plotted, "--snapshot-times", huge, "--dry-run"]) == 2
    assert "1 to 6 numbers" in capsys.readouterr().err
    assert main(["steps", "--config", CASE, "--snapshot-times", times, "--dry-run"]) == 2
    assert "draws none" in capsys.readouterr().err
    assert main(["steps", "--config", CASE, *plotted, "--snapshot-times", times,
                 "--export-case", "x.json"]) == 2
    assert not (tmp_path / "x.json").exists()
    assert main(["steps", "--resume", "RUN", "--snapshot-times", times]) == 2
    assert "remove --snapshot-times" in capsys.readouterr().err


def test_report_draws_the_step_plots_again_at_snapshot_times(tmp_path, monkeypatch, capsys):
    """report(snapshot_times=...) replays the saved cycle (no solve) and draws its step plots
    with the given times; it refuses before writing when the saved states are gone."""
    from pydac.post_processing.profiles.step_plotting import ProfilesPlotter
    from pydac.utils.plot_helpers import SNAPSHOT_TIME_OVERRIDES

    monkeypatch.chdir(tmp_path)
    settings = QUIET_CSS + ["execution.max_cycles=2", "execution.css_tolerance=1e6", "grid.nr=3"]
    result = pydac.run_css(pydac.load_case(CASE, settings=settings))
    assert result.cycle == 2
    maps = result.results_folder / "cycle_2" / "heat" / "profiles" / "internal_2D_profiles"
    assert not list(maps.glob("heat_Tbed_2D_snapshots.*"))
    before = json.dumps(SNAPSHOT_TIME_OVERRIDES, sort_keys=True)
    seen = []
    schedule = ProfilesPlotter._get_snapshot_schedule

    def spy(self, field_name, n_total, snapshot_times=None, time_unit=None, **options):
        if field_name == "T_bed":
            seen.append((self.pb.step, SNAPSHOT_TIME_OVERRIDES.get(self.pb.step, {}).get("T_bed")))
        return schedule(self, field_name, n_total, snapshot_times, time_unit, **options)

    monkeypatch.setattr(ProfilesPlotter, "_get_snapshot_schedule", spy)

    def run_files():
        """Every file of the run outside the plot folders, with its content hash."""
        return {path.relative_to(result.results_folder): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in result.results_folder.rglob("*")
                if path.is_file() and "profiles" not in path.parts
                and path.parts[len(result.results_folder.parts)].startswith("cycle_")}

    files_before = run_files()
    times = {"heat": {"T_bed": {"times": [0, 1, 2]}}}
    assert pydac.report(result.folder, snapshot_times=times, fmt="png") == result
    assert ("heat", {"times": (0.0, 1.0, 2.0), "unit": "s"}) in seen
    assert (maps / "heat_Tbed_2D_snapshots.png").is_file()
    assert json.dumps(SNAPSHOT_TIME_OVERRIDES, sort_keys=True) == before
    # Only plots: step logs, step metrics and their provenance stay as the run wrote them.
    assert run_files() == files_before

    with pytest.raises(ConfigurationError, match="is not in this run"):
        pydac.report(result.folder, snapshot_times={"startup": {"T_bed": {"times": [1]}}})
    # The replay starts from the final state of cycle 1: without it, refused before writing.
    (result.results_folder / "cycle_1" / "cool" / "sol.pkl").unlink()
    capsys.readouterr()
    assert main(["css", "--report", str(result.folder), "--snapshot-times", json.dumps(times)]) == 2
    assert "cannot be drawn again" in capsys.readouterr().err
