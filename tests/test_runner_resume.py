"""`--resume <folder>`: continue or replay a saved step/CSS run in place."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pydac.config import load_experiment
from pydac.config.errors import ConfigurationError, RunRefused
from pydac.config.loader import write_resolved_config
from pydac.utils.css.profiles import extract_final_fields
from pydac.utils import provenance
from pydac.utils.provenance import start_activity
from pydac.utils.run.resume import resolve_resume


CASE = "tests/cases/v3_smoke_startup.json"
CSS_STEPS = '["ads","blow","heat","s_des","press","cool"]'
CSS_SETTINGS = [
    f"execution.cycle_steps={CSS_STEPS}",
    "execution.css_tolerance=1e-12",
    # Capped runs: CSS is not the object here (a required CSS not reached exits 3).
    "execution.require_css=false",
    "execution.termination_mode=time",
    "execution.plot_profiles=false",
    "execution.plot_final_cycle_profiles=false",
    "execution.plot_cyclic_profiles=false",
    "execution.generate_kpi_outputs=false",
    "execution.print_to_terminal=false",
    "execution.print_cycle_build=false",
]


def _set_args(settings):
    return [token for setting in settings for token in ("--set", setting)]


def _only_run_folder(root: Path) -> Path:
    folders = [path for path in root.glob("*/*") if path.is_dir()]
    assert len(folders) == 1, folders
    return folders[0]


def _saved_run(tmp_path, workflow, *, completed=(), reached=False, **settings):
    """A saved run folder without solving: resolved config, provenance, cycles."""
    if workflow == "css":
        settings.setdefault("execution.cycle_steps", json.loads(CSS_STEPS))
    config = load_experiment(CASE, settings=settings or None)
    base = tmp_path / "saved"
    results = base / "1D_mode"
    results.mkdir(parents=True)
    write_resolved_config(config, results / "resolved_config.json")
    start_activity(results, "run", config=config, execution={"workflow": workflow})
    for cycle in completed:
        folder = results / f"cycle_{cycle}"
        folder.mkdir()
        (folder / "cycle_metadata.json").write_text("{}", encoding="utf-8")
        (folder / "css_diagnostics.json").write_text(
            json.dumps({"reached": reached and cycle == completed[-1]}), encoding="utf-8"
        )
    return base


# --------------------------------------------------------------------------
# Resolution rules (no solve)
# --------------------------------------------------------------------------

def test_css_resume_continues_after_last_completed_cycle(tmp_path):
    base = _saved_run(tmp_path, "css", completed=(1, 2), **{"execution.max_cycles": 5})
    (base / "1D_mode" / "cycle_3").mkdir()  # interrupted: no cycle_metadata.json

    plan = resolve_resume(base, workflow="css")

    assert plan.start_cycle == 3
    assert plan.config.execution.resume_from_cycle == 3
    assert Path(plan.base_folder) == base
    # The spatial folder itself is accepted too.
    assert resolve_resume(base / "1D_mode", workflow="css").start_cycle == 3


def test_css_resume_may_change_only_operational_settings(tmp_path):
    base = _saved_run(tmp_path, "css", completed=(1,), **{"execution.max_cycles": 2})

    plan = resolve_resume(base, workflow="css", settings=["execution.max_cycles=4",
                                                          "execution.css_tolerance=0.01"])
    assert plan.config.execution.max_cycles == 4
    assert plan.config.execution.css_tolerance == 0.01

    for setting in ("grid.nz=6", "execution.resume_from_cycle=1", "execution.solve=false",
                    "execution.output_root=elsewhere"):
        with pytest.raises(ConfigurationError, match="cannot change"):
            resolve_resume(base, workflow="css", settings=[setting])


def test_css_resume_refuses_finished_runs(tmp_path):
    converged = _saved_run(tmp_path / "a", "css", completed=(1, 2), reached=True)
    with pytest.raises(ConfigurationError, match="already reached cyclic steady state at cycle 2"):
        resolve_resume(converged, workflow="css")

    capped = _saved_run(tmp_path / "b", "css", completed=(1, 2), **{"execution.max_cycles": 2})
    with pytest.raises(ConfigurationError, match="max_cycles=2"):
        resolve_resume(capped, workflow="css")
    assert resolve_resume(capped, workflow="css",
                          settings=["execution.max_cycles=3"]).start_cycle == 3

    empty = _saved_run(tmp_path / "c", "css")
    with pytest.raises(ConfigurationError, match="no completed cycle"):
        resolve_resume(empty, workflow="css")


def test_resume_refuses_other_workflow_and_non_run_folders(tmp_path):
    steps = _saved_run(tmp_path / "a", "steps")
    with pytest.raises(ConfigurationError, match="holds a steps run"):
        resolve_resume(steps, workflow="css")
    with pytest.raises(ConfigurationError, match="does not exist"):
        resolve_resume(tmp_path / "missing", workflow="css")
    (tmp_path / "plain").mkdir()
    with pytest.raises(ConfigurationError, match="not a saved pyDAC run"):
        resolve_resume(tmp_path / "plain", workflow="steps")


def test_step_resume_must_reuse_saved_steps(tmp_path):
    base = _saved_run(tmp_path, "steps")
    with pytest.raises(ConfigurationError, match="reuses its saved steps"):
        resolve_resume(base, workflow="steps")
    # Nothing saved yet: reuse is refused before any step is solved again.
    with pytest.raises(ConfigurationError, match="cannot reuse startup: no saved solution"):
        resolve_resume(base, workflow="steps", settings=["execution.solve=false"])

    (base / "1D_mode" / "startup").mkdir()
    (base / "1D_mode" / "startup" / "sol.pkl").write_bytes(b"saved")
    replay = resolve_resume(base, workflow="steps", settings=["execution.solve=false"])
    assert replay.config.execution.solve is False
    partial = resolve_resume(
        base, workflow="steps", settings=['execution.step_options={"startup": {"solve": false}}']
    )
    assert "replaying all saved steps" in partial.summary


def test_step_resume_ignores_no_option_for_a_step_that_is_not_executed(tmp_path):
    """Reusing a step the run never executes must not count."""
    base = _saved_run(tmp_path, "steps")  # the smoke case executes startup only
    (base / "1D_mode" / "startup").mkdir()
    (base / "1D_mode" / "startup" / "sol.pkl").write_bytes(b"saved")
    with pytest.raises(ConfigurationError, match="does not execute: ads"):
        resolve_resume(base, workflow="steps",
                       settings=['execution.step_options={"ads": {"solve": false}}'])


def test_continuation_needs_the_pydac_version_that_wrote_the_run(tmp_path, monkeypatch):
    """Another pyDAC version may replay or report a run, never continue it."""
    css = _saved_run(tmp_path / "css", "css", completed=(1,), **{"execution.max_cycles": 3})
    steps = _saved_run(tmp_path / "steps", "steps", **{"execution.cycle_steps": ["ads", "blow"]})
    for step in ("startup", "ads", "blow"):  # a step run starts from startup
        (steps / "1D_mode" / step).mkdir()
        (steps / "1D_mode" / step / "sol.pkl").write_bytes(b"saved")
    # Reused steps form a leading block: keep startup, solve ads and blow again.
    reuse_startup = ['execution.step_options={"startup": {"solve": false}}']
    written_by = provenance.__version__
    monkeypatch.setattr(provenance, "__version__", "3.0.0.dev99")

    with pytest.raises(RunRefused, match=f"written by pyDAC {written_by}; this is pyDAC 3.0.0.dev99"):
        resolve_resume(css, workflow="css")
    with pytest.raises(RunRefused, match=r"replaying all its steps \(execution.solve=false\)"):
        resolve_resume(steps, workflow="steps", settings=reuse_startup)
    replay = resolve_resume(steps, workflow="steps", settings=["execution.solve=false"])
    assert "replaying all saved steps" in replay.summary

    # Runs saved before provenance recorded the software are not refused.
    shutil.rmtree(css / "1D_mode" / "provenance")
    assert resolve_resume(css, workflow="css").start_cycle == 2
    # An unreadable record is reported, never skipped.
    (steps / "1D_mode" / "provenance" / "broken.json").write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unreadable provenance record"):
        resolve_resume(steps, workflow="steps", settings=reuse_startup)

    monkeypatch.setattr(provenance, "__version__", written_by)
    (steps / "1D_mode" / "provenance" / "broken.json").unlink()
    assert "reusing saved startup, solving ads, blow" in resolve_resume(
        steps, workflow="steps", settings=reuse_startup).summary


def test_runner_resume_rejects_config_and_dry_run_writes_nothing(tmp_path, capsys):
    from runners import run_css_simulation as runner

    base = _saved_run(tmp_path, "css", completed=(1,), **{"execution.max_cycles": 3})
    before = sorted(path.relative_to(base) for path in base.rglob("*"))

    assert runner.main(["--resume", str(base), "--dry-run"]) == 0
    assert "Resuming CSS run" in capsys.readouterr().out
    assert sorted(path.relative_to(base) for path in base.rglob("*")) == before

    assert runner.main(["--resume", str(base), "--config", CASE, "--dry-run"]) == 2
    assert "remove --config" in capsys.readouterr().err


# --------------------------------------------------------------------------
# End to end: resumed trajectories are the uninterrupted ones
# --------------------------------------------------------------------------

def _terminal_fields(results: Path, cycle: int):
    return extract_final_fields(str(results / f"cycle_{cycle}" / "cool" / "sol.pkl"))


def test_resumed_css_run_matches_uninterrupted_run(monkeypatch, tmp_path):
    from runners import run_css_simulation as runner

    # The test states its case; a developer's runner settings must not apply.
    monkeypatch.setattr(runner, "DEVELOPMENT_SETTINGS", {})

    continuous_root = tmp_path / "continuous"
    assert runner.main(["--config", CASE, *_set_args(CSS_SETTINGS + [
        "execution.max_cycles=3", f"execution.output_root={continuous_root}"])]) == 0

    interrupted_root = tmp_path / "interrupted"
    assert runner.main(["--config", CASE, *_set_args(CSS_SETTINGS + [
        "execution.max_cycles=2", f"execution.output_root={interrupted_root}"])]) == 0
    base = _only_run_folder(interrupted_root)
    results = base / "1D_mode"
    first_cycle = (results / "cycle_1" / "cycle_metadata.json").read_bytes()
    first_log = (results / "log.txt").read_text(encoding="utf-8")

    assert runner.main(["--resume", str(base), "--set", "execution.max_cycles=3"]) == 0

    # Same folder, earlier cycles untouched, log extended, record of the resume.
    assert _only_run_folder(interrupted_root) == base
    assert (results / "cycle_1" / "cycle_metadata.json").read_bytes() == first_cycle
    log = (results / "log.txt").read_text(encoding="utf-8")
    assert log.startswith(first_log) and len(log) > len(first_log)
    saved = json.loads((results / "resolved_config.json").read_text(encoding="utf-8"))
    assert saved["execution"]["resume_from_cycle"] == 3
    assert saved["execution"]["max_cycles"] == 3

    continuous_results = _only_run_folder(continuous_root) / "1D_mode"
    continuous = _terminal_fields(continuous_results, 3)
    resumed = _terminal_fields(results, 3)
    # Cycle 3 is not a repeat of cycle 2, so matching it proves the resume
    # started from the saved cycle-2 state rather than any other state.
    previous = _terminal_fields(continuous_results, 2)
    assert any(not np.array_equal(previous[name], continuous[name]) for name in continuous)
    assert continuous.keys() == resumed.keys()
    for name in continuous:
        np.testing.assert_array_equal(resumed[name], continuous[name])


def test_step_run_replays_in_place(monkeypatch, tmp_path):
    from runners import run_steps_simulation as runner

    monkeypatch.setattr(runner, "DEVELOPMENT_SETTINGS", {})

    root = tmp_path / "steps"
    assert runner.main(["--config", CASE, "--set", f"execution.output_root={root}",
                        "--set", "execution.print_to_terminal=false"]) == 0
    base = _only_run_folder(root)
    solution = base / "1D_mode" / "startup" / "sol.pkl"
    saved_bytes = solution.read_bytes()

    assert runner.main(["--resume", str(base), "--set", "execution.solve=false"]) == 0

    assert _only_run_folder(root) == base
    assert solution.read_bytes() == saved_bytes


def test_step_replay_plots_in_the_case_figure_format(monkeypatch, tmp_path):
    from runners import run_steps_simulation as runner

    monkeypatch.setattr(runner, "DEVELOPMENT_SETTINGS", {})

    root = tmp_path / "steps"
    assert runner.main(["--config", CASE, "--set", f"execution.output_root={root}",
                        "--set", "execution.print_to_terminal=false"]) == 0
    base = _only_run_folder(root)

    assert runner.main(["--resume", str(base), "--set", "execution.solve=false",
                        "--set", "execution.plot_profiles=true",
                        "--set", "execution.plot_format=png"]) == 0

    figures = list((base / "1D_mode" / "startup" / "profiles").rglob("*.*"))
    assert figures
    assert {path.suffix for path in figures} == {".png"}
