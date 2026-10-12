"""NSGA-II campaign folders, robust_pareto continuation and `--resume`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import pytest


from tests.test_moo_strong_resume import CampaignToyProblem, _sorted_population
from pydac.config.errors import ConfigurationError, RunRefused
from pydac.utils.nsga2.checkpoint import restore_algorithm_from_checkpoint
from pydac.workflows import orchestrator_NSGA2 as nsga2


BOUNDS = {
    "t_ads": [900.0, 1800.0], "t_heat": [300.0, 600.0], "t_des": [300.0, 900.0],
    "P_vac": [5000.0, 10000.0], "v_feed": [0.01, 0.1], "v_steam": [0.001, 0.01],
}
ROBUST = nsga2.NSGA2TerminationConfig(mode="robust_pareto", ftol=0.05, period=2)


def _toy_config(folder):
    config = nsga2.NSGA2RunConfig(
        base_folder=str(folder),
        bounds={name: tuple(pair) for name, pair in BOUNDS.items()},
        constraints=nsga2.NSGA2Constraints(),
    )
    config.var_names = ("x0", "x1", "x2")
    config.bounds = {name: (-2.0, 2.0) for name in config.var_names}
    return config


def _identity(config, termination=ROBUST, **overrides):
    options = {"pop_size": 20, "n_generations": 30, "seed": 17}
    options.update(overrides)
    return nsga2.nsga2_campaign_identity(config, termination_config=termination, **options)


def test_robust_pareto_resume_stops_where_the_uninterrupted_campaign_stops(monkeypatch, tmp_path):
    class PlannedInterruption(RuntimeError):
        pass

    real_logger = nsga2.NSGA2Logger
    monkeypatch.setattr(nsga2, "TVSAOptimizationProblem", CampaignToyProblem)
    options = dict(pop_size=20, n_generations=30, seed=17, termination_config=ROBUST,
                   run_postprocessing=False)

    continuous = nsga2.run_nsga2_optimization(_toy_config(tmp_path / "continuous"), **options)
    # pymoo advances its counter once more after the final generation.
    stop_generation = continuous.algorithm.n_gen - 1
    assert stop_generation < 30
    # Interrupt one generation before convergence, with a full window: only a
    # restored window lets the next generation stop. Rebuilding the window
    # from the population alone (the pre-schema-3 restore) stops later.
    interrupt_at = stop_generation - 1
    assert interrupt_at >= ROBUST.period

    class InterruptBeforeConvergence(real_logger):
        def __call__(self, algorithm):
            super().__call__(algorithm)
            if algorithm.n_gen == interrupt_at:
                raise PlannedInterruption

    interrupted = tmp_path / "interrupted"
    monkeypatch.setattr(nsga2, "NSGA2Logger", InterruptBeforeConvergence)
    with pytest.raises(PlannedInterruption):
        nsga2.run_nsga2_optimization(_toy_config(interrupted), **options)
    state = json.loads((interrupted / "results/checkpoint_state.json").read_text())
    assert state["completed_generations"] == interrupt_at
    robust_state = state["termination_state"]["terminations"][0]
    assert robust_state["type"] == "RobustTermination"
    assert len(robust_state["history"]) == ROBUST.period

    monkeypatch.setattr(nsga2, "NSGA2Logger", real_logger)
    resumed = nsga2.run_nsga2_optimization(
        _toy_config(interrupted),
        warm_start=nsga2.WarmStartSpec(
            pareto_path=str(interrupted / "results/last_pareto_front.npz"),
            population_path=str(interrupted / "results/last_population.npz"),
            mode="continuation",
        ),
        **options,
    )

    assert resumed.algorithm.n_gen - 1 == stop_generation
    X_resumed, F_resumed = _sorted_population(resumed.algorithm)
    X_continuous, F_continuous = _sorted_population(continuous.algorithm)
    np.testing.assert_array_equal(X_resumed, X_continuous)
    np.testing.assert_array_equal(F_resumed, F_continuous)
    summary = json.loads((interrupted / "results/optimization_summary.json").read_text())
    assert summary["stop_reason"] == "robust_pareto_convergence"
    # The whole campaign (both runs) is counted from its committed records.
    assert summary["evaluations"] == {
        "evaluated": summary["final_evaluations"], "failed": 0, "errors": {}}

    # A converged campaign is complete: continuing it again is refused, and
    # the refusal writes and deletes nothing.
    def snapshot():
        return {str(path.relative_to(interrupted)): path.read_bytes()
                for path in sorted(interrupted.rglob("*")) if path.is_file()}

    before = snapshot()
    assert any(name.startswith("gen_") for name in before)
    with pytest.raises(RunRefused, match="already completed"):
        nsga2.run_nsga2_optimization(
            _toy_config(interrupted),
            warm_start=nsga2.WarmStartSpec(
                pareto_path=str(interrupted / "results/last_pareto_front.npz"),
                population_path=str(interrupted / "results/last_population.npz"),
                mode="continuation",
            ),
            **options,
        )
    assert snapshot() == before


def test_robust_campaign_cannot_resume_from_a_checkpoint_without_termination_state(tmp_path):
    from pymoo.algorithms.moo.nsga2 import NSGA2

    algorithm = NSGA2(pop_size=4)
    algorithm.setup(CampaignToyProblem(_toy_config(tmp_path)),
                    termination=nsga2.build_nsga2_termination(n_generations=5, config=ROBUST))
    with pytest.raises(ConfigurationError, match="predates saved termination state"):
        restore_algorithm_from_checkpoint(
            algorithm, algorithm.pop, {"completed_generations": 1, "completed_evaluations": 4,
                                       "rng_state": None},
        )


def test_campaign_identity_separates_campaigns_and_guards_folders(tmp_path):
    config = _toy_config(tmp_path)
    base = _identity(config)
    assert _identity(config)["sha256"] == base["sha256"]
    for changed in (_identity(config, seed=18), _identity(config, pop_size=21),
                    _identity(config, termination=nsga2.NSGA2TerminationConfig(mode="fixed_generations"))):
        assert changed["sha256"] != base["sha256"]
    wider = _toy_config(tmp_path)
    wider.bounds = {name: (-3.0, 2.0) for name in wider.var_names}
    assert _identity(wider)["sha256"] != base["sha256"]

    folder = tmp_path / "campaign"
    results = folder / "results"
    results.mkdir(parents=True)
    (results / nsga2.CAMPAIGN_IDENTITY_FILENAME).write_text(json.dumps(base))
    # Same campaign, nothing committed: a fresh start is allowed.
    nsga2.guard_campaign_folder(folder, base, continuation=False)
    with pytest.raises(RunRefused, match="different NSGA-II campaign"):
        nsga2.guard_campaign_folder(folder, _identity(config, seed=18), continuation=False)

    (results / "checkpoint_state.json").write_text(json.dumps({"campaign_identity": base}))
    with pytest.raises(RunRefused, match="continue it with --resume"):
        nsga2.guard_campaign_folder(folder, base, continuation=False)
    nsga2.guard_campaign_folder(folder, base, continuation=True)
    with pytest.raises(RunRefused, match="different NSGA-II campaign"):
        nsga2.guard_campaign_folder(folder, _identity(config, seed=18), continuation=True)


def test_guard_refuses_results_without_a_campaign_identity(tmp_path):
    """Unidentified results are never adopted by a new campaign."""
    identity = _identity(_toy_config(tmp_path))
    folder = tmp_path / "campaign"
    (folder / "results" / "checkpoints").mkdir(parents=True)
    # Caller files beside a campaign (the runner's base case, a study driver's
    # run manifest) and empty folders: a new campaign may start.
    for name in ("resolved_config.json", "case_metadata.json", "run_manifest.json"):
        (folder / name).write_text("{}")
    nsga2.guard_campaign_folder(folder, identity, continuation=False)

    (folder / "results" / "last_pareto_front.npz").write_bytes(b"old campaign")
    with pytest.raises(RunRefused, match="without a campaign identity"):
        nsga2.guard_campaign_folder(folder, identity, continuation=False)

    for artifact in ("pareto_bounds.csv", "gen_03/pid_1_eval_1/log.txt", "nsga2_log.txt"):
        other = tmp_path / artifact.replace("/", "_")
        (other / artifact).parent.mkdir(parents=True)
        (other / artifact).write_text("old campaign")
        with pytest.raises(RunRefused, match="without a campaign identity"):
            nsga2.guard_campaign_folder(other, identity, continuation=False)


def test_resume_rebuilds_the_exact_campaign_from_its_checkpoint(monkeypatch, tmp_path):
    folder = tmp_path / "campaign"
    config = nsga2.NSGA2RunConfig(
        base_folder=str(folder),
        bounds={name: tuple(pair) for name, pair in BOUNDS.items()},
        constraints=nsga2.NSGA2Constraints(purity_min=0.9, eq_work_max=25.0),
        sbx_eta=12.0,
    )
    identity = _identity(config, termination=ROBUST, seed=5)
    results = folder / "results"
    results.mkdir(parents=True)
    (results / "checkpoint_state.json").write_text(json.dumps({"campaign_identity": identity}))
    for name in ("last_pareto_front.npz", "last_population.npz"):
        np.savez(results / name, X=np.zeros((1, 6)))

    calls = []
    monkeypatch.setattr(nsga2, "run_nsga2_optimization",
                        lambda run_config, **kwargs: calls.append((run_config, kwargs)))
    nsga2.resume_nsga2_campaign(folder, n_procs=3)

    rebuilt, kwargs = calls[0]
    assert kwargs["n_procs"] == 3
    assert kwargs["warm_start"].mode == "continuation"
    assert nsga2.nsga2_campaign_identity(
        rebuilt, pop_size=kwargs["pop_size"], n_generations=kwargs["n_generations"],
        seed=kwargs["seed"], termination_config=kwargs["termination_config"],
    )["sha256"] == identity["sha256"]


def test_runner_names_folders_by_campaign_and_refuses_overwrite(monkeypatch, tmp_path, capsys):
    from runners import run_NSGA2 as runner

    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(nsga2, "run_nsga2_optimization",
                        lambda config, **kwargs: calls.append((config, kwargs)))

    def launch(bounds, *extra):
        path = tmp_path / f"bounds_{len(calls)}.json"
        path.write_text(json.dumps(bounds))
        return runner.main(["--bounds", str(path), "--pop-size", "4", "--generations", "2",
                            "--set", "execution.output_root=out", *extra])

    assert launch(BOUNDS) == 0
    wider = dict(BOUNDS, t_ads=[600.0, 1800.0])
    assert launch(wider, "--termination", "robust_pareto", "--period", "3") == 0
    first, second = (Path(config.base_folder) for config, _ in calls)
    assert first != second and first.parent == second.parent
    assert calls[1][1]["termination_config"] == nsga2.NSGA2TerminationConfig(
        mode="robust_pareto", period=3)

    # The first campaign has committed generations: a fresh start is refused.
    (first / "results").mkdir(parents=True)
    (first / "results/checkpoint_state.json").write_text(
        json.dumps({"campaign_identity": _runner_identity(calls[0])}))
    assert launch(BOUNDS) == 2
    assert "continue it with --resume" in capsys.readouterr().err
    assert len(calls) == 2


def _runner_identity(call):
    config, kwargs = call
    return nsga2.nsga2_campaign_identity(
        config, pop_size=kwargs["pop_size"], n_generations=kwargs["n_generations"],
        seed=kwargs["seed"], termination_config=kwargs["termination_config"],
    )


def test_runner_resume_accepts_only_workers(monkeypatch, tmp_path):
    from runners import run_NSGA2 as runner

    folder = tmp_path / "campaign"
    (folder / "results").mkdir(parents=True)
    (folder / "results/checkpoint_state.json").write_text("{}")
    calls = []
    # The continuation preflight is covered by its own tests; here only the flags.
    monkeypatch.setattr(nsga2, "preflight_nsga2_resume", lambda path: {})
    monkeypatch.setattr(nsga2, "resume_nsga2_campaign",
                        lambda path, **kwargs: calls.append((path, kwargs)))

    assert runner.main(["--resume", str(folder), "--workers", "4"]) == 0
    assert calls == [(str(folder), {"n_procs": 4})]
    for extra in (["--seed", "2"], ["--bounds", "b.json"], ["--set", "grid.nz=5"],
                  ["--termination", "robust_pareto"]):
        assert runner.main(["--resume", str(folder), *extra]) == 2
    assert runner.main(["--resume", str(tmp_path / "empty")]) == 2
    monkeypatch.setattr(runner, "RESUME_PATH", str(folder))
    assert runner.main(["--workers", "2"]) == 0
    assert calls[-1] == (str(folder), {"n_procs": 2})


def _files(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in sorted(folder.rglob("*")) if path.is_file()}


@pytest.mark.parametrize(
    "case",
    ["legacy_robust", "incompatible_termination", "corrupt_population",
     "completed_robust", "legacy_evaluation_cap"],
)
def test_refused_continuation_writes_nothing_and_starts_no_workers(monkeypatch, tmp_path, case):
    """Every refusal precedes files and worker pools."""

    class PlannedInterruption(RuntimeError):
        pass

    real_logger = nsga2.NSGA2Logger

    class InterruptAtThree(real_logger):
        def __call__(self, algorithm):
            super().__call__(algorithm)
            if algorithm.n_gen == 3:
                raise PlannedInterruption

    monkeypatch.setattr(nsga2, "TVSAOptimizationProblem", CampaignToyProblem)
    termination = ROBUST
    if case == "legacy_evaluation_cap":
        termination = nsga2.NSGA2TerminationConfig(mode="fixed_generations", max_evaluations=40)
    options = dict(pop_size=20, n_generations=30, seed=17, termination_config=termination,
                   run_postprocessing=False, delete_generation_folders=False)
    folder = tmp_path / case
    if case in ("completed_robust", "legacy_evaluation_cap"):
        nsga2.run_nsga2_optimization(_toy_config(folder), **options)
    else:
        monkeypatch.setattr(nsga2, "NSGA2Logger", InterruptAtThree)
        with pytest.raises(PlannedInterruption):
            nsga2.run_nsga2_optimization(_toy_config(folder), **options)
        monkeypatch.setattr(nsga2, "NSGA2Logger", real_logger)

    marker = folder / "results/checkpoint_state.json"
    state = json.loads(marker.read_text())
    if case in ("legacy_robust", "legacy_evaluation_cap"):
        state["schema_version"] = 2
        state.pop("termination_state")
    elif case == "incompatible_termination":
        state["termination_state"]["terminations"][0]["type"] = "WrongTermination"
    elif case == "corrupt_population":
        (folder / state["files"]["population"]["path"]).write_bytes(b"corrupt")
    marker.write_text(json.dumps(state))

    pools = []
    monkeypatch.setattr(nsga2.mp, "Pool", lambda *args, **kwargs: pools.append(args))
    before = _files(folder)
    with pytest.raises(ValueError):
        nsga2.run_nsga2_optimization(
            _toy_config(folder),
            warm_start=nsga2.WarmStartSpec(
                pareto_path=str(folder / "results/last_pareto_front.npz"),
                population_path=str(folder / "results/last_population.npz"),
                mode="continuation",
            ),
            n_procs=2,
            **options,
        )
    assert _files(folder) == before
    assert pools == []
