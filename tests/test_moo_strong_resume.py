"""Exact fixed-generation NSGA-II continuation from atomic checkpoints."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pytest


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.termination.max_gen import MaximumGenerationTermination

from pydac.config.errors import RunRefused
from pydac.utils.nsga2.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    campaign_identity,
    discard_uncommitted_checkpoint_artifacts,
    load_checkpoint_population,
    load_checkpoint_state,
    pymoo_has_algorithm_generator,
    restore_algorithm_from_checkpoint,
    save_generation_checkpoint,
)
from pydac.utils.nsga2.evaluation import make_eval_folder, write_candidate_record


class ToyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, xl=-2.0, xu=2.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array(
            [
                np.sum((x - 0.35) ** 2),
                np.sum((x + 0.65) ** 2),
            ]
        )


class AllInfeasibleToyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_ieq_constr=1, xl=-2.0, xu=2.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array([np.sum(x**2), np.sum((x - 1.0) ** 2)])
        out["G"] = np.array([1.0 + np.sum(x**2)])


def _new_algorithm(problem, *, total_generations, seed, callback=None):
    algorithm = NSGA2(
        pop_size=20,
        sampling=LatinHypercubeSampling(),
        crossover=SBX(prob=0.9, eta=15.0),
        mutation=PM(eta=20.0),
        eliminate_duplicates=True,
    )
    algorithm.setup(
        problem,
        termination=MaximumGenerationTermination(total_generations),
        seed=seed,
        callback=callback or Callback(),
        save_history=False,
        verbose=False,
    )
    return algorithm


class CheckpointEveryGeneration(Callback):
    def __init__(self, base_folder, identity):
        super().__init__()
        self.base_folder = base_folder
        self.identity = identity

    def notify(self, algorithm):
        save_generation_checkpoint(
            self.base_folder,
            algorithm,
            identity=self.identity,
            var_names=("x0", "x1", "x2"),
            constraint_names=(),
            candidate_records_path=None,
            candidate_records_count=0,
        )


def _sorted_population(algorithm):
    X = np.asarray(algorithm.pop.get("X"), dtype=float)
    F = np.asarray(algorithm.pop.get("F"), dtype=float)
    order = np.lexsort(tuple(X[:, index] for index in reversed(range(X.shape[1]))))
    return X[order], F[order]


def test_interrupted_plus_resumed_matches_uninterrupted_trajectory(tmp_path):
    seed = 17
    total_generations = 6
    identity = campaign_identity(
        {
            "problem": "deterministic-toy",
            "population_size": 20,
            "total_generations": total_generations,
            "seed": seed,
        }
    )

    continuous = _new_algorithm(
        ToyProblem(),
        total_generations=total_generations,
        seed=seed,
    )
    continuous.run()

    interrupted_root = tmp_path / "interrupted"
    interrupted = _new_algorithm(
        ToyProblem(),
        total_generations=3,
        seed=seed,
        callback=CheckpointEveryGeneration(interrupted_root, identity),
    )
    interrupted.run()

    state = load_checkpoint_state(
        interrupted_root,
        expected_identity=identity,
    )
    assert state["completed_generations"] == 3
    assert state["completed_evaluations"] == 60
    population = load_checkpoint_population(
        interrupted_root,
        state,
        expected_var_names=("x0", "x1", "x2"),
        expected_constraint_names=(),
    )

    resumed = _new_algorithm(
        ToyProblem(),
        total_generations=total_generations,
        seed=seed,
        callback=CheckpointEveryGeneration(interrupted_root, identity),
    )
    restore_algorithm_from_checkpoint(resumed, population, state)
    resumed.run()

    X_continuous, F_continuous = _sorted_population(continuous)
    X_resumed, F_resumed = _sorted_population(resumed)
    np.testing.assert_array_equal(X_resumed, X_continuous)
    np.testing.assert_array_equal(F_resumed, F_continuous)
    assert resumed.evaluator.n_eval == continuous.evaluator.n_eval == 120

    final_state = load_checkpoint_state(
        interrupted_root,
        expected_identity=identity,
    )
    assert final_state["completed_generations"] == total_generations
    assert final_state["completed_evaluations"] == 120


def test_checkpoint_rejects_identity_mismatch(tmp_path):
    identity = campaign_identity({"campaign": "a"})
    algorithm = _new_algorithm(
        ToyProblem(),
        total_generations=1,
        seed=1,
        callback=CheckpointEveryGeneration(tmp_path, identity),
    )
    algorithm.run()

    with pytest.raises(ValueError, match="identity does not match"):
        load_checkpoint_state(
            tmp_path,
            expected_identity=campaign_identity({"campaign": "b"}),
        )


def test_checkpoint_without_algorithm_generator_loads_only_where_pymoo_has_none(tmp_path):
    # A schema-3 state is a schema-4 one without the algorithm's generator.
    identity = campaign_identity({"campaign": "schema-3"})
    algorithm = _new_algorithm(
        ToyProblem(),
        total_generations=1,
        seed=5,
        callback=CheckpointEveryGeneration(tmp_path, identity),
    )
    algorithm.run()
    state_path = tmp_path / "results" / "checkpoint_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert (state["rng_state"]["algorithm"] is None) != pymoo_has_algorithm_generator()
    state["schema_version"] = 3
    del state["rng_state"]["algorithm"]
    state_path.write_text(json.dumps(state), encoding="utf-8")

    if pymoo_has_algorithm_generator():
        with pytest.raises(ValueError, match="no saved per-algorithm generator"):
            load_checkpoint_state(tmp_path, expected_identity=identity)
    else:
        assert load_checkpoint_state(tmp_path, expected_identity=identity)["schema_version"] == 3


def test_first_generation_checkpoint_preserves_missing_infeasible_metadata(tmp_path):
    identity = campaign_identity({"campaign": "all-infeasible"})
    algorithm = _new_algorithm(
        AllInfeasibleToyProblem(),
        total_generations=1,
        seed=3,
        callback=CheckpointEveryGeneration(tmp_path, identity),
    )
    algorithm.run()

    state = load_checkpoint_state(tmp_path, expected_identity=identity)
    population = load_checkpoint_population(
        tmp_path,
        state,
        expected_var_names=("x0", "x1", "x2"),
        expected_constraint_names=(),
    )

    assert state["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert population.get("rank").tolist() == [None] * len(population)
    assert population.get("crowding").tolist() == [None] * len(population)


def test_resume_removes_only_files_newer_than_commit_marker(tmp_path):
    checkpoint_root = tmp_path / "results" / "checkpoints"
    records_root = tmp_path / "results" / "candidate_records"
    checkpoint_root.mkdir(parents=True)
    records_root.mkdir(parents=True)
    for generation in (2, 3, 4):
        (checkpoint_root / f"generation_{generation:04d}_population.npz").write_bytes(
            b"checkpoint"
        )
        (records_root / f"generation_{generation:04d}.jsonl").write_text(
            json.dumps({"generation": generation}),
            encoding="utf-8",
        )

    removed = discard_uncommitted_checkpoint_artifacts(
        tmp_path,
        completed_generations=3,
    )

    assert [Path(item).as_posix() for item in removed] == [
        "results/candidate_records/generation_0004.jsonl",
        "results/checkpoints/generation_0004_population.npz",
    ]
    assert (checkpoint_root / "generation_0003_population.npz").is_file()
    assert (records_root / "generation_0003.jsonl").is_file()


class CampaignToyProblem(ElementwiseProblem):
    def __init__(self, run_config, **kwargs):
        self.run_config = run_config
        self.constraint_names = ("valid_evaluation_and_css",)
        super().__init__(
            n_var=3,
            n_obj=2,
            n_ieq_constr=1,
            xl=-2.0,
            xu=2.0,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        eval_folder, generation, eval_id = make_eval_folder(
            self.run_config.base_folder
        )
        objectives = np.array(
            [
                np.sum((x - 0.35) ** 2),
                np.sum((x + 0.65) ** 2),
            ]
        )
        write_candidate_record(
            eval_folder,
            {
                "generation": generation,
                "evaluation_id": eval_id,
                "decision_vector": x.tolist(),
                "pid": os.getpid(),
                "objectives": objectives.tolist(),
            },
        )
        out["F"] = objectives
        out["G"] = np.array([-1.0])
        out["Extra"] = {
            "co2_pur": 0.99,
            "o2_ppm": 1.0,
            "cwc": 1.0,
            "failed": False,
            "css_reached": True,
            "css_cycles_completed": 2,
            "evaluation_wall_time_s": 0.0,
            "deltaT_preheat": 1.0,
        }


@pytest.mark.parametrize(
    "method", [m for m in ("spawn", "forkserver") if m in mp.get_all_start_methods()]
)
def test_spawned_workers_read_the_parent_generation(tmp_path, method):
    # Windows, macOS and Linux from Python 3.14 do not fork: a worker re-imports the
    # module and, without the pool initializer, reads a fresh counter at 0.
    from pydac.utils.nsga2 import logging as shared
    from pydac.utils.nsga2.evaluation import make_eval_folder

    # Counter, lock and pool from one context, as the default context gives in a run.
    context = mp.get_context(method)
    with context.Pool(
        1,
        initializer=shared.share_worker_state,
        initargs=(context.Value("i", 4), context.Lock()),
    ) as pool:
        eval_folder, gen_id, _ = pool.apply(make_eval_folder, (str(tmp_path),))
    assert gen_id == 5
    assert Path(eval_folder).parent.name == "gen_05"


@pytest.mark.parametrize("n_procs", [1, 2])
def test_public_runner_resumes_to_cumulative_generation_cap(monkeypatch, tmp_path, n_procs):
    from pydac.workflows import orchestrator_NSGA2 as nsga2

    class PlannedInterruption(RuntimeError):
        pass

    real_logger = nsga2.NSGA2Logger

    class InterruptAfterThree(real_logger):
        def __call__(self, algorithm):
            super().__call__(algorithm)
            if algorithm.n_gen == 3:
                raise PlannedInterruption

    monkeypatch.setattr(nsga2, "TVSAOptimizationProblem", CampaignToyProblem)
    bounds = {
        "t_ads": (-2.0, 2.0),
        "t_heat": (-2.0, 2.0),
        "t_des": (-2.0, 2.0),
        "P_vac": (-2.0, 2.0),
        "v_feed": (-2.0, 2.0),
        "v_steam": (-2.0, 2.0),
    }

    def run_config(folder):
        config = nsga2.NSGA2RunConfig(
            base_folder=str(folder),
            bounds=bounds,
            constraints=nsga2.NSGA2Constraints(),
        )
        # The campaign toy has three variables while the real config surface
        # still supplies the scientific identity used by the runner.
        config.var_names = ("x0", "x1", "x2")
        config.bounds = {name: (-2.0, 2.0) for name in config.var_names}
        return config

    interrupted_root = tmp_path / "runner_interrupted"
    monkeypatch.setattr(nsga2, "NSGA2Logger", InterruptAfterThree)
    with pytest.raises(PlannedInterruption):
        nsga2.run_nsga2_optimization(
            run_config(interrupted_root),
            pop_size=20,
            n_generations=6,
            seed=17,
            n_procs=n_procs,
            termination_config=nsga2.NSGA2TerminationConfig(
                mode="fixed_generations"
            ),
            run_postprocessing=False,
        )

    state = json.loads(
        (interrupted_root / "results/checkpoint_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["completed_generations"] == 3

    from pydac.utils import provenance
    initial_attempts = list((interrupted_root / "results/provenance").glob("*.json"))
    assert len(initial_attempts) == 1
    initial_provenance = provenance.read_record(initial_attempts[0])
    checkpoint_artifact = provenance.file_artifact(interrupted_root / "results/checkpoint_state.json")
    monkeypatch.setattr(nsga2, "NSGA2Logger", real_logger)

    def continue_campaign():
        return nsga2.run_nsga2_optimization(
            run_config(interrupted_root),
            pop_size=20,
            n_generations=6,
            seed=17,
            n_procs=n_procs,
            warm_start=nsga2.WarmStartSpec(
                pareto_path=str(
                    interrupted_root / "results/last_pareto_front.npz"
                ),
                population_path=str(interrupted_root / "results/last_population.npz"),
                mode="continuation",
            ),
            termination_config=nsga2.NSGA2TerminationConfig(mode="fixed_generations"),
            run_postprocessing=False,
        )

    # Another pyDAC version must not continue the campaign or write into it.
    def snapshot():
        return {path: path.read_bytes() for path in interrupted_root.rglob("*") if path.is_file()}

    before = snapshot()
    written_by = provenance.__version__
    monkeypatch.setattr(provenance, "__version__", "3.0.0.dev99")
    with pytest.raises(RunRefused, match=f"written by pyDAC {written_by}; this is pyDAC 3.0.0.dev99"):
        continue_campaign()
    assert snapshot() == before
    monkeypatch.setattr(provenance, "__version__", written_by)

    resumed = continue_campaign()
    continuous = nsga2.run_nsga2_optimization(
        run_config(tmp_path / "runner_continuous"),
        pop_size=20,
        n_generations=6,
        seed=17,
        termination_config=nsga2.NSGA2TerminationConfig(mode="fixed_generations"),
        run_postprocessing=False,
    )

    X_resumed, F_resumed = _sorted_population(resumed.algorithm)
    X_continuous, F_continuous = _sorted_population(continuous.algorithm)
    np.testing.assert_array_equal(X_resumed, X_continuous)
    np.testing.assert_array_equal(F_resumed, F_continuous)
    assert resumed.algorithm.evaluator.n_eval == 120
    summary = json.loads((interrupted_root / "results/optimization_summary.json").read_text())
    assert summary["provenance"]["software"]["version"] == written_by
    assert summary["provenance"]["record_id"] != initial_provenance["record_id"]
    assert summary["provenance"]["sources"] == [{"artifact": checkpoint_artifact}]
    assert provenance.read_record(initial_attempts[0]) == initial_provenance
    assert not (interrupted_root / "gen_03").exists()
    assert Path(
        interrupted_root / "results/candidate_records/generation_0006.jsonl"
    ).is_file()

    records = [
        json.loads(line)
        for line in (interrupted_root / "results/candidate_records/generation_0006.jsonl")
        .read_text().splitlines()
    ]
    assert len(records) == 20
    if n_procs == 1:
        assert {record["pid"] for record in records} == {os.getpid()}
    else:
        assert all(record["pid"] != os.getpid() for record in records)
    summary = json.loads(
        (interrupted_root / "results/optimization_summary.json").read_text()
    )
    assert summary["timing"]["n_workers"] == n_procs


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_checkpoint_rejects_damaged_population(tmp_path, damage):
    identity = campaign_identity({"problem": "local-checkpoint"})
    algorithm = _new_algorithm(
        ToyProblem(), total_generations=1, seed=17,
        callback=CheckpointEveryGeneration(tmp_path, identity),
    )
    algorithm.run()
    state = load_checkpoint_state(tmp_path, expected_identity=identity)
    population_path = tmp_path / state["files"]["population"]["path"]
    if damage == "missing":
        population_path.unlink()
    else:
        population_path.write_bytes(b"incomplete population")
    with pytest.raises(ValueError, match="missing or corrupt"):
        load_checkpoint_state(tmp_path, expected_identity=identity)


def test_failed_checkpoint_commit_preserves_last_committed_generation(monkeypatch, tmp_path):
    identity = campaign_identity({"problem": "local-checkpoint"})
    algorithm = _new_algorithm(
        ToyProblem(), total_generations=3, seed=17,
        callback=CheckpointEveryGeneration(tmp_path, identity),
    )
    algorithm.next()
    committed = load_checkpoint_state(tmp_path, expected_identity=identity)
    assert committed["completed_generations"] == 1
    original_replace = Path.replace

    def interrupted_replace(source, destination):
        if Path(destination).name == "checkpoint_state.json":
            raise OSError("simulated interruption before marker replacement")
        return original_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", interrupted_replace)
        with pytest.raises(OSError, match="simulated interruption"):
            algorithm.next()

    state = load_checkpoint_state(tmp_path, expected_identity=identity)
    assert state == committed
    population = load_checkpoint_population(
        tmp_path, state, expected_var_names=("x0", "x1", "x2"),
        expected_constraint_names=(),
    )
    resumed = _new_algorithm(ToyProblem(), total_generations=3, seed=17)
    restore_algorithm_from_checkpoint(resumed, population, state)
    resumed.run()
    continuous = _new_algorithm(ToyProblem(), total_generations=3, seed=17)
    continuous.run()
    for actual, expected in zip(_sorted_population(resumed), _sorted_population(continuous)):
        np.testing.assert_array_equal(actual, expected)
    assert resumed.evaluator.n_eval == continuous.evaluator.n_eval == 60
