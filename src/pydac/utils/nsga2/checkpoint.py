"""Atomic, RNG- and termination-preserving checkpoints for NSGA-II runs."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pymoo
from pymoo.core.population import Population
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.robust import RobustTermination

from pydac.config.errors import ConfigurationError, RunRefused
from pydac.utils.campaign import atomic_write_json


CHECKPOINT_STATE_FILENAME = "checkpoint_state.json"
CHECKPOINT_SCHEMA_VERSION = 4
# Schema 4 adds the algorithm's own generator (`rng_state["algorithm"]`),
# which pymoo >= 0.6.2 draws from instead of the global ones. Schema 3 adds
# `termination_state`; 1 and 2 remain loadable for campaigns whose
# termination carries no history (fixed generation/evaluation caps). Schemas
# 1-3 load only under a pymoo without that generator (0.6.1.x).
SUPPORTED_CHECKPOINT_SCHEMAS = {1, 2, 3, 4}


def atomic_savez(path: str | Path, **arrays: Any) -> Path:
    """Write an NPZ through a same-directory temporary file and rename it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(destination)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def campaign_identity(definition: Mapping[str, Any]) -> dict[str, Any]:
    """Hash a JSON campaign definition while retaining it for diagnostics."""
    normalized = json.loads(
        json.dumps(definition, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "definition": normalized,
    }


def pymoo_has_algorithm_generator() -> bool:
    """Whether the installed pymoo draws from a per-algorithm generator.

    pymoo 0.6.2 gives every algorithm its own ``random_state``
    (``np.random.Generator``, seeded in ``setup``); 0.6.1.x draws from the
    global NumPy and Python generators only.
    """
    from pymoo.core.algorithm import Algorithm

    return "random_state" in vars(Algorithm())


def _algorithm_generator(algorithm):
    generator = getattr(algorithm, "random_state", None)
    if pymoo_has_algorithm_generator() and not isinstance(generator, np.random.Generator):
        raise RuntimeError(
            "pymoo draws from a per-algorithm generator, but this algorithm has "
            f"none ({type(generator).__name__}); call `setup` before checkpointing."
        )
    return generator if isinstance(generator, np.random.Generator) else None


def capture_rng_state(algorithm) -> dict[str, Any]:
    """Capture the global generators and, if pymoo has one, the algorithm's own."""
    numpy_state = np.random.get_state()
    generator = _algorithm_generator(algorithm)
    return {
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "python": _tuple_to_list(random.getstate()),
        "algorithm": None if generator is None else generator.bit_generator.state,
    }


def _tuple_to_list(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_tuple_to_list(item) for item in value]
    return value


def _list_to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_list_to_tuple(item) for item in value)
    return value


def restore_rng_state(state: Mapping[str, Any], algorithm) -> None:
    """Restore the RNG states captured by :func:`capture_rng_state`.

    ``algorithm`` must already be set up, so that a pymoo with a per-algorithm
    generator has created it; its seeded state is then replaced by the saved one.
    """
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    random.setstate(_list_to_tuple(state["python"]))
    generator = _algorithm_generator(algorithm)
    saved = state.get("algorithm")
    if (generator is None) != (saved is None):
        raise ConfigurationError(
            "Checkpoint per-algorithm generator does not match the installed pymoo: "
            f"saved {'none' if saved is None else saved.get('bit_generator')!r}, "
            f"pymoo {pymoo.__version__} {'has none' if generator is None else 'has one'}."
        )
    if generator is not None:
        generator.bit_generator.state = saved


def _front_data_to_json(data):
    if data is None:
        return None
    return {
        "feas": bool(data["feas"]),
        "ideal": None if data["ideal"] is None else np.asarray(data["ideal"], dtype=float).tolist(),
        "nadir": None if data["nadir"] is None else np.asarray(data["nadir"], dtype=float).tolist(),
        "F": np.asarray(data["F"], dtype=float).tolist(),
        "n_obj": int(np.shape(data["F"])[1]) if np.ndim(data["F"]) == 2 else None,
    }


def _front_data_from_json(data):
    if data is None:
        return None
    F = np.asarray(data["F"], dtype=float)
    if F.size == 0 and data.get("n_obj") is not None:
        F = F.reshape(0, int(data["n_obj"]))
    return {
        "feas": bool(data["feas"]),
        "ideal": None if data["ideal"] is None else np.asarray(data["ideal"], dtype=float),
        "nadir": None if data["nadir"] is None else np.asarray(data["nadir"], dtype=float),
        "F": F,
    }


def capture_termination_state(termination) -> dict[str, Any]:
    """Serialize the mutable state of the pymoo termination tree NSGA-II uses.

    Only the classes built by ``build_nsga2_termination`` are supported; any
    other termination is rejected rather than silently checkpointed without
    its history, which would change when a resumed campaign stops.
    """
    state: dict[str, Any] = {
        "type": type(termination).__name__,
        "perc": float(termination.perc),
        "force_termination": bool(termination.force_termination),
    }
    if isinstance(termination, TerminationCollection):
        state["terminations"] = [
            capture_termination_state(child) for child in termination.terminations
        ]
    elif isinstance(termination, RobustTermination):
        state["period"] = int(termination.history.size)
        state["history"] = [float(value) for value in termination.history]
        state["termination"] = capture_termination_state(termination.termination)
    elif isinstance(termination, MultiObjectiveSpaceTermination):
        state["counter"] = int(termination.counter)
        state["data"] = _front_data_to_json(termination.data)
        state["delta"] = [
            None if value is None else float(value)
            for value in (
                termination.delta_ideal,
                termination.delta_nadir,
                termination.delta_f,
            )
        ]
    elif not isinstance(
        termination, (MaximumGenerationTermination, MaximumFunctionCallTermination)
    ):
        raise TypeError(
            f"Cannot checkpoint termination {type(termination).__name__!r}."
        )
    return state


def restore_termination_state(termination, state: Mapping[str, Any], algorithm) -> None:
    """Restore a state captured by :func:`capture_termination_state` in place.

    History-carrying criteria get their saved state back. Generation and
    evaluation caps carry none: they are recomputed from ``algorithm``, which
    must already sit at the completed generation, so the progress reflects
    this campaign's own caps.
    """
    if type(termination).__name__ != state["type"]:
        raise ConfigurationError(
            "Checkpoint termination structure does not match the campaign: "
            f"{type(termination).__name__} != {state['type']}."
        )
    termination.force_termination = bool(state["force_termination"])
    if isinstance(termination, (MaximumGenerationTermination, MaximumFunctionCallTermination)):
        termination.update(algorithm)
        return
    if isinstance(termination, TerminationCollection):
        children = state["terminations"]
        if len(children) != len(termination.terminations):
            raise ConfigurationError("Checkpoint termination collection size mismatch.")
        for child, child_state in zip(termination.terminations, children):
            restore_termination_state(child, child_state, algorithm)
        termination.perc = (
            1.0
            if termination.force_termination
            else max(child.perc for child in termination.terminations)
        )
        return
    termination.perc = float(state["perc"])
    if isinstance(termination, RobustTermination):
        if int(state["period"]) != int(termination.history.size):
            raise ConfigurationError("Checkpoint robust-termination period mismatch.")
        termination.history.clear()
        for value in state["history"]:
            termination.history.append(float(value))
        restore_termination_state(termination.termination, state["termination"], algorithm)
    elif isinstance(termination, MultiObjectiveSpaceTermination):
        termination.counter = int(state["counter"])
        termination.data = _front_data_from_json(state["data"])
        (
            termination.delta_ideal,
            termination.delta_nadir,
            termination.delta_f,
        ) = state["delta"]


def termination_has_history(termination) -> bool:
    """Whether resuming this termination requires its checkpointed state."""
    if isinstance(termination, TerminationCollection):
        return any(termination_has_history(child) for child in termination.terminations)
    return isinstance(termination, (RobustTermination, MultiObjectiveSpaceTermination))


def restore_checkpoint_termination(termination, state: Mapping[str, Any], algorithm) -> None:
    """Bring ``termination`` to its state after the checkpoint's generation.

    ``algorithm`` must report the completed generation and evaluation count.
    Raises when the checkpoint cannot faithfully restore this termination.
    """
    termination_state = state.get("termination_state")
    if termination_state is not None:
        restore_termination_state(termination, termination_state, algorithm)
    elif termination_has_history(termination):
        raise ConfigurationError(
            "This checkpoint predates saved termination state; a campaign whose "
            "termination keeps a history (robust_pareto) cannot be resumed from it."
        )
    else:
        termination.update(algorithm)


def checkpoint_termination_met(termination, state: Mapping[str, Any]) -> bool:
    """Whether a fresh ``termination`` restored from ``state`` has already stopped.

    Uses a stand-in carrying only the committed counters, so a continuation
    can be validated and refused before any worker or file is created. Any
    incompatibility between checkpoint and termination raises here too.
    """
    completed = int(state["completed_generations"])
    probe = SimpleNamespace(
        n_gen=completed,
        n_iter=completed,
        evaluator=SimpleNamespace(n_eval=int(state["completed_evaluations"])),
    )
    restore_checkpoint_termination(termination, state, probe)
    return termination.has_terminated()


def _population_arrays(population, *, var_names, constraint_names) -> dict[str, Any]:
    raw_rank = np.asarray(population.get("rank"), dtype=object).reshape(-1)
    raw_crowding = np.asarray(population.get("crowding"), dtype=object).reshape(-1)
    if len(raw_rank) != len(population) or len(raw_crowding) != len(population):
        raise RuntimeError("NSGA-II population is missing rank/crowding metadata.")
    rank_present = np.asarray([value is not None for value in raw_rank], dtype=bool)
    crowding_present = np.asarray(
        [value is not None for value in raw_crowding], dtype=bool
    )
    rank = np.asarray(
        [int(value) if present else -1 for value, present in zip(raw_rank, rank_present)],
        dtype=int,
    )
    crowding = np.asarray(
        [
            float(value) if present else 0.0
            for value, present in zip(raw_crowding, crowding_present)
        ],
        dtype=float,
    )
    return {
        "X": np.asarray(population.get("X"), dtype=float),
        "F": np.asarray(population.get("F"), dtype=float),
        "G": np.asarray(population.get("G"), dtype=float),
        "H": np.asarray(population.get("H"), dtype=float),
        "CV": np.asarray(population.get("CV"), dtype=float),
        "feasible": np.asarray(population.get("feas"), dtype=bool).reshape(-1),
        "rank": rank,
        "crowding": crowding,
        "rank_present": rank_present,
        "crowding_present": crowding_present,
        "var_names": np.asarray(tuple(var_names), dtype=str),
        "constraint_names": np.asarray(tuple(constraint_names), dtype=str),
    }


def save_generation_checkpoint(
    base_folder: str | Path,
    algorithm,
    *,
    identity: Mapping[str, Any],
    var_names,
    constraint_names,
    candidate_records_path: str | Path | None,
    candidate_records_count: int,
) -> dict[str, Any]:
    """Commit one completed generation, with the state marker written last."""
    base = Path(base_folder)
    results = base / "results"
    checkpoint_root = results / "checkpoints"
    generation = int(algorithm.n_gen)
    evaluations = int(algorithm.evaluator.n_eval)
    token = f"generation_{generation:04d}"

    population_arrays = _population_arrays(
        algorithm.pop,
        var_names=var_names,
        constraint_names=constraint_names,
    )
    feasible = population_arrays["feasible"]
    rank = population_arrays["rank"]
    is_front = feasible & population_arrays["rank_present"] & (rank == 0)
    pareto_arrays = {
        "X": population_arrays["X"][is_front],
        "F": population_arrays["F"][is_front],
        "G": population_arrays["G"][is_front],
        "var_names": population_arrays["var_names"],
        "constraint_names": population_arrays["constraint_names"],
    }

    population_path = atomic_savez(
        checkpoint_root / f"{token}_population.npz",
        **population_arrays,
    )
    pareto_path = atomic_savez(
        checkpoint_root / f"{token}_pareto.npz",
        **pareto_arrays,
    )
    # Compatibility aliases are never used to decide what is committed; the
    # state marker below points to immutable generation-versioned files.
    atomic_savez(results / "last_population.npz", **population_arrays)
    atomic_savez(results / "last_pareto_front.npz", **pareto_arrays)

    records_path = (
        None if candidate_records_path is None else Path(candidate_records_path)
    )
    files = {
        "population": {
            "path": str(population_path.relative_to(base)),
            "sha256": _sha256(population_path),
        },
        "pareto": {
            "path": str(pareto_path.relative_to(base)),
            "sha256": _sha256(pareto_path),
        },
        "candidate_records": None,
    }
    if records_path is not None:
        files["candidate_records"] = {
            "path": str(records_path.relative_to(base)),
            "sha256": _sha256(records_path),
            "count": int(candidate_records_count),
        }

    state = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "pymoo_version": pymoo.__version__,
        "campaign_identity": dict(identity),
        "completed_generations": generation,
        "completed_evaluations": evaluations,
        "population_size": int(len(algorithm.pop)),
        "rng_state": capture_rng_state(algorithm),
        # pymoo updates the termination immediately before the callback that
        # commits this checkpoint, so this is its state after `generation`.
        "termination_state": capture_termination_state(algorithm.termination),
        "files": files,
    }
    atomic_write_json(results / CHECKPOINT_STATE_FILENAME, state, sort_keys=True)
    return state


def load_checkpoint_state(
    base_folder: str | Path,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and validate the latest committed checkpoint state."""
    base = Path(base_folder)
    state_path = base / "results" / CHECKPOINT_STATE_FILENAME
    if not state_path.is_file():
        raise ConfigurationError(
            f"Strong continuation requires {state_path}; no committed checkpoint exists."
        )
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    if state.get("schema_version") not in SUPPORTED_CHECKPOINT_SCHEMAS:
        raise ConfigurationError(
            f"Unsupported checkpoint schema: {state.get('schema_version')!r}."
        )
    if state.get("pymoo_version") != pymoo.__version__:
        raise ConfigurationError(
            "Checkpoint pymoo version mismatch: "
            f"{state.get('pymoo_version')!r} != {pymoo.__version__!r}."
        )
    if state["rng_state"].get("algorithm") is None and pymoo_has_algorithm_generator():
        raise ConfigurationError(
            f"This checkpoint (schema {state['schema_version']}) has no saved "
            f"per-algorithm generator, which pymoo {pymoo.__version__} draws from; "
            "a resumed campaign would not continue the uninterrupted one."
        )
    found_identity = state.get("campaign_identity", {})
    if found_identity.get("sha256") != expected_identity.get("sha256"):
        raise RunRefused(
            "Checkpoint campaign identity does not match the requested campaign "
            f"({found_identity.get('sha256')} != {expected_identity.get('sha256')})."
        )

    for name in ("population", "pareto"):
        entry = state["files"][name]
        path = base / entry["path"]
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise ConfigurationError(f"Committed checkpoint file is missing or corrupt: {path}")
    records = state["files"].get("candidate_records")
    if records is not None:
        path = base / records["path"]
        if not path.is_file() or _sha256(path) != records["sha256"]:
            raise ConfigurationError(f"Committed candidate record is missing or corrupt: {path}")
        with path.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        if count != int(records["count"]):
            raise ConfigurationError(
                f"Committed candidate record count mismatch in {path}: "
                f"{count} != {records['count']}."
            )
    return state


def load_checkpoint_population(
    base_folder: str | Path,
    state: Mapping[str, Any],
    *,
    expected_var_names,
    expected_constraint_names,
):
    """Rebuild an evaluated pymoo population, including tournament metadata."""
    base = Path(base_folder)
    path = base / state["files"]["population"]["path"]
    with np.load(path) as data:
        found_vars = tuple(str(value) for value in data["var_names"])
        found_constraints = tuple(str(value) for value in data["constraint_names"])
        if found_vars != tuple(expected_var_names):
            raise ConfigurationError(
                f"Checkpoint variables {found_vars} != {tuple(expected_var_names)}."
            )
        if found_constraints != tuple(expected_constraint_names):
            raise ConfigurationError(
                "Checkpoint constraints "
                f"{found_constraints} != {tuple(expected_constraint_names)}."
            )
        arrays = {name: np.array(data[name], copy=True) for name in data.files}

    population = Population.new(
        X=arrays["X"],
        F=arrays["F"],
        G=arrays["G"],
        H=arrays["H"],
        rank=arrays["rank"],
        crowding=arrays["crowding"],
    )
    rank_present = arrays.get(
        "rank_present", np.ones(len(population), dtype=bool)
    ).astype(bool)
    crowding_present = arrays.get(
        "crowding_present", np.ones(len(population), dtype=bool)
    ).astype(bool)
    for individual, has_rank, has_crowding in zip(
        population, rank_present, crowding_present
    ):
        if not has_rank:
            individual.set("rank", None)
        if not has_crowding:
            individual.set("crowding", None)
    population.apply(
        lambda individual: individual.evaluated.update(("F", "G", "H"))
    )
    if len(population) != int(state["population_size"]):
        raise ConfigurationError(
            f"Checkpoint population size {len(population)} != "
            f"{state['population_size']}."
        )
    for name in ("X", "F", "G", "H", "rank"):
        values = np.asarray(arrays[name])
        if not np.all(np.isfinite(values)):
            raise ConfigurationError(f"Checkpoint population field {name!r} is non-finite.")
    # Boundary points deliberately carry infinite NSGA-II crowding distance.
    if np.any(np.isnan(np.asarray(arrays["crowding"], dtype=float))):
        raise ConfigurationError("Checkpoint population field 'crowding' contains NaN.")
    return population


def restore_algorithm_from_checkpoint(algorithm, population, state) -> None:
    """Hydrate a configured NSGA-II instance at the next global generation."""
    if state.get("termination_state") is None and termination_has_history(algorithm.termination):
        raise ConfigurationError(
            "This checkpoint predates saved termination state; a campaign whose "
            "termination keeps a history (robust_pareto) cannot be resumed from it."
        )
    completed_generations = int(state["completed_generations"])
    algorithm.pop = population
    algorithm.off = None
    algorithm.opt = None
    algorithm.is_initialized = True
    algorithm.n_iter = completed_generations + 1
    algorithm.evaluator.n_eval = int(state["completed_evaluations"])
    algorithm.start_time = time.time()
    algorithm._set_optimum()

    # Termination is evaluated at the completed generation, while n_iter
    # must already point to the next offspring generation.
    next_generation = algorithm.n_iter
    algorithm.n_iter = completed_generations
    restore_checkpoint_termination(algorithm.termination, state, algorithm)
    algorithm.n_iter = next_generation
    restore_rng_state(state["rng_state"], algorithm)


def discard_uncommitted_checkpoint_artifacts(
    base_folder: str | Path,
    *,
    completed_generations: int,
) -> list[str]:
    """Remove versioned files newer than the latest committed state marker."""
    base = Path(base_folder)
    removed = []
    patterns = (
        base / "results" / "checkpoints",
        base / "results" / "candidate_records",
    )
    for folder in patterns:
        if not folder.exists():
            continue
        for path in folder.iterdir():
            stem = path.name
            if not stem.startswith("generation_"):
                continue
            try:
                generation = int(stem.split("_", 2)[1].split(".", 1)[0])
            except (IndexError, ValueError):
                continue
            if generation > int(completed_generations):
                path.unlink()
                removed.append(str(path.relative_to(base)))
    return sorted(removed)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_STATE_FILENAME",
    "atomic_savez",
    "campaign_identity",
    "capture_rng_state",
    "capture_termination_state",
    "checkpoint_termination_met",
    "discard_uncommitted_checkpoint_artifacts",
    "load_checkpoint_population",
    "load_checkpoint_state",
    "pymoo_has_algorithm_generator",
    "restore_algorithm_from_checkpoint",
    "restore_checkpoint_termination",
    "restore_rng_state",
    "restore_termination_state",
    "save_generation_checkpoint",
    "termination_has_history",
]
