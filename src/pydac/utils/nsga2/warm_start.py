"""Validated NSGA-II continuation and Pareto-refinement warm starts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.operators.sampling.lhs import LatinHypercubeSampling


@dataclass(frozen=True)
class WarmStartSpec:
    pareto_path: str
    population_path: str | None = None
    mode: Literal["continuation", "refining"] = "refining"
    source: Literal["population", "pareto"] = "pareto"
    fill: Literal["lhs", "none"] = "lhs"

    def __post_init__(self):
        if self.mode not in {"continuation", "refining"}:
            raise ValueError(
                f"Unsupported warm-start mode '{self.mode}'. "
                "Expected 'continuation' or 'refining'."
            )
        if self.source not in {"population", "pareto"}:
            raise ValueError(
                f"Unsupported warm-start source '{self.source}'. "
                "Expected 'population' or 'pareto'."
            )
        if self.fill not in {"lhs", "none"}:
            raise ValueError(
                f"Unsupported warm-start fill mode '{self.fill}'. "
                "Expected 'lhs' or 'none'."
            )

        pareto_path = Path(self.pareto_path)
        if not pareto_path.exists():
            raise FileNotFoundError(f"Pareto front file not found: {pareto_path}")
        needs_population = self.mode == "continuation" or self.source == "population"
        if needs_population:
            if self.population_path is None:
                raise ValueError(
                    "A population_path is required for continuation or a "
                    "population-sourced warm start."
                )
            population_path = Path(self.population_path)
            if not population_path.exists():
                raise FileNotFoundError(
                    f"Population file not found: {population_path}"
                )


def _expected_var_names(problem=None, expected_var_names=None):
    if expected_var_names is not None:
        return tuple(expected_var_names)
    run_config = getattr(problem, "run_config", None)
    if run_config is not None and hasattr(run_config, "var_names"):
        return tuple(run_config.var_names)
    return None


def _validate_var_names(data, *, path, expected_var_names=None):
    expected = tuple(expected_var_names) if expected_var_names is not None else None
    if expected is None:
        return

    if "var_names" not in data.files:
        raise ValueError(
            "Warm-start file has no var_names metadata: "
            f"{path}. Expected decision variables {list(expected)}. "
            "Regenerate the source Pareto with the velocity-space "
            "NSGA-II refactor, or explicitly migrate the old population "
            "before warm-starting."
        )

    found = tuple(str(v) for v in data["var_names"])
    if found != expected:
        raise ValueError(
            "Warm-start decision variables do not match this campaign. "
            f"Found {list(found)} in {path}, expected {list(expected)}. "
            "This usually means an old absolute-flow Pareto is being used "
            "with the velocity-space NSGA-II refactor."
        )


def _load_warm_start_X(path, *, expected_var_names=None):
    with np.load(path) as data:
        _validate_var_names(
            data,
            path=path,
            expected_var_names=expected_var_names,
        )
        if "X" not in data.files:
            raise ValueError(f"Warm-start file has no 'X' array: {path}")
        X = np.asarray(data["X"], dtype=float)
    if X.ndim != 2:
        raise ValueError(
            f"Warm-start array 'X' in {path} must be 2D. Received shape {X.shape}."
        )
    return X


def _load_evaluated_population(problem, path, *, expected_var_names=None):
    """Load a completed NSGA-II population without reevaluating its parents."""
    expected_constraints = tuple(getattr(problem, "constraint_names", ()))
    with np.load(path) as data:
        _validate_var_names(
            data,
            path=path,
            expected_var_names=expected_var_names,
        )
        missing = {"X", "F", "G"}.difference(data.files)
        if missing:
            raise ValueError(
                f"Continuation checkpoint {path} is missing arrays "
                f"{sorted(missing)}. A continuation requires saved X, F and G."
            )
        if expected_constraints:
            if "constraint_names" not in data.files:
                raise ValueError(
                    "Continuation checkpoint has no constraint_names metadata: "
                    f"{path}. Expected {list(expected_constraints)}."
                )
            found_constraints = tuple(str(v) for v in data["constraint_names"])
            if found_constraints != expected_constraints:
                raise ValueError(
                    "Continuation constraints do not match this campaign. "
                    f"Found {list(found_constraints)} in {path}, expected "
                    f"{list(expected_constraints)}."
                )

        X = np.asarray(data["X"], dtype=float)
        F = np.asarray(data["F"], dtype=float)
        G = np.asarray(data["G"], dtype=float)

    expected_shapes = {
        "X": (len(X), int(problem.n_var)),
        "F": (len(X), int(problem.n_obj)),
        "G": (len(X), int(problem.n_ieq_constr)),
    }
    for name, values in (("X", X), ("F", F), ("G", G)):
        if values.shape != expected_shapes[name]:
            raise ValueError(
                f"Continuation array '{name}' in {path} has shape {values.shape}; "
                f"expected {expected_shapes[name]}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Continuation array '{name}' in {path} contains non-finite values."
            )
    if len(X) == 0:
        raise ValueError(f"Continuation checkpoint contains no individuals: {path}")

    H = np.zeros((len(X), int(problem.n_eq_constr)), dtype=float)
    population = Population.new(X=X, F=F, G=G, H=H)
    population.apply(
        lambda individual: individual.evaluated.update(("F", "G", "H"))
    )
    return population


def _bounds_mask(X, *, xl, xu):
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)
    tol = 1e-12 * np.maximum(1.0, np.maximum(np.abs(xl), np.abs(xu)))
    return np.all((X >= (xl - tol)) & (X <= (xu + tol)), axis=1)


def _population_from_rows(X):
    if len(X) == 0:
        raise ValueError("Cannot build a pymoo population from an empty design matrix.")
    return Population.create(*[Individual(X=x) for x in X])


def _warm_start_path(spec: WarmStartSpec) -> str:
    if spec.mode == "continuation":
        assert spec.population_path is not None
        return spec.population_path
    if spec.source == "population":
        assert spec.population_path is not None
        return spec.population_path
    return spec.pareto_path


def build_initial_population(
    problem,
    pop_size: int,
    warm_start: WarmStartSpec,
    *,
    rng: np.random.Generator | None = None,
):
    """
    Build the initial pymoo population for a warm-started campaign.

    Returns
    -------
    tuple
        `(population, effective_pop_size, label, summary)` where `label` is a
        short human-readable description for logging and `summary` is a dict
        with detailed warm-start statistics.
    """
    expected_var_names = _expected_var_names(problem=problem)
    path = _warm_start_path(warm_start)
    X_saved = _load_warm_start_X(path, expected_var_names=expected_var_names)

    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    if X_saved.shape[1] != xl.size:
        raise ValueError(
            "Warm-start decision dimension does not match this campaign. "
            f"Found {X_saved.shape[1]} columns in {path}, expected {xl.size}."
        )

    mask_in_bounds = _bounds_mask(X_saved, xl=xl, xu=xu)

    if warm_start.mode == "continuation":
        if not np.all(mask_in_bounds):
            raise ValueError(
                "Continuation warm-start found individuals outside the current "
                f"bounds in {path}. This indicates the campaign bounds changed; "
                "use refining mode instead of continuation."
            )
        population = _load_evaluated_population(
            problem,
            path,
            expected_var_names=expected_var_names,
        )
        summary = {
            "mode": "continuation",
            "source": "population",
            "path": str(path),
            "loaded": int(len(X_saved)),
            "reused": int(len(X_saved)),
            "new": 0,
            "dropped_out_of_bounds": 0,
            "truncated": 0,
            "fill": "none",
            "effective_pop_size": int(len(population)),
            "pop_size_overridden": True,
            "parents_already_evaluated": True,
        }
        print(
            f"[INFO] Continuation loaded {len(population)} evaluated individuals "
            f"from {path}; parent reevaluation will be skipped and pop_size is "
            "overridden to the saved population size."
        )
        return population, len(population), "continuation", summary

    rng = np.random.default_rng() if rng is None else rng

    n_total = len(X_saved)
    X_in_bounds = X_saved[mask_in_bounds]
    n_dropped = n_total - len(X_in_bounds)
    if n_dropped:
        print(
            f"[INFO] Refining warm-start dropped {n_dropped} out-of-bounds "
            f"individuals from {path}."
        )

    if len(X_in_bounds) > pop_size:
        idx = rng.choice(len(X_in_bounds), size=pop_size, replace=False)
        X_seed = X_in_bounds[idx]
        n_truncated = len(X_in_bounds) - pop_size
        print(
            f"[INFO] Refining warm-start randomly truncated {len(X_in_bounds)} "
            f"reusable individuals down to pop_size={pop_size}."
        )
    else:
        X_seed = X_in_bounds
        n_truncated = 0

    reused = len(X_seed)
    n_missing = pop_size - reused

    individuals = [Individual(X=x) for x in X_seed]
    if n_missing > 0:
        if warm_start.fill == "none":
            if reused == 0:
                raise ValueError(
                    "Refining warm-start has no reusable in-bounds individuals "
                    f"in {path} and fill='none' prevents LHS backfill."
                )
        else:
            sampler = LatinHypercubeSampling()
            X_new = sampler.do(problem, n_samples=n_missing).get("X")
            individuals.extend(Individual(X=x) for x in X_new)

    if not individuals:
        raise ValueError(
            "Refining warm-start produced an empty initial population. "
            "Check the warm-start files and bounds."
        )

    population = Population.create(*individuals)
    label = f"refining ({warm_start.source})"
    summary = {
        "mode": "refining",
        "source": warm_start.source,
        "path": str(path),
        "loaded": int(n_total),
        "reused": int(reused),
        "new": int(len(population) - reused),
        "dropped_out_of_bounds": int(n_dropped),
        "truncated": int(n_truncated),
        "fill": warm_start.fill,
        "effective_pop_size": int(len(population)),
        "pop_size_overridden": False,
    }
    print(
        f"[INFO] Refining warm-start built {len(population)} individuals from {path} "
        f"({reused} reused, {len(population) - reused} new via "
        f"{'LHS' if len(population) > reused else 'no fill'})."
    )
    return population, len(population), label, summary
