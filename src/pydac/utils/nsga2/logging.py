"""NSGA-II shared state, candidate timing, and generation logging."""

import json
import shutil
import time
from multiprocessing import Lock, Value
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.core.callback import Callback
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.robust import RobustTermination

from pydac.utils.run import is_steam_assisted_desorption_step


# Shared generation counter (set externally)
CURRENT_GEN = Value("i", 0)
# Candidate workers share the campaign stdout/log stream. Serialize their
# short progress messages so separate PID lines never interleave.
WORKER_STDOUT_LOCK = Lock()


def share_worker_state(current_gen, stdout_lock):
    """Pool initializer: use the parent's generation counter and stdout lock.

    A forked worker inherits both objects. A spawned or forkserver worker
    (Windows, macOS, Linux from Python 3.14) re-imports this module and would
    read a fresh counter at 0, so every evaluation would land in `gen_01`.
    Workers read both through this module, never through an imported name.
    """
    global CURRENT_GEN, WORKER_STDOUT_LOCK
    CURRENT_GEN = current_gen
    WORKER_STDOUT_LOCK = stdout_lock

# Column layout for the human-readable per-generation table written to
# ``generation_log.txt``. Header and rows share this spec so they stay aligned.
_GEN_LOG_COLUMNS: tuple[tuple[str, int], ...] = (
    ("gen", 3),
    ("wall[min]", 9),
    ("new", 4),
    ("feasible", 9),
    ("eval mean/med/p90 [min]", 24),
    ("thr[ev/h]", 9),
    ("delta", 10),
    ("front", 5),
    ("robust", 7),
    ("STOP", 5),
)


def _gen_log_row(cells: tuple[str, ...]) -> str:
    """Render one aligned table row for the generation log."""
    return " | ".join(
        f"{cell:>{width}}" for cell, (_, width) in zip(cells, _GEN_LOG_COLUMNS)
    )


def _gen_log_header(context: str) -> tuple[str, ...]:
    """Build the banner and column header shown once atop the generation log."""
    header = _gen_log_row(tuple(name for name, _ in _GEN_LOG_COLUMNS))
    rule = "-+-".join("-" * width for _, width in _GEN_LOG_COLUMNS)
    banner = "=" * len(header)
    return (
        "",
        banner,
        f"NSGA-II generation log  |  {context}",
        banner,
        header,
        rule,
    )


def write_evaluation_timing_summary(
    eval_folder,
    *,
    generation: int,
    evaluation_id: str,
    elapsed_wall_time_s: float,
    finished_at_unix_s: float,
    failed: bool,
    css_reached: bool,
    css_cycles_completed: int,
    error: str | None,
    solver_failure: dict | None = None,
):
    """Persist lightweight, per-candidate timing metadata for a generation."""
    payload = {
        "generation": int(generation),
        "evaluation_id": str(evaluation_id),
        "elapsed_wall_time_s": float(elapsed_wall_time_s),
        "finished_at_unix_s": float(finished_at_unix_s),
        "failed": bool(failed),
        "css_reached": bool(css_reached),
        "css_cycles_completed": int(css_cycles_completed),
        "error": error,
        "solver_failure": solver_failure,
    }
    path = Path(eval_folder) / "evaluation_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def generation_evaluation_durations(
    base_folder,
    generation: int,
    *,
    finished_after_unix_s: float | None = None,
) -> np.ndarray:
    """Load wall times written by every candidate evaluated in one generation."""
    generation_folder = Path(base_folder) / f"gen_{generation:02d}"
    durations: list[float] = []
    for path in generation_folder.glob("pid_*_eval_*/evaluation_summary.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
                elapsed = float(payload["elapsed_wall_time_s"])
                finished_at = float(payload.get("finished_at_unix_s", np.inf))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        is_current_run = (
            finished_after_unix_s is None or finished_at >= finished_after_unix_s
        )
        if is_current_run and np.isfinite(elapsed) and elapsed >= 0.0:
            durations.append(elapsed)
    return np.asarray(durations, dtype=float)


def summarize_evaluation_durations(durations) -> dict[str, float | int]:
    """Return stable summary statistics for candidate wall times in seconds."""
    values = np.asarray(durations, dtype=float)
    values = values[np.isfinite(values) & (values >= 0.0)]
    if not len(values):
        return {
            "count": 0,
            "sum_s": 0.0,
            "mean_s": np.nan,
            "median_s": np.nan,
            "p90_s": np.nan,
            "min_s": np.nan,
            "max_s": np.nan,
        }
    return {
        "count": int(len(values)),
        "sum_s": float(np.sum(values)),
        "mean_s": float(np.mean(values)),
        "median_s": float(np.median(values)),
        "p90_s": float(np.percentile(values, 90.0)),
        "min_s": float(np.min(values)),
        "max_s": float(np.max(values)),
    }


def termination_diagnostics(algorithm) -> dict[str, float | int | bool | str]:
    """Extract current-generation stop checks from the native pymoo objects."""
    termination = algorithm.termination
    criteria = (
        list(termination.terminations)
        if isinstance(termination, TerminationCollection)
        else [termination]
    )
    robust = next(
        (criterion for criterion in criteria if isinstance(criterion, RobustTermination)),
        None,
    )
    max_gen = next(
        (
            criterion
            for criterion in criteria
            if isinstance(criterion, MaximumGenerationTermination)
        ),
        None,
    )
    max_eval = next(
        (
            criterion
            for criterion in criteria
            if isinstance(criterion, MaximumFunctionCallTermination)
        ),
        None,
    )

    feasible_front = bool(np.any(algorithm.opt.get("feas")))
    diagnostics: dict[str, float | int | bool | str] = {
        "termination_mode": "robust_pareto" if robust is not None else "fixed_generations",
        "termination_feasible_front": feasible_front,
        "termination_delta_ideal": np.nan,
        "termination_delta_nadir": np.nan,
        "termination_delta_igd": np.nan,
        "termination_delta_max": np.nan,
        "termination_ftol": np.nan,
        "termination_front_check": False,
        "termination_robust_window_passes": 0,
        "termination_robust_window_size": 0,
        "termination_robust_window_required": 0,
        "termination_robust_check": False,
        "termination_max_gen_check": bool(
            max_gen is not None and max_gen.has_terminated()
        ),
        "termination_max_eval_check": bool(
            max_eval is not None and max_eval.has_terminated()
        ),
        "termination_stop": bool(termination.has_terminated()),
    }

    if robust is None:
        return diagnostics

    front = robust.termination
    delta_ideal = getattr(front, "delta_ideal", None)
    delta_nadir = getattr(front, "delta_nadir", None)
    delta_igd = getattr(front, "delta_f", None)
    deltas = np.asarray(
        [
            np.nan if delta_ideal is None else delta_ideal,
            np.nan if delta_nadir is None else delta_nadir,
            np.nan if delta_igd is None else delta_igd,
        ],
        dtype=float,
    )
    finite_deltas = deltas[np.isfinite(deltas)]
    history = list(robust.history)
    diagnostics.update(
        termination_delta_ideal=float(deltas[0]),
        termination_delta_nadir=float(deltas[1]),
        termination_delta_igd=float(deltas[2]),
        termination_delta_max=(
            float(np.max(finite_deltas)) if len(finite_deltas) else np.nan
        ),
        termination_ftol=float(front.tol),
        termination_front_check=bool(front.has_terminated()),
        termination_robust_window_passes=int(
            sum(progress >= 1.0 for progress in history)
        ),
        termination_robust_window_size=int(len(history)),
        termination_robust_window_required=int(robust.history.size),
        termination_robust_check=bool(robust.has_terminated()),
    )
    return diagnostics


def _save_population_vs_front_figure(F_pop, F_front, figures_folder, filename="population_vs_front.pdf"):
    """
    Save a lightweight population-vs-Pareto snapshot for the current generation.
    """
    import matplotlib.pyplot as plt

    if F_pop is None or F_front is None or len(F_pop) == 0 or len(F_front) == 0:
        return

    figures_folder = Path(figures_folder)
    figures_folder.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(-F_pop[:, 1], F_pop[:, 0], alpha=0.4, label="Population", color="gray")
    plt.scatter(-F_front[:, 1], F_front[:, 0], label="Pareto Front", color="red")
    plt.ylabel("Equivalent Work [MJ/kg CO2]")
    plt.xlabel("Productivity [kg CO2/m3.day]")
    plt.title("Population vs Pareto Front")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figures_folder / filename, dpi=300)
    plt.close()

# Logger callback to track and compress generations
class AllCandidatesFailed(RuntimeError):
    """Every candidate of a campaign's first generation failed."""


class NSGA2Logger(Callback):
    def __init__(
        self,
        base_folder,
        pareto_folder,
        delete=True,
        tot_gens=100,
        desorption_step="s_des",
        var_names=None,
        constraint_names=None,
        n_procs=1,
        run_started_unix_s=None,
        checkpoint_identity=None,
        initial_evaluations=0,
    ):
        super().__init__()
        self.desorption_step = desorption_step
        self.has_preheat_constraint = is_steam_assisted_desorption_step(desorption_step)
        self.var_names = tuple(var_names) if var_names is not None else None
        self.constraint_names = (
            tuple(constraint_names) if constraint_names is not None else None
        )
        self.n_procs = int(n_procs)
        if self.n_procs < 1:
            raise ValueError("`n_procs` must be at least 1.")

        self.data = {
            "gen": [],
            "n_feasible": [],
            "min_energy": [],
            "max_prod": [],
            "gen_time": [],
            "generation_wall_time_s": [],
            "n_evaluations_total": [],
            "n_evaluations_new": [],
            "evaluation_timing_records": [],
            "evaluation_wall_time_sum_s": [],
            "evaluation_wall_time_mean_s": [],
            "evaluation_wall_time_median_s": [],
            "evaluation_wall_time_p90_s": [],
            "evaluation_wall_time_min_s": [],
            "evaluation_wall_time_max_s": [],
            "generation_throughput_eval_per_hour": [],
            "termination_mode": [],
            "termination_feasible_front": [],
            "termination_delta_ideal": [],
            "termination_delta_nadir": [],
            "termination_delta_igd": [],
            "termination_delta_max": [],
            "termination_ftol": [],
            "termination_front_check": [],
            "termination_robust_window_passes": [],
            "termination_robust_window_size": [],
            "termination_robust_window_required": [],
            "termination_robust_check": [],
            "termination_max_gen_check": [],
            "termination_max_eval_check": [],
            "termination_stop": [],
            "min_purity": [],
            "max_purity": [],
            "min_oxygen_ppm": [],
            "max_oxygen_ppm": [],
        }
        
        if self.has_preheat_constraint:
            self.data["min_deltaT_preheat"] = []
            self.data["max_deltaT_preheat"] = []  

        self.base_folder = Path(base_folder)
        self.pareto_folder = Path(pareto_folder)
        self.generation_log_path = self.base_folder / "generation_log.txt"
        self.figures_folder = self.pareto_folder / "figures"
        self.figures_folder.mkdir(parents=True, exist_ok=True)
        self.delete = delete
        self.tot_gens = tot_gens
        self._start_time = time.perf_counter()
        self._run_started_unix_s = (
            time.time() if run_started_unix_s is None else float(run_started_unix_s)
        )
        self.checkpoint_identity = (
            None if checkpoint_identity is None else dict(checkpoint_identity)
        )
        self._previous_n_eval = int(initial_evaluations)
        self._all_evaluation_durations: list[float] = []
        self._generation_log_started = False

    def timing_summary(self) -> dict[str, float | int]:
        """Aggregate candidate timing statistics over completed generations."""
        summary = summarize_evaluation_durations(self._all_evaluation_durations)
        summary["n_workers"] = self.n_procs
        return summary

    def _write_generation_log(self, *lines: str) -> None:
        """Append compact per-generation progress and termination telemetry."""
        with self.generation_log_path.open("a", buffering=1, encoding="utf-8") as handle:
            for line in lines:
                handle.write(f"{line}\n")
        
    def __call__(self, algorithm):
        gen = algorithm.n_gen
        CURRENT_GEN.value = gen

        now = time.perf_counter()
        elapsed = now - self._start_time
        self._start_time = now

        pop = algorithm.pop
        F = pop.get("F")
        X = pop.get("X")
        G = pop.get("G")
        extras = pop.get("Extra")  # List of dicts with additional info

        feasible = np.asarray(pop.get("feas"), dtype=bool).reshape(-1)
        rank = np.asarray(pop.get("rank")).reshape(-1)
        is_nd = feasible & (rank == 0)
        n_eval_total = int(algorithm.evaluator.n_eval)
        n_eval_new = n_eval_total - self._previous_n_eval
        self._previous_n_eval = n_eval_total
        evaluation_durations = generation_evaluation_durations(
            self.base_folder,
            gen,
            finished_after_unix_s=self._run_started_unix_s,
        )
        evaluation_timing = summarize_evaluation_durations(evaluation_durations)
        self._all_evaluation_durations.extend(evaluation_durations.tolist())
        throughput = (
            3600.0 * n_eval_new / elapsed if elapsed > 0.0 else np.nan
        )
        termination = termination_diagnostics(algorithm)

        def _extra_values(key):
            values = []
            for extra in extras:
                if isinstance(extra, dict):
                    values.append(extra.get(key, np.nan))
                else:
                    values.append(np.nan)
            return np.asarray(values, dtype=float)

        def _finite_min(values):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            return float(np.min(values)) if len(values) else np.nan

        def _finite_max(values):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            return float(np.max(values)) if len(values) else np.nan

        self.data["gen"].append(gen)
        self.data["gen_time"].append(elapsed)
        self.data["generation_wall_time_s"].append(elapsed)
        self.data["n_evaluations_total"].append(n_eval_total)
        self.data["n_evaluations_new"].append(n_eval_new)
        self.data["evaluation_timing_records"].append(evaluation_timing["count"])
        self.data["evaluation_wall_time_sum_s"].append(evaluation_timing["sum_s"])
        self.data["evaluation_wall_time_mean_s"].append(evaluation_timing["mean_s"])
        self.data["evaluation_wall_time_median_s"].append(evaluation_timing["median_s"])
        self.data["evaluation_wall_time_p90_s"].append(evaluation_timing["p90_s"])
        self.data["evaluation_wall_time_min_s"].append(evaluation_timing["min_s"])
        self.data["evaluation_wall_time_max_s"].append(evaluation_timing["max_s"])
        self.data["generation_throughput_eval_per_hour"].append(throughput)
        for key in (
            "termination_mode",
            "termination_feasible_front",
            "termination_delta_ideal",
            "termination_delta_nadir",
            "termination_delta_igd",
            "termination_delta_max",
            "termination_ftol",
            "termination_front_check",
            "termination_robust_window_passes",
            "termination_robust_window_size",
            "termination_robust_window_required",
            "termination_robust_check",
            "termination_max_gen_check",
            "termination_max_eval_check",
            "termination_stop",
        ):
            self.data[key].append(termination[key])
        self.data["n_feasible"].append(int(np.count_nonzero(feasible)))
        feasible_F = F[feasible]
        self.data["min_energy"].append(
            _finite_min(feasible_F[:, 0]) if len(feasible_F) else np.nan
        )
        self.data["max_prod"].append(
            _finite_max(-feasible_F[:, 1]) if len(feasible_F) else np.nan
        )
        purity = _extra_values("co2_pur")
        oxygen = _extra_values("o2_ppm")
        self.data["min_purity"].append(_finite_min(purity[feasible]))
        self.data["max_purity"].append(_finite_max(purity[feasible]))
        self.data["min_oxygen_ppm"].append(_finite_min(oxygen[feasible]))
        self.data["max_oxygen_ppm"].append(_finite_max(oxygen[feasible]))
        if self.has_preheat_constraint:
            delta_t = _extra_values("deltaT_preheat")
            self.data["max_deltaT_preheat"].append(_finite_max(delta_t[feasible]))
            self.data["min_deltaT_preheat"].append(_finite_min(delta_t[feasible]))
        
        eval_cell = (
            f"{evaluation_timing['mean_s'] / 60:.2f} / "
            f"{evaluation_timing['median_s'] / 60:.2f} / "
            f"{evaluation_timing['p90_s'] / 60:.2f}"
            if evaluation_timing["count"]
            else "n/a"
        )
        throughput_cell = (
            f"{throughput:.1f}" if np.isfinite(throughput) else "n/a"
        )
        if termination["termination_mode"] == "robust_pareto":
            delta_cell = (
                f"{termination['termination_delta_max']:.4g}"
                if np.isfinite(termination["termination_delta_max"])
                else "n/a"
            )
            front_cell = (
                "yes" if termination["termination_front_check"] else "no"
            )
            robust_cell = (
                f"{int(termination['termination_robust_window_passes'])}/"
                f"{int(termination['termination_robust_window_required'])}"
            )
            context = (
                f"mode=robust_pareto  ftol={termination['termination_ftol']:.4g}  "
                f"robust_window={int(termination['termination_robust_window_required'])}"
            )
        else:
            delta_cell = "-"
            front_cell = "-"
            robust_cell = "-"
            context = "mode=fixed_generations"
        row = _gen_log_row(
            (
                f"{gen:02d}",
                f"{elapsed / 60:.2f}",
                f"{n_eval_new}",
                f"{int(np.count_nonzero(feasible))}/{len(feasible)}",
                eval_cell,
                throughput_cell,
                delta_cell,
                front_cell,
                robust_cell,
                "STOP" if termination["termination_stop"] else "-",
            )
        )
        if not self._generation_log_started:
            self._write_generation_log(*_gen_log_header(context))
            self._generation_log_started = True
        self._write_generation_log(row)
        
        # === CSV export ===
        all_rows = []
        for i in np.where(is_nd)[0]:
            row = {
                "gen": gen,
                "eq_work": F[i, 0],
                "productivity": -F[i, 1],
            }
            for j, val in enumerate(X[i]):
                row[f"x{j}"] = val
                if self.var_names is not None and j < len(self.var_names):
                    row[self.var_names[j]] = val

            if isinstance(extras[i], dict):
                row.update(extras[i])
            all_rows.append(row)

        # Save generation CSV
        df = pd.DataFrame(all_rows)
        df.to_csv(self.pareto_folder / f"population_and_extras.csv", index=False)

        if not np.any(feasible):
            cv = np.asarray(pop.get("CV"), dtype=float).reshape(-1)
            best = np.flatnonzero(cv == np.nanmin(cv))
            np.savez(
                self.pareto_folder / "best_infeasible_population.npz",
                X=X[best],
                F=F[best],
                G=G[best],
                CV=cv[best],
            )

        # Stage the generation-scoped JSONL. It becomes committed only when
        # save_generation_checkpoint writes checkpoint_state.json below.
        from pydac.utils.nsga2.evaluation import commit_generation_candidate_records

        records_path, records_count = commit_generation_candidate_records(
            self.base_folder,
            gen,
            expected_count=n_eval_new,
        )
        print(
            f"[ARTIFACTS] Staged {records_count} candidate records at "
            f"{records_path}."
        )
        if gen == 1:
            self._stop_if_every_candidate_failed(extras)

        try:
            _save_population_vs_front_figure(
                F_pop=F,
                F_front=F[is_nd],
                figures_folder=self.figures_folder,
                filename="population_vs_front.pdf",
            )
        except Exception as exc:
            print(f"[WARN] Could not update population-vs-front figure: {exc}")

        # Save log dataframe
        df_log = pd.DataFrame(self.data)
        df_log.to_csv(self.base_folder / "optimization_log.csv", index=False)

        # Keep the latest generation even when convergence stops before the
        # hard generation cap; delete only the preceding generation.
        if self.delete and gen > 1:
            self.delete_generation(gen - 1)

        if self.checkpoint_identity is None:
            raise RuntimeError("NSGA2Logger requires a checkpoint campaign identity.")
        from pydac.utils.nsga2.checkpoint import save_generation_checkpoint

        checkpoint = save_generation_checkpoint(
            self.base_folder,
            algorithm,
            identity=self.checkpoint_identity,
            var_names=self.var_names or (),
            constraint_names=self.constraint_names or (),
            candidate_records_path=records_path,
            candidate_records_count=records_count,
        )
        print(
            "[CHECKPOINT] Committed generation "
            f"{checkpoint['completed_generations']} at "
            f"{self.pareto_folder / 'checkpoint_state.json'}."
        )

    def _stop_if_every_candidate_failed(self, extras) -> None:
        """A first generation without one successful candidate ends the campaign.

        Checked before the generation is committed, so nothing is checkpointed
        and the same campaign may start again once the cause is fixed.
        """
        failed = [isinstance(extra, dict) and bool(extra.get("failed")) for extra in extras]
        if not failed or not all(failed):
            return
        from pydac.utils.nsga2.evaluation import (
            campaign_evaluation_summary,
            format_evaluation_summary,
        )

        summary = format_evaluation_summary(campaign_evaluation_summary(self.base_folder))
        raise AllCandidatesFailed(
            f"Every candidate of generation 1 failed ({len(failed)} of {len(failed)}); the "
            f"campaign stops instead of continuing on an empty front.\n{summary}\n"
            f"Records: {self.pareto_folder / 'candidate_records' / 'generation_0001.jsonl'}"
        )

    def delete_generation(self, gen_number):
        gen_folder = self.base_folder / f"gen_{gen_number:02d}"
        if gen_folder.exists():
            try:
                shutil.rmtree(gen_folder)
                #print(f"[CLEANUP] Deleted full generation folder: {gen_folder.name}")
            except Exception as e:
                print(f"[ERROR] Could not delete {gen_folder.name}: {e}")
        else:
            print(f"[WARN] Generation folder not found: {gen_folder}")
                
    def delete_eval_folder(self, folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"[DEL] Deleted {folder_path.name}")
        except Exception as e:
            print(f"[ERROR] Could not delete {folder_path.name}: {e}")

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()
