"""
pyDAC - NSGA-II optimization orchestrator.

Reusable multi-objective optimization API that wraps the shared CSS runner
(`workflows.orchestrator_TVSA_cycle_css.run_css_simulation`) in a pymoo
ElementwiseProblem and a multiprocessing-friendly driver.

This module contains no operational campaign setup and no `__main__` block.
Campaign-specific settings (bounds, population, generations, CPU count,
warm-start, cycle definition) live in `runners/run_NSGA2.py`.

Canonical decision vector (steam-assisted desorption):

    [t_ads, t_heat, t_des, P_vac, v_feed, v_steam]

Canonical decision vector (wall-heated desorption):

    [t_ads, t_heat, t_des, P_vac, v_feed]

Before each CSS evaluation, superficial velocities are converted to the model
inputs expected by the shared runner:

    vol_flowrate = v_feed * A(tau_ar, bed_volume)
    m_dot_steam = v_steam * A(tau_ar, bed_volume) * rho_steam(P_vac, T_feed)

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-20
"""

from __future__ import annotations

import os
import sys


import contextlib
import fnmatch
import json
import multiprocessing as mp
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydac.config import (
    ExperimentConfig,
    default_experiment_config,
    experiment_to_document,
    load_experiment,
    operating_step_settings,
    write_resolved_config,
)
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.robust import RobustTermination
from pydac.post_processing.metrics.cycle_data_loader import CycleDataLoader
from pydac.post_processing.kpi.cycle_kpis import CycleKPIs, LedgerMetricMissing
from pydac.inputs.case_io import load_case
from pydac.config.errors import ConfigurationError, RunRefused
from pydac.utils.nsga2.artifacts import (
    save_flow_transform_metadata as _save_flow_transform_metadata,
    save_pareto_bounds as _save_pareto_bounds,
)
from pydac.utils.nsga2.checkpoint import (
    CHECKPOINT_STATE_FILENAME,
    atomic_savez,
    campaign_identity,
    checkpoint_termination_met,
    discard_uncommitted_checkpoint_artifacts,
    load_checkpoint_population,
    load_checkpoint_state,
    restore_algorithm_from_checkpoint,
)
from pydac.utils.nsga2.decisions import (
    decision_vector_to_model_inputs,
    resolve_var_names as _resolve_var_names,
    validate_var_names as _validate_var_names,
)
from pydac.utils.nsga2 import logging as nsga2_shared
from pydac.utils.nsga2.evaluation import (
    campaign_evaluation_summary,
    discard_transient_generation_folders,
    format_evaluation_summary,
    extract_candidate_kpis as _extract_candidate_kpis,
    format_candidate_summary as _format_candidate_summary,
    make_eval_folder as _make_eval_folder,
    prune_candidate_artifacts,
    write_candidate_record,
)
from pydac.utils.nsga2 import (
    CURRENT_GEN,
    NSGA2Logger,
    Tee,
    WORKER_STDOUT_LOCK,
    WarmStartSpec,
    build_initial_population,
    write_evaluation_timing_summary,
)
from pydac.utils.run import (
    infer_desorption_step,
    is_steam_assisted_desorption_step,
)
from pydac.utils.campaign import atomic_write_json
from pydac.utils.solver.outcome import FinalSolverFailure
from pydac.utils.solver.composition_domain import CompositionDomainGuard
from pydac.utils.solver.persistence import MOO_COMPACT, RUN_FULL
from pydac.utils.postprocessing.terminal_state import TERMINAL_STATE_SCHEMA_VERSION
from pydac.utils.tvsa.cycle_steps import DEFAULT_CYCLIC_SEQUENCE
from pydac.workflows.orchestrator_TVSA_cycle_css import run_css_simulation
# Penalty values returned by `evaluate_pydac_candidate` when a candidate fails.
_FAILED_OBJECTIVE: float = 1e6
_FAILED_CONSTRAINT: float = 1e6
_VALID_EVALUATION_CONSTRAINT: str = "valid_evaluation_and_css"
@dataclass
class NSGA2Constraints:
    """
    Toggle + threshold container for NSGA-II constraints. Each entry is
    evaluated as `g <= 0 ⇔ feasible`. Setting a numeric field to `None`
    (or a bool field to `False`) disables the corresponding constraint.

    Attributes
    ----------
    purity_min : float | None
        Lower bound on CO2 purity. `g = purity_min - purity`.
        `None` disables the constraint (default).
    oxygen_ppm_max : float | None
        Upper bound on O2 concentration in ppm.
        `g = oxygen_ppm - oxygen_ppm_max`. `None` disables (default).
    steam_condensation : bool
        If True and the cycle is steam-assisted, add
        `g = T_steam_condensation - T_bed_min_preheat`. Silently ignored
        when the cycle is wall-heated. Default False.
    eq_work_max : float | None
        Optional upper bound on equivalent work in MJ/kg CO2.
        `g = eq_work - eq_work_max`. `None` disables. Used by the
        refinement runner to tighten the Pareto set.
    """

    purity_min: float | None = None
    oxygen_ppm_max: float | None = None
    steam_condensation: bool = False
    eq_work_max: float | None = None

    def active_names(self, *, is_steam_assisted: bool) -> list[str]:
        """Ordered list of currently-active constraint names."""
        names: list[str] = []
        if self.purity_min is not None:
            names.append("purity_min")
        if self.oxygen_ppm_max is not None:
            names.append("oxygen_ppm_max")
        if self.steam_condensation and is_steam_assisted:
            names.append("steam_condensation")
        if self.eq_work_max is not None:
            names.append("eq_work_max")
        return names

    def count(self, *, is_steam_assisted: bool) -> int:
        return len(self.active_names(is_steam_assisted=is_steam_assisted))

    def build_g(self, result: Mapping[str, Any], *, is_steam_assisted: bool) -> list[float]:
        """
        Build the ordered list of g-values for a single evaluation `result`.
        Order matches `active_names(...)`.
        """
        g: list[float] = []
        if self.purity_min is not None:
            g.append(float(self.purity_min - result["purity"]))
        if self.oxygen_ppm_max is not None:
            g.append(float(result["oxygen_ppm"] - self.oxygen_ppm_max))
        if self.steam_condensation and is_steam_assisted:
            g.append(
                float(result["T_steam_condensation"] - result["T_bed_min_preheat"])
            )
        if self.eq_work_max is not None:
            g.append(float(result["eq_work"] - self.eq_work_max))
        return g

@dataclass(frozen=True)
class NSGA2TerminationConfig:
    """Termination controls built from pymoo's native termination classes.

    ``mode="fixed_generations"`` preserves the historical generation-only
    behavior. ``mode="robust_pareto"`` stops when the feasible Pareto front is
    stable in objective space over a robust sliding window. In both modes,
    ``n_generations`` passed to :func:`run_nsga2_optimization` remains the hard
    maximum-generation cap.
    """

    mode: Literal["fixed_generations", "robust_pareto"] = "robust_pareto"
    ftol: float = 5e-3
    period: int = 10
    n_skip: int = 0
    max_evaluations: int | None = None

    def __post_init__(self):
        if self.mode not in {"fixed_generations", "robust_pareto"}:
            raise ValueError(
                f"Unsupported termination mode '{self.mode}'. Expected "
                "'fixed_generations' or 'robust_pareto'."
            )
        if self.ftol < 0.0:
            raise ValueError("`ftol` must be non-negative.")
        if self.period < 1:
            raise ValueError("`period` must be at least 1.")
        if self.n_skip < 0:
            raise ValueError("`n_skip` must be non-negative.")
        if self.max_evaluations is not None and self.max_evaluations < 1:
            raise ValueError("`max_evaluations` must be at least 1 when provided.")


@dataclass
class NSGA2RunConfig:
    """
    Operational configuration for a single NSGA-II campaign.

    Attributes
    ----------
    base_folder : str
        Root folder for this campaign. Generation subfolders and paretos/
        output are created beneath it.
    bounds : Mapping[str, tuple[float, float]]
        Lower/upper bounds per decision variable. Must include all entries in
        the canonical decision vector for the selected cycle; `v_steam` is
        required only for steam-assisted desorption. Flow bounds are expressed
        as superficial velocities [m/s], not absolute volumetric/mass flows.
    decision_variables : Sequence[str] | None
        The decision vector this campaign optimizes, in order. `None` keeps
        the canonical vector implied by the cycle, which is what every
        campaign written before optional variables existed gets. Stating it
        is how a campaign opts into an optional variable such as
        `T_cool_target`; the value is validated against the shapes the
        evaluation path can actually map.
    experiment : ExperimentConfig | None
        The campaign's base case: geometry, bed, grid, cycle steps, CSS stop
        criteria, and outputs. Every candidate is this document plus its own
        operating knobs, so `tau_ar`, `bed_volume`, `bed_config`,
        `cycle_steps`, and `require_css` are read from it rather than being
        stated twice. `None` uses the reference profile.
    constraints : NSGA2Constraints
        Active constraints container. Defaults to no product constraint;
        the evaluation-validity/CSS constraint is always active.
    sbx_prob : float
        Crossover probability for SBX. Default 0.9.
    sbx_eta : float
        Distribution index for SBX crossover. Default 15.
    pm_eta : float
        Distribution index for Polynomial Mutation (PM). Default 20.
    """

    base_folder: str
    bounds: Mapping[str, tuple[float, float]]
    decision_variables: Sequence[str] | None = None
    experiment: ExperimentConfig | None = None
    constraints: NSGA2Constraints = field(default_factory=NSGA2Constraints)
    sbx_prob: float = 0.9
    sbx_eta: float = 15.0
    pm_eta: float = 20.0
    composition_domain_guard: CompositionDomainGuard = field(
        default_factory=CompositionDomainGuard
    )

    def __getstate__(self) -> dict[str, Any]:
        """Serialize immutable experiment mappings for multiprocessing."""
        state = dict(self.__dict__)
        state["experiment"] = experiment_to_document(self.experiment)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Rebuild the exact typed experiment inside a worker process."""
        restored = dict(state)
        restored["experiment"] = load_case(restored["experiment"])
        restored.setdefault("composition_domain_guard", CompositionDomainGuard())
        self.__dict__.update(restored)

    def __post_init__(self):
        # Everything every candidate shares is the campaign's base experiment:
        # the cycle, its steps, the CSS criteria, the geometry, the bed. A
        # candidate only states its own operating knobs on top of it.
        if self.experiment is None:
            self.experiment = default_experiment_config()
        if not isinstance(self.experiment, ExperimentConfig):
            raise TypeError("`experiment` must be an ExperimentConfig instance.")
        if not isinstance(self.composition_domain_guard, CompositionDomainGuard):
            raise TypeError(
                "`composition_domain_guard` must be a CompositionDomainGuard."
            )
        geometry = self.experiment.simulation.geometry
        execution = self.experiment.execution
        self.tau_ar = geometry.aspect_ratio
        self.bed_volume = geometry.volume_m3
        self.bounds = dict(self.bounds)
        self.cycle_steps = list(
            execution.cycle_steps
            if execution.cycle_steps is not None
            else DEFAULT_CYCLIC_SEQUENCE
        )
        self.bed_config = self.experiment.simulation.bed.to_mapping()
        self.require_css = execution.require_css
        # The feed condition of a campaign is a property of its adsorption
        # step; keeping it visible here is what the manifest and the banner
        # report, but the document remains the only place it is stated.
        adsorption = {step.name: step.config for step in self.experiment.simulation.cycle.steps}
        self.adsorption_T_feed = adsorption["ads"].thermal.feed_temperature_K
        self.adsorption_RH = adsorption["ads"].composition.relative_humidity
        self.desorption_step = infer_desorption_step(cycle_steps=self.cycle_steps)
        self.is_steam_assisted = is_steam_assisted_desorption_step(self.desorption_step)
        self.var_names = (
            _resolve_var_names(self.is_steam_assisted)
            if self.decision_variables is None
            else _validate_var_names(
                self.decision_variables,
                is_steam_assisted=self.is_steam_assisted,
            )
        )
        missing = [k for k in self.var_names if k not in self.bounds]
        if missing:
            raise ValueError(
                f"Missing bounds for decision variables: {missing}. "
                f"Required order: {list(self.var_names)}."
            )


def candidate_settings(operating_settings: Mapping[str, Any]) -> dict[str, Any]:
    """The settings that turn the campaign's base case into one candidate."""
    return {
        **operating_step_settings(operating_settings),
        "execution.solve": True,
        "execution.print_to_terminal": False,
        "execution.print_cycle_build": False,
        "execution.plot_profiles": False,
    }


def preflight_search_space(run_config: "NSGA2RunConfig") -> None:
    """Resolve the candidate case at the lower and the upper bound corner.

    A base case that disagrees with the search space (for example a startup
    ``event_max_time_s`` shorter than the ``t_ads`` bounds) would otherwise
    fail every candidate only once the campaign runs. Both corners are built
    exactly as an evaluation builds its case, without solving anything. A
    coupling that fails only inside the box is not caught here.
    """
    for label, index in (("lower", 0), ("upper", 1)):
        vector = [float(run_config.bounds[name][index]) for name in run_config.var_names]
        try:
            operating_settings, _ = decision_vector_to_model_inputs(
                vector,
                is_steam_assisted=run_config.is_steam_assisted,
                tau_ar=run_config.tau_ar,
                bed_volume=run_config.bed_volume,
                var_names=run_config.var_names,
            )
            load_experiment(run_config.experiment, settings=candidate_settings(operating_settings))
        except (TypeError, ValueError) as exc:
            point = ", ".join(f"{name}={value:g}" for name, value in zip(run_config.var_names, vector))
            raise ConfigurationError(
                f"Candidates at the {label} bounds ({point}) cannot run: {exc} "
                "Adjust the bounds or the base case."
            ) from exc


def evaluate_pydac_candidate(
    decision_vector: Sequence[float],
    *,
    run_config: NSGA2RunConfig,
    eval_folder: str | None = None,
    keep_trajectories: bool = False,
) -> dict:
    """
    Evaluate a single NSGA-II candidate by running a full CSS simulation
    through the shared orchestrator, then computing KPIs from the CSS cycle.

    Returns a dict with keys:
        eq_work, neg_productivity, purity, oxygen_ppm,
        T_bed_min_preheat, T_steam_condensation, energy_breakdown,
        cyclic_working_capacity, css_details, failed, error, solver_failure,
        solver_outcomes

    Ordinary optimization candidates retain only compact JSON metadata after
    evaluation. ``keep_trajectories=True`` is reserved for explicit debug or
    final-point re-evaluations; even then, CSS storage is bounded to two cycles.
    """
    evaluation_start_time = time.perf_counter()
    persistence_policy = RUN_FULL if keep_trajectories else MOO_COMPACT
    is_steam_assisted = run_config.is_steam_assisted
    operating_settings, flow_inputs = decision_vector_to_model_inputs(
        decision_vector,
        is_steam_assisted=is_steam_assisted,
        tau_ar=run_config.tau_ar,
        bed_volume=run_config.bed_volume,
        var_names=run_config.var_names,
    )

    if eval_folder is None:
        eval_folder, gen_id, eval_id = _make_eval_folder(run_config.base_folder)
    else:
        os.makedirs(eval_folder, exist_ok=True)
        gen_id = nsga2_shared.CURRENT_GEN.value + 1
        eval_id = Path(eval_folder).name

    summary = _format_candidate_summary(
        operating_settings,
        is_steam_assisted=is_steam_assisted,
        flow_inputs=flow_inputs,
    )
    # Per-candidate progress lives in a dedicated per-grid-point file so the
    # campaign log (`nsga2_log.txt`) stays readable. Appends
    # are atomic per line across worker processes on POSIX (O_APPEND).
    evaluation_line = (
        f"[PID {os.getpid()} | Gen {gen_id} | Eval {eval_id}] "
        f"Evaluating: {summary}\n"
    )
    evaluations_log_path = os.path.join(
        run_config.base_folder, "evaluations_log.txt"
    )
    with nsga2_shared.WORKER_STDOUT_LOCK:
        with open(evaluations_log_path, "a", encoding="utf-8") as eval_log_handle:
            eval_log_handle.write(evaluation_line)

    log_path = os.path.join(eval_folder, "eval_log.txt")
    from pydac.utils.provenance import new_record, source_record, read_record
    candidate_experiment = None
    candidate_sources = []
    failed = False
    error_msg: str | None = None
    result: dict[str, Any] = {
        "eq_work": _FAILED_OBJECTIVE,
        "neg_productivity": _FAILED_OBJECTIVE,
        "purity": 0.0,
        "oxygen_ppm": _FAILED_CONSTRAINT,
        "T_bed_min_preheat": 0.0,
        "T_steam_condensation": 0.0,
        "energy_breakdown": {},
        "cyclic_working_capacity": 0.0,
        "flow_inputs": dict(flow_inputs),
        "css_reached": False,
        "css_cycles_completed": 0,
        "css_history": [],
        "evaluation_wall_time_s": 0.0,
        "failed": True,
        "error": None,
        "solver_failure": None,
        "solver_outcomes": [],
        "persistence_policy": persistence_policy,
    }

    with open(log_path, "w", buffering=1, encoding="utf-8") as log_file_handle, \
            contextlib.redirect_stdout(log_file_handle), \
            contextlib.redirect_stderr(log_file_handle):
        try:
            print(
                f"Starting eval {eval_id}: tau_ar={run_config.tau_ar}, "
                f"persistence={persistence_policy}, {summary}",
                flush=True,
            )

            # This candidate is the campaign's base case plus its own
            # operating knobs, resolved through the public loader. The
            # document is saved next to the evaluation, so every point of the
            # front can be re-run from what is on disk.
            candidate_experiment = load_experiment(
                run_config.experiment,
                settings=candidate_settings(operating_settings),
            )
            write_resolved_config(
                candidate_experiment, os.path.join(eval_folder, "resolved_config.json")
            )
            css_details = run_css_simulation(
                experiment_config=candidate_experiment,
                base_folder=eval_folder,
                retain_cycles=2,
                composition_domain_guard=run_config.composition_domain_guard,
                persistence_policy=persistence_policy,
            )
            if not isinstance(css_details, dict):
                raise RuntimeError(
                    "run_css_simulation() did not return details."
                )
            result["css_reached"] = bool(css_details["css_reached"])
            result["css_cycles_completed"] = int(css_details["cycles_completed"])
            result["css_history"] = list(css_details.get("css_history", []))
            result["solver_outcomes"] = [
                dict(record["solver"])
                for record in css_details.get("last_step_records", [])
                if isinstance(record, Mapping)
                and isinstance(record.get("solver"), Mapping)
            ]
            if run_config.require_css and not result["css_reached"]:
                raise RuntimeError(
                    "Candidate reached max_cycles without cyclic steady state."
                )

            cycle_data = CycleDataLoader(
                css_details["results_base_folder"],
                cycle_index=css_details["css_cycle_idx"],
            ).load()
            cycle_kpis = CycleKPIs(cycle_data)
            candidate_sources = getattr(cycle_data, "provenance", None) or []
            candidate_kpis = _extract_candidate_kpis(cycle_kpis)

            eq_work = candidate_kpis["eq_work"]
            productivity = candidate_kpis["productivity"]
            purity = candidate_kpis["purity"]
            oxygen_ppm = candidate_kpis["oxygen_ppm"]

            result.update(
                eq_work=eq_work,
                neg_productivity=float(-productivity),
                purity=purity,
                oxygen_ppm=oxygen_ppm,
                T_bed_min_preheat=candidate_kpis["T_bed_min_preheat"],
                T_steam_condensation=candidate_kpis[
                    "T_steam_condensation"
                ],
                energy_breakdown=candidate_kpis["energy_breakdown"],
                cyclic_working_capacity=candidate_kpis[
                    "cyclic_working_capacity"
                ],
                failed=False,
                error=None,
            )

            print(
                f"=== Finished eval {eval_id}: "
                f"Energy={eq_work:.3f} MJ/kg, "
                f"Prod={productivity:.3f} kg/m3day, "
                f"Purity={purity:.4f}, "
                f"O2={oxygen_ppm:.2f} ppm ===",
                flush=True,
            )
        except LedgerMetricMissing:
            # Deliberately not swallowed.  The condenser split is absent or present for
            # the whole campaign, never for one candidate, so demoting this to a failed
            # evaluation would spend the entire budget producing an old-ledger front
            # indistinguishable from a physics failure.  Let it kill the pool.
            traceback.print_exc()
            raise
        except Exception as exc:  # noqa: BLE001 — evaluator must never crash the pool
            failed = True
            error_msg = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, FinalSolverFailure):
                result["solver_failure"] = dict(exc.summary)
                result["solver_outcomes"] = [
                    {"success": False, **result["solver_failure"]}
                ]
            traceback.print_exc()
            print(f"[EVAL FAILED] {error_msg}", flush=True)

    if failed:
        result["failed"] = True
        result["error"] = error_msg

    result["evaluation_wall_time_s"] = time.perf_counter() - evaluation_start_time
    evaluation_is_valid = bool(
        not result["failed"]
        and (result["css_reached"] or not run_config.require_css)
    )
    constraint_names = (
        _VALID_EVALUATION_CONSTRAINT,
        *run_config.constraints.active_names(
            is_steam_assisted=run_config.is_steam_assisted
        ),
    )
    constraint_values = [
        -1.0 if evaluation_is_valid else _FAILED_CONSTRAINT,
        *run_config.constraints.build_g(
            result,
            is_steam_assisted=run_config.is_steam_assisted,
        ),
    ]
    candidate_record = {
        "schema_version": 2,
        "generation": int(gen_id),
        "evaluation_id": str(eval_id),
        "decision_vector": {
            name: float(value)
            for name, value in zip(run_config.var_names, decision_vector)
        },
        "objectives": {
            "eq_work_MJ_per_kgCO2": float(result["eq_work"]),
            "neg_productivity_kgCO2_per_m3_day": float(
                result["neg_productivity"]
            ),
        },
        "constraints": {
            name: float(value)
            for name, value in zip(constraint_names, constraint_values)
        },
        "metrics": {
            "purity_wet": float(result["purity"]),
            "oxygen_ppm": float(result["oxygen_ppm"]),
            "T_bed_min_preheat_K": float(result["T_bed_min_preheat"]),
            "T_steam_condensation_K": float(result["T_steam_condensation"]),
            "cyclic_working_capacity": float(result["cyclic_working_capacity"]),
            "energy_breakdown": dict(result["energy_breakdown"]),
            "flow_inputs": dict(result["flow_inputs"]),
        },
        "css": {
            "reached": bool(result["css_reached"]),
            "cycles_completed": int(result["css_cycles_completed"]),
            "history": list(result["css_history"]),
        },
        "wall_time_s": float(result["evaluation_wall_time_s"]),
        "solver_outcomes": list(result["solver_outcomes"]),
        "persistence": {
            "policy": persistence_policy,
            "full_trajectories": persistence_policy == RUN_FULL,
            "terminal_state_schema_version": (
                TERMINAL_STATE_SCHEMA_VERSION
                if persistence_policy == MOO_COMPACT
                else None
            ),
        },
        "failure": (
            None
            if not result["failed"]
            else {
                "error": result["error"],
                "solver": result["solver_failure"],
            }
        ),
    }
    if not candidate_sources:
        # Preserve failed/unfinished solve attribution before compact pruning.
        # CSS already bounds the number of retained cycle folders.
        for path in sorted(Path(eval_folder).glob("**/provenance/*.json")):
            record = read_record(path)
            if record["activity"] == "solve":
                candidate_sources.append(source_record(record))
    candidate_record["provenance"] = new_record(
        "optimization", config=candidate_experiment,
        execution={"activity_scope": "candidate", "decision_variables": dict(operating_settings),
                   "generation": int(gen_id), "failed": bool(result["failed"]),
                   "persistence_policy": persistence_policy},
        sources=candidate_sources,
    )
    write_candidate_record(eval_folder, candidate_record)
    try:
        write_evaluation_timing_summary(
            eval_folder,
            generation=gen_id,
            evaluation_id=eval_id,
            elapsed_wall_time_s=result["evaluation_wall_time_s"],
            finished_at_unix_s=time.time(),
            failed=result["failed"],
            css_reached=result["css_reached"],
            css_cycles_completed=result["css_cycles_completed"],
            error=result["error"],
            solver_failure=result["solver_failure"],
        )
    except OSError as exc:
        with nsga2_shared.WORKER_STDOUT_LOCK:
            print(
                f"[WARN] Could not write timing summary for eval {eval_id}: {exc}",
                flush=True,
            )

    prune_candidate_artifacts(
        eval_folder,
        keep_trajectories=keep_trajectories,
    )

    return result


class TVSAOptimizationProblem(ElementwiseProblem):
    """
    pymoo ElementwiseProblem that evaluates a single TVSA decision vector via
    `evaluate_pydac_candidate`.

    Objectives:
        F[0] = eq_work            (min)
        F[1] = -productivity      (min; maximizes productivity)

    Constraints (g <= 0 is feasible) — active set is driven by
    `run_config.constraints` (`NSGA2Constraints`), see that class for the
    per-constraint formulas.
    """

    def __init__(self, run_config: NSGA2RunConfig, **kwargs):
        self.run_config = run_config
        xl = np.array([run_config.bounds[v][0] for v in run_config.var_names])
        xu = np.array([run_config.bounds[v][1] for v in run_config.var_names])
        n_physical_constraints = run_config.constraints.count(
            is_steam_assisted=run_config.is_steam_assisted
        )
        self.constraint_names = (
            _VALID_EVALUATION_CONSTRAINT,
            *run_config.constraints.active_names(
                is_steam_assisted=run_config.is_steam_assisted
            ),
        )
        super().__init__(
            n_var=len(run_config.var_names),
            n_obj=2,
            n_constr=1 + n_physical_constraints,
            xl=xl,
            xu=xu,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        result = evaluate_pydac_candidate(x, run_config=self.run_config)

        out["F"] = np.array([result["eq_work"], result["neg_productivity"]])

        evaluation_is_valid = bool(
            not result["failed"]
            and (result["css_reached"] or not self.run_config.require_css)
        )
        g_values = [-1.0 if evaluation_is_valid else _FAILED_CONSTRAINT]
        g_values.extend(self.run_config.constraints.build_g(
            result, is_steam_assisted=self.run_config.is_steam_assisted
        ))
        out["G"] = np.array(g_values)

        extra: dict[str, Any] = {
            "co2_pur": result["purity"],
            "o2_ppm": result["oxygen_ppm"],
            "cwc": result["cyclic_working_capacity"],
            "failed": result["failed"],
            "css_reached": result["css_reached"],
            "css_cycles_completed": result["css_cycles_completed"],
            "evaluation_wall_time_s": result.get("evaluation_wall_time_s", np.nan),
            "deltaT_preheat": (
                result["T_bed_min_preheat"] - result["T_steam_condensation"]
            ),
        }
        extra.update(result["flow_inputs"])
        if result["error"] is not None:
            extra["error"] = result["error"]
        if result["solver_failure"] is not None:
            extra["solver_failure"] = result["solver_failure"]
        extra.update(result["energy_breakdown"])
        out["Extra"] = extra


def build_nsga2_termination(
    *,
    n_generations: int,
    config: NSGA2TerminationConfig,
):
    """Build the native pymoo stop chain.

    The resulting logic is ``robust feasible-front convergence OR hard caps``.
    No project-specific convergence metric is introduced here.
    """
    if n_generations < 1:
        raise ValueError("`n_generations` must be at least 1.")

    criteria = []
    if config.mode == "robust_pareto":
        front_space = MultiObjectiveSpaceTermination(
            tol=config.ftol,
            only_feas=True,
            n_skip=config.n_skip,
        )
        criteria.append(RobustTermination(front_space, period=config.period))

    criteria.append(MaximumGenerationTermination(n_generations))
    if config.max_evaluations is not None:
        criteria.append(MaximumFunctionCallTermination(config.max_evaluations))

    if len(criteria) == 1:
        return criteria[0]
    return TerminationCollection(*criteria)


def _checkpoint_campaign_identity(
    run_config: NSGA2RunConfig,
    *,
    pop_size: int,
    n_generations: int,
    seed: int,
    termination_config: NSGA2TerminationConfig,
    constraint_names,
) -> dict[str, Any]:
    """Return the exact scientific/evolutionary contract a resume must match."""
    return campaign_identity(
        {
            "experiment": experiment_to_document(run_config.experiment),
            "decision_variables": list(run_config.var_names),
            "bounds": {
                name: [float(value) for value in run_config.bounds[name]]
                for name in run_config.var_names
            },
            "constraints": asdict(run_config.constraints),
            "composition_domain_guard": run_config.composition_domain_guard.as_dict(),
            "constraint_names": list(constraint_names),
            "optimizer": {
                "algorithm": "NSGA2",
                "population_size": int(pop_size),
                "total_generations": int(n_generations),
                "seed": int(seed),
                "sbx_probability": float(run_config.sbx_prob),
                "sbx_eta": float(run_config.sbx_eta),
                "polynomial_mutation_eta": float(run_config.pm_eta),
                "eliminate_duplicates": True,
                "termination": asdict(termination_config),
            },
        }
    )


CAMPAIGN_IDENTITY_FILENAME = "campaign_identity.json"


def nsga2_campaign_identity(
    run_config: NSGA2RunConfig,
    *,
    pop_size: int,
    n_generations: int,
    seed: int,
    termination_config: NSGA2TerminationConfig,
) -> dict[str, Any]:
    """The campaign identity, computed without building the problem.

    It is the same contract a checkpoint must match, so it also names the
    campaign folder: a different base case, search space, constraint set or
    optimizer setting is a different campaign and gets its own folder.
    """
    constraint_names = (
        _VALID_EVALUATION_CONSTRAINT,
        *run_config.constraints.active_names(
            is_steam_assisted=run_config.is_steam_assisted
        ),
    )
    return _checkpoint_campaign_identity(
        run_config,
        pop_size=pop_size,
        n_generations=n_generations,
        seed=seed,
        termination_config=termination_config,
        constraint_names=constraint_names,
    )


def _stored_campaign_sha(results_folder: Path) -> str | None:
    """The identity already committed in a campaign folder, if any."""
    for name in (CHECKPOINT_STATE_FILENAME, CAMPAIGN_IDENTITY_FILENAME):
        path = results_folder / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            identity = payload.get("campaign_identity", payload)
            return identity.get("sha256")
    return None


# What `run_nsga2_optimization` writes at the top of a campaign folder. Only
# these mark earlier campaign results; files a caller keeps beside a campaign
# (the runner's resolved_config.json, a driver's run manifest) are not.
_CAMPAIGN_ARTIFACTS = (
    "results",
    "gen_*",
    "pareto_bounds.csv",
    "pareto_flow_transform.json",
    "generation_log.txt",
    "nsga2_log.txt",
    "optimization_log.csv",
    "evaluations_log.txt",
)


def _unidentified_contents(folder: Path) -> list[str]:
    """Files of earlier NSGA-II campaign artifacts in ``folder`` (empty folders ignored)."""
    if not folder.is_dir():
        return []
    found = []
    for entry in folder.iterdir():
        if not any(fnmatch.fnmatchcase(entry.name, pattern) for pattern in _CAMPAIGN_ARTIFACTS):
            continue
        files = [entry] if entry.is_file() else [path for path in entry.rglob("*") if path.is_file()]
        found.extend(str(path.relative_to(folder)) for path in files)
    return found


def guard_campaign_folder(
    base_folder: str | os.PathLike,
    identity: Mapping[str, Any],
    *,
    continuation: bool,
) -> None:
    """Refuse to write a campaign into a folder that holds another one.

    A folder belongs to exactly one campaign identity. A fresh start is also
    refused when the same campaign has committed generations there: those are
    continued explicitly, never silently overwritten. A fresh start over the
    same campaign with nothing committed (it stopped inside generation 1) is
    allowed, because no completed work is lost.
    """
    results = Path(base_folder) / "results"
    found = _stored_campaign_sha(results)
    if found is not None and found != identity["sha256"]:
        raise RunRefused(
            f"{base_folder} holds a different NSGA-II campaign (identity {found[:12]}); "
            "a campaign never writes into another campaign's folder."
        )
    if found is None and not continuation:
        # Campaign results without an identity (saved before identities were
        # recorded) cannot be proven to be this campaign.
        foreign = sorted(_unidentified_contents(Path(base_folder)))
        if foreign:
            raise RunRefused(
                f"{base_folder} already holds results without a campaign identity "
                f"({', '.join(foreign[:5])}); choose another output root or remove them."
            )
    if not continuation and (results / CHECKPOINT_STATE_FILENAME).is_file():
        raise RunRefused(
            f"{base_folder} already holds committed generations of this campaign; "
            "continue it with --resume, or change the seed or output root for a new one."
        )


def _preflight_continuation(
    base_folder: str | os.PathLike,
    identity: Mapping[str, Any],
    run_config: NSGA2RunConfig,
    *,
    n_generations: int,
    termination_config: NSGA2TerminationConfig,
):
    """Validate a continuation completely before anything is written.

    Every reason to refuse (identity, checksums, population, termination
    compatibility, an already met termination) is checked here, before the
    worker pool, provenance record or any discard exists. Returns the
    committed state and population for the run to use.
    """
    from pydac.utils.provenance import require_same_version
    require_same_version(Path(base_folder) / "results", "optimization",
                         f"The NSGA-II campaign in `{base_folder}`")
    state = load_checkpoint_state(base_folder, expected_identity=identity)
    completed = int(state["completed_generations"])
    if completed >= n_generations:
        raise RunRefused(
            f"Campaign already completed {completed} of {n_generations} requested generations."
        )
    population = load_checkpoint_population(
        base_folder,
        state,
        expected_var_names=run_config.var_names,
        expected_constraint_names=identity["definition"]["constraint_names"],
    )
    termination = build_nsga2_termination(
        n_generations=n_generations,
        config=termination_config,
    )
    if checkpoint_termination_met(termination, state):
        raise RunRefused(
            f"Campaign already completed: its {termination_config.mode} "
            f"termination was met at generation {completed}."
        )
    return state, population


def run_nsga2_optimization(
    run_config: NSGA2RunConfig,
    *,
    pop_size: int,
    n_generations: int,
    n_procs: int = 1,
    seed: int = 1,
    warm_start: WarmStartSpec | None = None,
    termination_config: NSGA2TerminationConfig | None = None,
    delete_generation_folders: bool = True,
    run_postprocessing: bool = True,
) -> Any:
    """
    Execute a full NSGA-II optimization campaign for the given `run_config`.

    Parameters
    ----------
    run_config : NSGA2RunConfig
        Campaign configuration (bounds, cycle, CSS settings, folder).
    pop_size, n_generations : int
        Population size and cumulative hard generation target. On resume,
        only ``n_generations - completed_generations`` new generations run.
    n_procs : int
        Number of worker processes. `1` disables multiprocessing.
    seed : int
        Random seed used to initialize pymoo. Strong continuation restores the
        checkpointed NumPy/Python RNG state after setup.
    warm_start : WarmStartSpec | None
        Optional declarative warm-start configuration. `mode="continuation"`
        resumes the latest compatible atomic checkpoint, restores evaluated
        parents, rank/crowding, cumulative counters and RNG state, and skips
        parent reevaluation.
        `mode="refining"` seeds a new campaign from a saved Pareto/front or
        population, drops out-of-bounds individuals, and optionally fills with
        LHS to reach the requested `pop_size`.
    termination_config : NSGA2TerminationConfig | None
        Native pymoo robust-front termination settings. None preserves the
        historical generation-only stop.
    delete_generation_folders : bool
        Forwarded to `NSGA2Logger` to clean up non-final generation folders.
    run_postprocessing : bool
        If True, run `NSGA2_pp` plotting helpers after optimization.

    Returns
    -------
    pymoo result object.
    """
    if termination_config is None:
        termination_config = NSGA2TerminationConfig(mode="fixed_generations")

    base_folder = run_config.base_folder
    pareto_folder = os.path.join(base_folder, "results")
    continuation = warm_start is not None and warm_start.mode == "continuation"
    identity = nsga2_campaign_identity(
        run_config,
        pop_size=pop_size,
        n_generations=n_generations,
        seed=seed,
        termination_config=termination_config,
    )
    # Checked before anything is written, so a refused run leaves no trace.
    guard_campaign_folder(base_folder, identity, continuation=continuation)
    resume_state = None
    resume_population = None
    if continuation:
        resume_state, resume_population = _preflight_continuation(
            base_folder,
            identity,
            run_config,
            n_generations=n_generations,
            termination_config=termination_config,
        )
    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(pareto_folder, exist_ok=True)
    atomic_write_json(Path(pareto_folder) / CAMPAIGN_IDENTITY_FILENAME, identity, sort_keys=True)

    _save_pareto_bounds(run_config)
    _save_flow_transform_metadata(run_config)

    pool = None
    problem_kwargs: dict[str, Any] = {}
    if n_procs > 1:
        pool = mp.Pool(
            n_procs,
            initializer=nsga2_shared.share_worker_state,
            initargs=(CURRENT_GEN, WORKER_STDOUT_LOCK),
        )
        problem_kwargs["elementwise_runner"] = StarmapParallelization(pool.starmap)

    problem = TVSAOptimizationProblem(run_config, **problem_kwargs)
    if list(problem.constraint_names) != identity["definition"]["constraint_names"]:
        raise RuntimeError("Problem constraints disagree with the campaign identity.")

    from pydac.utils.provenance import start_activity, file_artifact
    checkpoint_path = Path(pareto_folder) / "checkpoint_state.json"
    checkpoint_sources = ([{"artifact": file_artifact(checkpoint_path)}]
                          if warm_start is not None and warm_start.mode == "continuation"
                          and checkpoint_path.is_file() else [])
    optimization_provenance = start_activity(
        pareto_folder, "optimization", config=run_config.experiment,
        execution={"activity_scope": "campaign", "n_workers": n_procs,
                   "campaign_identity_sha256": identity["sha256"],
                   "optimizer": identity["definition"]["optimizer"],
                   "bounds": identity["definition"]["bounds"],
                   "constraints": identity["definition"]["constraints"],
                   "warm_start_mode": None if warm_start is None else warm_start.mode},
        sources=checkpoint_sources,
    )

    completed_generations = 0
    completed_evaluations = 0
    if continuation:
        completed_generations = int(resume_state["completed_generations"])
        completed_evaluations = int(resume_state["completed_evaluations"])
        pop_size = len(resume_population)
        initial_population = resume_population
        warm_label = "strong continuation"
        initial_sampling_summary = {
            "mode": "continuation",
            "source": "checkpoint",
            "path": str(
                Path(base_folder) / "results" / "checkpoint_state.json"
            ),
            "loaded": int(pop_size),
            "reused": int(pop_size),
            "new": 0,
            "dropped_out_of_bounds": 0,
            "truncated": 0,
            "fill": "none",
            "effective_pop_size": int(pop_size),
            "pop_size_overridden": True,
            "parents_already_evaluated": True,
            "completed_generations": completed_generations,
            "remaining_generations": n_generations - completed_generations,
            "completed_evaluations": completed_evaluations,
        }
    elif warm_start is not None:
        initial_population, pop_size, warm_label, initial_sampling_summary = build_initial_population(
            problem,
            pop_size,
            warm_start,
            rng=np.random.default_rng(seed),
        )
    else:
        initial_population = LatinHypercubeSampling()
        warm_label = "LHS"
        initial_sampling_summary = {
            "mode": "fresh",
            "source": "lhs",
            "path": None,
            "loaded": 0,
            "reused": 0,
            "new": int(pop_size),
            "dropped_out_of_bounds": 0,
            "truncated": 0,
            "fill": "lhs",
            "effective_pop_size": int(pop_size),
            "pop_size_overridden": False,
        }

    CURRENT_GEN.value = completed_generations

    run_started_unix_s = time.time()
    callback = NSGA2Logger(
        base_folder=base_folder,
        pareto_folder=pareto_folder,
        delete=delete_generation_folders,
        tot_gens=n_generations,
        desorption_step=run_config.desorption_step,
        var_names=run_config.var_names,
        constraint_names=problem.constraint_names,
        n_procs=n_procs,
        run_started_unix_s=run_started_unix_s,
        checkpoint_identity=identity,
        initial_evaluations=completed_evaluations,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=initial_population,
        crossover=SBX(prob=run_config.sbx_prob, eta=run_config.sbx_eta),
        mutation=PM(eta=run_config.pm_eta),
        eliminate_duplicates=True,
    )

    termination = build_nsga2_termination(
        n_generations=n_generations,
        config=termination_config,
    )
    algorithm.setup(
        problem,
        termination=termination,
        seed=seed,
        callback=callback,
        save_history=False,
        verbose=True,
    )
    if resume_state is not None:
        restore_algorithm_from_checkpoint(
            algorithm,
            resume_population,
            resume_state,
        )
        if algorithm.termination.has_terminated():
            # Unreachable after _preflight_continuation; kept as a guard.
            if pool is not None:
                pool.terminate()
                pool.join()
            raise RunRefused(
                f"Campaign already completed: its {termination_config.mode} "
                f"termination was met at generation {completed_generations}."
            )
        # Only a campaign that will really continue loses its uncommitted
        # artifacts: never discard anything before every refusal above.
        discarded = discard_transient_generation_folders(base_folder)
        discarded.extend(
            discard_uncommitted_checkpoint_artifacts(
                base_folder,
                completed_generations=completed_generations,
            )
        )
        if discarded:
            print(
                "[RESUME] Discarded uncommitted artifacts: "
                f"{', '.join(discarded)}"
            )

    nsga_log_path = os.path.join(base_folder, "nsga2_log.txt")
    log_mode = "a" if continuation else "w"
    log_file_handle = open(nsga_log_path, log_mode, buffering=1, encoding="utf-8")
    tee_stdout = Tee(sys.stdout, log_file_handle)
    tee_stderr = Tee(sys.stderr, log_file_handle)

    optimization_completed = False
    optimization_wall_start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(tee_stdout), \
                contextlib.redirect_stderr(tee_stderr):
            print(f"\n{'=' * 120}")
            print("     Starting NSGA-II Optimization".center(120))
            print(f"{'=' * 120}\n")

            # --- Problem summary ---
            active = list(problem.constraint_names)
            print(f"  Decision variables ({len(run_config.var_names)}): {list(run_config.var_names)}")
            print(f"  Objectives:         2  (eq_work ↓, productivity ↑)")
            print(f"  Constraints ({len(active)}):    {active}")
            print(f"  Cycle steps:        {run_config.cycle_steps}")
            print(f"  Desorption mode:    {run_config.desorption_step}")
            print(f"  Bed architecture:   {run_config.bed_config['architecture']}")
            print(
                "  Composition guard:  "
                f"{run_config.composition_domain_guard.as_dict()}"
            )
            print(f"  Persistence:       {MOO_COMPACT} (ordinary candidates)")
            print(f"  tau_ar={run_config.tau_ar},  bed_volume={run_config.bed_volume}")
            simulation = run_config.experiment.simulation
            execution = run_config.experiment.execution
            print(
                f"  adsorption_T_feed={run_config.adsorption_T_feed} K,  "
                f"adsorption_RH={run_config.adsorption_RH}"
            )
            print(
                "  equilibrium_systems="
                f"{tuple(simulation.adsorption.equilibrium_systems)},  "
                f"max_cycles={execution.max_cycles},  css_tol={execution.css_tolerance}"
            )
            print()
            print(f"  Population size:    {pop_size}")
            print(f"  Generations:        {n_generations}")
            print(f"  Termination:        {termination_config.mode}")
            if termination_config.mode == "robust_pareto":
                print(
                    f"    ftol={termination_config.ftol}, "
                    f"period={termination_config.period}, "
                    f"n_skip={termination_config.n_skip}"
                )
            print(f"  Workers:            {n_procs}")
            print(f"  Seed:               {seed}")
            print(f"  Crossover:          SBX(prob={run_config.sbx_prob}, eta={run_config.sbx_eta})")
            print(f"  Mutation:           PM(eta={run_config.pm_eta})")
            print(f"  Initial sampling:   {warm_label}")
            if initial_sampling_summary["mode"] == "fresh":
                print(
                    f"    Sampled:          "
                    f"{initial_sampling_summary['effective_pop_size']} individuals via LHS"
                )
            elif initial_sampling_summary["mode"] == "continuation":
                print(f"    Source:           {initial_sampling_summary['source']}")
                print(f"    File:             {initial_sampling_summary['path']}")
                print(f"    Loaded:           {initial_sampling_summary['loaded']} individuals")
                print(
                    f"    Final sample:     "
                    f"{initial_sampling_summary['effective_pop_size']} individuals"
                )
                print(f"    pop_size rule:    overridden to saved population size")
                print(f"    Parent evals:     reused from checkpoint")
                print(
                    "    Generations:      "
                    f"{initial_sampling_summary['completed_generations']} "
                    "completed + "
                    f"{initial_sampling_summary['remaining_generations']} remaining"
                )
            else:
                fill_label = (
                    "LHS"
                    if initial_sampling_summary["new"] > 0
                    and initial_sampling_summary["fill"] == "lhs"
                    else "none"
                )
                print(f"    Source:           {initial_sampling_summary['source']}")
                print(f"    File:             {initial_sampling_summary['path']}")
                print(f"    Loaded:           {initial_sampling_summary['loaded']} individuals")
                print(
                    f"    Reused:           {initial_sampling_summary['reused']} individuals"
                )
                print(f"    New:              {initial_sampling_summary['new']} individuals")
                print(
                    "    Dropped OOB:      "
                    f"{initial_sampling_summary['dropped_out_of_bounds']} individuals"
                )
                print(
                    f"    Truncated:        {initial_sampling_summary['truncated']} individuals"
                )
                print(f"    Fill mode:        {fill_label}")
                print(
                    f"    Final sample:     "
                    f"{initial_sampling_summary['effective_pop_size']} individuals"
                )
            print()
            print(f"  Bounds:")
            for v in run_config.var_names:
                lo, hi = run_config.bounds[v]
                print(f"    {v:20s}  [{lo:.6g}, {hi:.6g}]")
            print(f"\n{'-' * 120}\n")

            res = algorithm.run()
            res.algorithm = algorithm
            optimization_completed = True

            print(f"\n{'=' * 120}")
            print("     Optimization Completed".center(120))
            print(f"{'=' * 120}\n")
    finally:
        log_file_handle.close()
        if pool is not None:
            if optimization_completed:
                pool.close()
            else:
                pool.terminate()
            pool.join()

    optimization_wall_time_s = time.perf_counter() - optimization_wall_start

    if res.X is None:
        pareto_X = np.empty((0, len(run_config.var_names)), dtype=float)
        pareto_F = np.empty((0, problem.n_obj), dtype=float)
    else:
        pareto_X = np.atleast_2d(np.asarray(res.X, dtype=float))
        pareto_F = np.atleast_2d(np.asarray(res.F, dtype=float))

    atomic_savez(
        os.path.join(pareto_folder, "pareto_front.npz"),
        X=pareto_X,
        F=pareto_F,
        var_names=np.asarray(run_config.var_names, dtype=str),
    )

    final_generation = int(callback.data["gen"][-1])
    final_evaluations = int(res.algorithm.evaluator.n_eval)
    if final_generation >= n_generations:
        stop_reason = "max_generations"
    elif (
        termination_config.max_evaluations is not None
        and final_evaluations >= termination_config.max_evaluations
    ):
        stop_reason = "max_evaluations"
    else:
        stop_reason = "robust_pareto_convergence"

    timing_summary = callback.timing_summary()
    evaluation_summary = campaign_evaluation_summary(base_folder)
    run_summary = {
        "status": "completed",
        "evaluations": evaluation_summary,
        "stop_reason": stop_reason,
        "final_generation": final_generation,
        "final_evaluations": final_evaluations,
        "n_feasible_pareto": int(len(pareto_X)),
        "termination": {
            "mode": termination_config.mode,
            "ftol": termination_config.ftol,
            "period": termination_config.period,
            "n_skip": termination_config.n_skip,
            "max_generations": int(n_generations),
            "max_evaluations": termination_config.max_evaluations,
        },
        "operating_condition": {
            "adsorption_T_feed_K": run_config.adsorption_T_feed,
            "adsorption_RH": run_config.adsorption_RH,
            "tau_ar": run_config.tau_ar,
            "bed_volume_m3": run_config.bed_volume,
            "bed_config": run_config.bed_config,
        },
        "warm_start": initial_sampling_summary,
        "timing": {
            "optimization_wall_time_s": optimization_wall_time_s,
            "n_workers": n_procs,
            "candidate_evaluations_with_timing": timing_summary["count"],
            "candidate_evaluation_wall_time_sum_s": timing_summary["sum_s"],
            "candidate_evaluation_wall_time_mean_s": timing_summary["mean_s"],
            "candidate_evaluation_wall_time_median_s": timing_summary["median_s"],
            "candidate_evaluation_wall_time_p90_s": timing_summary["p90_s"],
            "estimated_serial_evaluation_time_s": timing_summary["sum_s"],
            "ideal_parallel_lower_bound_s": (
                timing_summary["sum_s"] / n_procs
            ),
            "observed_worker_utilization": (
                timing_summary["sum_s"]
                / (n_procs * optimization_wall_time_s)
                if optimization_wall_time_s > 0.0
                else np.nan
            ),
        },
    }
    run_summary["provenance"] = optimization_provenance
    with open(
        os.path.join(pareto_folder, "optimization_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(run_summary, handle, indent=2)

    print(f"Saved: {pareto_folder}/pareto_front.npz")
    print(f"Saved: {pareto_folder}/last_population.npz")
    print(
        f"Stop reason: {stop_reason} at generation {final_generation} "
        f"after {final_evaluations} evaluations."
    )
    print(
        f"Timing: optimizer wall={optimization_wall_time_s / 60:.2f} min | "
        f"candidate mean/median/p90="
        f"{timing_summary['mean_s'] / 60:.2f}/"
        f"{timing_summary['median_s'] / 60:.2f}/"
        f"{timing_summary['p90_s'] / 60:.2f} min | "
        f"candidate sum={timing_summary['sum_s'] / 60:.2f} min."
    )
    print(format_evaluation_summary(evaluation_summary))

    if run_postprocessing and len(pareto_X):
        run_pareto_postprocessing(pareto_folder)
    elif run_postprocessing:
        print("[WARN] Post-processing skipped: no feasible Pareto front was found.")

    return res


def _campaign_from_checkpoint(base_folder: str | os.PathLike):
    """Rebuild a committed campaign's run config and optimizer options."""
    base = Path(base_folder)
    results = base / "results"
    state_path = results / CHECKPOINT_STATE_FILENAME
    if not state_path.is_file():
        raise ConfigurationError(
            f"{base} has no committed NSGA-II checkpoint ({state_path} is missing)."
        )
    # Before the saved case is loaded: a later version may not read it.
    # _preflight_continuation repeats the check for direct continuations.
    from pydac.utils.provenance import require_same_version
    require_same_version(results, "optimization", f"The NSGA-II campaign in `{base}`")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    definition = state["campaign_identity"]["definition"]
    optimizer = definition["optimizer"]
    run_config = NSGA2RunConfig(
        base_folder=str(base),
        bounds={name: tuple(pair) for name, pair in definition["bounds"].items()},
        decision_variables=list(definition["decision_variables"]),
        experiment=load_case(definition["experiment"]),
        constraints=NSGA2Constraints(**definition["constraints"]),
        sbx_prob=optimizer["sbx_probability"],
        sbx_eta=optimizer["sbx_eta"],
        pm_eta=optimizer["polynomial_mutation_eta"],
        composition_domain_guard=CompositionDomainGuard(
            **definition["composition_domain_guard"]
        ),
    )
    options = {
        "pop_size": int(optimizer["population_size"]),
        "n_generations": int(optimizer["total_generations"]),
        "seed": int(optimizer["seed"]),
        "termination_config": NSGA2TerminationConfig(**optimizer["termination"]),
    }
    return run_config, options


def preflight_nsga2_resume(base_folder: str | os.PathLike) -> dict[str, Any]:
    """Every check a continuation makes before it writes; return the committed state.

    This is what ``--resume CAMPAIGN --dry-run`` runs: version, identity,
    checksums, population, termination compatibility and a completed campaign
    are refused exactly as a real continuation would refuse them.
    """
    run_config, options = _campaign_from_checkpoint(base_folder)
    identity = nsga2_campaign_identity(
        run_config,
        pop_size=options["pop_size"],
        n_generations=options["n_generations"],
        seed=options["seed"],
        termination_config=options["termination_config"],
    )
    guard_campaign_folder(base_folder, identity, continuation=True)
    state, _ = _preflight_continuation(
        base_folder,
        identity,
        run_config,
        n_generations=options["n_generations"],
        termination_config=options["termination_config"],
    )
    return state


def resume_nsga2_campaign(
    base_folder: str | os.PathLike,
    *,
    n_procs: int = 1,
    delete_generation_folders: bool = True,
    run_postprocessing: bool = True,
) -> Any:
    """Continue a campaign from its latest committed checkpoint.

    Everything that defines the campaign (base case, decision variables,
    bounds, constraints, guard, optimizer and termination) is read back from
    the checkpoint, so a continuation cannot drift from the original run;
    only the number of local workers, which changes no result, is chosen here.
    """
    run_config, options = _campaign_from_checkpoint(base_folder)
    results = Path(base_folder) / "results"
    return run_nsga2_optimization(
        run_config,
        **options,
        n_procs=n_procs,
        warm_start=WarmStartSpec(
            pareto_path=str(results / "last_pareto_front.npz"),
            population_path=str(results / "last_population.npz"),
            mode="continuation",
        ),
        delete_generation_folders=delete_generation_folders,
        run_postprocessing=run_postprocessing,
    )


def run_pareto_postprocessing(pareto_folder: str) -> None:
    """
    Standard Pareto-plot post-processing for a completed campaign. Kept as a
    separate helper so runners can call it independently.
    """
    from pydac.post_processing.optimization.nsga2_post_processing import NSGA2_pp

    pp = NSGA2_pp(pareto_folder)
    # The front table first: a figure that fails must not cost the results.
    pp.export_pareto_csv()
    pp.plot_population_vs_front(save=True)
    pp.plot_pareto_front(save=True)
    pp.plot_scaled_decision_boxplot(save=True)
    pp.plot_scaled_decision_violinplot(save=True)
    #pp.pandas_postprocess(threshold=5, save=True)
