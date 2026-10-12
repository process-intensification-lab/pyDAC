"""
Helpers for the CSS cycle runner.

This module keeps orchestration-level details out of the public entry point in
``workflows/orchestrator_TVSA_cycle_css.py`` while reusing the lower-level CSS utilities from
``utils.css``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from pydac.utils.css import (
    CSSProfile,
    build_executed_steps,
    css_reached,
    final_profiles,
    load_resume_prev_fields,
    load_resume_prev_profile,
    plot_cycle_at_css,
    print_css_diagnostics,
)
from pydac.utils.tvsa import get_step_solution_path
from pydac.utils.solver.persistence import MOO_COMPACT, RUN_FULL


@dataclass(frozen=True)
class CSSCycleCheckResult:
    """Result of the CSS convergence check after a completed cycle."""

    prev_prof: CSSProfile | None
    reached: bool
    css_cycle_idx: int | None = None
    skipped: bool = False
    diagnostics: dict = field(default_factory=dict)
    limiting_field: str | None = None


def validate_css_run_setup(*, resume_from_cycle, max_cycles, css_tol):
    """
    Validate high-level CSS run controls and return the first cycle to run.
    """
    start_cycle = 1 if resume_from_cycle is None else resume_from_cycle
    if resume_from_cycle is not None and (
        isinstance(resume_from_cycle, bool) or not isinstance(resume_from_cycle, int)
    ):
        raise TypeError(
            f"`resume_from_cycle` must be int or None. Received type: {type(resume_from_cycle).__name__}."
        )
    if max_cycles < 1:
        raise ValueError("`max_cycles` must be at least 1.")
    if start_cycle < 1:
        raise ValueError("`resume_from_cycle` must be at least 1 when provided.")
    if start_cycle > max_cycles:
        raise ValueError("`resume_from_cycle` cannot be larger than `max_cycles`.")
    if css_tol < 0:
        raise ValueError("`css_tol` must be non-negative.")
    return start_cycle


def validate_css_runner_configuration(execution):
    """Validate CSS runner settings without creating folders or solving."""
    from pydac.utils.run import validate_flags, validate_output_options
    from pydac.utils.tvsa import build_repeating_cycle_steps

    from pydac.utils.css.cycle import resolve_css_probe_step, validate_cycle_options_structure

    validate_flags(
        solve=execution.solve,
        plot_profiles=execution.plot_profiles,
        use_sparsity=execution.use_sparsity,
        compute_numerical_sparsity=execution.compute_numerical_sparsity,
        termination_mode=execution.termination_mode,
    )
    validate_output_options(
        plot_profiles=execution.plot_profiles,
        plot_final_cycle_profiles=execution.plot_final_cycle_profiles,
        keep_only_final_cycle=execution.keep_only_final_cycle,
    )
    if execution.step_options:
        # A CSS run has no run-wide step options: they would be silently ignored.
        raise ValueError(
            "A CSS run takes per-step options per cycle: move execution.step_options into "
            'execution.cycle_options, e.g. {"1": {"step_options": {...}}}.'
        )
    cycle_options = {
        index: dict(values) for index, values in execution.cycle_options.items()
    } or None
    validate_cycle_options_structure(cycle_options)
    validate_css_run_setup(
        resume_from_cycle=execution.resume_from_cycle,
        max_cycles=execution.max_cycles,
        css_tol=execution.css_tolerance,
    )
    repeating_steps = build_repeating_cycle_steps(cycle_steps=execution.cycle_steps)
    resolve_css_probe_step(
        tvsa_cycle=None,
        repeating_steps=repeating_steps,
        css_probe_step=execution.css_probe_step,
    )
    return True


def validate_css_case(case):
    """Validate a CSS run or campaign: its runner settings and its product line.

    A cycle whose repeating steps all vent their outlet has no product, so every
    product KPI is undefined; it is refused here rather than after the cycles ran.
    """
    from pydac.utils.tvsa import build_repeating_cycle_steps

    from pydac.utils.run.validation import validate_steps_against_mixture

    validate_css_runner_configuration(case.execution)
    repeating_steps = build_repeating_cycle_steps(cycle_steps=case.execution.cycle_steps)
    validate_steps_against_mixture(case, ["startup", *repeating_steps])
    collected = [
        step.name
        for step in case.simulation.cycle.steps
        if step.name in repeating_steps and step.config.outlet.collect_product
    ]
    if not collected:
        raise ValueError(
            f"No step of the cycle {repeating_steps} collects product; set "
            "`steps.<step>.outlet.collect_product=true` on at least one of them."
        )
    return True


def resolve_cycle_initial_state(
    *,
    tvsa_cycle,
    cycle_idx,
    start_cycle,
    base_folder,
    repeating_steps,
    probe_step,
    previous_cycle_final_fields,
    prev_prof,
):
    """
    Resolve warm-start fields and CSS baseline profile for the next cycle.
    """
    if cycle_idx == start_cycle:
        if start_cycle == 1:
            return None, prev_prof

        previous_cycle_idx = cycle_idx - 1
        previous_executed_steps = build_executed_steps(repeating_steps, previous_cycle_idx)
        initial_prev_fields = load_resume_prev_fields(
            tvsa_cycle=tvsa_cycle,
            base_folder=base_folder,
            previous_cycle_idx=previous_cycle_idx,
            previous_executed_steps=previous_executed_steps,
        )
        prev_prof = load_resume_prev_profile(
            tvsa_cycle=tvsa_cycle,
            base_folder=base_folder,
            previous_cycle_idx=previous_cycle_idx,
            previous_executed_steps=previous_executed_steps,
            probe_step=probe_step,
        )
        print(
            f"Resuming from cycle {cycle_idx} using the saved terminal state of cycle {previous_cycle_idx}."
        )
        if prev_prof is None:
            print(
                f"Cycle {previous_cycle_idx} does not contain the CSS probe step '{probe_step}'. "
                f"A new CSS baseline will be stored when cycle {cycle_idx} completes."
            )
        return initial_prev_fields, prev_prof

    if previous_cycle_final_fields is None:
        raise ValueError(
            f"Cycle {cycle_idx} cannot start because cycle {cycle_idx - 1} did not produce "
            "terminal fields for warm-start initialization."
        )
    return previous_cycle_final_fields, prev_prof


def evaluate_css_after_cycle(
    *,
    tvsa_cycle,
    cycle_idx,
    step_records,
    probe_step,
    prev_prof,
    css_tol,
    cycle_run_config,
    results_base_folder,
    cycle_steps,
    print_to_terminal,
    cycle_config=None,
):
    """
    Update the CSS baseline and report whether cyclic steady state was reached.
    """
    executed_steps = [record["step"] for record in step_records]
    if probe_step not in executed_steps:
        print(
            f"Skipping CSS check for cycle {cycle_idx}: probe step '{probe_step}' is not executed "
            "in this startup cycle."
        )
        return CSSCycleCheckResult(prev_prof=prev_prof, reached=False, skipped=True)

    persistence_policy = getattr(tvsa_cycle, "persistence_policy", RUN_FULL)
    if persistence_policy == MOO_COMPACT:
        curr_prof = tvsa_cycle.final_profile_for_step(probe_step)
    else:
        sol_path = get_step_solution_path(
            tvsa_cycle.base_folder,
            probe_step,
            base_folder_is_spatial_root=tvsa_cycle.base_folder_is_spatial_root,
        )
        if not os.path.exists(sol_path):
            raise FileNotFoundError(f"Missing {sol_path} - cannot check CSS.")
        curr_prof = final_profiles(sol_path)
    if prev_prof is None:
        print(f"CSS baseline stored from cycle {cycle_idx} at step '{probe_step}'.")
        return CSSCycleCheckResult(prev_prof=curr_prof, reached=False)

    reached, diagnostics = css_reached(
        prev_prof,
        curr_prof,
        rel_tol=css_tol,
        return_details=True,
    )
    print_css_diagnostics(
        cycle_idx=cycle_idx,
        probe_step=probe_step,
        css_tol=css_tol,
        diagnostics=diagnostics,
    )
    limiting_field = (
        max(diagnostics, key=lambda name: diagnostics[name]["max_rel_diff"])
        if diagnostics
        else None
    )

    if not reached:
        return CSSCycleCheckResult(
            prev_prof=curr_prof,
            reached=False,
            diagnostics=diagnostics,
            limiting_field=limiting_field,
        )

    print(
        f"\nCSS reached after {cycle_idx} cycles "
        f"(difference < {css_tol * 100:.2f}% everywhere at step '{probe_step}')."
    )
    if (
        cycle_run_config["requested_cycle_plots"]
        and not cycle_run_config["force_live_cycle_plots"]
    ):
        plot_cycle_at_css(
            base_folder=results_base_folder,
            css_cycle_idx=cycle_idx,
            cycle_steps=cycle_steps,
            termination_mode=cycle_run_config["run_kwargs"]["termination_mode"],
            step_options=cycle_run_config["step_options"],
            print_to_terminal=print_to_terminal,
            cycle_config=cycle_config,
        )

    return CSSCycleCheckResult(
        prev_prof=curr_prof,
        reached=True,
        css_cycle_idx=cycle_idx,
        diagnostics=diagnostics,
        limiting_field=limiting_field,
    )
