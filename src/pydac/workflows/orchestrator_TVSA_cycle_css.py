"""
pyDAC - Cyclic steady-state TVSA entry point.

This script executes repeated TVSA cycles until the selected CSS criterion is
met, while handling per-cycle overrides, optional resume-from-cycle restarts,
and consolidated cycle-level logging.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

import os
import time
from pathlib import Path

_THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _thread_env_var in _THREAD_LIMIT_ENV_VARS:
    os.environ[_thread_env_var] = "1"


from pydac.core.TVSA_cycle import TVSA_Cycle
from pydac.config import ExperimentConfig
from pydac.config.launch import resolve_experiment_config
from pydac.config.loader import write_resolved_config
from pydac.utils.logger import log_file
from pydac.utils.tvsa import build_repeating_cycle_steps
from pydac.utils.css import (
    create_cycle_folder,
    discard_cycle_directories_before,
    plot_cycle_at_css,
    resolve_css_probe_step,
    resolve_cycle_run_config,
    resolve_spatial_results_base_folder,
    write_cycle_metadata,
)
from pydac.utils.css.runner import (
    evaluate_css_after_cycle,
    resolve_cycle_initial_state,
    validate_css_run_setup,
)
from pydac.utils.run import validate_flags
from pydac.utils.solver.persistence import RUN_FULL, MOO_COMPACT, normalize_persistence_policy
from pydac.utils.campaign import atomic_write_json


def run_css_simulation(
    experiment_config: ExperimentConfig,
    *,
    base_folder=None,
    retain_cycles=None,
    composition_domain_guard=None,
    persistence_policy=RUN_FULL,
):
    """
    Run successive TVSA cycles until cyclic steady state is reached or the
    maximum number of cycles is exceeded.

    The repeating `cycle_steps` sequence must begin with adsorption. In cycle 1,
    that leading adsorption step is replaced automatically by startup.

    `cycle_options` can override the global run flags on a per-cycle basis and
    can also contain nested per-step overrides plus `rerun_from_step`.
    `termination_mode` follows the same precedence: global CSS default,
    per-cycle override, then nested per-step override.
    `cvode_sparsity_mode=None` resolves to the recommended "pde_only" sparse
    mask; "full" is kept only for diagnostics.

    There is no top-level `step_options` argument in CSS mode. Per-step control
    lives only inside `cycle_options[n]["step_options"]`.

    `resume_from_cycle=None` starts from cycle 1. If an integer is provided,
    the loop restarts from that later cycle by loading the saved terminal state
    of the previous cycle.

    When `plot_final_cycle_profiles=True`, step-level internal 1D, internal 2D,
    and outlet profile PDFs are generated from the saved solutions of the last
    completed cycle only. The cycle is not solved again.

        Parameters:
        - experiment_config (ExperimentConfig): The resolved public document. It
            states the physics, the grid, the cycle, and what the CSS run does.
        - base_folder (str or None): Directory to save simulation results. None
            uses `execution.output_root`.

        - retain_cycles (int or None): How many of the most recent cycle folders
            to keep on disk while the run is going. None keeps every cycle
            until the run ends. This is a disk policy of the caller, not a
            property of the experiment: it changes no number the run produces
            and so is deliberately not part of the document. Two is the
            smallest useful value, because the after-the-fact CSS diagnostic
            reads the retained cycle and the one before it.
        - composition_domain_guard: Optional operational terminal guard. It
            rejects a step after a configured closure-species undershoot but
            never alters the state or the physical RHS.
        - persistence_policy: ``run_full`` keeps complete step trajectories;
            ``moo_compact`` keeps only versioned terminal states and step metrics.

        Run behavior is configured in the document: use `execution.*`, `grid.*`,
        and `solver.*` settings before calling this workflow rather than passing
        ad-hoc keyword overrides here. That keeps the API and saved
        `resolved_config.json` identical.
    """

    if retain_cycles is not None and (
        not isinstance(retain_cycles, int)
        or isinstance(retain_cycles, bool)
        or retain_cycles < 2
    ):
        raise ValueError("`retain_cycles` must be None or an integer of at least 2.")
    persistence_policy = normalize_persistence_policy(persistence_policy)

        # Entry point for a CSS run: resolve public config to runtime config.
    resolved, cycle_config = resolve_experiment_config(
        experiment_config=experiment_config,
    )

        # Read the run options from the execution section of the resolved document.
    execution = resolved.execution
    solve = execution.solve
    plot_profiles = execution.plot_profiles
    plot_final_cycle_profiles = execution.plot_final_cycle_profiles
    print_to_terminal = execution.print_to_terminal
    print_cycle_build = execution.print_cycle_build
    max_cycles = execution.max_cycles
    css_tol = execution.css_tolerance
    resume_from_cycle = execution.resume_from_cycle
    use_sparsity = execution.use_sparsity
    compute_numerical_sparsity = execution.compute_numerical_sparsity
    termination_mode = execution.termination_mode
    css_probe_step = execution.css_probe_step
    cycle_steps = execution.cycle_steps
    cycle_options = {
        index: dict(values) for index, values in execution.cycle_options.items()
    } or None
    # Returns True or raises; there is no silent "invalid" outcome to handle.
    validate_flags(
        solve=solve,
        plot_profiles=plot_profiles,
        use_sparsity=use_sparsity,
        compute_numerical_sparsity=compute_numerical_sparsity,
        termination_mode=termination_mode,
    )
    if not isinstance(plot_final_cycle_profiles, bool):
        raise TypeError("`plot_final_cycle_profiles` must be bool.")
    if persistence_policy == MOO_COMPACT and plot_final_cycle_profiles:
        raise ValueError(
            "`plot_final_cycle_profiles=True` requires `run_full`; compact MOO "
            "runs do not retain trajectories for post-hoc plotting."
        )
    bed_config = cycle_config.bed
    base_folder = execution.output_root if base_folder is None else base_folder
    start_cycle = validate_css_run_setup(
        resume_from_cycle=resume_from_cycle,
        max_cycles=max_cycles,
        css_tol=css_tol,
    )
    results_base_folder = resolve_spatial_results_base_folder(
        base_folder,
        grid_config=cycle_config.grid,
    )
    write_resolved_config(resolved, os.path.join(results_base_folder, "resolved_config.json"))
    from pydac.utils.provenance import start_activity
    start_activity(results_base_folder, "run", config=resolved,
                   execution={"workflow": "css", "solve": solve,
                              "persistence_policy": persistence_policy})
    overall_log_path = os.path.join(results_base_folder, "log.txt")
    # A resumed run extends the log of the cycles it continues.
    with log_file(
        overall_log_path,
        also_stdout=print_to_terminal,
        append=resume_from_cycle is not None,
    ):
        print(f"Overall CSS log will be saved in: {overall_log_path}")
        print(f"Persistence policy: {persistence_policy}")

        prev_prof = None
        previous_cycle_final_fields = None
        css_cycle_idx = None
        cycles_completed = start_cycle - 1
        css_probe_step_key = None
        last_step_records = []
        last_cycle_run_config = None
        css_history = []
        overall_start_time = time.time()

        for cycle_idx in range(start_cycle, max_cycles + 1):
            cycle_folder = create_cycle_folder(results_base_folder, cycle_idx)

            print(f"\n{'=' * 120}")
            print(f"         Starting TVSA Cycle #{cycle_idx}".center(120))
            print(f"{'=' * 120}\n")

            tvsa_cycle = TVSA_Cycle(
                cycle_config,
                base_folder=cycle_folder,
                cyclic_mode=True,
                cycle_idx=cycle_idx,
                last_cycle_idx=max_cycles,
                print_to_terminal=print_to_terminal,
                base_folder_is_spatial_root=True,
                composition_domain_guard=composition_domain_guard,
                persistence_policy=persistence_policy,
            )

            repeating_steps = build_repeating_cycle_steps(cycle_steps=cycle_steps)
            probe_step = resolve_css_probe_step(
                tvsa_cycle=tvsa_cycle,
                repeating_steps=repeating_steps,
                css_probe_step=css_probe_step,
            )
            cycle_run_config = resolve_cycle_run_config(
                tvsa_cycle=tvsa_cycle,
                cycle_idx=cycle_idx,
                repeating_steps=repeating_steps,
                solve=solve,
                plot_profiles=plot_profiles,
                use_sparsity=use_sparsity,
                compute_numerical_sparsity=compute_numerical_sparsity,
                termination_mode=termination_mode,
                print_cycle_build=print_cycle_build,
                cycle_options=cycle_options,
                start_cycle=start_cycle,
            )
            last_cycle_run_config = cycle_run_config

            if cycle_run_config["force_live_cycle_plots"]:
                tvsa_cycle.last_cycle_idx = cycle_idx

            initial_prev_fields, prev_prof = resolve_cycle_initial_state(
                tvsa_cycle=tvsa_cycle,
                cycle_idx=cycle_idx,
                start_cycle=start_cycle,
                base_folder=base_folder,
                repeating_steps=repeating_steps,
                probe_step=probe_step,
                previous_cycle_final_fields=previous_cycle_final_fields,
                prev_prof=prev_prof,
            )

            if cycle_run_config["rerun_step_key"] is not None:
                print(
                    f"Cycle {cycle_idx}: loading saved steps before '{cycle_run_config['rerun_step_key']}' "
                    "and rerunning from that step onward."
                )

            final_fields, step_records, cycle_total_time_s = tvsa_cycle.run_cycle(
                solve=cycle_run_config["run_kwargs"]["solve"],
                plot_profiles=cycle_run_config["run_kwargs"]["plot_profiles"],
                use_sparsity=cycle_run_config["run_kwargs"]["use_sparsity"],
                compute_numerical_sparsity=cycle_run_config["run_kwargs"]["compute_numerical_sparsity"],
                termination_mode=cycle_run_config["run_kwargs"]["termination_mode"],
                cycle_steps=cycle_steps,
                step_options=cycle_run_config["step_options"],
                initial_prev_fields=initial_prev_fields,
                print_cycle_build=cycle_run_config["run_kwargs"]["print_cycle_build"],
                print_summary=True,
            )

            cycles_completed = cycle_idx
            last_step_records = step_records
            css_probe_step_key = probe_step
            previous_cycle_final_fields = final_fields

            write_cycle_metadata(
                cycle_folder=cycle_folder,
                step_records=step_records,
                cycle_total_time_s=cycle_total_time_s,
                css_probe_step=probe_step,
            )

            css_check = evaluate_css_after_cycle(
                tvsa_cycle=tvsa_cycle,
                cycle_idx=cycle_idx,
                step_records=step_records,
                probe_step=probe_step,
                prev_prof=prev_prof,
                css_tol=css_tol,
                cycle_run_config=cycle_run_config,
                cycle_config=cycle_config,
                results_base_folder=results_base_folder,
                cycle_steps=cycle_steps,
                print_to_terminal=print_to_terminal,
            )
            css_record = {
                "cycle": int(cycle_idx),
                "probe_step": probe_step,
                "tolerance": float(css_tol),
                "reached": bool(css_check.reached),
                "skipped": bool(css_check.skipped),
                "limiting_field": css_check.limiting_field,
                "diagnostics": dict(css_check.diagnostics),
            }
            css_history.append(css_record)
            atomic_write_json(
                Path(cycle_folder) / "css_diagnostics.json",
                css_record,
                sort_keys=True,
            )
            prev_prof = css_check.prev_prof
            if retain_cycles is not None:
                dropped = discard_cycle_directories_before(
                    results_base_folder,
                    cycle_idx - retain_cycles + 1,
                )
                if dropped:
                    print(f"Retaining the last {retain_cycles} cycles; dropped {dropped}.")

            if css_check.reached:
                css_cycle_idx = css_check.css_cycle_idx
                break

        if css_cycle_idx is None:
            print(f"Stopped after reaching max_cycles = {max_cycles} without CSS.")
        else:
            print(f"CSS reached at cycle #{css_cycle_idx}.")

        final_cycle_idx = css_cycle_idx or cycles_completed
        if plot_final_cycle_profiles and final_cycle_idx >= start_cycle:
            print(
                f"Generating step profiles from saved CSS cycle #{final_cycle_idx} "
                "(no re-solve)."
            )
            plot_cycle_at_css(
                cycle_config=cycle_config,
                base_folder=results_base_folder,
                css_cycle_idx=final_cycle_idx,
                cycle_steps=cycle_steps,
                termination_mode=last_cycle_run_config["run_kwargs"]["termination_mode"],
                step_options=last_cycle_run_config["step_options"],
                print_to_terminal=print_to_terminal,
            )

        overall_total_time = time.time() - overall_start_time
        print(f"\n{'=' * 100}")
        print(f"      Total Wall-Clock Time Across All Cycles: {overall_total_time:.2f} s")
        print(f"{'=' * 100}\n")

        result = {
            "cycles_completed": cycles_completed,
            "css_reached": css_cycle_idx is not None,
            "css_cycle_idx": css_cycle_idx,
            "css_probe_step": css_probe_step_key,
            "last_step_records": last_step_records,
            "overall_total_time_s": overall_total_time,
            "results_base_folder": results_base_folder,
            "bed_config": bed_config,
            "persistence_policy": persistence_policy,
            "css_history": css_history,
        }
        return result


def replot_cycle_profiles(*, experiment_config: ExperimentConfig, results_base_folder, cycle_idx: int):
    """Draw the step plots of the saved CSS cycle ``cycle_idx`` again, without solving.

    The same replay as ``plot_final_cycle_profiles`` at the end of a run: every step of
    the cycle is rebuilt from ``experiment_config`` and reloaded from its ``sol.pkl``. The
    first step starts from the final state of cycle ``cycle_idx - 1``, which must still be
    on disk. ``pydac.report`` checks both before calling this. Only the plots are written:
    the step logs, step metrics and their provenance stay as the run wrote them.
    """
    resolved, cycle_config = resolve_experiment_config(experiment_config=experiment_config)
    execution = resolved.execution
    cycle_options = {
        index: dict(values) for index, values in execution.cycle_options.items()
    } or None
    tvsa_cycle = TVSA_Cycle(
        cycle_config,
        base_folder=os.path.join(results_base_folder, f"cycle_{cycle_idx}"),
        cyclic_mode=True,
        cycle_idx=cycle_idx,
        last_cycle_idx=cycle_idx,
        print_to_terminal=execution.print_to_terminal,
        base_folder_is_spatial_root=True,
    )
    # The termination mode and step options this cycle ran with, cycle_options included.
    cycle_run_config = resolve_cycle_run_config(
        tvsa_cycle=tvsa_cycle,
        cycle_idx=cycle_idx,
        repeating_steps=build_repeating_cycle_steps(cycle_steps=execution.cycle_steps),
        solve=False,
        plot_profiles=True,
        use_sparsity=execution.use_sparsity,
        compute_numerical_sparsity=False,
        termination_mode=execution.termination_mode,
        print_cycle_build=False,
        cycle_options=cycle_options,
        start_cycle=cycle_idx,
    )
    print(f"Drawing the step profiles of saved CSS cycle #{cycle_idx} again (no re-solve).")
    plot_cycle_at_css(
        cycle_config=cycle_config,
        base_folder=results_base_folder,
        css_cycle_idx=cycle_idx,
        cycle_steps=execution.cycle_steps,
        termination_mode=cycle_run_config["run_kwargs"]["termination_mode"],
        step_options=cycle_run_config["step_options"],
        print_to_terminal=execution.print_to_terminal,
        persist_step_outputs=False,
    )
