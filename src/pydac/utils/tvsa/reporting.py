"""Terminal summaries for standalone and cyclic TVSA execution."""

from __future__ import annotations

from pydac.utils.tvsa.steps import STEP_LABEL_BY_KEY, STEP_UNIT_BY_KEY
from pydac.utils.tvsa.termination import (
    describe_step_termination,
    format_termination_outcome,
)


def print_cycle_build_summary(
    steps,
    common_kwargs,
    step_options,
    *,
    cyclic_mode,
    cycle_idx,
    config,
):
    """Print the final sequence and effective options before execution."""
    print("\n" + "=" * 120)
    print("                 Cycle Builder Configuration                 ".center(120))
    print("-" * 120)

    if "s_des" in steps and "w_des" in steps:
        des_mode = "Mixed explicit desorption steps"
    elif "s_des" in steps:
        des_mode = "Steam-assisted (s_des)"
    elif "w_des" in steps:
        des_mode = "Wall-heated (w_des)"
    else:
        des_mode = "No desorption step in sequence"
    sequence = " -> ".join(STEP_LABEL_BY_KEY.get(step, step) for step in steps)
    if cyclic_mode:
        cycle_tag = f"Cyclic mode (cycle {cycle_idx})"
    else:
        cycle_tag = "Step mode"
    print(f"Run mode            : {cycle_tag}")
    print(f"Startup included    : {'startup' in steps}")
    if cyclic_mode and cycle_idx == 1:
        print("Cycle rule          : Startup replaces the leading adsorption step in cycle 1")
    print(f"Desorption mode     : {des_mode}")
    print(f"Built step sequence : {sequence}")

    print("-" * 120)
    print("Effective step options")
    print("-" * 120)
    for idx, step in enumerate(steps, start=1):
        effective = dict(common_kwargs)
        effective.update(step_options.get(step, {}))
        termination_setup = describe_step_termination(
            config,
            step,
            termination_mode=effective.get("termination_mode"),
        )
        label = STEP_LABEL_BY_KEY.get(step, step)
        s = str(bool(effective.get("solve", False)))
        pl = str(bool(effective.get("plot_profiles", False)))
        us = str(bool(effective.get("use_sparsity", False)))
        cs = str(bool(effective.get("compute_numerical_sparsity", False)))
        print(
            f"{idx:>2}. {label:<18} | "
            f"solve={s:<5} | "
            f"plot_profiles={pl:<5} | "
            f"use_sparsity={us:<5} | "
            f"compute_numerical_sparsity={cs:<5} | "
            f"termination={termination_setup['display']}"
        )

    print("=" * 120)


def print_cycle_completion(cycle_total_time_s):
    """Print the end-of-cycle completion banner."""
    print("\n" + "=" * 120)
    print("TVSA Cycle Successfully Completed".center(120))
    print("-" * 120)
    print(f"Total elapsed time: {cycle_total_time_s / 60:.2f} minutes".center(120))
    print("=" * 120)


def print_cycle_timing_summary(step_records, cycle_total_time_s):
    """Print simulated step durations and actual termination outcomes."""
    print("\n" + "-" * 120)
    print("          TVSA Cycle Step Timing And Termination Summary           ".center(120))
    print("-" * 120)

    for rec in step_records:
        step = rec["step"]
        t_s = rec["time_s"]
        label = STEP_LABEL_BY_KEY.get(step, step)
        unit = STEP_UNIT_BY_KEY.get(step, "s")
        factor = 3600.0 if unit == "hr" else 60.0 if unit == "min" else 1.0
        configured = rec.get("termination_display", rec.get("termination_mode", "unknown"))
        stopped = format_termination_outcome(rec.get("termination"))
        print(
            f"{label:<18}: {t_s / factor:.2f} {unit:<3} | "
            f"configured={configured} | "
            f"stopped={stopped}"
        )

    print("-" * 120)
    total_cycle_sim_time = sum(rec["time_s"] for rec in step_records)
    print(f"Total cycle time   : {total_cycle_sim_time / 3600:.2f} hr")
    print(f"Clock time elapsed : {cycle_total_time_s:.2f} s")
    print("=" * 120)


__all__ = [
    "print_cycle_build_summary",
    "print_cycle_completion",
    "print_cycle_timing_summary",
]
