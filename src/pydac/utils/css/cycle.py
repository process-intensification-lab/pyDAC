"""CSS cycle folders, overrides, diagnostics, and resume-state resolution."""

import json
import os
import re
import shutil
from pydac.utils.run import (
    infer_results_base_folder,
    normalize_termination_mode,
    validate_step_options_structure,
)
from pydac.utils.tvsa import (
    build_repeating_cycle_steps,
    get_step_solution_path,
    normalize_step_options as _normalize_step_options,
    normalize_step_token as _normalize_step_token,
    resolve_spatial_base_folder,
)

from pydac.utils.css.profiles import extract_final_fields, final_profiles
from pydac.utils.postprocessing.terminal_state import (
    TERMINAL_STATE_FILENAME,
    load_terminal_state,
    terminal_final_fields,
    terminal_profiles,
)
from pydac.utils.solver.persistence import MOO_COMPACT, RUN_FULL


CYCLE_OPTION_BOOL_KEYS = {
    "solve",
    "plot_profiles",
    "use_sparsity",
    "compute_numerical_sparsity",
    "print_cycle_build",
}
SUPPORTED_CYCLE_OPTION_KEYS = CYCLE_OPTION_BOOL_KEYS | {
    "termination_mode",
    "step_options",
    "rerun_from_step",
}

def _resolve_spatial_results_base_folder(base_folder, create=True, grid_config=None):
    return resolve_spatial_base_folder(base_folder, create=create, grid_config=grid_config)


def _create_cycle_folder(results_base_folder, cycle_idx):
    path = os.path.join(results_base_folder, f"cycle_{cycle_idx}")
    os.makedirs(path, exist_ok=True)
    return path


def _discard_cycle_directories_before(results_base_folder, oldest_kept_idx):
    """Delete completed cycle folders older than `oldest_kept_idx`.

    Used while a CSS run is still going, so that a long run holds a bounded
    number of cycles on disk instead of one per cycle. The CSS check itself
    carries the previous cycle in memory; what needs the folder is the
    after-the-fact diagnostic, which compares the retained cycle against the
    one before it.
    """
    removed = []
    for name in sorted(os.listdir(results_base_folder)):
        match = re.fullmatch(r"cycle_(\d+)", name)
        path = os.path.join(results_base_folder, name)
        if match is None or not os.path.isdir(path):
            continue
        if int(match.group(1)) >= oldest_kept_idx:
            continue
        shutil.rmtree(path)
        removed.append(name)
    return removed


def _find_existing_cycle_folder(base_folder, cycle_idx, grid_config=None):
    candidates = [
        os.path.join(_resolve_spatial_results_base_folder(base_folder, create=False, grid_config=grid_config), f"cycle_{cycle_idx}"),
        os.path.join(base_folder, f"cycle_{cycle_idx}"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def _resolve_cycle_folder(base_folder, cycle_idx):
    candidates = [
        os.path.join(infer_results_base_folder(base_folder), f"cycle_{cycle_idx}"),
        os.path.join(base_folder, f"cycle_{cycle_idx}"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def _cycle_folder_is_spatial_root(cycle_folder):
    return not any(
        os.path.isdir(os.path.join(cycle_folder, mode_tag))
        for mode_tag in ("1D_mode", "2D_mode")
    )


def _write_cycle_metadata(cycle_folder, step_records, cycle_total_time_s, css_probe_step):
    metadata = {
        "executed_steps": [record["step"] for record in step_records],
        "step_records": step_records,
        "cycle_total_time_s": float(cycle_total_time_s),
        "css_probe_step": css_probe_step,
    }
    policies = {
        record.get("persistence_policy")
        for record in step_records
        if record.get("persistence_policy") is not None
    }
    if len(policies) == 1:
        metadata["persistence_policy"] = policies.pop()
    metadata_path = os.path.join(cycle_folder, "cycle_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _get_step_solution_path_from_cycle_folder(tvsa_cycle, cycle_folder, step_name):
    step_key = _normalize_step_token(step_name)
    return os.path.join(
        infer_results_base_folder(cycle_folder),
        step_key,
        "sol.pkl",
    )


def _resolve_css_probe_step(tvsa_cycle, repeating_steps, css_probe_step):
    if css_probe_step is None:
        return repeating_steps[-1]

    probe_step = _normalize_step_token(css_probe_step)
    if probe_step == "startup":
        raise ValueError(
            "`css_probe_step` cannot be `startup` because startup is not part of the repeating CSS cycle. "
            "Use a repeated step such as `ads`, `press`, or the terminal step of the cycle."
        )
    if probe_step not in repeating_steps:
        raise ValueError(
            f"`css_probe_step` = '{probe_step}' is not part of the repeating cycle steps {repeating_steps}."
        )
    return probe_step


def _print_css_diagnostics(cycle_idx, probe_step, css_tol, diagnostics):
    if not diagnostics:
        return

    limiting_field, limiting_metrics = max(
        diagnostics.items(),
        key=lambda item: item[1]["max_rel_diff"],
    )
    max_rel_pct = 100.0 * limiting_metrics["max_rel_diff"]
    tol_pct = 100.0 * css_tol
    violations = limiting_metrics["violations"]
    print(
        f"CSS check @ cycle {cycle_idx}, step '{probe_step}': "
        f"max rel diff = {max_rel_pct:.3f}% in {limiting_field} "
        f"({violations} violating cells, tol = {tol_pct:.3f}%)."
    )


def _validate_cycle_options_structure(cycle_options):
    if cycle_options is None:
        return

    if not isinstance(cycle_options, dict):
        raise TypeError("cycle_options must be a dict[int, dict] or None.")

    for cycle_idx, opts in cycle_options.items():
        if isinstance(cycle_idx, bool) or not isinstance(cycle_idx, int):
            raise TypeError("Each cycle_options key must be an integer cycle index.")
        if cycle_idx < 1:
            raise ValueError("Each cycle_options key must be >= 1.")
        if not isinstance(opts, dict):
            raise TypeError(
                f"cycle_options[{cycle_idx}] must be a dict of cycle-specific overrides."
            )

        unknown_keys = set(opts.keys()) - SUPPORTED_CYCLE_OPTION_KEYS
        if unknown_keys:
            supported = ", ".join(sorted(SUPPORTED_CYCLE_OPTION_KEYS))
            bad = ", ".join(sorted(unknown_keys))
            raise ValueError(
                f"Unsupported key(s) in cycle_options[{cycle_idx}]: {bad}. "
                f"Supported keys: {supported}."
            )

        for key in sorted(CYCLE_OPTION_BOOL_KEYS & set(opts.keys())):
            if not isinstance(opts[key], bool):
                raise TypeError(
                    f"cycle_options[{cycle_idx}]['{key}'] must be bool. "
                    f"Received type: {type(opts[key]).__name__}."
                )

        if "termination_mode" in opts:
            normalize_termination_mode(
                opts["termination_mode"],
                option_name=f"cycle_options[{cycle_idx}]['termination_mode']",
                allow_none=False,
            )

        if "step_options" in opts:
            validate_step_options_structure(opts["step_options"])

        if "rerun_from_step" in opts:
            rerun_from_step = opts["rerun_from_step"]
            if rerun_from_step is not None and (
                not isinstance(rerun_from_step, str) or not rerun_from_step.strip()
            ):
                raise TypeError(
                    f"cycle_options[{cycle_idx}]['rerun_from_step'] must be a non-empty string or None."
                )


def _build_executed_steps(repeating_steps, cycle_idx):
    if cycle_idx == 1:
        return ["startup"] + list(repeating_steps[1:])
    return list(repeating_steps)


def _resolve_cycle_step_options(tvsa_cycle, cycle_step_options):
    return {
        step_key: dict(opts)
        for step_key, opts in _normalize_step_options(cycle_step_options).items()
    }


def _apply_rerun_from_step(tvsa_cycle, cycle_idx, executed_steps, step_options, rerun_from_step):
    effective_step_options = {
        step_key: dict(opts)
        for step_key, opts in step_options.items()
    }

    if rerun_from_step is None:
        return effective_step_options, None

    rerun_step_key = _normalize_step_token(rerun_from_step)
    if rerun_step_key not in executed_steps:
        raise ValueError(
            f"cycle_options[{cycle_idx}]['rerun_from_step'] = '{rerun_step_key}' is not executed "
            f"in cycle {cycle_idx}. Executed steps: {executed_steps}."
        )

    rerun_idx = executed_steps.index(rerun_step_key)

    # Steps before the rerun point are always loaded from disk.
    for step_key in executed_steps[:rerun_idx]:
        current = dict(effective_step_options.get(step_key, {}))
        current["solve"] = False
        effective_step_options[step_key] = current

    # From the rerun point onward, solve by default unless the user has
    # explicitly provided a per-step `solve` override for that step.
    for step_key in executed_steps[rerun_idx:]:
        current = dict(effective_step_options.get(step_key, {}))
        current.setdefault("solve", True)
        effective_step_options[step_key] = current

    return effective_step_options, rerun_step_key


def _resolve_cycle_run_config(
    tvsa_cycle,
    cycle_idx,
    repeating_steps,
    *,
    solve,
    plot_profiles,
    use_sparsity,
    compute_numerical_sparsity,
    termination_mode,
    print_cycle_build,
    cycle_options,
    start_cycle,
):
    cycle_override = dict((cycle_options or {}).get(cycle_idx, {}))
    cycle_step_options = cycle_override.pop("step_options", None)
    rerun_from_step = cycle_override.pop("rerun_from_step", None)

    executed_steps = _build_executed_steps(repeating_steps, cycle_idx)
    effective_step_options = _resolve_cycle_step_options(
        tvsa_cycle=tvsa_cycle,
        cycle_step_options=cycle_step_options,
    )
    effective_step_options, rerun_step_key = _apply_rerun_from_step(
        tvsa_cycle=tvsa_cycle,
        cycle_idx=cycle_idx,
        executed_steps=executed_steps,
        step_options=effective_step_options,
        rerun_from_step=rerun_from_step,
    )

    run_kwargs = {
        "solve": solve,
        "plot_profiles": plot_profiles,
        "use_sparsity": use_sparsity,
        "compute_numerical_sparsity": compute_numerical_sparsity,
        "termination_mode": normalize_termination_mode(
            termination_mode,
            allow_none=True,
        ),
        "print_cycle_build": print_cycle_build and cycle_idx == start_cycle,
    }
    for key in CYCLE_OPTION_BOOL_KEYS:
        if key in cycle_override:
            run_kwargs[key] = cycle_override[key]
    if "termination_mode" in cycle_override:
        run_kwargs["termination_mode"] = normalize_termination_mode(
            cycle_override["termination_mode"],
            option_name=f"cycle_options[{cycle_idx}]['termination_mode']",
            allow_none=False,
        )

    step_plot_requested = any(
        bool(opts.get("plot_profiles", False))
        for opts in effective_step_options.values()
    )
    requested_cycle_plots = bool(run_kwargs["plot_profiles"] or step_plot_requested)
    force_live_cycle_plots = requested_cycle_plots

    return {
        "executed_steps": executed_steps,
        "step_options": effective_step_options,
        "run_kwargs": run_kwargs,
        "rerun_step_key": rerun_step_key,
        "requested_cycle_plots": requested_cycle_plots,
        "force_live_cycle_plots": force_live_cycle_plots,
    }


def _load_resume_prev_fields(tvsa_cycle, base_folder, previous_cycle_idx, previous_executed_steps):
    if previous_cycle_idx < 1:
        return None

    previous_cycle_folder = _find_existing_cycle_folder(base_folder, previous_cycle_idx, grid_config=tvsa_cycle.column_factory.config.grid)
    terminal_step = previous_executed_steps[-1]
    sol_path = _get_step_solution_path_from_cycle_folder(
        tvsa_cycle=tvsa_cycle,
        cycle_folder=previous_cycle_folder,
        step_name=terminal_step,
    )
    persistence_policy = getattr(tvsa_cycle, "persistence_policy", RUN_FULL)
    state_path = (
        os.path.join(os.path.dirname(sol_path), TERMINAL_STATE_FILENAME)
        if persistence_policy == MOO_COMPACT
        else sol_path
    )
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Missing {state_path} - cannot resume cycle {previous_cycle_idx + 1}."
        )
    if persistence_policy == MOO_COMPACT:
        return terminal_final_fields(load_terminal_state(state_path))
    return extract_final_fields(state_path)


def _load_resume_prev_profile(
    tvsa_cycle,
    base_folder,
    previous_cycle_idx,
    previous_executed_steps,
    probe_step,
):
    if previous_cycle_idx < 1 or probe_step not in previous_executed_steps:
        return None

    previous_cycle_folder = _find_existing_cycle_folder(base_folder, previous_cycle_idx, grid_config=tvsa_cycle.column_factory.config.grid)
    sol_path = _get_step_solution_path_from_cycle_folder(
        tvsa_cycle=tvsa_cycle,
        cycle_folder=previous_cycle_folder,
        step_name=probe_step,
    )
    persistence_policy = getattr(tvsa_cycle, "persistence_policy", RUN_FULL)
    state_path = (
        os.path.join(os.path.dirname(sol_path), TERMINAL_STATE_FILENAME)
        if persistence_policy == MOO_COMPACT
        else sol_path
    )
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Missing {state_path} - cannot load the CSS baseline for cycle {previous_cycle_idx}."
        )
    if persistence_policy == MOO_COMPACT:
        return terminal_profiles(load_terminal_state(state_path))
    return final_profiles(state_path)


def _load_previous_cycle_terminal_fields(adsorption_cycle, base_folder, css_cycle_idx, cycle_steps):
    if css_cycle_idx <= 1:
        return None

    previous_cycle_folder = _resolve_cycle_folder(base_folder, css_cycle_idx - 1)
    repeating_steps = build_repeating_cycle_steps(cycle_steps=cycle_steps)
    previous_terminal_step = repeating_steps[-1]
    sol_path = get_step_solution_path(
        previous_cycle_folder,
        previous_terminal_step,
        base_folder_is_spatial_root=adsorption_cycle.base_folder_is_spatial_root,
    )

    if not os.path.exists(sol_path):
        raise FileNotFoundError(
            f"Missing {sol_path} while reconstructing the previous cycle state for CSS plotting."
        )

    return extract_final_fields(sol_path)

def resolve_spatial_results_base_folder(base_folder, create=True, grid_config=None):
    return _resolve_spatial_results_base_folder(base_folder, create=create, grid_config=grid_config)


def create_cycle_folder(results_base_folder, cycle_idx):
    return _create_cycle_folder(results_base_folder=results_base_folder, cycle_idx=cycle_idx)


# ---------------------------------------------------------------------------
# Cycle metadata
# ---------------------------------------------------------------------------

def write_cycle_metadata(cycle_folder, step_records, cycle_total_time_s, css_probe_step):
    _write_cycle_metadata(
        cycle_folder=cycle_folder,
        step_records=step_records,
        cycle_total_time_s=cycle_total_time_s,
        css_probe_step=css_probe_step,
    )


# ---------------------------------------------------------------------------
# CSS probe step & diagnostics
# ---------------------------------------------------------------------------

def resolve_css_probe_step(tvsa_cycle, repeating_steps, css_probe_step):
    return _resolve_css_probe_step(
        tvsa_cycle=tvsa_cycle,
        repeating_steps=repeating_steps,
        css_probe_step=css_probe_step,
    )


def print_css_diagnostics(cycle_idx, probe_step, css_tol, diagnostics):
    _print_css_diagnostics(
        cycle_idx=cycle_idx,
        probe_step=probe_step,
        css_tol=css_tol,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Cycle options validation
# ---------------------------------------------------------------------------

def validate_cycle_options_structure(cycle_options):
    _validate_cycle_options_structure(cycle_options)


# ---------------------------------------------------------------------------
# Cycle step execution logic
# ---------------------------------------------------------------------------

def build_executed_steps(repeating_steps, cycle_idx):
    return _build_executed_steps(repeating_steps=repeating_steps, cycle_idx=cycle_idx)


def resolve_cycle_run_config(
    tvsa_cycle,
    cycle_idx,
    repeating_steps,
    *,
    solve,
    plot_profiles,
    use_sparsity,
    compute_numerical_sparsity,
    termination_mode,
    print_cycle_build,
    cycle_options,
    start_cycle,
):
    return _resolve_cycle_run_config(
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


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_resume_prev_fields(tvsa_cycle, base_folder, previous_cycle_idx, previous_executed_steps):
    return _load_resume_prev_fields(
        tvsa_cycle=tvsa_cycle,
        base_folder=base_folder,
        previous_cycle_idx=previous_cycle_idx,
        previous_executed_steps=previous_executed_steps,
    )


def load_resume_prev_profile(
    tvsa_cycle,
    base_folder,
    previous_cycle_idx,
    previous_executed_steps,
    probe_step,
):
    return _load_resume_prev_profile(
        tvsa_cycle=tvsa_cycle,
        base_folder=base_folder,
        previous_cycle_idx=previous_cycle_idx,
        previous_executed_steps=previous_executed_steps,
        probe_step=probe_step,
    )


def discard_cycle_directories_before(results_base_folder, oldest_kept_idx):
    return _discard_cycle_directories_before(
        results_base_folder=results_base_folder,
        oldest_kept_idx=oldest_kept_idx,
    )
