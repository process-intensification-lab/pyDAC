"""
pyDAC - Standalone TVSA step-sequence entry point.

This script runs a startup plus user-selected TVSA steps without the outer CSS
loop and is the main driver for one-off simulations or step-by-step debugging.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

import os

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


from pydac.utils.run import validate_flags
from pydac.core.TVSA_cycle import TVSA_Cycle
from pydac.config import ExperimentConfig
from pydac.config.launch import resolve_experiment_config
from pydac.config.loader import write_resolved_config
from pydac.utils.tvsa import resolve_spatial_base_folder

def run_steps_simulation(
    experiment_config: ExperimentConfig,
    *,
    base_folder=None,
):
    """
    Run one standalone TVSA step sequence from a resolved experiment.

    Parameters:
    - experiment_config (ExperimentConfig): The resolved public document. It
            states the physics, the grid, the cycle, and what the run does.
    - base_folder (str or None): Directory to save simulation results. None
      uses `execution.output_root`.

        Run behavior is configured in the document: use `execution.*`, `grid.*`,
        and `solver.*` settings before calling this workflow rather than passing
        ad-hoc keyword overrides here. That keeps the API and saved
        `resolved_config.json` identical.
    """

    # Entry point for a standalone step run: resolve public config to runtime config.
    resolved, cycle_config = resolve_experiment_config(
        experiment_config=experiment_config,
    )

    # Read the run options from the execution section of the resolved document.
    execution = resolved.execution
    solve = execution.solve
    plot_profiles = execution.plot_profiles
    print_to_terminal = execution.print_to_terminal
    print_cycle_build = execution.print_cycle_build
    use_sparsity = execution.use_sparsity
    compute_numerical_sparsity = execution.compute_numerical_sparsity
    termination_mode = execution.termination_mode
    cycle_steps = execution.cycle_steps
    step_options = dict(execution.step_options) or None
    # Returns True or raises; there is no silent "invalid" outcome to handle.
    validate_flags(
        solve=solve,
        plot_profiles=plot_profiles,
        use_sparsity=use_sparsity,
        compute_numerical_sparsity=compute_numerical_sparsity,
        termination_mode=termination_mode,
        step_options=step_options,
    )

    # Save the exact case that will run before creating models or solver output.
    base_folder = execution.output_root if base_folder is None else base_folder
    results_base_folder = resolve_spatial_base_folder(
        base_folder,
        grid_config=cycle_config.grid,
    )
    write_resolved_config(resolved, os.path.join(results_base_folder, "resolved_config.json"))
    from pydac.utils.provenance import start_activity
    start_activity(results_base_folder, "run", config=resolved,
                   execution={"workflow": "steps", "solve": solve})

    # Create the cycle runner for this standalone sequence.
    tvsa_cycle = TVSA_Cycle(
        cycle_config,
        base_folder=base_folder,
        print_to_terminal=print_to_terminal,
        cyclic_mode=False,
    )

    print("\n" + "="*120)
    print("                Starting Adsorption Cycle                 ".center(120))
    print("="*120)

    # Execute the configured step sequence and return its final state/records.
    final_fields, step_records, cycle_total_time = tvsa_cycle.run_cycle(
        solve=solve,
        plot_profiles=plot_profiles,
        use_sparsity=use_sparsity,
        compute_numerical_sparsity=compute_numerical_sparsity,
        termination_mode=termination_mode,
        print_cycle_build=print_cycle_build,
        cycle_steps=cycle_steps,
        step_options=step_options,
    )

    return {
        "final_fields": final_fields,
        "step_records": step_records,
        "cycle_total_time_s": cycle_total_time,
    }
