"""
pyDAC - TVSA cycle orchestration.

This module defines `TVSA_Cycle`, which assembles step-specific column models,
normalizes step aliases, and executes startup plus repeating TVSA steps for both
standalone and cyclic runs.
It also coordinates solver execution, logging, and post-processing across saved
cycle folders.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

import os
import time
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.config.cycle import CycleConfig
from pydac.post_processing.context.simulation_context import SimulationContext
from pydac.post_processing.context.store import SimulationContextStore
from pydac.post_processing.context.runtime_store import RuntimeReportStore
from pydac.post_processing.profiles.step_post_processing import StepPostProcessor
from pydac.post_processing.metrics.step_metrics_store import StepMetricsStore
from pydac.core.solver import Solver
from pydac.utils.logger import log_to, terminal_only
from pydac.utils.solver.outcome import (
    compact_solver_outcome,
    is_state_domain_failure,
    require_final_solver_success,
)
from pydac.utils.solver.reporting import report_saved_solve_outcome
from pydac.utils.solver.persistence import (
    MOO_COMPACT,
    RUN_FULL,
    normalize_persistence_policy,
)
from pydac.utils.postprocessing.runtime_report import runtime_report_from_model
from pydac.utils.tvsa import (
    STEP_ALIAS_BY_TOKEN as _STEP_ALIAS_BY_TOKEN,
    STEP_LABEL_BY_KEY as _STEP_LABEL_BY_KEY,
    STEP_UNIT_BY_KEY as _STEP_UNIT_BY_KEY,
    DEFAULT_CYCLIC_SEQUENCE as _DEFAULT_CYCLIC_SEQUENCE,
    DEFAULT_STEP_MODE_SEQUENCE as _DEFAULT_STEP_MODE_SEQUENCE,
    build_cycle_steps as _build_cycle_steps,
    build_cyclic_cycle_steps as _build_cyclic_cycle_steps,
    build_repeating_cycle_steps as _build_repeating_cycle_steps,
    describe_step_termination as _describe_step_termination,
    describe_termination_setup_from_model as _describe_termination_setup_from_model,
    describe_termination_setup_from_summary as _describe_termination_setup_from_summary,
    format_termination_outcome as _format_termination_outcome,
    normalize_step_options as _normalize_step_options,
    normalize_termination_mode,
    print_cycle_build_summary as _print_cycle_build_summary,
    print_cycle_completion as _print_cycle_completion,
    print_cycle_timing_summary as _print_cycle_timing_summary,
    resolve_spatial_base_folder_from_solver_settings as _resolve_spatial_base_folder_from_solver_settings,
)

class TVSA_Cycle:
    # Step names, labels, and aliases used when building and reporting cycles.
    DEFAULT_STEP_MODE_SEQUENCE = list(_DEFAULT_STEP_MODE_SEQUENCE)
    DEFAULT_CYCLIC_SEQUENCE = list(_DEFAULT_CYCLIC_SEQUENCE)

    STEP_LABEL_BY_KEY = _STEP_LABEL_BY_KEY
    STEP_UNIT_BY_KEY = _STEP_UNIT_BY_KEY
    STEP_ALIAS_BY_TOKEN = _STEP_ALIAS_BY_TOKEN

    def __init__(
        self,
        cycle_config: CycleConfig,
        *,
        base_folder="simulation_results",
        cyclic_mode=False,
        cycle_idx=None,
        last_cycle_idx=None,
        print_to_terminal=True,
        base_folder_is_spatial_root=False,
        composition_domain_guard=None,
        persistence_policy=RUN_FULL,
    ):

        """
        Class representing a Temperature Vacuum Swing Adsorption (TVSA) cycle for CO₂ capture.

        Parameters
        ----------
        cycle_config : CycleConfig
            The resolved campaign configuration this cycle runs: every public
            simulation input, already validated. The cycle never resolves
            configuration of its own.
        base_folder : str, optional
            Directory to save simulation results, by default "simulation_results".
        cyclic_mode : bool, optional
            Whether the simulation is in cyclic mode, by default False.
        cycle_idx : int or None, optional
            Current cycle index if in cyclic mode, by default None.
        last_cycle_idx : int or None, optional
            Last cycle index if in cyclic mode, by default None.
        print_to_terminal : bool, optional
            Whether to print outputs to the terminal, by default True.
        base_folder_is_spatial_root : bool, optional
            When True, `base_folder` already points to the final 1D/2D results
            root and no extra `1D_mode`/`2D_mode` subfolder is added.

        Initializes the packed bed system.  Isotherm and kinetic model
        construction is owned by ``ColumnModelFactory``.
        """        
        if not isinstance(cycle_config, CycleConfig):
            raise TypeError("`cycle_config` must be a CycleConfig instance.")

        # Store the runtime config in a factory, which builds one model per step.
        self.column_factory = ColumnModelFactory(
            config=cycle_config
            )
        
        # Keep the output root and cycle position used by every step run.
        self.base_folder = base_folder
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
        print(f"Results will be saved in: {self.base_folder}")
        
        self.cyclic_mode = cyclic_mode
        self.cycle_idx = cycle_idx
        self.last_cycle_idx = last_cycle_idx
        self.print_to_terminal = print_to_terminal
        self.base_folder_is_spatial_root = bool(base_folder_is_spatial_root)
        self.composition_domain_guard = composition_domain_guard
        self.persistence_policy = normalize_persistence_policy(persistence_policy)
        self._step_final_profiles = {}
        # Step profile plots use the case's figure format, like the CSS and KPI figures.
        experiment = cycle_config.experiment
        self.plot_format = "pdf" if experiment is None else experiment.execution.plot_format

    def _should_plot_profiles(self, plot_profiles):
        """Return whether live profiles should be generated for this step."""
        # Avoid plotting every CSS cycle; standalone runs plot immediately.
        if not plot_profiles:
            return False
        if not self.cyclic_mode:
            return True
        return (
            self.cycle_idx is not None
            and self.last_cycle_idx is not None
            and self.cycle_idx == self.last_cycle_idx
        )

    def _run_step(
        self,
        step_name,
        solve=False,
        prev_fields=None,
        plot_profiles=False,
        use_sparsity=True,
        compute_numerical_sparsity=False,
        termination_mode=None,
        persist_step_outputs=True,
    ):
        """
        Internal helper to run any step of the TVSA cycle.

        Parameters
        ----------
        step_name : str
            One of ['startup', 'blow', 'heat', 'w_purge', 's_purge',
            'w_des', 's_des', 'press', 'cool', 'ads'].
        solve : bool
            Whether to solve the PDE model.
        prev_fields : dict or None
            Fields to use as initial condition from the previous step.
        plot_profiles : bool
            Whether to generate plots for this step.
        use_sparsity : bool
            Whether to use sparse linear algebra for this step.
        compute_numerical_sparsity : bool
            Whether to compute numerical sparsity snapshots for this step.
        termination_mode : str or None
            Optional override for the step termination mode: "time" or "event".
        persist_step_outputs : bool
            False only to redraw the plots of a saved step (``pydac.report``): the
            step log, step metrics and their provenance on disk are left untouched.
            Requires ``solve=False``.

        Notes
        -----
        Results saving and postprocessing are always enabled in this workflow:
        - If `solve=True`, solver output is saved to disk.
        - Postprocessing always runs and loads saved results when needed.
        
        Returns
        -------
        final_fields : dict
            Final state of the model at the end of this step.
        step_time : float
            Final time reached in the simulation [s].
        """
        persistence_policy = normalize_persistence_policy(
            getattr(self, "persistence_policy", RUN_FULL)
        )

        # Decide the folder for this step, including the 1D_mode/2D_mode suffix.
        run_base_folder = _resolve_spatial_base_folder_from_solver_settings(
            self.base_folder,
            base_folder_is_spatial_root=self.base_folder_is_spatial_root,
            grid_config=self.column_factory.config.grid,
        )

        if solve and not persist_step_outputs:
            raise ValueError("persist_step_outputs=False only redraws a saved step; it needs solve=False.")
        step_output = (
            log_to(step_name, run_base_folder, also_stdout=self.print_to_terminal, append=not solve)
            if persist_step_outputs
            else terminal_only(self.print_to_terminal)
        )

        with step_output:

            pretty_name = self.STEP_LABEL_BY_KEY.get(step_name, step_name.capitalize())

            print(f"\n{'='*100}")
            print(f"            Running {pretty_name} Step             ".center(100))
            print(f"{'='*100}")

            step_solution_path = os.path.join(run_base_folder, step_name, "sol.pkl")
            if solve:
                print("Execution mode      : solve=True (new simulation will be run)")
            else:
                if persistence_policy == MOO_COMPACT:
                    raise ValueError(
                        "`moo_compact` does not support solve=False step replay; "
                        "resume from a completed cycle terminal artifact or rerun the step."
                    )
                print("Execution mode      : solve=False (saved solution will be reused)")
                print(f"Saved solution path : {step_solution_path}")
                if not os.path.exists(step_solution_path):
                    raise FileNotFoundError(
                        f"solve=False for step '{step_name}', but no saved solution exists at "
                        f"'{step_solution_path}'. Set solve=True for this step or generate the file first."
                    )

            # Build this step's column model from the cycle configuration.
            solver_options = self.column_factory.config.solver_for_step(
                use_sparsity=use_sparsity,
                compute_numerical_sparsity=compute_numerical_sparsity,
            )
            adsorber = self.column_factory.build_column_model(
                step=step_name,
                initial_state=prev_fields,
                solver_options=solver_options,
            )
            normalized_termination_mode = normalize_termination_mode(
                termination_mode,
                allow_none=True,
            )
            if normalized_termination_mode is not None:
                adsorber.termination_mode = normalized_termination_mode

            # Show the stop criterion that the model will actually use.
            termination_setup = _describe_termination_setup_from_model(adsorber)

            print(f"Using spatial-mode results folder: {run_base_folder}")
            print(f"Termination setup   : {termination_setup['display']}")

            # Run or reload the step through the solver.
            solver = Solver(
                adsorber,
                base_folder=run_base_folder,
                composition_domain_guard=getattr(
                    self, "composition_domain_guard", None
                ),
            )

            # Save the static simulation context next to the step results.
            if solve:
                simulation_context = SimulationContext.from_model(adsorber)
                context_store = SimulationContextStore(run_base_folder)
                runtime_store = RuntimeReportStore(run_base_folder)
                runtime_report = (
                    runtime_store.from_model(adsorber)
                    if persistence_policy == RUN_FULL
                    else runtime_report_from_model(
                        adsorber,
                        persistence_policy=persistence_policy,
                    )
                )
                if context_store.exists():
                    context_store.validate_compatible(simulation_context)
                if runtime_store.exists():
                    runtime_store.validate_compatible(runtime_report)
                solver.solve_cvode()
                # A composition-domain event already carries its compact
                # dimensional crossing diagnostic. Do not materialize or
                # serialize a large invalid trajectory before rejecting it.
                if is_state_domain_failure(solver.sol):
                    require_final_solver_success(
                        solver.sol,
                        step=step_name,
                    )
                if persistence_policy == RUN_FULL:
                    solver.save_results()
                elif bool((solver.sol.get("cvode") or {}).get("success", False)):
                    solver.save_terminal_results()
                else:
                    print(
                        "Compact persistence: failed step keeps only its solver/"
                        "termination summary; no trajectory or terminal state is saved."
                    )
                # A failed final CVODE attempt is still valuable diagnostic
                # evidence, so save it first.  It must never reach step KPI
                # work or become the initial state of the following step.
                require_final_solver_success(
                    solver.sol,
                    step=step_name,
                )
                context_store.save(simulation_context)
                runtime_store.save(runtime_report)

            # Postprocess and persist typed metrics for this step (skipped when only
            # redrawing the plots of a saved step).
            if persist_step_outputs:
                from pydac.utils.provenance import new_record, source_record
                from pydac.config.serialization import to_jsonable
                processing_provenance = new_record(
                    "step_metrics", config=getattr(adsorber, "provenance_config", None),
                    sources=[source_record(solver.load_results().get("solve_provenance"))],
                    execution={"step": step_name, "operation": "compute",
                               "column_config": to_jsonable(adsorber.column_config)},
                )
            step_postprocessor = StepPostProcessor(adsorber, solver)
            step_postprocessor.describe()
            if persist_step_outputs:
                step_metrics = step_postprocessor.compute_step_metrics(step_name)
                metrics_store = StepMetricsStore(step_postprocessor.base_folder)
                metrics_store.show(step_metrics)
                metrics_store.save(
                    step_metrics, step_folder=step_name,
                    source_provenance=processing_provenance,
                )

            if self._should_plot_profiles(plot_profiles):
                step_postprocessor.plotter.plot_all_profiles(
                    save=True, show=False, fmt=self.plot_format
                )
                    
            # Return final fields and keep a small termination/timing record.
            final_fields = solver.get_final_fields()
            final_profile = solver.get_final_profile()
            if not hasattr(self, "_step_final_profiles"):
                self._step_final_profiles = {}
            self._step_final_profiles[step_name] = final_profile
            if not solve:
                report_saved_solve_outcome(solver.sol.get("cvode"))
            step_time = solver.sol["t"][-1]
            termination_summary = solver.sol.get("termination", None)
            recorded_termination_setup = _describe_termination_setup_from_summary(
                termination_summary,
                fallback=termination_setup,
            )
            actual_stop = _format_termination_outcome(termination_summary)
            self._last_step_run_record = {
                "step": step_name,
                "time_s": float(step_time),
                "termination_mode": recorded_termination_setup["mode"],
                "termination_display": recorded_termination_setup["display"],
                "termination_reason": (
                    termination_summary.get("reason", "unknown")
                    if isinstance(termination_summary, dict)
                    else "unknown"
                ),
                "termination": termination_summary,
                "solver": compact_solver_outcome(
                    solver.sol,
                    step=step_name,
                ),
                "persistence_policy": persistence_policy,
                "state_artifact": (
                    f"{step_name}/sol.pkl"
                    if persistence_policy == RUN_FULL
                    else f"{step_name}/terminal.pkl"
                ),
            }

            print(f"\n{'-'*100}")
            print(f"             {pretty_name} Step Complete            ".center(100))
            print(f"Configured stop     : {recorded_termination_setup['display']}")
            print(f"Actual stop         : {actual_stop}")
            print(f"{'-'*100}")

            if persistence_policy == MOO_COMPACT:
                # The post-processor owns large derived arrays and the solver owns
                # the prepared Nt-history. Release both before the next step starts.
                del step_postprocessor
                solver.release_results()
        
        return final_fields, step_time

    def final_profile_for_step(self, step_name):
        """Return the in-memory terminal primitive profile for a completed step."""
        step_key = str(step_name).strip().lower()
        try:
            return self._step_final_profiles[step_key]
        except KeyError as exc:
            raise KeyError(
                f"No terminal profile is available for completed step {step_key!r}."
            ) from exc
    
    def run_cycle(
        self,
        *,
        solve=False,
        plot_profiles=False,
        use_sparsity=True,
        compute_numerical_sparsity=False,
        termination_mode=None,
        cycle_steps=None,
        step_options=None,
        initial_prev_fields=None,
        print_cycle_build=True,
        print_summary=True,
    ):
        """
        Execute a full cycle from a step sequence.

        Parameters
        ----------
        solve : bool
            Whether to solve the PDE model in each step.
        plot_profiles : bool
            Whether to generate profile plots in each step.
        use_sparsity : bool
            Whether to enable sparse linear algebra in each step by default.
        compute_numerical_sparsity : bool
            Whether to compute numerical sparsity snapshots in each step by default.
        termination_mode : str or None
            Optional cycle-level step termination mode override. When set to
            "time" or "event", every step uses that mode unless overridden in
            `step_options`.
        cycle_steps : list[str] or None
            Optional list of explicit step tokens.
        step_options : dict[str, dict] or None
            Optional per-step kwargs override. Keys can be explicit aliases (e.g.
            "ADSORPTION", "BLOWDOWN", "STEAM-ASSISTED-DESORPTION", etc.) and are normalized.
            Supported per-step override keys in the TVSA step pipeline:
            `solve`, `plot_profiles`, `use_sparsity`,
            `compute_numerical_sparsity`, `termination_mode`.
        initial_prev_fields : dict or None
            Optional initial state to inject before the first executed step.
            This is used by cyclic restart workflows when resuming from a
            previously solved cycle.
        print_cycle_build : bool
            Print custom cycle build summary before running the cycle.
        print_summary : bool
            Print end-of-cycle banner and timing table.

        Returns
        -------
        tuple
            (final_fields, step_records, cycle_total_time_s)
            where step_records is a list of:
            [{"step": "<name>", "time_s": <float>, "termination_mode": "<time|event>", ...}, ...]
        """
        # Build the ordered list of steps for this standalone or CSS cycle.
        if self.cyclic_mode:
            steps = _build_cyclic_cycle_steps(
                self.cycle_idx,
                cycle_steps=cycle_steps,
            )
        else:
            steps = _build_cycle_steps(cycle_steps=cycle_steps)
        step_options = _normalize_step_options(step_options)

        # Defaults used for every step unless step_options overrides them.
        prev_fields = initial_prev_fields
        final_fields = None
        step_records = []

        common_kwargs = {
            "solve": solve,
            "plot_profiles": plot_profiles,
            "use_sparsity": use_sparsity,
            "compute_numerical_sparsity": compute_numerical_sparsity,
            "termination_mode": normalize_termination_mode(
                termination_mode,
                allow_none=True,
            ),
        }

        if print_cycle_build:
            _print_cycle_build_summary(
                steps=steps,
                common_kwargs=common_kwargs,
                step_options=step_options,
                cyclic_mode=self.cyclic_mode,
                cycle_idx=self.cycle_idx,
                config=self.column_factory.config,
            )

        cycle_start_time = time.time()
        for step in steps:
            # Use the previous step's final fields as this step's initial state.
            kwargs = dict(common_kwargs)
            if prev_fields is not None:
                kwargs["prev_fields"] = prev_fields

            if step in step_options:
                kwargs.update(step_options[step])

            # Print what this step is about to run.
            termination_setup = _describe_step_termination(
                self.column_factory.config,
                step,
                termination_mode=kwargs.get("termination_mode"),
            )
            step_label = self.STEP_LABEL_BY_KEY.get(step, step)
            run_label = (
                f"Cycle {self.cycle_idx}" if self.cyclic_mode and self.cycle_idx is not None else "TVSA"
            )
            print(
                f"[{run_label}] Starting step '{step_label}' with "
                f"solve={bool(kwargs.get('solve', False))}, "
                f"termination={termination_setup['display']}."
            )

            final_fields, step_time = self._run_step(step, **kwargs)
            prev_fields = final_fields
            # Store one compact record for summaries and metadata.
            step_record = dict(getattr(self, "_last_step_run_record", {}) or {})
            if not step_record:
                step_record = {
                    "step": step,
                    "time_s": float(step_time),
                    "termination_mode": termination_setup["mode"],
                    "termination_display": termination_setup["display"],
                    "termination_reason": "unknown",
                    "termination": None,
                }
            else:
                step_record.setdefault("step", step)
                step_record.setdefault("time_s", float(step_time))
                step_record.setdefault("termination_mode", termination_setup["mode"])
                step_record.setdefault("termination_display", termination_setup["display"])
                step_record.setdefault("termination_reason", "unknown")
                step_record.setdefault("termination", None)
            step_records.append(step_record)

            actual_stop = _format_termination_outcome(step_record.get("termination"))
            print(f"[{run_label}] Completed step '{step_label}' with stop={actual_stop}.")

        cycle_total_time_s = time.time() - cycle_start_time

        if print_summary:
            _print_cycle_completion(cycle_total_time_s)
            _print_cycle_timing_summary(step_records, cycle_total_time_s)

        return final_fields, step_records, cycle_total_time_s

TVSACycle = TVSA_Cycle
AdsorptionCycle = TVSA_Cycle
