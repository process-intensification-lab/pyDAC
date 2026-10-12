"""
pyDAC - Time integration of TVSA packed-bed steps.

This module wraps CVODE for `ColumnModelMoL`, handles dimensionalization,
sparsity loading, event-driven termination, and solution export for each
simulated TVSA step.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

#YOOOOOOOOOOOOOOOOO

import numpy as np
import time
import pickle
import os
from sksundae.cvode import CVODE
from pydac.utils.solver import (
    build_termination_summary,
    configure_cvode_linear_solver_options,
    format_metric_value,
    make_cvode_events_callback,
    pressure_drop_metadata_from_model,
)
from pydac.utils.solver.event_config import build_event_specs_from_model
from pydac.utils.solver.jacobian import finite_difference_jacobian
from pydac.utils.solver.reporting import report_initial_event_skip, report_solve_outcome
from pydac.utils.solver.outcome import compact_solver_outcome
from pydac.utils.postprocessing.terminal_state import (
    TERMINAL_STATE_FILENAME,
    save_terminal_state,
    terminal_state_from_solution,
)
from pydac.utils.postprocessing.saved_solution import (
    SAVED_SOLUTION_SCHEMA_KEY,
    SAVED_SOLUTION_SCHEMA_VERSION,
)
from pydac.core.outlet_reporting import outlet_face_history
from pydac.utils.provenance import new_record, save_record, source_record


INITIAL_EVENT_SAMPLE_TIME_S = 1e-12


class Solver:
    def __init__(
        self,
        model,
        base_folder="simulation_results",
        *,
        composition_domain_guard=None,
    ):
        """
        Initialize the Solver class with a model instance and output folder.

        Parameters:
        ----------
        model : ColumnModelMoL
            Instance of the packed-bed model already initialized.
        base_folder : str
            Path to the folder where results will be saved.
        """
        if model is None:
            raise ValueError("No Model Instance Provided.")
        
        # Model and output folder
        self.model = model
        self.base_folder = base_folder
        self.step = model.step
        self.composition_domain_guard = composition_domain_guard
        self._configure_time(model)
        self._cache_grid_data(model)
        self._cache_reference_scales(model)
        self._cache_solver_configuration(model)
        self._bind_model_interfaces(model)
        self.event_specs = build_event_specs_from_model(
            model=model,
            step=self.step,
            A_rings=self.A_rings,
            V_cells=self.V_cells,
            p_ref=self.p_ref,
            T_ref=self.T_ref,
            composition_domain_guard=self.composition_domain_guard,
        )

    def _configure_time(self, model):
        """Resolve the dimensional and nondimensional solve horizon."""
        self.termination_mode = str(getattr(model, "termination_mode", "time")).lower()
        self.t_end_nominal = float(getattr(model, "t_end", 0.0))
        raw_t_end_max = getattr(model, "t_end_max", None)
        self.t_end_max = None if raw_t_end_max is None else float(raw_t_end_max)

        # Time discretization
        self.tau_hyd = model.tau_hyd
        self.t_end = (
            self.t_end_max
            if self.termination_mode == "event" and self.t_end_max is not None
            else self.t_end_nominal
        )
        self.nondim_horizon = self.t_end / self.tau_hyd

        # The model RHS still reads this attribute for progress reporting.
        self.model.nondim_horizon = self.nondim_horizon
        self.t_eval = np.linspace(0.0, self.nondim_horizon, model.Nt)

    def _cache_grid_data(self, model):
        """Cache grid dimensions and geometry needed during integration."""
        # Grid and structure
        self.state_layout = model.state_layout
        self.Nz = model.Nz
        self.Nr = model.Nr
        self.A_rings = np.asarray(model.A_rings, dtype=float)
        self.V_cells = np.asarray(model.V_cells, dtype=float)

    def _cache_reference_scales(self, model):
        """Cache dimensional reference scales used for result export."""
        self.state_scale_by_field = dict(model.state_scale_by_field)
        self.T_ref = model.T_ref
        self.p_ref = model.p_ref

    def _cache_solver_configuration(self, model):
        """Cache tolerances, termination settings, and linear-solver options."""
        # Solver configuration
        self.rtol = model.rtol
        self.atol = model.atol
        self.atol_scalar_mass = float(getattr(model, "atol_scalar_mass", min(self.atol, 1e-12)))
        self.atol_scalar_energy = float(getattr(model, "atol_scalar_energy", min(self.atol, 1e-6)))

    def _bind_model_interfaces(self, model):
        """Bind the model callbacks and initialize solve-time diagnostics."""
        # PDE and initial condition interfaces
        self.get_initial_state = model.get_initial_state
        self.pde_system = model.pde_system_kpis

        # Sparsity computation and events 
        self.sparsity_ticks_done = set()   # empty set at start
        self.compute_numerical_sparsity = model.compute_numerical_sparsity
        self.use_sparsity= model.use_sparsity
        self.cvode_sparsity_mode = getattr(model, "cvode_sparsity_mode", None)

    def _initial_state(self):
        """Return the complete initial vector in generated layout order."""
        pde_state = np.asarray(self.get_initial_state(), dtype=float)
        if pde_state.shape != (self.state_layout.pde_size,):
            raise ValueError(
                f"Model initial PDE state has shape {pde_state.shape}; expected "
                f"{(self.state_layout.pde_size,)} from StateLayout."
            )
        initial = np.zeros(self.state_layout.state_size, dtype=float)
        initial[: self.state_layout.pde_size] = pde_state
        return initial

    def _solver_constraints(self):
        """Build CVODE non-negativity constraints from layout field metadata."""
        constrained = [
            index
            for field in self.state_layout.fields
            if field.constraint == "nonnegative"
            for index in range(field.start, field.stop)
        ]
        return constrained, [1] * len(constrained)

    def _absolute_tolerances(self, initial_state):
        """Build per-state absolute tolerances from layout policy groups."""
        tolerances = np.full_like(initial_state, self.atol, dtype=float)
        values = {
            "trace_gas": 1e-12,
            "mass_scalar": self.atol_scalar_mass,
            "energy_scalar": self.atol_scalar_energy,
        }
        for field in self.state_layout.fields:
            if field.tolerance_group in values:
                tolerances[field.state_slice] = values[field.tolerance_group]
        return tolerances

    def pde_callback(self, t, y, ydot):
        """
        ODE system callback required by the CVODE solver.

        Parameters
        ----------
        t : float
            Current nondimensional solver time.
        y : np.ndarray
            Current state vector.
        ydot : np.ndarray
            Output array to be filled with time derivatives.
        """
        self.pde_system(t, y, out=ydot)

        if self.compute_numerical_sparsity:
            tick = int(100 * t / self.nondim_horizon)   # coarse binning
            targets = {0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}

            if tick in targets and tick not in self.sparsity_ticks_done:
                self.sparsity_ticks_done.add(tick)

                print(f"Computing numerical Jacobian and sparsity pattern at ~{tick}% ...")

                def f_flat(x):
                    return self.pde_system(t, x).copy()

                J_fd = finite_difference_jacobian(f_flat, y, eps=1e-6)
                # Diagnostic output belongs to the run: next to this step's
                # sol.pkl, never inside the installed package.
                folder_path = os.path.join(
                    self.base_folder, self.step, "numerical_jacobian"
                )
                os.makedirs(folder_path, exist_ok=True)
                np.save(
                    os.path.join(folder_path, f"jacobian_sparsity_{tick:03d}.npy"),
                    J_fd,
                )

                print(f"Sparsity snapshot saved with shape {J_fd.shape}")

    def _find_initial_event(self, Y0):
        """Return the first primary event already satisfied by ``Y0``."""
        # Safety/failure events take precedence over a simultaneously
        # satisfied ordinary step-termination event.
        indexed_specs = sorted(
            enumerate(self.event_specs),
            key=lambda item: not bool(item[1].get("is_failure", False)),
        )
        for index, spec in indexed_specs:
            if (
                spec.get("name") != "step_termination"
                and not bool(spec.get("is_failure", False))
            ):
                continue
            args = (0.0, Y0) if spec.get("metric_uses_time", False) else (Y0,)
            metric_nd = float(spec["metric_fn"](*args))
            residual = metric_nd - float(spec["threshold_nd"])
            if not np.isfinite(residual):
                continue
            if (spec["direction"] < 0 and residual <= 0.0) or (
                spec["direction"] > 0 and residual >= 0.0
            ):
                return index, metric_nd
        return None

    def _store_initial_event_solution(self, Y0, event_index, metric_nd):
        """Persist a no-op trajectory while retaining the normal output shape."""
        spec = self.event_specs[event_index]
        sample_time_nd = INITIAL_EVENT_SAMPLE_TIME_S / self.tau_hyd
        metric_value_dim = metric_nd * spec["ref"]
        event = {
            "name": spec["name"],
            "label": spec["label"],
            "metric_label": spec["metric_label"],
            "unit": spec["unit"],
            "threshold_dim": float(spec["threshold_dim"]),
            "threshold_str": format_metric_value(spec["unit"], spec["threshold_dim"]),
            "metric_value_dim": float(metric_value_dim),
            "metric_value_str": format_metric_value(spec["unit"], metric_value_dim),
            "t_event_s": 0.0,
            "initially_satisfied": True,
        }
        if "arming" in spec:
            event["arming"] = dict(spec["arming"])

        is_failure = bool(spec.get("is_failure", False))
        status = (
            str(spec.get("failure_status", "event_failure"))
            if is_failure
            else 2
        )
        message = (
            str(spec.get("failure_message", "A terminal failure event was reached."))
            if is_failure
            else "Initial terminal event already satisfied; CVODE skipped."
        )
        diagnostic_fn = spec.get("diagnostic_fn")
        diagnostic = None if diagnostic_fn is None else diagnostic_fn(Y0)
        event["metadata"] = dict(spec.get("metadata", {}))
        if diagnostic is not None:
            event["diagnostic"] = dict(diagnostic)

        self.sol = {
            "solver": "scikit",
            "cvode": {
                "success": not is_failure,
                "status": status,
                "message": message,
                "nfev": 0,
                "njev": 0,
                "wall_time_s": 0.0,
            },
            # Keep two equal samples because step post-processing integrates traces.
            "t": np.array([0.0, sample_time_nd], dtype=float),
            "y": np.column_stack((Y0, Y0)).copy(),
            "t_events": [np.float64(0.0)],
            "y_events": np.asarray([Y0], dtype=float),
            "i_events": np.asarray([[-(event_index + 1)]], dtype=np.int32),
            "termination": {
                "configured_mode": self.termination_mode,
                "reason": (
                    str(spec.get("failure_reason", "event_failure"))
                    if is_failure
                    else "event"
                ),
                "success": not is_failure,
                "status": status if is_failure else "initial_event",
                "message": message,
                "t_final_s": INITIAL_EVENT_SAMPLE_TIME_S,
                "event": event,
            },
        }

    def _handle_initial_event(self, Y0):
        """Store and report an initial event, returning whether solving is skipped."""
        initial_event = self._find_initial_event(Y0)
        if initial_event is None:
            return False
        event_index, metric_nd = initial_event
        self._store_initial_event_solution(Y0, event_index, metric_nd)
        report_initial_event_skip(
            failure=bool(self.event_specs[event_index].get("is_failure", False))
        )
        return True

    def solve_cvode(self):
        """
        Solve the PDE system using the CVODE solver from scikits.odes.
        Stores the time and state history in `self.sol`.
        """
        step_label = str(self.step).upper()
        print("\n" + "-"*60)
        print(f"Solving PDE System for {step_label}".center(60))
        print("-"*60)
        if self.termination_mode == "event":
            horizon_source = "t_end_max" if self.t_end_max is not None else "t_end"
            print(f"Event-mode maximum time ({horizon_source}) : {self.t_end:.2f} s")
        
        Y0 = self._initial_state()

        from pydac.config.serialization import to_jsonable
        import hashlib
        self.solve_provenance = new_record(
            "solve", config=getattr(self.model, "provenance_config", None),
            execution={
                "step": self.step,
                "column_config": to_jsonable(getattr(self.model, "column_config", None)),
                "composition_domain_guard": to_jsonable(self.composition_domain_guard),
                "termination_mode": self.termination_mode,
                "time_horizon_s": float(self.t_end),
                "initial_state": {"sha256": hashlib.sha256(np.ascontiguousarray(Y0).tobytes()).hexdigest(),
                                  "dtype": str(Y0.dtype), "shape": list(Y0.shape)},
            },
        )
        save_record(os.path.join(self.base_folder, self.step), self.solve_provenance)

        if self.event_specs and self._handle_initial_event(Y0):
            self.sol["solve_provenance"] = self.solve_provenance
            return

        # Gas fields remain unconstrained because central-difference dispersion
        # is not positivity-preserving; the RHS clamps only nonlinear uses.
        constraints_idx, constraints_type = self._solver_constraints()
        atol_array = self._absolute_tolerances(Y0)
        
        options = {
            "rtol": self.rtol,
            "atol": atol_array,
            "method": "BDF",
            "max_order": 3,
            "max_num_steps": 100000,
            "max_nonlin_iters": 25,
            "max_conv_fails": 50,
            "constraints_idx": constraints_idx,
            "constraints_type": constraints_type,
        }

        if self.event_specs:
            step_events = make_cvode_events_callback(self.event_specs)
            options.update({
                "eventsfn": step_events,
                "num_events": len(self.event_specs),
            })
            for spec in self.event_specs:
                threshold_str = format_metric_value(spec["unit"], spec["threshold_dim"])
                direction_word = "below" if spec["direction"] < 0 else "above"
                arming = spec.get("arming")
                arming_text = (
                    ""
                    if arming is None
                    else f"; armed once {arming['metric_label']} "
                    f"{arming['operator']} {arming['threshold_str']}"
                )
                print(
                    f"Event termination enabled: {spec['label']} "
                    f"when {spec['metric_label']} goes {direction_word} {threshold_str}"
                    f"{arming_text}"
                )
        
        linsolver = configure_cvode_linear_solver_options(
            options,
            use_sparsity=self.use_sparsity,
            sparsity_mode=self.cvode_sparsity_mode,
            step=self.step,
            state_size=Y0.size,
            layout=self.state_layout,
        )

        start_time = time.time()

        cvode_solver = CVODE(self.pde_callback, **options)

        sol_raw = cvode_solver.solve(self.t_eval, Y0)
        duration = time.time() - start_time

        termination_summary = build_termination_summary(
            sol_raw,
            self.event_specs,
            tau_hyd=self.tau_hyd,
            configured_mode=self.termination_mode,
        )

        effective_success = bool(sol_raw.success) and bool(
            termination_summary.get("success", False)
        )
        effective_status = getattr(sol_raw, "status", None)
        effective_message = str(getattr(sol_raw, "message", ""))
        if bool(sol_raw.success) and not effective_success:
            effective_status = termination_summary.get("status", effective_status)
            effective_message = termination_summary.get("message", effective_message)

        self.sol = {
            "solver": "scikit",                      
            "cvode": {
                "success": effective_success,
                "status": effective_status,
                "message": effective_message,
                "raw_success": bool(sol_raw.success),
                "raw_status": getattr(sol_raw, "status", None),
                "raw_message": str(getattr(sol_raw, "message", "")),
                "nfev": getattr(sol_raw, "nfev", None),
                "njev": getattr(sol_raw, "njev", None),
                "linsolver": linsolver,
                "wall_time_s": duration,
            },
            "t": sol_raw.t,
            # Keep time as the contiguous last axis used by saved fields and
            # post-processing. The storage-group arrays later become zero-copy
            # views of this one canonical allocation.
            "y": np.ascontiguousarray(sol_raw.y.T),
            "t_events": sol_raw.t_events,
            "y_events": sol_raw.y_events,
            "i_events": sol_raw.i_events,
            "termination": termination_summary,
        }

        self.sol["solve_provenance"] = self.solve_provenance
        report_solve_outcome(
            sol_raw,
            termination_summary,
            duration,
            step=self.step,
            t_end_max=self.t_end_max,
        )

    def load_results(self):
        """Return prepared results, loading the saved full artifact when needed."""
        required_groups = ("y_2d", "y_1d", "y_scalars")
        live_solution = getattr(self, "sol", None)
        if isinstance(live_solution, dict) and all(
            group in live_solution for group in required_groups
        ):
            return live_solution

        sol_path = os.path.join(self.base_folder, self.step, "sol.pkl")
        if not os.path.exists(sol_path):
            raise FileNotFoundError(f"No saved results found at '{sol_path}'.")
        with open(sol_path, "rb") as f:
            self.sol = pickle.load(f)
        source_record(self.sol.get("solve_provenance"))
        return self.sol

    def get_final_fields(self):
        """Extract final physical fields, loading saved results when necessary."""
        live_solution = getattr(self, "sol", None)
        if not isinstance(live_solution, dict) or "y_2d" not in live_solution:
            print("\n" + "-"*60)
            print(" No simulation performed for this step ".center(60))
            print(" Loading saved solution to proceed with next step ".center(60))
            print("-"*60)
            self.load_results()

        return self._extract_final_fields_from_solution()

    def get_final_profile(self):
        """Return copied dimensional primitive fields at the final output time."""
        live_solution = getattr(self, "sol", None)
        if not isinstance(live_solution, dict) or "y_2d" not in live_solution:
            self.load_results()
        from pydac.utils.postprocessing.saved_solution import saved_solution_view

        saved = saved_solution_view(self.sol)
        return {
            field.name: np.array(saved.field(field.name)[..., -1], copy=True)
            for field in saved.layout.pde_fields
        }

    def _extract_final_fields_from_solution(self):
        """
        Internal method to unpack the final solution state vector into physical fields.
        Scales back from dimensionless to physical units.
        """
        y_fields_2d = self.sol["y_2d"]
        y_fields_1d = self.sol["y_1d"]
        extra = self.sol.get("extra", {})

        saved_layout = extra.get("state_layout") if isinstance(extra, dict) else None
        if (
            isinstance(saved_layout, dict)
            and saved_layout.get("signature") != self.state_layout.signature
        ):
            raise ValueError(
                "Saved solution layout does not match the current model "
                f"({saved_layout.get('signature')} != {self.state_layout.signature})."
            )

        Nz_saved = int(extra.get("Nz", y_fields_2d.shape[1]))
        Nr_saved = int(extra.get("Nr", y_fields_2d.shape[2]))
        Nt_saved = int(extra.get("Nt", y_fields_2d.shape[-1]))

        fields_2d = {
            field.name: y_fields_2d[index, ..., -1]
            for index, field in enumerate(self.state_layout.fields_in("y_2d"))
        }
        fields_1d = {
            field.name: y_fields_1d[index, ..., -1]
            for index, field in enumerate(self.state_layout.fields_in("y_1d"))
        }

        ci = np.stack(
            [fields_2d[f"c_{species}"] for species in self.state_layout.independent_gas_species],
            axis=-1,
        )
        qi = np.stack(
            [fields_2d[f"q_{species}"] for species in self.state_layout.adsorbates],
            axis=-1,
        )

        return {
            "ci": ci,
            "qi": qi,
            # These three fields would otherwise remain views into the complete
            # Nt-history and prevent compact runs from releasing it between steps.
            "c_tot": np.array(fields_2d["c_tot"], copy=True),
            "T": np.array(fields_2d["T_bed"], copy=True),
            "T_wall": np.array(fields_1d["T_wall"], copy=True),
            "layout_signature": self.state_layout.signature,
            "Nz": Nz_saved,
            "Nr": Nr_saved,
            "Nt": Nt_saved,
            "z_f": np.array(extra.get("z_f", self.model.z_f), copy=True),
            "z_c": np.array(extra.get("z_c", self.model.z_c), copy=True),
            "r_f": np.array(extra.get("r_f", self.model.r_f), copy=True),
            "r_c": np.array(extra.get("r_c", self.model.r_c), copy=True),
        }   
        
    def _history_view_for_storage(self, y_all, storage, Nt):
        """Return one named storage group as a zero-copy view of packed history."""
        fields = self.state_layout.fields_in(storage)
        if not fields:
            return np.empty((0, Nt), dtype=y_all.dtype)

        field_shape = fields[0].shape
        if any(field.shape != field_shape for field in fields):
            raise ValueError(
                f"StateLayout storage group {storage!r} contains mixed field shapes."
            )

        cursor = fields[0].start
        for field in fields:
            if field.start != cursor:
                raise ValueError(
                    f"StateLayout storage group {storage!r} is not contiguous at "
                    f"field {field.name!r}."
                )
            cursor = field.stop

        packed_group = y_all[fields[0].start:fields[-1].stop, :]
        expected_size = len(fields) * int(np.prod(field_shape, dtype=int))
        if packed_group.shape != (expected_size, Nt):
            raise ValueError(
                f"Packed history for {storage!r} has shape {packed_group.shape}; "
                f"expected {(expected_size, Nt)}."
            )

        grouped = packed_group.reshape((len(fields), *field_shape, Nt))
        if not np.shares_memory(grouped, y_all):
            raise RuntimeError(
                f"Preparing {storage!r} unexpectedly copied the full solution history."
            )
        return grouped

    def prepare_results(self):
        """Build the canonical dimensional solution representation in memory.

        The operation is idempotent. A newly solved packed history is scaled and
        concentration-clipped exactly once, exposed through the three StateLayout
        storage groups, and then removed from the public solution dictionary.
        """
        schema_version = self.sol.get(SAVED_SOLUTION_SCHEMA_KEY)
        if (
            schema_version is not None
            and schema_version != SAVED_SOLUTION_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Unsupported saved-solution schema version {schema_version}; "
                f"expected {SAVED_SOLUTION_SCHEMA_VERSION}."
            )
        if schema_version == SAVED_SOLUTION_SCHEMA_VERSION:
            required_groups = ("y_2d", "y_1d", "y_scalars")
            missing_groups = [
                group for group in required_groups if group not in self.sol
            ]
            if missing_groups:
                raise ValueError(
                    "Prepared solver results are missing storage groups: "
                    f"{missing_groups}."
                )
            if "y" in self.sol:
                raise ValueError(
                    "Canonical prepared results must not contain duplicate history `y`."
                )
            return self.sol

        if "y" not in self.sol:
            raise ValueError("Unprepared solver results are missing packed history `y`.")

        y_all = self.sol["y"]
        Nt = y_all.shape[1]
        if y_all.shape[0] != self.state_layout.state_size:
            raise ValueError(
                f"Solution state size {y_all.shape[0]} does not match layout size "
                f"{self.state_layout.state_size}."
            )

        self.sol["outlet_face"] = outlet_face_history(self.model, y_all)

        for field in self.state_layout.fields:
            history = y_all[field.state_slice].reshape((*field.shape, Nt))
            history *= self.state_scale_by_field[field.name]
            if field.kind in ("gas concentration", "total concentration"):
                np.maximum(history, 0.0, out=history)

        self.sol["y_2d"] = self._history_view_for_storage(y_all, "y_2d", Nt)
        self.sol["y_1d"] = self._history_view_for_storage(y_all, "y_1d", Nt)
        self.sol["y_scalars"] = self._history_view_for_storage(
            y_all, "y_scalars", Nt
        )

        self.sol["t"] *= self.tau_hyd
        if self.sol.get("t_events") is not None:
            self.sol["t_events"] = [
                np.asarray(t_event, dtype=float) * self.tau_hyd
                for t_event in self.sol["t_events"]
            ]

        self.sol["extra"] = {
            "Nz": self.Nz,
            "Nr": self.Nr,
            "Nt": Nt,
            "A_rings": self.model.A_rings,
            "V_cells": self.model.V_cells,
            "ring_gas_holdup_factor": self.model.ring_gas_holdup_factor.copy(),
            "ring_sorbent_factor": self.model.ring_sorbent_factor.copy(),
            "L_bed": self.model.L_b,
            "R_bed": self.model.R_b,
            "bed_volume": self.model.bed_volume,
            "z_f": np.array(self.model.z_f, copy=True),
            "z_c": np.array(self.model.z_c, copy=True),
            "r_f": np.array(self.model.r_f, copy=True),
            "r_c": np.array(self.model.r_c, copy=True),
            "pressure_drop": pressure_drop_metadata_from_model(self.model),
            "state_layout": self.state_layout.to_dict(),
        }
        self.sol[SAVED_SOLUTION_SCHEMA_KEY] = SAVED_SOLUTION_SCHEMA_VERSION

        # The split arrays above are views of this allocation. Removing the public
        # packed alias retains one backing buffer while eliminating duplicated output.
        del self.sol["y"]
        return self.sol

    def save_results(self):
        """
        Prepare and save the canonical full solution to ``<step>/sol.pkl``.

        The versioned artifact stores one dimensional history through the named
        ``y_2d``, ``y_1d``, and ``y_scalars`` groups. The duplicate packed ``y``
        alias is intentionally omitted.
        """
        print("Saving simulation results...")

        save_dir = os.path.join(self.base_folder, self.step)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "sol.pkl")

        self.prepare_results()

        with open(save_path, "wb") as f:
            pickle.dump(self.sol, f)

        print(f"Results successfully saved to '{save_path}'")

    def save_terminal_results(self):
        """Persist one compact restart/CSS endpoint without a trajectory artifact."""
        print("Saving compact terminal state...")
        save_dir = os.path.join(self.base_folder, self.step)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, TERMINAL_STATE_FILENAME)

        self.prepare_results()
        document = terminal_state_from_solution(
            self.sol,
            step=self.step,
            solver_outcome=compact_solver_outcome(self.sol, step=self.step),
        )
        save_terminal_state(document, save_path)
        print(f"Compact terminal state successfully saved to '{save_path}'")
        return save_path

    def release_results(self):
        """Drop all full-history arrays after compact post-processing is complete."""
        solution = getattr(self, "sol", None)
        if not isinstance(solution, dict):
            self.sol = None
            return
        time_values = np.asarray(solution.get("t", []), dtype=float)
        final_time = None if time_values.size == 0 else float(time_values[-1])
        self.sol = {
            "t_final_s": final_time,
            "cvode": solution.get("cvode"),
            "termination": solution.get("termination"),
            "results_released": True,
        }
