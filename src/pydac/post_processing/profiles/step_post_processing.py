"""
pyDAC - Saved step reconstruction and post-processing.

This module reloads step solutions, reconstructs dimensional fields and derived
quantities, and exposes typed step metrics and plotting helpers for one TVSA
step.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

from collections.abc import Mapping
import os

import numpy as np
from scipy.constants import gas_constant as R_gas
from scipy.integrate import simpson
import pandas as pd
from pydac.post_processing.metrics.step_metrics import MetricRecord, STEP_METRIC_CATALOG, StepMetrics
from pymrm import interp_cntr_to_stagg, construct_grad
from pydac.utils.vector_drag import normal_velocity, transverse_gradients
from pydac.core.thermodynamics_separators import Component, FlashSeparator, ThermoModels
from pydac.post_processing.profiles.step_plotting import ProfilesPlotter
from pydac.utils.postprocessing import cyclic_profiles as _profile_utils
from pydac.utils.postprocessing.saved_solution import saved_solution_view
from pydac.core.bed_architectures import packed_bed_ring_structure_factors
from pydac.utils.postprocessing.outlet import (
    outlet_net_species_flows,
    outlet_sensible_enthalpy_flow,
)

# Cooling water can only take the stream down to its own supply temperature plus an
# approach.  Everything the condenser rejects above that line costs no work -- the same
# credit `cycle_kpis.py` already gives the jacket -- but the split needs the flash, so it
# is measured here and priced there.  Kept equal to
# `EquivalentWorkConditions.T_low + cooling_water_approach`.
T_COOLING_WATER_LIMIT = 303.15  # [K]

# A sample where the outlet face carries no stream keeps its place on the time
# axis and contributes nothing, rather than being dropped and its neighbours
# joined across the gap.
_NO_STREAM_FLASH = {
    "y_CO2": 0.0,
    "y_H2O": 0.0,
    "x_CO2": 0.0,
    "x_H2O": 0.0,
    "nv": 0.0,
    "nl": 0.0,
}

class StepPostProcessor:
    CLOSED_OUTLET_STEPS = frozenset({"press", "cool"})
    # Largest inward net outlet flow, relative to the step's largest outward one,
    # accepted only at the initial t = 0 sample as a stagnant face. A bed already
    # at its outlet pressure has ~zero face flow at t = 0 whose sign is
    # noise: 5e-8..9e-4 of the step's peak flow in the tau = 8 MOO candidates.
    REFLUX_RTOL = 1e-2

    def __init__(self, model, solver):
        """
        Initialize the step post-processor from a common column model.

        Parameters:
        - model: an initialized ColumnModelMoL instance (metadata, not solution)
        """
        print("\n" + "="*60)
        print(" Initializing StepPostProcessor".center(60))
        print("="*60 + "\n")
        
        self.solver = solver
        self.pb = model
        self.base_folder = solver.base_folder
        self.step = model.step
        self.profiles_folder = os.path.join(self.base_folder, self.step, "profiles")
        self.internal_1D_profiles_folder = os.path.join(self.profiles_folder, "internal_1D_profiles")
        self.internal_2D_profiles_folder = os.path.join(self.profiles_folder, "internal_2D_profiles")
        self.outlet_profiles_folder = os.path.join(self.profiles_folder, "outlet_profiles")
        os.makedirs(self.outlet_profiles_folder, exist_ok=True)
        os.makedirs(self.internal_1D_profiles_folder, exist_ok=True)
        os.makedirs(self.internal_2D_profiles_folder, exist_ok=True)

        self.sol = self._load_results()
        self.step_long = self._translate_step(self.pb.step)  
        self._extract_fields()
        self.plotter = ProfilesPlotter(self)  

    def _translate_step(self, step):
        """
        Translate internal step names to directory names used in saving/loading.
        """
        step_map = {
            "startup": "startup",
            "ads": "adsorption",
            "blow": "blowdown",
            "heat": "heating",
            "w_purge": "wall purge",
            "s_purge": "steam purge",
            "w_des": "wall-heated desorption",
            "s_des": "steam-assisted desorption",
            "press": "pressurization",
            "cool": "cooling"
        }

        if step not in step_map:
            raise ValueError(f"Step '{step}' not recognized. Valid keys: {list(step_map.keys())}")
        return step_map[step]

    def _load_results(self):
        """Use live prepared results, falling back to the saved full artifact."""
        live_solution = getattr(self.solver, "sol", None)
        required_groups = ("y_2d", "y_1d", "y_scalars")
        if isinstance(live_solution, Mapping) and all(
            group in live_solution for group in required_groups
        ):
            print(f"Using in-memory simulation results for step '{self.pb.step}'.")
            return live_solution

        print(f"Loading saved simulation results for step '{self.pb.step}'...")
        solution = self.solver.load_results()
        print(f"Results successfully loaded for step '{self.pb.step}'.")
        return solution

    def _extract_fields(self):
        """Extract physical fields from the state vector (already dimensional)."""
        self.y_fields_2d = self.sol['y_2d']
        self.y_fields_1d = self.sol['y_1d']
        self.y_scalars = self.sol['y_scalars']
        self.saved = saved_solution_view(self.sol)
        self.layout = self.saved.layout
        model_layout = getattr(self.pb, "state_layout", None)
        if model_layout is not None and model_layout.signature != self.layout.signature:
            raise ValueError(
                "Saved solution StateLayout differs from the initialized post-processing "
                f"model ({self.layout.signature} != {model_layout.signature})."
            )
        self.time = self.saved.time_s
        self.termination = self.sol.get("termination", None)

        self._extract_geometry()

        self.gas_species = self.layout.mixture.species
        self.independent_species = self.layout.independent_gas_species
        self.closure_species = self.layout.closure_species
        self.adsorbates = self.layout.adsorbates
        self.c_tot = self.saved.field("c_tot")
        self.T_bed = self.saved.field("T_bed")
        self.T_wall = self.saved.field("T_wall")

        independent = {
            species: self.saved.field(f"c_{species}")
            for species in self.independent_species
        }
        closure = self.c_tot - sum(independent.values())
        self.raw_c_by_species = {
            species: closure if species == self.closure_species else independent[species]
            for species in self.gas_species
        }
        self.c_by_species = {
            species: np.array(values, copy=True)
            for species, values in self.raw_c_by_species.items()
        }
        self.q_by_species = {
            species: self.saved.field(f"q_{species}")
            for species in self.adsorbates
        }
        self.y_by_species = {
            species: concentration / (self.c_tot + 1e-32)
            for species, concentration in self.c_by_species.items()
        }
        self._clip_fields()

        self.p_tot = self.c_tot * R_gas * self.T_bed
        self.molecular_weight_by_species = {
            species: float(self.pb.species_properties[species].molecular_weight_kg_per_mol)
            for species in self.gas_species
        }
        self.ci = np.stack(
            [self.c_by_species[species] for species in self.gas_species], axis=-1
        )
        self.rho_gas = sum(
            self.c_by_species[species] * self.molecular_weight_by_species[species]
            for species in self.gas_species
        )
        self.u_ax = self._reconstruct_axial_velocity(self.rho_gas)
        self._extract_integrated_fields(self.y_scalars) 

        # Existing public DAC names remain available when the required fields exist.
        for species in self.gas_species:
            setattr(self, f"c_{species}", self.c_by_species[species])
            setattr(self, f"y_{species}", self.y_by_species[species])
        for species in self.adsorbates:
            setattr(self, f"q_{species}", self.q_by_species[species])
        
    def _clip_fields(self):
        """Clip species mole fractions to physical bounds, based on the solver tolerance."""
        self.y_by_species = {
            species: np.clip(values, 1e-12, 1.0)
            for species, values in self.y_by_species.items()
        }
        self.c_by_species = {
            species: np.clip(values, 1e-12, None)
            for species, values in self.c_by_species.items()
        }

    def _outlet_net_flows(self):
        """Net outlet-face species molar flows [mol/s] and sensible enthalpy flow [W].

        Signed, so rings flowing inward subtract exactly as the bed balance sees
        them. Prefers the face history the solver recorded itself, which replays
        the hydraulic law before dimensional scaling and clipping. Older results
        carry no such history and are rebuilt here from the saved fields with the
        same face weights; that path reads clipped concentrations, so it is
        reported under its own source name instead of being labelled exact.
        """
        flows = outlet_net_species_flows(self.sol)
        if flows is not None:
            return (
                flows,
                outlet_sensible_enthalpy_flow(self.sol),
                "solver_outlet_face",
            )
        self._get_flowrates()
        return (
            dict(self.molar_flow_out_by_species),
            self._reconstructed_sensible_enthalpy_flow(),
            "reconstructed_from_saved_fields",
        )

    def _reconstructed_sensible_enthalpy_flow(self):
        """Rebuild the outlet sensible enthalpy flow [W] from the saved fields.

        The energy balance transports ``sum_i c_i cp_i (T - T_h)`` as a single
        scalar, so the outlet face has to interpolate that product with the same
        weights the convection operator applies. Interpolating temperature and
        concentrations separately does not reproduce it, and the cross term is
        exactly the radial non-uniformity this reporting exists to resolve.
        """
        weight_last, weight_inner = self.pb.outlet_face_weights
        inner_index = -2 if self.Nz > 1 else -1
        cp = np.array(
            [self.pb.cp_g[self.pb.gas_index[species]] for species in self.gas_species],
            dtype=float,
        )
        sensible = np.tensordot(self.ci, cp, axes=([-1], [0])) * (
            self.T_bed - self.pb.T_enthalpy_ref_K
        )
        sensible_face = (
            weight_last * sensible[-1] + weight_inner * sensible[inner_index]
        )
        return np.sum(
            sensible_face * self.u_ax[-1, :] * self.A_rings[:, None], axis=0
        )

    def get_mixed_cups(self):
        """Outlet mixed-cup traces: stream composition and enthalpy temperature.

        Composition is the composition of the stream, one quantity:
        ``y_i = sum_r(n_i,r) / sum_r sum_j(n_j,r)``. Weighting mole fractions by
        area, or by mass flux, is a different number whenever temperature,
        composition and velocity vary radially -- which they do even at uniform
        porosity.

        The temperature is the enthalpy-balance mixing temperature, not the same
        weights applied to T: mixing conserves enthalpy, and the species heat
        capacities differ, so the two are not the same average.

        Both are NaN where the face carries no net outgoing stream. Callers that
        need a physical stream validate it; this reports what is there.
        """
        if hasattr(self, "_mixed_cups_cache"):
            return self._mixed_cups_cache

        flows, enthalpy_flow, source = self._outlet_net_flows()
        total = sum(flows.values())
        carries_stream = total > 0.0

        mixed_cups = {
            f"y_{species}": np.divide(
                flows[species],
                total,
                out=np.full_like(total, np.nan),
                where=carries_stream,
            )
            for species in self.gas_species
        }

        cp = np.array(
            [self.pb.cp_g[self.pb.gas_index[species]] for species in self.gas_species],
            dtype=float,
        )
        heat_capacity_flow = sum(
            flows[species] * cp[index]
            for index, species in enumerate(self.gas_species)
        )
        defined = carries_stream & (heat_capacity_flow > 0.0)
        T_out = np.full_like(total, np.nan)
        T_out[defined] = (
            self.pb.T_enthalpy_ref_K
            + enthalpy_flow[defined] / heat_capacity_flow[defined]
        )
        mixed_cups["T_out"] = T_out

        self.outlet_composition_source = source
        for species in self.gas_species:
            setattr(
                self,
                f"y_{species}_out_mixed_cup",
                mixed_cups[f"y_{species}"],
            )
        self.T_out_mixed_cup = mixed_cups["T_out"]
        self._mixed_cups_cache = mixed_cups
        return mixed_cups

    def _condenser_stream_mask(self):
        """Validate the aggregate stream handed to the condenser, sample by sample.

        The condenser is an aggregate unit operation: it receives a pipe, with a
        total flow, a composition, a temperature and a pressure. It does not see
        the radial structure of the bed face, so counter-flow between individual
        rings is not disqualifying on its own -- only what the aggregate turns
        out to be.

        Returns the mask of samples that carry a stream at all. Samples without
        one keep their place on the time axis and contribute nothing.
        """
        if self.step in self.CLOSED_OUTLET_STEPS:
            raise ValueError(
                f"Step '{self.step}' has a closed outlet, so it carries no stream "
                "for the condenser. The caller must exclude it."
            )

        flows, _, _ = self._outlet_net_flows()
        # Validate before masking: NaN must not poison the scale or disappear
        # behind the positive-flow mask.
        for species, flow in flows.items():
            if not np.all(np.isfinite(flow)):
                raise ValueError(f"Outlet molar flow of {species} is not finite.")
        with np.errstate(over="ignore", invalid="ignore"):
            total = sum(flows.values())
        if not np.all(np.isfinite(total)):
            raise ValueError("Total outlet molar flow is not finite.")

        # Only the initial stagnant face may be treated as no stream. Every
        # later inward sample is reflux, regardless of its relative magnitude.
        scale = float(np.max(total, initial=0.0))
        reflux = total < 0.0
        if total.size and self.time[0] == 0.0 and scale > 0.0:
            reflux[0] = reflux[0] and (-total[0] > self.REFLUX_RTOL * scale)
        if np.any(reflux):
            raise ValueError(
                "The outlet carries a net inward stream in "
                f"{int(np.sum(reflux))} of {total.size} "
                "samples. The condenser model does not describe reflux."
            )

        carries_stream = total > 0.0
        # Individual mole fractions are deliberately not range-checked. The outlet
        # face weights extrapolate past the last cell centre (1.125 / -0.125), so a
        # species being flushed out reaches the face slightly negative whenever it
        # drops by more than a factor nine across the last cell. That is the
        # boundary condition, not a defect, and bounding it would mean bounding the
        # near-outlet gradient rather than any numerical noise.
        return carries_stream
        
    def _extract_geometry(self):    
        """
        Extract geometry-related fields from the state vector.
        """    
        extra = self.sol.get("extra", {})
        Nz_from_fields = self.y_fields_2d.shape[1]
        Nr_from_fields = self.y_fields_2d.shape[2]
        Nt_from_fields = self.y_fields_2d.shape[-1]

        self.Nz = int(extra.get("Nz", Nz_from_fields))
        self.Nr = int(extra.get("Nr", Nr_from_fields))
        self.Nt = int(extra.get("Nt", Nt_from_fields))

        if self.Nz != Nz_from_fields or self.Nr != Nr_from_fields or self.Nt != Nt_from_fields:
            raise ValueError(
                "Saved grid metadata in `sol.pkl` is inconsistent with the saved field shapes. "
                f"Metadata says (Nz={self.Nz}, Nr={self.Nr}, Nt={self.Nt}), "
                f"but y_2d has ({Nz_from_fields}, {Nr_from_fields}, {Nt_from_fields})."
            )

        self.V_cells = extra["V_cells"]
        self.A_rings = extra["A_rings"]
        self.R_b = extra["R_bed"]
        self.L_b = extra["L_bed"]

        self.z_f = np.array(extra.get("z_f", np.linspace(0, 1, self.Nz + 1)), copy=True)
        self.z_c = np.array(extra.get("z_c", 0.5 * (self.z_f[:-1] + self.z_f[1:])), copy=True)
        self.r_f = np.array(extra.get("r_f", np.linspace(0, 1, self.Nr + 1)), copy=True)
        self.r_c = np.array(extra.get("r_c", 0.5 * (self.r_f[:-1] + self.r_f[1:])), copy=True)

        # Saved factors belong to this grid. Older results resolve the same
        # structural input on the saved faces, rather than using the live grid.
        if "ring_gas_holdup_factor" in extra and "ring_sorbent_factor" in extra:
            gas = np.asarray(extra["ring_gas_holdup_factor"])
            solid = np.asarray(extra["ring_sorbent_factor"])
        elif self.pb.bed_architecture == "packed_bed" and self.Nr > 1:
            gas, solid = packed_bed_ring_structure_factors(
                self.pb.bed_design, self.pb.particle_properties, self.r_f
            )
        else:
            gas = solid = np.ones(self.Nr)
        self.gas_holdup = self.pb.eps_total * gas[None, :]
        self.sorbent_density = self.pb.rho_sorbent * solid[None, :]

        if self.z_f.shape[0] != self.Nz + 1 or self.z_c.shape[0] != self.Nz:
            raise ValueError(
                "Saved axial grid metadata in `sol.pkl` is inconsistent with Nz. "
                f"Expected z_f length {self.Nz + 1} and z_c length {self.Nz}."
            )
        if self.r_f.shape[0] != self.Nr + 1 or self.r_c.shape[0] != self.Nr:
            raise ValueError(
                "Saved radial grid metadata in `sol.pkl` is inconsistent with Nr. "
                f"Expected r_f length {self.Nr + 1} and r_c length {self.Nr}."
            )

        self.shape_single_saved = (self.Nz, self.Nr)
        self.Grad_ax_p_saved, grad_ax_p_bc_saved = construct_grad(
            self.shape_single_saved,
            self.z_f,
            self.z_c,
            self.pb.bc_ax_p,
            axis=0,
        )
        self.grad_ax_p_bc_dense_saved = grad_ax_p_bc_saved.toarray().ravel()

        if self.Nz < 3:
            raise ValueError(
                "Post-processing requires at least Nz >= 3 to reconstruct the inlet axial velocity."
            )
        self.dz1_saved = self.z_f[2] - self.z_f[0]
        self.dz2_saved = self.z_f[1] - self.z_f[0]
        self.dz3_saved = self.z_f[2] - self.z_f[1]

        self.is_1d = bool(self.Nr == 1)
        self.Nr_plot_2d = self.Nr if not self.is_1d else max(25, int(getattr(self.pb, "Nr_plot_2d", 25)))
        self.r_f_plot = np.linspace(0, 1, self.Nr_plot_2d + 1)
        self.r_c_plot = 0.5 * (self.r_f_plot[:-1] + self.r_f_plot[1:])

        if self.Nz != self.pb.Nz or self.Nr != self.pb.Nr:
            print(
                "[INFO] Loaded results were saved on grid "
                f"(Nz={self.Nz}, Nr={self.Nr}) while the current model uses "
                f"(Nz={self.pb.Nz}, Nr={self.pb.Nr}). Post-processing uses the saved grid."
            )

        if self.is_1d:
            print(
                "[INFO] Post-processing in 1D mode: radial plots use a single cell, "
                "outlet traces use the single radial value, and 2D plots are "
                "rendered with a pseudo-radial expansion."
            )
        
    def _apply_operator_time(self, field_3d, Op, out_spatial_shape, bc=None):
        """
        Apply the operator to a time stack field_3d.

        field_3d: (*in_spatial, Nt)
        Op:      (N_out, N_in) sparse/dense
        bc:      (N_out,) or (N_out,1) or None   (added after Op @ field)
        out_spatial_shape: tuple like (Nz+1, Nr) or (Nz, Nr+1) etc. (NO Nt)

        returns: (*out_spatial_shape, Nt)
        """
        *in_spatial, Nt = field_3d.shape
        N_in = int(np.prod(in_spatial))
        field_flat = field_3d.reshape(N_in, Nt)     # (N_in, Nt)

        out_flat = Op @ field_flat                  # (N_out, Nt)

        if bc is not None:
            bc_dense = np.asarray(bc)
            if bc_dense.ndim == 1:
                bc_dense = bc_dense[:, None]        # (N_out, 1)
            out_flat = out_flat + bc_dense          # broadcast over Nt

        return out_flat.reshape(*out_spatial_shape, Nt)

    def _reconstruct_axial_velocity(self, rho):
        """Reconstruct dimensional axial velocity from saved pressure metadata."""
        rho_stagg_ax = interp_cntr_to_stagg(rho, self.z_f, self.z_c, axis=0)

        # The saved gradient operator acts on dimensionless z/L_b and p/p_ref.
        p_nd = self.p_tot / self.pb.p_ref
        grad_p_nd = self._apply_operator_time(
            p_nd,
            self.Grad_ax_p_saved,
            (self.Nz + 1, self.Nr),
            bc=self.grad_ax_p_bc_dense_saved,
        )
        driving_gradient = -(self.pb.p_ref / self.L_b) * grad_p_nd
        pressure_parameters = _profile_utils.pressure_drop_parameters_from_metadata(
            self.sol.get("extra", {}),
            self.step,
        )
        if pressure_parameters.get("formulation") == "isotropic_vector":
            radial_operator, radial_bc = construct_grad(
                self.shape_single_saved, self.r_f, self.r_c,
                ({'a': 1, 'b': 0, 'd': 0}, {'a': 1, 'b': 0, 'd': 0}), axis=1)
            grad_rad = (self.pb.p_ref / self.R_b) * self._apply_operator_time(
                p_nd, radial_operator, (self.Nz, self.Nr + 1),
                bc=radial_bc.toarray().ravel())
            fixed = tuple(float(bc['a']) == 0.0 and float(bc['b']) != 0.0 for bc in self.pb.bc_ax_p)
            transverse_ax, _ = transverse_gradients(
                -driving_gradient, grad_rad, self.z_f, self.z_c, self.r_f, self.r_c,
                axial_dirichlet=fixed)
            u_ax = normal_velocity(
                -driving_gradient, transverse_ax, rho_stagg_ax,
                linear_coefficient=np.asarray(pressure_parameters["linear_coefficient"]).reshape(1, -1, 1),
                inertial_coefficient_per_density=np.asarray(pressure_parameters["inertial_coefficient_per_density"]).reshape(1, -1, 1))
        else:
            # Archived scalar/componentwise runs retain their original law.
            u_ax = _profile_utils.solve_pressure_drop_velocity(
                driving_gradient, rho_stagg_ax, **pressure_parameters)

        # Reconstruct boundary faces consistently with the solver: only
        # overwrite faces that are physically closed.
        if self.step in ("blow", "heat", "w_purge", "w_des", "cool"):
            u_ax[0, :, :] = 0.0
        elif self.step != "press":
            u_ax[0] = (
                self.dz1_saved * u_ax[1, :, :] - self.dz2_saved * u_ax[2, :, :]
            ) / self.dz3_saved

        if self.step in self.CLOSED_OUTLET_STEPS:
            u_ax[-1, :, :] = 0.0

        u_ax[np.abs(u_ax) < 1e-8] = 0.0
        return u_ax
          
    def _extract_integrated_fields(self, y_scalars):
        """Extract integrated KPI traces by their persisted layout names."""
        del y_scalars  # StateLayout is the only scalar-order owner.
        self.scalar_histories = {
            field.name: self.saved.field(field.name)
            for field in self.layout.fields_in("y_scalars")
        }
        self.mass_integrals_by_species = {
            species: self.scalar_histories[f"m_{species}"]
            for species in self.adsorbates
            if f"m_{species}" in self.scalar_histories
        }
        for name, history in self.scalar_histories.items():
            setattr(self, name, history)
 
    def describe(self):
        """Print a quick summary of time vector and key fields."""
        print("\n" + "-"*60)
        print(" Overview of Simulation Results".center(60))
        print("-"*60)
        print(f"Spatial discretization: {self.Nz} grid points")
        print(f"Radial discretization: {self.Nr} grid points")
        if self.is_1d:
            print("Simulation mode: 1D (`Nr = 1`)")
        print(f"Bed length: {self.pb.L_b:.3f} m")
        print(f"Bed diameter: {self.pb.d_b:.3f} m")
        print(f"Simulation time: {self.time[-1]/3600:.2f} hr")
        for species in self.adsorbates:
            loading = self.q_by_species[species]
            print(
                f"{species} loading range: {loading.min():.2f} - "
                f"{loading.max():.2f} mol/kg"
            )
        print(f"Bed temperature range: {self.T_bed.min()-273.15:.1f} - {self.T_bed.max()-273.15:.2f} °C")
        print(f"Interstitial velocity range: {self.u_ax.min():.3f} - {self.u_ax.max():.3f} m/s")
        print(f"Total pressure range: {self.p_tot.min()/100:.2f} - {self.p_tot.max()/100:.2f} mbar")
        print(f"Time steps for plotting: {self.time.size}")
        if self.termination is not None:
            reason = self.termination.get("reason", "unknown")
            if reason == "event" and self.termination.get("event") is not None:
                event = self.termination["event"]
                print(
                    f"Termination: event '{event['label']}' at {event['t_event_s']:.2f} s"
                )
                print(
                    f"Termination metric: {event['metric_label']} = {event['metric_value_str']} "
                    f"(target {event['threshold_str']})"
                )
            elif reason == "final_time":
                print(
                    f"Termination: final time reached at {self.termination.get('t_final_s', self.time[-1]):.2f} s"
                )
            elif reason == "failure":
                print(
                    f"Termination: solver failure ({self.termination.get('status', 'unknown')})"
                )

        for species, values in self.get_mass_balance_summary().items():
            change = values.get("delta_integrated", values["delta_state"])
            print(f"{species} net mass change: {change * 1e3:.2f} g")
 
    def get_masses(self):
        """
        Return the step-end masses used by the step-metrics layer.

        CO2 and H2O mass changes are the integrated KPI states saved by the
        solver. O2 and N2 are reconstructed from the final gas inventory.
        """
        self._require_species(("CO2", "H2O", "O2", "N2"), "legacy get_masses tuple")
        mass_summary = self.get_mass_balance_summary()

        return (
            mass_summary["CO2"]["initial"],
            mass_summary["CO2"]["delta_integrated"],
            mass_summary["H2O"]["initial"],
            mass_summary["H2O"]["delta_integrated"],
            mass_summary["O2"]["initial"],
            mass_summary["O2"]["left_state"],
            mass_summary["N2"]["initial"],
            mass_summary["N2"]["left_state"],
        )

    def _calculate_species_mass(self, species, phase="both", time_index=-1):
        """
        Reconstruct a species mass directly from the saved fields.

        Parameters
        ----------
        species : str
            One declared gas species.
        phase : str, optional
            "gas", "solid", or "both" (default). Non-adsorbates have no solid mass.
        time_index : int or None, optional
            Time index to evaluate. Use `None` to return the full time trace.
        """
        sp = species.strip().upper()
        if sp not in self.c_by_species:
            raise ValueError(
                f"Unknown species {species!r}. Declared mixture: {self.gas_species}."
            )

        phase = phase.lower()
        if phase not in {"gas", "solid", "both"}:
            raise ValueError("phase must be 'gas', 'solid', or 'both'")

        if time_index is None:
            cell_weights = self.V_cells[..., None]
            sum_axes = (0, 1)
            c_species = self.c_by_species[sp]
            q_species = self.q_by_species.get(sp)
        else:
            cell_weights = self.V_cells
            sum_axes = None
            c_species = self.c_by_species[sp][:, :, time_index]
            q_species = (
                None
                if sp not in self.q_by_species
                else self.q_by_species[sp][:, :, time_index]
            )

        gas_holdup = self.gas_holdup
        sorbent_density = self.sorbent_density
        if time_index is None:
            gas_holdup = gas_holdup[..., None]
            sorbent_density = sorbent_density[..., None]
        molecular_weight = self.molecular_weight_by_species[sp]
        m_gas = molecular_weight * np.sum(
            c_species * cell_weights * gas_holdup, axis=sum_axes
        )

        m_solid = 0.0
        if q_species is not None:
            m_solid = (
                molecular_weight
                * np.sum(q_species * cell_weights * sorbent_density, axis=sum_axes)
            )

        if phase == "gas":
            return m_gas
        if phase == "solid":
            return m_solid
        return {"gas": m_gas, "solid": m_solid, "total": m_gas + m_solid}

    def get_mass_balance_summary(self):
        """
        Collect integrated and state-based end-of-step mass balances.
        """
        if hasattr(self, "_mass_balance_summary"):
            return self._mass_balance_summary

        summary = {}
        for species in self.gas_species:
            initial = self._initial_species_mass(species)
            final = float(self._calculate_species_mass(species, phase="both")["total"])
            final = self._check_current_dac_residual_inventory(species, initial, final)
            delta_state = final - initial
            values = {
                "initial": float(initial),
                "delta_state": float(delta_state),
                "left_state": float(final),
            }
            if species in self.mass_integrals_by_species:
                delta_integrated = float(self.mass_integrals_by_species[species][-1])
                values.update(
                    {
                        "delta_integrated": delta_integrated,
                        "left_integrated": float(initial + delta_integrated),
                        "delta_gap": float(delta_integrated - delta_state),
                        "relative_error_pct": float(
                            self._relative_mass_error_pct(
                                delta_integrated, delta_state
                            )
                        ),
                    }
                )
            summary[species] = values

        self._mass_balance_summary = summary
        return self._mass_balance_summary

    def compute_mass_metrics(self):
        """Return the canonical mass quantities produced by this step."""
        summary = self.get_mass_balance_summary()
        metrics = {}

        for species in self.gas_species:
            species_summary = summary[species]
            values = {
                "initial": species_summary["initial"],
                "change": species_summary.get(
                    "delta_integrated", species_summary["delta_state"]
                ),
                "left": species_summary.get(
                    "left_integrated", species_summary["left_state"]
                ),
            }
            if "delta_integrated" in species_summary:
                values.update(
                    {
                        "state_change": species_summary["delta_state"],
                        "integrated_state_gap": species_summary["delta_gap"],
                        "integrated_state_error_pct": species_summary[
                            "relative_error_pct"
                        ],
                        "state_left": species_summary["left_state"],
                    }
                )
            metrics[species] = values

        return metrics

    def compute_final_gas_masses(self):
        """Return final gas-phase species inventories from the saved state [kg].

        Trace concentrations retain the solver values used by the historical
        residual-gas calculation. Only reconstructed nitrogen is clipped at
        zero to prevent a negative balance remainder.
        """
        gas_holdup = self.gas_holdup
        cell_volumes = np.asarray(self.V_cells, dtype=float)
        return {
            species: float(
                self.molecular_weight_by_species[species]
                * np.sum(
                    (
                        np.clip(concentration[:, :, -1], 0.0, None)
                        if species == self.closure_species
                        else concentration[:, :, -1]
                    )
                    * cell_volumes
                    * gas_holdup
                )
            )
            for species, concentration in self.raw_c_by_species.items()
        }

    def compute_bed_loading_fraction_co2(self):
        """Return the final bed-average CO2 loading relative to q_eq_CO2_ref.

        Weight by local sorbent inventory, including radial voidage variations.
        """
        self._require_species(("CO2",), "CO2 bed saturation")
        if "CO2" not in self.adsorbates:
            raise ValueError("CO2 bed saturation requires CO2 to be an adsorbate.")
        q_ref = float(getattr(self.pb, "q_eq_CO2_ref", 0.0))
        if q_ref <= 0.0:
            return np.nan

        q_co2 = np.asarray(self.q_by_species["CO2"][:, :, -1], dtype=float)
        solid_weights = (
            np.asarray(self.V_cells, dtype=float)
            * self.sorbent_density
        )
        den = q_ref * float(np.sum(solid_weights))
        if den <= 0.0:
            return np.nan

        return float(np.sum(q_co2 * solid_weights) / den)

    def _relative_mass_error_pct(self, integrated_delta, state_delta):
        """
        Return the absolute relative error [%] between integrated and state-based mass changes.
        """
        gap = integrated_delta - state_delta
        denom = abs(state_delta)

        if denom < 1e-16:
            return 0.0 if abs(gap) < 1e-16 else np.nan

        return 100.0 * abs(gap) / denom

    @property
    def _is_current_dac_layout(self):
        return (
            self.gas_species == ("CO2", "H2O", "O2", "N2")
            and self.closure_species == "N2"
            and self.adsorbates == ("CO2", "H2O")
        )

    def _require_species(self, species, purpose):
        missing = [name for name in species if name not in self.gas_species]
        if missing:
            raise ValueError(
                f"{purpose} requires species {tuple(species)}; missing={missing}, "
                f"declared mixture={self.gas_species}."
            )

    def _initial_species_mass(self, species):
        """Return the step-start inventory, retaining exact DAC startup aliases."""
        if self._is_current_dac_layout:
            if species in ("CO2", "H2O"):
                return float(
                    getattr(self.pb, f"mass_{species}_initial_f")
                    + getattr(self.pb, f"mass_{species}_initial_s")
                )
            if species in ("O2", "N2"):
                return float(getattr(self.pb, f"mass_{species}_initial"))

        masses = getattr(self.pb, "initial_mass_by_species", {}).get(species)
        if not isinstance(masses, dict) or "total" not in masses:
            raise ValueError(f"Model has no initial mass inventory for {species}.")
        return float(np.asarray(masses["total"]).item())

    def _check_current_dac_residual_inventory(self, species, initial, final):
        """Warn when O2/N2 rises in a DAC step that admits no air; never alter the fields.

        The final inventory is reconstructed from concentrations floored at
        1e-12 mol/m3 (``_clip_fields``, the order of the solver tolerance). The
        initial one comes from the model state: floored at 1e-32 for an
        independent species, unfloored for the closure species (c_tot - sum(ci)),
        which solver undershoot can leave slightly negative. A species the step
        has already removed therefore reads as a rise of up to the floor mass
        over the gas volume, so only a change beyond that budget is reported.
        """
        if not self._is_current_dac_layout or species not in ("O2", "N2"):
            return final
        allowed_steps = {
            "O2": {"startup", "ads", "press"},
            "N2": {"startup", "ads", "press", "cool"},
        }[species]
        # The relative margin absorbs rounding: a species at the floor in every
        # cell sums to the budget itself, only in a different order.
        budget = 1e-12 * self.molecular_weight_by_species[species] * float(
            np.sum(self.V_cells * self.gas_holdup)
        ) * (1.0 + 1e-9)
        # A negative step-start inventory is solver undershoot, not gas that can
        # later read as a rise; measure the rise from zero in that case.
        rise = final - max(initial, 0.0)
        if (rise > budget and self.step not in allowed_steps) or final < -budget:
            print(
                f"Warning: {species} mass in step {self.step} went from {initial:.3e} kg "
                f"to {final:.3e} kg (change {final - initial:+.3e} kg) although the step admits no "
                f"air; the post-processing floor accounts for {budget:.3e} kg."
            )
        return final

    def _CO2_mass_change(self):
        """
        Reconstruct the initial and final CO2 mass [kg] from the saved state.
        """
        self._require_species(("CO2",), "CO2 mass change")
        m_CO2_initial = self._initial_species_mass("CO2")
        m_CO2_left = self._calculate_species_mass("CO2", phase="both")["total"]
        return m_CO2_initial, m_CO2_left

    def _H2O_mass_change(self):
        """
        Reconstruct the initial and final H2O mass [kg] from the saved state.
        """
        self._require_species(("H2O",), "H2O mass change")
        m_H2O_initial = self._initial_species_mass("H2O")
        m_H2O_left = self._calculate_species_mass("H2O", phase="both")["total"]
        return m_H2O_initial, m_H2O_left
    
    def _O2_mass_change(self):
        """
        Initial and final gas-phase O2 mass [kg], checked against the DAC air-admission rule.
        """
        self._require_species(("O2",), "O2 mass change")
        m_O2_initial = self._initial_species_mass("O2")
        m_O2_left = self._calculate_species_mass("O2", phase="gas")
        m_O2_left = self._check_current_dac_residual_inventory(
            "O2", m_O2_initial, m_O2_left
        )
        return m_O2_initial, m_O2_left
    
    def _N2_mass_change(self):
        """
        Initial and final gas-phase N2 mass [kg], checked against the DAC air-admission rule.
        """
        self._require_species(("N2",), "N2 mass change")
        m_N2_initial = self._initial_species_mass("N2")
        m_N2_left = self._calculate_species_mass("N2", phase="gas")
        m_N2_left = self._check_current_dac_residual_inventory(
            "N2", m_N2_initial, m_N2_left
        )
        return m_N2_initial, m_N2_left
    
    def _get_flowrates(self):
        """
        Calculate the outlet species molar flow rates.

        The concentration is taken at the outlet **face**, using the same
        two-cell weights the model transports mass through (`bc_ax[1]` imposes
        a zero axial gradient there, so the face value is not the last cell
        centre). Reading the cell centre instead biases every outlet flow by one
        cell of the axial gradient, which is invisible on a step whose net
        change is a decent fraction of its throughput and severe on the steam
        steps, where kilograms pass for a gram of inventory change.

        Returns:
        """
        weight_last, weight_inner = self.pb.outlet_face_weights
        inner_index = -2 if self.Nz > 1 else -1
        u_ax_out = self.u_ax[-1, :]  # [m/s]
        ci_out = (
            weight_last * self.ci[-1, :, :]
            + weight_inner * self.ci[inner_index, :, :]
        )                            # [mol/m^3] at the outlet face
        flux_ni_out = u_ax_out[..., None] * ci_out  # [mol/m^2/s]
        
        self.molar_flow_out_by_species = {
            species: np.sum(
                flux_ni_out[..., index] * self.A_rings[:, None], axis=0
            )
            for index, species in enumerate(self.gas_species)
        }
        for species, flow in self.molar_flow_out_by_species.items():
            setattr(self, f"n_{species}_out", flow)
            
    def get_min_bed_T_and_steam_cond_T(self):
        """Return the minimum internal bed temperature at the end of the step
        and the steam condensation temperature.

        The minimum internal bed temperature is the coldest point inside the
        2D bed at the final time step. It is used to check whether steam
        would condense anywhere inside the bed.

        Returns
        -------
        T_bed_min_final : float
            Minimum bed temperature at the final time step [K].
        T_steam_condensation : float
            Steam condensation temperature [K].
        """
        T_bed_min_final = float(self.T_bed[:, :, -1].min())
        T_steam_condensation = self.pb.T_cond
        return T_bed_min_final, T_steam_condensation
    
    def get_energies(self):
        """
        Return integrated energy consumptions for the current step.

        Returns:
        - E_compression: compression energy [MJ].
        - E_vacuum: vacuum-pump energy [MJ].
        - E_jacket: jacket energy [MJ].
        """
        E_compression_final = self.E_compression[-1] / 1e6  # [MJ]
        E_vacuum_final = self.E_vacuum[-1] / 1e6            # [MJ]
        E_jacket_final = self.E_jacket[-1] / 1e6            # [MJ]
        
        return E_compression_final, E_vacuum_final, E_jacket_final

    def compute_average_pressure_drop(self):
        """Return the time-average inlet-to-outlet pressure drop [mbar]."""
        p_avg_in = np.sum(self.p_tot[0] * self.A_rings[:, None], axis=0) / np.sum(
            self.A_rings
        )
        p_avg_out = np.sum(
            self.p_tot[-1] * self.A_rings[:, None], axis=0
        ) / np.sum(self.A_rings)
        return simpson(p_avg_in - p_avg_out, x=self.time) / self.time[-1] / 100

    def compute_steam_metrics(self):
        """Return injected steam mass [kg] and steam energy [MJ].

        Steam vaporization temperature and latent heat use the pressure
        correlations from Ward et al. retained from the legacy calculation.
        """
        steam_mass_flowrate = self.pb.m_dot_steam  # [kg/s]
        cp_water = 4.186  # [kJ/kg/K]
        a, b, c = 1.012e-6, -3.594e-4, 1.882

        steam_mass = steam_mass_flowrate * self.time[-1]  # [kg]
        p_feed_bar = self.pb.p_feed / 1e5
        T_vap = 197 * p_feed_bar**0.1367 - 97.97 + 273.15  # [K]
        dh1 = cp_water * (self.pb.T_ref - T_vap)  # [kJ/kg_steam]
        dh_vap = -446.2 * p_feed_bar**0.1508 + 2707  # [kJ/kg_steam]
        dh2 = (
            a * (T_vap**3 - self.pb.T_feed**3) / 3
            + b * (T_vap**2 - self.pb.T_feed**2) / 2
            + c * (T_vap - self.pb.T_feed)
        )

        steam_energy = -steam_mass * (dh1 - dh_vap + dh2) / 1000  # [MJ]
        return steam_mass, steam_energy

    @staticmethod
    def _step_metric_record(
        metric_id,
        species,
        value,
        basis=None,
        definition_basis=None,
    ):
        """Build one typed metric while preserving its legacy representation."""
        lookup_basis = basis if definition_basis is None else definition_basis
        definition = STEP_METRIC_CATALOG.get(metric_id, basis=lookup_basis)
        return MetricRecord(
            definition=definition,
            species=species,
            value=value,
            basis=basis,
        )

    def _common_step_metric_records(self):
        """Return common grid, mass, and saturation records for one step."""
        record = self._step_metric_record
        records = [
            record("grid.nz", "system", self.Nz),
            record("grid.nr", "system", self.Nr),
        ]
        mass_metrics = self.compute_mass_metrics()

        for species in self.gas_species:
            species_metrics = mass_metrics[species]
            mass_basis = (
                "integrated"
                if species in self.mass_integrals_by_species
                else "state"
            )
            records.extend(
                [
                    record("mass.initial", species, species_metrics["initial"]),
                    record(
                        "mass.change",
                        species,
                        species_metrics["change"],
                        basis=mass_basis,
                        definition_basis="integrated",
                    ),
                    record(
                        "mass.final",
                        species,
                        species_metrics["left"],
                        basis=mass_basis,
                        definition_basis="integrated",
                    ),
                ]
            )

        for species in self.mass_integrals_by_species:
            species_metrics = mass_metrics[species]
            records.extend(
                [
                    record(
                        "mass.change",
                        species,
                        species_metrics["state_change"],
                        basis="state",
                    ),
                    record(
                        "mass.balance.gap",
                        species,
                        species_metrics["integrated_state_gap"],
                        basis="integrated-state",
                    ),
                    record(
                        "mass.balance.error",
                        species,
                        species_metrics["integrated_state_error_pct"],
                        basis="integrated-state",
                    ),
                    record(
                        "mass.final",
                        species,
                        species_metrics["state_left"],
                        basis="state",
                    ),
                ]
            )

        if "CO2" in self.adsorbates:
            saturation_pct = 100.0 * self.compute_bed_loading_fraction_co2()
            records.append(record("bed.saturation.co2", "CO2", saturation_pct))
        final_gas_masses = self.compute_final_gas_masses()
        records.extend(
            record(
                "mass.gas.final",
                species,
                final_gas_masses[species],
                basis="state",
            )
            for species in self.gas_species
        )
        return records, mass_metrics

    def compute_step_metrics(self, step=None):
        """Compute the complete typed scalar output for one simulated step."""
        step_key = self.step if step is None else str(step).strip()
        step_names = {
            "startup": "startup",
            "ads": "adsorption",
            "blow": "blowdown",
            "heat": "preheating",
            "w_purge": "wall purge",
            "s_purge": "steam purge",
            "w_des": "wall heated desorption",
            "s_des": "steam assisted desorption",
            "cool": "cooldown",
            "press": "pressurization",
        }
        if step_key not in step_names:
            raise ValueError(
                f"Unknown step: {step_key}. Valid steps are: {tuple(step_names)}."
            )

        record = self._step_metric_record
        records, mass_metrics = self._common_step_metric_records()

        if step_key in {"startup", "ads"}:
            energy_compression, _, energy_jacket = self.get_energies()
            records.extend(
                [
                    record(
                        "pressure.drop.average",
                        "system",
                        self.compute_average_pressure_drop(),
                    ),
                    record("energy.compression", "system", energy_compression),
                    record("energy.jacket", "system", energy_jacket),
                ]
            )
        elif step_key == "blow":
            _, _, energy_jacket = self.get_energies()
            outlet_records, energy_vacuum, condenser_records = (
                self._outlet_metric_records(mass_metrics)
            )
            records.extend(
                [
                    record("energy.vacuum", "system", energy_vacuum),
                    record("energy.jacket", "system", energy_jacket),
                ]
            )
            records.extend(outlet_records)
            records.extend(condenser_records)
        elif step_key in {"heat", "w_purge"}:
            T_bed_min, T_steam_condensation = self.get_min_bed_T_and_steam_cond_T()
            _, _, energy_jacket = self.get_energies()
            outlet_records, energy_vacuum, condenser_records = (
                self._outlet_metric_records(mass_metrics)
            )
            records.append(record("temperature.bed.minimum", "system", T_bed_min))
            # The preheating must exceed the steam condensation temperature before
            # steam enters; a mixture without H2O has no steam, so no such limit.
            if "H2O" in self.gas_species:
                records.append(
                    record(
                        "temperature.steam.condensation",
                        "system",
                        T_steam_condensation,
                    )
                )
            records.extend(
                [
                    record("energy.vacuum", "system", energy_vacuum),
                    record("energy.jacket", "system", energy_jacket),
                ]
            )
            records.extend(outlet_records)
            records.extend(condenser_records)
        elif step_key in {"s_purge", "w_des", "s_des"}:
            outlet_records, energy_vacuum, condenser_records = (
                self._outlet_metric_records(mass_metrics)
            )
            records.extend(outlet_records)
            energy_compression, _, energy_jacket = self.get_energies()

            if step_key == "w_des":
                records.extend(
                    [
                        record("energy.jacket", "system", energy_jacket),
                        record("energy.vacuum", "system", energy_vacuum),
                    ]
                )
                records.extend(condenser_records)
            else:
                steam_mass, steam_energy = self.compute_steam_metrics()
                records.extend(condenser_records)
                records.extend(
                    [
                        record(
                            "pressure.drop.average",
                            "system",
                            self.compute_average_pressure_drop(),
                        ),
                        record("energy.compression", "system", energy_compression),
                        record("energy.vacuum", "system", energy_vacuum),
                        record("energy.jacket", "system", energy_jacket),
                        record("steam.mass", "H2O", steam_mass),
                        record("energy.steam", "system", steam_energy),
                    ]
                )
        elif step_key == "cool":
            _, energy_vacuum, energy_jacket = self.get_energies()
            records.extend(
                [
                    record("energy.vacuum", "system", energy_vacuum),
                    record("energy.jacket", "system", energy_jacket),
                ]
            )
        elif step_key == "press":
            energy_compression, _, energy_jacket = self.get_energies()
            records.extend(
                [
                    record(
                        "pressure.drop.average",
                        "system",
                        self.compute_average_pressure_drop(),
                    ),
                    record("energy.compression", "system", energy_compression),
                    record("energy.jacket", "system", energy_jacket),
                ]
            )

        records.append(record("step.duration", "system", self.time[-1]))
        return StepMetrics(step=step_names[step_key], records=tuple(records))

    def _outlet_settings(self):
        """Return the step's resolved `outlet` group: where its outlet stream goes."""
        return self.pb.column_config.step_settings.outlet

    def _outlet_metric_records(self, mass_metrics):
        """Return `(mass records, vacuum energy [MJ], condenser energy records)`.

        The step's `outlet` group decides where the outlet stream goes.  A collected
        step feeds the product line -- an optional condenser, the vacuum pump, the
        vacuum-discharge flash -- and records what leaves each stage.  A vented step
        records what it throws away under `mass.vented`, which never enters a product
        KPI.  The vacuum energy follows the same line: behind a condenser the pump
        moves only the condenser vapour and the inerts; without one it moves the whole
        outlet stream, which is the solver's own figure.
        """
        record = self._step_metric_record
        outlet = self._outlet_settings()
        if not outlet.collect_product:
            _, energy_vacuum, _ = self.get_energies()
            return self._vented_mass_records(mass_metrics), energy_vacuum, []

        self._require_species(("CO2",), "product-line KPIs")
        # Every species other than CO2 and H2O passes the separators unchanged
        # (the binary flash is CO2/H2O); with the DAC mixture these are O2 and N2.
        inert_records = lambda metric_id: [  # noqa: E731
            record(metric_id, species, abs(mass_metrics[species]["change"]))
            for species in self._inert_species()
        ]
        if "H2O" not in self.gas_species:
            # Nothing condenses: no condenser and no vacuum-discharge flash; the
            # product is the outlet stream itself.
            print(
                f"No H2O in the mixture: step '{self.step}' collects its outlet "
                "without condenser or vacuum-discharge flash."
            )
            self._get_flowrates()
            m_CO2 = self.pb.MW_CO2 * simpson(self.n_CO2_out, x=self.time)
            records = [record("mass.after.vacuum", "CO2", m_CO2)]
            records.extend(inert_records("mass.after.vacuum"))
            _, energy_vacuum, _ = self.get_energies()
            return records, energy_vacuum, []

        records = []
        T_cond = float(outlet.condenser_temperature_K)
        if outlet.condenser:
            m_CO2_cond, m_H2O_cond = self.run_condensation_separator(T_cond=T_cond)
            records.extend(
                [
                    record("mass.after.condenser", "CO2", m_CO2_cond),
                    record("mass.after.condenser", "H2O", m_H2O_cond),
                ]
            )
            records.extend(inert_records("mass.after.condenser"))
        m_CO2_vac, m_H2O_vac = self.run_vacuum_separator(
            after_condenser=outlet.condenser
        )
        records.extend(
            [
                record("mass.after.vacuum", "CO2", m_CO2_vac),
                record("mass.after.vacuum", "H2O", m_H2O_vac),
            ]
        )
        records.extend(inert_records("mass.after.vacuum"))
        if not outlet.condenser:
            _, energy_vacuum, _ = self.get_energies()
            return records, energy_vacuum, []

        energy_vacuum = self.update_vacuum_pump_energy(T_cond=T_cond)
        energy_condenser = self.compute_condenser_duty(T_cond=T_cond)
        # Cooling water carries the duty down to its own limit; a condenser set above
        # that limit is cooled by water alone, so the split ends at the product
        # temperature and nothing is left for a chiller.
        energy_condenser_above_ambient = self.compute_condenser_duty_above_ambient(
            max(T_COOLING_WATER_LIMIT, T_cond)
        )
        condenser_records = [
            record("energy.condenser", "system", -energy_condenser),
            record(
                "energy.condenser.ambient",
                "system",
                -energy_condenser_above_ambient,
            ),
            # The cycle KPIs price this step's chiller at the temperature it delivers.
            record("temperature.condenser", "system", T_cond),
        ]
        return records, energy_vacuum, condenser_records

    def _inert_species(self):
        """Species the CO2/H2O separators pass unchanged, in mixture order."""
        return tuple(
            species for species in self.gas_species if species not in ("CO2", "H2O")
        )

    def _vented_mass_records(self, mass_metrics):
        """Return the `mass.vented` records of a step whose outlet is not collected.

        What a vented step discharges is lost, not recovered, so it is recorded under
        its own metric id and never enters a product KPI.  This is the bed outlet
        before any separation: for the loss that matters -- the CO2 a preheating step
        throws away -- separation is irrelevant, a condenser only takes water.  The
        outlet is what left the bed plus what was fed: the injected steam of a steam
        step leaves through the vent too.  Signed, positive when the species leaves.
        """
        record = self._step_metric_record
        steam_in = (
            self.compute_steam_metrics()[0] if self.step in {"s_purge", "s_des"} else None
        )
        records = []
        for species in self.gas_species:
            vented = -mass_metrics[species]["change"]
            if species == "H2O" and steam_in is not None:
                vented += steam_in
            records.append(record("mass.vented", species, vented))
        return records

    def run_condensation_separator(self, T_cond=278.15):
        """
        Run condensation separator on outlet flowrates using FlashSeparator.
        """
        time   = self.time      # [s], shape: (Nt,)
        MW_CO2, MW_H2O = self.pb.MW_CO2, self.pb.MW_H2O  # [kg/mol]
        self._get_flowrates()
        carries_stream = self._condenser_stream_mask()
        n_CO2 = self.n_CO2_out            # [mol/s]
        n_H2O = self.n_H2O_out            # [mol/s]

        n_total = n_CO2 + n_H2O
        z_CO2 = n_CO2 / (n_total + 1e-32)

        p_cond = np.sum(self.p_tot[-1, :] * self.A_rings[:, None], axis=0) / np.sum(self.A_rings)  # outlet pressure

        # Set up separator model
        CO2 = Component("CO2", 304.2, 73.74)
        H2O = Component("H2O", 647.096, 220.64)
        separator = FlashSeparator(CO2, H2O)

        # Loop over time to run flash separator
        results = []
        for i in range(self.Nt):
            if not carries_stream[i]:
                results.append(dict(_NO_STREAM_FLASH))
                continue
            results.append(
                self._binary_flash(separator, n_total[i], z_CO2[i], T_cond, p_cond[i])
            )
            
        # Store as DataFrame for easier plotting/postprocessing
        self.condenser_results_df = pd.DataFrame(results)
        mean_values = self.condenser_results_df.mean()

        # Preview result
        separator.pretty_print(mean_values, title=f"Average Condenser Output (0 - {self.time[-1]:.2f} s)", n_feed_in=n_total.mean())

        # Extract arrays
        nv     = self.condenser_results_df["nv"].to_numpy()
        y_CO2  = self.condenser_results_df["y_CO2"].to_numpy()
        y_H2O  = self.condenser_results_df["y_H2O"].to_numpy()
        
        # Instantaneous molar flows [mol/s]
        n_CO2 = nv * y_CO2
        n_H2O = nv * y_H2O

        # Total mass [kg] using Simpson's rule
        m_CO2_before = MW_CO2 * simpson(self.n_CO2_out, x=time)
        m_H2O_before = MW_H2O * simpson(self.n_H2O_out, x=time)
        
        m_CO2 = MW_CO2 * simpson(n_CO2, x=time)
        m_H2O = MW_H2O * simpson(n_H2O, x=time)

        # Recovery = retained / original
        recovery_CO2 = m_CO2 / (m_CO2_before + 1e-32)
        recovery_H2O = m_H2O / (m_H2O_before + 1e-32)

        print("\n[Condenser Product Summary]")
        print(f"  Mass CO2 (after condenser) : {1000 * m_CO2:.2f} g")
        print(f"  Mass CO2 (before condenser): {1000 * m_CO2_before:.2f} g")
        print(f"  CO2 ratio         : {100 * recovery_CO2:.4f} %")

        print()
        print(f"  Mass H2O (after condenser) : {1000 * m_H2O:.2f} g")
        print(f"  Mass H2O (before condenser): {1000 * m_H2O_before:.2f} g")
        print(f"  H2O ratio         : {100 * recovery_H2O:.2f} %")
        
        return m_CO2, m_H2O

    def compute_condenser_duty(self, T_cond=278.15):
        """
        Compute condenser duty over time using enthalpy balance.
        Assumes condenser output is at T_cond (reference temp).
        """
        # Required data
        if not hasattr(self, "n_CO2_out"):
            self._get_flowrates()

        n_l   = self.condenser_results_df["nl"].to_numpy()        # [mol/s]
        x_CO2 = self.condenser_results_df["x_CO2"].to_numpy()
        x_H2O = self.condenser_results_df["x_H2O"].to_numpy()

        return self._condenser_duty_from_flash(T_cond, n_l, x_CO2, x_H2O)

    def compute_condenser_duty_above_ambient(self, T_ambient_limit):
        """
        Compute the part of the condenser duty [MJ] released before the stream reaches
        `T_ambient_limit`.  Cooling water carries that part, so only the remainder needs
        a chiller.

        The stream leaves the bed well above ambient and condenses along a dew line, so
        the duty is not released at one temperature -- charging all of it at the product
        temperature prices heat that ambient takes for free.  This is the same enthalpy
        path and the same flash as the total, evaluated at a warmer end temperature:
        `total - this` is therefore a duty difference, not a second model.
        """
        if not hasattr(self, "n_CO2_out"):
            self._get_flowrates()

        T_end = float(T_ambient_limit)
        n_l, x_CO2, x_H2O = self._condenser_flash_at(T_end)
        return self._condenser_duty_from_flash(T_end, n_l, x_CO2, x_H2O)

    def _condenser_flash_at(self, T_end):
        """
        Flash the bed outlet at `T_end`, returning `(n_l, x_CO2, x_H2O)`.

        Feed basis is the one `run_condensation_separator` uses -- CO2 and H2O only --
        so that a duty computed here and the recorded total are on one basis.

        `FlashSeparator.solve` assumes two phases exist: it reads y and x off the
        equilibrium at (T, p) alone and splits the feed with the lever rule.  Above the
        dew point that assumption is false, and the algebra says so -- the denominator
        changes sign, y_CO2 goes negative, and the lever rule hands back more liquid
        than there was feed.  It raises nothing, it returns a number.  The product
        temperature is always far below the dew point so the recorded total never meets
        this; the cooling-water limit is a site parameter and can be pushed above it,
        hence the gate.
        """
        n_CO2 = self.n_CO2_out                                    # [mol/s]
        n_H2O = self.n_H2O_out                                    # [mol/s]
        n_total = n_CO2 + n_H2O
        z_CO2 = n_CO2 / (n_total + 1e-32)

        p_cond = np.sum(self.p_tot[-1, :] * self.A_rings[:, None], axis=0) / np.sum(self.A_rings)

        separator = FlashSeparator(
            Component("CO2", 304.2, 73.74), Component("H2O", 647.096, 220.64)
        )
        p_sat_end = float(ThermoModels.p_sat_H2O(T_end))

        carries_stream = self._condenser_stream_mask()
        n_l = np.zeros(self.Nt)
        x_CO2 = np.zeros(self.Nt)
        x_H2O = np.zeros(self.Nt)
        for i in range(self.Nt):
            if not carries_stream[i]:
                continue                                          # no stream: nothing to flash
            if (1.0 - z_CO2[i]) * p_cond[i] <= p_sat_end:
                continue                                          # dry at T_end: no liquid
            res = separator.solve(
                n_feed=n_total[i], z_CO2_feed=z_CO2[i], T=T_end, p=p_cond[i]
            )
            nl = res["nl"]
            n_l[i] = min(max(nl, 0.0), n_total[i]) if np.isfinite(nl) else 0.0
            x_CO2[i], x_H2O[i] = res["x_CO2"], res["x_H2O"]
        return n_l, x_CO2, x_H2O

    def _condenser_duty_from_flash(self, T_cond, n_l, x_CO2, x_H2O):
        """
        Enthalpy path of the condenser duty [MJ] for an outlet cooled to `T_cond`.

        Single home for the enthalpy constants: `compute_condenser_duty` feeds it the
        flash cached at the product temperature and `compute_condenser_duty_above_ambient`
        feeds it a flash run at the cooling-water limit.
        """
        mixed_cups = self.get_mixed_cups()

        # Every species of the outlet stream, in mixture order (DAC: CO2, H2O, O2, N2).
        n_feed = sum(getattr(self, f"n_{species}_out") for species in self.gas_species)  # [mol/s]

        p_cond = np.sum(self.p_tot[-1, :] * self.A_rings[:, None], axis=0) / np.sum(self.A_rings)  # [Pa]
        T_feed = mixed_cups["T_out"]  # [K]
        time = self.time                                           # [s]
        
        # Constants
        cp_gas = {species: getattr(self.pb, f"cp_g_{species}") for species in self.gas_species}  # [J/mol/K]
        cp_liq_H2O = 75.3                                                                       # [J/mol/K]
        dh_sol_CO2 = (106.56 - (6.2634e4 / T_cond) + (7.475e6 / T_cond**2)) * 1000              # [J/mol]  (Carrol et al 1991)
        
        # Sensible cooling of inlet gas
        # Mixture heat capacity of the outlet stream, summed in mixture order.
        Q_sensible = (
            n_feed * sum(
                mixed_cups[f"y_{species}"] * cp_gas[species]
                for species in self.gas_species
            ) * (T_feed - T_cond)
        )  # [W]

        # CO2 dissolution
        Q_dissolution = n_l * x_CO2 * dh_sol_CO2  # [W]

        # T boiling and dhvap (Ward correlation)
        p_bar = p_cond / 1e5
        T_bp = 197 * (p_bar)**0.1367 - 97.97 + 273.15        # [K]
        dh_vap_kJ_per_kg = -446.2 * (p_bar)**0.1508 + 2707   # [kJ/kg]
        dh_vap = dh_vap_kJ_per_kg * 1e3 * self.pb.MW_H2O     # [J/mol]

        # H2O condensation termk
        Q_cond_H2O = n_l * x_H2O * (
            cp_gas["H2O"] * (T_bp - T_cond) - dh_vap + cp_liq_H2O * (T_cond - T_bp)
        )  # [W]

        # Total condenser duty
        Q_cond = Q_sensible - Q_dissolution - Q_cond_H2O     # [W]

        # A sample without a stream has an undefined composition and temperature,
        # so it draws no duty. It stays on the time axis: dropping it and joining
        # its neighbours would integrate across a gap that carries no flow.
        Q_cond = np.where(self._condenser_stream_mask(), Q_cond, 0.0)

        # Integrate over time to get energy in MJ
        E_cond_MJ = simpson(Q_cond, x=time) / 1e6

        return E_cond_MJ

    def run_vacuum_separator(self, after_condenser=True):
        """
        Run vacuum separator on outlet flowrates using FlashSeparator.

        Behind a condenser the flash takes the condenser vapour; without one it takes
        the bed outlet itself (CO2 and H2O, the condenser's feed basis).
        """
        time   = self.time
        MW_CO2, MW_H2O = self.pb.MW_CO2, self.pb.MW_H2O  # [kg/mol]
        T_discharge = self.pb.T_amb  # outlet temperature at each time step
        p_discharge = self.pb.p_amb  # outlet pressure at each time step

        # Create separator object
        CO2 = Component("CO2", 304.2, 73.74)
        H2O = Component("H2O", 647.096, 220.64)
        separator = FlashSeparator(CO2, H2O)

        if after_condenser:
            # Extract vapor output from condenser
            z_CO2_vac = self.condenser_results_df["y_CO2"].to_numpy()
            n_feed_vac = self.condenser_results_df["nv"].to_numpy()

            # Loop over time to run flash separator
            results = []
            for i in range(self.Nt):
                results.append(
                    self._binary_flash(
                        separator, n_feed_vac[i], z_CO2_vac[i], T_discharge, p_discharge
                    )
                )
        else:
            n_feed_vac, results = self._raw_outlet_vacuum_flash(
                separator, T_discharge, p_discharge
            )
            
         # Store results
        self.vacuum_results_df = pd.DataFrame(results)
        mean_values = self.vacuum_results_df.mean()

        # Preview result
        separator.pretty_print(mean_values, title=f"Average Vacuum Output (0 - {self.time[-1]:.2f} s)", n_feed_in=n_feed_vac.mean())
        
        # Extract arrays
        nv     = self.vacuum_results_df["nv"].to_numpy()       # vapor out
        y_CO2  = self.vacuum_results_df["y_CO2"].to_numpy()
        y_H2O  = self.vacuum_results_df["y_H2O"].to_numpy()

        n_CO2 = nv * y_CO2
        n_H2O = nv * y_H2O

        # Total mass [kg] using Simpson's rule
        m_CO2_before = MW_CO2 * simpson(self.n_CO2_out, x=time)
        m_H2O_before = MW_H2O * simpson(self.n_H2O_out, x=time)
        
        m_CO2 = MW_CO2 * simpson(n_CO2, x=time)
        m_H2O = MW_H2O * simpson(n_H2O, x=time)

        # Recovery = retained / original
        lost_CO2 = (m_CO2_before - m_CO2) / (m_CO2_before + 1e-32)
        lost_H2O = (m_H2O_before - m_H2O) / (m_H2O_before + 1e-32)

        print("\n[Vacuum Discharge Product Summary]")
        print(f"  Mass CO2 (after vacuum) : {1000 * m_CO2:.2f} g")
        print(f"  Mass CO2 (before condenser): {1000 * m_CO2_before:.2f} g")
        print(f"  CO2 recovered         : {100 * (1 - lost_CO2):.4f} %")

        print()
        print(f"  Mass H2O (after vacuum) : {1000 * m_H2O:.2f} g")
        print(f"  Mass H2O (before condenser): {1000 * m_H2O_before:.2f} g")
        print(f"  H2O removed         : {100 * lost_H2O:.4f} %")
        
        return m_CO2, m_H2O
    
    @staticmethod
    def _binary_flash(separator, n_feed, z_CO2, T, p):
        """Flash one CO2/H2O sample at (T, p), or pass it through if it is dry.

        `FlashSeparator.solve` assumes two phases: it reads y and x off the
        equilibrium at (T, p) alone and splits the feed with the lever rule.  A stream
        drier than its dew point (binary basis, as in `_condenser_flash_at`) has no
        liquid, and the solve would return a split anyway, with no error.  Such a
        sample leaves as vapour, unchanged.  A two-phase result is held inside the
        feed: 0 <= nv <= n_feed.
        """
        if (1.0 - z_CO2) * p <= float(ThermoModels.p_sat_H2O(T)):
            return {
                "y_CO2": z_CO2,
                "y_H2O": 1.0 - z_CO2,
                "x_CO2": 0.0,
                "x_H2O": 0.0,
                "nv": n_feed,
                "nl": 0.0,
            }
        res = separator.solve(n_feed=n_feed, z_CO2_feed=z_CO2, T=T, p=p)
        nv = res["nv"]
        if not (np.isfinite(nv) and 0.0 <= nv <= n_feed):
            res["nv"] = min(max(nv, 0.0), n_feed) if np.isfinite(nv) else 0.0
            res["nl"] = n_feed - res["nv"]
        return res

    def _raw_outlet_vacuum_flash(self, separator, T, p):
        """Flash the bed outlet at the vacuum discharge, with no condenser before it.

        Returns `(n_feed, results)` per sample; a sample without a stream contributes
        nothing.
        """
        self._get_flowrates()
        carries_stream = self._condenser_stream_mask()
        n_CO2 = self.n_CO2_out                                    # [mol/s]
        n_H2O = self.n_H2O_out                                    # [mol/s]
        n_total = n_CO2 + n_H2O
        z_CO2 = n_CO2 / (n_total + 1e-32)

        n_feed = np.where(carries_stream, n_total, 0.0)
        results = [
            self._binary_flash(separator, n_total[i], z_CO2[i], T, p)
            if carries_stream[i]
            else dict(_NO_STREAM_FLASH)
            for i in range(self.Nt)
        ]
        return n_feed, results

    def update_vacuum_pump_energy(self, T_cond=278.15):
        """
        Update the vacuum pump power based on the condenser outlet.
        """
        if not hasattr(self, 'condenser_results_df'):
            raise RuntimeError("Run vacuum separator first to get results.")

        # Calculate the average volumetric flow rate at the outlet fo the condenser
        inert_flow = sum(getattr(self, f"n_{species}_out") for species in self._inert_species())
        nv_out_cond = self.condenser_results_df["nv"].to_numpy() + inert_flow  # [mol/s]

        # Mixture properties
        cp_mix = self.pb.cp_g_mix  # [J/mol/K]
        cv_mix = cp_mix - R_gas
        gamma = cp_mix / cv_mix

        # Pressure ratios
        p_outlet = np.sum(self.p_tot[-1, :] * self.A_rings[:, None], axis=0) / np.sum(self.A_rings)  # outlet pressure
        pressure_ratio_vac = self.pb.p_amb / p_outlet                       # Discharge at atmospheric pressure
        Q_out_cond = nv_out_cond * R_gas * T_cond / p_outlet       # [m³/s] using ideal gas law

        # Vacuum Pump Model (Fully Compressible, Polytropic) 
        eta_vacuum_pump = 0.8 * 19.55 * (p_outlet/1e5) / (1 + 19.55 * p_outlet/1e5)                 # Ward et al.
        power_vacuum_pump1 = (1 / eta_vacuum_pump) * (gamma / (gamma - 1)) * p_outlet * Q_out_cond 
        power_vacuum_pump2 = (pressure_ratio_vac) ** ((gamma - 1) / gamma) - 1
        power_vacuum_pump = power_vacuum_pump1 * power_vacuum_pump2  
        
        # Integrate the power over time to get energy consumption
        time = self.time  # [s]
        E_vacuum_after_condenser = simpson(power_vacuum_pump, x=time) / 1e6  # Convert to MJ
        
        return E_vacuum_after_condenser
                
    
