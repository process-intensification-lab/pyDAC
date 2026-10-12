"""
pyDAC - Cyclic steady-state profile plots.
This module defines the CSS plotting specifications and routines used to
visualize outlet histories across a whole converged cycle, including step
styling, field aliases, and cooldown-jump handling.

Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
"""

import json
import os
import pickle
from dataclasses import dataclass

from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.constants import gas_constant as R_gas

from pydac.core.species_catalog import get_species_properties

from pydac.utils.postprocessing import cyclic_profiles as _profile_utils
from pydac.utils.postprocessing.saved_solution import saved_solution_view
from pydac.utils.postprocessing.outlet import outlet_mole_fractions
from pydac.utils.run import (
    infer_desorption_step_from_results,
    infer_results_base_folder,
    ordered_saved_step_folders,
)


@dataclass(frozen=True)
class CSSFieldSpec:
    ylabel: str
    title: str
    filename: str
    scale: float = 1.0
    offset: float = 0.0


class CyclicSteadyStateProfiles:
    FALLBACK_STEP_ORDER = [
        "startup", "ads", "blow", "heat", "w_purge", "s_purge",
        "s_des", "w_des", "press", "cool",
    ]
    CLOSED_OUTLET_STEPS = {"cool", "press"}
    CLOSED_INLET_STEPS = {"blow", "heat", "w_purge", "w_des", "cool"}
    O2_PPM_LIMIT = 40.0
    PPM_LOG_FLOOR = 1e-12
    STEP_LABELS = {
        "startup": "Startup",
        "ads": "Adsorption",
        "blow": "Blowdown",
        "heat": "Heating",
        "w_purge": "Wall purge",
        "s_purge": "Steam purge",
        "s_des": "Steam Desorption",
        "w_des": "Wall-Heated Desorption",
        "press": "Pressurization",
        "cool": "Cooldown",
    }

    STEP_COLORS = {
        "startup": "tab:gray",
        "ads": "tab:blue",
        "blow": "tab:purple",
        "heat": "tab:orange",
        "w_purge": "#B8860B",
        "s_purge": "#008C95",
        "s_des": "tab:red",
        "w_des": "tab:red",
        "press": "tab:brown",
        "cool": "tab:green",
    }

    FIELD_ALIASES = {
        "Pressure": "pressure",
        "p_tot": "pressure",
        "O2_ppm": "ppm_O2",
        "N2_ppm": "ppm_N2",
    }

    FIELD_SPECS = {
        "y_CO2": CSSFieldSpec(
            ylabel=r"Outlet $y_{CO_2}$ [-]",
            title=r"CSS Outlet $y_{CO_2}$",
            filename="css_outlet_y_CO2",
        ),
        "y_H2O": CSSFieldSpec(
            ylabel=r"Outlet $y_{H_2O}$ [-]",
            title=r"CSS Outlet $y_{H_2O}$",
            filename="css_outlet_y_H2O",
        ),
        "ppm_O2": CSSFieldSpec(
            ylabel=r"Outlet $O_2$ [ppmmol]",
            title=r"CSS Outlet $O_2$",
            filename="css_outlet_ppm_O2",
        ),
        "ppm_N2": CSSFieldSpec(
            ylabel=r"Outlet $N_2$ [ppmmol]",
            title=r"CSS Outlet $N_2$",
            filename="css_outlet_ppm_N2",
        ),
        "q_CO2": CSSFieldSpec(
            ylabel=r"Outlet $q_{CO_2}$ [mol/kg]",
            title=r"CSS Outlet $q_{CO_2}$",
            filename="css_outlet_q_CO2",
        ),
        "q_H2O": CSSFieldSpec(
            ylabel=r"Outlet $q_{H_2O}$ [mol/kg]",
            title=r"CSS Outlet $q_{H_2O}$",
            filename="css_outlet_q_H2O",
        ),
        "T_bed": CSSFieldSpec(
            ylabel=r"Outlet $T_{bed}$ [$^\circ$C]",
            title=r"CSS Outlet $T_{bed}$",
            filename="css_outlet_T_bed",
            offset=-273.15,
        ),
        "T_wall": CSSFieldSpec(
            ylabel=r"Outlet $T_{wall}$ [$^\circ$C]",
            title=r"CSS Outlet $T_{wall}$",
            filename="css_outlet_T_wall",
            offset=-273.15,
        ),
        "pressure": CSSFieldSpec(
            ylabel=r"Outlet $p_{tot}$ [bar]",
            title=r"CSS Outlet Pressure",
            filename="css_outlet_pressure",
            scale=1.0 / 1e5,
        ),
        "u_out": CSSFieldSpec(
            ylabel=r"Outlet $u_{ax}$ [m/s]",
            title=r"CSS Outlet Velocity",
            filename="css_outlet_velocity",
        ),
    }

    DEFAULT_PLOT_FIELDS = ("y_CO2", "y_H2O", "ppm_O2", "q_CO2", "q_H2O", "T_bed", "T_wall", "pressure")
    COOL_JUMP_FIELDS = frozenset(DEFAULT_PLOT_FIELDS)
    COOL_JUMP_MAX_TIME_S = 3600.0
    COOL_JUMP_MAX_FRACTION = 0.05
    COOL_JUMP_MIN_IMPROVEMENT = 1.5
    COOL_JUMP_TOL_REL = 0.02
    COOL_JUMP_ABS_FLOORS = {
        "y_CO2": 2e-6,
        "y_H2O": 2e-6,
        "ppm_O2": 1.0,
        "ppm_N2": 1.0,
        "q_CO2": 5e-4,
        "q_H2O": 5e-4,
        "T_bed": 0.5,
        "T_wall": 0.5,
        "pressure": 50.0,
        "u_out": 1e-3,
    }

    PRESSURE_DROP_UNITS = _profile_utils.PRESSURE_DROP_UNITS

    _radial_average = staticmethod(_profile_utils.radial_average)
    _clip_mole_fraction = staticmethod(_profile_utils.clip_mole_fraction)
    _solve_pressure_drop_velocity = staticmethod(
        _profile_utils.solve_pressure_drop_velocity
    )
    _pressure_drop_parameters_from_metadata = staticmethod(
        _profile_utils.pressure_drop_parameters_from_metadata
    )

    @staticmethod
    def _set_global_style():
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans"],
            "axes.labelsize": 15,
            "axes.labelweight": "regular",
            "axes.titlesize": 17,
            "axes.linewidth": 1.2,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "legend.fontsize": 14,
            "lines.linewidth": 3,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        })

    @staticmethod
    def _sig3_formatter():
        return mticker.FuncFormatter(lambda x, pos: f"{x:.3g}")

    def __init__(self, base_folder="simulation_results/cyclic_mode", css_cycle_idx=None):
        self.base_folder = base_folder
        self.results_base_folder = infer_results_base_folder(self.base_folder)
        self.css_cycle_idx = self._find_last_cycle() if css_cycle_idx is None else css_cycle_idx
        self.desorption_step = None
        self.step_order = None
        self.cycle_metadata = None
        self.outlet_profiles = {}
        self.is_1d = None
        self._mixed_cup_zero_den_warned = False
        self.layout_signature = None
        self.gas_species = ()
        self.adsorbates = ()
        self.field_specs = dict(self.FIELD_SPECS)
        self.default_plot_fields = tuple(self.DEFAULT_PLOT_FIELDS)

        self._load_css_cycle()

        # One folder per reported cycle: reporting another cycle must not overwrite these.
        self.plot_folder = os.path.join(
            self.results_base_folder, "plots_css", f"cycle_{self.css_cycle_idx}"
        )
        os.makedirs(self.plot_folder, exist_ok=True)

    def _normalize_field_name(self, field):
        key = self.FIELD_ALIASES.get(field, field)
        if key not in self.field_specs:
            raise KeyError(f"Unsupported CSS field '{field}'. Available fields: {sorted(self.field_specs)}")
        return key

    def _register_layout(self, layout):
        if self.layout_signature is not None and layout.signature != self.layout_signature:
            raise ValueError(
                "CSS cycle contains incompatible StateLayouts "
                f"({layout.signature} != {self.layout_signature})."
            )
        if self.layout_signature is not None:
            return
        self.layout_signature = layout.signature
        self.gas_species = layout.mixture.species
        self.adsorbates = layout.adsorbates
        for species in self.gas_species:
            self.field_specs.setdefault(
                f"y_{species}",
                CSSFieldSpec(
                    ylabel=f"Outlet y({species}) [-]",
                    title=f"CSS Outlet {species} Mole Fraction",
                    filename=f"css_outlet_y_{species}",
                ),
            )
            self.field_specs.setdefault(
                f"ppm_{species}",
                CSSFieldSpec(
                    ylabel=f"Outlet {species} [ppmmol]",
                    title=f"CSS Outlet {species}",
                    filename=f"css_outlet_ppm_{species}",
                ),
            )
        for species in self.adsorbates:
            self.field_specs.setdefault(
                f"q_{species}",
                CSSFieldSpec(
                    ylabel=f"Outlet q({species}) [mol/kg]",
                    title=f"CSS Outlet {species} Loading",
                    filename=f"css_outlet_q_{species}",
                ),
            )
        self.default_plot_fields = (
            *(f"y_{species}" for species in self.gas_species),
            *(f"q_{species}" for species in self.adsorbates),
            "T_bed",
            "T_wall",
            "pressure",
        )

    def _find_last_cycle(self):
        search_roots = [self.results_base_folder]
        if self.results_base_folder != self.base_folder:
            search_roots.append(self.base_folder)

        subdirs = []
        for root in search_roots:
            subdirs = [
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d)) and d.startswith("cycle_")
            ]
            if subdirs:
                self.results_base_folder = root
                break

        if not subdirs:
            raise FileNotFoundError(f"No 'cycle_*' folders found in {self.results_base_folder}")

        cycle_ids = [int(d.split("_")[1]) for d in subdirs]
        print(f"[CyclicProfiles] Found cycles: {cycle_ids}, using last as CSS cycle.")
        return max(cycle_ids)

    def _load_cycle_metadata(self, cycle_folder):
        metadata_path = os.path.join(cycle_folder, "cycle_metadata.json")
        if not os.path.isfile(metadata_path):
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.cycle_metadata = metadata
        steps = metadata.get("executed_steps")
        if not isinstance(steps, list) or not all(isinstance(step, str) for step in steps):
            return None
        return steps

    def _get_cycle_step_order(self):
        if self.step_order is not None:
            ordered = [step for step in self.step_order if step in self.outlet_profiles]
            if ordered:
                return ordered
        return [step for step in self.FALLBACK_STEP_ORDER if step in self.outlet_profiles]

    def _load_css_cycle(self):
        cycle_folder = os.path.join(self.results_base_folder, f"cycle_{self.css_cycle_idx}")
        if not os.path.isdir(cycle_folder) and self.results_base_folder != self.base_folder:
            cycle_folder = os.path.join(self.base_folder, f"cycle_{self.css_cycle_idx}")
        if not os.path.isdir(cycle_folder):
            raise FileNotFoundError(f"CSS cycle folder not found: {cycle_folder}")

        self.desorption_step = infer_desorption_step_from_results(cycle_folder)
        self.step_order = self._load_cycle_metadata(cycle_folder)
        self._load_cycle(infer_results_base_folder(cycle_folder))

    def _load_cycle(self, folder):
        preferred = list(self.step_order or []) + self.FALLBACK_STEP_ORDER
        for folder_step_name, step_name in ordered_saved_step_folders(folder, preferred):
            if (
                step_name in {"s_des", "w_des"}
                and self.desorption_step is not None
                and step_name != self.desorption_step
            ):
                continue

            sol_path = os.path.join(folder, folder_step_name, "sol.pkl")
            if not os.path.isfile(sol_path):
                continue

            with open(sol_path, "rb") as f:
                sol = pickle.load(f)

            outlet_data = self._extract_outlet_profiles(sol, step_name)
            if step_name in self.outlet_profiles:
                raise ValueError(
                    f"Multiple result folders normalize to step '{step_name}' in '{folder}'."
                )
            self.outlet_profiles[step_name] = outlet_data

    def _estimate_outlet_velocity(
        self,
        step_name,
        p_tot,
        rho_out,
        z_c,
        bed_length,
        pressure_drop_parameters,
    ):
        if step_name in self.CLOSED_OUTLET_STEPS:
            return np.zeros_like(rho_out)

        if p_tot.shape[0] < 2 or np.asarray(z_c).size < 2:
            return np.zeros_like(rho_out)

        dz = float(z_c[-1] - z_c[-2]) * float(bed_length)
        if dz <= 0.0:
            return np.zeros_like(rho_out)

        grad_p_out = (p_tot[-2, :, :] - p_tot[-1, :, :]) / dz
        u_out = self._solve_pressure_drop_velocity(
            grad_p_out,
            rho_out,
            **pressure_drop_parameters,
        )
        u_out[np.abs(u_out) < 1e-10] = 0.0
        return u_out

    def _outlet_average(self, step_name, field_z, A_rings, rho_z, u_out_z):
        """Mixing-cup where the outlet is a real advected stream, else area-weighted.

        A mass-flux weighted mean is the right outlet reading only when gas is
        actually being carried through the bed. On a closed-inlet step the sole
        motion is gas venting as it expands, biased towards the hot near-wall
        ring, so the flow weights collapse onto the wall and the mean reports the
        wall rather than the bed; on a closed-outlet step there is no flow at all
        and the weights are degenerate. Both fall back to the area-weighted mean,
        which is the cross-section the 1D model represents.

        The closed-inlet set is the pressure-BC classification of
        `src/pydac/core/column_model.py` (`_set_boundary_conditions`, `_gas_property_composition`),
        so mixing-cup survives on `startup`, `ads`, `s_purge` and `s_des` only.
        """
        if step_name in self.CLOSED_INLET_STEPS or step_name in self.CLOSED_OUTLET_STEPS:
            return self._radial_average(field_z, A_rings)
        return self._outlet_measured_average(field_z, A_rings, rho_z, u_out_z)

    def _outlet_measured_average(self, field_z, A_rings, rho_z, u_out_z):
        if field_z.shape[0] == 1:
            return np.asarray(field_z[0, :], dtype=float)

        mass_flux = np.clip(rho_z * u_out_z, 0.0, None)

        weights = mass_flux * np.asarray(A_rings, dtype=float)[:, None]
        den = np.sum(weights, axis=0)
        num = np.sum(field_z * weights, axis=0)

        out = np.empty_like(den)
        mask = np.abs(den) > 1e-32
        out[mask] = num[mask] / den[mask]

        if not np.all(mask):
            if not self._mixed_cup_zero_den_warned:
                print(
                    "[CyclicProfiles] Mixed-cup outlet average had zero denominator; "
                    "falling back to radial average for those points."
                )
                self._mixed_cup_zero_den_warned = True
            out[~mask] = self._radial_average(field_z, A_rings)[~mask]

        return out

    def _extract_outlet_profiles(self, sol, step_name):
        saved = saved_solution_view(sol)
        self._register_layout(saved.layout)
        time = saved.time_s
        extra = sol.get("extra", {})

        A_rings = np.asarray(extra["A_rings"], dtype=float).reshape(-1)
        c_tot = saved.field("c_tot")
        T_bed = saved.field("T_bed")
        T_wall = saved.field("T_wall")
        z_c = np.asarray(
            extra.get("z_c", np.arange(saved.layout.nz, dtype=float)), dtype=float
        )
        pressure_drop_parameters = self._pressure_drop_parameters_from_metadata(
            extra,
            step_name,
        )

        is_1d = saved.layout.nr == 1
        if self.is_1d is not None and self.is_1d != is_1d:
            raise ValueError("CSS cycle mixes 1D and 2D saved solutions.")
        self.is_1d = is_1d

        independent = {
            species: saved.field(f"c_{species}")
            for species in saved.layout.independent_gas_species
        }
        closure = c_tot - sum(independent.values())
        concentrations = {
            species: (
                closure
                if species == saved.layout.closure_species
                else independent[species]
            )
            for species in saved.layout.mixture.species
        }
        fractions = {
            species: self._clip_mole_fraction(concentration / (c_tot + 1e-32))
            for species, concentration in concentrations.items()
        }
        molecular_weights = {
            species: float(get_species_properties(species).molecular_weight_kg_per_mol)
            for species in saved.layout.mixture.species
        }
        rho = sum(
            concentrations[species] * molecular_weights[species]
            for species in saved.layout.mixture.species
        )
        p_tot = c_tot * R_gas * T_bed

        z_out = -1
        if pressure_drop_parameters.get("formulation") == "isotropic_vector":
            if "outlet_face" not in sol:
                raise ValueError("Vector-drag cyclic profiles require saved native outlet-face fluxes.")
            u_out_z = np.asarray(sol["outlet_face"]["velocity_m_s"])
        else:
            u_out_z = self._estimate_outlet_velocity(
                step_name=step_name,
                p_tot=p_tot,
                rho_out=rho[z_out, :, :],
                z_c=z_c,
                bed_length=float(extra["L_bed"]),
                pressure_drop_parameters=pressure_drop_parameters,
            )

        fraction_out = {
            species: self._outlet_average(
                step_name,
                values[z_out, :, :],
                A_rings=A_rings,
                rho_z=rho[z_out, :, :],
                u_out_z=u_out_z,
            )
            for species, values in fractions.items()
        }
        T_bed_out = self._outlet_average(
            step_name,
            T_bed[z_out, :, :],
            A_rings=A_rings,
            rho_z=rho[z_out, :, :],
            u_out_z=u_out_z,
        )

        exact_fractions = outlet_mole_fractions(sol)
        if exact_fractions is not None and step_name not in self.CLOSED_OUTLET_STEPS:
            fraction_out = exact_fractions
            u_out_z = np.asarray(sol["outlet_face"]["velocity_m_s"])

        pressure_out = self._radial_average(p_tot[z_out, :, :], A_rings)
        u_out = self._radial_average(u_out_z, A_rings)
        if step_name in self.CLOSED_OUTLET_STEPS:
            u_out = np.zeros_like(u_out)

        outlet = {
            "time": time,
            "T_bed": T_bed_out,
            "T_wall": np.asarray(T_wall[z_out, :], dtype=float),
            "pressure": pressure_out,
            "p_tot": pressure_out,
            "u_out": u_out,
        }
        for species in saved.layout.mixture.species:
            outlet[f"y_{species}"] = fraction_out[species]
            outlet[f"ppm_{species}"] = 1e6 * fraction_out[species]
        for species in saved.layout.adsorbates:
            outlet[f"q_{species}"] = self._radial_average(
                saved.field(f"q_{species}")[z_out, :, :], A_rings
            )
        return outlet

    def get_step_profile(self, step_name, field):
        field_key = self.FIELD_ALIASES.get(field, field)
        return self.outlet_profiles.get(step_name, {}).get(field_key, None)

    def available_steps(self):
        return list(self.outlet_profiles.keys())

    def available_fields(self, step_name):
        return list(self.outlet_profiles.get(step_name, {}).keys())

    def _get_time_scaling(self, steps):
        total_time = 0.0
        for step in steps:
            t = self.get_step_profile(step, "time")
            if t is not None and len(t) > 1:
                total_time += max(float(t[-1] - t[0]), 0.0)

        if total_time > 3600:
            return 1.0 / 3600.0, "Time [h]"
        if total_time > 300:
            return 1.0 / 60.0, "Time [min]"
        return 1.0, "Time [s]"

    _normalize_cool_jump_mode = staticmethod(
        _profile_utils.normalize_cool_jump_mode
    )
    _step_duration_s = staticmethod(_profile_utils.step_duration_s)

    def _cool_jump_search_limit(self, time):
        return _profile_utils.cool_jump_search_limit(
            time,
            max_fraction=self.COOL_JUMP_MAX_FRACTION,
            max_time_s=self.COOL_JUMP_MAX_TIME_S,
        )

    def _cool_jump_match_tolerance(self, field, prev_values, curr_values):
        return _profile_utils.cool_jump_match_tolerance(
            field,
            prev_values,
            curr_values,
            absolute_floors=self.COOL_JUMP_ABS_FLOORS,
            relative_tolerance=self.COOL_JUMP_TOL_REL,
        )

    _smoothstep = staticmethod(_profile_utils.smoothstep)

    def _get_cool_jump_context(self, field, prev_step, step):
        field_key = self._normalize_field_name(field)
        if step != "cool" or prev_step is None or field_key not in self.COOL_JUMP_FIELDS:
            return None

        prev_values = self.get_step_profile(prev_step, field_key)
        curr_values = self.get_step_profile(step, field_key)
        curr_time = self.get_step_profile(step, "time")
        if prev_values is None or curr_values is None or curr_time is None:
            return None

        prev_values = np.asarray(prev_values, dtype=float)
        curr_values = np.asarray(curr_values, dtype=float)
        curr_time = np.asarray(curr_time, dtype=float)
        if prev_values.size == 0 or curr_values.size <= 1 or curr_time.size != curr_values.size:
            return None

        upper = self._cool_jump_search_limit(curr_time)
        if upper <= 0:
            return None

        prev_end = float(prev_values[-1])
        diff_window = np.abs(curr_values[:upper + 1] - prev_end)
        idx_best = int(np.argmin(diff_window))

        diff_0 = float(diff_window[0])
        diff_best = float(diff_window[idx_best])
        if diff_best <= 0.0:
            improvement = np.inf
        else:
            improvement = diff_0 / diff_best

        tol_abs = self._cool_jump_match_tolerance(field_key, prev_values, curr_values[:upper + 1])
        dt = float(curr_time[1] - curr_time[0]) if curr_time.size > 1 else 0.0
        bridge_horizon_s = max(float(curr_time[upper]), 3.0 * max(dt, 0.0))

        return {
            "field_key": field_key,
            "prev_end": prev_end,
            "curr_values": curr_values,
            "curr_time": curr_time,
            "upper": upper,
            "idx_best": idx_best,
            "diff_0": diff_0,
            "diff_best": diff_best,
            "improvement": improvement,
            "tol_abs": tol_abs,
            "dt": dt,
            "bridge_horizon_s": bridge_horizon_s,
        }

    def _get_cool_jump_adjustment(self, field, prev_step, step):
        context = self._get_cool_jump_context(field, prev_step, step)
        if context is None:
            return None
        if context["idx_best"] == 0:
            return None
        if context["diff_0"] <= context["tol_abs"]:
            return None
        if context["diff_best"] > context["tol_abs"]:
            return None
        if context["improvement"] < self.COOL_JUMP_MIN_IMPROVEMENT:
            return None

        curr_values = context["curr_values"]
        curr_time = context["curr_time"]
        idx_best = context["idx_best"]

        return {
            "idx_match": idx_best,
            "time_match_s": float(curr_time[idx_best]),
            "prev_end": context["prev_end"],
            "curr_match": float(curr_values[idx_best]),
        }

    def _build_concatenated_profiles(self, fields, steps=None):
        normalized_fields = [self._normalize_field_name(field) for field in fields]
        steps = self._get_cycle_step_order() if steps is None else list(steps)
        steps = [step for step in steps if step in self.outlet_profiles]
        if not steps:
            raise RuntimeError("[CyclicProfiles] No steps found to build concatenated profiles.")

        t_factor, t_label = self._get_time_scaling(steps)
        time_list = []
        field_arrays = {field: [] for field in normalized_fields}
        step_segments = {}
        idx_cursor = 0
        t_shift = 0.0

        for step in steps:
            time = self.get_step_profile(step, "time")
            if time is None or len(time) == 0:
                continue

            time = np.asarray(time, dtype=float)
            time_shifted = (time + t_shift) * t_factor
            time_list.append(time_shifted)

            idx_start = idx_cursor
            idx_end = idx_cursor + len(time)
            step_segments[step] = (idx_start, idx_end)
            idx_cursor = idx_end

            for field in normalized_fields:
                arr = self.get_step_profile(step, field)
                if arr is None:
                    raise KeyError(f"Field '{field}' not found in step '{step}'.")
                field_arrays[field].append(np.asarray(arr, dtype=float))

            if len(time) > 1:
                t_shift += float(time[-1] - time[0])

        time_concat = np.concatenate(time_list)
        for field in normalized_fields:
            field_arrays[field] = np.concatenate(field_arrays[field])

        return {
            "time": time_concat,
            "time_label": t_label,
            "fields": field_arrays,
            "steps_used": steps,
            "t_factor": t_factor,
            "step_segments": step_segments,
        }

    def _transform_field_for_plot(self, field, values):
        spec = self.field_specs[self._normalize_field_name(field)]
        return spec.scale * np.asarray(values, dtype=float) + spec.offset

    def _step_display_label(self, step):
        return self.STEP_LABELS.get(step, step.replace("_", " ").title())

    def plot_field(self, field, save=True, show=False, fmt="pdf", steps=None, cool_jump_mode=None):
        field_key = self._normalize_field_name(field)
        spec = self.field_specs[field_key]
        cool_jump_mode = self._normalize_cool_jump_mode(cool_jump_mode)
        ordered_steps = self._get_cycle_step_order() if steps is None else list(steps)
        ordered_steps = [step for step in ordered_steps if step in self.outlet_profiles]
        if not ordered_steps:
            raise RuntimeError("[CyclicProfiles] No steps found to plot.")

        t_factor, t_label = self._get_time_scaling(ordered_steps)

        self._set_global_style()

        fig, ax = plt.subplots(figsize=(12, 5))
        lines = []
        labels = []
        locator = mticker.MaxNLocator(nbins=8)
        boundary_times = []
        t_shift = 0.0
        positive_values = []

        for idx, step in enumerate(ordered_steps):
            prev_step = ordered_steps[idx - 1] if idx > 0 else None
            time_raw = np.asarray(self.get_step_profile(step, "time"), dtype=float)
            values_raw = np.asarray(self.get_step_profile(step, field_key), dtype=float)
            values_plot = self._transform_field_for_plot(field_key, values_raw)
            color = self.STEP_COLORS.get(step, "tab:gray")
            duration_s = self._step_duration_s(time_raw)
            jump_context = self._get_cool_jump_context(field_key, prev_step, step) if cool_jump_mode else None
            adjustment = self._get_cool_jump_adjustment(field_key, prev_step, step) if cool_jump_mode == "trim" else None

            if cool_jump_mode == "trim" and adjustment is not None:
                idx_start = adjustment["idx_match"]
                time_display = time_raw[idx_start:] - time_raw[idx_start]
                values_display = values_plot[idx_start:]
                duration_s = self._step_duration_s(time_display)
            elif cool_jump_mode == "bridge" and jump_context is not None and jump_context["diff_0"] > jump_context["tol_abs"]:
                time_display = time_raw
                values_display = values_plot.copy()
                prev_end_plot = float(self._transform_field_for_plot(field_key, jump_context["prev_end"]))
                t_bridge = max(jump_context["bridge_horizon_s"], 1e-32)
                s = self._smoothstep(time_raw / t_bridge)
                bridge_mask = time_raw < t_bridge
                values_display[bridge_mask] = (
                    (1.0 - s[bridge_mask]) * prev_end_plot
                    + s[bridge_mask] * values_plot[bridge_mask]
                )
            else:
                time_display = time_raw
                values_display = values_plot

            if field_key in {"ppm_O2", "ppm_N2"}:
                positive_values.append(values_display[np.isfinite(values_display) & (values_display > 0.0)])
            t_step = (time_display + t_shift) * t_factor

            plot_values = values_display
            if field_key in {"ppm_O2", "ppm_N2"}:
                plot_values = np.where(values_display > 0.0, values_display, np.nan)

            line, = ax.plot(t_step, plot_values, color=color, label=self._step_display_label(step))
            lines.append(line)
            labels.append(self._step_display_label(step))
            t_shift += duration_s
            boundary_times.append(t_shift * t_factor)

        for x_boundary in boundary_times[:-1]:
            ax.axvline(x_boundary, color="0.85", linestyle="--", linewidth=1.0, zorder=0)

        ax.set_title(spec.title)
        ax.set_ylabel(spec.ylabel)
        ax.set_xlabel(t_label)
        ax.grid(True)
        ax.xaxis.set_major_locator(locator)
        if field_key in {"ppm_O2", "ppm_N2"}:
            positive_arrays = [np.ravel(vals) for vals in positive_values if np.size(vals) > 0]
            positive_concat = np.concatenate(positive_arrays) if positive_arrays else np.array([self.PPM_LOG_FLOOR])
            y_min_data = float(np.min(positive_concat))
            y_min = max(self.PPM_LOG_FLOOR, 0.2 * y_min_data)
            y_max = max(float(np.max(positive_concat)), self.O2_PPM_LIMIT if field_key == "ppm_O2" else self.PPM_LOG_FLOOR)
            ax.set_yscale("log")
            ax.set_ylim(y_min, max(1.25 * y_max, y_min * 10.0))
            if field_key == "ppm_O2":
                spec_line = ax.axhline(
                    self.O2_PPM_LIMIT,
                    color="black",
                    linestyle="--",
                    linewidth=1.3,
                    label=r"O$_2$ spec = 40 ppmmol",
                )
                lines.append(spec_line)
                labels.append(r"O$_2$ spec = 40 ppmmol")
        elif field_key in {"y_CO2", "y_H2O", "q_CO2", "q_H2O", "pressure"}:
            ax.yaxis.set_major_formatter(self._sig3_formatter())

        fig.tight_layout()
        if lines:
            fig.legend(
                lines,
                labels,
                loc="lower center", bbox_to_anchor=(0.5, -0.0325),
                ncol=min(len(labels), 4),
                frameon=False,
            )
            fig.subplots_adjust(bottom=0.22)

        out_path = None
        if save:
            out_path = os.path.join(self.plot_folder, f"{spec.filename}.{fmt}")
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[Saved] {out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return out_path

    def plot_fields(self, fields=None, save=True, show=False, fmt="pdf", steps=None, cool_jump_mode=None):
        selected_fields = self.default_plot_fields if fields is None else tuple(fields)
        saved = []
        for field in selected_fields:
            saved.append(
                self.plot_field(
                    field,
                    save=save,
                    show=show,
                    fmt=fmt,
                    steps=steps,
                    cool_jump_mode=cool_jump_mode,
                )
            )
        return saved

    def plot_all_css_fields(self, save=True, show=False, fmt="pdf", steps=None, cool_jump_mode=None):
        return self.plot_fields(
            fields=self.default_plot_fields,
            save=save,
            show=show,
            fmt=fmt,
            steps=steps,
            cool_jump_mode=cool_jump_mode,
        )

    def plot_cycle_species_outlet(self, save=True, show=False, fmt="pdf", cool_jump_mode=None):
        return self.plot_fields(
            fields=tuple(f"y_{species}" for species in self.gas_species),
            save=save,
            show=show,
            fmt=fmt,
            cool_jump_mode=cool_jump_mode,
        )

    def plot_cycle_impurity_outlet(self, save=True, show=False, fmt="pdf", cool_jump_mode=None):
        if "O2" not in self.gas_species:
            raise ValueError(
                "DAC impurity plotting requires O2 in the saved mixture; "
                f"declared mixture={self.gas_species}."
            )
        return self.plot_fields(
            fields=("ppm_O2",),
            save=save,
            show=show,
            fmt=fmt,
            cool_jump_mode=cool_jump_mode,
        )

    def plot_cycle_solid_species_profiles(self, save=True, show=False, fmt="pdf", cool_jump_mode=None):
        return self.plot_fields(
            fields=tuple(f"q_{species}" for species in self.adsorbates),
            save=save,
            show=show,
            fmt=fmt,
            cool_jump_mode=cool_jump_mode,
        )

    def plot_cycle_temp_pressure_outlet(self, save=True, show=False, fmt="pdf", cool_jump_mode=None):
        return self.plot_fields(
            fields=("T_bed", "T_wall", "pressure"),
            save=save,
            show=show,
            fmt=fmt,
            cool_jump_mode=cool_jump_mode,
        )

    def save_css_signature(self, fname=None, fields=None, steps=None):
        save_root = os.path.join("comparison_css", "css_signatures")
        os.makedirs(save_root, exist_ok=True)

        selected_fields = tuple(fields) if fields is not None else self.default_plot_fields + ("u_out",)
        normalized_fields = tuple(self._normalize_field_name(field) for field in selected_fields)

        if fname is None:
            mode_tag = "1D" if self.is_1d else "2D"
            fname = os.path.join(save_root, f"css_signature_cycle_{self.css_cycle_idx}_{mode_tag}.npz")
        elif not os.path.isabs(fname):
            fname = os.path.join(save_root, fname)

        data = self._build_concatenated_profiles(fields=normalized_fields, steps=steps)

        metadata = {
            "css_cycle_idx": self.css_cycle_idx,
            "desorption_step": self.desorption_step,
            "base_folder": self.base_folder,
            "steps_used": data["steps_used"],
            "t_factor": data["t_factor"],
            "time_label": data["time_label"],
            "step_segments": data["step_segments"],
            "step_colors": self.STEP_COLORS,
            "closed_outlet_steps": sorted(self.CLOSED_OUTLET_STEPS),
            "fields_saved": normalized_fields,
        }

        save_dict = {
            "metadata": metadata,
            "fields": data["fields"],
            "time": data["time"],
            "time_label": data["time_label"],
        }

        np.savez(fname, **save_dict)
        print(f"[Saved CSS signature] {fname}")
        return fname
