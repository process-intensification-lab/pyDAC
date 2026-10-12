"""
pyDAC - Step-profile visualization.

This module defines `ProfilesPlotter`, which renders internal 1D/2D fields and
outlet histories using the data extracted by `StepPostProcessor`.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

import os
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pydac.post_processing.profiles.cyclic_plotting import CyclicSteadyStateProfiles
from pydac.utils.plot_helpers import SNAPSHOT_TIME_OVERRIDES, T_END_OVERRIDES
import seaborn as sns

class ProfilesPlotter:
    """
    Handles all plotting related to internal/external profiles,
    using a StepPostProcessor instance as data backend.
    """

    def __init__(self, post_processing_instance):
        self.pp = post_processing_instance   
        step = self.pb.step

        if step not in ("blow", "heat", "w_purge", "w_des"):
            self.y_ref_by_species = dict(self.pb.y_feed_by_species)
            self.T_ref     = self.pb.T_feed
            self.p_ref     = self.pb.p_out
            self.u_ref     = self.pb.u_inlet

            # Labels used in dashed reference lines
            self.y_ref_label = "Feed"
            self.T_ref_label = "Feed"
            self.p_ref_label = "Outlet"
            self.u_ref_label = "Feed"

        else:
            self.y_ref_by_species = dict(self.pb.y_initial_by_species)
            self.T_ref     = self.pb.T_jacket
            self.p_ref     = self.pb.p_out        
            self.u_ref     = self.pb.u_inlet

            self.y_ref_label = "Initial"
            self.T_ref_label = "Jacket"
            self.p_ref_label = "Outlet setpoint"
            self.u_ref_label = "Nominal"

        for species, value in self.y_ref_by_species.items():
            setattr(self, f"y_{species}_ref", float(value))
            
    # --- optional convenience properties ---
    @property
    def pb(self):
        return self.pp.pb

    @property
    def solver(self):
        return self.pp.solver

    @property
    def internal_1D_profiles_folder(self):
        return self.pp.internal_1D_profiles_folder
    
    @property
    def internal_2D_profiles_folder(self):
        return self.pp.internal_2D_profiles_folder
    
    @property
    def outlet_profiles_folder(self):
        return self.pp.outlet_profiles_folder
    
    @property
    def step_long(self):
        return self.pp.step_long

    @property
    def step(self):
        return self.pp.step

    @property
    def is_1d(self):
        return bool(getattr(self.pp, "is_1d", False))

    # ---------- GLOBAL STYLE ----------
    @staticmethod
    def _set_global_style():
        plt.rcParams.update({
            "text.usetex": False,             # keep TeX off unless you really need math typesetting
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans"],   # or "Arial" if you prefer

            # Axes and labels
            "axes.labelsize": 15,
            "axes.labelweight": "regular",
            "axes.titlesize": 17,
            "axes.linewidth": 1.2,

            # Ticks
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,

            # Legend
            "legend.fontsize": 14,

            # Cleaner plots
            "figure.dpi": 120,       # you also export at 300 DPI anyway
            "savefig.dpi": 300,
        })

    @staticmethod
    def _sig3_formatter():
        """Formatter with 3 significant digits for compact axis labels."""
        return mticker.FuncFormatter(lambda x, pos: f"{x:.3g}")

    def _radial_average(self, field_z):
        """
        Area-weighted radial average at a given axial plane.

        Parameters
        ----------
        field_z : np.ndarray
            Field slice with shape `(Nr, Nt)`.
        """
        weights = self.pp.A_rings[:, None]
        return np.sum(field_z * weights, axis=0) / np.sum(self.pp.A_rings)

    def _plot_r_centers(self):
        """Return dimensional radial centers used for visualizations."""
        if self.is_1d:
            return self.pp.r_c_plot * self.pb.R_b
        return self.pp.r_c * self.pb.R_b

    def _plot_r_faces(self):
        """Return dimensional radial faces used for visualizations."""
        if self.is_1d:
            return self.pp.r_f_plot * self.pb.R_b
        return self.pp.r_f * self.pb.R_b

    def _expand_field_for_2d_plot(self, field):
        """
        Expand a 1D-in-radius field for 2D visualization.

        For `Nr > 1` the input is returned unchanged. For `Nr == 1`, the single
        radial value is repeated across a pseudo-radial plotting grid.
        """
        arr = np.asarray(field)
        if not self.is_1d:
            return arr

        if arr.ndim < 2 or arr.shape[1] != 1:
            return arr
        return np.repeat(arr, self.pp.Nr_plot_2d, axis=1)

    def _outlet_reduce(self, field_z, mode="radial", **kwargs):
        """
        Reduce an outlet radial profile to a single time trace.

        In 1D mode (`Nr == 1`) the single stored radial value is returned
        directly. In 2D mode, either radial or mixed-cup averaging is applied.
        """
        if self.is_1d:
            return np.asarray(field_z[0, :])

        if mode == "mixed_cup":
            return self._mixed_cup_average(field_z=field_z, **kwargs)
        if mode == "radial":
            return self._radial_average(field_z)
        raise ValueError(f"Unknown outlet reduction mode: {mode}")

    def _plot_center_edge(self, ax, x, center, edge, label=None):
        """
        Plot center/edge axial traces, or a single trace in 1D mode.
        """
        line, = ax.plot(x, center, linestyle='-', label=label)
        color = line.get_color()
        if not self.is_1d:
            ax.plot(x, edge, linestyle='--', color=color, label=None)
        return color

    def _mixed_cup_average(
        self,
        field_z,
        u_ax_z,
        rho_z=None,
        c_tot_z=None,
        y_CO2_z=None,
        y_H2O_z=None,
        y_O2_z=None,
        y_N2_z=None,
    ):
        """
        Mass-flux weighted average at a given axial plane.

        All inputs are expected to have shape `(Nr, Nt)`.
        """
        if rho_z is None:
            required = (y_CO2_z, y_H2O_z, y_O2_z, y_N2_z, c_tot_z)
            if any(value is None for value in required):
                raise ValueError("Mixed-cup averaging requires `rho_z`.")
            rho_z = c_tot_z * (
                y_CO2_z * self.pb.MW_CO2
                + y_H2O_z * self.pb.MW_H2O
                + y_O2_z * self.pb.MW_O2
                + y_N2_z * self.pb.MW_N2
            )
        mass_flux = np.clip(rho_z * u_ax_z, 0, None)

        weights = mass_flux * self.pp.A_rings[:, None]
        den = np.sum(weights, axis=0)
        num = np.sum(field_z * weights, axis=0)

        eps = 1e-32
        out = np.empty_like(den)
        mask = np.abs(den) > eps
        out[mask] = num[mask] / den[mask]

        if not np.all(mask):
            print("[Warning] Mixed-cup average had zero denominator, falling back to radial average for those.")
            out[~mask] = self._radial_average(field_z)[~mask]
        return out

    def plot_all_profiles(self, save=True, show=False, fmt="pdf"):
        """
        Run the full plotting suite for one simulation step.

        This is the main orchestration entrypoint used by the cycle runner.
        It keeps the 1D/2D branching logic inside the plotter instead of
        scattering it across the caller.
        """
        if self.pp._is_current_dac_layout:
            self.plot_internal_profiles_2D(save=save, show=show, fmt=fmt)
            self.plot_temperature_profiles_2D_with_wall(save=save, show=show, fmt=fmt)
            self.plot_internal_profiles_1D(save=save, show=show, fmt=fmt)
            self.plot_outlet_profiles(save=save, show=show, fmt=fmt)
            return

        fields = (
            *(f"y_{species}" for species in self.pp.gas_species),
            *(f"q_{species}" for species in self.pp.adsorbates),
            "c_tot",
            "T_bed",
            "p_tot",
        )
        self.plot_internal_profiles_2D(
            fields=fields, save=save, show=show, fmt=fmt
        )
        self.plot_temperature_profiles_2D_with_wall(save=save, show=show, fmt=fmt)
        self.plot_generic_internal_profiles_1D(save=save, show=show, fmt=fmt)
        self.plot_generic_outlet_profiles(save=save, show=show, fmt=fmt)

    def plot_generic_internal_profiles_1D(
        self, n_times=10, save=True, show=False, fmt="pdf"
    ):
        """Plot declared gas species and adsorbate loadings without DAC branches."""
        self._set_global_style()
        pp = self.pp
        z_c = pp.z_c * self.pb.L_b

        for species in pp.gas_species:
            has_loading = species in pp.q_by_species
            fig, axes = plt.subplots(
                1, 2 if has_loading else 1, figsize=(14 if has_loading else 7, 5)
            )
            axes = np.atleast_1d(axes)
            y = pp.y_by_species[species]
            for index, label in self._profile_schedule(f"y_{species}", n_times):
                self._plot_center_edge(
                    axes[0], z_c, y[:, 0, index], y[:, -1, index], label=label
                )
            if has_loading:
                q = pp.q_by_species[species]
                for index, label in self._profile_schedule(f"q_{species}", n_times):
                    self._plot_center_edge(
                        axes[1], z_c, q[:, 0, index], q[:, -1, index], label=label
                    )

            axes[0].set_title(f"Gas-phase {species}")
            axes[0].set_ylabel("Molar fraction [-]")
            if has_loading:
                axes[1].set_title(f"Solid-phase {species}")
                axes[1].set_ylabel("Loading [mol/kg]")
            for axis in axes:
                axis.set_xlabel("z [m]")
                axis.grid(True)
                axis.legend(title="Time")
            fig.tight_layout()
            if save:
                path = os.path.join(
                    self.internal_1D_profiles_folder,
                    f"{self.step}_{species}_alltimes.{fmt}",
                )
                fig.savefig(path, dpi=300, bbox_inches="tight")
                print(f"[Saved] {path}")
            if not show:
                plt.close(fig)
        if show:
            plt.show()

    def plot_generic_outlet_profiles(self, save=True, show=False, fmt="pdf"):
        """Plot declared outlet compositions plus temperature and hydraulics.

        The stream traces are mixed-cup values, NaN while no stream leaves the
        column (a closed step such as ``cool``). The bed temperature in the
        outlet cell is always drawn, and so is the gas composition there while
        the stream is missing, so a closed step does not give empty panels.
        """
        self._set_global_style()
        pp = self.pp
        cups = pp.get_mixed_cups()
        time = pp.time
        figures = []
        no_stream = ~np.isfinite(cups["T_out"])

        fig_species, axes = plt.subplots(
            len(pp.gas_species), 1, figsize=(10, 3.5 * len(pp.gas_species)), sharex=True
        )
        axes = np.atleast_1d(axes)
        for axis, species in zip(axes, pp.gas_species, strict=True):
            axis.plot(time, cups[f"y_{species}"], label=f"Model {species}")
            if no_stream.any():
                y_cell = self._outlet_reduce(
                    pp.y_by_species[species][-1], mode="radial"
                )
                axis.plot(
                    time,
                    np.where(no_stream, y_cell, np.nan),
                    linestyle=":",
                    label="Gas in the outlet cell (no outgoing stream)",
                )
            axis.axhline(
                self.y_ref_by_species[species],
                linestyle="--",
                label=self.y_ref_label,
            )
            axis.set_ylabel("Mole fraction [-]")
            axis.set_title(f"Outlet gas-phase {species}")
            axis.grid(True)
            axis.legend()
        axes[-1].set_xlabel("Time [s]")
        fig_species.tight_layout()
        figures.append((fig_species, f"{self.step}_outlet_species.{fmt}"))

        fig_state, axes_state = plt.subplots(1, 3, figsize=(18, 5))
        T_bed_out = self._outlet_reduce(pp.T_bed[-1], mode="radial")
        axes_state[0].plot(time, T_bed_out - 273.15, label="Bed in the outlet cell")
        axes_state[0].plot(
            time, cups["T_out"] - 273.15, linestyle="--", label="Outlet stream"
        )
        axes_state[0].set_ylabel("Temperature [°C]")
        axes_state[0].set_title("Outlet temperature")
        axes_state[0].legend()
        p_out = self._outlet_reduce(pp.p_tot[-1], mode="radial")
        axes_state[1].plot(time, p_out / 100.0)
        axes_state[1].set_ylabel("Pressure [mbar]")
        axes_state[1].set_title("Outlet pressure")
        u_out = self._outlet_reduce(pp.u_ax[-1], mode="radial")
        axes_state[2].plot(time, u_out)
        axes_state[2].set_ylabel("Velocity [m/s]")
        axes_state[2].set_title("Outlet axial velocity")
        for axis in axes_state:
            axis.set_xlabel("Time [s]")
            axis.grid(True)
        fig_state.tight_layout()
        figures.append((fig_state, f"{self.step}_outlet_state.{fmt}"))

        if save:
            for figure, filename in figures:
                path = os.path.join(self.outlet_profiles_folder, filename)
                figure.savefig(path, dpi=300, bbox_inches="tight")
                print(f"[Saved] {path}")
        if show:
            plt.show()
        else:
            for figure, _ in figures:
                plt.close(figure)

    # ============================================================
    # 1D INTERNAL PROFILES
    # ============================================================
    def plot_internal_profiles_1D(
        self,
        n_times=10,
        species=True,
        temperature_pressure=True,
        wall_temperature=True,
        velocities=True,
        save=True,
        show=False,
        fmt='pdf'
    ):
        self._set_global_style()

        pp = self.pp

        z_c = pp.z_c * self.pb.L_b
        z_f = pp.z_f * self.pb.L_b
        r_c = pp.r_c * self.pb.R_b
        is_1d = self.is_1d

        # Each panel is drawn at the times of its field (see `_profile_schedule`).
        def schedule(field_name):
            return self._profile_schedule(field_name, n_times)

        base_folder = self.internal_1D_profiles_folder
        os.makedirs(base_folder, exist_ok=True)

        if species:
            fig_co2, axs_co2 = plt.subplots(1, 2, figsize=(14, 5))
            for idx, label in schedule("y_CO2"):
                y_center = pp.y_CO2[:, 0, idx]
                y_edge = pp.y_CO2[:, -1, idx]

                self._plot_center_edge(axs_co2[0], z_c, y_center, y_edge, label=label)

            for idx, label in schedule("q_CO2"):
                q_center = pp.q_CO2[:, 0, idx]
                q_edge = pp.q_CO2[:, -1, idx]

                self._plot_center_edge(axs_co2[1], z_c, q_center, q_edge, label=label)

            if is_1d:
                axs_co2[0].set_title(r'Gas-phase CO$_2$')
                axs_co2[1].set_title(r'Solid-phase CO$_2$')
            else:
                axs_co2[0].set_title(r'Gas-phase CO$_2$ (solid: r=0, dashed: r=R)')
                axs_co2[1].set_title(r'Solid-phase CO$_2$ (solid: r=0, dashed: r=R)')
            axs_co2[0].set_ylabel('Molar fraction [-]')
            axs_co2[1].set_ylabel('Loading [mol/kg]')
            axs_co2[0].set_xlabel('z [m]')
            axs_co2[1].set_xlabel('z [m]')
            axs_co2[0].grid(True)
            axs_co2[1].grid(True)
            axs_co2[0].legend(title='Time')
            axs_co2[1].legend(title='Time')
            fig_co2.tight_layout()

            if save:
                fname_co2 = os.path.join(
                    base_folder, f"{self.step}_CO2_alltimes.{fmt}"
                )
                fig_co2.savefig(fname_co2, dpi=300, bbox_inches="tight")
                print(f"[Saved] {fname_co2}")

            fig_h2o, axs_h2o = plt.subplots(1, 2, figsize=(14, 5))
            for idx, label in schedule("y_H2O"):
                y_center = pp.y_H2O[:, 0, idx]
                y_edge = pp.y_H2O[:, -1, idx]

                self._plot_center_edge(axs_h2o[0], z_c, y_center, y_edge, label=label)

            for idx, label in schedule("q_H2O"):
                q_center = pp.q_H2O[:, 0, idx]
                q_edge = pp.q_H2O[:, -1, idx]

                self._plot_center_edge(axs_h2o[1], z_c, q_center, q_edge, label=label)

            if is_1d:
                axs_h2o[0].set_title(r'Gas-phase H$_2$O')
                axs_h2o[1].set_title(r'Solid-phase H$_2$O')
            else:
                axs_h2o[0].set_title(r'Gas-phase H$_2$O (solid: r=0, dashed: r=R)')
                axs_h2o[1].set_title(r'Solid-phase H$_2$O (solid: r=0, dashed: r=R)')
            axs_h2o[0].set_ylabel('Molar fraction [-]')
            axs_h2o[1].set_ylabel('Loading [mol/kg]')
            axs_h2o[0].set_xlabel('z [m]')
            axs_h2o[1].set_xlabel('z [m]')
            axs_h2o[0].grid(True)
            axs_h2o[1].grid(True)
            axs_h2o[0].legend(title='Time')
            axs_h2o[1].legend(title='Time')
            fig_h2o.tight_layout()

            if save:
                fname_h2o = os.path.join(
                    base_folder, f"{self.step}_H2O_alltimes.{fmt}"
                )
                fig_h2o.savefig(fname_h2o, dpi=300, bbox_inches="tight")
                print(f"[Saved] {fname_h2o}")

            fig_o2n2, axs_o2n2 = plt.subplots(1, 2, figsize=(14, 5))
            for idx, label in schedule("y_O2"):
                y_center = pp.y_O2[:, 0, idx]
                y_edge = pp.y_O2[:, -1, idx]
                self._plot_center_edge(axs_o2n2[0], z_c, y_center, y_edge, label=label)

            for idx, label in schedule("y_N2"):
                y_center = pp.y_N2[:, 0, idx]
                y_edge = pp.y_N2[:, -1, idx]
                self._plot_center_edge(axs_o2n2[1], z_c, y_center, y_edge, label=label)

            if is_1d:
                axs_o2n2[0].set_title(r'Gas-phase O$_2$')
                axs_o2n2[1].set_title(r'Gas-phase N$_2$')
            else:
                axs_o2n2[0].set_title(r'Gas-phase O$_2$ (solid: r=0, dashed: r=R)')
                axs_o2n2[1].set_title(r'Gas-phase N$_2$ (solid: r=0, dashed: r=R)')
            axs_o2n2[0].set_ylabel('Molar fraction [-]')
            axs_o2n2[1].set_ylabel('Molar fraction [-]')
            axs_o2n2[0].set_xlabel('z [m]')
            axs_o2n2[1].set_xlabel('z [m]')
            axs_o2n2[0].grid(True)
            axs_o2n2[1].grid(True)
            axs_o2n2[0].legend(title='Time')
            axs_o2n2[1].legend(title='Time')
            fig_o2n2.tight_layout()

            if save:
                fname_o2n2 = os.path.join(
                    base_folder, f"{self.step}_O2_N2_alltimes.{fmt}"
                )
                fig_o2n2.savefig(fname_o2n2, dpi=300, bbox_inches="tight")
                print(f"[Saved] {fname_o2n2}")

            if not show:
                plt.close(fig_co2)
                plt.close(fig_h2o)
                plt.close(fig_o2n2)

        if wall_temperature:
            fig_tz, (axTb_z, axTw_z) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            Tmin = np.inf
            Tmax = -np.inf

            # The wall has no field of its own: it follows T_bed.
            for idx, label in schedule("T_bed"):
                T_center = pp.T_bed[:, 0, idx] - 273.15
                T_edge = pp.T_bed[:, -1, idx] - 273.15
                T_wall = pp.T_wall[:, idx] - 273.15

                Tmin = min(Tmin, np.min(T_center), np.min(T_edge), np.min(T_wall))
                Tmax = max(Tmax, np.max(T_center), np.max(T_edge), np.max(T_wall))

                color_T = self._plot_center_edge(axTb_z, z_c, T_center, T_edge, label=label)
                axTw_z.plot(z_c, T_wall, linestyle='-', color=color_T, label=label)

            if np.isfinite(Tmin) and np.isfinite(Tmax):
                spread = Tmax - Tmin
                pad = 0.03 * spread if spread > 0 else 1.0
                axTb_z.set_ylim(Tmin - pad, Tmax + pad)
                axTw_z.set_ylim(Tmin - pad, Tmax + pad)

            if is_1d:
                axTb_z.set_title(r'Bed Temperature $T_{bed}$')
            else:
                axTb_z.set_title(r'Bed Temperature $T_{bed}$ (solid: r=0, dashed: r=R)')
            axTw_z.set_title(r'Wall Temperature $T_w$ (z-dependent)')
            axTb_z.set_ylabel('Temperature [°C]')
            axTw_z.set_ylabel('Temperature [°C]')
            axTb_z.set_xlabel('z [m]')
            axTw_z.set_xlabel('z [m]')
            axTb_z.grid(True)
            axTw_z.grid(True)
            axTb_z.legend(title='Time')
            axTw_z.legend(title='Time')
            fig_tz.tight_layout()

            if save:
                fname_tz = os.path.join(base_folder, f"{self.step}_Tbed_Twall_axial_alltimes.{fmt}")
                fig_tz.savefig(fname_tz, dpi=300, bbox_inches="tight")
                print(f"[Saved] {fname_tz}")

            if not show:
                plt.close(fig_tz)

            if not is_1d:
                fig_tr, (axTb_r, axTw_r) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
                z_mid_idx = pp.Nz // 2
                z_end_idx = pp.Nz - 1
                Tmin = np.inf
                Tmax = -np.inf

                for idx, label in schedule("T_bed"):
                    T_bed_mid = pp.T_bed[z_mid_idx, :, idx] - 273.15
                    T_bed_end = pp.T_bed[z_end_idx, :, idx] - 273.15
                    T_wall_mid = pp.T_wall[z_mid_idx, idx] - 273.15
                    T_wall_end = pp.T_wall[z_end_idx, idx] - 273.15

                    Tmin = min(Tmin, np.min(T_bed_mid), np.min(T_bed_end), T_wall_mid, T_wall_end)
                    Tmax = max(Tmax, np.max(T_bed_mid), np.max(T_bed_end), T_wall_mid, T_wall_end)

                    line_Tr, = axTb_r.plot(r_c, T_bed_mid, linestyle='-', label=label)
                    color_Tr = line_Tr.get_color()
                    axTb_r.plot(r_c, T_bed_end, linestyle='--', color=color_Tr, label=None)
                    axTw_r.plot(r_c, np.full_like(r_c, T_wall_mid), linestyle='-', color=color_Tr, label=label)
                    axTw_r.plot(r_c, np.full_like(r_c, T_wall_end), linestyle='--', color=color_Tr, label=None)

                if np.isfinite(Tmin) and np.isfinite(Tmax):
                    spread = Tmax - Tmin
                    pad = 0.03 * spread if spread > 0 else 1.0
                    axTb_r.set_ylim(Tmin - pad, Tmax + pad)
                    axTw_r.set_ylim(Tmin - pad, Tmax + pad)

                axTb_r.set_title(r'Bed Temperature vs $r$ (solid: z=L/2, dashed: z=L)')
                axTw_r.set_title(r'Wall Temperature vs $r$ (solid: z=L/2, dashed: z=L)')
                axTb_r.set_ylabel('Temperature [°C]')
                axTw_r.set_ylabel('Temperature [°C]')
                axTb_r.set_xlabel('r [m]')
                axTw_r.set_xlabel('r [m]')
                axTb_r.grid(True)
                axTw_r.grid(True)
                axTb_r.legend(title='Time')
                axTw_r.legend(title='Time')
                fig_tr.tight_layout()

                if save:
                    fname_tr = os.path.join(base_folder, f"{self.step}_Tbed_Twall_radial_alltimes.{fmt}")
                    fig_tr.savefig(fname_tr, dpi=300, bbox_inches="tight")
                    print(f"[Saved] {fname_tr}")

                if not show:
                    plt.close(fig_tr)

        if temperature_pressure:
            fig_p, (axPz, axCz) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

            for idx, label in schedule("p_tot"):
                p_center = pp.p_tot[:, 0, idx] / 100.0
                p_edge = pp.p_tot[:, -1, idx] / 100.0

                self._plot_center_edge(axPz, z_c, p_center, p_edge, label=label)

            for idx, label in schedule("c_tot"):
                c_center = pp.c_tot[:, 0, idx]
                c_edge = pp.c_tot[:, -1, idx]
                self._plot_center_edge(axCz, z_c, c_center, c_edge, label=label)

            if is_1d:
                axPz.set_title('Axial Pressure')
                axCz.set_title(r'Gas total concentration $c_{tot}$')
            else:
                axPz.set_title('Axial Pressure (solid: r=0, dashed: r=R)')
                axCz.set_title(r'Total concentration $c_{tot}$ (solid: r=0, dashed: r=R)')
            axPz.set_ylabel('Pressure [mbar]')
            axPz.set_xlabel('z [m]')
            axPz.grid(True)
            axPz.legend(title='Time')
            axCz.set_ylabel(r'Concentration [mol/m$^3$]')
            axCz.set_xlabel('z [m]')
            axCz.grid(True)
            axCz.legend(title='Time')
            fig_p.tight_layout()

            if save:
                fname_p = os.path.join(base_folder, f"{self.step}_P_ax_ctot_ax_alltimes.{fmt}")
                fig_p.savefig(fname_p, dpi=300, bbox_inches="tight")
                print(f"[Saved] {fname_p}")

            if not show:
                plt.close(fig_p)

        if velocities:
            if is_1d:
                fig_vel, axUz_z = plt.subplots(figsize=(8, 5))
            else:
                fig_vel, (axUz_z, axUz_r) = plt.subplots(1, 2, figsize=(14, 5))
                z_mid_ax = pp.Nz // 2

            # The velocity has no snapshot field: evenly spaced times.
            for idx, label in schedule(None):
                uz_center = pp.u_ax[:, 0, idx]
                uz_edge = pp.u_ax[:, -1, idx]

                self._plot_center_edge(axUz_z, z_f, uz_center, uz_edge, label=label)

                if not is_1d:
                    uz_mid = pp.u_ax[z_mid_ax, :, idx]
                    uz_end = pp.u_ax[-1, :, idx]
                    line_uz_r, = axUz_r.plot(r_c, uz_mid, linestyle='-', label=label)
                    color_uz_r = line_uz_r.get_color()
                    axUz_r.plot(r_c, uz_end, linestyle='--', color=color_uz_r, label=None)

            axUz_z.axhline(
                self.u_ref,
                linestyle='--',
                color='k',
                linewidth=1.8,
                label=self.u_ref_label
            )

            if is_1d:
                axUz_z.set_title(r'Axial velocity $u_z$')
            else:
                axUz_z.set_title(r'$u_z$ vs $z$ (solid: r=0, dashed: r=R)')
                axUz_r.set_title(r'$u_z$ vs $r$ (solid: z=L/2, dashed: z=L)')
            axUz_z.set_ylabel('Velocity [m/s]')
            axUz_z.set_xlabel('z [m]')
            axUz_z.grid(True)
            axUz_z.legend(title='Time / Ref')
            if not is_1d:
                axUz_r.set_ylabel('Velocity [m/s]')
                axUz_r.set_xlabel('r [m]')
                axUz_r.grid(True)
                axUz_r.legend(title='Time')

            fig_vel.tight_layout()

            if save:
                fname_vel = os.path.join(base_folder, f"{self.step}_Uz_ax_Uz_rad_alltimes.{fmt}")
                fig_vel.savefig(fname_vel, dpi=300, bbox_inches="tight")
                print(f"[Saved] {fname_vel}")

            if not show:
                plt.close(fig_vel)

        if show:
            plt.show()

    # ============================================================
    # 2D INTERNAL PROFILES
    # ============================================================
    def plot_internal_profiles_2D(
        self,
        fields=("y_CO2", "q_CO2", "y_H2O", "q_H2O", "y_O2", "y_N2", "c_tot", "T_bed", "p_tot"),
        snapshot_times=None,
        time_unit=None,
        save=True,
        show=False,
        fmt="pdf",
    ):
        """
        Plot 2D snapshots on the `(r, z)` domain.

        Snapshot selection priority is:
        1. explicit `snapshot_times` passed to this method,
        2. per-step/per-field entries in `SNAPSHOT_TIME_OVERRIDES`,
        3. uniform sampling up to the field-specific `t_end` from
           `T_END_OVERRIDES`.

        The layout is a fixed 2x3 boxed grid, while the rendering uses
        `contourf` with the configured colormap.
        """
        self._set_global_style()

        pp = self.pp
        Nt = pp.Nt
        base_folder = self.internal_2D_profiles_folder
        os.makedirs(base_folder, exist_ok=True)

        r_c = self._plot_r_centers()
        z_c = pp.z_c * self.pb.L_b
        Rg, Zg = np.meshgrid(r_c, z_c)

        field_meta = {
            "q_CO2": {
                "suptitle": r"Solid-phase CO$_2$ loading",
                "cbar_label": "Loading [mol/kg]",
                "file_tag": "qCO2",
                "cbar_format": "%.2f",
                "transform": lambda x: x,
            },
            "q_H2O": {
                "suptitle": r"Solid-phase H$_2$O loading",
                "cbar_label": "Loading [mol/kg]",
                "file_tag": "qH2O",
                "cbar_format": "%.2f",
                "transform": lambda x: x,
            },
            "y_CO2": {
                "suptitle": r"Gas-phase CO$_2$ molar fraction",
                "cbar_label": "Molar fraction [-]",
                "file_tag": "yCO2",
                "cbar_format": "%.2e",
                "transform": lambda x: x,
            },
            "y_H2O": {
                "suptitle": r"Gas-phase H$_2$O molar fraction",
                "cbar_label": "Molar fraction [-]",
                "file_tag": "yH2O",
                "cbar_format": "%.2e",
                "transform": lambda x: x,
            },
            "y_O2": {
                "suptitle": r"Gas-phase O$_2$ molar fraction",
                "cbar_label": "Molar fraction [-]",
                "file_tag": "yO2",
                "cbar_format": "%.3f",
                "transform": lambda x: x,
            },
            "y_N2": {
                "suptitle": r"Gas-phase N$_2$ molar fraction",
                "cbar_label": "Molar fraction [-]",
                "file_tag": "yN2",
                "cbar_format": "%.3f",
                "transform": lambda x: x,
            },
            "T_bed": {
                "suptitle": r"Bed temperature",
                "cbar_label": "Temperature [°C]",
                "file_tag": "Tbed",
                "cbar_format": "%.2f",
                "transform": lambda x: x - 273.15,
            },
            "p_tot": {
                "suptitle": r"Total pressure",
                "cbar_label": "Pressure [mbar]",
                "file_tag": "ptot",
                "cbar_format": "%.2f",
                "transform": lambda x: x / 100.0,
            },
            "c_tot": {
                "suptitle": r"Total concentration",
                "cbar_label": r"Concentration [mol/m$^3$]",
                "file_tag": "ctot",
                "cbar_format": "%.2f",
                "transform": lambda x: x,
            },
        }
        for species in pp.gas_species:
            field_meta.setdefault(
                f"y_{species}",
                {
                    "suptitle": f"Gas-phase {species} molar fraction",
                    "cbar_label": "Molar fraction [-]",
                    "file_tag": f"y{species}",
                    "cbar_format": "%.3g",
                    "transform": lambda x: x,
                },
            )
        for species in pp.adsorbates:
            field_meta.setdefault(
                f"q_{species}",
                {
                    "suptitle": f"Solid-phase {species} loading",
                    "cbar_label": "Loading [mol/kg]",
                    "file_tag": f"q{species}",
                    "cbar_format": "%.3g",
                    "transform": lambda x: x,
                },
            )

        for field in fields:
            if not hasattr(pp, field):
                print(f"[Warning] pp has no attribute '{field}', skipping.")
                continue

            data = getattr(pp, field)
            if data.ndim != 3:
                print(f"[Warning] Field '{field}' is not 3D (Nz, Nr, Nt), skipping.")
                continue

            meta = field_meta.get(
                field,
                {
                    "suptitle": field,
                    "cbar_label": field,
                    "file_tag": field,
                    "cbar_format": None,
                    "vmin": None,
                    "vmax": None,
                    "cbar_ticks": None,
                    "transform": lambda x: x,
                },
            )

            time_indices, time_labels, _ = self._get_snapshot_schedule(
                field_name=field,
                n_total=Nt,
                snapshot_times=snapshot_times,
                time_unit=time_unit,
            )
            if len(time_indices) == 0:
                print(f"[WARN] No 2D snapshots available for '{field}', skipping.")
                continue

            data_plot = self._expand_field_for_2d_plot(data)
            snapshots = [meta["transform"](data_plot[:, :, idx]) for idx in time_indices]
            vmin = min(np.nanmin(snap) for snap in snapshots)
            vmax = max(np.nanmax(snap) for snap in snapshots)
            if meta.get("vmin") is not None:
                vmin = float(meta["vmin"])
            if meta.get("vmax") is not None:
                vmax = float(meta["vmax"])
            if np.isclose(vmin, vmax):
                delta = 1e-12 if vmin == 0 else 1e-3 * abs(vmin) + 1e-12
                vmin -= delta
                vmax += delta

            fig, axs = plt.subplots(
                2,
                3,
                figsize=(14, 8),
                sharex=True,
                sharey=True,
            )
            axs_flat = np.asarray(axs).ravel()

            levels = np.linspace(vmin, vmax, 40)
            mappable = None
            cmap = sns.color_palette("Spectral_r", as_cmap=True)
            
            for j, (idx, field_2d) in enumerate(zip(time_indices, snapshots)):
                ax = axs_flat[j]
                mappable = ax.contourf(
                    Rg,
                    Zg,
                    field_2d,
                    levels=levels,
                    cmap=cmap,
                    extend="both",
                )
                ax.set_title(time_labels[j])
                ax.set_xlim(r_c[0], r_c[-1])
                ax.set_ylim(z_c[0], z_c[-1])

            for k in range(len(time_indices), axs_flat.size):
                axs_flat[k].axis("off")

            for irow in range(2):
                for icol in range(3):
                    ax = axs[irow, icol]
                    if not ax.has_data():
                        continue
                    if icol == 0:
                        ax.set_ylabel("z [m]")
                    else:
                        ax.set_ylabel("")
                    if irow == 1:
                        ax.set_xlabel("r [m]")
                    else:
                        ax.set_xlabel("")
                    ax.xaxis.set_major_formatter(self._sig3_formatter())

            suptitle = meta["suptitle"]
            if self.is_1d:
                suptitle += r" (1D mode, uniform in $r$)"
            fig.suptitle(suptitle, y=0.93, x=0.45, fontsize=18)
            fig.tight_layout(rect=[0.05, 0.05, 0.86, 0.92])

            if mappable is not None:
                cbar = fig.colorbar(
                    mappable,
                    ax=axs_flat,
                    orientation="vertical",
                    fraction=0.08,
                    pad=0.02,
                )
                cbar.set_label(meta["cbar_label"])
                cbar_ticks = meta.get("cbar_ticks")
                if cbar_ticks is not None:
                    cbar.set_ticks(cbar_ticks)
                cbar_format = meta.get("cbar_format")
                if cbar_format is not None:
                    cbar.formatter = mticker.FormatStrFormatter(cbar_format)
                    cbar.update_ticks()
                    cbar.ax.yaxis.get_offset_text().set_visible(False)

            if save:
                out_name = f"{self.step}_{meta['file_tag']}_2D_snapshots.{fmt}"
                out_path = os.path.join(base_folder, out_name)
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"[Saved] {out_path}")

            if show:
                plt.show()
            else:
                plt.close(fig)

    def plot_temperature_snapshot_with_wall(
        self,
        snapshot_time=10.0,
        time_unit="min",
        save=True,
        show=False,
        fmt="pdf",
    ):
        """
        Plot one temperature snapshot with bed and wall temperatures side by side.

        The bed temperature is shown on the full `(r, z)` domain. The wall
        temperature is drawn as a narrow strip attached to the outer boundary
        at `r = R_b`, so it reads visually as the external wall of the bed.
        """
        self._set_global_style()

        pp = self.pp
        base_folder = self.internal_2D_profiles_folder
        os.makedirs(base_folder, exist_ok=True)

        time_indices, time_labels, _ = self._get_snapshot_schedule(
            field_name="T_bed",
            n_total=pp.Nt,
            snapshot_times=[snapshot_time],
            time_unit=time_unit,
        )
        if len(time_indices) == 0:
            print("[WARN] No temperature snapshot available, skipping wall-temperature figure.")
            return

        idx = time_indices[0]
        time_label = time_labels[0]

        z_c = pp.z_c * self.pb.L_b
        z_f = pp.z_f * self.pb.L_b
        r_c = self._plot_r_centers()
        Rg, Zg = np.meshgrid(r_c, z_c)

        T_bed_2d = self._expand_field_for_2d_plot(pp.T_bed[:, :, idx]) - 273.15
        T_wall_1d = pp.T_wall[:, idx] - 273.15
        wall_width = 0.10 * self.pb.R_b
        wall_cols = max(6, pp.Nr // 4)
        wall_r_edges = np.linspace(self.pb.R_b, self.pb.R_b + wall_width, wall_cols + 1)
        Rw, Zw = np.meshgrid(wall_r_edges, z_f)
        T_wall_strip = np.repeat(T_wall_1d[:, None], wall_cols, axis=1)

        vmin = float(min(np.nanmin(T_bed_2d), np.nanmin(T_wall_strip)))
        vmax = float(max(np.nanmax(T_bed_2d), np.nanmax(T_wall_strip)))
        if np.isclose(vmin, vmax):
            delta = 1e-12 if vmin == 0 else 1e-3 * abs(vmin) + 1e-12
            vmin -= delta
            vmax += delta

        levels = np.linspace(vmin, vmax, 40)
        cmap = sns.color_palette("Spectral_r", as_cmap=True)

        fig, ax = plt.subplots(figsize=(9.8, 8.0))

        mappable = ax.contourf(
            Rg,
            Zg,
            T_bed_2d,
            levels=levels,
            cmap=cmap,
            extend="both",
        )
        ax.pcolormesh(
            Rw,
            Zw,
            T_wall_strip,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )

        # Mark the bed-wall interface and the outer wall edge.
        ax.axvline(self.pb.R_b, color="k", linewidth=1.1)
        ax.axvline(self.pb.R_b + wall_width, color="k", linewidth=1.1)

        ax.set_xlabel("r [m]")
        ax.set_ylabel("z [m]")
        ax.set_xlim(0.0, self.pb.R_b + wall_width)
        ax.set_ylim(z_c[0], z_c[-1])
        ax.set_xticks([0.0, 0.5 * self.pb.R_b, self.pb.R_b])
        ax.xaxis.set_major_formatter(self._sig3_formatter())
        ax.set_title(f"Temperature Snapshot ({time_label})")

        ax.text(
            self.pb.R_b + 0.5 * wall_width,
            0.5 * self.pb.L_b,
            r"$T_{wall}$",
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 2},
        )

        fig.suptitle(r"$T_{bed}$ with external wall condition", y=0.95, fontsize=18)

        cbar = fig.colorbar(
            mappable,
            ax=ax,
            orientation="vertical",
            fraction=0.06,
            pad=0.03,
        )
        cbar.set_label("Temperature [°C]")
        cbar.formatter = mticker.FormatStrFormatter("%.2f")
        cbar.update_ticks()
        cbar.ax.yaxis.get_offset_text().set_visible(False)

        fig.tight_layout(rect=[0.03, 0.03, 0.92, 0.93])

        if save:
            out_name = f"{self.step}_Tbed_Twall_snapshot.{fmt}"
            out_path = os.path.join(base_folder, out_name)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[Saved] {out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_temperature_profiles_2D_with_wall(
        self,
        snapshot_times=None,
        time_unit=None,
        save=True,
        show=False,
        fmt="pdf",
    ):
        """
        Plot a 2x3 temperature snapshot figure with a wall-temperature strip
        attached to the right side of every panel.

        The snapshot schedule follows the same rules as `plot_internal_profiles_2D`
        for `T_bed`: explicit times first, then `SNAPSHOT_TIME_OVERRIDES`, then
        uniform sampling up to the `T_bed` plotting horizon.
        """
        self._set_global_style()

        pp = self.pp
        base_folder = self.internal_2D_profiles_folder
        os.makedirs(base_folder, exist_ok=True)

        time_indices, time_labels, _ = self._get_snapshot_schedule(
            field_name="T_bed",
            n_total=pp.Nt,
            snapshot_times=snapshot_times,
            time_unit=time_unit,
        )
        if len(time_indices) == 0:
            print("[WARN] No temperature snapshots available, skipping 2D wall-temperature figure.")
            return

        z_c = pp.z_c * self.pb.L_b
        z_f = pp.z_f * self.pb.L_b
        r_c = self._plot_r_centers()
        Rg, Zg = np.meshgrid(r_c, z_c)

        wall_width = 0.10 * self.pb.R_b
        wall_cols = max(6, pp.Nr // 4)
        wall_r_edges = np.linspace(self.pb.R_b, self.pb.R_b + wall_width, wall_cols + 1)
        Rw, Zw = np.meshgrid(wall_r_edges, z_f)

        T_bed_snaps = [self._expand_field_for_2d_plot(pp.T_bed[:, :, idx]) - 273.15 for idx in time_indices]
        T_wall_snaps = [pp.T_wall[:, idx] - 273.15 for idx in time_indices]

        vmin = min(
            min(np.nanmin(Tb) for Tb in T_bed_snaps),
            min(np.nanmin(Tw) for Tw in T_wall_snaps),
        )
        vmax = max(
            max(np.nanmax(Tb) for Tb in T_bed_snaps),
            max(np.nanmax(Tw) for Tw in T_wall_snaps),
        )
        if np.isclose(vmin, vmax):
            delta = 1e-12 if vmin == 0 else 1e-3 * abs(vmin) + 1e-12
            vmin -= delta
            vmax += delta

        levels = np.linspace(vmin, vmax, 40)
        cmap = sns.color_palette("Spectral_r", as_cmap=True)

        fig, axs = plt.subplots(
            2,
            3,
            figsize=(15.5, 8.8),
            sharex=True,
            sharey=True,
        )
        axs_flat = np.asarray(axs).ravel()
        mappable = None

        for j, (idx, Tb, Tw, label) in enumerate(zip(time_indices, T_bed_snaps, T_wall_snaps, time_labels)):
            ax = axs_flat[j]
            mappable = ax.contourf(
                Rg,
                Zg,
                Tb,
                levels=levels,
                cmap=cmap,
                extend="both",
            )
            T_wall_strip = np.repeat(Tw[:, None], wall_cols, axis=1)
            ax.pcolormesh(
                Rw,
                Zw,
                T_wall_strip,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="flat",
            )

            ax.axvline(self.pb.R_b, color="k", linewidth=0.9)
            ax.axvline(self.pb.R_b + wall_width, color="k", linewidth=0.9)
            ax.set_title(label)
            ax.set_xlim(0.0, self.pb.R_b + wall_width)
            ax.set_ylim(z_c[0], z_c[-1])

            ax.text(
                self.pb.R_b + 0.5 * wall_width,
                0.5 * self.pb.L_b,
                r"$T_w$",
                rotation=90,
                ha="center",
                va="center",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 1.5},
            )

        for k in range(len(time_indices), axs_flat.size):
            axs_flat[k].axis("off")

        for irow in range(2):
            for icol in range(3):
                ax = axs[irow, icol]
                if not ax.has_data():
                    continue
                if icol == 0:
                    ax.set_ylabel("z [m]")
                else:
                    ax.set_ylabel("")
                if irow == 1:
                    ax.set_xlabel("r [m]")
                else:
                    ax.set_xlabel("")
                ax.set_xticks([0.0, 0.5 * self.pb.R_b, self.pb.R_b])
                ax.xaxis.set_major_formatter(self._sig3_formatter())

        title = r"$T_{bed}$ snapshots with external wall temperature"
        if self.is_1d:
            title += r" (1D mode, uniform in $r$)"
        fig.suptitle(title, y=0.94, fontsize=18)
        fig.tight_layout(rect=[0.04, 0.04, 0.90, 0.93])

        if mappable is not None:
            cbar = fig.colorbar(
                mappable,
                ax=axs_flat,
                orientation="vertical",
                fraction=0.06,
                pad=0.02,
            )
            cbar.set_label("Temperature [°C]")
            cbar.formatter = mticker.FormatStrFormatter("%.2f")
            cbar.update_ticks()
            cbar.ax.yaxis.get_offset_text().set_visible(False)

        if save:
            out_name = f"{self.step}_Tbed_Twall_2D_snapshots.{fmt}"
            out_path = os.path.join(base_folder, out_name)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[Saved] {out_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ============================================================
    # 1D OUTLET PROFILES
    # ============================================================
    def plot_outlet_profiles(
        self,
        species=True,
        temperature=True,
        pressure=True,
        save=True,
        show=False,
        fmt='pdf'
    ):
        """
        Plot the temporal evolution of OUTLET (z = L) profiles.

        In 2D mode, species and temperature are mixed-cup
        (mass-flux weighted) on steps that carry an advected stream and
        area-weighted on closed-inlet and closed-outlet steps (the step classes
        of the CSS figures), while pressure and axial velocity are radially
        averaged. In 1D mode, the single stored radial value is plotted
        directly for every outlet quantity.

        Layout:
        - 1x2: gas-phase CO2 and H2O molar fractions
        - 1x2: gas-phase O2 and N2 molar fractions
        - 1x1: bed temperature
        - 1x2: pressure and axial velocity
        """
        self._set_global_style()

        pp = self.pp
        line_width = 2.5
        zL_idx = -1

        t_max_dim = pp.time[-1]
        if t_max_dim > 3600:
            t = pp.time / 3600
            t_unit = "Time [h]"
        elif t_max_dim > 300:
            t = pp.time / 60
            t_unit = "Time [min]"
        else:
            t = pp.time
            t_unit = "Time [s]"
        x_locator = mticker.MaxNLocator(nbins=8)

        fig_list = []

        # Closed-inlet and closed-outlet steps carry no advected stream: area mean,
        # the same step classes as the CSS figures.
        area_mean = self.step in (
            CyclicSteadyStateProfiles.CLOSED_INLET_STEPS
            | CyclicSteadyStateProfiles.CLOSED_OUTLET_STEPS
        )
        reduce_mode = "radial" if area_mean else "mixed_cup"
        average_label = "area mean" if area_mean else "mixed-cup"

        if species or temperature:
            c_tot_z = pp.c_tot[zL_idx, :, :]
            y_CO2_z = pp.y_CO2[zL_idx, :, :]
            y_H2O_z = pp.y_H2O[zL_idx, :, :]
            y_O2_z = pp.y_O2[zL_idx, :, :]
            y_N2_z = pp.y_N2[zL_idx, :, :]
            u_ax_z = pp.u_ax[zL_idx, :, :]

        if species:
            y_CO2_out = self._outlet_reduce(
                y_CO2_z,
                mode=reduce_mode,
                c_tot_z=c_tot_z,
                y_CO2_z=y_CO2_z,
                y_H2O_z=y_H2O_z,
                y_O2_z=y_O2_z,
                y_N2_z=y_N2_z,
                u_ax_z=u_ax_z,
            )
            y_H2O_out = self._outlet_reduce(
                y_H2O_z,
                mode=reduce_mode,
                c_tot_z=c_tot_z,
                y_CO2_z=y_CO2_z,
                y_H2O_z=y_H2O_z,
                y_O2_z=y_O2_z,
                y_N2_z=y_N2_z,
                u_ax_z=u_ax_z,
            )
            y_O2_out = self._outlet_reduce(
                y_O2_z,
                mode=reduce_mode,
                c_tot_z=c_tot_z,
                y_CO2_z=y_CO2_z,
                y_H2O_z=y_H2O_z,
                y_O2_z=y_O2_z,
                y_N2_z=y_N2_z,
                u_ax_z=u_ax_z,
            )
            y_N2_out = self._outlet_reduce(
                y_N2_z,
                mode=reduce_mode,
                c_tot_z=c_tot_z,
                y_CO2_z=y_CO2_z,
                y_H2O_z=y_H2O_z,
                y_O2_z=y_O2_z,
                y_N2_z=y_N2_z,
                u_ax_z=u_ax_z,
            )

            fig1, axs1 = plt.subplots(1, 2, figsize=(14, 5))
            axs1[0].plot(t, y_CO2_out, label=r'Model CO$_2$', linewidth=line_width, color='tab:blue')
            axs1[0].axhline(self.y_CO2_ref, linestyle='--', label=self.y_ref_label, linewidth=2, color='tab:blue')
            axs1[1].plot(t, y_H2O_out, label=r'Model H$_2$O', linewidth=line_width, color='tab:orange')
            axs1[1].axhline(self.y_H2O_ref, linestyle='--', label=self.y_ref_label, linewidth=2, color='tab:orange')

            if self.is_1d:
                axs1[0].set_title(r'Outlet Gas-phase CO$_2$')
                axs1[1].set_title(r'Outlet Gas-phase H$_2$O')
            else:
                axs1[0].set_title(r'Outlet Gas-phase CO$_2$ (' + average_label + ')')
                axs1[1].set_title(r'Outlet Gas-phase H$_2$O (' + average_label + ')')
            axs1[0].set_ylabel('Molar fraction [-]')
            axs1[0].set_xlabel(t_unit)
            axs1[1].set_xlabel(t_unit)
            axs1[0].grid(True)
            axs1[1].grid(True)
            axs1[0].legend()
            axs1[1].legend()
            if x_locator is not None:
                for ax in axs1:
                    ax.xaxis.set_major_locator(x_locator)
            fig1.tight_layout()
            fig_list.append((fig1, f"{self.step}_outlet_species.{fmt}"))

            fig1b, axs1b = plt.subplots(1, 2, figsize=(14, 5))
            axs1b[0].plot(t, y_O2_out, label=r'Model O$_2$', linewidth=line_width, color='tab:green')
            axs1b[0].axhline(self.y_O2_ref, linestyle='--', label=self.y_ref_label, linewidth=2, color='tab:green')
            axs1b[1].plot(t, y_N2_out, label=r'Model N$_2$', linewidth=line_width, color='tab:brown')
            axs1b[1].axhline(self.y_N2_ref, linestyle='--', label=self.y_ref_label, linewidth=2, color='tab:brown')

            if self.is_1d:
                axs1b[0].set_title(r'Outlet Gas-phase O$_2$')
                axs1b[1].set_title(r'Outlet Gas-phase N$_2$')
            else:
                axs1b[0].set_title(r'Outlet Gas-phase O$_2$ (' + average_label + ')')
                axs1b[1].set_title(r'Outlet Gas-phase N$_2$ (' + average_label + ')')
            axs1b[0].set_ylabel('Molar fraction [-]')
            axs1b[0].set_xlabel(t_unit)
            axs1b[1].set_xlabel(t_unit)
            axs1b[0].grid(True)
            axs1b[1].grid(True)
            axs1b[0].legend()
            axs1b[1].legend()
            if x_locator is not None:
                for ax in axs1b:
                    ax.xaxis.set_major_locator(x_locator)
            fig1b.tight_layout()
            fig_list.append((fig1b, f"{self.step}_outlet_species_O2_N2.{fmt}"))

        if temperature:
            T_bed_out = self._outlet_reduce(
                pp.T_bed[zL_idx, :, :],
                mode=reduce_mode,
                c_tot_z=c_tot_z,
                y_CO2_z=y_CO2_z,
                y_H2O_z=y_H2O_z,
                y_O2_z=y_O2_z,
                y_N2_z=y_N2_z,
                u_ax_z=u_ax_z,
            )

            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(t, T_bed_out - 273.15,  label='Bed',  linewidth=line_width)
            ax2.axhline(self.pb.T_ref - 273.15, linestyle='--',
                        label=self.T_ref_label, linewidth=2)

            if x_locator is not None:
                ax2.xaxis.set_major_locator(x_locator)
            if self.is_1d:
                ax2.set_title('Outlet Temperature')
            else:
                ax2.set_title('Outlet Temperature (' + average_label + ')')
            ax2.set_ylabel('Temperature [°C]')
            ax2.set_xlabel(t_unit)
            ax2.grid(True)
            ax2.legend()
            fig2.tight_layout()
            fig_list.append((fig2, f"{self.step}_outlet_temperature.{fmt}"))

        if pressure:
            p_out = self._outlet_reduce(pp.p_tot[zL_idx, :, :], mode="radial")
            uz_out = self._outlet_reduce(pp.u_ax[zL_idx, :, :], mode="radial")

            fig3, axs3 = plt.subplots(1, 2, figsize=(14, 5))
            axs3[0].plot(
                t, p_out / 100.0,
                label='Total Pressure',
                linewidth=line_width,
                color='tab:red',
            )
            axs3[0].axhline(self.pb.p_out / 100.0,
                            linestyle='--', label=self.p_ref_label, linewidth=2, color='tab:red')
            axs3[1].plot(
                t, uz_out,
                label='Velocity',
                linewidth=line_width,
                color='tab:green',
            )
            axs3[1].axhline(self.pb.u_inlet,
                            linestyle='--', label=self.u_ref_label, linewidth=2, color='tab:green')

            if self.is_1d:
                axs3[0].set_title('Outlet Pressure')
                axs3[1].set_title('Outlet Axial Velocity')
            else:
                axs3[0].set_title('Outlet Pressure (radially averaged)')
                axs3[1].set_title('Outlet Axial Velocity (radially averaged)')
            axs3[0].set_ylabel('Pressure [mbar]')
            axs3[1].set_ylabel('Superficial Velocity [m/s]')
            axs3[0].set_xlabel(t_unit)
            axs3[1].set_xlabel(t_unit)
            axs3[0].grid(True)
            axs3[1].grid(True)
            axs3[0].legend()
            axs3[1].legend()
            if x_locator is not None:
                for ax in axs3:
                    ax.xaxis.set_major_locator(x_locator)
            fig3.tight_layout()
            fig_list.append((fig3, f"{self.step}_outlet_pressure_velocity.{fmt}"))

        if save:
            for fig, fname in fig_list:
                out_path = os.path.join(self.outlet_profiles_folder, fname)
                fig.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"[Saved] {out_path}")

        if show:
            plt.show()
        else:
            for fig, _ in fig_list:
                plt.close(fig)

    # Helpers 
    def _get_t_end_for_field(self, field_name):
        """
        Return the plotting time limit for a field.

        Defaults to the full simulated time window unless a step-specific
        override is provided in `T_END_OVERRIDES`.
        """
        full = self.pp.time[-1]
        step = self.pb.step

        if step not in T_END_OVERRIDES:
            return full

        overrides = T_END_OVERRIDES[step]
        if field_name not in overrides:
            return full

        value = overrides[field_name]
        if value is None:
            return full

        return min(value, full)

    def _profile_schedule(self, field_name, n_times=10):
        """
        Return `(time index, legend label)` pairs for the 1D profile panels of a field.

        A field with snapshot times for this step (`SNAPSHOT_TIME_OVERRIDES`, where a
        run also puts its `snapshot_times`) is drawn at the times of its 2D map, so the
        map and the profiles of a field show the same instants. Any other field, and
        `field_name=None` (a panel without a snapshot field), is drawn at `n_times`
        evenly spaced saved times.
        """
        pp = self.pp
        if field_name in SNAPSHOT_TIME_OVERRIDES.get(self.pb.step, {}):
            time_indices, time_labels, _ = self._get_snapshot_schedule(
                field_name, pp.Nt, report=False
            )
            return list(zip(time_indices, time_labels))

        time_indices = np.linspace(0, pp.Nt - 1, min(n_times, pp.Nt), dtype=int)
        t_max_dim = pp.time[-1]
        if t_max_dim > 3600:
            time_labels = [f"t = {pp.time[i] / 3600:.2f} h" for i in time_indices]
        elif t_max_dim > 300:
            time_labels = [f"t = {pp.time[i] / 60:.1f} min" for i in time_indices]
        else:
            time_labels = [f"t = {pp.time[i]:.1f} s" for i in time_indices]
        return list(zip(time_indices, time_labels))

    def _get_snapshot_schedule(
        self, field_name, n_total, snapshot_times=None, time_unit=None, report=True
    ):
        """
        Build snapshot indices, labels, and filename tag for a 2D field.

        Explicit `snapshot_times` take precedence. If omitted, the method looks
        for a per-step/per-field entry in `SNAPSHOT_TIME_OVERRIDES`. If no exact
        override exists, snapshots are uniformly sampled from `t=0` to the
        field-specific `t_end`. `report=False` skips the printed time mapping
        (the 1D profiles reuse the schedule the 2D map already reported).
        """
        times_s = None
        label_unit = time_unit
        from_explicit_times = False

        if snapshot_times is not None:
            times_req = np.asarray(snapshot_times, dtype=float).ravel()
            if times_req.size == 0:
                return [], [], "empty"
            label_unit = "s" if time_unit is None else time_unit
            label_unit = self._normalize_time_unit(label_unit)
            times_s = self._convert_times_to_seconds(times_req, label_unit)
            from_explicit_times = True
        else:
            step_overrides = SNAPSHOT_TIME_OVERRIDES.get(self.pb.step, {})
            field_override = step_overrides.get(field_name)
            if field_override is not None:
                times_req = np.asarray(field_override.get("times", ()), dtype=float).ravel()
                if times_req.size > 0:
                    label_unit = self._normalize_time_unit(field_override.get("unit", "s"))
                    times_s = self._convert_times_to_seconds(times_req, label_unit)
                    from_explicit_times = True

        if times_s is not None:
            t_end = float(self.pp.time[-1])
            if report and np.any(times_s > t_end + 1e-12):
                print(
                    f"[WARN] Requested 2D times for '{field_name}' exceed simulated window "
                    f"(t_end={t_end:.1f} s). Nearest available snapshots are used."
                )
            time_indices = [int(np.argmin(np.abs(self.pp.time - t_s))) for t_s in times_s]
            if from_explicit_times and report:
                time_indices = self._report_snapshot_mapping(
                    field_name=field_name,
                    requested_times_s=times_s,
                    time_indices=time_indices,
                )
            elif from_explicit_times:
                time_indices = list(dict.fromkeys(time_indices))  # same times, once each
            time_labels = self._format_time_labels(time_indices, label_unit=label_unit)
            tag = self._build_snapshot_tag(time_indices, label_unit=label_unit)
            return time_indices, time_labels, tag

        t_end = self._get_t_end_for_field(field_name)
        valid = np.where(self.pp.time <= t_end)[0]
        if valid.size > 0:
            last_idx = int(valid[-1])
        else:
            last_idx = n_total - 1

        n_snap = min(6, last_idx + 1)
        time_indices = np.linspace(0, last_idx, n_snap, dtype=int).tolist()
        time_labels = self._format_time_labels(time_indices)
        tag = self._build_snapshot_tag(time_indices)
        return time_indices, time_labels, tag

    def _normalize_time_unit(self, unit):
        """Map common time-unit aliases to `s`, `min`, or `h`."""
        if unit is None:
            return "s"

        unit_norm = str(unit).strip().lower()
        unit_map = {
            "s": "s",
            "sec": "s",
            "secs": "s",
            "second": "s",
            "seconds": "s",
            "min": "min",
            "mins": "min",
            "minute": "min",
            "minutes": "min",
            "h": "h",
            "hr": "h",
            "hrs": "h",
            "hour": "h",
            "hours": "h",
        }
        return unit_map.get(unit_norm, "s")

    def _convert_times_to_seconds(self, times, unit):
        """Convert a sequence of times from `s`, `min`, or `h` to seconds."""
        unit = self._normalize_time_unit(unit)
        if unit == "h":
            return times * 3600.0
        if unit == "min":
            return times * 60.0
        return times

    def _format_time_labels(self, time_indices, label_unit=None):
        """Return formatted subplot titles for the selected snapshot indices."""
        if len(time_indices) == 0:
            return []

        if label_unit is None:
            t_max = self.pp.time[time_indices[-1]]
            if t_max > 3600:
                label_unit = "h"
            elif t_max > 300:
                label_unit = "min"
            else:
                label_unit = "s"
        else:
            label_unit = self._normalize_time_unit(label_unit)

        if label_unit == "h":
            scale = 3600.0
            decimals = 2
        elif label_unit == "min":
            scale = 60.0
            decimals = 1
        else:
            scale = 1.0
            decimals = 1

        return [
            f"t = {self.pp.time[idx] / scale:.{decimals}f} {label_unit}"
            for idx in time_indices
        ]

    def _build_snapshot_tag(self, time_indices, label_unit=None):
        """Build a compact filename suffix from the selected snapshot times."""
        if len(time_indices) == 0:
            return "empty"

        if label_unit is None:
            t_max = self.pp.time[time_indices[-1]]
            if t_max > 3600:
                label_unit = "h"
            elif t_max > 300:
                label_unit = "min"
            else:
                label_unit = "s"
        else:
            label_unit = self._normalize_time_unit(label_unit)

        if label_unit == "h":
            scale = 3600.0
        elif label_unit == "min":
            scale = 60.0
        else:
            scale = 1.0

        values = self.pp.time[np.asarray(time_indices, dtype=int)] / scale
        return "_".join(f"{value:g}" for value in values)

    def _report_snapshot_mapping(self, field_name, requested_times_s, time_indices):
        """
        Print how requested snapshot times map to the stored time grid.

        This is useful when very early requested times collapse to `t=0`
        because the saved time vector is too coarse.
        """
        time_grid = np.asarray(self.pp.time, dtype=float)
        if time_grid.size < 2:
            return list(dict.fromkeys(time_indices))  # one saved state: one panel

        dt = np.diff(time_grid)
        dt_min = float(np.min(dt))
        first_nonzero = float(time_grid[1])
        actual_times = time_grid[np.asarray(time_indices, dtype=int)]

        too_early = [
            req for req, act in zip(requested_times_s, actual_times)
            if req > 0.0 and act == 0.0
        ]
        if too_early:
            requested_list = ", ".join(f"{req:g} s" for req in too_early)
            print(
                f"[WARN] Requested 2D snapshots for '{field_name}' include times that are "
                f"too small for the saved time grid.\n"
                f"  Minimum available nonzero time: {first_nonzero:g} s\n"
                f"  Minimum saved time spacing:    {dt_min:g} s\n"
                f"  Requested too-early times:    {requested_list}\n"
                f"  These snapshots will be mapped to t = 0 s."
            )

        mapping_lines = "\n".join(
            f"  {req:g} s -> {act:g} s"
            for req, act in zip(requested_times_s, actual_times)
        )
        print(
            f"[INFO] 2D snapshot mapping for '{field_name}':\n"
            f"{mapping_lines}"
        )

        unique_indices = []
        seen = set()
        for idx in time_indices:
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)

        if len(unique_indices) < len(time_indices):
            print(
                f"[WARN] Requested 2D snapshots for '{field_name}' collapse onto "
                f"{len(unique_indices)} unique stored times because the saved grid "
                f"is coarser than some requested snapshot times."
            )

        return unique_indices
       
       
    
