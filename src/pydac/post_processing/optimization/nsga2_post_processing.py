"""
pyDAC - Post-processing for single NSGA-II campaigns.

This module loads saved Pareto fronts and full populations, then generates
exploratory plots and comparisons for a specific optimization campaign.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde


from pydac.utils.flow_transforms import STEAM_FEED_TEMPERATURE_K, steam_density_from_pressure
from pydac.utils.paths import display_path


class NSGA2_pp:
    def __init__(self, pareto_folder):
        self.pareto_folder = Path(pareto_folder)
        self.pareto_path = self.pareto_folder / "last_pareto_front.npz"
        self.population_path = self.pareto_folder / "last_population.npz"
        self.figures_folder = self.pareto_folder / "figures"
        self.figures_folder.mkdir(parents=True, exist_ok=True)
        self.load_data()

    def load_data(self):
        """Load Pareto front and full population data from .npz files."""
        if self.pareto_path.exists():
            data = np.load(self.pareto_path)
            self.X = data["X"]
            self.F = data["F"]
        else:
            print(f"[ERROR] Pareto front not found: {display_path(self.pareto_path)}")
            self.X = self.F = None

        if self.population_path.exists():
            pop_data = np.load(self.population_path)
            self.X_pop = pop_data["X"]
            self.F_pop = pop_data["F"]
        else:
            self.X_pop = self.F_pop = None

        from pydac.utils.provenance import file_artifact, source_record, start_activity
        sources = [{"artifact": file_artifact(path)}
                   for path in (self.pareto_path, self.population_path) if path.is_file()]
        summary_path = self.pareto_folder / "optimization_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            sources.append(source_record(summary.get("provenance")))
        self.analysis_provenance = start_activity(
            self.pareto_folder, "analysis", sources=sources,
            execution={"operation": "load_pareto_and_population"},
        )
            

            

    def plot_pareto_front(self, save=False, filename="pareto_front.pdf"):
        """Plot the Pareto front only."""
        if self.F is None:
            print("[ERROR] No Pareto front loaded.")
            return

        plt.figure(figsize=(8, 6))
        plt.scatter(-self.F[:, 1], self.F[:, 0], label="Pareto Front", edgecolors="k")
        plt.ylabel("Equivalent Work [MJ/kg CO₂]")
        plt.xlabel("Productivity [kg CO₂/m³·day]")
        plt.title("NSGA-II Pareto Front")
        plt.grid(True)
        plt.tight_layout()

        if save:
            fig_path = self.figures_folder / filename
            plt.savefig(fig_path, dpi=300)
            print(f"[FIG] Saved Pareto front to {display_path(fig_path)}")
        else:
            plt.show()
        plt.close()

    def plot_full_population(self, save=True, filename="full_population.pdf"):
        """Plot all individuals from the final saved population."""
        if self.F_pop is None:
            print("[ERROR] No saved population found.")
            return

        plt.figure(figsize=(8, 6))
        plt.scatter(-self.F_pop[:, 1], self.F_pop[:, 0], s=40, edgecolors='k', alpha=0.8)
        plt.ylabel("Equivalent Work [MJ/kg CO₂]")
        plt.xlabel("Productivity [kg CO₂/m³·day]")
        plt.title("NSGA-II Final Population (All Individuals)")
        plt.grid(True)
        plt.tight_layout()

        if save:
            fig_path = self.figures_folder / filename
            plt.savefig(fig_path, dpi=300)
            print(f"[FIG] Saved full population plot to {display_path(fig_path)}")
        else:
            plt.show()
        plt.close()

    def plot_population_vs_front(self, save=True, filename="population_vs_front.pdf"):
        """Plot both the final population and the Pareto front for comparison."""
        if self.F_pop is None or self.F is None:
            print("[ERROR] Population or Pareto front missing.")
            return

        plt.figure(figsize=(8, 6))
        plt.scatter(-self.F_pop[:, 1], self.F_pop[:, 0], alpha=0.4, label="Population", color="gray")
        plt.scatter(-self.F[:, 1], self.F[:, 0], label="Pareto Front", color="red")
        plt.ylabel("Equivalent Work [MJ/kg CO₂]")
        plt.xlabel("Productivity [kg CO₂/m³·day]")
        plt.title("Population vs Pareto Front")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save:
            fig_path = self.figures_folder / filename
            plt.savefig(fig_path, dpi=300)
            print(f"[FIG] Saved population vs Pareto front to {display_path(fig_path)}")
        else:
            plt.show()
        plt.close()

    def plot_scaled_decision_boxplot(self, save=True, show_stats=False):
        """
        Enhanced boxplot of scaled decision variables with mean, min, and max overlay.

        Parameters
        ----------
        save : bool
            Whether to save the figure.
        show_stats : bool
            Whether to plot mean, min, and max markers on top of the boxplot.
        """

        # Load bounds
        bounds_path = self.pareto_folder.parent / "pareto_bounds.csv"
        bounds_df = pd.read_csv(bounds_path)
        bounds = {row['variable']: (row['lower_bound'], row['upper_bound']) for _, row in bounds_df.iterrows()}

        def latex_label(name):
            if "_" in name:
                base, sub = name.split("_", 1)
                return fr"${base}_{{{sub}}}$"
            return fr"${name}$"
        
        var_names = list(bounds.keys())
        var_labels = [latex_label(v) for v in var_names]
        var_indices = [var_names.index(v) for v in var_names]
        X = self.X[:, var_indices]

        # Scale variables
        X_scaled = np.zeros_like(X)
        for i, name in enumerate(var_names):
            lb, ub = bounds[name]
            X_scaled[:, i] = np.clip((X[:, i] - lb) / (ub - lb), 0, None)

        # Compute statistics
        mins = np.min(X_scaled, axis=0)
        maxs = np.max(X_scaled, axis=0)

        # Plot boxplot
        plt.figure(figsize=(10, 6))
        # Tick labels set apart: boxplot's `labels` is gone in matplotlib 3.11.
        box = plt.boxplot(X_scaled, patch_artist=True, showfliers=True)
        plt.xticks(range(1, len(var_labels) + 1), var_labels, fontsize=12)

        # Customize boxes
        for patch in box['boxes']:
            patch.set(facecolor="#ADD8E6")

        if show_stats:
            x_pos = np.arange(1, len(var_names) + 1)
            plt.scatter(x_pos, mins,  color="red", marker="d", label="Min", zorder=3, s=10)
            plt.scatter(x_pos, maxs,  color="green", marker="d", label="Max", zorder=3, s=15)
            median_line = mlines.Line2D([], [], color='orange', label='Median', linewidth=2)
            plt.legend(handles=[  # custom order
            mlines.Line2D([], [], color='red', marker='d', linestyle='None', label='Min'),
            mlines.Line2D([], [], color='green', marker='d', linestyle='None', label='Max'),
            median_line
        ])

        plt.ylabel("Scaled Value [0-1]", fontsize=12)
        plt.title("Distribution of Scaled Decision Variables (Box Plot)", fontsize=14)
        plt.grid(True)
        plt.tight_layout()

        if save:
            path = self.figures_folder / "distributions_boxplot.pdf"
            plt.savefig(path, dpi=300)
            print(f"[FIG] Saved enhanced boxplot to {display_path(path)}")
        else:
            plt.show()
        plt.close()

    def plot_scaled_decision_violinplot(self, save=True):
        """
        Violin plot of scaled decision variables with optional overlaid baseline markers and median line.

        Parameters
        ----------
        save : bool
            Whether to save the figure.
        show_baseline : bool
            Whether to plot baseline horizontal lines (e.g. reference from literature).
        baseline_dict : dict or None
            Optional dictionary {var_name: scaled_baseline_value}.
        """

        # Load bounds
        bounds_path = self.pareto_folder.parent / "pareto_bounds.csv"
        bounds_df = pd.read_csv(bounds_path)
        bounds = {row['variable']: (row['lower_bound'], row['upper_bound']) for _, row in bounds_df.iterrows()}
        def latex_label(name):
            if "_" in name:
                base, sub = name.split("_", 1)
                return fr"${base}_{{{sub}}}$"
            return fr"${name}$"
        
        var_names = list(bounds.keys())
        var_labels = [latex_label(v) for v in var_names]
        var_indices = [var_names.index(v) for v in var_names]
        X = self.X[:, var_indices]

        # Scale variables
        X_scaled = np.zeros_like(X)
        for i, name in enumerate(var_names):
            lb, ub = bounds[name]
            X_scaled[:, i] = np.clip((X[:, i] - lb) / (ub - lb), 0, None)
            
        # Compute modes
        mode_dict = self.compute_variable_modes(X_scaled, var_names, save_csv=True)

        # Create DataFrame
        df = pd.DataFrame(X_scaled, columns=var_names)
        df_melted = df.melt(var_name="Variable", value_name="Scaled Value")

        # Set color palette with transparency
        pastel_transparent = to_rgba("#ADD8E6")
        palette = [pastel_transparent] * len(var_names)

        # Plot
        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(
            data=df_melted,
            x="Variable",
            y="Scaled Value",
            cut=0,
            hue="Variable",
            density_norm="count",
            legend=False,
            palette=palette,
            saturation=0.8,
            native_scale=True,
            linecolor="#262B2C",  # edge color
            zorder = 1,
        )

        # Overlay modes as red diamonds
        for i, var in enumerate(var_names):
            mode = mode_dict[var]
            ax.scatter(i, mode, color="red", marker="D", s=50, label="Mode" if i == 0 else None)

        # Final formatting
        ax.set_xticks(np.arange(len(var_labels)))
        ax.set_xticklabels(var_labels, fontsize=12)
        ax.set_ylabel("Scaled Value [0-1]", fontsize=12)
        ax.set_title("Distribution of Scaled Decision Variables (Violin Plot)", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            path = self.figures_folder / "distributions_violinplot.pdf"
            plt.savefig(path, dpi=300)
            print(f"[FIG] Saved violin plot to {display_path(path)}")
        else:
            plt.show()
        plt.close()

    def compute_variable_modes(self, X_scaled, var_names, resolution=1000, save_csv=False):
        """
        Compute KDE-based modes for each scaled decision variable and optionally save to CSV.

        Parameters
        ----------
        X_scaled : ndarray
            Scaled decision matrix [n_points x n_vars]
        var_names : list of str
            Names of the variables
        resolution : int
            Resolution for KDE grid
        save_csv : bool
            Whether to save the modes to CSV

        Returns
        -------
        mode_dict : dict
            Dictionary {var_name: mode_scaled}
        """
        from scipy.stats import gaussian_kde
        import pandas as pd
        import numpy as np

        # Load bounds
        bounds_path = self.pareto_folder.parent / "pareto_bounds.csv"
        bounds_df = pd.read_csv(bounds_path)
        bounds = {row['variable']: (row['lower_bound'], row['upper_bound']) for _, row in bounds_df.iterrows()}

        mode_dict = {}
        mode_unscaled_dict = {}

        for i, var in enumerate(var_names):
            x_scaled = X_scaled[:, i]
            if np.ptp(x_scaled) == 0:
                # One front point, or every point at one value: no spread for a
                # KDE (it raises), and that value is the mode.
                mode_scaled = float(x_scaled[0])
            else:
                kde = gaussian_kde(x_scaled)
                x_grid = np.linspace(0, 1, resolution)
                density = kde(x_grid)
                mode_scaled = x_grid[np.argmax(density)]
            mode_dict[var] = mode_scaled

            # Invert scaling
            lb, ub = bounds[var]
            mode_unscaled = mode_scaled * (ub - lb) + lb
            mode_unscaled_dict[var] = mode_unscaled

        # Save both scaled and unscaled
        save_path = self.pareto_folder / "decision_variable_modes.csv"
        if save_csv:
            df_modes = pd.DataFrame({
                "Variable": var_names,
                "Mode (scaled)": [mode_dict[v] for v in var_names],
                "Mode (unscaled)": [mode_unscaled_dict[v] for v in var_names]
            })
            df_modes.to_csv(save_path, index=False)
            print(f"[CSV] Saved scaled and unscaled modes to: {display_path(save_path)}")

        return mode_dict

    def _load_flow_transform_metadata(self):
        metadata_path = self.pareto_folder.parent / "pareto_flow_transform.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _add_model_flow_columns(self, df):
        """
        Add derived model-flow columns to velocity-space Pareto data.

        These columns are not decision variables; they are the values passed to
        the CSS model after applying the geometry and steam-density transform.
        """
        if "v_feed" not in df.columns and "v_steam" not in df.columns:
            return df

        metadata = self._load_flow_transform_metadata()
        if metadata is None or "inlet_area_model" not in metadata:
            print(
                "[WARN] Velocity-space Pareto detected, but "
                "pareto_flow_transform.json is missing. Skipping derived "
                "model-flow columns."
            )
            return df

        inlet_area = float(metadata["inlet_area_model"])
        if "v_feed" in df.columns:
            df["vol_flowrate_model"] = df["v_feed"] * inlet_area

        if "v_steam" in df.columns:
            if "P_vac" not in df.columns:
                print("[WARN] Cannot derive m_dot_steam_model without P_vac.")
                return df

            T_feed = float(
                metadata.get("steam_feed_temperature_K", STEAM_FEED_TEMPERATURE_K)
            )
            rho_values = []
            pressure_used_values = []
            p_sat_values = []
            for pressure in df["P_vac"]:
                rho, pressure_used, p_sat = steam_density_from_pressure(
                    pressure,
                    T_feed=T_feed,
                    cap_at_saturation=True,
                )
                rho_values.append(rho)
                pressure_used_values.append(pressure_used)
                p_sat_values.append(p_sat)

            df["rho_steam_model"] = rho_values
            df["p_steam_density_model"] = pressure_used_values
            df["p_steam_saturation_model"] = p_sat_values
            df["m_dot_steam_model"] = (
                df["v_steam"] * inlet_area * df["rho_steam_model"]
            )

        return df

    def export_pareto_csv(self, filename="pareto_front.csv"):
        """Export the Pareto front as a CSV file with decision variables and objective values."""
        if self.X is None or self.F is None:
            print("[ERROR] No Pareto data to export.")
            return

        bounds_path = self.pareto_folder.parent / "pareto_bounds.csv"
        bounds_df = pd.read_csv(bounds_path)
        columns = bounds_df["variable"].tolist()
        df = pd.DataFrame(self.X, columns=columns)
        df = self._add_model_flow_columns(df)
        df["eq_work"] = self.F[:, 0]
        df["productivity"] = -self.F[:, 1]

        csv_path = self.pareto_folder / filename
        df.to_csv(csv_path, index=False)
        print(f"[CSV] Saved Pareto front to {display_path(csv_path)}")
