"""Plot CSS KPI tables produced by the pure cycle-level KPI layer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pydac.post_processing.kpi.cycle_kpis import (
    DAC_SPECIES,
    CycleKPIs,
    EquivalentWorkConditions,
)
from pydac.post_processing.metrics.cycle_metrics import STEP_KEY_TO_CYCLE_STEP


O2_PPM_LIMIT = 40.0
PPM_LOG_FLOOR = 1e-12

STEP_DISPLAY_NAMES = {
    "startup": "Startup",
    "adsorption": "Adsorption",
    "blowdown": "Blowdown",
    "preheating": "Preheating",
    "wall purge": "Wall purge",
    "steam purge": "Steam purge",
    "desorption": "Desorption",
    "pressurization": "Pressurization",
    "cooldown": "Cooldown",
}

MASS_COLORS = {
    "CO2": "#1f77b4",
    "H2O": "#2ca02c",
    "O2": "#9467bd",
    "N2": "#d62728",
}

ENERGY_COLORS = {
    "Compression": "#FBC15E",
    "Vacuum": "#8EBA42",
    "Jacket": "#E24A33",
    "Steam": "#B276B2",
    "Condenser": "#348ABD",
}

UTILITY_COLORS = {
    "Compression": "#FBC15E",
    "Vacuum": "#8EBA42",
    "Jacket heating": "#E24A33",
    "Steam heating": "#B276B2",
    "Jacket cooling": "#64B5CD",
    "Condenser cooling": "#348ABD",
}


class KPIPlotting:
    """Render KPI figures without loading solver results or report files."""

    def __init__(self, cycle_kpis, output_folder=None):
        if not isinstance(cycle_kpis, CycleKPIs):
            raise TypeError("cycle_kpis must be a CycleKPIs instance.")
        self.kpis = cycle_kpis
        self.output_folder = (
            None if output_folder is None else Path(output_folder)
        )

    @property
    def cycle_label(self):
        return self.kpis.cycle_data.metrics.cycle_label

    # Shared plot appearance and file naming
    @staticmethod
    def _set_plot_style():
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.sans-serif": ["Liberation Sans"],
                "axes.labelsize": 15,
                "axes.titlesize": 17,
                "axes.linewidth": 1.2,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "xtick.major.width": 1.2,
                "ytick.major.width": 1.2,
                "legend.fontsize": 12,
                "figure.dpi": 120,
                "savefig.dpi": 300,
            }
        )

    @staticmethod
    def _filename_suffix_from_species(species):
        return "_".join(str(item).strip().lower() for item in species)

    @staticmethod
    def _compute_axis_limits(
        positive_max,
        negative_min,
        top_fraction=0.20,
        bottom_fraction=0.15,
    ):
        top = float(max(positive_max, 0.0))
        bottom = float(min(negative_min, 0.0))

        top_pad = max(top_fraction * top, 1e-6 if top > 0 else 1.0)
        bottom_pad = max(
            bottom_fraction * abs(bottom),
            1e-6 if bottom < 0 else 0.1 * max(top, 1.0),
        )
        top_limit = top + top_pad
        bottom_limit = bottom - bottom_pad
        if abs(top_limit - bottom_limit) < 1e-12:
            top_limit += 1.0
            bottom_limit -= 1.0
        return bottom_limit, top_limit

    def _output_path(self, stem, fmt):
        if self.output_folder is None:
            raise ValueError("output_folder is required to save KPI figures.")
        format_token = str(fmt).strip().lower().lstrip(".")
        if not format_token:
            raise ValueError("fmt must identify a non-empty figure format.")
        format_folder = self.output_folder / format_token
        format_folder.mkdir(parents=True, exist_ok=True)
        return format_folder / f"{self.cycle_label}_{stem}.{format_token}"

    @staticmethod
    def _finish_figure(fig, *, show):
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _save_and_finish(self, fig, stem, fmt, *, save, show):
        out_path = None
        if save:
            out_path = self._output_path(stem, fmt)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[Saved] {out_path}")
        self._finish_figure(fig, show=show)
        return None if out_path is None else str(out_path)

    @staticmethod
    def _step_tick_labels(index):
        return [
            STEP_DISPLAY_NAMES.get(step, str(step)).replace(" ", "\n")
            for step in index
        ]

    @staticmethod
    def _annotate_bars(ax, bars, *, fmt="{:.3g}"):
        for bar in bars:
            height = float(bar.get_height())
            if not np.isfinite(height):
                continue
            ax.annotate(
                fmt.format(height),
                (bar.get_x() + 0.5 * bar.get_width(), height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    @staticmethod
    def _plot_stacked_utility_table(ax, table, *, ylabel, title, hatch_share=None):
        """Stack one utility table.

        `hatch_share` maps a column to the part of it to draw hatched, in the same
        units, inside that column's own segment: the bar keeps its total height and
        only says how the duty divides.  It is used for the condenser, most of whose
        duty is rejected to ambient at no work cost.
        """
        hatch_share = {} if hatch_share is None else hatch_share
        x = np.arange(len(table.index))
        bottom = np.zeros(len(table.index))
        for column in table.columns:
            values = table[column].to_numpy(dtype=float)
            color = UTILITY_COLORS.get(column, "gray")
            if column in hatch_share:
                free = np.clip(
                    np.asarray(hatch_share[column], dtype=float), 0.0, values
                )
                ax.bar(
                    x,
                    values - free,
                    bottom=bottom,
                    label=f"{column} (chiller)",
                    color=color,
                    edgecolor="black",
                    linewidth=0.8,
                )
                ax.bar(
                    x,
                    free,
                    bottom=bottom + values - free,
                    label=f"{column} (free, to ambient)",
                    color=color,
                    alpha=0.45,
                    hatch="//",
                    edgecolor="black",
                    linewidth=0.8,
                )
            else:
                ax.bar(
                    x,
                    values,
                    bottom=bottom,
                    label=column,
                    color=color,
                    edgecolor="black",
                    linewidth=0.8,
                )
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels(KPIPlotting._step_tick_labels(table.index))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(
            frameon=False,
            ncol=min(len(table.columns) + len(hatch_share), 4),
        )

    # Per-step mass-change figures
    def plot_css_mass_changes(
        self,
        species=None,
        basis="state",
        unit="g",
        save=True,
        show=False,
        fmt="png",
        filename_stem=None,
        title=None,
    ):
        self._set_plot_style()
        table = self.kpis.mass_change_table(species=species, basis=basis)
        if table.empty:
            print("[KPIPlotting] No mass-change KPIs available to plot.")
            return None
        if save and self.output_folder is None:
            raise ValueError("output_folder is required to save KPI figures.")

        unit_token = str(unit).strip().lower()
        factor = 1000.0 if unit_token == "g" else 1.0
        ylabel = f"Mass change [{'g' if unit_token == 'g' else 'kg'}]"
        table_plot = factor * table

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(table_plot.index))
        species_tokens = tuple(table.columns)
        width = 0.8 / max(len(species_tokens), 1)

        for index, species_token in enumerate(species_tokens):
            offset = (index - 0.5 * (len(species_tokens) - 1)) * width
            ax.bar(
                x + offset,
                table_plot[species_token].to_numpy(dtype=float),
                width=width,
                label=species_token,
                color=MASS_COLORS.get(species_token, "gray"),
                edgecolor="black",
                linewidth=0.8,
            )

        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                STEP_DISPLAY_NAMES.get(step, step).replace(" ", "\n")
                for step in table_plot.index
            ]
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title or "CSS Mass Change Across Steps")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(frameon=False, ncol=min(len(species_tokens), 4))
        fig.tight_layout()

        out_path = None
        if save:
            stem = filename_stem or (
                "css_mass_changes_"
                f"{self._filename_suffix_from_species(species_tokens)}"
            )
            out_path = self._output_path(stem, fmt)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[Saved] {out_path}")

        self._finish_figure(fig, show=show)
        return None if out_path is None else str(out_path)

    def plot_css_mass_change_panels(
        self,
        unit="g",
        save=True,
        show=False,
        fmt="png",
    ):
        self.kpis._require_species(
            ("CO2", "H2O", "O2", "N2"), "DAC mass-change panels"
        )
        return {
            "co2_h2o": self.plot_css_mass_changes(
                species=("CO2", "H2O"),
                unit=unit,
                save=save,
                show=show,
                fmt=fmt,
                filename_stem="css_mass_changes_co2_h2o",
                title="CSS Mass Change Across Steps: CO2 and H2O",
            ),
            "n2_o2": self.plot_css_mass_changes(
                species=("N2", "O2"),
                unit=unit,
                save=save,
                show=show,
                fmt=fmt,
                filename_stem="css_mass_changes_n2_o2",
                title="CSS Mass Change Across Steps: N2 and O2",
            ),
        }

    # Final residual gas inventory across cycle steps
    def plot_css_residual_gas_ppm(
        self,
        species=None,
        save=True,
        show=False,
        fmt="png",
        filename_stem=None,
        title=None,
    ):
        self._set_plot_style()
        table = self.kpis.residual_gas_ppm_by_step(species=species)
        if table.empty:
            print("[KPIPlotting] No residual gas-phase ppm data available to plot.")
            return None
        if save and self.output_folder is None:
            raise ValueError("output_folder is required to save KPI figures.")

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(table.index))
        species_tokens = tuple(table.columns)
        width = 0.8 / max(len(species_tokens), 1)
        plotted_values = []

        for index, species_token in enumerate(species_tokens):
            offset = (index - 0.5 * (len(species_tokens) - 1)) * width
            values = table[species_token].to_numpy(dtype=float)
            positive_values = values[np.isfinite(values) & (values > 0.0)]
            if positive_values.size > 0:
                display_values = np.where(values > 0.0, values, np.nan)
                plotted_values.append(positive_values)
            else:
                display_values = np.full_like(values, np.nan, dtype=float)
                plotted_values.append(
                    np.array([PPM_LOG_FLOOR], dtype=float)
                )
            ax.bar(
                x + offset,
                display_values,
                width=width,
                label=species_token,
                color=MASS_COLORS.get(species_token, "gray"),
                edgecolor="black",
                linewidth=0.8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                STEP_DISPLAY_NAMES.get(step, step).replace(" ", "\n")
                for step in table.index
            ]
        )
        ax.set_ylabel("Residual gas-phase content [ppmmol]")
        ax.set_title(title or "CSS Residual Gas-Phase Composition Across Steps")
        ax.set_yscale("log")
        positive_arrays = [
            np.ravel(values)
            for values in plotted_values
            if np.size(values) > 0
        ]
        positive_concat = (
            np.concatenate(positive_arrays)
            if positive_arrays
            else np.array([PPM_LOG_FLOOR])
        )
        y_min = max(PPM_LOG_FLOOR, 0.2 * float(np.min(positive_concat)))
        y_max = max(float(np.max(positive_concat)), O2_PPM_LIMIT)
        ax.set_ylim(y_min, max(1.25 * y_max, y_min * 10.0))
        if "O2" in species_tokens:
            ax.axhline(
                O2_PPM_LIMIT,
                color="black",
                linestyle="--",
                linewidth=1.3,
                label=r"O$_2$ spec = 40 ppmmol",
            )
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.06),
                ncol=min(len(labels), 4),
                frameon=False,
            )
            fig.subplots_adjust(bottom=0.24)

        out_path = None
        if save:
            stem = filename_stem or (
                "css_residual_gas_ppm_"
                f"{self._filename_suffix_from_species(species_tokens)}"
            )
            out_path = self._output_path(stem, fmt)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[Saved] {out_path}")

        self._finish_figure(fig, show=show)
        return None if out_path is None else str(out_path)

    # Per-step energy contribution figures
    def plot_css_energy_breakdown(
        self,
        specific=False,
        equivalent=False,
        absolute=False,
        T_L=298.15,
        T_H=373.15,
        T_cond=278.15,
        eta_turb=0.75,
        eta_refr=0.5,
        save=True,
        show=False,
        fmt="png",
    ):
        self._set_plot_style()
        conditions = EquivalentWorkConditions(
            T_low=T_L,
            T_high=T_H,
            T_condensation=T_cond,
            turbine_efficiency=eta_turb,
            refrigeration_efficiency=eta_refr,
        )
        if equivalent:
            table = self.kpis.equivalent_work_breakdown(
                specific=specific,
                conditions=conditions,
            )
        else:
            table = self.kpis.energy_breakdown(
                specific=specific,
                equivalent=False,
                absolute=absolute,
            )
        if table.empty:
            print("[KPIPlotting] No energy KPIs available to plot.")
            return None
        if save and self.output_folder is None:
            raise ValueError("output_folder is required to save KPI figures.")

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(table.index))
        bottom_positive = np.zeros(len(table.index))
        bottom_negative = np.zeros(len(table.index))

        for column in table.columns:
            values = table[column].to_numpy(dtype=float)
            positive = np.clip(values, 0.0, None)
            negative = np.clip(values, None, 0.0)
            color = ENERGY_COLORS.get(column, "gray")

            ax.bar(
                x,
                positive,
                bottom=bottom_positive,
                label=column,
                color=color,
                edgecolor="black",
                linewidth=0.8,
            )
            ax.bar(
                x,
                negative,
                bottom=bottom_negative,
                color=color,
                edgecolor="black",
                linewidth=0.8,
            )
            bottom_positive += positive
            bottom_negative += negative

        if specific and equivalent:
            ylabel = "Specific equivalent work [kWh_eq/kgCO2]"
            title = "CSS Equivalent Work Breakdown Across Steps"
            stem = "css_equivalent_work_breakdown_specific"
        elif specific:
            ylabel = "Specific energy [MJ/kgCO2]"
            title = "CSS Specific Energy Breakdown Across Steps"
            stem = "css_energy_breakdown_specific"
        elif equivalent:
            ylabel = "Equivalent work [kWh_eq]"
            title = "CSS Equivalent Work Breakdown Across Steps"
            stem = "css_equivalent_work_breakdown"
        else:
            ylabel = "Energy [MJ]"
            title = "CSS Energy Breakdown Across Steps"
            stem = "css_energy_breakdown"

        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                STEP_DISPLAY_NAMES.get(step, step).replace(" ", "\n")
                for step in table.index
            ]
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        positive_max = (
            float(bottom_positive.max()) if len(bottom_positive) else 0.0
        )
        negative_min = (
            float(bottom_negative.min()) if len(bottom_negative) else 0.0
        )
        y_min, y_max = self._compute_axis_limits(
            positive_max,
            negative_min,
            top_fraction=0.28,
            bottom_fraction=0.18,
        )
        ax.set_ylim(y_min, y_max)
        ax.legend(
            frameon=False,
            ncol=min(len(table.columns), 5),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

        out_path = None
        if save:
            out_path = self._output_path(stem, fmt)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[Saved] {out_path}")

        self._finish_figure(fig, show=show)
        return None if out_path is None else str(out_path)

    # Direct electrical and thermal utility demand
    def plot_css_electrical_energy_breakdown(
        self,
        *,
        save=True,
        show=False,
        fmt="pdf",
    ):
        self.kpis._require_species(("CO2",), "CO2-specific energy plot")
        self._set_plot_style()
        raw = self.kpis.electrical_energy_breakdown(specific=False)
        specific = self.kpis.electrical_energy_breakdown(specific=True)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        self._plot_stacked_utility_table(
            axes[0],
            raw,
            ylabel="Electrical demand [kWh/cycle]",
            title="Direct Electrical Demand by Step",
        )
        self._plot_stacked_utility_table(
            axes[1],
            specific,
            ylabel="Specific electrical demand [kWh/kgCO2]",
            title="CO2-Specific Direct Electrical Demand by Step",
        )
        fig.tight_layout()
        return self._save_and_finish(
            fig,
            "css_electrical_energy_breakdown",
            fmt,
            save=save,
            show=show,
        )

    def plot_css_thermal_energy_breakdown(
        self,
        *,
        save=True,
        show=False,
        fmt="pdf",
    ):
        self.kpis._require_species(("CO2",), "CO2-specific energy plot")
        self._set_plot_style()
        raw = self.kpis.thermal_energy_breakdown(specific=False)
        specific = self.kpis.thermal_energy_breakdown(specific=True)
        # The condenser bar is the whole duty; the hatched part of it is rejected above
        # ambient on cooling water and costs no work.  Only the solid part is charged.
        raw_split = self.kpis.condenser_cooling_split(specific=False)
        specific_split = self.kpis.condenser_cooling_split(specific=True)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        self._plot_stacked_utility_table(
            axes[0],
            raw,
            ylabel="Thermal utility demand [MJ/cycle]",
            title="Thermal Heating and Cooling Utilities by Step",
            hatch_share={
                "Condenser cooling": raw_split["Condenser to ambient"].to_numpy()
            },
        )
        self._plot_stacked_utility_table(
            axes[1],
            specific,
            ylabel="Specific thermal demand [MJ/kgCO2]",
            title="CO2-Specific Thermal Utilities by Step",
            hatch_share={
                "Condenser cooling": specific_split["Condenser to ambient"].to_numpy()
            },
        )
        fig.tight_layout()
        return self._save_and_finish(
            fig,
            "css_thermal_energy_breakdown",
            fmt,
            save=save,
            show=show,
        )

    # Adsorption-normalized material release diagnostics
    def plot_css_mass_changes_percent_of_adsorption(
        self,
        *,
        save=True,
        show=False,
        fmt="pdf",
    ):
        self.kpis._require_species(
            ("CO2", "H2O"), "DAC adsorption-normalized mass plot"
        )
        self._set_plot_style()
        table = self.kpis.mass_change_percent_of_adsorption(
            species=("CO2", "H2O")
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(table.index))
        width = 0.36
        for offset, species in ((-0.5, "CO2"), (0.5, "H2O")):
            ax.bar(
                x + offset * width,
                table[species].to_numpy(dtype=float),
                width=width,
                label=species,
                color=MASS_COLORS[species],
                edgecolor="black",
                linewidth=0.8,
            )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.axhline(100.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(self._step_tick_labels(table.index))
        ax.set_ylabel("State mass change [% of adsorption uptake]")
        ax.set_title("CO2 and H2O Change Relative to Adsorption Uptake")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(frameon=False)
        fig.tight_layout()
        return self._save_and_finish(
            fig,
            "css_mass_changes_percent_of_adsorption",
            fmt,
            save=save,
            show=show,
        )

    # Compact process-performance dashboard
    def plot_css_process_kpis_overview(
        self,
        *,
        T_L=298.15,
        T_H=373.15,
        T_cond=278.15,
        eta_turb=0.75,
        eta_refr=0.5,
        save=True,
        show=False,
        fmt="pdf",
    ):
        self.kpis._require_species(
            ("CO2", "H2O", "O2", "N2"), "DAC process KPI overview"
        )
        self._set_plot_style()
        conditions = EquivalentWorkConditions(
            T_low=T_L,
            T_high=T_H,
            T_condensation=T_cond,
            turbine_efficiency=eta_turb,
            refrigeration_efficiency=eta_refr,
        )
        purity = self.kpis.purity()
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        axes[0, 0].axis("off")
        axes[0, 0].set_title("Performance")
        axes[0, 0].text(
            0.5,
            0.64,
            f"Productivity\n{self.kpis.productivity():.3f} kgCO2/m3/day",
            ha="center",
            va="center",
            fontsize=14,
        )
        axes[0, 0].text(
            0.5,
            0.28,
            "Equivalent work\n"
            f"{self.kpis.equivalent_work(conditions=conditions) / 3.6:.3f} "
            "kWh_eq/kgCO2",
            ha="center",
            va="center",
            fontsize=14,
        )

        quality_labels = ("CO2 recovery", "Wet purity", "Dry purity")
        quality_values = (
            100.0 * self.kpis.recovery("CO2"),
            100.0 * purity["purity_wet"],
            100.0 * purity["purity_dry"],
        )
        bars = axes[0, 1].bar(
            quality_labels,
            quality_values,
            color=("#4C72B0", "#55A868", "#2A9D8F"),
            edgecolor="black",
        )
        axes[0, 1].set_ylabel("[%]")
        axes[0, 1].set_title("Recovery and Product Purity")
        axes[0, 1].tick_params(axis="x", rotation=20)
        axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)
        self._annotate_bars(axes[0, 1], bars, fmt="{:.3f}")

        capacity_values = (
            self.kpis.cyclic_working_capacity("CO2"),
            self.kpis.cyclic_working_capacity("H2O"),
        )
        bars = axes[0, 2].bar(
            ("CO2", "H2O"),
            capacity_values,
            color=(MASS_COLORS["CO2"], MASS_COLORS["H2O"]),
            edgecolor="black",
        )
        axes[0, 2].set_ylabel("Cyclic working capacity [mol/kg adsorbent]")
        axes[0, 2].set_title("Cyclic Working Capacity")
        axes[0, 2].grid(axis="y", linestyle="--", alpha=0.5)
        self._annotate_bars(axes[0, 2], bars)

        ratio_labels = ("Adsorbed", "Desorbed", "Recovered")
        ratio_values = (
            self.kpis.step_h2o_to_co2_mass_ratio("adsorption"),
            self.kpis.regeneration_h2o_to_co2_mass_ratio(),
            self.kpis.h2o_to_co2_ratio("mass"),
        )
        bars = axes[1, 0].bar(
            ratio_labels,
            ratio_values,
            color=("#4C72B0", "#DD8452", "#55A868"),
            edgecolor="black",
        )
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_ylabel("H2O/CO2 mass ratio [kg/kg]")
        axes[1, 0].set_title("Water-to-CO2 Ratios")
        axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)
        self._annotate_bars(axes[1, 0], bars)

        steam_ratio = self.kpis.steam_to_co2_ratio()
        if steam_ratio is None:
            axes[1, 1].axis("off")
            axes[1, 1].text(
                0.5,
                0.5,
                "Steam/CO2\nnot applicable",
                ha="center",
                va="center",
                fontsize=14,
            )
        else:
            bars = axes[1, 1].bar(
                ("Steam/CO2",),
                (steam_ratio,),
                color=("#B276B2",),
                edgecolor="black",
            )
            axes[1, 1].set_ylabel("Steam/CO2 [kg/kgCO2]")
            axes[1, 1].set_title("Steam Demand")
            axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)
            self._annotate_bars(axes[1, 1], bars)

        impurity_values = np.array(
            [purity["oxygen_ppm"], purity["nitrogen_ppm"]], dtype=float
        )
        impurity_values[np.isclose(impurity_values, 0.0, atol=1e-9)] = 0.0
        bars = axes[1, 2].bar(
            ("O2", "N2"),
            impurity_values,
            color=(MASS_COLORS["O2"], MASS_COLORS["N2"]),
            edgecolor="black",
        )
        axes[1, 2].axhline(
            O2_PPM_LIMIT,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="O2 spec = 40 ppmmol",
        )
        axes[1, 2].set_ylabel("Product content [ppmmol]")
        axes[1, 2].set_title("Product O2 and N2")
        axes[1, 2].grid(axis="y", linestyle="--", alpha=0.5)
        axes[1, 2].legend(frameon=False)
        axes[1, 2].set_ylim(
            0.0,
            1.20 * max(O2_PPM_LIMIT, float(np.max(impurity_values)), 1.0),
        )
        self._annotate_bars(axes[1, 2], bars)

        fig.suptitle("CSS Process KPI Overview", fontsize=19)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        return self._save_and_finish(
            fig,
            "css_process_kpis_overview",
            fmt,
            save=save,
            show=show,
        )

    # Recovered product composition after condenser and vacuum discharge
    def plot_css_product_composition(
        self,
        *,
        save=True,
        show=False,
        fmt="pdf",
    ):
        self.kpis._require_species(
            ("CO2", "H2O", "O2", "N2"), "DAC product-composition plot"
        )
        self._set_plot_style()
        table = self.kpis.product_composition().set_index("Species")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        major_species = ("CO2", "H2O")
        major_values = table.loc[
            list(major_species), "Wet mole fraction [%mol]"
        ].to_numpy(dtype=float)
        bars = axes[0].bar(
            major_species,
            major_values,
            color=(MASS_COLORS["CO2"], MASS_COLORS["H2O"]),
            edgecolor="black",
        )
        axes[0].set_ylabel("Wet product composition [%mol]")
        axes[0].set_title("Major Product Components")
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)
        self._annotate_bars(axes[0], bars, fmt="{:.4g}")

        trace_species = ("O2", "N2")
        trace_values = table.loc[
            list(trace_species), "Wet content [ppmmol]"
        ].to_numpy(dtype=float, copy=True)
        trace_values[np.isclose(trace_values, 0.0, atol=1e-9)] = 0.0
        bars = axes[1].bar(
            trace_species,
            trace_values,
            color=(MASS_COLORS["O2"], MASS_COLORS["N2"]),
            edgecolor="black",
        )
        axes[1].axhline(
            O2_PPM_LIMIT,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="O2 spec = 40 ppmmol",
        )
        axes[1].set_ylabel("Wet product content [ppmmol]")
        axes[1].set_title("Trace Product Components")
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)
        axes[1].legend(frameon=False)
        axes[1].set_ylim(
            0.0,
            1.20 * max(O2_PPM_LIMIT, float(np.max(trace_values)), 1.0),
        )
        self._annotate_bars(axes[1], bars)

        fig.suptitle("Recovered Product Composition After Vacuum Discharge")
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
        return self._save_and_finish(
            fig,
            "css_product_composition",
            fmt,
            save=save,
            show=show,
        )

    def plot_css_equivalent_work_conditions(
        self,
        *,
        T_L=298.15,
        T_H=373.15,
        T_cond=278.15,
        eta_turb=0.75,
        eta_refr=0.5,
        save=True,
        show=False,
        fmt="pdf",
    ):
        self._set_plot_style()
        conditions = EquivalentWorkConditions(
            T_low=T_L,
            T_high=T_H,
            T_condensation=T_cond,
            turbine_efficiency=eta_turb,
            refrigeration_efficiency=eta_refr,
        )
        heat_to_work, refrigeration_cop = (
            self.kpis.thermodynamic_conversion_factors(conditions)
        )
        rows = (
            ("Low temperature", f"{conditions.T_low:.2f}", "K"),
            ("High temperature", f"{conditions.T_high:.2f}", "K"),
            ("Condensation temperature", f"{conditions.T_condensation:.2f}", "K"),
            ("Turbine efficiency", f"{conditions.turbine_efficiency:.4f}", "-"),
            (
                "Refrigeration efficiency",
                f"{conditions.refrigeration_efficiency:.4f}",
                "-",
            ),
            ("Heat-to-work factor", f"{heat_to_work:.6f}", "-"),
            ("Effective refrigeration COP", f"{refrigeration_cop:.6f}", "-"),
            *(
                row
                for step_key, (T_condenser, cop) in self.kpis.condenser_pricing(
                    conditions
                ).items()
                for row in (
                    (
                        f"Condenser temperature, {STEP_KEY_TO_CYCLE_STEP[step_key]}",
                        f"{T_condenser:.2f}",
                        "K",
                    ),
                    (f"Chiller COP, {STEP_KEY_TO_CYCLE_STEP[step_key]}", f"{cop:.6f}", "-"),
                )
            ),
        )
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=("Parameter", "Value", "Unit"),
            cellLoc="left",
            colLoc="left",
            loc="center",
            colWidths=(0.58, 0.24, 0.14),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.45)
        ax.set_title("Equivalent-Work Conversion Conditions", pad=18)
        fig.tight_layout()
        return self._save_and_finish(
            fig,
            "css_equivalent_work_conditions",
            fmt,
            save=save,
            show=show,
        )

    # KPI-only figure orchestration and legacy convenience aliases
    def plot_all_kpi_figures(
        self,
        fmt="pdf",
        mass_species=None,
        energy_specific=True,
        energy_equivalent=True,
        energy_absolute=False,
        T_L=298.15,
        T_H=373.15,
        T_cond=278.15,
        eta_turb=0.75,
        eta_refr=0.5,
        save=True,
        show=False,
    ):
        self.kpis._require_species(("CO2",), "KPI plot suite")
        available = set(self.kpis.available_species)

        def needs(name, species, plot):
            """Draw ``plot`` only if the mixture has ``species``; say why otherwise."""
            missing = [item for item in species if item not in available]
            if missing:
                print(f"KPI figure '{name}' skipped: the mixture has no {', '.join(missing)}.")
                return None
            return plot()

        if mass_species is None and set(DAC_SPECIES) <= available:
            mass_plots = self.plot_css_mass_change_panels(
                save=save,
                show=show,
                fmt=fmt,
            )
        elif mass_species is None:
            mass_plots = self.plot_css_mass_changes(
                species=self.kpis.available_species,
                save=save,
                show=show,
                fmt=fmt,
            )
        else:
            species_tokens = tuple(
                str(item).strip().upper() for item in mass_species
            )
            mass_plots = self.plot_css_mass_changes(
                species=species_tokens,
                save=save,
                show=show,
                fmt=fmt,
            )

        return {
            "mass_changes": mass_plots,
            "residual_gas_ppm": (
                self.plot_css_residual_gas_ppm(
                    species=self.kpis.present_species(("N2", "O2")),
                    save=save,
                    show=show,
                    fmt=fmt,
                )
                if self.kpis.present_species(("N2", "O2"))
                else needs("residual gas ppm", ("N2", "O2"), lambda: None)
            ),
            "energy_breakdown": self.plot_css_energy_breakdown(
                specific=energy_specific,
                equivalent=energy_equivalent,
                absolute=energy_absolute,
                T_L=T_L,
                T_H=T_H,
                T_cond=T_cond,
                eta_turb=eta_turb,
                eta_refr=eta_refr,
                save=save,
                show=show,
                fmt=fmt,
            ),
            "electrical_energy_breakdown": (
                self.plot_css_electrical_energy_breakdown(
                    save=save,
                    show=show,
                    fmt=fmt,
                )
            ),
            "thermal_energy_breakdown": self.plot_css_thermal_energy_breakdown(
                save=save,
                show=show,
                fmt=fmt,
            ),
            "mass_changes_percent_of_adsorption": needs(
                "mass changes, percent of adsorption",
                ("CO2", "H2O"),
                lambda: self.plot_css_mass_changes_percent_of_adsorption(
                    save=save,
                    show=show,
                    fmt=fmt,
                ),
            ),
            "process_kpis_overview": needs(
                "process KPI overview",
                DAC_SPECIES,
                lambda: self.plot_css_process_kpis_overview(
                    T_L=T_L,
                    T_H=T_H,
                    T_cond=T_cond,
                    eta_turb=eta_turb,
                    eta_refr=eta_refr,
                    save=save,
                    show=show,
                    fmt=fmt,
                ),
            ),
            "product_composition": needs(
                "product composition",
                DAC_SPECIES,
                lambda: self.plot_css_product_composition(
                    save=save,
                    show=show,
                    fmt=fmt,
                ),
            ),
            "equivalent_work_conditions": (
                self.plot_css_equivalent_work_conditions(
                    T_L=T_L,
                    T_H=T_H,
                    T_cond=T_cond,
                    eta_turb=eta_turb,
                    eta_refr=eta_refr,
                    save=save,
                    show=show,
                    fmt=fmt,
                )
            ),
        }

    def plot_masses(self, species=None):
        if species is None:
            species = self.kpis.available_species
        species_tokens = tuple(
            str(item).strip().upper() for item in species
        )
        if species_tokens == ("CO2", "H2O", "O2", "N2"):
            return self.plot_css_mass_change_panels(
                unit="g",
                save=True,
                show=False,
                fmt="png",
            )
        return self.plot_css_mass_changes(
            species=species_tokens,
            unit="g",
            save=True,
            show=False,
            fmt="png",
        )

    def plot_equivalent_specific_energy_contributions(
        self,
        T_L=288.15,
        T_H=373.15,
        T_cond=278.15,
        eta_turb=0.75,
        eta_refr=0.5,
        plot_absolute=True,
    ):
        return self.plot_css_energy_breakdown(
            specific=True,
            equivalent=True,
            absolute=plot_absolute,
            T_L=T_L,
            T_H=T_H,
            T_cond=T_cond,
            eta_turb=eta_turb,
            eta_refr=eta_refr,
            save=True,
            show=False,
            fmt="png",
        )
