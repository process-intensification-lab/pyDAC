"""CSS KPI report assembly and persistence from pure cycle-level KPIs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pydac.post_processing.metrics.cycle_metrics import STEP_KEY_TO_CYCLE_STEP

from pydac.post_processing.kpi.cycle_kpis import CycleKPIs, EquivalentWorkConditions, MJ_PER_KWH


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


class KPIReporting:
    """Build and save the current CSS reports without recomputing physical KPIs."""

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

    # Shared display conversions used throughout the report
    @staticmethod
    def _kg_to_g(value):
        return 1000.0 * float(value)

    @staticmethod
    def _k_to_c(value):
        return float(value) - 273.15

    @staticmethod
    def _round_numeric_table(table, decimals=3):
        rounded = table.copy()
        numeric_columns = rounded.select_dtypes(include=[np.number]).columns
        rounded[numeric_columns] = rounded[numeric_columns].round(decimals)
        return rounded

    @staticmethod
    def _with_uniform_unit(table, unit):
        annotated = table.copy()
        annotated["Unit"] = str(unit)
        return annotated

    @staticmethod
    def _equivalent_work_conditions(
        T_L,
        T_H,
        T_cond,
        eta_turb,
        eta_refr,
    ):
        return EquivalentWorkConditions(
            T_low=T_L,
            T_high=T_H,
            T_condensation=T_cond,
            turbine_efficiency=eta_turb,
            refrigeration_efficiency=eta_refr,
        )

    def _with_cycle_label(self, filename):
        return f"{self.cycle_label}_{filename}"

    # Headline KPI assembly
    def _collect_css_summary_metrics(
        self,
        species="CO2",
        T_L=298.15,
        T_H=373.15,
        T_cond=278.15,
        eta_turb=0.75,
        eta_refr=0.5,
    ):
        self.kpis._require_species(("CO2",), "CSS summary report")
        species_token = str(species).strip().upper()
        purity = self.kpis.purity()
        residual_ppm_table = self.kpis.residual_gas_ppm_by_step(
            species=self.kpis.present_species(("O2", "N2"))
        )
        mass_transfer = self.kpis.mass_transfer_summary(
            species=self.kpis.present_species(("CO2", "H2O"))
        )
        transfer_map = mass_transfer.set_index("Species").to_dict(orient="index")
        # A species the mixture lacks reports NaN: "not applicable", not zero.
        not_applicable = {
            "Adsorbed in adsorption [kg]": np.nan,
            "Desorbed in desorption [kg]": np.nan,
            "Recovered after vacuum discharge [kg]": np.nan,
        }
        transfer_map.setdefault("H2O", not_applicable)
        has_h2o = "H2O" in self.kpis.available_species
        preheating = self.kpis.preheating_temperatures()
        working_capacity_map = {
            "CO2": self.kpis.cyclic_working_capacity(species="CO2"),
            "H2O": self.kpis.cyclic_working_capacity(species="H2O") if has_h2o else np.nan,
        }
        selected_working_capacity = working_capacity_map.get(species_token)
        if selected_working_capacity is None:
            selected_working_capacity = self.kpis.cyclic_working_capacity(
                species=species_token
            )

        co2_recovered = float(
            transfer_map["CO2"]["Recovered after vacuum discharge [kg]"]
        )
        steam_mass = self.kpis.steam_mass_used()
        conditions = self._equivalent_work_conditions(
            T_L,
            T_H,
            T_cond,
            eta_turb,
            eta_refr,
        )

        return {
            "species_token": species_token,
            "cycle_label": self.cycle_label,
            "cycle_time_min": self.kpis.cycle_time() / 60.0,
            "bed_volume_m3": self.kpis.context.value("bed.volume"),
            "solid_adsorbent_mass_g": self._kg_to_g(
                self.kpis.context.adsorbent_mass
            ),
            "co2_adsorbed_g": self._kg_to_g(
                transfer_map["CO2"]["Adsorbed in adsorption [kg]"]
            ),
            "co2_desorbed_g": self._kg_to_g(
                transfer_map["CO2"]["Desorbed in desorption [kg]"]
            ),
            "co2_recovered_g": self._kg_to_g(co2_recovered),
            "h2o_adsorbed_g": self._kg_to_g(
                transfer_map["H2O"]["Adsorbed in adsorption [kg]"]
            ),
            "h2o_desorbed_g": self._kg_to_g(
                transfer_map["H2O"]["Desorbed in desorption [kg]"]
            ),
            "h2o_recovered_g": self._kg_to_g(
                transfer_map["H2O"]["Recovered after vacuum discharge [kg]"]
            ),
            "saturation_after_adsorption_percent": self.kpis.step_saturation(
                "adsorption"
            ),
            "saturation_after_desorption_percent": self.kpis.step_saturation(
                self.kpis.last_desorption_step_key()
            ),
            "cyclic_working_capacity_mol_per_kg": selected_working_capacity,
            "co2_cyclic_working_capacity_mol_per_kg": working_capacity_map["CO2"],
            "h2o_cyclic_working_capacity_mol_per_kg": working_capacity_map["H2O"],
            "co2_recovery_percent": 100.0 * self.kpis.recovery(species="CO2"),
            "h2o_recovery_percent": (
                100.0 * self.kpis.recovery(species="H2O") if has_h2o else np.nan
            ),
            # Vented by every step whose outlet is not collected (preheating, purges,
            # blowdown by default): lost, not recovered, and the largest lever on
            # recovery in a cycle whose preheating sweeps the bed.
            # NaN, not zero, on a result that predates the record: "not measured" and
            # "vented nothing" must not read the same.
            "co2_vented_in_purge_g": (
                self._kg_to_g(self.kpis.vented_mass("CO2"))
                if self.kpis.has_vent_record("CO2")
                else float("nan")
            ),
            "co2_vented_percent_of_adsorbed": (
                100.0 * self.kpis.vented_fraction_of_adsorbed("CO2")
                if self.kpis.has_vent_record("CO2")
                else float("nan")
            ),
            "equivalent_work_MJ_per_kgCO2": self.kpis.equivalent_work(
                conditions=conditions
            ),
            "equivalent_work_kWh_eq_per_kgCO2": (
                self.kpis.equivalent_work(conditions=conditions) / MJ_PER_KWH
            ),
            "productivity_kgCO2_per_m3_day": self.kpis.productivity(),
            "purity_wet_percent": 100.0 * purity["purity_wet"],
            "purity_dry_percent": 100.0 * purity["purity_dry"],
            "oxygen_content_ppmmol": purity["oxygen_ppm"],
            "nitrogen_content_ppmmol": purity["nitrogen_ppm"],
            "h2o_to_co2_ratio_kg_per_kg": self.kpis.h2o_to_co2_ratio(
                basis="mass"
            ),
            "preheating_min_bed_temperature_degC": self._k_to_c(
                preheating["bed_minimum_K"]
            ),
            "preheating_steam_condensation_temperature_degC": (
                self._k_to_c(preheating["steam_condensation_K"])
                if preheating["steam_condensation_K"] is not None
                else np.nan
            ),
            "steam_used_g": self._kg_to_g(steam_mass),
            "steam_to_co2_ratio_kg_per_kgCO2": self.kpis.steam_to_co2_ratio(),
            "residual_gas_ppm_by_step": residual_ppm_table,
        }

    def get_css_summary(
        self,
        species="CO2",
        T_L=298.15,
        T_H=373.15,
        T_cond=278.15,
        eta_turb=0.75,
        eta_refr=0.5,
    ):
        metrics = self._collect_css_summary_metrics(
            species=species,
            T_L=T_L,
            T_H=T_H,
            T_cond=T_cond,
            eta_turb=eta_turb,
            eta_refr=eta_refr,
        )
        return {
            "saturation_after_adsorption_percent": metrics[
                "saturation_after_adsorption_percent"
            ],
            "saturation_after_desorption_percent": metrics[
                "saturation_after_desorption_percent"
            ],
            "cyclic_working_capacity_mol_per_kg": metrics[
                "cyclic_working_capacity_mol_per_kg"
            ],
            "co2_cyclic_working_capacity_mol_per_kg": metrics[
                "co2_cyclic_working_capacity_mol_per_kg"
            ],
            "h2o_cyclic_working_capacity_mol_per_kg": metrics[
                "h2o_cyclic_working_capacity_mol_per_kg"
            ],
            "equivalent_work_MJ_per_kgCO2": metrics[
                "equivalent_work_MJ_per_kgCO2"
            ],
            "recovery_fraction": 0.01 * metrics["co2_recovery_percent"],
            "purity_mole_fraction": 0.01 * metrics["purity_wet_percent"],
            "purity_dry_mole_fraction": 0.01 * metrics["purity_dry_percent"],
            "oxygen_content_ppmmol": metrics["oxygen_content_ppmmol"],
            "nitrogen_content_ppmmol": metrics["nitrogen_content_ppmmol"],
            "h2o_to_co2_ratio_kg_per_kg": metrics[
                "h2o_to_co2_ratio_kg_per_kg"
            ],
            "productivity_kgCO2_per_m3_day": metrics[
                "productivity_kgCO2_per_m3_day"
            ],
        }

    def get_css_summary_table(
        self,
        species="CO2",
        T_L=298.15,
        T_H=373.15,
        T_cond=278.15,
        eta_turb=0.75,
        eta_refr=0.5,
    ):
        metrics = self._collect_css_summary_metrics(
            species=species,
            T_L=T_L,
            T_H=T_H,
            T_cond=T_cond,
            eta_turb=eta_turb,
            eta_refr=eta_refr,
        )
        summary_rows = [
            {"Metric": "Cycle label", "Value": metrics["cycle_label"], "Unit": "-"},
            {"Metric": "Cycle time", "Value": metrics["cycle_time_min"], "Unit": "min"},
            {"Metric": "Bed volume", "Value": metrics["bed_volume_m3"], "Unit": "m3"},
            {
                "Metric": "Solid adsorbent mass",
                "Value": metrics["solid_adsorbent_mass_g"],
                "Unit": "g",
            },
            {
                "Metric": "CO2 adsorbed in adsorption",
                "Value": metrics["co2_adsorbed_g"],
                "Unit": "g",
            },
            {
                "Metric": "CO2 desorbed in desorption",
                "Value": metrics["co2_desorbed_g"],
                "Unit": "g",
            },
            {
                "Metric": "CO2 recovered after vacuum discharge",
                "Value": metrics["co2_recovered_g"],
                "Unit": "g",
            },
            {
                "Metric": "H2O adsorbed in adsorption",
                "Value": metrics["h2o_adsorbed_g"],
                "Unit": "g",
            },
            {
                "Metric": "H2O desorbed in desorption",
                "Value": metrics["h2o_desorbed_g"],
                "Unit": "g",
            },
            {
                "Metric": "H2O recovered after vacuum discharge",
                "Value": metrics["h2o_recovered_g"],
                "Unit": "g",
            },
            {
                "Metric": "Saturation after adsorption",
                "Value": metrics["saturation_after_adsorption_percent"],
                "Unit": "%",
            },
            {
                "Metric": "Saturation after desorption",
                "Value": metrics["saturation_after_desorption_percent"],
                "Unit": "%",
            },
            {
                "Metric": "CO2 cyclic working capacity",
                "Value": metrics["co2_cyclic_working_capacity_mol_per_kg"],
                "Unit": "mol/kg",
            },
            {
                "Metric": "H2O cyclic working capacity",
                "Value": metrics["h2o_cyclic_working_capacity_mol_per_kg"],
                "Unit": "mol/kg",
            },
            {"Metric": "CO2 recovery", "Value": metrics["co2_recovery_percent"], "Unit": "%"},
            {"Metric": "H2O recovery", "Value": metrics["h2o_recovery_percent"], "Unit": "%"},
            {
                "Metric": "Equivalent work",
                "Value": metrics["equivalent_work_kWh_eq_per_kgCO2"],
                "Unit": "kWh_eq/kgCO2",
            },
            {
                "Metric": "Productivity",
                "Value": metrics["productivity_kgCO2_per_m3_day"],
                "Unit": "kgCO2/m3/day",
            },
            {
                "Metric": "Purity (wet, mole basis)",
                "Value": metrics["purity_wet_percent"],
                "Unit": "%",
            },
            {
                "Metric": "Purity (dry, mole basis)",
                "Value": metrics["purity_dry_percent"],
                "Unit": "%",
            },
            {"Metric": "O2 content", "Value": metrics["oxygen_content_ppmmol"], "Unit": "ppmmol"},
            {"Metric": "N2 content", "Value": metrics["nitrogen_content_ppmmol"], "Unit": "ppmmol"},
            {
                "Metric": "H2O/CO2 co-desorption ratio",
                "Value": metrics["h2o_to_co2_ratio_kg_per_kg"],
                "Unit": "kg/kg",
            },
            {
                "Metric": "Preheating minimum bed temperature",
                "Value": metrics["preheating_min_bed_temperature_degC"],
                "Unit": "degC",
            },
            {
                "Metric": "Preheating steam condensation temperature",
                "Value": metrics[
                    "preheating_steam_condensation_temperature_degC"
                ],
                "Unit": "degC",
            },
        ]
        if metrics["steam_to_co2_ratio_kg_per_kgCO2"] is not None:
            summary_rows.extend(
                [
                    {
                        "Metric": "Steam used in desorption",
                        "Value": metrics["steam_used_g"],
                        "Unit": "g",
                    },
                    {
                        "Metric": "Steam/CO2 ratio",
                        "Value": metrics["steam_to_co2_ratio_kg_per_kgCO2"],
                        "Unit": "kg/kgCO2",
                    },
                ]
            )

        return self._round_numeric_table(pd.DataFrame(summary_rows), decimals=3)

    def get_process_kpis_table(
        self,
        T_L=298.15,
        T_H=373.15,
        T_cond=278.15,
        eta_turb=0.75,
        eta_refr=0.5,
    ):
        metrics = self._collect_css_summary_metrics(
            species="CO2",
            T_L=T_L,
            T_H=T_H,
            T_cond=T_cond,
            eta_turb=eta_turb,
            eta_refr=eta_refr,
        )
        rows = [
            {
                "KPI": "Productivity",
                "Value": metrics["productivity_kgCO2_per_m3_day"],
                "Unit": "kgCO2/m3/day",
                "Basis": "Recovered CO2 after vacuum discharge / bed volume / cycle time",
            },
            {
                "KPI": "Equivalent work",
                "Value": metrics["equivalent_work_kWh_eq_per_kgCO2"],
                "Unit": "kWh_eq/kgCO2",
                "Basis": "Total absolute work-equivalent demand / recovered CO2 after vacuum discharge",
            },
            {
                "KPI": "CO2 recovery",
                "Value": metrics["co2_recovery_percent"],
                "Unit": "%",
                "Basis": "Recovered CO2 after vacuum discharge / CO2 adsorbed on state basis",
            },
            {
                "KPI": "CO2 vented",
                "Value": metrics["co2_vented_in_purge_g"],
                "Unit": "g",
                "Basis": "Bed outlet of the steps not collected as product; vented, not recovered",
            },
            {
                "KPI": "CO2 vented",
                "Value": metrics["co2_vented_percent_of_adsorbed"],
                "Unit": "%",
                "Basis": "Vented CO2 / CO2 adsorbed on state basis",
            },
            {
                "KPI": "CO2 purity (wet)",
                "Value": metrics["purity_wet_percent"],
                "Unit": "%mol",
                "Basis": "Recovered product after vacuum discharge, including H2O",
            },
            {
                "KPI": "CO2 purity (dry)",
                "Value": metrics["purity_dry_percent"],
                "Unit": "%mol",
                "Basis": "Recovered product after vacuum discharge, excluding H2O",
            },
            {
                "KPI": "CO2 cyclic working capacity",
                "Value": metrics["co2_cyclic_working_capacity_mol_per_kg"],
                "Unit": "mol/kg_adsorbent",
                "Basis": "End-adsorption minus end-desorption bed inventory / adsorbent mass",
            },
            {
                "KPI": "H2O cyclic working capacity",
                "Value": metrics["h2o_cyclic_working_capacity_mol_per_kg"],
                "Unit": "mol/kg_adsorbent",
                "Basis": "End-adsorption minus end-desorption bed inventory / adsorbent mass",
            },
            {
                "KPI": "Steam/CO2 ratio",
                "Value": metrics["steam_to_co2_ratio_kg_per_kgCO2"],
                "Unit": "kg/kgCO2",
                "Basis": "Steam supplied / recovered CO2 after vacuum discharge",
            },
            {
                "KPI": "Adsorbed H2O/CO2 mass ratio",
                "Value": self.kpis.step_h2o_to_co2_mass_ratio("adsorption"),
                "Unit": "kg/kg",
                "Basis": "Absolute state delta H2O / CO2 during adsorption",
            },
            {
                "KPI": "Desorbed H2O/CO2 mass ratio",
                "Value": self.kpis.regeneration_h2o_to_co2_mass_ratio(),
                "Unit": "kg/kg",
                "Basis": "Summed absolute state delta H2O / CO2 over every product step",
            },
            {
                "KPI": "Recovered H2O/CO2 product mass ratio",
                "Value": metrics["h2o_to_co2_ratio_kg_per_kg"],
                "Unit": "kg/kg",
                "Basis": "Recovered H2O / recovered CO2 after vacuum discharge",
            },
            {
                "KPI": "O2 content in product",
                "Value": metrics["oxygen_content_ppmmol"],
                "Unit": "ppmmol",
                "Basis": "Recovered product after vacuum discharge, wet mole basis",
            },
            {
                "KPI": "N2 content in product",
                "Value": metrics["nitrogen_content_ppmmol"],
                "Unit": "ppmmol",
                "Basis": "Recovered product after vacuum discharge, wet mole basis",
            },
        ]
        return self._round_numeric_table(pd.DataFrame(rows), decimals=6)

    # Transform numerical CycleKPIs tables into report tables
    def _build_report_mass_transfer_summary(self, species=None):
        table = self.kpis.mass_transfer_summary(species=species).copy()
        mass_columns = [
            "Adsorbed in adsorption [kg]",
            "Desorbed in desorption [kg]",
            "Recovered after vacuum discharge [kg]",
        ]
        for column in mass_columns:
            table[column] = table[column].astype(float).map(self._kg_to_g)
        table = table.rename(
            columns={column: column.replace("[kg]", "[g]") for column in mass_columns}
        )
        return self._round_numeric_table(table, decimals=3)

    def _build_report_mass_change_table(self, species, basis="integrated"):
        table = self.kpis.mass_change_table(
            species=species, basis=basis
        ).astype(float) * 1000.0
        return table.round(3)

    def _build_report_mass_change_percent_of_adsorption(self):
        table = self.kpis.mass_change_percent_of_adsorption().round(3)
        table["Unit"] = "%"
        table["Reference"] = (
            "100 * state delta m(step) / abs(state delta m(adsorption))"
        )
        return table

    def _build_report_mass_change_comparison_table(self, species=None):
        if species is None:
            species = self.kpis.integrated_mass_species
        species_tokens = [str(item).strip().upper() for item in species]
        integrated = self.kpis.mass_change_table(
            species=species_tokens, basis="integrated"
        ).astype(float) * 1000.0
        state = self.kpis.mass_change_table(
            species=species_tokens, basis="state"
        ).astype(float) * 1000.0

        combined = pd.DataFrame(index=integrated.index)
        for species_token in species_tokens:
            combined[f"{species_token} (state)"] = state[species_token]
            combined[f"{species_token} (integrated)"] = integrated[species_token]
        combined.index.name = "Step"
        return combined.round(3)

    def _build_report_cycle_time_table(self):
        table = self.kpis.cycle_time_breakdown().copy()
        report = pd.DataFrame(
            {
                "Step": table["Step"],
                "Duration [min]": table["Duration [s]"].astype(float) / 60.0,
                "Cumulative time [min]": table["Cumulative time [s]"].astype(float) / 60.0,
                "Fraction [%]": 100.0 * table["Fraction [-]"].astype(float),
            }
        )
        return self._round_numeric_table(report, decimals=3)

    def _build_report_bed_closure_step_table(self):
        table = self.kpis.bed_closure_by_step().copy()
        kg_columns = [
            "Initial bed inventory [kg]",
            "Final bed inventory [kg]",
            "Delta m integrated [kg]",
            "Delta m state [kg]",
            "Closure gap [kg]",
        ]
        for column in kg_columns:
            table[column.replace("[kg]", "[g]")] = table[column].astype(float).map(
                self._kg_to_g
            )
        table = table.drop(columns=kg_columns)
        ordered = [
            "Step",
            "Species",
            "Initial bed inventory [g]",
            "Final bed inventory [g]",
            "Delta m integrated [g]",
            "Delta m state [g]",
            "Closure gap [g]",
            "Relative error [%]",
        ]
        return self._round_numeric_table(table[ordered], decimals=6)

    def _build_report_desorption_process_check_table(self):
        table = self.kpis.desorption_process_check().copy()
        kg_columns = [
            "Initial bed inventory [kg]",
            "External input [kg]",
            "Bed outlet before separation [kg]",
            "Condenser vapor outlet [kg]",
            "Condenser liquid removed [kg]",
            "Vacuum discharge vapor [kg]",
            "Vacuum liquid removed [kg]",
            "Final bed inventory [kg]",
            "Closure gap [kg]",
        ]
        for column in kg_columns:
            table[column.replace("[kg]", "[g]")] = table[column].astype(float).map(
                self._kg_to_g
            )
        table = table.drop(columns=kg_columns)
        ordered = [
            "Step",
            "Species",
            "Initial bed inventory [g]",
            "External input [g]",
            "Bed outlet before separation [g]",
            "Condenser vapor outlet [g]",
            "Condenser liquid removed [g]",
            "Vacuum discharge vapor [g]",
            "Vacuum liquid removed [g]",
            "Final bed inventory [g]",
            "Closure gap [g]",
            "Relative error [%]",
        ]
        report = self._round_numeric_table(table[ordered], decimals=6)
        for column in ("Closure gap [g]", "Relative error [%]"):
            report[column] = report[column].astype(float).map(
                lambda value: f"{value:.3f}"
            )
        return report

    def _build_report_saturation_by_step_table(self):
        rows = []
        for step_key in self.kpis.cycle_data.metrics.step_order:
            step = STEP_KEY_TO_CYCLE_STEP[step_key]
            rows.append(
                {
                    "Step": STEP_DISPLAY_NAMES.get(step, step.title()),
                    "Saturation [%]": self.kpis.step_saturation(step_key),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["Step", "Saturation [%]"])
        return self._round_numeric_table(pd.DataFrame(rows), decimals=3)

    def _build_report_residual_gas_ppm_table(self):
        table = self.kpis.residual_gas_ppm_by_step().copy()
        if table.empty:
            return pd.DataFrame(columns=["Step", *self.kpis.available_species])
        table = table.rename(
            index=lambda step: STEP_DISPLAY_NAMES.get(
                step, str(step).replace("_", " ").title()
            )
        )
        table.index.name = "Step"
        return table.round(3)

    def _build_report_equivalent_work_conditions(self, conditions):
        carnot_factor, refrigeration_cop = (
            self.kpis.thermodynamic_conversion_factors(conditions)
        )
        return pd.DataFrame(
            [
                {"Parameter": "Low temperature", "Value": conditions.T_low, "Unit": "K"},
                {"Parameter": "High temperature", "Value": conditions.T_high, "Unit": "K"},
                {
                    "Parameter": "Condensation temperature",
                    "Value": conditions.T_condensation,
                    "Unit": "K",
                },
                {
                    "Parameter": "Turbine efficiency",
                    "Value": conditions.turbine_efficiency,
                    "Unit": "-",
                },
                {
                    "Parameter": "Refrigeration efficiency",
                    "Value": conditions.refrigeration_efficiency,
                    "Unit": "-",
                },
                {
                    "Parameter": "Heat-to-work factor",
                    "Value": carnot_factor,
                    "Unit": "-",
                },
                {
                    "Parameter": "Effective refrigeration COP",
                    "Value": refrigeration_cop,
                    "Unit": "-",
                },
                *self._condenser_pricing_rows(conditions),
            ]
        )

    def _condenser_pricing_rows(self, conditions):
        """One temperature and one chiller COP row per step with a condenser."""
        rows = []
        for step_key, (T_condenser, cop) in self.kpis.condenser_pricing(conditions).items():
            step = STEP_KEY_TO_CYCLE_STEP[step_key]
            rows.append(
                {"Parameter": f"Condenser temperature, {step}", "Value": T_condenser, "Unit": "K"}
            )
            rows.append({"Parameter": f"Chiller COP, {step}", "Value": cop, "Unit": "-"})
        return rows

    def get_css_report_tables(
        self,
        species="CO2",
        T_L=298.15,
        T_H=373.15,
        T_cond=278.15,
        eta_turb=0.75,
        eta_refr=0.5,
    ):
        self.kpis._require_species(("CO2",), "CSS report suite")
        conditions = self._equivalent_work_conditions(
            T_L,
            T_H,
            T_cond,
            eta_turb,
            eta_refr,
        )
        return {
            "summary": self.get_css_summary_table(
                species=species,
                T_L=T_L,
                T_H=T_H,
                T_cond=T_cond,
                eta_turb=eta_turb,
                eta_refr=eta_refr,
            ),
            "process_kpis": self.get_process_kpis_table(
                T_L=T_L,
                T_H=T_H,
                T_cond=T_cond,
                eta_turb=eta_turb,
                eta_refr=eta_refr,
            ),
            "product_composition": self._round_numeric_table(
                self.kpis.product_composition(), decimals=6
            ),
            "step_durations": self._build_report_cycle_time_table(),
            "mass_transfer": self._with_uniform_unit(
                self._build_report_mass_transfer_summary(), "g"
            ),
            "mass_changes_state": self._with_uniform_unit(
                self._build_report_mass_change_table(
                    species=self.kpis.present_species(("CO2", "H2O", "N2", "O2")),
                    basis="state",
                ),
                "g",
            ),
            "mass_changes_state_percent_of_adsorption": (
                self._build_report_mass_change_percent_of_adsorption()
            ),
            "residual_gas_ppm_by_step": self._with_uniform_unit(
                self._build_report_residual_gas_ppm_table(), "ppmmol"
            ),
            "saturation_by_step": self._with_uniform_unit(
                self._build_report_saturation_by_step_table(), "%"
            ),
            "bed_closure_by_step": self._build_report_bed_closure_step_table(),
            "process_closure": self._build_report_desorption_process_check_table(),
            "electrical_energy_kWh": self._round_numeric_table(
                self._with_uniform_unit(
                    self.kpis.electrical_energy_breakdown(specific=False),
                    "kWh",
                ),
                decimals=6,
            ),
            "electrical_energy_specific_kWh_per_kgCO2": self._round_numeric_table(
                self._with_uniform_unit(
                    self.kpis.electrical_energy_breakdown(specific=True),
                    "kWh/kgCO2",
                ),
                decimals=6,
            ),
            "thermal_energy_MJ": self._round_numeric_table(
                self._with_uniform_unit(
                    self.kpis.thermal_energy_breakdown(specific=False), "MJ"
                ),
                decimals=6,
            ),
            "thermal_energy_specific_MJ_per_kgCO2": self._round_numeric_table(
                self._with_uniform_unit(
                    self.kpis.thermal_energy_breakdown(specific=True),
                    "MJ/kgCO2",
                ),
                decimals=6,
            ),
            "equivalent_work_specific_kWh_eq_per_kgCO2": self._round_numeric_table(
                self._with_uniform_unit(
                    self.kpis.equivalent_work_breakdown(
                        specific=True, conditions=conditions
                    ),
                    "kWh_eq/kgCO2",
                ),
                decimals=6,
            ),
            "equivalent_work_conditions": self._round_numeric_table(
                self._build_report_equivalent_work_conditions(conditions),
                decimals=6,
            ),
        }

    # Plain-text report formatting
    def _format_css_report_text(self, report_tables):
        lines = [
            "=" * 84,
            f"CSS KPI Summary - {self.cycle_label}".center(84),
            "=" * 84,
            "",
            report_tables["summary"].to_string(index=False),
            "",
            "Mass convention: reported mass changes use the state/inventory basis.",
            "Detailed mass, energy, and closure diagnostics are saved under kpis/csv/.",
        ]
        return "\n".join(lines)

    def print_css_summary_report(
        self,
        species="CO2",
        T_L=298.15,
        T_H=373.15,
        T_cond=278.15,
        eta_turb=0.75,
        eta_refr=0.5,
    ):
        report_tables = self.get_css_report_tables(
            species=species,
            T_L=T_L,
            T_H=T_H,
            T_cond=T_cond,
            eta_turb=eta_turb,
            eta_refr=eta_refr,
        )
        report_text = self._format_css_report_text(report_tables)
        print(report_text)
        return report_text

    # Filesystem persistence kept separate from report assembly
    def save_css_summary_report(
        self,
        species="CO2",
        T_L=298.15,
        T_H=373.15,
        T_cond=278.15,
        eta_turb=0.75,
        eta_refr=0.5,
        print_report=False,
    ):
        if self.output_folder is None:
            raise ValueError("output_folder is required to save CSS KPI reports.")

        report_tables = self.get_css_report_tables(
            species=species,
            T_L=T_L,
            T_H=T_H,
            T_cond=T_cond,
            eta_turb=eta_turb,
            eta_refr=eta_refr,
        )
        report_text = self._format_css_report_text(report_tables)
        if print_report:
            print(report_text)

        self.output_folder.mkdir(parents=True, exist_ok=True)
        csv_folder = self.output_folder / "csv"
        csv_folder.mkdir(parents=True, exist_ok=True)
        txt_path = self.output_folder / self._with_cycle_label("css_kpi_summary.txt")
        txt_path.write_text(report_text + "\n", encoding="utf-8")

        summary_table = report_tables["summary"]
        csv_path = csv_folder / self._with_cycle_label("css_kpi_summary.csv")
        summary_table.to_csv(csv_path, index=False)

        json_path = self.output_folder / self._with_cycle_label("css_kpi_summary.json")
        json_payload = {}
        for _, row in summary_table.iterrows():
            value = row["Value"]
            if isinstance(value, np.generic):
                value = value.item()
            json_payload[row["Metric"]] = {"value": value, "unit": row["Unit"]}
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(json_payload, handle, indent=2)

        extra_tables = {
            Path("css_step_durations.csv"): report_tables["step_durations"],
            Path("process/css_process_kpis.csv"): report_tables["process_kpis"],
            Path("process/css_product_composition.csv"): report_tables[
                "product_composition"
            ],
            Path("mass/css_mass_transfer_summary.csv"): report_tables[
                "mass_transfer"
            ],
            Path("mass/css_mass_changes_state.csv"): report_tables[
                "mass_changes_state"
            ],
            Path(
                "mass/css_mass_changes_state_percent_of_adsorption.csv"
            ): report_tables["mass_changes_state_percent_of_adsorption"],
            Path("mass/css_residual_gas_ppm_by_step.csv"): report_tables[
                "residual_gas_ppm_by_step"
            ],
            Path("mass/css_saturation_by_step.csv"): report_tables[
                "saturation_by_step"
            ],
            Path("closures/css_bed_closure_by_step.csv"): report_tables[
                "bed_closure_by_step"
            ],
            Path("closures/css_process_closure.csv"): report_tables[
                "process_closure"
            ],
            Path("energy/css_electrical_energy_breakdown_kWh.csv"): report_tables[
                "electrical_energy_kWh"
            ],
            Path(
                "energy/css_electrical_energy_breakdown_specific_kWh_per_kgCO2.csv"
            ): report_tables["electrical_energy_specific_kWh_per_kgCO2"],
            Path("energy/css_thermal_energy_breakdown_MJ.csv"): report_tables[
                "thermal_energy_MJ"
            ],
            Path(
                "energy/css_thermal_energy_breakdown_specific_MJ_per_kgCO2.csv"
            ): report_tables["thermal_energy_specific_MJ_per_kgCO2"],
            Path(
                "energy/css_equivalent_work_breakdown_specific_kWh_eq_per_kgCO2.csv"
            ): report_tables["equivalent_work_specific_kWh_eq_per_kgCO2"],
            Path("energy/css_equivalent_work_conditions.csv"): report_tables[
                "equivalent_work_conditions"
            ],
        }

        saved_paths = {
            "summary_txt": str(txt_path),
            "summary_csv": str(csv_path),
            "summary_json": str(json_path),
        }
        for relative_path, table in extra_tables.items():
            filename = self._with_cycle_label(relative_path.name)
            path = csv_folder / relative_path.parent / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=table.index.name is not None)
            saved_paths[str(relative_path)] = str(path)

        from pydac.utils.provenance import file_artifact, new_record, save_record
        provenance = new_record(
            "cycle_report", sources=self.kpis.cycle_data.provenance,
            execution={"species": species, "T_L": T_L, "T_H": T_H,
                       "T_cond": T_cond, "eta_turb": eta_turb, "eta_refr": eta_refr},
            artifacts=[file_artifact(path, name=str(Path(path).relative_to(self.output_folder)))
                       for path in saved_paths.values()],
        )
        save_record(self.output_folder, provenance,
                    pointer=self._with_cycle_label("report_provenance.json"))

        print(f"[Saved] {txt_path}")
        print(f"[Saved] {csv_path}")
        print(f"[Saved] {json_path}")
        return saved_paths
