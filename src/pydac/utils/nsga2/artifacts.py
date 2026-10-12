"""Persist stable NSGA-II campaign configuration artifacts."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from pydac.utils.flow_transforms import (
    STEAM_FEED_TEMPERATURE_K,
    bed_cross_section_area,
)


def save_pareto_bounds(run_config) -> None:
    active_bounds = {key: run_config.bounds[key] for key in run_config.var_names}
    dataframe = (
        pd.DataFrame.from_dict(
            active_bounds,
            orient="index",
            columns=["lower_bound", "upper_bound"],
        )
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    dataframe["space"] = "optimizer_velocity"
    dataframe.to_csv(
        os.path.join(run_config.base_folder, "pareto_bounds.csv"),
        index=False,
    )


def flow_transform_metadata(run_config) -> dict[str, Any]:
    inlet_area = bed_cross_section_area(
        tau_ar=run_config.tau_ar,
        bed_volume=run_config.bed_volume,
    )
    return {
        "optimizer_flow_basis": "superficial_velocity",
        "model_flow_basis": "vol_flowrate_and_m_dot_steam",
        "decision_variables": list(run_config.var_names),
        "model_operating_setting_keys": [
            "adsorption_time_s",
            "heating_time_s",
            "desorption_time_s",
            "low_pressure_Pa",
            "volumetric_flow_m3_s",
            "steam_mass_flow_kg_s",
        ],
        "tau_ar": run_config.tau_ar,
        "bed_volume": run_config.bed_volume,
        "bed_config": run_config.bed_config,
        "inlet_area_model": inlet_area,
        "steam_feed_temperature_K": STEAM_FEED_TEMPERATURE_K,
        "adsorption_feed_temperature_K": run_config.adsorption_T_feed,
        "adsorption_relative_humidity": run_config.adsorption_RH,
        "steam_density": (
            "rho = min(P_vac, P_sat_H2O(T_feed)) * MW_H2O / "
            "(R * T_feed)"
        ),
        "vol_flowrate_model": "v_feed * inlet_area_model",
        "m_dot_steam_model": (
            "v_steam * inlet_area_model * rho_steam_model(P_vac, T_feed)"
        ),
    }


def save_flow_transform_metadata(run_config) -> None:
    path = os.path.join(run_config.base_folder, "pareto_flow_transform.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(flow_transform_metadata(run_config), handle, indent=2)
