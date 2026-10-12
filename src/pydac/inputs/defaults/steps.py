"""Reference TVSA cycle-step defaults."""

from __future__ import annotations

from typing import Any

from pydac.inputs.defaults._rules import (
    Rule,
    fraction,
    nonempty_string,
    nonnegative_number,
    positive_number,
    species_keys,
)


#: The startup composition field that sets the initial solid instead of
#: equilibrating it with the initial gas; every step declares it, startup reads it.
INITIAL_SOLID_LOADINGS_FIELD = "initial_solid_loadings_mol_per_kg"


def cycle_defaults() -> dict[str, Any]:
    """Return the complete ten-step reference cycle mapping."""
    return {
        "steps": {
            "startup": {
                "composition": {
                    "co2_feed_mole_fraction": 4e-4,
                    "relative_humidity": 0.5,
                    "co2_initial_mole_fraction": 0.0,
                    "initial_relative_humidity": 0.0,
                    "o2_feed_mole_fraction": None,
                    "n2_feed_mole_fraction": None,
                    "o2_initial_mole_fraction": None,
                    "n2_initial_mole_fraction": None,
                    "initial_solid_loadings_mol_per_kg": None,
                },
                "flow": {
                    "volumetric_flow_m3_s": 1e-3,
                    "steam_mass_flow_kg_s": None,
                    "blowdown_valve_conductance": None,
                    "pressurization_valve_conductance": None,
                },
                "thermal": {
                    "feed_temperature_K": 298.15,
                    "initial_temperature_K": 298.15,
                    "jacket_temperature_K": 298.15,
                    "ambient_temperature_K": None,
                    "external_htc_W_m2K": 1000.0,
                },
                "pressure": {
                    "initial_pressure_Pa": 101325.0,
                    "outlet_pressure_Pa": 101325.0,
                    "feed_pressure_Pa": 101325.0,
                    "ambient_pressure_Pa": 101325.0,
                },
                "horizon": {
                    "nominal_time_s": 39000.0,
                    "event_max_time_s": 43200.0,
                },
                "termination": {
                    "mode": "time",
                    "metric": "avg_bed_loading_fraction_CO2",
                    "direction": "above",
                    "threshold_value": 0.95,
                    "threshold_source": None,
                    "threshold_offset": 0.0,
                    "label": "Average bed CO2 loading reached 95% of the reference capacity",
                    "stop_on_low_pressure": False,
                    "low_pressure_stop_Pa": None,
                    "stop_on_pressure_plateau": False,
                    "pressure_plateau_rate_tolerance_nd": None,
                    "pressure_plateau_arm_below_Pa": None,
                    "pressure_plateau_arm_above_Pa": None,
                },
                "outlet": {
                    "collect_product": False,
                    "condenser": False,
                    "condenser_temperature_K": 278.15,
                },
            },
            "ads": {
                "composition": {
                    "co2_feed_mole_fraction": 4e-4,
                    "relative_humidity": 0.5,
                    "co2_initial_mole_fraction": None,
                    "initial_relative_humidity": None,
                    "o2_feed_mole_fraction": None,
                    "n2_feed_mole_fraction": None,
                    "o2_initial_mole_fraction": None,
                    "n2_initial_mole_fraction": None,
                    "initial_solid_loadings_mol_per_kg": None,
                },
                "flow": {
                    "volumetric_flow_m3_s": 1e-3,
                    "steam_mass_flow_kg_s": None,
                    "blowdown_valve_conductance": None,
                    "pressurization_valve_conductance": None,
                },
                "thermal": {
                    "feed_temperature_K": 298.15,
                    "initial_temperature_K": None,
                    "jacket_temperature_K": 298.15,
                    "ambient_temperature_K": None,
                    "external_htc_W_m2K": 1000.0,
                },
                "pressure": {
                    "initial_pressure_Pa": None,
                    "outlet_pressure_Pa": 101325.0,
                    "feed_pressure_Pa": 101325.0,
                    "ambient_pressure_Pa": 101325.0,
                },
                "horizon": {
                    "nominal_time_s": 39000.0,
                    "event_max_time_s": 43200.0,
                },
                "termination": {
                    "mode": "time",
                    "metric": "avg_bed_loading_fraction_CO2",
                    "direction": "above",
                    "threshold_value": 0.95,
                    "threshold_source": None,
                    "threshold_offset": 0.0,
                    "label": "Average bed CO2 loading reached 95% of the reference capacity",
                    "stop_on_low_pressure": False,
                    "low_pressure_stop_Pa": None,
                    "stop_on_pressure_plateau": False,
                    "pressure_plateau_rate_tolerance_nd": None,
                    "pressure_plateau_arm_below_Pa": None,
                    "pressure_plateau_arm_above_Pa": None,
                },
                "outlet": {
                    "collect_product": False,
                    "condenser": False,
                    "condenser_temperature_K": 278.15,
                },
            },
            "blow": {
                "composition": {
                    "co2_feed_mole_fraction": 0.0,
                    "relative_humidity": 0.0,
                    "co2_initial_mole_fraction": None,
                    "initial_relative_humidity": None,
                    "o2_feed_mole_fraction": 0.0,
                    "n2_feed_mole_fraction": 0.0,
                    "o2_initial_mole_fraction": None,
                    "n2_initial_mole_fraction": None,
                    "initial_solid_loadings_mol_per_kg": None,
                },
                "flow": {
                    "volumetric_flow_m3_s": 0.0,
                    "steam_mass_flow_kg_s": None,
                    "blowdown_valve_conductance": 0.01,
                    "pressurization_valve_conductance": None,
                },
                "thermal": {
                    "feed_temperature_K": 298.15,
                    "initial_temperature_K": None,
                    "jacket_temperature_K": 298.15,
                    "ambient_temperature_K": None,
                    "external_htc_W_m2K": 1000.0,
                },
                "pressure": {
                    "initial_pressure_Pa": None,
                    "outlet_pressure_Pa": 20000.0,
                    "feed_pressure_Pa": 20000.0,
                    "ambient_pressure_Pa": 101325.0,
                },
                "horizon": {
                    "nominal_time_s": 180.0,
                    "event_max_time_s": 900.0,
                },
                "termination": {
                    "mode": "event",
                    "metric": "max_internal_pressure",
                    "direction": "below",
                    "threshold_value": None,
                    "threshold_source": "p_out",
                    "threshold_offset": 25.0,
                    "label": "Maximum internal pressure reached the target vacuum level",
                    "stop_on_low_pressure": False,
                    "low_pressure_stop_Pa": None,
                    "stop_on_pressure_plateau": True,
                    "pressure_plateau_rate_tolerance_nd": 1e-5,
                    "pressure_plateau_arm_below_Pa": 50000.0,
                    "pressure_plateau_arm_above_Pa": None,
                },
                "outlet": {
                    "collect_product": False,
                    "condenser": False,
                    "condenser_temperature_K": 278.15,
                },
            },
            "heat": {
                "composition": {
                    "co2_feed_mole_fraction": 0.0,
                    "relative_humidity": 0.0,
                    "co2_initial_mole_fraction": None,
                    "initial_relative_humidity": None,
                    "o2_feed_mole_fraction": 0.0,
                    "n2_feed_mole_fraction": 0.0,
                    "o2_initial_mole_fraction": None,
                    "n2_initial_mole_fraction": None,
                    "initial_solid_loadings_mol_per_kg": None,
                },
                "flow": {
                    "volumetric_flow_m3_s": 0.0,
                    "steam_mass_flow_kg_s": None,
                    "blowdown_valve_conductance": None,
                    "pressurization_valve_conductance": None,
                },
                "thermal": {
                    "feed_temperature_K": 373.15,
                    "initial_temperature_K": None,
                    "jacket_temperature_K": 373.15,
                    "ambient_temperature_K": None,
                    "external_htc_W_m2K": 1000.0,
                },
                "pressure": {
                    "initial_pressure_Pa": None,
                    "outlet_pressure_Pa": 20000.0,
                    "feed_pressure_Pa": 20000.0,
                    "ambient_pressure_Pa": 101325.0,
                },
                "horizon": {
                    "nominal_time_s": 9000.0,
                    "event_max_time_s": 21600.0,
                },
                "termination": {
                    "mode": "time",
                    "metric": "min_bed_temperature",
                    "direction": "above",
                    "threshold_value": None,
                    "threshold_source": "T_cond",
                    "threshold_offset": 0.0,
                    "label": "Minimum bed temperature exceeded the steam condensation temperature",
                    "stop_on_low_pressure": False,
                    "low_pressure_stop_Pa": None,
                    "stop_on_pressure_plateau": False,
                    "pressure_plateau_rate_tolerance_nd": None,
                    "pressure_plateau_arm_below_Pa": None,
                    "pressure_plateau_arm_above_Pa": None,
                },
                "outlet": {
                    "collect_product": False,
                    "condenser": False,
                    "condenser_temperature_K": 278.15,
                },
            },
            "w_purge": {
                "composition": {
                    "co2_feed_mole_fraction": 0.0,
                    "relative_humidity": 0.0,
                    "co2_initial_mole_fraction": None,
                    "initial_relative_humidity": None,
                    "o2_feed_mole_fraction": 0.0,
                    "n2_feed_mole_fraction": 0.0,
                    "o2_initial_mole_fraction": None,
                    "n2_initial_mole_fraction": None,
                    "initial_solid_loadings_mol_per_kg": None,
                },
                "flow": {
                    "volumetric_flow_m3_s": 0.0,
                    "steam_mass_flow_kg_s": None,
                    "blowdown_valve_conductance": None,
                    "pressurization_valve_conductance": None,
                },
                "thermal": {
                    "feed_temperature_K": 373.15,
                    "initial_temperature_K": None,
                    "jacket_temperature_K": 373.15,
                    "ambient_temperature_K": None,
                    "external_htc_W_m2K": 1000.0,
                },
                "pressure": {
                    "initial_pressure_Pa": None,
                    "outlet_pressure_Pa": 20000.0,
                    "feed_pressure_Pa": 20000.0,
                    "ambient_pressure_Pa": 101325.0,
                },
                "horizon": {
                    "nominal_time_s": 900.0,
                    "event_max_time_s": 9000,
                },
                "termination": {
                    "mode": "event",
                    "metric": "max_bed_oxygen_mole_fraction",
                    "direction": "below",
                    "threshold_value": 1e-7,
                    "threshold_source": None,
                    "threshold_offset": 0.0,
                    "label": "Maximum bed oxygen mole fraction dropped below 0.1 ppm",
                    "stop_on_low_pressure": False,
                    "low_pressure_stop_Pa": None,
                    "stop_on_pressure_plateau": False,
                    "pressure_plateau_rate_tolerance_nd": None,
                    "pressure_plateau_arm_below_Pa": None,
                    "pressure_plateau_arm_above_Pa": None,
                },
                "outlet": {
                    "collect_product": False,
                    "condenser": False,
                    "condenser_temperature_K": 278.15,
                },
            },
            "s_purge": {
                "composition": {
                    "co2_feed_mole_fraction": 0.0,
                    "relative_humidity": 1.0,
                    "co2_initial_mole_fraction": None,
                    "initial_relative_humidity": None,
                    "o2_feed_mole_fraction": 0.0,
                    "n2_feed_mole_fraction": 0.0,
                    "o2_initial_mole_fraction": None,
                    "n2_initial_mole_fraction": None,
                    "initial_solid_loadings_mol_per_kg": None,
                },
                "flow": {
                    "volumetric_flow_m3_s": None,
                    "steam_mass_flow_kg_s": 1e-3,
                    "blowdown_valve_conductance": None,
                    "pressurization_valve_conductance": None,
                },
                "thermal": {
                    "feed_temperature_K": 373.15,
                    "initial_temperature_K": None,
                    "jacket_temperature_K": 373.15,
                    "ambient_temperature_K": 298.15,
                    "external_htc_W_m2K": 1000.0,
                },
                "pressure": {
                    "initial_pressure_Pa": None,
                    "outlet_pressure_Pa": 20000.0,
                    "feed_pressure_Pa": 20000.0,
                    "ambient_pressure_Pa": 101325.0,
                },
                "horizon": {
                    "nominal_time_s": 300,
                    "event_max_time_s": 600,
                },
                "termination": {
                    "mode": "event",
                    "metric": "max_bed_oxygen_mole_fraction",
                    "direction": "below",
                    "threshold_value": 1e-7,
                    "threshold_source": None,
                    "threshold_offset": 0.0,
                    "label": "Maximum bed oxygen mole fraction dropped below 0.1 ppm",
                    "stop_on_low_pressure": False,
                    "low_pressure_stop_Pa": None,
                    "stop_on_pressure_plateau": False,
                    "pressure_plateau_rate_tolerance_nd": None,
                    "pressure_plateau_arm_below_Pa": None,
                    "pressure_plateau_arm_above_Pa": None,
                },
                "outlet": {
                    "collect_product": True,
                    "condenser": True,
                    "condenser_temperature_K": 278.15,
                },
            },
            "s_des": {
                "composition": {
                    "co2_feed_mole_fraction": 0.0,
                    "relative_humidity": 1.0,
                    "co2_initial_mole_fraction": None,
                    "initial_relative_humidity": None,
                    "o2_feed_mole_fraction": 0.0,
                    "n2_feed_mole_fraction": 0.0,
                    "o2_initial_mole_fraction": None,
                    "n2_initial_mole_fraction": None,
                    "initial_solid_loadings_mol_per_kg": None,
                },
                "flow": {
                    "volumetric_flow_m3_s": None,
                    "steam_mass_flow_kg_s": 1e-3,
                    "blowdown_valve_conductance": None,
                    "pressurization_valve_conductance": None,
                },
                "thermal": {
                    "feed_temperature_K": 373.15,
                    "initial_temperature_K": None,
                    "jacket_temperature_K": 373.15,
                    "ambient_temperature_K": 298.15,
                    "external_htc_W_m2K": 1000.0,
                },
                "pressure": {
                    "initial_pressure_Pa": None,
                    "outlet_pressure_Pa": 20000.0,
                    "feed_pressure_Pa": 20000.0,
                    "ambient_pressure_Pa": 101325.0,
                },
                "horizon": {
                    "nominal_time_s": 3600.0,
                    "event_max_time_s": 10800.0,
                },
                "termination": {
                    "mode": "time",
                    "metric": "avg_bed_loading_fraction_CO2",
                    "direction": "below",
                    "threshold_value": 0.05,
                    "threshold_source": None,
                    "threshold_offset": 0.0,
                    "label": "Average bed CO2 loading dropped below 5% of the reference capacity",
                    "stop_on_low_pressure": False,
                    "low_pressure_stop_Pa": None,
                    "stop_on_pressure_plateau": False,
                    "pressure_plateau_rate_tolerance_nd": None,
                    "pressure_plateau_arm_below_Pa": None,
                    "pressure_plateau_arm_above_Pa": None,
                },
                "outlet": {
                    "collect_product": True,
                    "condenser": True,
                    "condenser_temperature_K": 278.15,
                },
            },
            "w_des": {
                "composition": {
                    "co2_feed_mole_fraction": 0.0,
                    "relative_humidity": 0.0,
                    "co2_initial_mole_fraction": None,
                    "initial_relative_humidity": None,
                    "o2_feed_mole_fraction": 0.0,
                    "n2_feed_mole_fraction": 0.0,
                    "o2_initial_mole_fraction": None,
                    "n2_initial_mole_fraction": None,
                    "initial_solid_loadings_mol_per_kg": None,
                },
                "flow": {
                    "volumetric_flow_m3_s": 0.0,
                    "steam_mass_flow_kg_s": None,
                    "blowdown_valve_conductance": None,
                    "pressurization_valve_conductance": None,
                },
                "thermal": {
                    "feed_temperature_K": 373.15,
                    "initial_temperature_K": None,
                    "jacket_temperature_K": 373.15,
                    "ambient_temperature_K": 298.15,
                    "external_htc_W_m2K": 1000.0,
                },
                "pressure": {
                    "initial_pressure_Pa": None,
                    "outlet_pressure_Pa": 20000.0,
                    "feed_pressure_Pa": 20000.0,
                    "ambient_pressure_Pa": 101325.0,
                },
                "horizon": {
                    "nominal_time_s": 86400.0,
                    "event_max_time_s": 86400.0,
                },
                "termination": {
                    "mode": "event",
                    "metric": "avg_bed_loading_fraction_CO2",
                    "direction": "below",
                    "threshold_value": 0.05,
                    "threshold_source": None,
                    "threshold_offset": 0.0,
                    "label": "Average bed CO2 loading dropped below 5% of the reference capacity",
                    "stop_on_low_pressure": False,
                    "low_pressure_stop_Pa": None,
                    "stop_on_pressure_plateau": False,
                    "pressure_plateau_rate_tolerance_nd": None,
                    "pressure_plateau_arm_below_Pa": None,
                    "pressure_plateau_arm_above_Pa": None,
                },
                "outlet": {
                    "collect_product": True,
                    "condenser": True,
                    "condenser_temperature_K": 278.15,
                },
            },
            "cool": {
                "composition": {
                    "co2_feed_mole_fraction": 0.0,
                    "relative_humidity": 0.0,
                    "co2_initial_mole_fraction": None,
                    "initial_relative_humidity": None,
                    "o2_feed_mole_fraction": 0.0,
                    "n2_feed_mole_fraction": 0.0,
                    "o2_initial_mole_fraction": None,
                    "n2_initial_mole_fraction": None,
                    "initial_solid_loadings_mol_per_kg": None,
                },
                "flow": {
                    "volumetric_flow_m3_s": 0.0,
                    "steam_mass_flow_kg_s": None,
                    "blowdown_valve_conductance": None,
                    "pressurization_valve_conductance": None,
                },
                "thermal": {
                    "feed_temperature_K": 298.15,
                    "initial_temperature_K": None,
                    "jacket_temperature_K": 298.15,
                    "ambient_temperature_K": None,
                    "external_htc_W_m2K": 1000.0,
                },
                "pressure": {
                    "initial_pressure_Pa": None,
                    "outlet_pressure_Pa": 20000.0,
                    "feed_pressure_Pa": 20000.0,
                    "ambient_pressure_Pa": 101325.0,
                },
                "horizon": {
                    "nominal_time_s": 9000.0,
                    "event_max_time_s": 10800.0,
                },
                "termination": {
                    "mode": "event",
                    "metric": "max_bed_temperature",
                    "direction": "below",
                    "threshold_value": 323.15,
                    "threshold_source": None,
                    "threshold_offset": 0.0,
                    "label": "Maximum bed temperature dropped below the cooldown target",
                    "stop_on_low_pressure": True,
                    "low_pressure_stop_Pa": 100.0,
                    "stop_on_pressure_plateau": False,
                    "pressure_plateau_rate_tolerance_nd": None,
                    "pressure_plateau_arm_below_Pa": None,
                    "pressure_plateau_arm_above_Pa": None,
                },
                "outlet": {
                    "collect_product": False,
                    "condenser": False,
                    "condenser_temperature_K": 278.15,
                },
            },
            "press": {
                "composition": {
                    "co2_feed_mole_fraction": 4e-4,
                    "relative_humidity": 0.5,
                    "co2_initial_mole_fraction": None,
                    "initial_relative_humidity": None,
                    "o2_feed_mole_fraction": None,
                    "n2_feed_mole_fraction": None,
                    "o2_initial_mole_fraction": None,
                    "n2_initial_mole_fraction": None,
                    "initial_solid_loadings_mol_per_kg": None,
                },
                "flow": {
                    "volumetric_flow_m3_s": 1e-3,
                    "steam_mass_flow_kg_s": None,
                    "blowdown_valve_conductance": None,
                    "pressurization_valve_conductance": 0.01,
                },
                "thermal": {
                    "feed_temperature_K": 298.15,
                    "initial_temperature_K": None,
                    "jacket_temperature_K": 298.15,
                    "ambient_temperature_K": None,
                    "external_htc_W_m2K": 1000.0,
                },
                "pressure": {
                    "initial_pressure_Pa": None,
                    "outlet_pressure_Pa": 101325.0,
                    "feed_pressure_Pa": 101325.0,
                    "ambient_pressure_Pa": 101325.0,
                },
                "horizon": {
                    "nominal_time_s": 300.0,
                    "event_max_time_s": 900.0,
                },
                "termination": {
                    "mode": "event",
                    "metric": "min_internal_pressure",
                    "direction": "above",
                    "threshold_value": None,
                    "threshold_source": "p_feed",
                    "threshold_offset": -25.0,
                    "label": "Minimum internal pressure reached the feed pressure",
                    "stop_on_low_pressure": False,
                    "low_pressure_stop_Pa": None,
                    "stop_on_pressure_plateau": True,
                    "pressure_plateau_rate_tolerance_nd": 1e-5,
                    "pressure_plateau_arm_below_Pa": None,
                    "pressure_plateau_arm_above_Pa": 50000.0,
                },
                "outlet": {
                    "collect_product": False,
                    "condenser": False,
                    "condenser_temperature_K": 278.15,
                },
            },
        }
    }


def _lowercase(value: Any, _name: str) -> Any:
    return value.lower() if isinstance(value, str) else value


def step_rules():
    """Return constraints shared by every named step in the default cycle."""
    from pydac.utils.tvsa.termination import EVENT_METRICS, THRESHOLD_SOURCES

    rules = {}
    steps = cycle_defaults()["steps"]
    for step_name, step in steps.items():
        prefix = ("simulation", "cycle", "steps", step_name)
        for name in step["composition"]:
            rules[prefix + ("composition", name)] = fraction(nullable=True)
        rules[prefix + ("composition", INITIAL_SOLID_LOADINGS_FIELD)] = Rule(
            kind="object",
            nullable=True,
            opaque=True,
            normalize=species_keys,
            schema={
                "anyOf": [
                    {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": {"type": "number", "minimum": 0},
                    },
                    {"type": "null"},
                ]
            },
            description=(
                "Initial solid loading [mol/kg] of each adsorbate, keyed by "
                "species name (any case: co2 = CO2), the same in the whole bed; every species of "
                "mixture.adsorbates needs its own value, others are refused; "
                "startup only. null equilibrates the solid with the initial gas."
            ),
        )
        for name in step["flow"]:
            rules[prefix + ("flow", name)] = nonnegative_number(nullable=True)
        for valve, boundary in (("blowdown", "outlet"), ("pressurization", "inlet")):
            rules[prefix + ("flow", f"{valve}_valve_conductance")] = nonnegative_number(
                f"Valve conductance alpha [1/m] of the {boundary} relation "
                "dp/dn = alpha (p_set - p), n the outward normal; null is an ideal valve.",
                nullable=True,
            )
        for group in ("thermal", "pressure"):
            for name in step[group]:
                rules[prefix + (group, name)] = positive_number(nullable=True)
        rules[prefix + ("horizon", "nominal_time_s")] = positive_number()
        rules[prefix + ("horizon", "event_max_time_s")] = positive_number(
            nullable=True
        )
        rules[prefix + ("termination", "mode")] = Rule(
            kind="string",
            choices=("time", "event"),
            normalize=_lowercase,
        )
        rules[prefix + ("termination", "metric")] = Rule(
            kind="string", choices=EVENT_METRICS
        )
        rules[prefix + ("termination", "direction")] = Rule(
            kind="string",
            choices=("above", "below"),
            normalize=_lowercase,
        )
        for name in (
            "threshold_value",
            "low_pressure_stop_Pa",
            "pressure_plateau_rate_tolerance_nd",
            "pressure_plateau_arm_below_Pa",
            "pressure_plateau_arm_above_Pa",
        ):
            rules[prefix + ("termination", name)] = positive_number(nullable=True)
        rules[prefix + ("termination", "threshold_source")] = Rule(
            kind="string", nullable=True, choices=THRESHOLD_SOURCES
        )
        rules[prefix + ("termination", "threshold_offset")] = Rule(kind="number")
        rules[prefix + ("termination", "label")] = nonempty_string(nullable=True)
        rules[prefix + ("outlet", "collect_product")] = Rule(
            kind="boolean",
            description=(
                "Send the step outlet to the product line (condenser, vacuum pump, "
                "vacuum-discharge flash); false vents it."
            ),
        )
        rules[prefix + ("outlet", "condenser")] = Rule(
            kind="boolean",
            description="Put a condenser before the vacuum pump of a collected step.",
        )
        rules[prefix + ("outlet", "condenser_temperature_K")] = positive_number(
            "Temperature the condenser cools the collected stream to."
        )
    return rules


__all__ = ["INITIAL_SOLID_LOADINGS_FIELD", "cycle_defaults", "step_rules"]
