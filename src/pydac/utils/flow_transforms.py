"""
Flow-coordinate transforms used by NSGA-II campaigns.

The optimizer uses superficial velocities as decision variables. These helpers
convert them to the model inputs expected by the CSS runner:
`vol_flowrate` for air-fed steps and `m_dot_steam` for steam-assisted
desorption.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.constants import gas_constant as R_gas

from pydac.core.isotherm_models import water_saturation_pressure
from pydac.core.species_catalog import get_species_properties

from pydac.utils.aspect_ratios import get_L_and_R


STEAM_FEED_TEMPERATURE_K: float = 373.15
# Keep the historical helper name, but make the catalog the sole owner of the
# molecular weight used by the steam-density transform.
MW_H2O: float = get_species_properties("H2O").molecular_weight_kg_per_mol


def bed_cross_section_area(tau_ar: float, bed_volume: float | None = None) -> float:
    """Return the bed inlet cross-sectional area [m2]."""
    _length, radius = get_L_and_R(tau_ar=tau_ar, volume=bed_volume)
    if radius is None or radius <= 0.0:
        raise ValueError(
            f"Could not compute a positive bed radius for tau_ar={tau_ar}, "
            f"bed_volume={bed_volume}."
        )
    return float(np.pi * radius**2)


def water_saturation_pressure_iapws(T: float) -> float:
    """
    Compute H2O saturation pressure [Pa] from the relation the GAB water model
    uses, so a config-time or optimizer-time steam density and the simulation
    itself cannot be built on two different vapor pressures.
    """
    return float(water_saturation_pressure(float(T)))


def ideal_vapor_density(
    pressure: float,
    temperature_K: float,
    molar_mass_kg_mol: float,
) -> float:
    """Return the ideal-gas density [kg/m3] of a vapor-only stream."""
    return float(pressure) * float(molar_mass_kg_mol) / (R_gas * float(temperature_K))


def check_saturation_pressure_agrees(
    saturation_pressure: float,
    temperature_K: float,
    *,
    context: str,
    rtol: float = 1e-9,
) -> bool:
    """Warn when a model's own H2O EOS is not the correlation used before it exists.

    `config.derivations` turns `m_dot_steam` into `volumetric_flow_m3_s`, and the
    NSGA-II transforms turn a steam velocity into a mass flow -- both long before
    any equilibrium system is built, so both call `water_saturation_pressure`.
    Today every H2O-bearing equilibrium system resolves to that same relation, so
    this never fires; adding an H2O model with a different vapor pressure would
    otherwise make those config-time numbers describe a different stream than the
    one actually simulated. Returns whether the two agree.
    """
    reference = float(water_saturation_pressure(float(temperature_K)))
    resolved = float(saturation_pressure)
    if abs(resolved - reference) <= rtol * abs(reference):
        return True
    warnings.warn(
        f"{context}: the configured H2O equilibrium component reports a saturation "
        f"pressure of {resolved:.2f} Pa at {float(temperature_K):.2f} K, while "
        f"`isotherm_models.water_saturation_pressure` gives {reference:.2f} Pa. The "
        "steam flow already derived at configuration time used the latter, so the "
        "resolved `volumetric_flow_m3_s` describes a different stream than this step.",
        RuntimeWarning,
    )
    return False


def cap_pressure_at_saturation(
    pressure: float,
    saturation_pressure: float,
    *,
    temperature_K: float,
    context: str | None = None,
) -> float:
    """Return the pressure usable for a vapor-only density, capped at saturation.

    Above `saturation_pressure` the stream would be condensing, so the ideal-gas
    density is evaluated at saturation instead. Callers own the saturation
    pressure itself: the model passes the EOS of its configured H2O equilibrium
    component, while configuration-time helpers pass the IAPWS-1995 correlation
    above. Passing a `context` (the step name) turns a silent cap into a
    `RuntimeWarning`.
    """
    p_used = float(pressure)
    p_sat = float(saturation_pressure)
    if p_used <= p_sat:
        return p_used
    if context is not None:
        warnings.warn(
            f"{context} feed pressure ({p_used:.2f} Pa) exceeds H2O saturation pressure at "
            f"{float(temperature_K):.2f} K ({p_sat:.2f} Pa). Using saturation pressure for "
            "steam density estimate.",
            RuntimeWarning,
        )
    return p_sat


def steam_density_from_pressure(
    pressure: float,
    *,
    T_feed: float = STEAM_FEED_TEMPERATURE_K,
    cap_at_saturation: bool = True,
) -> tuple[float, float, float]:
    """
    Estimate inlet steam density [kg/m3] from pressure and feed temperature.

    Returns `(rho, pressure_used, p_sat)`. When `cap_at_saturation` is True,
    `pressure_used` is capped at the saturation pressure, matching the model's
    vapor-only density estimate.
    """
    p_feed = float(pressure)
    temperature = float(T_feed)
    if p_feed < 0.0:
        raise ValueError(f"Steam pressure must be non-negative. Received {pressure}.")
    if temperature <= 0.0:
        raise ValueError(f"Steam feed temperature must be > 0 K. Received {T_feed}.")

    p_sat = water_saturation_pressure_iapws(temperature)
    p_used = (
        cap_pressure_at_saturation(p_feed, p_sat, temperature_K=temperature)
        if cap_at_saturation
        else p_feed
    )
    rho = ideal_vapor_density(p_used, temperature, MW_H2O)
    return float(rho), float(p_used), float(p_sat)


def model_flow_inputs_from_velocities(
    *,
    v_feed: float,
    tau_ar: float,
    bed_volume: float | None,
    v_steam: float | None = None,
    P_vac: float | None = None,
    steam_feed_temperature: float = STEAM_FEED_TEMPERATURE_K,
) -> dict[str, float]:
    """
    Convert optimizer velocities to model flow inputs.

    `P_vac` is intentionally accepted per candidate because steam density, and
    therefore `m_dot_steam`, depends on the candidate vacuum pressure.
    """
    inlet_area = bed_cross_section_area(tau_ar=tau_ar, bed_volume=bed_volume)
    outputs: dict[str, float] = {
        "inlet_area_model": inlet_area,
        "v_feed": float(v_feed),
        "vol_flowrate_model": float(v_feed) * inlet_area,
    }

    if v_steam is not None:
        if P_vac is None:
            raise ValueError("P_vac is required when converting v_steam to m_dot_steam.")
        rho, p_used, p_sat = steam_density_from_pressure(
            P_vac,
            T_feed=steam_feed_temperature,
            cap_at_saturation=True,
        )
        outputs.update(
            v_steam=float(v_steam),
            rho_steam_model=rho,
            p_steam_density_model=p_used,
            p_steam_saturation_model=p_sat,
            m_dot_steam_model=float(v_steam) * inlet_area * rho,
        )

    return outputs
