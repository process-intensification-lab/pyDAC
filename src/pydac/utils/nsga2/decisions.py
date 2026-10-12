"""Decision-vector contracts and model-flow transforms for NSGA-II."""

from __future__ import annotations

from typing import Sequence

from pydac.utils.flow_transforms import model_flow_inputs_from_velocities


BASE_VAR_NAMES: tuple[str, ...] = (
    "t_ads",
    "t_heat",
    "t_des",
    "P_vac",
    "v_feed",
)
STEAM_VAR_NAME = "v_steam"
COOLING_TARGET_VAR_NAME = "T_cool_target"


def resolve_var_names(
    is_steam_assisted: bool,
    *,
    optimize_cooling_target: bool = False,
) -> tuple[str, ...]:
    """Return the decision variables of one campaign, in vector order.

    Optional variables are *appended*, never interleaved, so a five- or
    six-variable campaign keeps the exact vector it has always had and its
    persisted decision columns stay comparable.
    """
    names = BASE_VAR_NAMES
    if is_steam_assisted:
        names = (*names, STEAM_VAR_NAME)
    if optimize_cooling_target:
        names = (*names, COOLING_TARGET_VAR_NAME)
    return names


def supported_var_name_sets(is_steam_assisted: bool) -> tuple[tuple[str, ...], ...]:
    """Return every vector shape a campaign may declare for this cycle."""
    return (
        resolve_var_names(is_steam_assisted),
        resolve_var_names(is_steam_assisted, optimize_cooling_target=True),
    )


def validate_var_names(
    var_names: Sequence[str],
    *,
    is_steam_assisted: bool,
) -> tuple[str, ...]:
    """Check a campaign-declared vector against the supported shapes.

    A campaign states which variables it optimizes; this refuses anything the
    evaluation path could not map, rather than letting an unknown name reach
    the model as a silently ignored knob.
    """
    names = tuple(str(name) for name in var_names)
    supported = supported_var_name_sets(is_steam_assisted)
    if names not in supported:
        raise ValueError(
            f"Unsupported decision variables {list(names)} for a "
            f"{'steam-assisted' if is_steam_assisted else 'dry'} cycle. "
            f"Supported: {[list(option) for option in supported]}."
        )
    return names


def decision_vector_to_model_inputs(
    decision_vector: Sequence[float],
    *,
    is_steam_assisted: bool,
    tau_ar: float,
    bed_volume: float | None,
    var_names: Sequence[str] | None = None,
) -> tuple[dict[str, float | None], dict[str, float]]:
    """Map an optimizer vector to named settings for the shared CSS runner."""
    names = (
        resolve_var_names(is_steam_assisted)
        if var_names is None
        else validate_var_names(var_names, is_steam_assisted=is_steam_assisted)
    )
    if len(decision_vector) != len(names):
        raise ValueError(
            f"Decision vector length mismatch: expected {len(names)}, "
            f"received {len(decision_vector)}."
        )
    values = {
        name: float(value) for name, value in zip(names, decision_vector)
    }
    v_steam = values.get(STEAM_VAR_NAME)

    flow_inputs = model_flow_inputs_from_velocities(
        v_feed=values["v_feed"],
        v_steam=v_steam,
        P_vac=values["P_vac"],
        tau_ar=tau_ar,
        bed_volume=bed_volume,
    )
    vol_flowrate = flow_inputs["vol_flowrate_model"]
    m_dot_steam = (
        flow_inputs["m_dot_steam_model"] if is_steam_assisted else None
    )
    operating_settings: dict[str, float | None] = {
        "adsorption_time_s": values["t_ads"],
        "heating_time_s": values["t_heat"],
        "desorption_time_s": values["t_des"],
        "low_pressure_Pa": values["P_vac"],
        "volumetric_flow_m3_s": float(vol_flowrate),
        "steam_mass_flow_kg_s": (
            None if m_dot_steam is None else float(m_dot_steam)
        ),
    }
    # Present only when the campaign optimizes it, so a campaign that does not
    # writes exactly the settings it wrote before this variable existed.
    if COOLING_TARGET_VAR_NAME in values:
        operating_settings["cooling_target_temperature_K"] = values[
            COOLING_TARGET_VAR_NAME
        ]
    return operating_settings, flow_inputs
