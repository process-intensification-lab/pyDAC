"""Derivations for public fields that a configuration may leave implicit.

Two public inputs are derived rather than typed by hand: the column geometry
(one pair of length/radius and aspect ratio/volume from the other, as
`geometry.specified_by` names) and the volumetric flow of a
steam-fed step (from its steam mass flow). They live here, outside the loader,
so the loader stays a pure document orchestrator and so each formula keeps a
single owner in ``utils``.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : stage 4 — runs after settings, on EVERY resolution. That is what
#            makes resolution idempotent: a derived value is never carried
#            forward, it is recomputed from the fields it depends on.
# Owns     : the adaptation of two `utils` formulas to the config contract.
# Not here : the formulas themselves (utils.aspect_ratios, flow_transforms).
# Entry    : derive_geometry()  ·  derive_geometry_pair()  ·  derive_steam_volumetric_flow()
# Guide    : docs/manual/01_inputs.md §4.3
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from pydac.config.errors import ConfigurationError

from pydac.utils.aspect_ratios import get_L_and_R
from pydac.utils.flow_transforms import steam_density_from_pressure


def derive_geometry(aspect_ratio: float, volume_m3: float) -> tuple[float, float]:
    """Return ``(length_m, radius_m)`` for an ``L/D`` ratio and a bed volume.

    The formula itself is owned by :func:`utils.aspect_ratios.get_L_and_R`,
    which is also what the legacy initializers use; this wrapper only adapts
    its contract to the configuration error type.
    """
    try:
        length, radius = get_L_and_R(tau_ar=float(aspect_ratio), volume=float(volume_m3))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Could not derive the column geometry: {exc}") from exc
    if length is None or radius is None:
        raise ConfigurationError(
            "`geometry.aspect_ratio` and `geometry.volume_m3` are required to derive "
            "`length_m` and `radius_m`."
        )
    return float(length), float(radius)


#: The two ways to give the column size, and the pair each one computes.
GEOMETRY_INPUT_PAIRS = {
    "volume_aspect_ratio": ("aspect_ratio", "volume_m3"),
    "length_radius": ("length_m", "radius_m"),
}
#: Relative tolerance for a computed field that a case also states: a frozen case
#: stores all four, computed on another machine possibly in the last digits.
# A written derived value must agree with the computed one to this tolerance.
DERIVED_RTOL = 1e-9
GEOMETRY_RTOL = DERIVED_RTOL


def derive_geometry_pair(geometry: dict, keep: frozenset[str] | None = None) -> None:
    """Fill the computed pair of ``geometry`` in place, from the input pair.

    ``keep`` names the computed fields whose written value must be checked rather
    than replaced (``None``: every written one). A written value that differs from
    the computed one is refused: a case gives the column size in one way only.
    """
    import math

    mode = geometry.get("specified_by")
    if mode not in GEOMETRY_INPUT_PAIRS:
        raise ConfigurationError(
            f"`geometry.specified_by` must be one of {sorted(GEOMETRY_INPUT_PAIRS)}; "
            f"got {mode!r}."
        )
    inputs = GEOMETRY_INPUT_PAIRS[mode]
    computed = next(pair for pair in GEOMETRY_INPUT_PAIRS.values() if pair != inputs)
    missing = [name for name in inputs if geometry.get(name) is None]
    if missing:
        raise ConfigurationError(
            f"`geometry.specified_by={mode}` needs `geometry.{missing[0]}`."
        )
    if mode == "volume_aspect_ratio":
        length, radius = derive_geometry(geometry["aspect_ratio"], geometry["volume_m3"])
        values = {"length_m": length, "radius_m": radius}
    else:
        length, radius = float(geometry["length_m"]), float(geometry["radius_m"])
        values = {
            "aspect_ratio": length / (2.0 * radius),
            "volume_m3": math.pi * radius**2 * length,
        }
    for name in computed:
        written = geometry.get(name)
        if written is not None and (keep is None or name in keep):
            if not math.isclose(float(written), values[name], rel_tol=GEOMETRY_RTOL):
                raise ConfigurationError(
                    f"`geometry.{name}` is computed from `geometry.{inputs[0]}` and "
                    f"`geometry.{inputs[1]}` (specified_by={mode}): {written} differs from "
                    f"{values[name]:.12g}. Leave it null, or give the size by "
                    f"`{name}` and its partner with the other specified_by."
                )
        geometry[name] = values[name]


def derive_steam_volumetric_flow(
    steam_mass_flow_kg_s: float,
    *,
    feed_pressure_Pa: float,
    feed_temperature_K: float,
) -> float:
    """Return the inlet volumetric flow [m3/s] equivalent to a steam mass flow.

    The vapor density comes from
    :func:`utils.flow_transforms.steam_density_from_pressure`, capped at
    saturation exactly as the model's own steam estimate is, and is evaluated
    at the same feed state the model uses.
    """
    try:
        density, _pressure_used, _p_sat = steam_density_from_pressure(
            feed_pressure_Pa,
            T_feed=feed_temperature_K,
            cap_at_saturation=True,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Could not derive the volumetric flow from the steam settings: {exc}"
        ) from exc
    if density <= 0.0:
        raise ConfigurationError("The derived steam density must be positive.")
    return float(steam_mass_flow_kg_s) / density


__all__ = [
    "DERIVED_RTOL",
    "GEOMETRY_INPUT_PAIRS",
    "derive_geometry",
    "derive_geometry_pair",
    "derive_steam_volumetric_flow",
]
