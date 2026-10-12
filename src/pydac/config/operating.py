"""Campaign operating knobs, expressed as settings on the reference cycle.

A campaign never restates a whole step. It states a handful of named,
unit-bearing knobs — how long adsorption runs, which vacuum level is held,
how much steam is fed — and each knob stands for the step fields it drives.
That fan-out is written down once, here, so the meaning of ``low_pressure_Pa``
cannot drift between the optimizer and the model.

The table reproduces exactly which fields the retired per-step initializers
touched when a campaign supplied the corresponding argument. Two consequences
of that history are deliberate and worth reading twice:

* ``low_pressure_Pa`` reaches the *feed* pressure only of the four steps whose
  supply/evaluation state sits at the vacuum level (``heat``, ``w_purge``,
  ``s_purge``, and ``s_des``). In the
  closed-inlet steps the feed pressure is a fixed evaluation state, and it now
  says so in the profile instead of being a literal in Python.
* both desorption variants can terminate on an event, so their fallback
  horizons must follow the optimized nominal duration; ``desorption_time_s``
  drives both fields for both steps.
* ``cooling_target_temperature_K`` writes a *threshold*, not a horizon. It is
  the first knob of that kind, and it is what lets an optimizer choose how
  cold the bed must be before feed air re-enters while the step still stops on
  the physical event rather than on a clock.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : before stage 3 — a campaign knob becomes ordinary settings, then
#            travels through the very same loader as everything else.
# Owns     : the knob -> (step, group, field) fan-out table below.
# Not here : the step fields themselves (steps.py).
# Entry    : operating_step_settings()
# Guide    : docs/manual/01_inputs.md §8.1
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydac.config.errors import ConfigurationError


# knob -> the (step, group, field) targets it writes.
OPERATING_SETTING_TARGETS: Mapping[str, tuple[tuple[str, str, str], ...]] = MappingProxyType(
    {
        "adsorption_time_s": (
            ("startup", "horizon", "nominal_time_s"),
            ("ads", "horizon", "nominal_time_s"),
        ),
        "heating_time_s": (("heat", "horizon", "nominal_time_s"),),
        "desorption_time_s": (
            ("s_des", "horizon", "nominal_time_s"),
            ("s_des", "horizon", "event_max_time_s"),
            ("w_des", "horizon", "nominal_time_s"),
            ("w_des", "horizon", "event_max_time_s"),
        ),
        "cooling_time_s": (("cool", "horizon", "nominal_time_s"),),
        # The cooldown is expressed as *how cold*, not *how long*: the step
        # ends when the bed has actually reached the target, so its duration
        # adapts to the column shape by itself. Bounding the target below the
        # oxidation limit makes every candidate safe by construction, which a
        # clock plus a terminal-temperature constraint could not do.
        "cooling_target_temperature_K": (
            ("cool", "termination", "threshold_value"),
        ),
        "low_pressure_Pa": (
            ("blow", "pressure", "outlet_pressure_Pa"),
            ("heat", "pressure", "outlet_pressure_Pa"),
            ("heat", "pressure", "feed_pressure_Pa"),
            ("w_purge", "pressure", "outlet_pressure_Pa"),
            ("w_purge", "pressure", "feed_pressure_Pa"),
            ("s_purge", "pressure", "outlet_pressure_Pa"),
            ("s_purge", "pressure", "feed_pressure_Pa"),
            ("s_des", "pressure", "outlet_pressure_Pa"),
            ("s_des", "pressure", "feed_pressure_Pa"),
            ("w_des", "pressure", "outlet_pressure_Pa"),
            ("cool", "pressure", "outlet_pressure_Pa"),
        ),
        "volumetric_flow_m3_s": (
            ("startup", "flow", "volumetric_flow_m3_s"),
            ("ads", "flow", "volumetric_flow_m3_s"),
        ),
        "steam_mass_flow_kg_s": (
            ("s_purge", "flow", "steam_mass_flow_kg_s"),
            ("s_des", "flow", "steam_mass_flow_kg_s"),
        ),
        "adsorption_feed_temperature_K": (
            ("startup", "thermal", "feed_temperature_K"),
            ("ads", "thermal", "feed_temperature_K"),
        ),
        "adsorption_relative_humidity": (
            ("startup", "composition", "relative_humidity"),
            ("ads", "composition", "relative_humidity"),
        ),
    }
)

OPERATING_SETTING_KEYS: tuple[str, ...] = tuple(OPERATING_SETTING_TARGETS)

def operating_step_settings(values: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the canonical dotted settings a campaign's knobs stand for.

    An omitted knob, or one left ``None``, contributes nothing: the step keeps
    the value the reference profile gives it.
    """
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ConfigurationError("Operating settings must be a mapping or None.")

    settings: dict[str, Any] = {}
    for knob, targets in OPERATING_SETTING_TARGETS.items():
        value = values.get(knob)
        if value is None:
            continue
        for step, group, field in targets:
            settings[f"steps.{step}.{group}.{field}"] = float(value)
    return settings


__all__ = [
    "OPERATING_SETTING_KEYS",
    "OPERATING_SETTING_TARGETS",
    "operating_step_settings",
]
