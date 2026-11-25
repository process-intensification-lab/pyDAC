"""Capacity-limited, supply-referred inlet for 1D pressurization.

The header is at p_feed, T_feed. Closure prevents addition, not pressure
rises due to heat or desorption. Internal velocities need not obey this cap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

PRESS_INLET_MODELS = ("valve", "feed_blower")
FEED_BLOWER_FIELDS = ("feed_blower_flow_m3_s", "feed_blower_band_fraction")
DEFAULT_FEED_BLOWER_BAND_FRACTION = 1e-3
MAX_FEED_BLOWER_BAND_FRACTION = 0.1
#: Smallest band: delta must stay resolvable next to the setpoint in double precision
#: (a band of 1e-20 rounds p_set - delta to p_set and would close the supply everywhere).
MIN_FEED_BLOWER_BAND_FRACTION = 1e-6


@dataclass(frozen=True)
class FeedBlower:
    """Dimensionless law: u_cap=Q_max/(A u_ref), pressures divided by p_ref."""
    u_cap: float
    p_set: float
    delta: float

    def throughput_fraction(self, p):
        # Clip finite trial pressures before division to avoid overflow far outside the band.
        bounded = np.clip(np.asarray(p, dtype=float), self.p_set - self.delta, self.p_set)
        x = np.clip((self.p_set - bounded) / self.delta, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def inlet_velocity(self, p):
        return self.u_cap * self.throughput_fraction(p)


def _finite_number(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float)) and math.isfinite(value)


def check_feed_blower(*, path, step, is_1d, model, flow_m3_s, band_fraction, setpoint_Pa):
    """Shared validation and direct model-build guard; null inherits before here."""
    if model not in PRESS_INLET_MODELS:
        raise ValueError(f"`{path}.press_inlet_model` must be one of {PRESS_INLET_MODELS}, got {model!r}.")
    if model == "valve":
        for name, value in zip(FEED_BLOWER_FIELDS, (flow_m3_s, band_fraction)):
            if value is not None:
                raise ValueError(f"`{path}.{name}` is read only with press_inlet_model 'feed_blower'. "
                                 "A null setting inherits and cannot clear a preset value.")
        return
    if step != "press":
        raise ValueError(f"`{path}.press_inlet_model` = 'feed_blower' applies only to press.")
    if not is_1d:
        raise ValueError(f"`{path}.press_inlet_model` = 'feed_blower' is implemented only on a 1D grid (grid.nr = 1).")
    if not _finite_number(flow_m3_s) or flow_m3_s <= 0:
        raise ValueError(f"`{path}.feed_blower_flow_m3_s` must be a finite positive number [m3/s].")
    if band_fraction is not None and (not _finite_number(band_fraction) or not MIN_FEED_BLOWER_BAND_FRACTION <= band_fraction <= MAX_FEED_BLOWER_BAND_FRACTION):
        raise ValueError(f"`{path}.feed_blower_band_fraction` must be finite and in [1e-6, 0.1].")
    if not _finite_number(setpoint_Pa) or setpoint_Pa <= 0:
        raise ValueError(f"`{path}` feed setpoint must be a finite positive pressure.")


def target_inside_band_warning(*, path, termination, setpoint_Pa, band_fraction, outlet_pressure_Pa):
    """Warn for ascending pressure targets at or above the band's lower edge."""
    if termination.metric not in {"min_internal_pressure", "max_internal_pressure", "avg_outlet_pressure"} or termination.direction != "above":
        return None
    sources = {"p_feed": setpoint_Pa, "p_out": outlet_pressure_Pa}
    if termination.threshold_source is not None:
        if termination.threshold_source not in sources:
            return None
        threshold = sources[termination.threshold_source]
    elif termination.threshold_value is not None:
        threshold = termination.threshold_value
    else:
        return None
    threshold = float(threshold) + float(termination.threshold_offset or 0.0)
    band = DEFAULT_FEED_BLOWER_BAND_FRACTION if band_fraction is None else band_fraction
    if threshold < setpoint_Pa * (1.0 - band):
        return None
    return (f"`{path}`: the {termination.metric} target {threshold:g} Pa is at or above the lower "
            f"edge of the feed blower's throttle band [{setpoint_Pa * (1-band):g}, {setpoint_Pa:g}] Pa. "
            "The supply closes at the setpoint; the step may end on the pressure plateau or time cap. "
            "Place the target below the band and check the first-cell pressure at termination.")


def build_feed_blower(model):
    check_feed_blower(path=f"simulation.cycle.steps.{model.step}.flow", step=model.step,
                      is_1d=model.is_1d, model=model.press_inlet_model,
                      flow_m3_s=model.feed_blower_flow_m3_s,
                      band_fraction=model.feed_blower_band_fraction, setpoint_Pa=model.p_feed)
    if model.press_inlet_model == "valve":
        return None
    if not math.isinf(model.alpha_ps):
        print(f"Feed blower inlet: pressurization_valve_conductance = {model.alpha_ps} 1/m is not read.")
    band = DEFAULT_FEED_BLOWER_BAND_FRACTION if model.feed_blower_band_fraction is None else model.feed_blower_band_fraction
    p_set = model.p_feed / model.p_ref
    return FeedBlower(model.feed_blower_flow_m3_s / (model.inlet_area * model.u_ref), p_set, band * p_set)
