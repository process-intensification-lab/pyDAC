"""Vacuum pump with pressure control: the 1D outlet law and its standby power.

A vacuum-connected step (`blow`, `heat`, `w_purge`, `w_des`) has two outlet modes,
`flow.vacuum_outlet_model`:

- ``"valve"`` (default): a throttling valve into an ideal vacuum reservoir held
  at the step's outlet pressure, of infinite capacity (the 3.1.0 model, in
  `column_model.py`);
- ``"vacuum_pump"``: a pump of finite volumetric speed with pressure control at
  the setpoint P_set (the step's outlet pressure), this module.

With the pump the outlet superficial velocity on the last axial face is

    u_out = (S(p) / A) * g(x),   S(p) = S0 (1 - p_ult / p),   x = (p - P_set) / delta

with p the pressure of the last cell, S0 the pump's asymptotic speed, p_ult its
ultimate pressure and delta = band * P_set. The law states two things only: at
full opening (g = 1) the pump removes gas at its speed at the bed pressure,
S(p), and a control reduces the removal to hold P_set. The control g is a C1
smoothstep: closed at and below P_set, fully open above P_set + delta. Pumping
stops at the setpoint; the pump supplies no gas, so other processes can still
lower the pressure. g is a throughput fraction: the law does not say how the
control realises it (on/off with an isolation valve, a variable-speed drive,
or a valve throttling a fixed-speed pump); in this averaged law the bed cannot
tell them apart. The energy can: `idle_power` charges P_idle (1 - g) for the
declared standby mode.
Documented in docs/manual/05_cycle_steps.md §2.7 (law) and
docs/manual/06_post_processing.md §8.2 (energy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


VALVE_OUTLET_MODEL = "valve"
VACUUM_PUMP_OUTLET_MODEL = "vacuum_pump"
VACUUM_OUTLET_MODELS = (VALVE_OUTLET_MODEL, VACUUM_PUMP_OUTLET_MODEL)
#: The steps with a closed inlet and an outlet to the vacuum line.
VACUUM_PUMP_STEPS = frozenset({"blow", "heat", "w_purge", "w_des"})
#: Throttle band delta / P_set when the case leaves it null.
DEFAULT_SETPOINT_BAND_FRACTION = 1e-3
MAX_SETPOINT_BAND_FRACTION = 0.1
#: Smallest band: delta must stay resolvable next to the setpoint in double precision
#: (a band of 5e-324 makes delta 0 and the smoothstep 0/0 at the setpoint). Same
#: bound as the feed blower (`feed_blower.MIN_FEED_BLOWER_BAND_FRACTION`).
MIN_SETPOINT_BAND_FRACTION = 1e-6
#: What the pump draws while the control holds it without removing gas: "off"
#: (on/off with an isolation valve, pump off: no standby power), "minimum_speed"
#: (variable-speed drive at its minimum speed) or "fixed_speed" (a fixed-speed pump
#: running against the closed control valve). The last two need the vendor's power.
IDLE_OFF = "off"
IDLE_MODES = (IDLE_OFF, "minimum_speed", "fixed_speed")
#: The flow fields that only the vacuum pump reads.
VACUUM_PUMP_FIELDS = (
    "vacuum_pump_speed_m3_s",
    "vacuum_pump_ultimate_pressure_Pa",
    "vacuum_setpoint_band_fraction",
    "vacuum_pump_idle_mode",
    "vacuum_pump_idle_power_W",
)


@dataclass(frozen=True)
class VacuumManifold:
    """Nondimensional constants of the outlet law (pressures over p_ref).

    `u_cap` is S0 / (A u_ref), the outlet superficial velocity of the fully
    open pump far above its ultimate pressure.
    """

    u_cap: float
    p_set: float
    p_ult: float
    delta: float
    #: Standby electric power [W] (dimensional), charged as p_idle_W (1 - g); 0 for "off".
    p_idle_W: float = 0.0

    def throughput_fraction(self, p):
        """Return g, exactly 0 at and below P_set and exactly 1 above P_set + delta."""
        x = np.clip((np.asarray(p, dtype=float) - self.p_set) / self.delta, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def outlet_velocity(self, p):
        """Return the nondimensional outlet superficial velocity for last-cell pressure `p`.

        The closed branch never divides by p, so non-positive trial pressures
        of a Newton iteration return 0 without a warning.
        """
        g = self.throughput_fraction(p)
        open_pressure = np.where(g > 0.0, p, 1.0)
        return self.u_cap * g * (1.0 - self.p_ult / open_pressure)

    def idle_power(self, p):
        """Return the standby power P_idle (1 - g) [W] at last-cell pressure `p` (nondimensional).

        At g = 1 the pump is at full load and the isentropic term is its whole
        power, at g = 0 it removes nothing and draws P_idle; the weight is exact
        for on/off operation (on a fraction g of the time). Exactly 0.0 in the
        "off" mode, so every number of a case without standby power is kept.
        """
        if self.p_idle_W == 0.0:
            return 0.0
        return self.p_idle_W * (1.0 - self.throughput_fraction(p))

    def suction_pressure(self, p):
        """Return p_ult + g (p - p_ult), the suction pressure if a valve throttles a fixed-speed pump.

        For ideal gas at one temperature and no line storage, a pump at this
        suction pressure moves the law's throughput S0 (p - p_ult) g / (R T).
        It is one realisation of the control, a diagnostic only: on/off and
        variable-speed operation keep the pump inlet at p. At zero throughput it
        returns p_ult, which has no operational meaning.
        """
        g = self.throughput_fraction(p)
        return self.p_ult + g * (np.asarray(p, dtype=float) - self.p_ult)


def _finite_number(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def check_vacuum_outlet(
    *,
    path: str,
    step: str,
    is_1d: bool,
    model,
    speed_m3_s,
    ultimate_pressure_Pa,
    band_fraction,
    setpoint_Pa,
    idle_mode=None,
    idle_power_W=None,
) -> None:
    """Refuse a vacuum-outlet declaration the model would not read as stated.

    Called by the case validation and again by the model build, so `--validate`
    and `--dry-run` refuse what the solve would refuse. `path` names the step's
    `flow` group in the messages. The step's `blowdown_valve_conductance` is not
    checked: the pump does not read it (an explicit null on a step field inherits
    the default, so the reference `blow` could never clear its 0.01).
    """
    if model not in VACUUM_OUTLET_MODELS:
        raise ValueError(
            f"`{path}.vacuum_outlet_model` must be one of "
            f"{', '.join(map(repr, VACUUM_OUTLET_MODELS))}, got {model!r}."
        )
    values = {
        "vacuum_pump_speed_m3_s": speed_m3_s,
        "vacuum_pump_ultimate_pressure_Pa": ultimate_pressure_Pa,
        "vacuum_setpoint_band_fraction": band_fraction,
        "vacuum_pump_idle_mode": idle_mode,
        "vacuum_pump_idle_power_W": idle_power_W,
    }
    if model == VALVE_OUTLET_MODEL:
        stated = [name for name, value in values.items() if value is not None]
        if stated:
            raise ValueError(
                f"`{path}.{stated[0]}` is read only with "
                f"`vacuum_outlet_model` = {VACUUM_PUMP_OUTLET_MODEL!r}; this step uses "
                f"the valve. Use a base case in which this field resolves to null. A null "
                f"setting inherits the preset value and cannot clear it."
            )
        return
    if step not in VACUUM_PUMP_STEPS:
        raise ValueError(
            f"`{path}.vacuum_outlet_model` = {VACUUM_PUMP_OUTLET_MODEL!r} applies only to "
            f"the vacuum-connected steps {', '.join(sorted(VACUUM_PUMP_STEPS))}; "
            f"step {step!r} does not draw its outlet through the vacuum line."
        )
    if not is_1d:
        raise ValueError(
            f"`{path}.vacuum_outlet_model` = {VACUUM_PUMP_OUTLET_MODEL!r} is implemented "
            "only on a 1D grid (grid.nr = 1)."
        )
    for name, value in values.items():
        if name == "vacuum_pump_idle_mode":
            continue
        if value is not None and not _finite_number(value):
            raise ValueError(f"`{path}.{name}` must be a finite number, got {value!r}.")
    if speed_m3_s is None or speed_m3_s <= 0.0:
        raise ValueError(
            f"`{path}.vacuum_pump_speed_m3_s` must be a positive pump speed [m3/s] "
            f"with `vacuum_outlet_model` = {VACUUM_PUMP_OUTLET_MODEL!r}, got {speed_m3_s!r}."
        )
    if ultimate_pressure_Pa is not None and ultimate_pressure_Pa < 0.0:
        raise ValueError(
            f"`{path}.vacuum_pump_ultimate_pressure_Pa` must be >= 0, got "
            f"{ultimate_pressure_Pa!r}."
        )
    if band_fraction is not None and not (
        MIN_SETPOINT_BAND_FRACTION <= band_fraction <= MAX_SETPOINT_BAND_FRACTION
    ):
        raise ValueError(
            f"`{path}.vacuum_setpoint_band_fraction` must be in "
            f"[{MIN_SETPOINT_BAND_FRACTION:g}, {MAX_SETPOINT_BAND_FRACTION:g}], got {band_fraction!r}."
        )
    if not _finite_number(setpoint_Pa) or setpoint_Pa <= 0.0:
        raise ValueError(
            f"The manifold setpoint (`pressure.outlet_pressure_Pa` of the step at {path}) "
            f"must be a positive pressure, got {setpoint_Pa!r}."
        )
    mode = IDLE_OFF if idle_mode is None else idle_mode
    if mode not in IDLE_MODES:
        raise ValueError(
            f"`{path}.vacuum_pump_idle_mode` must be one of "
            f"{', '.join(map(repr, IDLE_MODES))} (null is {IDLE_OFF!r}), got {idle_mode!r}."
        )
    if mode == IDLE_OFF and idle_power_W is not None:
        raise ValueError(
            f"`{path}.vacuum_pump_idle_power_W` is read only with `vacuum_pump_idle_mode` "
            f"'minimum_speed' or 'fixed_speed'; the mode {IDLE_OFF!r} (the pump is off while "
            "the control holds the setpoint) draws no standby power. Declare 'minimum_speed' "
            "or 'fixed_speed', or use a base case in which the idle power resolves to null. "
            "A null setting inherits the preset value and cannot clear it."
        )
    if mode != IDLE_OFF and (idle_power_W is None or idle_power_W <= 0.0):
        raise ValueError(
            f"`{path}.vacuum_pump_idle_power_W` must be a positive power [W] with "
            f"`vacuum_pump_idle_mode` = {mode!r}: the vendor's absorbed power "
            + ("at minimum speed" if mode == "minimum_speed" else "at the ultimate pressure")
            + f", got {idle_power_W!r}."
        )
    p_ult = 0.0 if ultimate_pressure_Pa is None else ultimate_pressure_Pa
    if p_ult >= setpoint_Pa:
        raise ValueError(
            f"`{path}.vacuum_pump_ultimate_pressure_Pa` ({p_ult} Pa) must be below the "
            f"manifold setpoint `pressure.outlet_pressure_Pa` ({setpoint_Pa} Pa)."
        )


#: Pressure events that, with direction "below", wait for the bed to fall to a level.
_PRESSURE_DESCENT_METRICS = frozenset({"max_internal_pressure", "avg_outlet_pressure", "min_internal_pressure"})


def target_inside_band_warning(*, path, termination, setpoint_Pa, band_fraction, feed_pressure_Pa):
    """Return a warning text when a pump step's pressure target lies in the throttle band, else None.

    The control throttles inside the band [P_set, P_set + delta] and stops pumping
    at or below P_set, so the approach to a "pressure below" target at or under
    P_set + delta slows down: a target above P_set is still crossed by pumping in
    finite time, one at or below P_set only by cooling or adsorption (or if it is
    already met at the start). The step can therefore end on the pressure plateau
    or the time cap, and that stop is ill-conditioned (the plateau metric crosses
    its tolerance at a shallow angle), so the target should sit above the band,
    e.g. P_set + 0.5 %.
    """
    if termination.metric not in _PRESSURE_DESCENT_METRICS or termination.direction != "below":
        return None
    sources = {"p_out": setpoint_Pa, "p_feed": feed_pressure_Pa}
    if termination.threshold_source is not None:
        if termination.threshold_source not in sources:
            return None
        threshold = sources[termination.threshold_source]
    elif termination.threshold_value is not None:
        threshold = termination.threshold_value
    else:
        return None
    threshold = float(threshold) + float(termination.threshold_offset or 0.0)
    delta = (DEFAULT_SETPOINT_BAND_FRACTION if band_fraction is None else band_fraction) * setpoint_Pa
    if threshold > setpoint_Pa + delta:
        return None
    return (
        f"`{path}`: the {termination.metric} target {threshold:g} Pa is at or below the upper "
        f"edge of the vacuum pump's throttle band [P_set, P_set + delta] = [{setpoint_Pa:g}, "
        f"{setpoint_Pa + delta:g}] Pa. The flow throttles above P_set and stops at or below it, "
        "so the step may end on the pressure plateau or the time cap instead of this target, "
        "and that stop instant is ill-conditioned; place the target above the band "
        "(e.g. P_set + 0.5 %)."
    )


def build_vacuum_manifold(model) -> VacuumManifold | None:
    """Return the outlet law of a column model, or None for the valve.

    Reads the step fields the model bound from its `flow` group, its geometry
    and its reference scales. The valve returns after the declaration check and
    before any pump arithmetic.
    """
    check_vacuum_outlet(
        path=f"simulation.cycle.steps.{model.step}.flow",
        step=model.step,
        is_1d=model.is_1d,
        model=model.vacuum_outlet_model,
        speed_m3_s=model.vacuum_pump_speed_m3_s,
        ultimate_pressure_Pa=model.vacuum_pump_ultimate_pressure_Pa,
        band_fraction=model.vacuum_setpoint_band_fraction,
        setpoint_Pa=model.p_out,
        idle_mode=model.vacuum_pump_idle_mode,
        idle_power_W=model.vacuum_pump_idle_power_W,
    )
    if model.vacuum_outlet_model == VALVE_OUTLET_MODEL:
        return None
    if not math.isinf(model.alpha_bd):
        print(
            f"Vacuum pump outlet: blowdown_valve_conductance = {model.alpha_bd} 1/m "
            "is not read."
        )
    p_ult = model.vacuum_pump_ultimate_pressure_Pa
    band = model.vacuum_setpoint_band_fraction
    p_set = model.p_out / model.p_ref
    return VacuumManifold(
        u_cap=model.vacuum_pump_speed_m3_s / (model.inlet_area * model.u_ref),
        p_set=p_set,
        p_ult=(0.0 if p_ult is None else p_ult) / model.p_ref,
        delta=(DEFAULT_SETPOINT_BAND_FRACTION if band is None else band) * p_set,
        p_idle_W=0.0 if model.vacuum_pump_idle_power_W is None else float(model.vacuum_pump_idle_power_W),
    )


__all__ = [
    "DEFAULT_SETPOINT_BAND_FRACTION",
    "IDLE_MODES",
    "IDLE_OFF",
    "MAX_SETPOINT_BAND_FRACTION",
    "VACUUM_OUTLET_MODELS",
    "VACUUM_PUMP_FIELDS",
    "VACUUM_PUMP_OUTLET_MODEL",
    "VACUUM_PUMP_STEPS",
    "VALVE_OUTLET_MODEL",
    "VacuumManifold",
    "build_vacuum_manifold",
    "check_vacuum_outlet",
    "target_inside_band_warning",
]
