"""Explicit dimensional output requests; independent of adaptive integration steps."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class OutputSchedule:
    """Versioned local step times in seconds, including zero and the horizon.

    Transition times are known control switches, not predicted terminal roots.
    Each step starts a fresh schedule; CVODE retains the actual terminal root.
    """
    times_s: tuple
    transition_times_s: tuple = ()
    version: int = 1

    def resolve(self, horizon_s):
        if self.version != 1:
            raise ValueError("Unsupported output schedule version")
        t = np.asarray(self.times_s, dtype=float)
        switches = np.asarray(self.transition_times_s, dtype=float)
        if (t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all()
                or t[0] != 0 or t[-1] != horizon_s or np.any(np.diff(t) <= 0)):
            raise ValueError("Output times must strictly increase from zero to the horizon")
        if (switches.ndim != 1 or not np.isfinite(switches).all()
                or np.any(switches < 0) or np.any(switches > horizon_s)):
            raise ValueError("Transition times must lie inside the horizon")
        return np.unique(np.concatenate((t, switches)))

    def to_dict(self):
        return {"version": self.version, "times_s": list(self.times_s),
                "transition_times_s": list(self.transition_times_s)}

    @classmethod
    def dense_start(cls, horizon_s, *, count=4001, first_s=1e-7,
                    transition_times_s=(), transition_window_s=1., transition_count=101):
        if (not np.isfinite(horizon_s) or not 0 < first_s < horizon_s
                or count < 3 or transition_count < 3 or transition_window_s <= 0):
            raise ValueError("Invalid dense schedule parameters")
        parts = [np.array([0.]), np.geomspace(first_s, horizon_s, count-1)]
        for switch in transition_times_s:
            offsets = np.geomspace(first_s, transition_window_s, transition_count)
            around = np.concatenate(([switch], switch-offsets, switch+offsets))
            parts.append(around[(around >= 0) & (around <= horizon_s)])
        schedule = cls(tuple(np.unique(np.concatenate(parts))), tuple(transition_times_s))
        schedule.resolve(horizon_s)
        return schedule


SCHEDULE_DEFAULTS = dict(dense_start_s=None, dense_start_points=None,
                         first_output_s=None, max_output_spacing_s=None)


def validate_schedule_spec(spec, *, horizon_s=None, nt=None):
    """Validate paired controls without allocating output arrays."""
    from numbers import Integral, Real
    import math
    if set(spec) != set(SCHEDULE_DEFAULTS):
        raise ValueError("output_schedule requires exactly the four schema-11 fields")
    for key, value in spec.items():
        if value is None:
            continue
        kind = Integral if key == "dense_start_points" else Real
        if isinstance(value, bool) or not isinstance(value, kind) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"output_schedule.{key} must be finite and positive")
    window, points = spec["dense_start_s"], spec["dense_start_points"]
    first = spec["first_output_s"]
    if (window is None) != (points is None):
        raise ValueError("dense_start_s and dense_start_points must be set together")
    if points is not None and points < 2:
        raise ValueError("dense_start_points must be an integer >= 2")
    if window is None and first is not None:
        raise ValueError("first_output_s requires a dense start window")
    if window is not None:
        first = 1e-6 if first is None else first
        if first > window or (horizon_s is not None and window > horizon_s):
            raise ValueError("output_schedule requires 0 < first_output_s <= dense_start_s <= horizon")
    enabled = any(v is not None for v in spec.values())
    if enabled and nt is not None and nt < 2:
        raise ValueError("output_schedule requires grid.nt >= 2")
    return enabled


def schedule_from_spec(spec, horizon_s, nt, tau_hyd):
    """Union a dimensional dense window/spacing with the exact legacy backbone.

    The all-null path is selected by Solver before any arithmetic here.
    """
    if not validate_schedule_spec(spec, horizon_s=horizon_s, nt=nt):
        return None
    # Convert the backbone to seconds for the versioned object. Solver restores
    # its exact nondimensional values after conversion, avoiding roundoff drift.
    backbone = np.linspace(0., horizon_s / tau_hyd, nt) * tau_hyd
    backbone[-1] = horizon_s
    parts = [backbone]
    if spec["dense_start_s"] is not None:
        first = 1e-6 if spec["first_output_s"] is None else spec["first_output_s"]
        parts.append(np.geomspace(first, spec["dense_start_s"], spec["dense_start_points"]))
    if spec["max_output_spacing_s"] is not None:
        parts.append(np.linspace(0., horizon_s, int(np.ceil(horizon_s/spec["max_output_spacing_s"])) + 1))
    return OutputSchedule(tuple(np.unique(np.concatenate(parts))))
