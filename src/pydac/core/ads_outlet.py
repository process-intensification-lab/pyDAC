"""One-way ads/startup outlet; velocities are superficial, bed-referred."""
from dataclasses import dataclass
from numbers import Real
import math
import numpy as np


@dataclass(frozen=True)
class AdsOutletValve:
    conductance_m_s_Pa: float
    p_out_Pa: float
    p_ref_Pa: float
    u_ref_m_s: float
    overpressure_Pa: float
    feed_velocity_m_s: float

    def outlet_velocity(self, pressure_nd):
        """Dimensionless velocity at the final axial face (linear check valve)."""
        return self.conductance_m_s_Pa * np.maximum(
            np.asarray(pressure_nd) * self.p_ref_Pa - self.p_out_Pa, 0.) / self.u_ref_m_s

    def to_dict(self):
        return {"version": 1, "law": "one_way_linear", "overpressure_Pa": self.overpressure_Pa,
                "conductance_m_s_Pa": self.conductance_m_s_Pa, "p_out_Pa": self.p_out_Pa,
                "p_ref_Pa": self.p_ref_Pa, "u_ref_m_s": self.u_ref_m_s,
                "feed_velocity_m_s": self.feed_velocity_m_s}


def check_ads_outlet(*, step, is_1d, overpressure_Pa, feed_flow_m3_s):
    if overpressure_Pa is None:
        return
    path = f"simulation.cycle.steps.{step}.flow.ads_outlet_overpressure_Pa"
    if step not in ("ads", "startup") or not is_1d:
        raise ValueError(f"{path} requires 1D ads/startup")
    for name, value in ((path, overpressure_Pa), ("volumetric_flow_m3_s", feed_flow_m3_s)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive with the one-way outlet")


def build_ads_outlet(model):
    delta = model.ads_outlet_overpressure_Pa
    check_ads_outlet(step=model.step, is_1d=model.is_1d, overpressure_Pa=delta,
                     feed_flow_m3_s=model.vol_flowrate)
    if delta is None:
        return None
    feed_velocity = model.vol_flowrate / model.inlet_area
    return AdsOutletValve(feed_velocity / delta, model.p_out, model.p_ref, model.u_ref,
                          delta, feed_velocity)
