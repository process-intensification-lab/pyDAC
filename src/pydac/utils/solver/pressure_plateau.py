"""Pressure-rate plateau event construction for blowdown and pressurization."""

from __future__ import annotations

import numpy as np


def _build_pressure_rate_plateau_spec(
    *,
    layout,
    p_ref,
    rate_tolerance_nd,
    arm_pressure_pa,
    trend,
    rhs_fn,
):
    """Build a guarded pressure ascent/descent-rate plateau event."""
    p_ref = float(p_ref)
    tolerance = float(rate_tolerance_nd)
    arm_pressure_pa = float(arm_pressure_pa)
    trend = str(trend).strip().lower()
    if trend not in {"descent", "ascent"}:
        raise ValueError("Pressure-rate plateau trend must be 'descent' or 'ascent'.")
    if p_ref <= 0.0 or tolerance <= 0.0 or arm_pressure_pa <= 0.0:
        raise ValueError("Pressure-plateau event parameters must be positive.")
    if not callable(rhs_fn):
        raise TypeError("The pressure-plateau event requires a callable model RHS.")

    arm_pressure_nd = arm_pressure_pa / p_ref
    is_descent = trend == "descent"

    def mean_pressure_rate_nd(t, y):
        c_tot_field = layout.field("c_tot")
        temperature_field = layout.field("T_bed")
        c_tot = np.asarray(y)[c_tot_field.state_slice]
        T_bed = np.asarray(y)[temperature_field.state_slice]
        pressure_nd = c_tot * T_bed
        armed = (
            float(np.max(pressure_nd)) <= arm_pressure_nd
            if is_descent
            else float(np.min(pressure_nd)) >= arm_pressure_nd
        )
        if not armed:
            # The detector is deliberately inactive during the fast transient.
            return tolerance + 1.0
        ydot = np.asarray(rhs_fn(t, y))
        dc_tot = ydot[c_tot_field.state_slice]
        dT_bed = ydot[temperature_field.state_slice]
        pressure_rate = float(np.mean(T_bed * dc_tot + c_tot * dT_bed))
        return -pressure_rate if is_descent else pressure_rate

    if is_descent:
        name = "blowdown_pressure_rate_plateau"
        label = "Blowdown mean pressure descent rate reached plateau tolerance"
        metric_label = "-d(mean(p/p_ref))/d(t/tau_hyd)"
        arm_metric_label = "max(p_internal)"
        arm_operator = "<="
    else:
        name = "pressurization_pressure_rate_plateau"
        label = "Pressurization mean pressure rise rate reached plateau tolerance"
        metric_label = "d(mean(p/p_ref))/d(t/tau_hyd)"
        arm_metric_label = "min(p_internal)"
        arm_operator = ">="

    return {
        "name": name,
        "metric_fn": mean_pressure_rate_nd,
        "metric_uses_time": True,
        "threshold_nd": tolerance,
        "threshold_dim": tolerance,
        "direction": -1,
        "label": label,
        "metric_label": metric_label,
        "unit": "nd_pressure_rate",
        "ref": 1.0,
        "arming": {
            "role": "arming_condition",
            "metric_label": arm_metric_label,
            "operator": arm_operator,
            "threshold_dim": arm_pressure_pa,
            "threshold_str": f"{arm_pressure_pa / 100.0:.2f} mbar",
            "unit": "mbar",
        },
    }


def build_blowdown_pressure_plateau_spec(
    *,
    layout,
    p_ref,
    rate_tolerance_nd,
    arm_below_pa,
    rhs_fn,
):
    """Return the guarded ``-d(mean(p_nd))/d(tau)`` blowdown event."""
    return _build_pressure_rate_plateau_spec(
        layout=layout,
        p_ref=p_ref,
        rate_tolerance_nd=rate_tolerance_nd,
        arm_pressure_pa=arm_below_pa,
        trend="descent",
        rhs_fn=rhs_fn,
    )


def build_pressurization_pressure_plateau_spec(
    *,
    layout,
    p_ref,
    rate_tolerance_nd,
    arm_above_pa,
    rhs_fn,
):
    """Return the guarded ``+d(mean(p_nd))/d(tau)`` pressurization event."""
    return _build_pressure_rate_plateau_spec(
        layout=layout,
        p_ref=p_ref,
        rate_tolerance_nd=rate_tolerance_nd,
        arm_pressure_pa=arm_above_pa,
        trend="ascent",
        rhs_fn=rhs_fn,
    )


__all__ = [
    "build_blowdown_pressure_plateau_spec",
    "build_pressurization_pressure_plateau_spec",
]
