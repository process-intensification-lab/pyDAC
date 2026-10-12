"""CVODE event metrics, callbacks, and termination summaries."""

import numpy as np


def _state_field(y, layout, name):
    field = layout.field(name)
    return np.asarray(y)[field.state_slice].reshape(field.shape)


def avg_outlet_pressure_nd(y, *, layout, A_rings):
    """Return the area-weighted radial average outlet pressure in nondimensional form."""
    c_tot = _state_field(y, layout, "c_tot")
    T_bed = _state_field(y, layout, "T_bed")
    p_outlet = c_tot[-1, :] * T_bed[-1, :]
    return float(np.sum(p_outlet * A_rings) / np.sum(A_rings))


def max_internal_pressure_nd(y, *, layout):
    """Return the maximum internal pressure over the full domain in nondimensional form."""
    c_tot = _state_field(y, layout, "c_tot")
    T_bed = _state_field(y, layout, "T_bed")
    return float(np.max(c_tot * T_bed))


def min_internal_pressure_nd(y, *, layout):
    """Return the minimum internal pressure over the full domain in nondimensional form."""
    c_tot = _state_field(y, layout, "c_tot")
    T_bed = _state_field(y, layout, "T_bed")
    return float(np.min(c_tot * T_bed))


def avg_outlet_temperature_nd(y, *, layout, A_rings):
    """Return the area-weighted radial average outlet bed temperature."""
    T_bed = _state_field(y, layout, "T_bed")
    T_outlet = T_bed[-1, :]
    return float(np.sum(T_outlet * A_rings) / np.sum(A_rings))


def max_bed_temperature_nd(y, *, layout):
    """Return the maximum bed temperature over the full domain."""
    T_bed = _state_field(y, layout, "T_bed")
    return float(np.max(T_bed))


def min_bed_temperature_nd(y, *, layout):
    """Return the minimum bed temperature over the full domain."""
    T_bed = _state_field(y, layout, "T_bed")
    return float(np.min(T_bed))


def max_bed_species_mole_fraction(
    y,
    *,
    layout,
    species,
    reference_concentrations,
    total_reference_concentration,
):
    """Return a named species mole-fraction maximum, including closure species."""
    c_tot_nd = np.maximum(_state_field(y, layout, "c_tot"), 1e-32)
    c_tot_ref = float(total_reference_concentration)
    independent_fractions = {}
    for independent in layout.independent_gas_species:
        c_nd = np.maximum(_state_field(y, layout, f"c_{independent}"), 0.0)
        independent_fractions[independent] = (
            c_nd * float(reference_concentrations[independent])
            / (c_tot_nd * c_tot_ref)
        )

    if species == layout.closure_species:
        fraction = 1.0 - sum(independent_fractions.values())
    else:
        fraction = independent_fractions[species]
    return float(np.max(fraction))


def avg_bed_loading_fraction_nd(y, *, layout, species, sorbent_weights):
    """
    Return the sorbent-mass-weighted mean loading fraction in nondimensional form.

    The solid loading is already stored relative to its reference capacity in the state
    vector, so the weighted average directly represents a bed-averaged loading
    fraction relative to the reference adsorption capacity.

    The weights must be proportional to local sorbent MASS, not to cell volume: in a
    radially structured bed the two differ, and a volume-weighted mean would report a
    fraction of a capacity the bed does not have, shifting event termination.
    """
    loading = _state_field(y, layout, f"q_{species}")
    weights = np.asarray(sorbent_weights, dtype=float)
    den = float(np.sum(weights))
    if den <= 0.0:
        return float(np.mean(loading))
    return float(np.sum(loading * weights) / den)


def avg_bed_loading_fraction_CO2_nd(y, *, layout, sorbent_weights):
    """Compatibility name for the configured CO2 loading event."""
    return avg_bed_loading_fraction_nd(
        y, layout=layout, species="CO2", sorbent_weights=sorbent_weights
    )


def resolve_event_threshold_dim(
    *,
    model,
    step,
    threshold_source=None,
    threshold_value=None,
    threshold_offset=0.0,
):
    """
    Resolve the dimensional threshold for the configured step event.
    """
    if threshold_source:
        if not hasattr(model, threshold_source):
            raise AttributeError(
                f"Unknown termination threshold source '{threshold_source}' "
                f"for step '{step}'."
            )
        threshold_dim = float(getattr(model, threshold_source))
    elif threshold_value is not None:
        threshold_dim = float(threshold_value)
    else:
        raise ValueError(
            f"Step '{step}' uses event termination but no threshold value/source was provided."
        )

    return threshold_dim + float(threshold_offset)


def format_metric_value(unit, value_dim):
    """Format an event metric value for logging."""
    if unit == "mbar":
        return f"{value_dim / 100.0:.2f} mbar"
    if unit == "degC":
        return f"{value_dim - 273.15:.2f} °C"
    if unit == "fraction":
        return f"{100.0 * value_dim:.2f}%"
    if unit == "ppmmol":
        return f"{1e6 * value_dim:.2f} ppmmol"
    if unit == "nd_pressure_rate":
        return f"{value_dim:.3g} p_ref/tau_hyd"
    return f"{value_dim:.6g}"


def build_event_specs(
    *,
    model,
    step,
    layout,
    A_rings,
    sorbent_weights,
    p_ref,
    T_ref,
    reference_concentrations,
    total_reference_concentration,
    termination_mode="time",
    termination_metric=None,
    termination_direction=None,
    termination_threshold_value=None,
    termination_threshold_source=None,
    termination_threshold_offset=0.0,
    termination_label=None,
    cool_stop_on_low_pressure=False,
    cool_pressure_stop_pa=0.0,
):
    """
    Build the list of terminal CVODE events for the current step.
    """
    metric_specs = {
        "avg_outlet_pressure": {
            "metric_fn": lambda y: avg_outlet_pressure_nd(
                y, layout=layout, A_rings=A_rings
            ),
            "ref": p_ref,
            "unit": "mbar",
            "label": "p_outlet,radial_avg",
        },
        "max_internal_pressure": {
            "metric_fn": lambda y: max_internal_pressure_nd(y, layout=layout),
            "ref": p_ref,
            "unit": "mbar",
            "label": "max(p_internal)",
        },
        "min_internal_pressure": {
            "metric_fn": lambda y: min_internal_pressure_nd(y, layout=layout),
            "ref": p_ref,
            "unit": "mbar",
            "label": "min(p_internal)",
        },
        "avg_outlet_temperature": {
            "metric_fn": lambda y: avg_outlet_temperature_nd(
                y, layout=layout, A_rings=A_rings
            ),
            "ref": T_ref,
            "unit": "degC",
            "label": "T_outlet,radial_avg",
        },
        "max_bed_temperature": {
            "metric_fn": lambda y: max_bed_temperature_nd(y, layout=layout),
            "ref": T_ref,
            "unit": "degC",
            "label": "max(T_bed)",
        },
        "min_bed_temperature": {
            "metric_fn": lambda y: min_bed_temperature_nd(y, layout=layout),
            "ref": T_ref,
            "unit": "degC",
            "label": "min(T_bed)",
        },
    }
    if "O2" in layout.mixture.species:
        metric_specs["max_bed_oxygen_mole_fraction"] = {
            "metric_fn": lambda y: max_bed_species_mole_fraction(
                y,
                layout=layout,
                species="O2",
                reference_concentrations=reference_concentrations,
                total_reference_concentration=total_reference_concentration,
            ),
            "ref": 1.0,
            "unit": "ppmmol",
            "label": "max(y_O2,bed)",
        }
    if "CO2" in layout.adsorbates:
        metric_specs["avg_bed_loading_fraction_CO2"] = {
            "metric_fn": lambda y: avg_bed_loading_fraction_CO2_nd(
                y, layout=layout, sorbent_weights=sorbent_weights
            ),
            "ref": 1.0,
            "unit": "fraction",
            "label": "avg bed CO2 loading fraction",
        }

    event_specs = []
    termination_mode = str(termination_mode).lower()

    if termination_mode == "event":
        if termination_metric not in metric_specs:
            raise ValueError(
                f"Unsupported termination metric '{termination_metric}' for step '{step}'."
            )

        if termination_direction not in ("below", "above"):
            raise ValueError(
                f"Unsupported termination direction '{termination_direction}' for step '{step}'."
            )

        metric_spec = metric_specs[termination_metric]
        threshold_dim = resolve_event_threshold_dim(
            model=model,
            step=step,
            threshold_source=termination_threshold_source,
            threshold_value=termination_threshold_value,
            threshold_offset=termination_threshold_offset,
        )
        event_specs.append({
            "name": "step_termination",
            "metric_fn": metric_spec["metric_fn"],
            "threshold_nd": threshold_dim / metric_spec["ref"],
            "threshold_dim": threshold_dim,
            "direction": -1 if termination_direction == "below" else 1,
            "label": termination_label or metric_spec["label"],
            "metric_label": metric_spec["label"],
            "unit": metric_spec["unit"],
            "ref": metric_spec["ref"],
        })
    elif termination_mode != "time":
        raise ValueError(
            f"Unsupported termination_mode '{termination_mode}' for step '{step}'."
        )

    if cool_stop_on_low_pressure:
        event_specs.append({
            "name": "cool_low_pressure_safety",
            "metric_fn": lambda y: min_internal_pressure_nd(y, layout=layout),
            "threshold_nd": float(cool_pressure_stop_pa) / p_ref,
            "threshold_dim": float(cool_pressure_stop_pa),
            "direction": -1,
            "label": "Cooldown minimum internal pressure floor reached",
            "metric_label": "min(p_internal)",
            "unit": "mbar",
            "ref": p_ref,
        })

    return event_specs

def make_cvode_events_callback(event_specs):
    """Build a CVODE-compatible event callback with terminal metadata."""
    def step_events(t, y, events):
        for idx, spec in enumerate(event_specs):
            args = (t, y) if spec.get("metric_uses_time", False) else (y,)
            events[idx] = spec["metric_fn"](*args) - spec["threshold_nd"]

    step_events.terminal = [True] * len(event_specs)
    step_events.direction = [spec["direction"] for spec in event_specs]
    return step_events


def resolve_triggered_event(sol_raw, event_specs):
    """Decode the last triggered event from a CVODE result."""
    if sol_raw.t_events is None or len(sol_raw.t_events) == 0:
        return None

    t_event = float(sol_raw.t_events[-1])
    y_event = sol_raw.y_events[-1]
    i_event = np.asarray(sol_raw.i_events[-1]).ravel() if sol_raw.i_events is not None else np.array([])
    hit_idx = int(np.flatnonzero(i_event)[0]) if i_event.size and np.any(i_event != 0) else 0
    spec = event_specs[hit_idx]
    args = (t_event, y_event) if spec.get("metric_uses_time", False) else (y_event,)
    metric_value_dim = spec["metric_fn"](*args) * spec["ref"]

    diagnostic_fn = spec.get("diagnostic_fn")
    diagnostic = None if diagnostic_fn is None else diagnostic_fn(y_event)
    return {
        "t_event": t_event,
        "y_event": y_event,
        "spec": spec,
        "metric_value_dim": metric_value_dim,
        "diagnostic": diagnostic,
    }


def build_termination_summary(
    sol_raw,
    event_specs,
    *,
    tau_hyd,
    configured_mode,
):
    """Build a lightweight termination summary for logs and persistence."""
    time_scale = float(tau_hyd)
    t_final_s = None
    if sol_raw.t is not None and len(sol_raw.t):
        t_final_s = float(sol_raw.t[-1]) * time_scale

    summary = {
        "configured_mode": str(configured_mode).lower(),
        "reason": "failure",
        "success": bool(sol_raw.success),
        "status": str(getattr(sol_raw, "status", "")),
        "message": str(getattr(sol_raw, "message", "")),
        "t_final_s": t_final_s,
        "event": None,
    }

    if not sol_raw.success:
        return summary

    event_info = resolve_triggered_event(sol_raw, event_specs) if event_specs else None
    if event_info is None:
        if str(configured_mode).lower() == "event":
            summary["reason"] = "event_not_reached"
        else:
            summary["reason"] = "final_time"
        return summary

    spec = event_info["spec"]
    is_failure = bool(spec.get("is_failure", False))
    summary["reason"] = (
        str(spec.get("failure_reason", "event_failure"))
        if is_failure
        else "event"
    )
    if is_failure:
        summary["success"] = False
        summary["status"] = str(spec.get("failure_status", "event_failure"))
        summary["message"] = str(
            spec.get("failure_message", "A terminal failure event was reached.")
        )
    summary["event"] = {
        "name": spec["name"],
        "label": spec["label"],
        "metric_label": spec["metric_label"],
        "unit": spec["unit"],
        "threshold_dim": float(spec["threshold_dim"]),
        "threshold_str": format_metric_value(spec["unit"], spec["threshold_dim"]),
        "metric_value_dim": float(event_info["metric_value_dim"]),
        "metric_value_str": format_metric_value(spec["unit"], event_info["metric_value_dim"]),
        "t_event_s": float(event_info["t_event"]) * time_scale,
    }
    if "arming" in spec:
        summary["event"]["arming"] = dict(spec["arming"])
    if "metadata" in spec:
        summary["event"]["metadata"] = dict(spec["metadata"])
    if event_info["diagnostic"] is not None:
        summary["event"]["diagnostic"] = dict(event_info["diagnostic"])
    return summary
