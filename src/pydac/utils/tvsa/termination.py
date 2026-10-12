"""TVSA termination-mode validation and result presentation helpers."""

from __future__ import annotations

SUPPORTED_TERMINATION_MODES = {"time", "event"}

# The metrics an event can stop a step on (`build_event_specs`), each with what it
# needs from the mixture: a gas species, an adsorbate, or nothing.
EVENT_METRIC_REQUIREMENTS = {
    "avg_outlet_pressure": None,
    "max_internal_pressure": None,
    "min_internal_pressure": None,
    "avg_outlet_temperature": None,
    "max_bed_temperature": None,
    "min_bed_temperature": None,
    "max_bed_oxygen_mole_fraction": ("species", "O2"),
    "avg_bed_loading_fraction_CO2": ("adsorbate", "CO2"),
}
EVENT_METRICS = tuple(EVENT_METRIC_REQUIREMENTS)

# The step-resolved model attributes an event threshold may be read from.
THRESHOLD_SOURCES = ("p_out", "p_feed", "T_cond")


def normalize_termination_mode(
    termination_mode,
    *,
    option_name="termination_mode",
    allow_none=True,
):
    """Validate and normalize a termination mode override."""
    if termination_mode is None:
        if allow_none:
            return None
        raise TypeError(f"`{option_name}` must be one of: event, time.")

    if not isinstance(termination_mode, str):
        raise TypeError(
            f"`{option_name}` must be str with value 'time' or 'event'. "
            f"Received type: {type(termination_mode).__name__}."
        )

    mode = termination_mode.strip().lower()
    if mode not in SUPPORTED_TERMINATION_MODES:
        valid = ", ".join(sorted(SUPPORTED_TERMINATION_MODES))
        raise ValueError(
            f"`{option_name}` must be one of: {valid}. "
            f"Received: {termination_mode!r}."
        )

    return mode


def describe_termination_setup_from_model(model):
    """Return display metadata for a model's configured termination."""
    return _describe_termination(
        mode=str(getattr(model, "termination_mode", "time")).strip().lower(),
        metric=getattr(model, "termination_metric", None),
        label=getattr(model, "termination_label", None),
        has_cool_safety=bool(getattr(model, "cool_stop_on_low_pressure", False)),
    )


def _describe_termination(*, mode, metric, label, has_cool_safety):
    """Format one termination setup, whatever it was read from."""
    if mode == "event":
        detail = label or metric
        display = f"event ({detail})" if detail else "event"
    elif mode == "time":
        display = "time"
    else:
        display = mode or "unknown"

    if has_cool_safety:
        display = f"{display} + low-pressure safety event"

    return {
        "mode": mode or "unknown",
        "metric": metric,
        "label": label,
        "cool_low_pressure_safety": has_cool_safety,
        "display": display,
    }


def describe_step_termination(cycle_config, step, termination_mode=None):
    """Resolve termination metadata for a configured cycle step.

    Read straight from the resolved step configuration: describing a step no
    longer requires building a stand-in model to mutate.
    """
    termination = cycle_config.step_config(step).step_settings.termination
    override = normalize_termination_mode(termination_mode, allow_none=True)
    return _describe_termination(
        mode=termination.mode if override is None else override,
        metric=termination.metric,
        label=termination.label,
        has_cool_safety=termination.stop_on_low_pressure,
    )


def describe_termination_setup_from_summary(termination_summary, fallback=None):
    """Return display metadata reconstructed from a solver summary."""
    if not isinstance(termination_summary, dict):
        return fallback if fallback is not None else {
            "mode": "unknown",
            "metric": None,
            "label": None,
            "cool_low_pressure_safety": False,
            "display": "unknown",
        }

    mode = str(termination_summary.get("configured_mode", "")).strip().lower()
    if not mode and fallback is not None:
        mode = fallback["mode"]

    event = termination_summary.get("event") or {}
    label = event.get("label")
    has_cool_safety = bool(fallback.get("cool_low_pressure_safety", False)) if fallback else False

    if mode == "event":
        display = f"event ({label})" if label else "event"
    elif mode == "time":
        display = "time"
    else:
        display = mode or "unknown"

    if has_cool_safety:
        display = f"{display} + low-pressure safety event"

    return {
        "mode": mode or "unknown",
        "metric": event.get("metric_label"),
        "label": label,
        "cool_low_pressure_safety": has_cool_safety,
        "display": display,
    }


def format_termination_outcome(termination_summary):
    """Format the actual solver stop reason for terminal summaries."""
    if not isinstance(termination_summary, dict):
        return "unknown"

    reason = str(termination_summary.get("reason", "unknown"))
    if reason == "event":
        event = termination_summary.get("event") or {}
        label = event.get("label")
        value_str = event.get("metric_value_str")
        t_event_s = event.get("t_event_s")
        if label is not None and value_str is not None:
            detail = f"{label} = {value_str}"
        elif label is not None:
            detail = label
        else:
            detail = None
        if detail is not None and t_event_s is not None:
            return f"event ({detail}) @ {float(t_event_s):.2f} s"
        if detail is not None:
            return f"event ({detail})"
        return "event"

    if reason == "final_time":
        t_final_s = termination_summary.get("t_final_s")
        if t_final_s is not None:
            return f"final time @ {float(t_final_s):.2f} s"
        return "final time"

    if reason == "failure":
        status = termination_summary.get("status")
        return f"failure ({status})" if status else "failure"

    return reason


__all__ = [
    "EVENT_METRIC_REQUIREMENTS",
    "EVENT_METRICS",
    "SUPPORTED_TERMINATION_MODES",
    "THRESHOLD_SOURCES",
    "describe_step_termination",
    "describe_termination_setup_from_model",
    "describe_termination_setup_from_summary",
    "format_termination_outcome",
    "normalize_termination_mode",
]
