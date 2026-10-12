"""Model-to-event configuration boundary for the solver."""

import numpy as np

from pydac.utils.solver.composition_domain import (
    CompositionDomainGuard,
    build_composition_domain_event_spec,
)
from pydac.utils.solver.events import build_event_specs
from pydac.utils.solver.pressure_plateau import (
    build_blowdown_pressure_plateau_spec,
    build_pressurization_pressure_plateau_spec,
)


def build_event_specs_from_model(
    *,
    model,
    step,
    A_rings,
    V_cells,
    p_ref,
    T_ref,
    composition_domain_guard: CompositionDomainGuard | None = None,
):
    """Build event specs from a configured column model and solver geometry."""
    termination_mode = str(getattr(model, "termination_mode", "time")).lower()
    # A loading-capacity fraction is per unit sorbent mass, which the radial
    # structure factor carries.  `A_rings` stays untouched: the outlet metrics
    # are area averages by definition.
    sorbent_weights = np.asarray(V_cells, dtype=float) * np.asarray(
        model._solid_2d, dtype=float
    )
    event_specs = build_event_specs(
        model=model,
        step=step,
        layout=model.state_layout,
        A_rings=A_rings,
        sorbent_weights=sorbent_weights,
        p_ref=p_ref,
        T_ref=T_ref,
        reference_concentrations=model.c_ref_by_species,
        total_reference_concentration=model.c_tot_ref,
        termination_mode=termination_mode,
        termination_metric=getattr(model, "termination_metric", None),
        termination_direction=getattr(model, "termination_direction", None),
        termination_threshold_value=getattr(model, "termination_threshold_value", None),
        termination_threshold_source=getattr(model, "termination_threshold_source", None),
        termination_threshold_offset=float(
            getattr(model, "termination_threshold_offset", 0.0)
        ),
        termination_label=getattr(model, "termination_label", None),
        cool_stop_on_low_pressure=bool(
            step == "cool" and getattr(model, "cool_stop_on_low_pressure", False)
        ),
        cool_pressure_stop_pa=float(getattr(model, "cool_pressure_stop_pa", 0.0)),
    )
    if (
        step == "blow"
        and termination_mode == "event"
        and bool(getattr(model, "stop_on_pressure_plateau", False))
    ):
        event_specs.append(
            build_blowdown_pressure_plateau_spec(
                layout=model.state_layout,
                p_ref=p_ref,
                rate_tolerance_nd=model.pressure_plateau_rate_tolerance_nd,
                arm_below_pa=model.pressure_plateau_arm_below_Pa,
                rhs_fn=model.pde_system_kpis,
            )
        )
    if (
        step == "press"
        and termination_mode == "event"
        and bool(getattr(model, "stop_on_pressure_plateau", False))
    ):
        step_settings = getattr(getattr(model, "column_config", None), "step_settings", None)
        termination = getattr(step_settings, "termination", None)
        arm_above_pa = getattr(termination, "pressure_plateau_arm_above_Pa", None)
        if arm_above_pa is None:
            # Compatibility for lightweight model doubles and old pickled configs.
            arm_above_pa = getattr(model, "pressure_plateau_arm_below_Pa", None)
        event_specs.append(
            build_pressurization_pressure_plateau_spec(
                layout=model.state_layout,
                p_ref=p_ref,
                rate_tolerance_nd=model.pressure_plateau_rate_tolerance_nd,
                arm_above_pa=arm_above_pa,
                rhs_fn=model.pde_system_kpis,
            )
        )
    if composition_domain_guard is not None:
        domain_spec = build_composition_domain_event_spec(
            model, composition_domain_guard
        )
        if domain_spec is not None:
            event_specs.append(domain_spec)
    return event_specs


__all__ = ["build_event_specs_from_model"]
