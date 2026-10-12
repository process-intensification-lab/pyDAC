"""Stable facade for solver metadata, sparsity, and event utilities."""

import numpy as np

from pydac.utils.solver.flags import FALSE_TOKENS, env_flag
from pydac.utils.solver.composition_domain import (
    CompositionDomainGuard,
    build_composition_domain_event_spec,
    closure_domain_diagnostic,
)
from pydac.utils.solver.events import (
    avg_bed_loading_fraction_CO2_nd,
    avg_outlet_pressure_nd,
    avg_outlet_temperature_nd,
    build_event_specs,
    build_termination_summary,
    format_metric_value,
    make_cvode_events_callback,
    max_bed_temperature_nd,
    max_internal_pressure_nd,
    min_bed_temperature_nd,
    min_internal_pressure_nd,
    resolve_event_threshold_dim,
    resolve_triggered_event,
)
from pydac.utils.solver.sparsity import (
    DEFAULT_CVODE_SPARSITY_MODE,
    SPARSITY_MODE_ALIASES,
    configure_cvode_linear_solver_options,
    configure_sparse_linsolver,
    estimate_sparsity_colors,
    normalize_cvode_sparsity_mode,
    pde_state_size,
    prepare_cvode_solver_sparsity,
    try_get_solver_sparsity,
)


def pressure_drop_metadata_from_model(model):
    """Serialize the resolved dimensional quadratic pressure-drop law."""
    law = getattr(model, "pressure_drop_law", None)
    if law is None:
        raise AttributeError(
            "Model has no resolved 'pressure_drop_law' for result persistence."
        )

    architecture = str(getattr(model, "bed_architecture", "")).strip()
    pressure_drop_model = str(getattr(law, "name", "")).strip()
    reference_pressure = float(model.p_ref)
    bed_length = float(model.L_b)
    reference_velocity = float(model.u_ref)
    linear_nd = np.asarray(law.linear_coefficient, dtype=float)
    inertial_nd = np.asarray(law.inertial_coefficient_per_density, dtype=float)
    is_radial = linear_nd.size > 1
    if is_radial:
        # A radial law carries one coefficient pair per ring.  It is written
        # under its own schema version, with the radial axis named, so a reader
        # cannot mistake it for a scalar law and reconstruct velocities from a
        # single coefficient that was never applied anywhere in the bed.
        linear_nd = linear_nd.reshape(-1)
        inertial_nd = inertial_nd.reshape(-1)
        radial_faces = np.asarray(model.r_f, dtype=float)
        if radial_faces.size != linear_nd.size + 1:
            raise ValueError(
                "Radial pressure-drop coefficients and the radial grid "
                f"disagree: {linear_nd.size} rings against "
                f"{radial_faces.size} faces."
            )
    linear_dimensionless = linear_nd
    inertial_dimensionless = inertial_nd

    values = np.concatenate(
        (
            np.array(
                [reference_pressure, bed_length, reference_velocity], dtype=float
            ),
            np.atleast_1d(linear_dimensionless),
            np.atleast_1d(inertial_dimensionless),
        )
    )
    if not architecture or not pressure_drop_model:
        raise ValueError(
            "Bed architecture and pressure-drop model names must be non-empty."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Resolved pressure-drop metadata must be finite.")
    if min(reference_pressure, bed_length, reference_velocity) <= 0.0:
        raise ValueError(
            "Pressure, bed length, and reference velocity must be positive."
        )

    linear_coefficient = (
        linear_dimensionless
        * reference_pressure
        / (bed_length * reference_velocity)
    )
    inertial_coefficient_per_density = (
        inertial_dimensionless
        * reference_pressure
        / (bed_length * reference_velocity**2)
    )
    if np.any(linear_coefficient <= 0.0) or np.any(
        inertial_coefficient_per_density <= 0.0
    ):
        raise ValueError(
            "Resolved dimensional pressure-drop coefficients must be positive."
        )

    units = {
        "linear_coefficient": "Pa*s/m2",
        "inertial_coefficient_per_density": "1/m",
    }
    if not is_radial:
        return {
            **({"formulation": "isotropic_vector"}
               if getattr(model, "pressure_drop_formulation", "scalar") == "isotropic_vector" else {}),
            "architecture": architecture,
            "model": pressure_drop_model,
            "linear_coefficient": float(linear_coefficient),
            "inertial_coefficient_per_density": float(
                inertial_coefficient_per_density
            ),
            "units": units,
        }

    ring_area = radial_faces[1:] ** 2 - radial_faces[:-1] ** 2
    return {
        **({"formulation": "isotropic_vector"}
           if getattr(model, "pressure_drop_formulation", "scalar") == "isotropic_vector" else {}),
        "architecture": architecture,
        "model": pressure_drop_model,
        "schema_version": 2,
        "radial_axis": {
            "name": "radial_cell",
            "size": int(linear_coefficient.size),
            "r_faces_over_R": radial_faces.tolist(),
        },
        "linear_coefficient": linear_coefficient.tolist(),
        "inertial_coefficient_per_density": (
            inertial_coefficient_per_density.tolist()
        ),
        # Area means are a summary for scalar context tables only.  The law
        # applied in the bed is the per-ring one above; Ergun on an averaged
        # voidage is not the average of the Ergun responses.
        "area_mean": {
            "linear_coefficient": float(
                np.sum(linear_coefficient * ring_area) / np.sum(ring_area)
            ),
            "inertial_coefficient_per_density": float(
                np.sum(inertial_coefficient_per_density * ring_area)
                / np.sum(ring_area)
            ),
        },
        "units": units,
    }
