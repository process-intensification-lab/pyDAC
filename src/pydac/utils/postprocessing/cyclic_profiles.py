"""Pure numerical and trace helpers for cyclic profile post-processing."""

from __future__ import annotations

import numpy as np

from pydac.core.bed_architectures.packed_bed import solve_quadratic_drag_velocity


PRESSURE_DROP_UNITS = {
    "linear_coefficient": "Pa*s/m2",
    "inertial_coefficient_per_density": "1/m",
}


def radial_average(field_z, A_rings):
    weights = np.asarray(A_rings, dtype=float)[:, None]
    return np.sum(field_z * weights, axis=0) / np.sum(A_rings)


def clip_mole_fraction(y):
    return np.clip(y, 1e-12, 1.0)


def solve_pressure_drop_velocity(
    grad_p,
    rho,
    *,
    linear_coefficient,
    inertial_coefficient_per_density,
    **_metadata,
):
    """Solve ``A*u + B*rho*u|u| = driving pressure gradient``."""
    grad_p = np.asarray(grad_p, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if not np.all(np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("Pressure-drop reconstruction requires finite nonnegative density.")

    # A radial law carries one coefficient per ring.  The radial axis is axis 1
    # of the (Nz+1, Nr, Nt) fields and axis 0 of a single-face (Nr, Nt) slice,
    # so the coefficients are reshaped onto it explicitly rather than left to
    # broadcast against whichever axis happens to match.
    A = np.asarray(linear_coefficient, dtype=float)
    B_per_rho = np.asarray(inertial_coefficient_per_density, dtype=float)
    if A.size > 1:
        axis = 1 if grad_p.ndim >= 3 else 0
        shape = [1] * grad_p.ndim
        shape[axis] = A.size
        if grad_p.shape[axis] != A.size:
            raise ValueError(
                "Radial pressure-drop coefficients do not match the field's "
                f"radial axis: {A.size} rings against {grad_p.shape[axis]}."
            )
        A = A.reshape(shape)
        B_per_rho = B_per_rho.reshape(shape)
    B = B_per_rho * rho

    return solve_quadratic_drag_velocity(grad_p, A, B)


def pressure_drop_parameters_from_metadata(extra, step_name):
    metadata = extra.get("pressure_drop")
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Saved solution for step '{step_name}' is missing required "
            "architecture-aware pressure-drop metadata. Legacy pyDAC "
            "results are unsupported by the current post-processing; "
            "regenerate the result."
        )

    schema_version = metadata.get("schema_version", 1)
    if schema_version not in (1, 2):
        raise ValueError(
            f"Pressure-drop metadata for step '{step_name}' declares "
            f"unsupported schema version {schema_version!r}."
        )
    required = {
        "architecture",
        "model",
        "linear_coefficient",
        "inertial_coefficient_per_density",
        "units",
    }
    if schema_version == 2:
        required |= {"schema_version", "radial_axis", "area_mean"}
    missing = sorted(required - set(metadata))
    unknown = sorted(set(metadata) - required - {"formulation"})
    if missing or unknown:
        raise ValueError(
            f"Pressure-drop metadata for step '{step_name}' does not match "
            f"the current schema; missing={missing}, unknown={unknown}."
        )
    architecture = str(metadata["architecture"]).strip()
    model = str(metadata["model"]).strip()
    if not architecture or not model:
        raise ValueError(
            f"Pressure-drop architecture/model names for step '{step_name}' "
            "must be non-empty."
        )
    if metadata["units"] != PRESSURE_DROP_UNITS:
        raise ValueError(
            f"Pressure-drop metadata for step '{step_name}' must use units "
            f"{PRESSURE_DROP_UNITS}."
        )
    if schema_version == 1:
        linear = float(metadata["linear_coefficient"])
        inertial = float(metadata["inertial_coefficient_per_density"])
    else:
        # Reconstruct with the per-ring coefficients the solver actually used.
        # The area means in the metadata are a scalar summary for context
        # tables and must never be substituted for the applied law here.
        linear = np.asarray(metadata["linear_coefficient"], dtype=float)
        inertial = np.asarray(
            metadata["inertial_coefficient_per_density"], dtype=float
        )
        axis = metadata["radial_axis"]
        if not isinstance(axis, dict) or "size" not in axis:
            raise ValueError(
                f"Radial pressure-drop metadata for step '{step_name}' is "
                "missing its radial-axis description."
            )
        if linear.shape != (int(axis["size"]),) or inertial.shape != linear.shape:
            raise ValueError(
                f"Radial pressure-drop coefficients for step '{step_name}' do "
                f"not match the declared radial axis of {axis['size']} rings."
            )
    parameters = {
        "architecture": architecture,
        "model": model,
        "linear_coefficient": linear,
        "inertial_coefficient_per_density": inertial,
    }
    if "formulation" in metadata:
        if metadata["formulation"] != "isotropic_vector":
            raise ValueError("Unsupported pressure-drop formulation.")
        parameters["formulation"] = metadata["formulation"]
    numeric = np.concatenate(
        (np.atleast_1d(linear), np.atleast_1d(inertial))
    )
    if not np.all(np.isfinite(numeric)) or np.min(numeric) <= 0.0:
        raise ValueError(
            f"Pressure-drop coefficients for step '{step_name}' must be "
            "finite and positive."
        )
    return parameters


def normalize_cool_jump_mode(cool_jump_mode):
    if cool_jump_mode is None:
        return None

    mode = str(cool_jump_mode).strip().lower()
    if mode in {"none", "off", "false"}:
        return None
    if mode not in {"trim", "bridge"}:
        raise ValueError(
            "`cool_jump_mode` must be one of: None, 'trim', or 'bridge'."
        )
    return mode


def step_duration_s(time):
    time = np.asarray(time, dtype=float)
    if time.size <= 1:
        return 0.0
    return float(time[-1] - time[0])


def cool_jump_search_limit(
    time,
    *,
    max_fraction,
    max_time_s,
):
    time = np.asarray(time, dtype=float)
    if time.size <= 1:
        return 0

    idx_frac = max(1, int(np.ceil((time.size - 1) * max_fraction)))
    idx_time = np.searchsorted(time, max_time_s, side="right") - 1
    idx_time = max(1, idx_time)
    return min(time.size - 1, idx_frac, idx_time)


def cool_jump_match_tolerance(
    field,
    prev_values,
    curr_values,
    *,
    absolute_floors,
    relative_tolerance,
):
    prev_values = np.asarray(prev_values, dtype=float)
    curr_values = np.asarray(curr_values, dtype=float)

    combined = np.concatenate([prev_values[-3:], curr_values])
    span = float(np.max(combined) - np.min(combined))
    abs_floor = absolute_floors.get(field, 0.0)
    return max(abs_floor, relative_tolerance * max(span, 1e-32))


def smoothstep(x):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)
