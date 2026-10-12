"""Read-only decomposition of the coupled gas-species balance."""

from __future__ import annotations

import numpy as np


def build_species_balance_diagnostic(
    model,
    *,
    t,
    packed_state,
    c_tot_dim,
    conv_ctot,
    source_ctot,
    conv_flux_ax_ci,
    conv_flux_rad_ci,
    disp_flux_ax_i,
    disp_flux_rad_i,
    dq_ordered,
    dc_tot_dt,
    dc_independent_dt,
):
    """Return a JSON-safe closure/RHS diagnostic for one evaluated state.

    The separated transport matvecs are recomputed for attribution only.  The
    production RHS remains based on its existing fused arithmetic.
    """
    # Match both transport directions and the local gas-storage normalization.
    inv_gas_storage = 1.0 / (model.eps_total * model._gas_2d)
    conv_independent_nd = (
        -(model.Div_ax_ci.dot(conv_flux_ax_ci)
          + model.L_R_ratio * model.Div_rad_ci.dot(conv_flux_rad_ci)
          ).reshape(model.shape_ci) * inv_gas_storage[..., None]
    )
    disp_ax_independent_nd = (
        -model.Div_ax_ci.dot(disp_flux_ax_i).reshape(model.shape_ci)
        * inv_gas_storage[..., None]
    )
    disp_rad_independent_nd = (
        -model.Div_rad_ci.dot(disp_flux_rad_i).reshape(model.shape_ci)
        * inv_gas_storage[..., None]
    )
    adsorption_independent_nd = np.zeros(model.shape_ci, dtype=float)
    for species in model.adsorbates:
        if species == model.closure_species:
            continue
        independent = model.independent_index[species]
        adsorbate = model.adsorbate_index[species]
        adsorption_independent_nd[..., independent] = (
            -model.psi_adsorbate[adsorbate]
            * np.reshape(dq_ordered[..., adsorbate], model._solid_2d.shape)
            * model._solid_2d
            * inv_gas_storage
        )

    independent_scale = np.asarray(
        [
            model.c_ref_by_species[species] / model.tau_hyd
            for species in model.independent_species
        ],
        dtype=float,
    ).reshape(1, 1, -1)
    independent_components = {
        "convection": conv_independent_nd * independent_scale,
        "axial_dispersion": disp_ax_independent_nd * independent_scale,
        "radial_dispersion": disp_rad_independent_nd * independent_scale,
        "adsorption": adsorption_independent_nd * independent_scale,
    }
    independent_total = dc_independent_dt * independent_scale

    total_scale = model.c_tot_ref / model.tau_hyd
    total_convection = (
        np.asarray(conv_ctot).reshape(model.Nz, model.Nr)
        * inv_gas_storage
        * total_scale
    )
    total_adsorption = (
        np.asarray(source_ctot).reshape(model.Nz, model.Nr)
        * inv_gas_storage
        * total_scale
    )
    total_derivative = (
        np.asarray(dc_tot_dt).reshape(model.Nz, model.Nr) * total_scale
    )

    closure_components = {
        "convection": total_convection
        - np.sum(independent_components["convection"], axis=-1),
        "axial_dispersion": -np.sum(
            independent_components["axial_dispersion"], axis=-1
        ),
        "radial_dispersion": -np.sum(
            independent_components["radial_dispersion"], axis=-1
        ),
        "adsorption": total_adsorption
        - np.sum(independent_components["adsorption"], axis=-1),
    }
    closure_total_from_components = sum(closure_components.values())
    closure_total = total_derivative - np.sum(independent_total, axis=-1)

    c_tot_dim_view = np.asarray(c_tot_dim).reshape(model.Nz, model.Nr)
    independent_concentrations = np.empty(model.shape_ci, dtype=float)
    for independent, (species, field) in enumerate(
        zip(model.independent_species, model.gas_state_fields)
    ):
        independent_concentrations[..., independent] = (
            packed_state[field.state_slice].reshape(model.Nz, model.Nr)
            * model.c_ref_by_species[species]
        )
    independent_sum = np.sum(independent_concentrations, axis=-1)
    closure_concentration = c_tot_dim_view - independent_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        independent_to_total = independent_sum / c_tot_dim_view

    state_min_index = tuple(
        int(value)
        for value in np.unravel_index(
            np.nanargmin(closure_concentration),
            closure_concentration.shape,
        )
    )
    derivative_min_index = tuple(
        int(value)
        for value in np.unravel_index(
            np.nanargmin(closure_total),
            closure_total.shape,
        )
    )

    def cell_record(index):
        z_index, r_index = index
        independent_values = {
            species: float(independent_concentrations[index][independent])
            for independent, species in enumerate(model.independent_species)
        }
        independent_derivatives = {
            species: {
                component: float(values[index][independent])
                for component, values in independent_components.items()
            }
            | {"total": float(independent_total[index][independent])}
            for independent, species in enumerate(model.independent_species)
        }
        closure_derivatives = {
            component: float(values[index])
            for component, values in closure_components.items()
        }
        closure_derivatives["total"] = float(closure_total[index])
        return {
            "index": [z_index, r_index],
            "z_m": float(model.z_c[z_index] * model.L_b),
            "r_m": float(model.r_c[r_index] * model.R_b),
            "c_tot_mol_m3": float(c_tot_dim_view[index]),
            "independent_concentrations_mol_m3": independent_values,
            "independent_sum_mol_m3": float(independent_sum[index]),
            "closure_concentration_mol_m3": float(closure_concentration[index]),
            "independent_derivatives_mol_m3_s": independent_derivatives,
            "closure_derivative_mol_m3_s": closure_derivatives,
        }

    independent_sum_from_components = sum(independent_components.values())
    decomposition_error = max(
        float(
            np.nanmax(
                np.abs(independent_total - independent_sum_from_components)
            )
        ),
        float(
            np.nanmax(
                np.abs(closure_total - closure_total_from_components)
            )
        ),
    )
    return {
        "schema_version": 1,
        "time_s": float(t * model.tau_hyd),
        "closure_species": model.closure_species,
        "all_state_values_finite": bool(np.all(np.isfinite(packed_state))),
        "negative_closure_cell_count": int(
            np.count_nonzero(closure_concentration < 0.0)
        ),
        "minimum_closure_concentration_mol_m3": float(
            np.nanmin(closure_concentration)
        ),
        "maximum_independent_sum_over_total": float(
            np.nanmax(independent_to_total)
        ),
        "decomposition_max_abs_error_mol_m3_s": decomposition_error,
        "minimum_closure_cell": cell_record(state_min_index),
        "most_negative_closure_derivative_cell": cell_record(
            derivative_min_index
        ),
    }


__all__ = ["build_species_balance_diagnostic"]
