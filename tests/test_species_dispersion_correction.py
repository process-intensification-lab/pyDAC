from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from pydac.core.column_model_factory import ColumnModelFactory
from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pymrm import interp_cntr_to_stagg
from pydac.core.sparsity_patterns.analytical_sparsity import generate_sparsity_pattern
from pydac.utils.solver.sparsity import try_get_solver_sparsity


CASE = "tests/cases/v3_smoke_startup.json"


def _build_model(*, nr):
    config = load_experiment(
        CASE,
        settings={"grid.nz": 5, "grid.nr": nr},
    )
    cycle = CycleConfig.from_experiment(config)
    with contextlib.redirect_stdout(io.StringIO()):
        return ColumnModelFactory(cycle).build_column_model(
            step="startup",
            solver_options=cycle.solver_for_step(
                use_sparsity=False,
                compute_numerical_sparsity=False,
            ),
        )


def _initial_state(model):
    state = np.zeros(model.state_layout.state_size)
    state[: model.state_layout.pde_size] = model.get_initial_state()
    return state


def _smooth_independent_mole_fractions(model):
    z = ((np.arange(model.Nz) + 0.5) / model.Nz)[:, None]
    r = ((np.arange(model.Nr) + 0.5) / model.Nr)[None, :]
    co2 = 0.02 + 0.08 * z + 0.01 * r
    h2o = 0.01 + 0.02 * (1.0 - z) + 0.005 * r
    inert = 1.0 - co2 - h2o
    oxygen_fraction = (
        model.y_feed_by_species["O2"]
        / (
            model.y_feed_by_species["O2"]
            + model.y_feed_by_species[model.closure_species]
        )
    )
    return np.stack(
        (co2, h2o, oxygen_fraction * inert),
        axis=-1,
    )


@pytest.mark.parametrize("nr", [1, 3], ids=("1d", "2d"))
def test_dispersion_matrices_use_each_independent_species_coefficient(nr):
    model = _build_model(nr=nr)
    independent = model.independent_gas_indices

    axial_diagonal = model.Diff_mat_ax.diagonal().reshape(model.shape_u_ax)
    radial_diagonal = model.Diff_mat_rad.diagonal().reshape(model.shape_u_rad)
    np.testing.assert_allclose(
        axial_diagonal,
        np.broadcast_to(model.D_ax[..., independent], model.shape_u_ax),
    )
    np.testing.assert_allclose(
        radial_diagonal,
        np.broadcast_to(model.D_rad[..., independent], model.shape_u_rad),
    )


@pytest.mark.parametrize(
    ("nr", "axis"),
    [(1, 0), (3, 0), (3, 1)],
    ids=("1d_axial", "2d_axial", "2d_radial"),
)
def test_correction_velocity_makes_the_full_mixture_flux_sum_zero(nr, axis):
    model = _build_model(nr=nr)
    yi = _smooth_independent_mole_fractions(model)
    independent_flux = model._corrected_species_dispersion_flux(
        yi.ravel(), axis=axis
    )

    if axis == 0:
        gradients = (
            model.Grad_ax_c.dot(yi.ravel()) + model.grad_ax_c_bc_dense
        ).reshape(model.shape_u_ax)
        yi_face = interp_cntr_to_stagg(
            yi, model.z_f, model.z_c, axis=0
        )
        inv_pe_ind = model._dispersion_inv_pe_ax_ind
        inv_pe_delta = model._dispersion_inv_pe_ax_delta
        face_shape = model.shape_u_ax
        fick_operator_flux = (
            model.Disp_flux_ax.dot(yi.ravel())
            + model.disp_flux_ax_bc_dense
        ).reshape(face_shape)
    else:
        gradients = (
            model.Grad_rad_c.dot(yi.ravel()) + model.grad_rad_c_bc_dense
        ).reshape(model.shape_u_rad)
        yi_face = interp_cntr_to_stagg(
            yi, model.r_f, model.r_c, axis=1
        )
        inv_pe_ind = model._dispersion_inv_pe_rad_ind
        inv_pe_delta = model._dispersion_inv_pe_rad_delta
        face_shape = model.shape_u_rad
        fick_operator_flux = (
            model.Disp_flux_rad.dot(yi.ravel())
            + model.disp_flux_rad_bc_dense
        ).reshape(face_shape)

    # Reconstruct the closure flux from the same full-mixture formula.  The
    # correction sum itself is layout-generic and needs no closure state.
    correction_velocity = np.sum(
        inv_pe_delta * gradients, axis=-1, keepdims=True
    )
    closure_inv_pe = float((inv_pe_ind - inv_pe_delta).ravel()[0])
    closure_gradient = -np.sum(gradients, axis=-1, keepdims=True)
    closure_y_face = 1.0 - np.sum(yi_face, axis=-1, keepdims=True)
    closure_flux = (
        -closure_inv_pe * closure_gradient
        + closure_y_face * correction_velocity
    )
    independent_flux_common_scale = (
        independent_flux.reshape(face_shape) / model._dispersion_ratio_ind
    )

    np.testing.assert_allclose(
        fick_operator_flux,
        model._dispersion_ratio_ind * (-inv_pe_ind * gradients),
        rtol=2e-15,
        atol=2e-16,
    )

    np.testing.assert_allclose(
        np.sum(independent_flux_common_scale, axis=-1, keepdims=True)
        + closure_flux,
        0.0,
        rtol=0.0,
        atol=5e-17,
    )


@pytest.mark.parametrize("nr", [1, 3], ids=("1d", "2d"))
def test_uniform_o2_n2_ratio_is_preserved_when_their_diffusivities_match(nr):
    model = _build_model(nr=nr)
    state = _initial_state(model)
    yi = _smooth_independent_mole_fractions(model)
    c_tot_dim = (
        state[model.state_field_by_name["c_tot"].state_slice]
        * model.c_tot_ref
    ).reshape(model.Nz, model.Nr)

    for independent, species in enumerate(model.independent_species):
        field = model.state_field_by_name[f"c_{species}"]
        state[field.state_slice] = (
            yi[..., independent]
            * c_tot_dim
            / model.c_ref_by_species[species]
        ).ravel()

    oxygen = model.independent_index["O2"]
    model._dispersion_inv_pe_ax_ind[..., oxygen] -= (
        model._dispersion_inv_pe_ax_delta[..., oxygen]
    )
    model._dispersion_inv_pe_ax_delta[..., oxygen] = 0.0
    model._dispersion_inv_pe_rad_ind[..., oxygen] -= (
        model._dispersion_inv_pe_rad_delta[..., oxygen]
    )
    model._dispersion_inv_pe_rad_delta[..., oxygen] = 0.0

    derivative = model.pde_system_kpis(0.0, state).copy()
    oxygen_field = model.state_field_by_name["c_O2"]
    oxygen_concentration = (
        state[oxygen_field.state_slice] * model.c_ref_by_species["O2"]
    )
    oxygen_rate = (
        derivative[oxygen_field.state_slice] * model.c_ref_by_species["O2"]
    )
    closure_concentration = (
        state[model.state_field_by_name["c_tot"].state_slice] * model.c_tot_ref
    )
    closure_rate = (
        derivative[model.state_field_by_name["c_tot"].state_slice]
        * model.c_tot_ref
    )
    for species in model.independent_species:
        field = model.state_field_by_name[f"c_{species}"]
        closure_concentration -= (
            state[field.state_slice] * model.c_ref_by_species[species]
        )
        closure_rate -= (
            derivative[field.state_slice] * model.c_ref_by_species[species]
        )

    np.testing.assert_allclose(
        oxygen_rate / oxygen_concentration,
        closure_rate / closure_concentration,
        rtol=2e-11,
        atol=2e-13,
    )


def test_2d_sparsity_covers_cross_species_radial_dispersion():
    model = _build_model(nr=3)
    pattern = generate_sparsity_pattern(model.state_layout)
    co2 = model.state_layout.gas_field("CO2")
    oxygen = model.state_layout.gas_field("O2")
    axial_cell = 2
    radial_cell = 1
    row = co2.start + axial_cell * model.Nr + radial_cell
    radial_neighbour = oxygen.start + axial_cell * model.Nr + radial_cell + 1

    assert pattern[row, radial_neighbour]


def test_solver_sparsity_is_the_current_analytical_dependencies():
    model = _build_model(nr=3)
    current = generate_sparsity_pattern(model.state_layout)

    mask, source, origin = try_get_solver_sparsity(
        step="startup", layout=model.state_layout
    )

    assert (source, origin) == ("analytical", "generated in memory")
    assert (mask != current).nnz == 0
