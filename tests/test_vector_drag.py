"""Nondegenerate checks of vector drag, sparsity and saved reconstruction."""
import numpy as np
import pytest

from tests.test_phase5b_rhs_solver_sparsity import build_model, complete_initial_state
from tests.test_phase6_postprocessing_runtime import _warm_state
from pydac.core.sparsity_patterns.analytical_sparsity import generate_sparsity_pattern
from pydac.utils.solver.jacobian import finite_difference_jacobian
from pydac.utils.vector_drag import normal_velocity, transverse_gradients
from pydac.post_processing.profiles.step_post_processing import StepPostProcessor
from pydac.core.solver import Solver


@pytest.mark.parametrize('angle', [0., .3, 1.2, 2.5])
@pytest.mark.parametrize('magnitude', [0., 1e-18, 1e-8, 2., 1e3])
def test_vector_inversion_satisfies_forward_law_and_rotation(angle, magnitude):
    g = magnitude*np.array([np.cos(angle), np.sin(angle)])
    kw = dict(linear_coefficient=1.7, inertial_coefficient_per_density=.8)
    u = np.array([normal_velocity(g[0], g[1], 2., **kw),
                  normal_velocity(g[1], g[0], 2., **kw)])
    np.testing.assert_allclose((1.7 + 1.6*np.linalg.norm(u))*u, -g,
                               rtol=3e-15, atol=1e-32)
    if magnitude:
        np.testing.assert_allclose(u/np.linalg.norm(u), -g/magnitude, atol=1e-15)


def test_transverse_collocation_preserves_affine_gradient():
    zf=np.linspace(0,1,6);rf=np.linspace(0,1,5)
    zc=(zf[:-1]+zf[1:])/2;rc=(rf[:-1]+rf[1:])/2
    cross_ax,cross_rad=transverse_gradients(np.full((6,4,1),3.),
        np.full((5,5,1),-2.),zf,zc,rf,rc)
    np.testing.assert_allclose(cross_ax,-2.)
    np.testing.assert_allclose(cross_rad,3.)


def nonuniform_state(model):
    y=complete_initial_state(model)
    z=np.arange(model.Nz)[:,None];r=np.arange(model.Nr)[None,:]
    for name,values in [('c_tot',1.+.01*z+.005*np.cos(2*r)),
                        ('T_bed',1.1+.01*z+.02*np.sin(r))]:
        y[model.state_field_by_name[name].state_slice]=values.ravel()
    return y


@pytest.mark.parametrize('channeling', [False, True])
@pytest.mark.parametrize('step', ['startup','press','blow','cool','heat','s_des'])
def test_nonuniform_vector_jacobian_and_postprocessing(tmp_path, step, channeling):
    settings={'grid.nz':5,'grid.nr':4,'geometry.aspect_ratio':2.}
    if channeling:
        settings['bed.design.flow_model'] = 'wall-channeling-reduced'
    model=build_model(settings=settings)
    if step != 'startup':
        model=build_model(settings=settings,step=step,initial_state=_warm_state(model))
    y=nonuniform_state(model)
    jac=finite_difference_jacobian(lambda x:model.pde_system_kpis(0.,x).copy(),y,eps=1e-7)
    mask=generate_sparsity_pattern(model.state_layout).toarray()
    assert not np.any((abs(jac)>1e-7)&~mask),np.argwhere((abs(jac)>1e-7)&~mask)
    solver=Solver(model,base_folder=str(tmp_path))
    solver.sol={'y':np.column_stack([y,y]),'t':np.array([0.,1.])}
    solver.prepare_results()
    processor=StepPostProcessor(model,solver)
    model.pde_system_kpis(0.,y)
    expected=model.u_ax_single[...,0]*model.u_ref
    expected[abs(expected)<1e-8]=0.
    np.testing.assert_allclose(processor.u_ax[...,0],expected,rtol=2e-10,atol=2e-10)
    assert solver.sol['extra']['pressure_drop']['formulation']=='isotropic_vector'
    assert np.max(abs(model.u_rad_single[:,1:-1]))>0
    np.testing.assert_array_equal(model.u_rad_single[:,[0,-1]],0.)


def test_one_dimensional_solver_keeps_scalar_formulation():
    model=build_model(settings={'grid.nz':5,'grid.nr':1,'geometry.aspect_ratio':8.})
    model.pde_system_kpis(0.,nonuniform_state(model))
    assert model.pressure_drop_formulation=='scalar'
    np.testing.assert_array_equal(model.u_rad_single,0.)


def test_drag_voidage_is_interpolated_before_nonlinear_face_resistance():
    from pydac.core.bed_architectures import RadialErgunPressureDrop
    faces = np.array([0., .1, .4, 1.])
    centres = (faces[:-1] + faces[1:]) / 2
    eps = np.array([.3, .4, .8])
    law = RadialErgunPressureDrop(eps, 1.8e-5, .002, .2, .1, 1.e5)
    a, b = law.coefficients_on_radial_faces(faces, centres)
    expected_eps = np.interp(faces, centres, eps)
    expected = RadialErgunPressureDrop(expected_eps, 1.8e-5, .002, .2, .1, 1.e5)
    assert a.shape == b.shape == (1, 4, 1)
    np.testing.assert_allclose(a, expected.linear_coefficient, rtol=1e-14)
    np.testing.assert_allclose(b, expected.inertial_coefficient_per_density, rtol=1e-14)
    assert np.all(np.isfinite(a)) and np.all(a > 0)
    assert not np.allclose(a.ravel()[1:-1], np.interp(faces, centres, law.linear_coefficient.ravel())[1:-1], rtol=1e-3, atol=0)
