"""Signed drag inversion against independent high-precision roots and limits."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from pydac.core.bed_architectures import solve_quadratic_drag_velocity
from pydac.utils.postprocessing.cyclic_profiles import solve_pressure_drop_velocity


@pytest.mark.parametrize("a,b,g", [
    (1747., 1.24e4, 300.),
    (1747., 1.24e4, 1e-100),
    (1747., 1.24e4, 1e-300),
    (1e200, 1e200, 1e200),  # The old discriminant overflows.
    (1e-200, 1e-200, 1e-200),  # Its squared coefficients underflow.
    (1e308, 1e308, 1e308),  # Even the unscaled rationalized denominator overflows.
    (0.8, 0.64 / 1.7e308, 1.7e308),  # G/scale overflows but the root fits.
    (0., 4., 9.),
    (4., 0., 9.),
    (1e-100, 0., 1e-100),  # No artificial linear-coefficient floor.
    (1., 1e-20, 1e30),  # Tiny b alone does NOT justify dropping inertia.
])
def test_signed_root_matches_high_precision(a, b, g):
    with localcontext() as ctx:
        ctx.prec = 90
        da, db, dg = map(Decimal.from_float, (a, b, g))
        expected = float(2 * dg / (da + (da * da + 4 * db * dg).sqrt()))
    with np.errstate(all="raise"):
        actual = solve_quadratic_drag_velocity(np.array([-g, 0., g]), a, b)
    assert actual[1] == 0.
    np.testing.assert_allclose(actual, [-expected, 0., expected], rtol=8e-15, atol=0)


@pytest.mark.parametrize("g,a,b", [
    (np.nan, 1., 1.), (np.inf, 1., 1.), (1., np.inf, 1.),
    (1., 1., np.nan), (1., -1., 1.), (1., 1., -1.),
    (1., 0., 0.), (0., 0., 0.),
    (1e308, 1e-308, 0.),  # Velocity cannot be represented.
    (1e-308, 1e308, 0.),
])
def test_invalid_or_unrepresentable_root_raises_instead_of_returning_nan(g, a, b):
    with np.errstate(all="raise"), pytest.raises(ValueError, match="Drag"):
        solve_quadratic_drag_velocity(g, a, b)


def test_broadcasting_and_near_zero_slope_are_preserved():
    a = np.array([1., 2., 4.]).reshape(1, 3, 1)
    b = np.array([2., 3., 5.]).reshape(1, 3, 1)
    force = np.linspace(-1e-16, 1e-16, 35).reshape(5, 1, 7)
    u = solve_quadratic_drag_velocity(force, a, b)
    assert u.shape == (5, 3, 7)
    np.testing.assert_allclose(a*u + b*u*np.abs(u), np.broadcast_to(force, u.shape),
                               rtol=1e-14, atol=0.)
    h = 1e-20
    slope = (solve_quadratic_drag_velocity(h, a, b)
             - solve_quadratic_drag_velocity(-h, a, b)) / (2*h)
    np.testing.assert_allclose(slope, 1/a, rtol=1e-14, atol=0.)


def test_saved_result_reader_uses_the_same_stable_signed_inversion():
    g = np.array([-1e-100, 0., 1e-100])[None, :, None] * np.ones((4, 3, 2))
    a = np.array([2., 3., 4.])
    b = np.array([5., 6., 7.])
    rho = np.full_like(g, 1.2)
    expected = solve_quadratic_drag_velocity(g, a[None, :, None], b[None, :, None]*rho)
    actual = solve_pressure_drop_velocity(
        g, rho, linear_coefficient=a, inertial_coefficient_per_density=b)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("density", [-1., np.nan, np.inf])
def test_reader_does_not_hide_invalid_density(density):
    with pytest.raises(ValueError, match="density"):
        solve_pressure_drop_velocity(-1., density, linear_coefficient=2.,
                                     inertial_coefficient_per_density=3.)
