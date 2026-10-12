"""Numerical Jacobian diagnostics."""

import numpy as np


def finite_difference_jacobian(fun, x, eps=1e-8):
    """
    Approximate a vector-valued function's Jacobian by central differences.

    The perturbation for each component is scaled by ``max(1, abs(x_i))``.
    """
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(fun(x), dtype=float)
    n = x.size
    jacobian = np.zeros((f0.size, n))
    steps = eps * np.maximum(1.0, np.abs(x))

    for i, step in enumerate(steps):
        delta = np.zeros(n)
        delta[i] = step
        f_plus = np.asarray(fun(x + delta), dtype=float)
        f_minus = np.asarray(fun(x - delta), dtype=float)
        jacobian[:, i] = (f_plus - f_minus) / (2.0 * step)

    return jacobian


__all__ = ["finite_difference_jacobian"]
