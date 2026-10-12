"""Isotropic quadratic drag and transverse gradients on staggered faces.

Both gradient components must use the same physical units (or the same
length/reference-pressure scale). Transport divergence scaling is separate.
"""
from __future__ import annotations

import numpy as np
from pymrm import interp_cntr_to_stagg


ISOTROPIC_VECTOR = "isotropic_vector"


def normal_velocity(normal_gradient, transverse_gradient, density, *,
                    linear_coefficient, inertial_coefficient_per_density):
    """Invert (A + B*rho*|u|) u = -grad(p) for one face-normal component.

    The rationalized root is finite at zero pressure gradient and also works
    in the Darcy limit B=0. The two components share the gradient magnitude.
    """
    a = np.asarray(linear_coefficient, dtype=float)
    b = np.asarray(inertial_coefficient_per_density, dtype=float) * density
    magnitude = np.hypot(normal_gradient, transverse_gradient)
    return -2.0 * normal_gradient / (a + np.sqrt(a*a + 4.0*b*magnitude))


def transverse_gradients(grad_ax, grad_rad, z_f, z_c, r_f, r_c,
                         *, axial_dirichlet=(False, False)):
    """Collocate the other gradient component at each normal flux face.

    Average each face-normal derivative to its cell centre, then interpolate
    along the other direction. Interior stencils span the surrounding 2x3
    cells, giving a nine-point pressure stencil after divergence. The mesh
    centres are midpoints, as in ColumnModelMoL. Trailing dimensions may be
    species or saved time. Constant Dirichlet pressure has zero tangential
    gradient at the corresponding axial boundary.
    """
    radial_at_centres = 0.5 * (grad_rad[:, :-1] + grad_rad[:, 1:])
    axial_at_centres = 0.5 * (grad_ax[:-1] + grad_ax[1:])
    radial_at_axial_faces = interp_cntr_to_stagg(
        radial_at_centres, z_f, z_c, axis=0)
    axial_at_radial_faces = interp_cntr_to_stagg(
        axial_at_centres, r_f, r_c, axis=1)
    for index, fixed in zip((0, -1), axial_dirichlet):
        if fixed:
            radial_at_axial_faces[index] = 0.0
    return radial_at_axial_faces, axial_at_radial_faces
