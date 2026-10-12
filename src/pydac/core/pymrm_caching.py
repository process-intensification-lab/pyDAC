"""
pyDAC - Cached sparse operators built on pymrm.

This module stores reusable convection and gradient operator structures so
repeated RHS evaluations can update sparse data in place instead of rebuilding
matrices at every solver call.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

import math
import os
import numpy as np
from scipy.sparse import csr_array
from pymrm import generate_grid
from pymrm.interpolate import create_staggered_array
from pymrm.convect import construct_convflux_bc
from pymrm.operators import construct_grad_int, construct_grad_bc


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

_CACHE_SLICE_WRITE = os.environ.get(
    "PYDAC_CACHE_SLICE_WRITE", "1"
).strip().lower() not in ("0", "off", "false", "no")


def _contiguous_slice(positions):
    """Return the equivalent slice for an ascending contiguous index run."""
    if not _CACHE_SLICE_WRITE or positions.size == 0:
        return None
    start = int(positions[0])
    expected = np.arange(
        start,
        start + positions.size,
        dtype=positions.dtype,
    )
    if not np.array_equal(positions, expected):
        return None
    return slice(start, start + positions.size)


def _shape_triplet(shape, axis):
    """
    Collapse an N-D field shape into the 3-block layout used by `pymrm`.

    Returns `(shape_t, shape_f_t)` where `shape_t` is the cell-centered
    triplet (prefix, along-axis, suffix) and `shape_f_t` is the staggered
    face triplet (same prefix/suffix, along-axis+1).
    """
    if not isinstance(shape, (list, tuple)):
        shape_f = [shape]
    else:
        shape_f = list(shape)
    if axis < 0:
        axis += len(shape)
    shape_t = [
        math.prod(shape_f[0:axis]),
        math.prod(shape_f[axis:axis + 1]),
        math.prod(shape_f[axis + 1:]),
    ]
    shape_f[axis] = shape_f[axis] + 1
    shape_f_t = shape_t.copy()
    shape_f_t[1] = shape_t[1] + 1
    return shape_t, shape_f_t


def _internal_indices(shape, axis):
    """
    Precompute sparse row/column indices for internal faces only.

    `i_f` indexes face rows in the staggered operator; `i_c` indexes
    donor-cell columns in the cell-centered field. These indices depend
    only on grid topology and axis, not on the velocity field.
    """
    shape_t, shape_f_t = _shape_triplet(shape, axis)
    i_f = (
        (shape_t[1] + 1) * shape_t[2] * np.arange(shape_t[0]).reshape(-1, 1, 1)
        + shape_t[2] * np.arange(1, shape_t[1]).reshape((1, -1, 1))
        + np.arange(shape_t[2]).reshape((1, 1, -1))
    )
    i_c = (
        shape_t[1] * shape_t[2] * np.arange(shape_t[0]).reshape((-1, 1, 1))
        + shape_t[2] * np.arange(1, shape_t[1]).reshape((1, -1, 1))
        + np.arange(shape_t[2]).reshape((1, 1, -1))
    )
    return i_f, i_c, shape_t, shape_f_t


# ---------------------------------------------------------------------------
#  ConvfluxUpwindCache
# ---------------------------------------------------------------------------

class ConvfluxUpwindCache:
    """
    Unified upwind convective-flux cache (internal + BC, in-place updates).

    On `__init__`:
    1. Compute internal face indices from the grid shape.
    2. If BCs are provided, call `pymrm.convect.construct_convflux_bc`
       **once** with a unit velocity to capture the BC sparsity pattern
       and coefficient template.
    3. Assemble a single combined CSR matrix (internal + BC) and
       pre-compute index maps locating internal vs BC entries within
       `combined.data`.
    4. Pre-allocate a dense BC vector buffer.

    On every `update(v)`:
    1. If `v` is the same Python object as the previous call, return
       the cached references immediately.
    2. Otherwise, interpolate `v` to faces, compute the upwind donor
       columns for internal faces, and write into `combined.data` and
       `combined.indices` in place.
    3. Scale the BC coefficient template by the current boundary
       velocity and write into `combined.data`.
    4. Scatter the BC inhomogeneous vector into the dense buffer.

    No sparse matrix is constructed after `__init__`. No sparse addition
    is performed after `__init__`.

    Returns
    -------
    `(combined_matrix, bc_dense)` where:
    - `combined_matrix` is a `csr_array` of shape `(n_faces, n_cells)`
      containing the full stencil.
    - `bc_dense` is a dense 1D `ndarray` of length `n_faces` containing
      the inhomogeneous BC contribution. **Callers must `.copy()` this
      array before mutating it**, since it points to the cache's
      internal buffer.
    """

    def __init__(self, shape, x_f, x_c=None, bc_inlet=None, bc_outlet=None, axis=0):
        self.shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        self.axis = axis
        self.x_f, self.x_c = generate_grid(
            self.shape[axis], x_f, generate_x_c=True, x_c=x_c
        )
        self.bc = (bc_inlet, bc_outlet)

        i_f, i_c, shape_t, shape_f_t = _internal_indices(self.shape, axis)
        self._shape_t = shape_t
        self._shape_f_t = shape_f_t
        self._i_c_base = i_c

        n_faces = math.prod(shape_f_t)
        n_cells = math.prod(shape_t)
        self._has_bc = self.bc is not None and self.bc != (None, None)

        # Same-velocity shortcut state
        self._last_v = None

        # ---- Case 1: no BC -- internal-only matrix ----
        if not self._has_bc:
            dummy = np.ones(i_f.size)
            self._combined = csr_array(
                (dummy, (i_f.ravel(), i_c.ravel())),
                shape=(n_faces, n_cells),
            )
            self._combined.sort_indices()
            self._int_data_pos = np.arange(
                self._combined.nnz, dtype=self._combined.indices.dtype
            )
            self._int_write = _contiguous_slice(self._int_data_pos)
            self._bc_dense = np.zeros(n_faces)
            return

        # ---- Case 2: with BC -- combined internal + BC matrix ----
        # 1. internal CSR with dummy data
        int_mat = csr_array(
            (np.ones(i_f.size), (i_f.ravel(), i_c.ravel())),
            shape=(n_faces, n_cells),
        )
        int_face_rows = set(i_f.ravel().tolist())

        # 2. BC CSR from a single pymrm call with unit velocity
        shape_f = list(self.shape)
        shape_f[self.axis] += 1
        v_unit = np.ones(shape_f)
        bc_mat_unit, bc_vec_unit = construct_convflux_bc(
            self.shape, self.x_f, self.x_c, self.bc, v_unit, self.axis,
        )

        # 3. one-time sparse addition to fix the combined sparsity pattern
        combined = (int_mat + bc_mat_unit.tocsr()).tocsr()
        combined.sort_indices()
        self._combined = combined

        # 4. Build data-position maps
        int_data_pos = []
        bc_data_pos = []
        bc_vel_b = []
        bc_vel_f = []
        bc_vel_a = []
        b1_b2 = shape_f_t[1] * shape_f_t[2]
        b2 = shape_f_t[2]

        for r in range(n_faces):
            s, e = combined.indptr[r], combined.indptr[r + 1]
            if e == s:
                continue
            if r in int_face_rows:
                int_data_pos.extend(range(s, e))
            else:
                b = r // b1_b2
                rem = r % b1_b2
                f = rem // b2
                a = rem % b2
                for pos in range(s, e):
                    bc_data_pos.append(pos)
                    bc_vel_b.append(b)
                    bc_vel_f.append(f)
                    bc_vel_a.append(a)

        self._int_data_pos = np.array(int_data_pos, dtype=np.intp)
        self._int_write = _contiguous_slice(self._int_data_pos)
        self._bc_data_pos = np.array(bc_data_pos, dtype=np.intp)
        self._bc_template = combined.data[self._bc_data_pos].copy()
        self._bc_vel_b = np.array(bc_vel_b, dtype=np.intp)
        self._bc_vel_f = np.array(bc_vel_f, dtype=np.intp)
        self._bc_vel_a = np.array(bc_vel_a, dtype=np.intp)

        # 5. BC inhomogeneous vector template (dense)
        self._bc_dense = np.zeros(n_faces)
        if bc_vec_unit.nnz > 0:
            self._bc_vec_idx = bc_vec_unit.indices.copy().astype(np.intp)
            self._bc_vec_template = bc_vec_unit.data.copy()
            bv_b, bv_f, bv_a = [], [], []
            for flat_idx in self._bc_vec_idx:
                b = flat_idx // b1_b2
                rem = flat_idx % b1_b2
                f = rem // b2
                a = rem % b2
                bv_b.append(b)
                bv_f.append(f)
                bv_a.append(a)
            self._bc_vec_vel_b = np.array(bv_b, dtype=np.intp)
            self._bc_vec_vel_f = np.array(bv_f, dtype=np.intp)
            self._bc_vec_vel_a = np.array(bv_a, dtype=np.intp)
        else:
            self._bc_vec_idx = np.array([], dtype=np.intp)
            self._bc_vec_template = np.array([], dtype=float)
            self._bc_vec_vel_b = np.array([], dtype=np.intp)
            self._bc_vec_vel_f = np.array([], dtype=np.intp)
            self._bc_vec_vel_a = np.array([], dtype=np.intp)

    # ------------------------------------------------------------------ #
    def update(self, v):
        """
        Refresh the combined convection operator for a new velocity field.

        If `v` is the same Python object as the velocity used in the
        previous call, the cached matrix and BC vector are returned
        immediately without recomputation (same-velocity shortcut).
        """
        # Same-velocity shortcut: avoids redundant work when the same
        # cache is called multiple times within one RHS evaluation with
        # the same velocity (e.g. for c_tot and for the T balance).
        if v is self._last_v:
            return self._combined, self._bc_dense
        self._last_v = v

        v_f = create_staggered_array(
            v, self.shape, self.axis, x_f=self.x_f, x_c=self.x_c,
        )
        v_t = v_f.reshape(self._shape_f_t)

        # ---- Internal upwind coefficients ----
        v_int = v_t[:, 1:-1, :]
        i_c = self._i_c_base - self._shape_t[2] * (v_int > 0)

        # A contiguous slice writes the same values to the same sparse-data
        # positions without NumPy's fancy-index scatter. Old restored cache
        # instances and non-contiguous layouts keep the indexed path.
        write_positions = getattr(self, "_int_write", None)
        if write_positions is None:
            write_positions = self._int_data_pos
        self._combined.data[write_positions] = (
            v_int.ravel().astype(self._combined.data.dtype, copy=False)
        )
        self._combined.indices[write_positions] = (
            i_c.ravel().astype(self._combined.indices.dtype, copy=False)
        )

        if not self._has_bc:
            return self._combined, self._bc_dense

        # ---- BC matrix coefficients: template * face velocity ----
        bc_vel = v_t[self._bc_vel_b, self._bc_vel_f, self._bc_vel_a]
        self._combined.data[self._bc_data_pos] = self._bc_template * bc_vel

        # ---- BC inhomogeneous vector (dense scatter) ----
        self._bc_dense[:] = 0.0
        if self._bc_vec_idx.size > 0:
            bv_vel = v_t[
                self._bc_vec_vel_b, self._bc_vec_vel_f, self._bc_vec_vel_a
            ]
            self._bc_dense[self._bc_vec_idx] = self._bc_vec_template * bv_vel

        return self._combined, self._bc_dense


# ---------------------------------------------------------------------------
#  GradCache
# ---------------------------------------------------------------------------

class GradCache:
    """
    Unified gradient-operator cache for state-dependent BC values.

    The gradient matrix `grad_int + grad_bc_mat` depends on the grid and
    on the BC coefficients `a`, `b`. For the packed-bed model these are
    constant between RHS calls (only the Dirichlet/Robin value `d` changes
    with the wall temperature). This cache builds the combined matrix
    once on first use and reuses it for every subsequent call.

    The inhomogeneous BC vector is rebuilt each call from pre-extracted
    scaling coefficients and scattered into a dense buffer.

    Returns
    -------
    `(combined_matrix, bc_dense)` where:
    - `combined_matrix` is a sparse matrix containing the full gradient
      stencil (interior + BC).
    - `bc_dense` is a dense 1D `ndarray` of length `n_faces`. **Callers
      must `.copy()` before mutating.**
    """

    def __init__(self, shape, x_f, x_c=None, axis=0):
        self.shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        self.axis = axis
        self.x_f, self.x_c = generate_grid(
            self.shape[axis], x_f, generate_x_c=True, x_c=x_c
        )
        self.grad_int = construct_grad_int(
            self.shape, self.x_f, self.x_c, axis=self.axis
        )
        self._combined = None
        self._bc_dense = None
        self._single_cell = False

    # ------------------------------------------------------------------ #
    def update_bc(self, bc):
        """
        Return `(grad_matrix, bc_dense)` with the refreshed BC data.

        On the first call the combined matrix is built and cached;
        subsequent calls only recompute the dense BC vector.
        """
        if self._combined is None:
            return self._first_update(bc)
        if self._single_cell:
            # Single-cell fallback: full recompute (rare in 2D runs).
            grad_bc_mat, grad_bc = construct_grad_bc(
                self.shape, self.x_f, self.x_c, bc, axis=self.axis,
            )
            mat = self.grad_int + grad_bc_mat
            self._bc_dense[:] = 0.0
            if grad_bc.nnz > 0:
                self._bc_dense[grad_bc.indices] = grad_bc.data
            return mat, self._bc_dense
        return self._fast_update(bc)

    # ------------------------------------------------------------------ #
    def _first_update(self, bc):
        grad_bc_mat, grad_bc = construct_grad_bc(
            self.shape, self.x_f, self.x_c, bc, axis=self.axis,
        )
        self._combined = self.grad_int + grad_bc_mat
        self._n_faces = grad_bc.shape[0]
        self._bc_dense = np.zeros(self._n_faces)
        if grad_bc.nnz > 0:
            self._bc_dense[grad_bc.indices] = grad_bc.data
        self._precompute_d_scale(bc)
        return self._combined, self._bc_dense

    # ------------------------------------------------------------------ #
    def _precompute_d_scale(self, bc):
        """
        Cache the constant multipliers that convert `d` values into the
        entries of the BC inhomogeneous vector. Uses the same formulas
        as `pymrm.operators.construct_grad_bc`, but only the pieces that
        depend on `a`, `b`, and the grid (constants).
        """
        from pymrm.helpers import unwrap_bc_coeff

        shape = self.shape
        axis = self.axis
        x_f, x_c = self.x_f, self.x_c
        shape_t, shape_f_t = _shape_triplet(shape, axis)

        if shape_t[1] == 1:
            self._single_cell = True
            return
        self._single_cell = False

        shape_f = list(shape)
        shape_f[axis] += 1
        shape_bc = list(shape_f)
        shape_bc[axis] = 1
        shape_bc_d = [shape_t[0], shape_t[2]]

        # Face-row indices for the left and right BC vector entries.
        i_f_bc = (
            shape_f_t[1] * shape_f_t[2]
            * np.arange(shape_f_t[0]).reshape((-1, 1, 1))
            + shape_f_t[2]
            * np.array([0, shape_f_t[1] - 1]).reshape((1, -1, 1))
            + np.arange(shape_f_t[2]).reshape((1, 1, -1))
        )
        self._bc_i_f = i_f_bc
        self._bc_i_f_flat = i_f_bc.ravel()
        self._bc_n = i_f_bc.size

        # Left BC: values_bc[:, 0, :] = -d * fctr
        a_l = (
            unwrap_bc_coeff(shape, bc[0]['a'], axis=axis)
            if bc[0] else np.zeros((1,) * len(shape))
        )
        b_l = (
            unwrap_bc_coeff(shape, bc[0]['b'], axis=axis)
            if bc[0] else np.zeros((1,) * len(shape))
        )
        alpha_0_l = (
            (x_c[1] - x_f[0]) / ((x_c[0] - x_f[0]) * (x_c[1] - x_c[0]))
            - (x_c[0] - x_f[0]) / ((x_c[1] - x_f[0]) * (x_c[1] - x_c[0]))
        )
        fctr_l = a_l + b_l / alpha_0_l
        np.divide(1, fctr_l, out=fctr_l, where=(fctr_l != 0))
        self._d_scale_left = (
            np.broadcast_to(-fctr_l, shape_bc).reshape(shape_bc_d).copy()
        )

        # Right BC: values_bc[:, 1, :] = +d * fctr
        a_r = (
            unwrap_bc_coeff(shape, bc[1]['a'], axis=axis)
            if bc[1] else np.zeros((1,) * len(shape))
        )
        b_r = (
            unwrap_bc_coeff(shape, bc[1]['b'], axis=axis)
            if bc[1] else np.zeros((1,) * len(shape))
        )
        alpha_0_r = (
            -(x_c[-2] - x_f[-1]) / ((x_c[-1] - x_f[-1]) * (x_c[-2] - x_c[-1]))
            - (-(x_c[-1] - x_f[-1]) / ((x_c[-2] - x_f[-1]) * (x_c[-2] - x_c[-1])))
        )
        fctr_r = a_r + b_r / alpha_0_r
        np.divide(1, fctr_r, out=fctr_r, where=(fctr_r != 0))
        self._d_scale_right = (
            np.broadcast_to(fctr_r, shape_bc).reshape(shape_bc_d).copy()
        )

        self._bc_vals = np.empty((shape_t[0], 2, shape_t[2]))
        self._shape_bc = shape_bc
        self._shape_bc_d = shape_bc_d

    # ------------------------------------------------------------------ #
    def _fast_update(self, bc):
        """Per-call path: refresh only the dense BC vector."""
        from pymrm.helpers import unwrap_bc_coeff

        shape = self.shape
        axis = self.axis

        d_l = (
            unwrap_bc_coeff(shape, bc[0]['d'], axis=axis)
            if bc[0] else np.zeros((1,) * len(shape))
        )
        d_r = (
            unwrap_bc_coeff(shape, bc[1]['d'], axis=axis)
            if bc[1] else np.zeros((1,) * len(shape))
        )

        self._bc_vals[:, 0, :] = self._d_scale_left * (
            np.broadcast_to(d_l, self._shape_bc).reshape(self._shape_bc_d)
        )
        self._bc_vals[:, 1, :] = self._d_scale_right * (
            np.broadcast_to(d_r, self._shape_bc).reshape(self._shape_bc_d)
        )

        self._bc_dense[:] = 0.0
        self._bc_dense[self._bc_i_f_flat] = self._bc_vals.ravel()
        return self._combined, self._bc_dense
