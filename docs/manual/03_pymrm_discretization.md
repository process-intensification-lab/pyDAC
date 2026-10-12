---
title: "03 — pymrm Discretization Engine"
toc: true
toc-depth: 3
embed-resources: true
---

> **This manual is the detailed technical reference: how pyDAC works inside, for contributors
> and for checking the model.** To learn to use pyDAC, do the [tutorials](../../examples/README.md)
> first, then read the [user guide](../guide/01_installation.md).

This chapter describes the `pymrm` spatial-operator library and the caching layer in
`src/pydac/core/pymrm_caching.py` used by the column model (`src/pydac/core/column_model.py`).

---

## 1. Overview

`pymrm` (v2.1.10, pinned exactly in `pyproject.toml`) is a Python library for building sparse
matrix operators on structured grids. It implements the **finite volume method (FVM)** with:

- Cell-centered storage for primary unknowns (concentrations, temperature, pressure)
- Staggered (face-centered) storage for fluxes and velocities
- SciPy sparse arrays for the operators: the convection caches return CSR arrays, and the
  model converts selected fixed operators to CSR after assembly when
  `PYDAC_RHS_CSR_OPERATORS` is enabled (the default); operators not converted can remain CSC
- Boundary conditions encoded in a single linear relation per face

**Authors:** E.A.J.F. Peters, M. van Sint Annaland, M. Galanti, D.R. Rieder
**Reference:** [pymrm](https://github.com/multiscale-modelling-multiphase-flows/pymrm) and its
[book sources](https://github.com/multiscale-modelling-multiphase-flows/pymrm-book)

---

## 2. Grid Layout

pymrm operates on structured grids with two coordinate arrays per axis:

| Array | Symbol | Description | Size |
|-------|--------|-------------|------|
| `x_f` | z_f, r_f | **Face** positions (cell boundaries) | N+1 |
| `x_c` | z_c, r_c | **Center** positions (cell midpoints) | N |

For 2D problems, operators are constructed **one axis at a time** using the `axis` parameter and then combined.

The model creates uniform, nondimensional face grids with `np.linspace(0, 1, N + 1)` and
computes the centers as midpoints of neighboring faces (`ColumnModelMoL.__init__`):

```python
import numpy as np

Nz, Nr = 50, 25
z_f = np.linspace(0, 1, Nz + 1)    # Axial faces (nondimensional)
z_c = 0.5 * (z_f[:-1] + z_f[1:])   # Axial centers
r_f = np.linspace(0, 1, Nr + 1)    # Radial faces (nondimensional)
r_c = 0.5 * (r_f[:-1] + r_f[1:])   # Radial centers
```

The caches of §6 receive these arrays and call
`pymrm.generate_grid(size, x_f, generate_x_c=True, x_c=x_c)` to normalize them;
`generate_grid` returns the `(x_f, x_c)` pair only with `generate_x_c=True`.

### Shape Convention

The solver packs the state into one flat vector described by `StateLayout`
(`src/pydac/core/state_layout.py`), where a bed field has shape `(Nz, Nr)` and the wall
temperature `(Nz,)`. Inside the model, work arrays and operators use the shapes below
(attributes of `ColumnModelMoL`); operators act on the flattened (raveled) form of these arrays:

| Shape | Meaning | Example |
|-------|---------|---------|
| `shape_single` | (Nz, Nr, 1) | One scalar field (c_tot, T, p) |
| `shape_ci` | (Nz, Nr, Nc_ci) | Declared independent gas species |
| `shape_u_ax_single` | (Nz+1, Nr, 1) | Axial velocity on faces |
| `shape_u_ax` | (Nz+1, Nr, Nc_ci) | Axial face field with the independent-species axis |
| `shape_u_rad_single` | (Nz, Nr+1, 1) | Radial velocity on faces |
| `shape_u_rad` | (Nz, Nr+1, Nc_ci) | Radial face field with the independent-species axis |
| `shape_Tw` | (Nz, 1, 1) | Wall temperature (1D along z) |

---

## 3. Boundary Condition Notation

All boundary conditions in pymrm use the **generalized linear form**:

$$a \, \frac{\partial\phi}{\partial n}\bigg|_{face} + b \, \phi\big|_{face} = d$$

where $\partial/\partial n$ is the derivative along the **outward** normal of that boundary
face: $-\partial/\partial z$ at the inlet ($z = 0$), $+\partial/\partial z$ at the outlet. The
relation is stored as a Python dictionary: `{'a': a_val, 'b': b_val, 'd': d_val}`.

### Standard Types

| Type | a | b | d | Effect |
|------|---|---|---|--------|
| **Dirichlet** | 0 | 1 | φ_in | φ = φ_in at boundary |
| **Neumann** (zero gradient) | 1 | 0 | 0 | ∂φ/∂n = 0: no diffusive or conductive flux; a convective flux remains unless the normal velocity is zero |
| **Robin** | a ≠ 0 | b ≠ 0 | d | Mixed condition (e.g. valve) |
| **Danckwerts-type** (marker) | 0 | 0 | 0 | No gradient relation is imposed; the RHS replaces the inlet contribution by the prescribed total, species or sensible-enthalpy flux |

### How pymrm Encodes BCs

When `construct_grad` is called, it returns:

1. **`Grad`** — a sparse matrix that maps cell-center values to face gradients (interior faces only have standard stencils).
2. **`grad_bc`** — a sparse column vector containing the boundary contribution.

The full gradient at all faces is then:

```python
grad_phi = Grad @ phi + grad_bc  # grad_bc is converted to dense for efficiency
```

The boundary face values are reconstructed from the `{a, b, d}` relation by a quadratic
fit through the face and the two nearest cells; no ghost-cell state is stored. Fixed boundary contributions are assembled once during initialization. The 2D
bed-temperature boundary is the exception: it uses `GradCache` (§6.2), which builds its
combined matrix on first use and refreshes the boundary-value vector during RHS evaluations.

---

## 4. Core pymrm Functions

### 4.1 `construct_grad(shape, x_f, x_c, bc, axis)`

Builds the gradient operator [Grad] mapping cell-center values → face gradients.

- **Input:** field shape, face/center coordinates, BC tuple `(bc_left, bc_right)`, axis index.
- **Output:** `(Grad, grad_bc)` — sparse matrix and BC correction vector.
- **Stencil:** centered difference between adjacent cell centres on interior faces (second
  order on the uniform grid); quadratic reconstruction satisfying the face BC on boundaries.

### 4.2 `construct_div(shape, x_f, nu, axis)`

Builds the divergence operator [Div] mapping face fluxes → cell-center divergences.

- **`nu` parameter** controls geometry:
  - `nu=0` → Cartesian/planar: `div = (J_{i+1/2} − J_{i−1/2}) / Δx`
  - `nu=1` → Cylindrical (radial): `div = (1/r) ∂(rJ)/∂r`
  - `nu=2` → Spherical

In pyDAC: `nu=0` for axial, `nu=1` for radial.

### 4.3 `construct_convflux_upwind(shape, x_f, x_c, bc, v, axis)`

Builds the upwind convective flux operator.

- Velocity `v` at faces determines the upwind direction.
- Returns `(Conv, conv_bc)` — the convective flux matrix and BC correction.
- The full convective flux is: `flux = Conv @ phi + conv_bc`.

### 4.4 `construct_coefficient_matrix(coefficients, shape, axis)`

Creates a diagonal sparse matrix from spatially varying coefficients (e.g. dispersion D,
conductivity λ). Used to build diffusive flux operators. In the model the dispersion
coefficients cover all gas species, so the species block selects the independent ones first
(`ColumnModelMoL.init_Jac`; fragment, not a standalone script):

```python
Diff_mat_ax = construct_coefficient_matrix(
    D_ax[..., independent_gas_indices], shape=shape_ci, axis=0
)
Disp_flux_ax = -Diff_mat_ax @ Grad_ax_c  # Fick's law
```

A coefficient matrix times the gradient gives only the gradient-based part of a flux. The
live species dispersion in the RHS also carries the face total concentration and the
mixture correction velocity (§5.1).

### 4.5 Interpolation Functions

| Function | Direction | Description |
|----------|-----------|-------------|
| `interp_cntr_to_stagg(φ, x_f, x_c, axis)` | Center → Face | Linear interpolation |
| `interp_stagg_to_cntr(φ, x_f, x_c, axis)` | Face → Center | Linear interpolation |

### 4.6 TVD Limiters

For higher-order convection schemes, pymrm provides flux limiters:

`upwind`, `minmod`, `osher`, `clam`, `muscl`, `smart`, `stoic`, `vanleer`

pyDAC currently uses first-order upwind exclusively.

---

## 5. Assembly in pyDAC

### 5.1 One-Time Operators (built in `init_Jac`)

These operators depend on the grid, the step-resolved transport coefficients and the fixed
BC coefficients, not on the solution state:

| Operator | Equation | Construction |
|----------|----------|-------------|
| `Disp_flux_ax` | −D_ax · Grad_ax | Fixed Fick-type matrix, axial (not used by the RHS) |
| `Disp_flux_rad` | −D_rad · Grad_rad | Fixed Fick-type matrix, radial (not used by the RHS) |
| `Cond_flux_ax` | −λ̄_ax · Grad_ax | Temperature conduction, axial |
| `Cond_flux_rad` | −(L_b/R_b) λ̄_rad · Grad_rad | Temperature conduction, radial (fixed BC; not used by the RHS) |
| `Grad_ax_p` | Grad(p) | Pressure gradient for Ergun |
| `Div_ax_*`, `Div_rad_*` | ∂/∂z, (1/r)∂(r·)/∂r | Divergence operators |

Here λ̄_ax = λ_ax/(v⁰L_b) and λ̄_rad = λ_rad/(v⁰R_b) are the scaled conductivities
(`inv_Pe_h_ax`, `inv_Pe_h_rad`). The live species dispersion is assembled by
`_corrected_species_dispersion_flux()` from the raw mole-fraction gradients, the face total
concentration and the mixture correction velocity. The live 2D radial conduction uses
`Div_Diff_rad_T` with the `GradCache` operator, whose boundary vector follows the current wall
temperature (§5.2).

All BC correction vectors are converted to dense arrays (`*_bc_dense`) to avoid sparse-dense conversion overhead in the RHS loop.

### 5.2 Per-RHS Operators (refreshed during RHS evaluation)

These operators depend on the velocity field. The convection caches are refreshed during each
RHS evaluation (including CVODE's internal evaluations) from the current face-velocity arrays;
passing the same array object again returns the cached operator and boundary vector:

| Operator (RHS local) | Update call |
|----------|-----------------|
| `Conv_ax_single` (c_tot and T balances) | `self.conv_ax_cache.update(self.u_ax_single)` |
| `Conv_ax_ci` | `self.conv_ax_species_cache.update(self.u_ax_single)` |
| `Conv_rad_single` | `self.conv_rad_cache.update(self.u_rad_single)` |
| `Conv_rad_ci` | `self.conv_rad_species_cache.update(self.u_rad_single)` |
| `Grad_rad_T`, `grad_rad_T_bc` (2D only) | `self.grad_rad_T_cache.update_bc(bc_rad_T)` (wall T changes) |

---

## 6. Caching Layer (`src/pydac/core/pymrm_caching.py`)

The caching layer avoids rebuilding operators at every RHS call by separating the
**geometry-dependent** parts (fixed) from the **state-dependent** parts (updated). It exposes
two classes, `ConvfluxUpwindCache` and `GradCache`.

### 6.1 `ConvfluxUpwindCache`

Stores one combined CSR stencil at initialization and only updates the velocity-dependent
data in place. Outline (not the implementation):

```python
class ConvfluxUpwindCache:
    def __init__(self, shape, x_f, x_c=None, bc_inlet=None, bc_outlet=None, axis=0):
        # Build the combined internal+BC CSR pattern once
        ...

    def update(self, v):
        # Same-object shortcut if the exact same velocity array is reused
        # Otherwise update CSR data/indices and the dense BC buffer in place
        return self._combined, self._bc_dense
```

`update(v)` returns the reusable combined CSR matrix and dense boundary vector; use the
returned pair (the buffers are private attributes). In the current implementation:

- no new sparse convection matrix is constructed after `__init__`
- no sparse matrix addition is performed inside `update()`
- the returned BC contribution is already a dense 1D array, so the RHS can overwrite inlet entries after a defensive `.copy()`

### 6.2 `GradCache`

`GradCache` builds the gradient matrix once (it depends on the grid and on the BC
coefficients `a`, `b`) and then refreshes only the BC value `d` on each call. It assumes `a`
and `b` stay fixed and does not check them: a step whose `a` or `b` changes needs a new cache.
Outline (not the implementation):

```python
class GradCache:
    def __init__(self, shape, x_f, x_c=None, axis=0):
        ...

    def update_bc(self, bc):
        # bc = (bc_left, bc_right) dictionaries
        # First call: build and cache the combined gradient matrix
        # Later calls: refresh only the dense BC vector
        return self._combined, self._bc_dense
```

`update_bc(bc)` returns the combined sparse gradient operator and the dense boundary vector.
With more than one cell along the selected axis, the combined matrix is cached on first use
and later calls only update the boundary vector. With one cell along that axis, the boundary
operator is reassembled on every call (the returned matrix is new, the dense buffer is reused).

For the packed-bed model this is especially important for the radial bed-temperature boundary condition, where `T_wall` changes every RHS call but the BC coefficients `a` and `b` remain fixed.

---

## 7. Operator Assembly Pattern (Summary)

```
Initialization (once per step):
  construct_grad / construct_div   → static sparse operators
  -D @ Grad                        → static diffusive operators
  Div_Diff_rad_T = Div_rad_T @ Diff_mat_rad_T
  ConvfluxUpwindCache(...)         → in-place convection caches
  GradCache(...)                   → in-place BC-aware gradient cache
  _jacket_quad_weights             → precomputed wall-energy quadrature weights
  _rhs_buf                         → reusable RHS output buffer

Each RHS evaluation (in execution order):
  gas state, density rho = c_tot·M   → reconstructed from the packed state
  pressure gradient + law inversion  → axial and radial face velocities
  update_inlet_fluxes()              → pressure-driven inlet fluxes (press only)
  axial/radial caches.update(...)    → convection matrices and boundary fluxes
  Div @ (Conv @ φ + bc)              → transport assembly
  grad_rad_T_cache.update_bc(...)    → Grad_rad_T, grad_rad_T_bc (2D only)
  Div_Diff_rad_T @ (...)             → radial bed conduction
  np.dot(_jacket_quad_weights, ...)  → jacket power
```

In 2D RHS evaluations, `grad_rad_T_cache.update_bc(bc_rad_T)` refreshes the radial
temperature-boundary contribution; the 1D branch does not call it.

In 2D the total, species and sensible-enthalpy balances include radial convection
with cylindrical divergence and the model's L/R scaling. Radial convective and
dispersive mass fluxes vanish at axis and wall; the radial heat flux is zero at the axis and
follows the bed-wall Robin condition at the wall. Both directions use first-order upwind convection; dispersion and
conduction use central gradients. Channeling storage factors enter the cell inventories,
while drag uses a separate field averaged over fixed physical layers.
