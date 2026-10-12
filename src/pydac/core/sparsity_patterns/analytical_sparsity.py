"""
pyDAC - Analytical Jacobian sparsity masks.

This module generates the analytical sparsity pattern of the TVSA Jacobian
from a state layout and draws it. The solver generates the mask in memory for
every run (see `pydac.utils.solver.sparsity`); nothing is read from or written
to the package.

Draw the mask of the default DAC layout from the command line:

    python -m pydac.core.sparsity_patterns.analytical_sparsity --nz 10 --nr 10 --out DIR

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-10-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from pydac.core.state_layout import StateLayout, current_dac_state_layout


# Stencil types:
#   '5pt'  = 5-point stencil: offsets [-Nr, -1, 0, +1, +Nr]
#   '3ax'  = 3-point axial:   offsets [-Nr, 0, +Nr]
#   'diag' = diagonal:        offset [0]
#   None   = no coupling

def _field_stencil(equation, variable):
    """Return a conservative physical coupling stencil for two cell fields."""
    if (equation.kind in ("gas concentration", "total concentration", "bed temperature")
            and variable.kind in ("total concentration", "bed temperature")):
        # The transverse pressure derivative in isotropic vector drag couples
        # diagonal cells as well as the normal neighbours.
        return "9pt"
    if equation.kind == "gas concentration":
        if variable.kind == "gas concentration":
            # The mixture-averaged dispersion correction couples every species
            # equation to every species gradient, axially and radially.
            return "5pt"
        if variable.kind == "solid loading":
            return "diag"
        if variable.kind == "total concentration":
            return "5pt"
        if variable.kind == "bed temperature":
            # Radial convection of each species follows the radial velocity,
            # which is inverted from grad(p) = grad(c_tot * T).
            return "5pt"

    if equation.kind == "solid loading":
        if variable.kind in (
            "gas concentration",
            "solid loading",
            "total concentration",
            "bed temperature",
        ):
            return "diag"

    if equation.kind == "total concentration":
        if variable.kind == "gas concentration":
            # The radial bulk flux is inverted from the Ergun law, whose
            # inertial term carries the local density, hence the composition
            # of the radial neighbours.
            return "5pt"
        if variable.kind == "solid loading":
            return "diag"
        if variable.kind in ("total concentration", "bed temperature"):
            # Radial convection of the total concentration: the radial
            # velocity follows grad(p) = grad(c_tot * T), so both fields
            # couple to their radial neighbours.
            return "5pt"

    if equation.kind == "bed temperature":
        if variable.kind == "gas concentration":
            # The sensible-energy accumulation source consumes the complete
            # gas-rate balance, including radial species dispersion.
            return "5pt"
        if variable.kind == "solid loading":
            return "diag"
        if variable.kind == "total concentration":
            # Gas mole fractions divide by total concentration before radial
            # dispersion enters the sensible-energy accumulation source.
            return "5pt"
        if variable.kind == "bed temperature":
            return "5pt"
    return None


def generate_sparsity_pattern(layout: StateLayout, as_dense: bool = False):
    """Generate a conservative Jacobian mask from one complete state layout."""
    if not isinstance(layout, StateLayout):
        raise TypeError("`layout` must be a StateLayout instance.")
    Nz, Nr = layout.nz, layout.nr
    N = Nz * Nr
    n_total = layout.state_size
    fields_2d = layout.fields_in("y_2d")

    cell_idx = np.arange(N, dtype=np.int32)
    iz_all = cell_idx // Nr
    ir_all = cell_idx % Nr

    stencil_entries = {}
    stencil_entries["diag"] = (cell_idx, cell_idx)

    rows_3ax = [cell_idx]
    cols_3ax = [cell_idx]
    mask_zp = iz_all < Nz - 1
    rows_3ax.append(cell_idx[mask_zp])
    cols_3ax.append(cell_idx[mask_zp] + Nr)
    mask_zm = iz_all > 0
    rows_3ax.append(cell_idx[mask_zm])
    cols_3ax.append(cell_idx[mask_zm] - Nr)
    stencil_entries["3ax"] = (np.concatenate(rows_3ax), np.concatenate(cols_3ax))

    rows_5pt = list(rows_3ax)
    cols_5pt = list(cols_3ax)
    mask_rp = ir_all < Nr - 1
    rows_5pt.append(cell_idx[mask_rp])
    cols_5pt.append(cell_idx[mask_rp] + 1)
    mask_rm = ir_all > 0
    rows_5pt.append(cell_idx[mask_rm])
    cols_5pt.append(cell_idx[mask_rm] - 1)
    stencil_entries["5pt"] = (np.concatenate(rows_5pt), np.concatenate(cols_5pt))

    rows_9pt = list(rows_5pt)
    cols_9pt = list(cols_5pt)
    for dz in (-1, 1):
        for dr in (-1, 1):
            mask = ((iz_all + dz >= 0) & (iz_all + dz < Nz)
                    & (ir_all + dr >= 0) & (ir_all + dr < Nr))
            rows_9pt.append(cell_idx[mask])
            cols_9pt.append(cell_idx[mask] + dz*Nr + dr)
    stencil_entries["9pt"] = (np.concatenate(rows_9pt), np.concatenate(cols_9pt))

    all_rows = []
    all_cols = []

    for equation in fields_2d:
        row_base = equation.start
        for variable in fields_2d:
            stype = _field_stencil(equation, variable)
            if stype is None:
                continue
            col_base = variable.start
            r_off, c_off = stencil_entries[stype]
            all_rows.append(r_off + row_base)
            all_cols.append(c_off + col_base)

    wall = layout.field("T_wall")
    bed_temperature = layout.field("T_bed")
    Tw_start = wall.start
    Tbed_start = bed_temperature.start

    iz_arr = np.arange(Nz, dtype=np.int32)
    all_rows.append(Tw_start + iz_arr)
    all_cols.append(Tw_start + iz_arr)
    all_rows.append(Tw_start + iz_arr[:-1])
    all_cols.append(Tw_start + iz_arr[:-1] + 1)
    all_rows.append(Tw_start + iz_arr[1:])
    all_cols.append(Tw_start + iz_arr[1:] - 1)

    wall_bed_radial_indices = [Nr - 1]
    if Nr > 1:
        # The wall-face interpolation used by the 2D energy balance contains
        # both the wall-adjacent cell and its inner radial neighbour.
        wall_bed_radial_indices.append(Nr - 2)
    for ir in wall_bed_radial_indices:
        all_rows.append(Tw_start + iz_arr)
        all_cols.append(Tbed_start + iz_arr * Nr + ir)
    all_rows.append(Tbed_start + iz_arr * Nr + (Nr - 1))
    all_cols.append(Tw_start + iz_arr)

    ir_range = np.arange(Nr, dtype=np.int32)

    boundary_fields = tuple(
        field
        for field in fields_2d
        if field.kind in (
            "gas concentration",
            "total concentration",
            "bed temperature",
        )
    )
    boundary_z = np.unique(
        [
            iz
            for iz in (0, 1, 2, Nz - 3, Nz - 2, Nz - 1)
            if 0 <= iz < Nz
        ]
    )
    for scalar in layout.fields_in("y_scalars"):
        if scalar.name == "E_jacket":
            all_rows.append(np.full(Nz, scalar.start, dtype=np.int32))
            all_cols.append(Tw_start + iz_arr)
            continue
        for variable in boundary_fields:
            for iz in boundary_z:
                all_rows.append(np.full(Nr, scalar.start, dtype=np.int32))
                all_cols.append(variable.start + iz * Nr + ir_range)

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    data = np.ones(len(rows), dtype=bool)

    S = coo_matrix((data, (rows, cols)), shape=(n_total, n_total))
    S = csc_matrix(S, dtype=bool)
    S.sum_duplicates()

    if as_dense:
        return S.toarray()
    return S



def field_boundaries(layout: StateLayout) -> list[int]:
    """Return the state indices where one field ends and the next begins."""
    return [
        field.stop for field in layout.fields if 0 < field.stop < layout.state_size
    ]


def plot_sparsity(
    mask,
    layout: StateLayout,
    out_pdf: str | Path,
    title: str | None = None,
    markersize: float = 0.5,
) -> Path:
    """
    Draw a Jacobian mask with the field boundaries of `layout` and save it.

    `mask` may be sparse or dense. The figure is written only to `out_pdf`;
    its folder is created if needed. Returns the path written.
    """
    from matplotlib.figure import Figure

    mask = csc_matrix(mask, dtype=bool)
    if mask.shape != (layout.state_size, layout.state_size):
        raise ValueError(
            f"Mask shape {mask.shape} does not match the layout state size "
            f"{layout.state_size}."
        )
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(8, 8))
    ax = fig.subplots()
    ax.spy(mask, markersize=markersize)
    for index in field_boundaries(layout):
        ax.axvline(index, color="red", linewidth=0.3)
        ax.axhline(index, color="red", linewidth=0.3)
    ax.set_title(title or f"Jacobian sparsity - Nz{layout.nz}_Nr{layout.nr}")
    ax.set_xlabel("State index (column)")
    ax.set_ylabel("State index (row)")
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    return out_pdf


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Draw the analytical Jacobian mask of the default DAC layout."
    )
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--nr", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    args = parser.parse_args(argv)

    layout = current_dac_state_layout(args.nz, args.nr)
    mask = generate_sparsity_pattern(layout)
    tag = f"Nz{args.nz}_Nr{args.nr}"
    path = plot_sparsity(
        mask,
        layout,
        args.out / f"jacobian_sparsity_analytical_{tag}.pdf",
        title=f"Analytical Jacobian sparsity - {tag}",
    )
    density = mask.nnz / layout.state_size**2 * 100.0
    print(
        f"{tag}: n={layout.state_size}, nnz={mask.nnz}, "
        f"density={density:.4f}% -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
