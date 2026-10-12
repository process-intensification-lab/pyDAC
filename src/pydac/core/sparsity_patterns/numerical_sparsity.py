"""
pyDAC - Numerical Jacobian sparsity aggregation.

With `execution.compute_numerical_sparsity=true` the solver saves dense
finite-difference Jacobians of a step into its results folder,
`<step folder>/numerical_jacobian/jacobian_sparsity_<tick>.npy`. This module
unions those snapshots into one numerical sparsity mask, draws it, and checks
it against the analytical mask: every coupling found numerically must be in
the analytical mask, otherwise the solver's mask misses a dependency.

    python -m pydac.core.sparsity_patterns.numerical_sparsity STEP_FOLDER --nz 10 --nr 10

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-10-01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pydac.core.sparsity_patterns.analytical_sparsity import (
    generate_sparsity_pattern,
    plot_sparsity,
)
from pydac.core.state_layout import StateLayout, current_dac_state_layout

SNAPSHOT_FOLDER = "numerical_jacobian"


def sparsity_from_J(J, rel_tol=1e-8, abs_tol=1e-12):
    """
    Column-wise relative threshold + absolute threshold:
      keep (i,j) if |J_ij| > abs_tol AND |J_ij| > rel_tol * max_i |J_ij|
    """
    absJ = np.abs(J)
    col_max = absJ.max(axis=0)
    col_max[col_max == 0.0] = 1.0
    return (absJ > abs_tol) & (absJ > (rel_tol * col_max))


def aggregate_numerical_sparsity(
    step_folder: str | Path,
    layout: StateLayout,
    *,
    rel_tol: float = 1e-16,
    abs_tol: float = 0.0,
    save_plot: bool = True,
) -> dict:
    """
    Union the Jacobian snapshots of one step and compare with the analytical mask.

    Reads `<step_folder>/numerical_jacobian/jacobian_sparsity_*.npy` and writes
    the union mask (`sparsity_union_bool.npy`) and, if requested, its spy plot
    into the same folder. Returns the counts; `missing` (numerical couplings
    absent from the analytical mask) must be 0.
    """
    snapshot_dir = Path(step_folder) / SNAPSHOT_FOLDER
    files = sorted(snapshot_dir.glob("jacobian_sparsity_*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No Jacobian snapshots in {snapshot_dir}. Run the step with "
            "execution.compute_numerical_sparsity=true first."
        )

    expected_shape = (layout.state_size, layout.state_size)
    union = None
    densities = []
    for path in files:
        J = np.load(path)
        if J.shape != expected_shape:
            raise ValueError(
                f"Jacobian {path.name} has shape {J.shape}; the layout "
                f"{layout.signature[:16]} expects {expected_shape}."
            )
        S = sparsity_from_J(J, rel_tol=rel_tol, abs_tol=abs_tol)
        densities.append(S.sum() / S.size)
        union = S if union is None else (union | S)

    analytical = generate_sparsity_pattern(layout, as_dense=True)
    missing = int((union & ~analytical).sum())
    extra = int((analytical & ~union).sum())
    np.save(snapshot_dir / "sparsity_union_bool.npy", union)

    plot_path = None
    if save_plot:
        plot_path = plot_sparsity(
            union,
            layout,
            snapshot_dir / "jacobian_sparsity_union.pdf",
            title=f"Numerical Jacobian sparsity union - {Path(step_folder).name}",
            markersize=1,
        )

    summary = {
        "snapshots": len(files),
        "nnz_union": int(union.sum()),
        "density_range": (float(min(densities)), float(max(densities))),
        "missing_from_analytical": missing,
        "analytical_extra": extra,
        "plot": None if plot_path is None else str(plot_path),
    }
    status = "PASS" if missing == 0 else f"FAIL ({missing} numerical couplings missing)"
    print(
        f"{Path(step_folder).name}: {len(files)} snapshots, union nnz={summary['nnz_union']}, "
        f"analytical extra={extra} -> {status}"
    )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Union a step's Jacobian snapshots and check them against "
        "the analytical mask (default DAC layout)."
    )
    parser.add_argument("step_folder", type=Path)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--nr", type=int, required=True)
    args = parser.parse_args(argv)

    summary = aggregate_numerical_sparsity(
        args.step_folder, current_dac_state_layout(args.nz, args.nr)
    )
    return 0 if summary["missing_from_analytical"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
