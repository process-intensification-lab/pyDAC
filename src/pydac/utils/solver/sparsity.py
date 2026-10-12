"""CVODE sparsity-pattern generation, pruning, and solver configuration."""

from functools import lru_cache

from scipy.sparse import csc_matrix

try:
    from scipy.optimize._numdiff import group_columns
except Exception:  # pragma: no cover - diagnostic-only fallback
    group_columns = None

from pydac.core.sparsity_patterns.analytical_sparsity import generate_sparsity_pattern


DEFAULT_CVODE_SPARSITY_MODE = "pde_only"
SPARSITY_MODE_ALIASES = {
    "": DEFAULT_CVODE_SPARSITY_MODE,
    "default": DEFAULT_CVODE_SPARSITY_MODE,
    "legacy": "full",
    "full": "full",
    "pde": "pde_only",
    "pde_only": "pde_only",
    "drop_kpi": "pde_only",
    "drop_kpi_rows": "pde_only",
}


@lru_cache(maxsize=64)
def _analytical_sparsity(layout):
    """Generate the analytical mask once per state layout."""
    return generate_sparsity_pattern(layout)


def try_get_solver_sparsity(*, step, layout):
    """
    Return the analytical sparsity pattern for the current step/grid.

    The mask is generated in memory from the state layout and never read from
    or written to disk. Every step shares the same mask; `step` is kept for the
    messages. A copy is returned so a caller can never alter the cached mask.
    """
    return _analytical_sparsity(layout).copy(), "analytical", "generated in memory"


def normalize_cvode_sparsity_mode(sparsity_mode):
    """
    Normalize the advanced CVODE sparse-Jacobian mask mode.
    """
    if sparsity_mode is None:
        return DEFAULT_CVODE_SPARSITY_MODE

    requested = str(sparsity_mode).strip().lower()
    if requested not in SPARSITY_MODE_ALIASES:
        valid = ", ".join(sorted(set(SPARSITY_MODE_ALIASES) - {""}))
        raise ValueError(f"Unsupported cvode_sparsity_mode '{sparsity_mode}'. Valid options: {valid}.")
    return SPARSITY_MODE_ALIASES[requested]


def pde_state_size(layout):
    """
    Return the number of non-KPI states in the packed-bed state vector.
    """
    return int(layout.pde_size)


def estimate_sparsity_colors(sparsity):
    """
    Estimate the finite-difference coloring count used by scikit-sundae.
    """
    if group_columns is None:
        return None
    try:
        groups = group_columns(sparsity)
    except Exception:
        return None
    if groups.size == 0:
        return 0
    return int(groups.max()) + 1


def prepare_cvode_solver_sparsity(sparsity, *, layout, state_size, sparsity_mode):
    """
    Return the actual sparsity mask passed to CVODE.

    `full` preserves the legacy analytical mask. The default `pde_only` keeps
    the full matrix shape, but drops the passive KPI rows from the Jacobian mask
    used by CVODE's colored finite-difference Jacobian builder. The KPI states
    are still integrated by the RHS; only their Newton linearization rows are
    omitted.
    """
    mode = normalize_cvode_sparsity_mode(sparsity_mode)
    solver_sparsity = csc_matrix(sparsity, dtype=bool)

    if mode == "full":
        return solver_sparsity, mode, None

    non_kpi_size = pde_state_size(layout)
    if non_kpi_size > state_size:
        raise ValueError(
            f"Computed non-KPI state size {non_kpi_size} exceeds total state size {state_size}."
        )
    if non_kpi_size == state_size:
        return solver_sparsity, mode, {
            "non_kpi_size": non_kpi_size,
            "dropped_rows": 0,
            "original_nnz": int(solver_sparsity.nnz),
        }

    pruned = solver_sparsity.tolil(copy=True)
    for row in range(non_kpi_size, state_size):
        pruned.rows[row] = []
        pruned.data[row] = []

    pruned = pruned.tocsc()
    pruned.eliminate_zeros()
    return pruned, mode, {
        "non_kpi_size": non_kpi_size,
        "dropped_rows": state_size - non_kpi_size,
        "original_nnz": int(solver_sparsity.nnz),
    }


def configure_cvode_linear_solver_options(
    options,
    *,
    use_sparsity,
    sparsity_mode,
    step,
    state_size,
    layout,
):
    """
    Add the CVODE linear-solver options for one step.

    Every step is solved with the direct sparse factorization on the analytical
    mask; `use_sparsity=False` selects the dense diagnostic path instead. This
    mutates `options` in place and returns the name of the solver that will
    run, `sparse` or `dense`, so the caller can record it.
    """
    if not use_sparsity:
        options["linsolver"] = "dense"
        print(
            "CVODE warning        : use_sparsity=False disables the recommended "
            "analytical sparse solver and is intended only for diagnostics."
        )
        print("CVODE linear solver : dense (diagnostic non-sparse path)")
        return "dense"

    return configure_sparse_linsolver(
        options,
        step=step,
        state_size=state_size,
        layout=layout,
        sparsity_mode=sparsity_mode,
    )


def configure_sparse_linsolver(
    options,
    *,
    step,
    state_size,
    layout,
    sparsity_mode,
):
    """
    Configure CVODE's direct sparse linear solver for one step.

    Returns `sparse`, the name of the solver that will run.
    """
    sparsity, source, sparsity_origin = try_get_solver_sparsity(step=step, layout=layout)
    if sparsity.shape != (state_size, state_size):
        raise ValueError(
            f"Sparsity matrix shape {sparsity.shape} does not match state size "
            f"{(state_size, state_size)}."
        )

    solver_sparsity, sparsity_mode, pruning_info = prepare_cvode_solver_sparsity(
        sparsity,
        layout=layout,
        state_size=state_size,
        sparsity_mode=sparsity_mode,
    )

    density_pct = sparsity.nnz / (sparsity.shape[0] * sparsity.shape[1]) * 100.0
    solver_density_pct = (
        solver_sparsity.nnz
        / (solver_sparsity.shape[0] * solver_sparsity.shape[1])
        * 100.0
    )
    color_count = estimate_sparsity_colors(solver_sparsity)
    color_msg = "unknown" if color_count is None else str(color_count)
    print(
        "CVODE linear solver : sparse (direct sparse)\n"
        f"Using {source} sparsity pattern ({sparsity_origin}) "
        f"with shape {sparsity.shape}, nnz={sparsity.nnz}, density={density_pct:.3f}%\n"
        f"CVODE sparsity mode : {sparsity_mode}, solver nnz={solver_sparsity.nnz}, "
        f"density={solver_density_pct:.3f}%, estimated colors={color_msg}"
    )
    if pruning_info is not None and pruning_info["dropped_rows"]:
        print(
            "CVODE sparsity note : passive KPI rows "
            f"{pruning_info['non_kpi_size']}:{state_size} are omitted from the solver Jacobian mask."
        )

    options.update({
        "linsolver": "sparse",
        "sparsity": solver_sparsity,
        "nthreads": 1,
    })
    return "sparse"
