"""Saved-solution profile extraction and CSS convergence checks."""

import pickle
from typing import Dict, Literal, Tuple, Union, overload

import numpy as np

from pydac.utils.postprocessing.saved_solution import saved_solution_view


CSSProfile = Dict[str, np.ndarray]
CSSDiagnostics = Dict[str, Dict[str, Union[float, int]]]


def _load_solution(sol_path):
    with open(sol_path, "rb") as f:
        return pickle.load(f)

def extract_final_fields(sol_path):
    """
    Reconstruct the `final_fields` payload expected by non-startup steps.
    """
    sol = _load_solution(sol_path)
    saved = saved_solution_view(sol)
    extra = sol.get("extra", {})

    return {
        "ci": np.stack(
            [saved.field(f"c_{species}")[..., -1]
             for species in saved.layout.independent_gas_species],
            axis=-1,
        ),
        "qi": np.stack(
            [saved.field(f"q_{species}")[..., -1]
             for species in saved.layout.adsorbates],
            axis=-1,
        ),
        "c_tot": saved.field("c_tot")[..., -1],
        "T": saved.field("T_bed")[..., -1],
        "T_wall": saved.field("T_wall")[..., -1],
        "layout_signature": saved.layout.signature,
        "Nz": int(extra.get("Nz", saved.layout.nz)),
        "Nr": int(extra.get("Nr", saved.layout.nr)),
        "Nt": int(extra.get("Nt", saved.time_s.size)),
        "z_f": np.array(extra.get("z_f", []), copy=True),
        "z_c": np.array(extra.get("z_c", []), copy=True),
        "r_f": np.array(extra.get("r_f", []), copy=True),
        "r_c": np.array(extra.get("r_c", []), copy=True),
    }


def final_profiles(sol_path) -> CSSProfile:
    """
    Extract end-of-step primitive fields for CSS comparison.

    The CSS check operates on primitive internal fields rather than on derived
    quantities such as mole fractions or pressure. This is numerically more
    robust and avoids amplifying tiny changes through algebraic ratios.
    """
    sol = _load_solution(sol_path)
    saved = saved_solution_view(sol)
    return {
        field.name: saved.field(field.name)[..., -1]
        for field in saved.layout.pde_fields
    }


# ---------------------------------------------------------------------------
# CSS convergence check
# ---------------------------------------------------------------------------

@overload
def css_reached(
    prev: CSSProfile | None,
    curr: CSSProfile,
    rel_tol: float = 0.005,
    abs_tol: float = 1e-8,
    return_details: Literal[True] = True,
) -> Tuple[bool, CSSDiagnostics]:
    ...


@overload
def css_reached(
    prev: CSSProfile | None,
    curr: CSSProfile,
    rel_tol: float = 0.005,
    abs_tol: float = 1e-8,
    return_details: Literal[False] = False,
) -> bool:
    ...


def css_reached(
    prev: CSSProfile | None,
    curr: CSSProfile,
    rel_tol=0.005,
    abs_tol=1e-8,
    return_details=False,
):
    """
    Compare previous-cycle and current-cycle fields.

    Relative difference is computed as `|a - b| / |b|`. A field only fails if
    it exceeds both the absolute and relative tolerance somewhere in the domain.
    """
    if prev is None:
        if return_details:
            return False, {}
        return False

    eps = 1e-12
    diagnostics = {}
    reached = True

    for key in curr:
        a = np.asarray(prev[key])
        b = np.asarray(curr[key])

        if a.shape != b.shape:
            raise ValueError(
                f"CSS comparison shape mismatch for '{key}': {a.shape} vs {b.shape}."
            )

        abs_diff = np.abs(a - b)
        rel_diff = abs_diff / (np.abs(b) + eps)
        mask = (abs_diff > abs_tol) & (rel_diff > rel_tol)

        diagnostics[key] = {
            "max_abs_diff": float(np.max(abs_diff)),
            "max_rel_diff": float(np.max(rel_diff)),
            "violations": int(np.count_nonzero(mask)),
        }

        if np.any(mask):
            reached = False

    if return_details:
        return reached, diagnostics
    return reached
