"""Spatial result-folder and saved-step path resolution for TVSA runs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydac.config import GridConfig

from pydac.utils.tvsa.steps import normalize_step_token


def _is_1d_model(model):
    if hasattr(model, "is_1d"):
        return bool(model.is_1d)
    return int(model.Nr) == 1


def _resolve_grid_config(grid_config):
    from pydac.utils.run.config import resolve_grid_config

    return resolve_grid_config(grid_config)


def resolve_spatial_base_folder(
    base_folder,
    *,
    model=None,
    base_folder_is_spatial_root=False,
    create=True,
    grid_config: GridConfig | None = None,
):
    """Resolve the result root for the model's 1D/2D spatial mode."""
    if base_folder_is_spatial_root:
        path = base_folder
    else:
        if model is not None:
            mode_tag = "1D_mode" if _is_1d_model(model) else "2D_mode"
        else:
            grid = _resolve_grid_config(grid_config)
            mode_tag = "1D_mode" if grid.nr == 1 else "2D_mode"
        path = os.path.join(base_folder, mode_tag)

    if create:
        os.makedirs(path, exist_ok=True)
    return path


def resolve_spatial_base_folder_from_solver_settings(
    base_folder,
    *,
    base_folder_is_spatial_root=False,
    create=True,
    grid_config: GridConfig | None = None,
):
    """Resolve the result root before a column model is constructed."""
    return resolve_spatial_base_folder(
        base_folder,
        base_folder_is_spatial_root=base_folder_is_spatial_root,
        create=create,
        grid_config=grid_config,
    )


def get_step_solution_path(
    base_folder,
    step_name,
    *,
    base_folder_is_spatial_root=False,
    grid_config: GridConfig | None = None,
):
    """Return the saved ``sol.pkl`` path for a canonical TVSA step."""
    step_key = normalize_step_token(step_name)
    results_base_folder = resolve_spatial_base_folder_from_solver_settings(
        base_folder,
        base_folder_is_spatial_root=base_folder_is_spatial_root,
        grid_config=grid_config,
    )
    return os.path.join(results_base_folder, step_key, "sol.pkl")


__all__ = [
    "get_step_solution_path",
    "resolve_spatial_base_folder",
    "resolve_spatial_base_folder_from_solver_settings",
]
