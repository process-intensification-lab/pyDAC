"""Stable facade for high-level run configuration utilities."""

from pydac.core.bed_architectures import build_bed_config
from pydac.utils.run.config import (
    make_base_folder,
    resolve_grid_config,
)
from pydac.utils.run.steps import (
    DESORPTION_STEP_ALIASES,
    GENERIC_DESORPTION_STEP_TOKENS,
    infer_desorption_step,
    infer_desorption_step_from_results,
    infer_results_base_folder,
    is_steam_assisted_desorption_step,
    normalize_desorption_step,
    normalize_step_token,
    ordered_saved_step_folders,
)
from pydac.utils.run.validation import (
    SUPPORTED_STEP_OPTION_KEYS,
    SUPPORTED_STEP_TOKENS,
    SUPPORTED_TERMINATION_MODES,
    normalize_termination_mode,
    validate_flags,
    validate_output_options,
    validate_step_runner_configuration,
    validate_step_options_structure,
    warn_advanced_sparsity_options,
)


def current_grid_tag(grid_config=None) -> str:
    """Return a campaign-folder tag derived from one typed solver grid."""
    grid = resolve_grid_config(grid_config)
    return f"1D_{grid.nz}z" if grid.nr == 1 else f"2D_{grid.nz}x{grid.nr}"
