"""Shared TVSA step, path, termination, and reporting utilities."""

from pydac.utils.tvsa.paths import (
    get_step_solution_path,
    resolve_spatial_base_folder,
    resolve_spatial_base_folder_from_solver_settings,
)
from pydac.utils.tvsa.reporting import (
    print_cycle_build_summary,
    print_cycle_completion,
    print_cycle_timing_summary,
)
from pydac.utils.tvsa.cycle_steps import (
    DEFAULT_CYCLIC_SEQUENCE,
    DEFAULT_STEP_MODE_SEQUENCE,
    build_cycle_steps,
    build_cyclic_cycle_steps,
    build_repeating_cycle_steps,
)
from pydac.utils.tvsa.steps import (
    CANONICAL_STEP_KEYS,
    DESORPTION_STEP_ALIASES,
    GENERIC_DESORPTION_STEP_TOKENS,
    STEP_ALIAS_BY_TOKEN,
    STEP_LABEL_BY_KEY,
    STEP_UNIT_BY_KEY,
    SUPPORTED_STEP_TOKENS,
    normalize_desorption_step,
    normalize_step_options,
    normalize_step_token,
    normalize_step_token_text,
)
from pydac.utils.tvsa.termination import (
    EVENT_METRIC_REQUIREMENTS,
    EVENT_METRICS,
    SUPPORTED_TERMINATION_MODES,
    THRESHOLD_SOURCES,
    describe_step_termination,
    describe_termination_setup_from_model,
    describe_termination_setup_from_summary,
    format_termination_outcome,
    normalize_termination_mode,
)

__all__ = [
    "CANONICAL_STEP_KEYS",
    "DEFAULT_CYCLIC_SEQUENCE",
    "DEFAULT_STEP_MODE_SEQUENCE",
    "DESORPTION_STEP_ALIASES",
    "GENERIC_DESORPTION_STEP_TOKENS",
    "STEP_ALIAS_BY_TOKEN",
    "STEP_LABEL_BY_KEY",
    "STEP_UNIT_BY_KEY",
    "SUPPORTED_STEP_TOKENS",
    "build_cycle_steps",
    "build_cyclic_cycle_steps",
    "build_repeating_cycle_steps",
    "SUPPORTED_TERMINATION_MODES",
    "EVENT_METRIC_REQUIREMENTS",
    "EVENT_METRICS",
    "THRESHOLD_SOURCES",
    "describe_step_termination",
    "describe_termination_setup_from_model",
    "describe_termination_setup_from_summary",
    "format_termination_outcome",
    "get_step_solution_path",
    "normalize_desorption_step",
    "normalize_step_options",
    "normalize_step_token",
    "normalize_step_token_text",
    "normalize_termination_mode",
    "print_cycle_build_summary",
    "print_cycle_completion",
    "print_cycle_timing_summary",
    "resolve_spatial_base_folder",
    "resolve_spatial_base_folder_from_solver_settings",
]
