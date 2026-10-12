"""Stable facade for NSGA-II logging and warm-start utilities."""

from pydac.utils.nsga2.logging import (
    CURRENT_GEN,
    WORKER_STDOUT_LOCK,
    NSGA2Logger,
    Tee,
    generation_evaluation_durations,
    summarize_evaluation_durations,
    termination_diagnostics,
    write_evaluation_timing_summary,
)
from pydac.utils.nsga2.warm_start import WarmStartSpec, build_initial_population
