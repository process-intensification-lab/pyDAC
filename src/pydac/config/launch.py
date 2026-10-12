"""Shared resolution bridge between a public document and one running cycle.

A workflow receives the resolved :class:`config.schema.ExperimentConfig` and
needs two things from it: the document to save beside the results, and the
runtime :class:`config.cycle.CycleConfig` the factory and the model consume.
Both come from here, so the two never disagree — and the execution overrides a
workflow wants must already be in the document before this boundary, which is
what makes the saved configuration a faithful record of the run.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : the workflow boundary — every public workflow starts here.
# Owns     : returning the (document, runtime view) pair as ONE act, so the
#            `resolved_config.json` on disk always describes what ran.
# Not here : running anything; deciding anything about the case.
# Entry    : resolve_experiment_config()
# Guide    : docs/manual/01_inputs.md §3.3, §8.3
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from pydac.config.cycle import CycleConfig
from pydac.config.schema import ExperimentConfig


def resolve_experiment_config(
    *,
    experiment_config: ExperimentConfig,
) -> tuple[ExperimentConfig, CycleConfig]:
    """Resolve the typed experiment and its runtime campaign view together."""
    if not isinstance(experiment_config, ExperimentConfig):
        raise TypeError("`experiment_config` must be an ExperimentConfig instance.")

    return experiment_config, CycleConfig.from_experiment(experiment_config)


__all__ = ["resolve_experiment_config"]
