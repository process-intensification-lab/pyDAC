"""Validated aggregate passed from cycle loading to KPI calculations."""

from __future__ import annotations

from dataclasses import dataclass

from pydac.post_processing.context.simulation_context import SimulationContext

from pydac.post_processing.metrics.cycle_metrics import CycleMetrics


@dataclass(frozen=True, slots=True)
class CycleData:
    """One cycle's dynamic metrics and required architecture-aware context."""

    metrics: CycleMetrics
    context: SimulationContext
    provenance: list | None = None

    def __post_init__(self):
        if not isinstance(self.metrics, CycleMetrics):
            raise TypeError("metrics must be a CycleMetrics instance.")
        if not isinstance(self.context, SimulationContext):
            raise TypeError("context must be a SimulationContext instance.")
