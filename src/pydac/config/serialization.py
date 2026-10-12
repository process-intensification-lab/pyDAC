"""JSON-safe serialization for typed experiment configuration records.

The inverse direction of the loader: typed records back to the document the
user reads. Because resolution is idempotent, the output is also a complete
frozen-case document and is re-runnable with ``--case``.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : the way back out — typed records -> JSON-safe document.
# Owns     : the canonical external layout (steps keyed by name, not a list).
# Not here : file writing (loader.write_resolved_config), field meanings.
# Entry    : experiment_to_document()
# Guide    : docs/manual/01_inputs.md §8.3
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Mapping

from pydac.config.schema import SIMULATION_SECTIONS, ExperimentConfig


def to_jsonable(value: Any) -> Any:
    """Convert frozen dataclasses and immutable mappings to JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def experiment_to_document(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize an experiment using the canonical external JSON layout.

    The cycle is the one section that does not map field-for-field: steps are
    keyed by name so a JSON document can override a single step.
    """
    if not isinstance(config, ExperimentConfig):
        raise TypeError("`config` must be an ExperimentConfig instance.")

    simulation = config.simulation
    sections: dict[str, Any] = {
        name: to_jsonable(getattr(simulation, name)) for name in SIMULATION_SECTIONS
    }
    sections["cycle"] = {
        "steps": {step.name: to_jsonable(step.config) for step in simulation.cycle.steps}
    }
    return {
        "schema_version": config.schema_version,
        "simulation": sections,
        "execution": to_jsonable(config.execution),
    }


__all__ = ["experiment_to_document", "to_jsonable"]
