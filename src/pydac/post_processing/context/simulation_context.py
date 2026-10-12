"""Validated, immutable snapshot of the resolved simulation configuration.

The context stores static bed parameters and step-specific operating
parameters in a canonical DataFrame. It provides stable access to persisted
simulation metadata and prevents incompatible results from being merged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd

from pydac.utils.postprocessing import context_schema as _context_schema
from pydac.utils.postprocessing.context_model import (
    simulation_context_dataframe_from_model,
)


ARCHITECTURE_STATIC_PARAMETERS = _context_schema.ARCHITECTURE_STATIC_PARAMETERS
COMMON_STATIC_PARAMETERS = _context_schema.COMMON_STATIC_PARAMETERS
ContextParameter = _context_schema.ContextParameter
REQUIRED_STATIC_PARAMETER_IDS = _context_schema.REQUIRED_STATIC_PARAMETER_IDS
REQUIRED_STEP_PARAMETER_IDS = _context_schema.REQUIRED_STEP_PARAMETER_IDS
SIMULATION_CONTEXT_COLUMNS = _context_schema.SIMULATION_CONTEXT_COLUMNS
SIMULATION_CONTEXT_SCHEMA_VERSION = _context_schema.SIMULATION_CONTEXT_SCHEMA_VERSION
STATIC_STEP = _context_schema.STATIC_STEP
STEP_ORDER = _context_schema.STEP_ORDER

_definition_row = _context_schema.context_definition_row
_finite_number = _context_schema.finite_context_number
_nonempty_text = _context_schema.nonempty_context_text
validate_simulation_context_dataframe = (
    _context_schema.validate_simulation_context_dataframe
)


@dataclass(frozen=True, slots=True, init=False)
class SimulationContext:
    """Immutable static standardized and per-step context for the current pyDAC framework."""

    _dataframe: pd.DataFrame = field(repr=False)
    schema_version: int = SIMULATION_CONTEXT_SCHEMA_VERSION

    REQUIRED_STATIC_PARAMETER_IDS: ClassVar[tuple[str, ...]] = (
        REQUIRED_STATIC_PARAMETER_IDS
    )
    REQUIRED_STEP_PARAMETER_IDS: ClassVar[tuple[str, ...]] = (
        REQUIRED_STEP_PARAMETER_IDS
    )

    _validate_dataframe = staticmethod(validate_simulation_context_dataframe)

    def __init__(
        self,
        dataframe,
        schema_version=SIMULATION_CONTEXT_SCHEMA_VERSION,
    ):
        if schema_version != SIMULATION_CONTEXT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported SimulationContext schema version {schema_version}; "
                f"only version {SIMULATION_CONTEXT_SCHEMA_VERSION} from the "
                "current pyDAC framework is supported. Regenerate the result."
            )
        validated = self._validate_dataframe(dataframe)
        object.__setattr__(self, "_dataframe", validated)
        object.__setattr__(self, "schema_version", schema_version)

    @classmethod
    def from_model(cls, model):
        """Capture the actual resolved context from one initialized step model."""
        return cls(simulation_context_dataframe_from_model(model))

    @property
    def architecture(self):
        """Return the explicitly persisted bed architecture."""
        return self.text("bed.architecture")

    @property
    def steps(self):
        """Return persisted step blocks in canonical cycle order."""
        return tuple(
            step
            for step in STEP_ORDER
            if self._dataframe["Step"].eq(step).any()
        )

    @property
    def species(self):
        """Return persisted gas species in declared mixture order."""
        return tuple(
            name.upper()
            for name in _context_schema.simulation_context_species(self._dataframe)
        )

    @property
    def required_step_parameter_ids(self):
        """Return the complete dynamic per-step schema for this mixture."""
        return tuple(
            definition.parameter_id
            for definition in _context_schema.step_parameters(self.species)
        )

    def to_dataframe(self):
        """Return a deep copy of the canonical context table."""
        return self._dataframe.copy(deep=True)

    def _raw_value(self, parameter_id, *, step=None):
        """Retrieve the raw stored value for one static or step-specific parameter."""
        token = str(parameter_id).strip().lower()
        step_token = STATIC_STEP if step is None else str(step).strip().lower()
        matches = self._dataframe.loc[
            self._dataframe["Parameter ID"].eq(token)
            & self._dataframe["Step"].eq(step_token),
            "Value",
        ]
        if matches.empty:
            scope = "static context" if step is None else f"step '{step_token}'"
            raise KeyError(
                f"Unknown simulation-context parameter '{parameter_id}' in {scope}."
            )
        return matches.iloc[0]

    def value(self, parameter_id, *, step=None):
        """Return one numeric context value by stable identity and optional step."""
        value = self._raw_value(parameter_id, step=step)
        return _finite_number(value, str(parameter_id))

    def text(self, parameter_id, *, step=None):
        """Return one textual context value by stable identity and optional step."""
        value = self._raw_value(parameter_id, step=step)
        return _nonempty_text(value, str(parameter_id))

    @property
    def adsorbent_mass(self):
        """Return adsorbent mass [kg] from persisted resolved inventory."""
        return self.value("bed.volume") * self.value(
            "bed.resolved.rho_sorbent"
        )

    def merged_with(self, other):
        """Merge compatible step contexts.

        Static parameters must match exactly. Step blocks present in both
        contexts must also be identical; otherwise, an error is raised.
        """
        if not isinstance(other, SimulationContext):
            raise TypeError("other must be a SimulationContext instance.")

        left_static = self._dataframe.loc[
            self._dataframe["Step"].eq(STATIC_STEP)
        ].reset_index(drop=True)
        right_static = other._dataframe.loc[
            other._dataframe["Step"].eq(STATIC_STEP)
        ].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                left_static,
                right_static,
                check_exact=True,
            )
        except AssertionError as exc:
            raise ValueError(
                "Static simulation context differs from the existing result folder. "
                "Use a separate results folder."
            ) from exc

        frames = [self._dataframe]
        for step in other.steps:
            right_step = other._dataframe.loc[
                other._dataframe["Step"].eq(step)
            ].reset_index(drop=True)
            if step in self.steps:
                left_step = self._dataframe.loc[
                    self._dataframe["Step"].eq(step)
                ].reset_index(drop=True)
                try:
                    pd.testing.assert_frame_equal(
                        left_step,
                        right_step,
                        check_exact=True,
                    )
                except AssertionError as exc:
                    raise ValueError(
                        f"Simulation context for step '{step}' differs from the "
                        "existing result folder. Use a separate results folder."
                    ) from exc
            else:
                frames.append(right_step)

        merged = pd.concat(frames, ignore_index=True)
        return SimulationContext(merged)
