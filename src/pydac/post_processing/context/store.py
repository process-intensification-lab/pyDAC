"""Persistence adapter for architecture-aware simulation context."""

from pathlib import Path
import pickle

import pandas as pd

from pydac.post_processing.context.simulation_context import (
    SIMULATION_CONTEXT_SCHEMA_VERSION,
    SimulationContext,
)


CONTEXT_FILENAME = "simulation_context.pkl"


class SimulationContextStore:
    """Save and load one validated, incrementally merged cycle context."""

    def __init__(self, base_folder):
        self.base_folder = Path(base_folder)

    @property
    def path(self):
        return self.base_folder / CONTEXT_FILENAME

    def exists(self):
        return self.path.is_file()

    def validate_compatible(self, context):
        """Check whether a context can be saved without changing the file."""
        if not isinstance(context, SimulationContext):
            raise TypeError("context must be a SimulationContext instance.")
        if not self.exists():
            return context
        return self.load().merged_with(context)

    def save(self, context):
        if not isinstance(context, SimulationContext):
            raise TypeError("context must be a SimulationContext instance.")

        if self.exists():
            context = self.load().merged_with(context)

        self.base_folder.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as handle:
            pickle.dump(context.to_dataframe(), handle)
        print(f"Simulation context successfully saved to '{self.path}'")
        return self.path

    def load(self):
        if not self.exists():
            raise FileNotFoundError(f"No saved simulation context found at '{self.path}'.")

        with self.path.open("rb") as handle:
            dataframe = pickle.load(handle)
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame in '{self.path}'.")
        schema_version = dataframe.attrs.get("schema_version")
        if schema_version != SIMULATION_CONTEXT_SCHEMA_VERSION:
            raise ValueError(
                f"Simulation context at '{self.path}' uses unsupported schema "
                f"version {schema_version!r}. Only current pyDAC schema version "
                f"{SIMULATION_CONTEXT_SCHEMA_VERSION} is supported; regenerate "
                "the result."
            )
        return SimulationContext(dataframe, schema_version=schema_version)
