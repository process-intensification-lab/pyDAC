"""Typed contract for scalar metrics produced by one simulated step.

This module contains data definitions and serialization helpers only. Physical
calculations remain the responsibility of ``step_post_processing``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

import pandas as pd

from pydac.utils.postprocessing.normalization import normalize_metric_species


SCHEMA_VERSION = 1
LEGACY_COLUMNS = ("Step", "Species", "KPI", "Value")
INTERNAL_COLUMNS = ("Step", "Species", "Metric ID", "Value", "Unit", "Basis")

_METRIC_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Stable identity and legacy representation of a step metric."""

    metric_id: str
    legacy_name: str
    unit: str
    basis: str | None = None

    def __post_init__(self) -> None:
        metric_id = str(self.metric_id).strip()
        legacy_name = str(self.legacy_name).strip()
        unit = str(self.unit).strip()
        basis = None if self.basis is None else str(self.basis).strip().lower()

        if not _METRIC_ID_PATTERN.fullmatch(metric_id):
            raise ValueError(
                f"Invalid metric_id '{self.metric_id}'. Use lower-case dotted identifiers."
            )
        if not legacy_name:
            raise ValueError("legacy_name must not be empty.")
        if not unit:
            raise ValueError("unit must not be empty; use '-' for dimensionless metrics.")
        if basis == "":
            raise ValueError("basis must be None or a non-empty string.")

        object.__setattr__(self, "metric_id", metric_id)
        object.__setattr__(self, "legacy_name", legacy_name)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "basis", basis)

    @property
    def key(self) -> tuple[str, str | None]:
        return self.metric_id, self.basis


class MetricCatalog:
    """Validated lookup table for metric definitions."""

    def __init__(self, definitions: Iterable[MetricDefinition]):
        self.definitions = tuple(definitions)
        self._by_key: dict[tuple[str, str | None], MetricDefinition] = {}
        self._by_legacy_name: dict[str, MetricDefinition] = {}

        for definition in self.definitions:
            if definition.key in self._by_key:
                raise ValueError(f"Duplicate metric definition key: {definition.key}.")
            if definition.legacy_name in self._by_legacy_name:
                raise ValueError(
                    f"Duplicate legacy metric name: '{definition.legacy_name}'."
                )
            self._by_key[definition.key] = definition
            self._by_legacy_name[definition.legacy_name] = definition

    def get(self, metric_id: str, basis: str | None = None) -> MetricDefinition:
        normalized_basis = None if basis is None else str(basis).strip().lower()
        key = (str(metric_id).strip(), normalized_basis)
        try:
            return self._by_key[key]
        except KeyError as exc:
            raise KeyError(f"Unknown metric definition: {key}.") from exc

    def from_legacy_name(self, legacy_name: str) -> MetricDefinition:
        name = str(legacy_name).strip()
        try:
            return self._by_legacy_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown legacy KPI name: '{legacy_name}'.") from exc


STEP_METRIC_DEFINITIONS = (
    MetricDefinition("grid.nz", "Nz [-]", "-"),
    MetricDefinition("grid.nr", "Nr [-]", "-"),
    MetricDefinition("mass.initial", "initial mass [kg]", "kg"),
    MetricDefinition("mass.change", "mass change [kg]", "kg", basis="integrated"),
    MetricDefinition(
        "mass.change", "state-based mass change [kg]", "kg", basis="state"
    ),
    MetricDefinition(
        "mass.balance.gap",
        "integrated-state mass gap [kg]",
        "kg",
        basis="integrated-state",
    ),
    MetricDefinition(
        "mass.balance.error",
        "integrated-state error [%]",
        "%",
        basis="integrated-state",
    ),
    MetricDefinition("mass.final", "left-over mass [kg]", "kg", basis="integrated"),
    MetricDefinition(
        "mass.final", "state-based left-over mass [kg]", "kg", basis="state"
    ),
    MetricDefinition(
        "mass.gas.final", "final gas-phase mass [kg]", "kg", basis="state"
    ),
    MetricDefinition("mass.after.condenser", "mass after condenser [kg]", "kg"),
    MetricDefinition(
        "mass.after.vacuum", "mass after vacuum discharge [kg]", "kg"
    ),
    # Purge/preheating discharge.  Deliberately NOT "mass.after.vacuum": what a purge
    # vents is lost, not recovered, so it must never be picked up by a product KPI.
    # Signed, positive when the species leaves the bed.
    MetricDefinition("mass.vented", "vented mass [kg]", "kg"),
    MetricDefinition("steam.mass", "Steam Mass [kg]", "kg"),
    MetricDefinition(
        "pressure.drop.average", "Average pressure drop [mbar]", "mbar"
    ),
    MetricDefinition("temperature.bed.minimum", "T_bed_min [K]", "K"),
    MetricDefinition(
        "temperature.steam.condensation", "T_steam_condensation [K]", "K"
    ),
    MetricDefinition("temperature.condenser", "T_condenser [K]", "K"),
    MetricDefinition("step.duration", "Step duration [s]", "s"),
    MetricDefinition("bed.saturation.co2", "Saturation [%]", "%"),
    MetricDefinition("energy.compression", "Compression Energy [MJ]", "MJ"),
    MetricDefinition("energy.vacuum", "Vacuum Pump Energy [MJ]", "MJ"),
    MetricDefinition("energy.jacket", "Jacket Energy [MJ]", "MJ"),
    MetricDefinition("energy.steam", "Steam Energy [MJ]", "MJ"),
    MetricDefinition("energy.condenser", "Condenser Duty [MJ]", "MJ"),
    MetricDefinition(
        "energy.condenser.ambient",
        "Condenser Duty Rejected To Ambient [MJ]",
        "MJ",
    ),
)

STEP_METRIC_CATALOG = MetricCatalog(STEP_METRIC_DEFINITIONS)


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """One scalar metric associated with a species or the overall system."""

    definition: MetricDefinition
    species: str
    value: float
    basis: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.definition, MetricDefinition):
            raise TypeError("definition must be a MetricDefinition instance.")
        if isinstance(self.value, bool):
            raise TypeError("metric values must be numeric, not bool.")
        try:
            value = float(self.value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Metric value must be numeric, got {self.value!r}.") from exc

        basis = self.definition.basis if self.basis is None else str(self.basis).strip().lower()
        if basis == "":
            raise ValueError("basis must be None or a non-empty string.")

        object.__setattr__(
            self,
            "species",
            normalize_metric_species(self.species),
        )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "basis", basis)

    @property
    def key(self) -> tuple[str, str, str | None]:
        return self.species, self.definition.metric_id, self.basis


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Immutable collection of scalar metrics for one simulated step."""

    step: str
    records: tuple[MetricRecord, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        step = str(self.step).strip()
        records = tuple(self.records)

        if not step:
            raise ValueError("step must not be empty.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported StepMetrics schema version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}."
            )
        if not all(isinstance(record, MetricRecord) for record in records):
            raise TypeError("records must contain only MetricRecord instances.")

        seen: set[tuple[str, str, str | None]] = set()
        duplicates = set()
        for record in records:
            if record.key in seen:
                duplicates.add(record.key)
            seen.add(record.key)
        if duplicates:
            duplicate_list = ", ".join(map(str, sorted(duplicates)))
            raise ValueError(f"Duplicate step metric records: {duplicate_list}.")

        object.__setattr__(self, "step", step)
        object.__setattr__(self, "records", records)

    def to_internal_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "Step": self.step,
                "Species": record.species,
                "Metric ID": record.definition.metric_id,
                "Value": record.value,
                "Unit": record.definition.unit,
                "Basis": record.basis,
            }
            for record in self.records
        ]
        return pd.DataFrame(rows, columns=INTERNAL_COLUMNS)

    def to_legacy_dataframe(self) -> pd.DataFrame:
        """Return the historical four-column DataFrame consumed by pyDAC."""
        rows = [
            {
                "Step": self.step,
                "Species": record.species,
                "KPI": record.definition.legacy_name,
                "Value": record.value,
            }
            for record in self.records
        ]
        return pd.DataFrame(rows, columns=LEGACY_COLUMNS)

    @classmethod
    def from_legacy_dataframe(
        cls,
        dataframe: pd.DataFrame,
        catalog: MetricCatalog = STEP_METRIC_CATALOG,
    ) -> StepMetrics:
        """Build the typed contract from one historical per-step KPI table."""
        missing = [column for column in LEGACY_COLUMNS if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Legacy KPI DataFrame is missing columns: {missing}.")
        if dataframe.empty:
            raise ValueError("Cannot infer a step from an empty legacy KPI DataFrame.")

        steps = tuple(dict.fromkeys(str(value).strip() for value in dataframe["Step"]))
        if len(steps) != 1 or not steps[0]:
            raise ValueError(
                "A StepMetrics object requires exactly one non-empty step; "
                f"found {steps}."
            )

        records = []
        legacy_names_by_species = {
            normalize_metric_species(species): set(group["KPI"])
            for species, group in dataframe.groupby("Species", sort=False)
        }
        for row in dataframe.loc[:, LEGACY_COLUMNS].itertuples(index=False):
            definition = catalog.from_legacy_name(row.KPI)
            species = normalize_metric_species(row.Species)
            basis = definition.basis
            has_explicit_state_record = any(
                name.startswith("state-based ")
                for name in legacy_names_by_species[species]
            )
            if (
                species != "system"
                and not has_explicit_state_record
                and definition.metric_id in {
                "mass.change",
                "mass.final",
                }
            ):
                basis = "state"
            records.append(
                MetricRecord(
                    definition=definition,
                    species=species,
                    value=row.Value,
                    basis=basis,
                )
            )
        return cls(step=steps[0], records=tuple(records))
