"""Validated DataFrame contract for metrics collected across one cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Mapping, Sequence

import pandas as pd

from pydac.utils.postprocessing.normalization import (
    STEP_KEY_TO_CYCLE_STEP,
    normalize_metric_species,
    normalize_step_key,
)

from pydac.post_processing.metrics.step_metrics import StepMetrics


CYCLE_METRICS_SCHEMA_VERSION = 1
CYCLE_METRIC_COLUMNS = (
    "Cycle",
    "Step Index",
    "Step Key",
    "Step",
    "Species",
    "Metric ID",
    "Value",
    "Unit",
    "Basis",
)

STEP_KEY_TO_METRICS_LABEL = {
    "startup": "startup",
    "ads": "adsorption",
    "blow": "blowdown",
    "heat": "preheating",
    "w_purge": "wall purge",
    "s_purge": "steam purge",
    "w_des": "wall heated desorption",
    "s_des": "steam assisted desorption",
    "press": "pressurization",
    "cool": "cooldown",
}

_UNSET = object()
_MISSING = object()
_METRIC_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z0-9]+)*$")


def _normalize_cycle_label(cycle_label):
    label = str(cycle_label).strip()
    if not label:
        raise ValueError("cycle label must not be empty.")
    return label


def _normalize_metric_id(metric_id):
    token = str(metric_id).strip().lower()
    if not _METRIC_ID_PATTERN.fullmatch(token):
        raise ValueError(f"Invalid Metric ID '{metric_id}'.")
    return token


@dataclass(frozen=True, slots=True, init=False)
class CycleMetrics:
    """Immutable, validated collection of typed metrics for one cycle."""

    _dataframe: pd.DataFrame = field(repr=False)
    schema_version: int = CYCLE_METRICS_SCHEMA_VERSION

    def __init__(self, dataframe, schema_version=CYCLE_METRICS_SCHEMA_VERSION):
        if schema_version != CYCLE_METRICS_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported CycleMetrics schema version {schema_version}; "
                f"expected {CYCLE_METRICS_SCHEMA_VERSION}."
            )
        validated = self._validate_dataframe(dataframe)
        object.__setattr__(self, "_dataframe", validated)
        object.__setattr__(self, "schema_version", schema_version)

    @staticmethod
    def _validate_dataframe(dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        if tuple(dataframe.columns) != CYCLE_METRIC_COLUMNS:
            raise ValueError(
                "CycleMetrics DataFrame columns must be exactly "
                f"{CYCLE_METRIC_COLUMNS}; found {tuple(dataframe.columns)}."
            )
        if dataframe.empty:
            raise ValueError("CycleMetrics DataFrame must not be empty.")

        table = dataframe.copy(deep=True).reset_index(drop=True)
        cycle_labels = tuple(
            dict.fromkeys(_normalize_cycle_label(value) for value in table["Cycle"])
        )
        if len(cycle_labels) != 1:
            raise ValueError(
                "CycleMetrics requires exactly one cycle label; "
                f"found {cycle_labels}."
            )
        table["Cycle"] = cycle_labels[0]

        numeric_values = pd.to_numeric(table["Value"], errors="coerce")
        invalid_numeric = numeric_values.isna() & table["Value"].notna()
        if invalid_numeric.any():
            bad_values = table.loc[invalid_numeric, "Value"].tolist()
            raise TypeError(f"Cycle metric values must be numeric; found {bad_values}.")
        table["Value"] = numeric_values.astype(float)

        step_indices = pd.to_numeric(table["Step Index"], errors="coerce")
        if step_indices.isna().any() or (step_indices < 0).any():
            raise ValueError("Step Index values must be non-negative integers.")
        if not (step_indices == step_indices.astype(int)).all():
            raise ValueError("Step Index values must be non-negative integers.")
        table["Step Index"] = step_indices.astype(int)

        table["Step Key"] = table["Step Key"].map(normalize_step_key)
        table["Step"] = table["Step"].astype(str).str.strip().str.lower()
        table["Species"] = table["Species"].map(
            lambda species: normalize_metric_species(
                species,
                empty_message="Species values must not be empty.",
            )
        )
        table["Metric ID"] = table["Metric ID"].map(_normalize_metric_id)
        table["Unit"] = table["Unit"].astype(str).str.strip()
        table["Basis"] = table["Basis"].map(
            lambda value: None if pd.isna(value) else str(value).strip().lower()
        )

        if table["Unit"].eq("").any():
            raise ValueError("Unit values must not be empty.")
        if table["Basis"].eq("").any():
            raise ValueError("Basis values must be None or non-empty strings.")

        step_rows = table.loc[:, ["Step Index", "Step Key", "Step"]].drop_duplicates()
        if step_rows["Step Key"].duplicated().any():
            raise ValueError("Each Step Key must map to exactly one Step Index and Step.")
        if step_rows["Step Index"].duplicated().any():
            raise ValueError("Each Step Index must map to exactly one Step Key.")

        expected_indices = list(range(len(step_rows)))
        actual_indices = sorted(step_rows["Step Index"].tolist())
        if actual_indices != expected_indices:
            raise ValueError(
                "Step Index values must be contiguous and start at zero; "
                f"found {actual_indices}."
            )

        for row in step_rows.itertuples(index=False):
            expected_step = STEP_KEY_TO_CYCLE_STEP[row[1]]
            if row[2] != expected_step:
                raise ValueError(
                    f"Step key '{row[1]}' must map to cycle step '{expected_step}', "
                    f"not '{row[2]}'."
                )

        seen = set()
        duplicates = set()
        for key in table.loc[
            :, ["Step Key", "Species", "Metric ID", "Basis"]
        ].itertuples(index=False, name=None):
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        if duplicates:
            raise ValueError(
                f"Duplicate cycle metric records: {sorted(duplicates)}."
            )

        return table.loc[:, CYCLE_METRIC_COLUMNS]

    @classmethod
    def from_step_metrics(
        cls,
        cycle_label: str,
        step_metrics_by_key: Mapping[str, StepMetrics],
        step_order: Sequence[str] | None = None,
    ) -> CycleMetrics:
        """Aggregate ordered per-step contracts into one canonical cycle table."""
        if not isinstance(step_metrics_by_key, Mapping):
            raise TypeError("step_metrics_by_key must be a mapping.")
        if not step_metrics_by_key:
            raise ValueError("step_metrics_by_key must not be empty.")

        normalized_metrics = {}
        for raw_key, metrics in step_metrics_by_key.items():
            step_key = normalize_step_key(raw_key)
            if step_key in normalized_metrics:
                raise ValueError(f"Duplicate normalized step key '{step_key}'.")
            if not isinstance(metrics, StepMetrics):
                raise TypeError(
                    f"Metrics for step '{raw_key}' must be a StepMetrics instance."
                )
            expected_label = STEP_KEY_TO_METRICS_LABEL[step_key]
            if metrics.step != expected_label:
                raise ValueError(
                    f"StepMetrics for key '{step_key}' must use step label "
                    f"'{expected_label}', not '{metrics.step}'."
                )
            normalized_metrics[step_key] = metrics

        if step_order is None:
            ordered_keys = tuple(normalized_metrics)
        else:
            ordered_keys = tuple(normalize_step_key(key) for key in step_order)
            if len(set(ordered_keys)) != len(ordered_keys):
                raise ValueError("step_order must not contain duplicate step keys.")
            missing = set(normalized_metrics) - set(ordered_keys)
            unknown = set(ordered_keys) - set(normalized_metrics)
            if missing or unknown:
                raise ValueError(
                    "step_order must contain exactly the supplied step metrics; "
                    f"missing={sorted(missing)}, unknown={sorted(unknown)}."
                )

        cycle = _normalize_cycle_label(cycle_label)
        rows = []
        for step_index, step_key in enumerate(ordered_keys):
            cycle_step = STEP_KEY_TO_CYCLE_STEP[step_key]
            metrics = normalized_metrics[step_key]
            for record in metrics.records:
                rows.append(
                    {
                        "Cycle": cycle,
                        "Step Index": step_index,
                        "Step Key": step_key,
                        "Step": cycle_step,
                        "Species": record.species,
                        "Metric ID": record.definition.metric_id,
                        "Value": record.value,
                        "Unit": record.definition.unit,
                        "Basis": record.basis,
                    }
                )

        return cls(pd.DataFrame(rows, columns=CYCLE_METRIC_COLUMNS))

    @property
    def cycle_label(self):
        return str(self._dataframe["Cycle"].iloc[0])

    @property
    def step_order(self):
        ordered = (
            self._dataframe.loc[:, ["Step Index", "Step Key"]]
            .drop_duplicates()
            .sort_values("Step Index")
        )
        return tuple(ordered["Step Key"])

    def to_dataframe(self):
        """Return a deep copy of the canonical cycle metrics table."""
        return self._dataframe.copy(deep=True)

    def select(
        self,
        *,
        step_key=None,
        step=None,
        species=None,
        metric_id=None,
        basis=_UNSET,
    ):
        """Select records using stable identities rather than legacy labels."""
        table = self._dataframe
        mask = pd.Series(True, index=table.index)

        if step_key is not None:
            mask &= table["Step Key"].eq(normalize_step_key(step_key))
        if step is not None:
            mask &= table["Step"].eq(str(step).strip().lower())
        if species is not None:
            species_token = normalize_metric_species(
                species,
                allow_empty=True,
            )
            mask &= table["Species"].eq(species_token)
        if metric_id is not None:
            mask &= table["Metric ID"].eq(_normalize_metric_id(metric_id))
        if basis is not _UNSET:
            normalized_basis = None if basis is None else str(basis).strip().lower()
            if normalized_basis is None:
                mask &= table["Basis"].isna()
            else:
                mask &= table["Basis"].eq(normalized_basis)

        return table.loc[mask].copy(deep=True).reset_index(drop=True)

    def value(self, *, default=_MISSING, **selection):
        """Return one selected numeric value, or ``default`` when absent."""
        rows = self.select(**selection)
        if rows.empty:
            if default is _MISSING:
                raise KeyError(f"No cycle metric matches {selection}.")
            return default
        if len(rows) != 1:
            raise ValueError(
                f"Expected one cycle metric for {selection}; found {len(rows)}."
            )
        return float(rows["Value"].iloc[0])
