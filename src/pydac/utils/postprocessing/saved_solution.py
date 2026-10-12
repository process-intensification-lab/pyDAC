"""Read dimensional saved fields through their persisted :class:`StateLayout`.

New solutions carry the complete generated layout.  A deliberately narrow
fallback recognizes only the historical current-DAC storage bundle; arbitrary
layout-less files are rejected because their field positions are ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from pydac.core.state_layout import StateLayout, current_dac_state_layout


SAVED_SOLUTION_SCHEMA_KEY = "saved_solution_schema_version"
SAVED_SOLUTION_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class SavedSolutionView:
    """Validated name-keyed view of one dimensional saved trajectory."""

    layout: StateLayout
    fields: Mapping[str, np.ndarray]
    time_s: np.ndarray
    compatibility_mode: str

    def field(self, name: str) -> np.ndarray:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise KeyError(
                f"Saved solution has no field {name!r}; available fields: "
                f"{tuple(self.fields)}."
            ) from exc


def saved_solution_view(solution: Mapping[str, object]) -> SavedSolutionView:
    """Validate storage groups and return histories keyed by layout field name."""
    if not isinstance(solution, Mapping):
        raise TypeError("Saved solution must be a mapping.")
    if "solve_provenance" in solution:
        from pydac.utils.provenance import source_record
        source_record(solution["solve_provenance"])

    schema_version = _saved_solution_schema_version(solution)
    arrays = {
        "y_2d": _saved_array(solution, "y_2d", ndim=4),
        "y_1d": _saved_array(solution, "y_1d", ndim=3),
        "y_scalars": _saved_array(solution, "y_scalars", ndim=2),
    }
    time_s = np.asarray(solution.get("t"), dtype=float)
    if time_s.ndim != 1 or time_s.size < 1 or not np.all(np.isfinite(time_s)):
        raise ValueError("Saved solution `t` must be a non-empty finite 1D array.")

    nt = time_s.size
    if any(array.shape[-1] != nt for array in arrays.values()):
        shapes = {name: array.shape for name, array in arrays.items()}
        raise ValueError(
            f"Saved field histories must share Nt={nt} from `t`; found {shapes}."
        )

    layout_document = solution.get("extra", {})
    layout_document = (
        layout_document.get("state_layout")
        if isinstance(layout_document, Mapping)
        else None
    )
    if layout_document is None:
        if schema_version is not None:
            raise ValueError(
                "Versioned saved solutions must contain a persisted StateLayout."
            )
        layout = _legacy_current_dac_layout(arrays)
        compatibility_mode = "legacy_current_dac"
    else:
        layout = StateLayout.from_dict(layout_document)
        compatibility_mode = (
            "versioned_canonical"
            if schema_version is not None
            else "persisted_layout"
        )

    fields: dict[str, np.ndarray] = {}
    for storage, array in arrays.items():
        storage_fields = layout.fields_in(storage)
        if array.shape[0] != len(storage_fields):
            raise ValueError(
                f"Saved `{storage}` contains {array.shape[0]} fields, but the "
                f"persisted StateLayout declares {len(storage_fields)}."
            )
        for index, field in enumerate(storage_fields):
            history = array[index]
            expected_shape = (*field.shape, nt)
            if history.shape != expected_shape:
                raise ValueError(
                    f"Saved field {field.name!r} has shape {history.shape}; "
                    f"StateLayout requires {expected_shape}."
                )
            fields[field.name] = history

    return SavedSolutionView(
        layout=layout,
        fields=MappingProxyType(fields),
        time_s=time_s,
        compatibility_mode=compatibility_mode,
    )


def _saved_solution_schema_version(
    solution: Mapping[str, object],
) -> int | None:
    """Return and validate the optional canonical saved-solution version."""
    raw_version = solution.get(SAVED_SOLUTION_SCHEMA_KEY)
    if raw_version is None:
        return None
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise TypeError(f"`{SAVED_SOLUTION_SCHEMA_KEY}` must be an integer.")
    if raw_version != SAVED_SOLUTION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported saved-solution schema version {raw_version}; "
            f"expected {SAVED_SOLUTION_SCHEMA_VERSION}."
        )
    if "y" in solution:
        raise ValueError(
            "Canonical saved-solution schema 2 must not contain the duplicate `y` history."
        )
    return raw_version


def _saved_array(solution: Mapping[str, object], name: str, *, ndim: int) -> np.ndarray:
    if name not in solution:
        raise ValueError(f"Saved solution is missing `{name}`.")
    array = np.asarray(solution[name], dtype=float)
    if array.ndim != ndim:
        raise ValueError(
            f"Saved `{name}` must have {ndim} dimensions; found shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Saved `{name}` contains non-finite values.")
    return array


def _legacy_current_dac_layout(arrays: Mapping[str, np.ndarray]) -> StateLayout:
    y_2d = arrays["y_2d"]
    y_1d = arrays["y_1d"]
    y_scalars = arrays["y_scalars"]
    if y_2d.shape[0] != 7 or y_1d.shape[0] != 1 or y_scalars.shape[0] != 5:
        raise ValueError(
            "Layout-less `sol.pkl` is supported only for the exact historical "
            "current-DAC storage bundle: 7 y_2d fields, 1 y_1d field, and 5 "
            "scalar fields. Regenerate any other result with layout metadata."
        )
    nz, nr = y_2d.shape[1:3]
    if y_1d.shape[1] != nz:
        raise ValueError(
            "Layout-less current-DAC solution has inconsistent axial dimensions."
        )
    return current_dac_state_layout(nz, nr)


__all__ = [
    "SAVED_SOLUTION_SCHEMA_KEY",
    "SAVED_SOLUTION_SCHEMA_VERSION",
    "SavedSolutionView",
    "saved_solution_view",
]
