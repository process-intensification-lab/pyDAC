"""Versioned compact terminal-state artifacts for trajectory-free MOO runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import pickle
from typing import Mapping

import numpy as np

from pydac.core.state_layout import StateLayout

from pydac.utils.solver.persistence import MOO_COMPACT, normalize_persistence_policy
from pydac.utils.postprocessing.saved_solution import saved_solution_view
from pydac.utils.provenance import source_record


TERMINAL_STATE_FILENAME = "terminal.pkl"
TERMINAL_STATE_SCHEMA_KEY = "terminal_state_schema_version"
TERMINAL_STATE_SCHEMA_VERSION = 1
_REQUIRED_KEYS = {
    TERMINAL_STATE_SCHEMA_KEY,
    "artifact_kind",
    "persistence_policy",
    "step",
    "time_s",
    "state_layout",
    "fields",
    "grid",
    "termination",
    "solver",
}


def terminal_state_from_solution(
    solution: Mapping[str, object],
    *,
    step: str,
    solver_outcome: Mapping[str, object],
) -> dict[str, object]:
    """Copy only one dimensional PDE endpoint out of a prepared trajectory."""
    saved = saved_solution_view(solution)
    extra = solution.get("extra", {})
    if not isinstance(extra, Mapping):
        raise TypeError("Prepared solution `extra` metadata must be a mapping.")

    artifact = {
        TERMINAL_STATE_SCHEMA_KEY: TERMINAL_STATE_SCHEMA_VERSION,
        "artifact_kind": "terminal_state",
        "persistence_policy": MOO_COMPACT,
        "step": str(step),
        "time_s": float(saved.time_s[-1]),
        "state_layout": saved.layout.to_dict(),
        "fields": {
            field.name: np.array(saved.field(field.name)[..., -1], copy=True)
            for field in saved.layout.pde_fields
        },
        "grid": {
            "Nz": int(extra.get("Nz", saved.layout.nz)),
            "Nr": int(extra.get("Nr", saved.layout.nr)),
            "Nt": int(extra.get("Nt", saved.time_s.size)),
            "z_f": np.array(extra.get("z_f", []), dtype=float, copy=True),
            "z_c": np.array(extra.get("z_c", []), dtype=float, copy=True),
            "r_f": np.array(extra.get("r_f", []), dtype=float, copy=True),
            "r_c": np.array(extra.get("r_c", []), dtype=float, copy=True),
        },
        "termination": deepcopy(solution.get("termination")),
        "solver": deepcopy(dict(solver_outcome)),
    }
    if "solve_provenance" in solution:
        artifact["solve_provenance"] = source_record(solution["solve_provenance"])["record"]
    return validate_terminal_state(artifact)


def validate_terminal_state(document: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize one compact terminal-state document."""
    if not isinstance(document, Mapping):
        raise TypeError("Terminal-state artifact must be a mapping.")
    if set(document) - {"solve_provenance"} != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - set(document))
        unexpected = sorted(set(document) - _REQUIRED_KEYS)
        raise ValueError(
            "Terminal-state artifact keys differ from the schema "
            f"(missing={missing}, unexpected={unexpected})."
        )
    if "solve_provenance" in document:
        source_record(document["solve_provenance"])
    if document[TERMINAL_STATE_SCHEMA_KEY] != TERMINAL_STATE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported terminal-state schema version "
            f"{document[TERMINAL_STATE_SCHEMA_KEY]!r}."
        )
    if document["artifact_kind"] != "terminal_state":
        raise ValueError("Compact artifact kind must be 'terminal_state'.")
    if normalize_persistence_policy(document["persistence_policy"]) != MOO_COMPACT:
        raise ValueError("Terminal-state artifacts are reserved for `moo_compact`.")
    step = document["step"]
    if not isinstance(step, str) or not step.strip():
        raise ValueError("Terminal-state step must be non-empty text.")
    time_s = float(document["time_s"])
    if not np.isfinite(time_s) or time_s < 0.0:
        raise ValueError("Terminal-state time must be finite and non-negative.")

    layout = StateLayout.from_dict(document["state_layout"])
    raw_fields = document["fields"]
    if not isinstance(raw_fields, Mapping):
        raise TypeError("Terminal-state fields must be a mapping.")
    expected_names = tuple(field.name for field in layout.pde_fields)
    if tuple(raw_fields) != expected_names:
        raise ValueError(
            "Terminal-state fields must follow the persisted StateLayout order."
        )
    fields: dict[str, np.ndarray] = {}
    for field in layout.pde_fields:
        values = np.asarray(raw_fields[field.name], dtype=float)
        if values.shape != field.shape:
            raise ValueError(
                f"Terminal field {field.name!r} has shape {values.shape}; "
                f"expected {field.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Terminal field {field.name!r} contains non-finite values.")
        fields[field.name] = values

    raw_grid = document["grid"]
    if not isinstance(raw_grid, Mapping):
        raise TypeError("Terminal-state grid metadata must be a mapping.")
    if set(raw_grid) != {"Nz", "Nr", "Nt", "z_f", "z_c", "r_f", "r_c"}:
        raise ValueError("Terminal-state grid metadata keys differ from the schema.")
    nz = int(raw_grid["Nz"])
    nr = int(raw_grid["Nr"])
    nt = int(raw_grid["Nt"])
    if (nz, nr) != (layout.nz, layout.nr) or nt < 1:
        raise ValueError("Terminal-state grid metadata differs from StateLayout or has Nt < 1.")
    coordinates = {}
    expected_sizes = {"z_f": nz + 1, "z_c": nz, "r_f": nr + 1, "r_c": nr}
    for name, expected_size in expected_sizes.items():
        values = np.asarray(raw_grid[name], dtype=float)
        if values.shape != (expected_size,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Terminal-state coordinate {name!r} must be finite with "
                f"shape {(expected_size,)}."
            )
        coordinates[name] = values

    for name in ("termination", "solver"):
        if document[name] is not None and not isinstance(document[name], Mapping):
            raise TypeError(f"Terminal-state {name} must be a mapping or None.")

    return {
        TERMINAL_STATE_SCHEMA_KEY: TERMINAL_STATE_SCHEMA_VERSION,
        "artifact_kind": "terminal_state",
        "persistence_policy": MOO_COMPACT,
        "step": step.strip(),
        "time_s": time_s,
        "state_layout": layout.to_dict(),
        "fields": fields,
        "grid": {"Nz": nz, "Nr": nr, "Nt": nt, **coordinates},
        "termination": deepcopy(document["termination"]),
        "solver": deepcopy(dict(document["solver"] or {})),
        **({"solve_provenance": deepcopy(document["solve_provenance"])}
           if "solve_provenance" in document else {}),
    }


def save_terminal_state(document: Mapping[str, object], path: str | Path) -> Path:
    """Validate and pickle one compact terminal artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(validate_terminal_state(document), handle)
    return destination


def load_terminal_state(path: str | Path) -> dict[str, object]:
    """Load and validate one compact terminal artifact."""
    with Path(path).open("rb") as handle:
        return validate_terminal_state(pickle.load(handle))


def terminal_profiles(document: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Return copied primitive PDE fields for CSS comparison."""
    terminal = validate_terminal_state(document)
    return {
        name: np.array(values, copy=True)
        for name, values in terminal["fields"].items()
    }


def terminal_final_fields(document: Mapping[str, object]) -> dict[str, object]:
    """Reconstruct the next-step initial-state payload from a compact endpoint."""
    terminal = validate_terminal_state(document)
    layout = StateLayout.from_dict(terminal["state_layout"])
    fields = terminal["fields"]
    grid = terminal["grid"]
    return {
        "ci": np.stack(
            [fields[f"c_{species}"] for species in layout.independent_gas_species],
            axis=-1,
        ),
        "qi": np.stack(
            [fields[f"q_{species}"] for species in layout.adsorbates],
            axis=-1,
        ),
        "c_tot": np.array(fields["c_tot"], copy=True),
        "T": np.array(fields["T_bed"], copy=True),
        "T_wall": np.array(fields["T_wall"], copy=True),
        "layout_signature": layout.signature,
        "Nz": grid["Nz"],
        "Nr": grid["Nr"],
        "Nt": grid["Nt"],
        "z_f": np.array(grid["z_f"], copy=True),
        "z_c": np.array(grid["z_c"], copy=True),
        "r_f": np.array(grid["r_f"], copy=True),
        "r_c": np.array(grid["r_c"], copy=True),
    }


__all__ = [
    "TERMINAL_STATE_FILENAME",
    "TERMINAL_STATE_SCHEMA_KEY",
    "TERMINAL_STATE_SCHEMA_VERSION",
    "load_terminal_state",
    "save_terminal_state",
    "terminal_final_fields",
    "terminal_profiles",
    "terminal_state_from_solution",
    "validate_terminal_state",
]
