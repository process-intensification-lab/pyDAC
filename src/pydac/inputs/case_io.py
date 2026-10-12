"""Exact, standalone frozen-case I/O for human-authored experiments.

The regular configuration loader intentionally supports partial overlays. A
case has a different contract: it is a complete resolved document, and its
meaning must not change when the current Python defaults change. This module
owns that distinction, canonical JSON serialization, and the case
fingerprint. It does not import the physical model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydac.config.errors import ConfigurationError
from pydac.config.loader import (
    _construct_config,
    _resolve_derivations,
    load_experiment,
    merge_documents,
)
from pydac.config.migrations import (
    CURRENT_SCHEMA_VERSION,
    LEGACY_SCHEMA_VERSION,
    PRE_INITIAL_LOADING_SCHEMA_VERSION,
    PRE_OUTLET_SCHEMA_VERSION,
    PRE_SORBENT_MAPPING_SCHEMA_VERSION,
    PRE_VALVE_RENAME_SCHEMA_VERSION,
    PRE_WALL_RENAME_SCHEMA_VERSION,
    SOLVER_CHOICE_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    legacy_v1_vocabulary,
    pre_initial_loading_v7_vocabulary,
    pre_outlet_v3_vocabulary,
    pre_sorbent_mapping_v6_vocabulary,
    pre_valve_rename_v4_vocabulary,
    pre_wall_rename_v5_vocabulary,
    refuse_initial_loadings_before_v8,
    solver_choice_v2_vocabulary,
    upgrade_experiment_document,
)
from pydac.config.profile import read_json_document
from pydac.config.schema import (
    ExperimentConfig,
    default_bed_mapping,
)
from pydac.config.serialization import experiment_to_document
from pydac.config.steps import STEP_NAMES
from pydac.inputs.defaults import default_profile
from pydac.inputs.defaults.steps import INITIAL_SOLID_LOADINGS_FIELD


CASE_SCHEMA_VERSION = CURRENT_SCHEMA_VERSION
CASE_METADATA_FILENAME = "case_metadata.json"
CANONICALIZATION_DESCRIPTION = (
    "json.dumps(document, indent=2, sort_keys=True, "
    "ensure_ascii=False, allow_nan=False) + newline"
)


def _path_text(path: tuple[str, ...]) -> str:
    return ".".join(path) or "<root>"


def _copy_json_value(value: Any, path: tuple[str, ...]) -> Any:
    """Copy a JSON value while rejecting non-standard values and keys."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigurationError(
                    f"Keys at `{_path_text(path)}` must be strings."
                )
            result[key] = _copy_json_value(item, path + (key,))
        return result
    if isinstance(value, (list, tuple)):
        return [_copy_json_value(item, path + (str(index),)) for index, item in enumerate(value)]
    if value is None or isinstance(value, (bool, int, str)):
        return deepcopy(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError(
                f"Non-standard JSON number at `{_path_text(path)}` is not allowed."
            )
        return value
    raise ConfigurationError(
        f"Value at `{_path_text(path)}` is not JSON serializable."
    )


def _source_document(source: Any) -> dict[str, Any]:
    if isinstance(source, ExperimentConfig):
        return _copy_json_value(experiment_to_document(source), ())
    if isinstance(source, (str, Path)):
        return _copy_json_value(read_json_document(source), ())
    if isinstance(source, Mapping):
        return _copy_json_value(source, ())
    raise ConfigurationError(
        "A frozen case must be a path, mapping, or ExperimentConfig; "
        f"received {type(source).__name__}."
    )


def _require_object(value: Any, path: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Case section `{_path_text(path)}` must be a JSON object.")
    return value


def _require_exact_keys(
    value: Any,
    path: tuple[str, ...],
    expected: set[str] | frozenset[str] | tuple[str, ...],
) -> Mapping[str, Any]:
    mapping = _require_object(value, path)
    expected_set = set(expected)
    unknown = sorted(set(mapping) - expected_set)
    if unknown:
        raise ConfigurationError(
            f"Unknown case field `{_path_text(path + (unknown[0],))}`."
        )
    missing = sorted(expected_set - set(mapping))
    if missing:
        raise ConfigurationError(
            f"Incomplete case: missing field `{_path_text(path + (missing[0],))}`."
        )
    return mapping


_OPAQUE_PATHS = frozenset(
    {
        ("simulation", "bed", "design"),
        ("simulation", "bed", "material"),
        ("simulation", "adsorption", "equilibrium_systems"),
        ("simulation", "adsorption", "kinetics"),
        ("simulation", "sorbent", "heats_of_adsorption_J_mol"),
        ("simulation", "sorbent", "heat_capacity"),
        ("simulation", "sorbent", "fixed_adsorbates"),
        ("execution", "step_options"),
        ("execution", "cycle_options"),
    }
    | {
        ("simulation", "cycle", "steps", step, "composition", INITIAL_SOLID_LOADINGS_FIELD)
        for step in STEP_NAMES
    }
)


def _case_vocabulary(source: Any, document: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, ExperimentConfig):
        return deepcopy(document)
    vocabulary = default_profile()
    vocabulary["simulation"]["bed"] = default_bed_mapping()
    return vocabulary


def _require_complete(value: Any, expected: Any, path: tuple[str, ...]) -> None:
    if path in _OPAQUE_PATHS:
        if value is not None:
            _require_object(value, path)
        return
    if not isinstance(expected, Mapping):
        return
    mapping = _require_exact_keys(value, path, set(expected))
    for key, expected_value in expected.items():
        _require_complete(mapping[key], expected_value, path + (key,))


def validate_case_document(source: Any) -> dict[str, Any]:
    """Validate and return a complete standalone case document.

    Structural completeness is checked without reading the current reference
    profile. Architecture-specific ``bed.design`` and ``bed.material`` keys
    are validated by the existing detached architecture resolver when the
    typed configuration is constructed.
    """
    original = _source_document(source)
    version = original.get("schema_version")
    refuse_initial_loadings_before_v8(original, "frozen case")
    current_vocabulary = _case_vocabulary(source, original)
    if version == LEGACY_SCHEMA_VERSION:
        _require_complete(
            original,
            legacy_v1_vocabulary(current_vocabulary),
            (),
        )
    elif version == SOLVER_CHOICE_SCHEMA_VERSION:
        _require_complete(
            original,
            solver_choice_v2_vocabulary(current_vocabulary),
            (),
        )
    elif version == PRE_OUTLET_SCHEMA_VERSION:
        _require_complete(
            original,
            pre_outlet_v3_vocabulary(current_vocabulary),
            (),
        )
    elif version == PRE_VALVE_RENAME_SCHEMA_VERSION:
        _require_complete(
            original,
            pre_valve_rename_v4_vocabulary(current_vocabulary),
            (),
        )
    elif version == PRE_WALL_RENAME_SCHEMA_VERSION:
        _require_complete(
            original,
            pre_wall_rename_v5_vocabulary(current_vocabulary),
            (),
        )
    elif version == PRE_SORBENT_MAPPING_SCHEMA_VERSION:
        _require_complete(
            original,
            pre_sorbent_mapping_v6_vocabulary(current_vocabulary),
            (),
        )
    elif version == PRE_INITIAL_LOADING_SCHEMA_VERSION:
        _require_complete(
            original,
            pre_initial_loading_v7_vocabulary(current_vocabulary),
            (),
        )
    elif version == CURRENT_SCHEMA_VERSION:
        _require_complete(original, current_vocabulary, ())
    else:
        raise ConfigurationError(
            f"Unsupported case `schema_version` {version!r}; this loader supports "
            f"{', '.join(map(str, SUPPORTED_SCHEMA_VERSIONS))}."
        )

    document = upgrade_experiment_document(original, label="frozen case")
    vocabulary = _case_vocabulary(source, document)
    _require_complete(document, vocabulary, ())

    try:
        resolved = _resolve_derivations(merge_documents(document, document))
        steps = resolved["simulation"]["cycle"]["steps"]
        resolved["simulation"]["cycle"]["steps"] = {
            name: steps[name] for name in STEP_NAMES
        }
        _construct_config(resolved, vocabulary=vocabulary)
    except ConfigurationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"Invalid frozen case: {exc}") from exc
    return document


def canonical_case_bytes(source: Any) -> bytes:
    """Return the deterministic UTF-8 representation used for case hashes."""
    return _canonical_bytes(validate_case_document(source))


def _canonical_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Could not serialize case as standard JSON: {exc}") from exc
    return rendered.encode("utf-8")


def case_fingerprint(source: Any) -> str:
    """Return the stable SHA-256 fingerprint of a complete case."""
    return hashlib.sha256(canonical_case_bytes(source)).hexdigest()


def load_case(source: Any) -> ExperimentConfig:
    """Load a complete case without consulting current Python defaults."""
    document = validate_case_document(source)
    # Canonical JSON sorts keys, while the runtime record keeps the deliberate
    # cycle order. No owned-default stage is called: the case is its own base.
    steps = document["simulation"]["cycle"]["steps"]
    document["simulation"]["cycle"]["steps"] = {name: steps[name] for name in STEP_NAMES}
    vocabulary = _case_vocabulary(source, document)
    return _construct_config(
        _resolve_derivations(merge_documents(document, document)),
        vocabulary=vocabulary,
    )


def export_case(
    source: Any = None,
    path: str | Path = "case.json",
    *,
    settings: Mapping[str, Any] | list[str] | tuple[str, ...] | None = None,
    overwrite: bool = False,
) -> Path:
    """Resolve development input and write one complete frozen case.

    ``source`` may be a partial preset, a complete mapping, or an already
    resolved ``ExperimentConfig``. Existing targets require ``overwrite=True``.
    """
    config = source if isinstance(source, ExperimentConfig) else load_experiment(
        source,
        settings=settings,
    )
    document = validate_case_document(config)
    target = refuse_existing_case(path, overwrite=overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_case_bytes(document))
    return target


def refuse_existing_case(path: str | Path, *, overwrite: bool) -> Path:
    """Refuse an existing export target unless ``overwrite``; return the target."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise ConfigurationError(
            f"Refusing to overwrite existing case `{target}`; use --overwrite-case "
            "(overwrite=True in Python) to replace it."
        )
    return target


def case_metadata(source: Any) -> dict[str, Any]:
    """Return metadata suitable for a run sidecar, excluding self-reference."""
    fingerprint = case_fingerprint(source)
    document = validate_case_document(source)
    return {
        "case_fingerprint": fingerprint,
        "schema_version": document["schema_version"],
        "canonicalization": CANONICALIZATION_DESCRIPTION,
    }


def stored_case_metadata(document: Any) -> dict[str, Any]:
    """Return :func:`case_metadata` of a stored case document as it is stored.

    Saved runs record the complete, already validated document, so their
    fingerprint is checked without today's case vocabulary: a later version
    that adds a case field still reads them.
    """
    if not isinstance(document, Mapping):
        raise ConfigurationError("A stored case document must be a JSON object.")
    return {
        "case_fingerprint": hashlib.sha256(_canonical_bytes(document)).hexdigest(),
        "schema_version": document.get("schema_version"),
        "canonicalization": CANONICALIZATION_DESCRIPTION,
    }


def write_case_metadata(source: Any, resolved_config_path: str | Path) -> Path:
    """Write the case fingerprint beside a persisted resolved configuration."""
    output = Path(resolved_config_path)
    metadata_path = output.with_name(CASE_METADATA_FILENAME)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(case_metadata(source), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata_path


__all__ = [
    "CASE_METADATA_FILENAME",
    "CASE_SCHEMA_VERSION",
    "CANONICALIZATION_DESCRIPTION",
    "canonical_case_bytes",
    "case_fingerprint",
    "case_metadata",
    "export_case",
    "refuse_existing_case",
    "load_case",
    "stored_case_metadata",
    "validate_case_document",
    "write_case_metadata",
]
