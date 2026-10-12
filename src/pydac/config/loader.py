"""Resolve public experiment documents into immutable typed configuration.

The loader orchestrates one pipeline and owns no formula of its own:

    reference profile -> case document -> settings -> directives -> typed config

Each stage is total and deterministic, so resolving an already-resolved
document returns the same document. Values come from ``config.profile``,
formulas from ``config.derivations``, the ``null`` policy from
``config.nulls``, and contracts from ``config.schema`` and ``config.steps``;
the loader never imports or mutates the physical model.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : all five stages — this module is the pipeline.
# Owns     : the ORDER of the stages and the document-level error messages.
# Not here : values (profile.py), meanings (schema.py, steps.py, execution.py),
#            `null` policy (nulls.py), path grammar (settings.py), formulas
#            (derivations.py).
# Entry    : load_experiment() — read it first; everything above it is a stage.
# Guide    : docs/manual/01_inputs.md §2.1
# Order matters: owned defaults are filled BEFORE settings are applied, so
#            `--set bed.design.eps_b=0.4` finds a concrete mapping to walk
#            into; derivations run LAST, after every value is final.
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydac.config.derivations import (
    GEOMETRY_INPUT_PAIRS,
    DERIVED_RTOL,
    derive_geometry_pair,
    derive_steam_volumetric_flow,
)
from pydac.config.errors import ConfigurationError, SettingsError
from pydac.config.migrations import upgrade_experiment_document
from pydac.config.nulls import INHERIT, canonical_values, null_policy, resolve_null
from pydac.config.profile import read_json_document, reference_profile
from pydac.config.schema import SIMULATION_SECTIONS, ExperimentConfig, experiment_config_from_mapping
from pydac.config.serialization import experiment_to_document
from pydac.config.settings import OPAQUE_MAPPING_PATHS, resolve_settings, setting_path_parts
from pydac.config.steps import STEP_NAMES


_STEPS_PREFIX = ("simulation", "cycle", "steps")

# Document sources
def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    # Accept only JSON-object-like inputs and copy them before editing.
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{location} must be a JSON object.")
    return {str(key): deepcopy(item) for key, item in value.items()}


def _read_source(source: Any, *, label: str) -> dict[str, Any]:
    # Normalize every supported source into the same mutable document dict.
    if isinstance(source, ExperimentConfig):
        return experiment_to_document(source)
    if isinstance(source, Mapping):
        return _require_mapping(source, label)
    if isinstance(source, (str, Path)):
        return read_json_document(source)
    raise ConfigurationError(
        f"{label} must be a path, mapping, or ExperimentConfig; "
        f"received {type(source).__name__}."
    )


def _path_text(path: tuple[str, ...]) -> str:
    # Turn an internal tuple path into the dotted text shown to users.
    return ".".join(path) or "<root>"


def _document_path(canonical_path: tuple[str, ...]) -> tuple[str, ...]:
    """Expand a canonical setting path to its position in the document."""
    # Settings may omit structural wrappers; put them back for document access.
    if canonical_path[:1] == ("steps",):
        return _STEPS_PREFIX[:2] + canonical_path
    if canonical_path and canonical_path[0] in SIMULATION_SECTIONS:
        return ("simulation",) + canonical_path
    return canonical_path


# The user-owned mappings are replaced wholesale rather than merged key by
# key. The policy itself is stated once, in canonical setting paths, by
# ``config.settings``; here it is only expanded to document positions.
_REPLACED_MAPPING_PATHS = frozenset(
    _document_path(setting_path_parts(path)) for path in OPAQUE_MAPPING_PATHS
)


# Inheritance
def merge_documents(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge an input document over a complete profile.

    Mapping nodes merge by key; scalar values replace the profile. Unknown
    keys fail immediately, and an explicit ``null`` is resolved through the
    field's documented policy.
    """
    base_document = _require_mapping(base, "base document")
    overlay_document = _require_mapping(overlay, "overlay document")

    def merge_node(current: Any, incoming: Any, path: tuple[str, ...]) -> Any:
        # Mapping overlays usually recurse; opaque mappings replace wholesale.
        if isinstance(incoming, Mapping):
            if path in _REPLACED_MAPPING_PATHS or not isinstance(current, Mapping):
                return deepcopy(incoming)
            result = {str(key): deepcopy(value) for key, value in current.items()}
            for key, value in incoming.items():
                if not isinstance(key, str):
                    raise ConfigurationError(f"Keys at `{_path_text(path)}` must be strings.")
                if key not in current:
                    raise ConfigurationError(
                        f"Unknown configuration field `{_path_text(path + (key,))}`."
                    )
                result[key] = merge_node(current[key], value, path + (key,))
            return result
        # Explicit null is meaningful: resolve it through the path policy.
        if incoming is None:
            return resolve_null(path, current)
        return deepcopy(incoming)

    return merge_node(base_document, overlay_document, ())


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


def _get_path(document: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    # Walk an existing document path; missing prefixes mean user-facing typo.
    current: Any = document
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise SettingsError(f"Unknown setting path `{_path_text(path)}`.")
        current = current[component]
    return current


_BED_DESIGN_PATH = ("simulation", "bed", "design")
_CHANNELING_PATH = _BED_DESIGN_PATH + ("channeling",)


def _is_new_bed_design_field(path: tuple[str, ...]) -> bool:
    """A design field may be absent from the packed-bed profile yet be valid.

    The same holds for a field of the nested ``channeling`` option block.
    Whether it really is valid for the selected architecture is decided by
    ``bed_architectures.resolve_bed_config`` when the typed config is built.
    """
    if len(path) == 4 and path[:3] == _BED_DESIGN_PATH:
        return True
    return len(path) == 5 and path[:4] == _CHANNELING_PATH


def _resolve_bed_profile_alias(document: dict[str, Any]) -> dict[str, Any]:
    """Expand a development-time bed alias before typed construction."""
    simulation = document.get("simulation")
    if not isinstance(simulation, Mapping):
        return document

    bed = simulation.get("bed")
    if not isinstance(bed, str):
        return document

    from pydac.config.bed_profiles import resolve_bed_profile

    document["simulation"]["bed"] = resolve_bed_profile(bed)
    return document


def _set_path(document: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    # Walk to the parent object; settings may not create arbitrary branches.
    current: Any = document
    for depth, component in enumerate(path[:-1], start=1):
        # The optional channeling block is created by its first field.
        if (
            _is_new_bed_design_field(path)
            and path[:depth] == _CHANNELING_PATH
            and isinstance(current, dict)
            and current.get(component) is None
        ):
            current[component] = {}
        if not isinstance(current, dict) or component not in current:
            raise SettingsError(f"Unknown setting path `{_path_text(path)}`.")
        current = current[component]
    if not isinstance(current, dict):
        raise SettingsError(f"Unknown setting path `{_path_text(path)}`.")
    # Ordinary fields must exist; architecture-owned bed.design fields may be new.
    if path[-1] not in current and not _is_new_bed_design_field(path):
        raise SettingsError(f"Unknown setting path `{_path_text(path)}`.")
    current[path[-1]] = deepcopy(value)


def apply_settings(
    document: Mapping[str, Any],
    settings: Mapping[str, Any] | list[str] | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Apply canonical dotted settings without mutating ``document``."""
    result = _require_mapping(document, "configuration document")
    for raw_path, value in resolve_settings(settings).items():
        # Normalize dotted/nested/user spelling into the canonical setting path.
        canonical = setting_path_parts(raw_path)
        if canonical[:1] == ("steps",) and canonical[1:2] and canonical[1] not in STEP_NAMES:
            raise SettingsError(f"Unknown cycle step `{canonical[1]}` in `{raw_path}`.")
        # Expand canonical shortcuts, resolve explicit nulls, then assign.
        path = _document_path(canonical)
        if value is None:
            value = resolve_null(path, _current_value(result, path))
        _set_path(result, path, value)
    return _resolve_bed_profile_alias(result)


def _current_value(document: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """Return the value a ``null`` setting would inherit, if the path exists."""
    # A new bed-design leaf has no current packed-bed profile value to inherit.
    if _is_new_bed_design_field(path):
        return None
    return _get_path(document, path)


# --------------------------------------------------------------------------
# Directives: owned defaults, then derivations
# --------------------------------------------------------------------------


def _resolve_owned_defaults(document: dict[str, Any]) -> dict[str, Any]:
    """Fill the `null` directives whose value another module owns.

    This runs before settings are applied, so `--set bed.design.eps_b=0.4`
    finds a concrete mapping to walk into.
    """
    result = deepcopy(document)
    for path, value in canonical_values().items():
        # Example: `bed: null` becomes the architecture-owned packed-bed default.
        if _get_path(result, path) is None:
            _set_path(result, path, value)
    return result


_GEOMETRY_PATH = ("simulation", "geometry")
_GEOMETRY_FIELDS = ("specified_by", "length_m", "radius_m", "aspect_ratio", "volume_m3")


def _geometry_layers(case: Mapping[str, Any], settings) -> dict[str, int]:
    """Where each geometry field was last given: 0 defaults, 1 preset/case, 2 settings."""
    layers = {name: 0 for name in _GEOMETRY_FIELDS}
    simulation = case.get("simulation") if isinstance(case, Mapping) else None
    case_geometry = simulation.get("geometry") if isinstance(simulation, Mapping) else None
    if isinstance(case_geometry, Mapping):
        layers.update({name: 1 for name in case_geometry if name in layers})
    for raw_path in resolve_settings(settings):
        path = _document_path(setting_path_parts(raw_path))
        if path == _GEOMETRY_PATH or path[:1] == ("simulation",) and len(path) == 1:
            layers.update({name: 2 for name in layers})
        elif path[:2] == _GEOMETRY_PATH and len(path) == 3 and path[2] in layers:
            layers[path[2]] = 2
    return layers


def _computed_geometry_to_keep(geometry: Mapping[str, Any], layers: Mapping[str, int]):
    """The computed fields stated at least as recently as the inputs they follow.

    An older one is stale (a frozen case whose aspect ratio a setting changed, or a
    built-in default under the other `specified_by`) and is recomputed; a newer or
    equal one is the user's own statement and must agree with the inputs.
    """
    inputs = GEOMETRY_INPUT_PAIRS.get(geometry.get("specified_by"), ())
    newest = max([layers["specified_by"], *(layers[name] for name in inputs)])
    return frozenset(
        name for name in _GEOMETRY_FIELDS[1:]
        if name not in inputs and layers[name] >= newest
    )


# A steam-fed step's volumetric flow is computed from these fields of the same step.
_STEAM_FLOW_INPUTS = (
    ("flow", "steam_mass_flow_kg_s"),
    ("pressure", "feed_pressure_Pa"),
    ("thermal", "feed_temperature_K"),
)


def _field_layer(path: tuple[str, ...], case: Any, setting_paths) -> int:
    """Where a document field was last given: 0 defaults, 1 preset/case, 2 settings."""
    if any(path[: len(setting)] == setting for setting in setting_paths):
        return 2
    node = case
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return 0
        node = node[key]
    return 1


def _steam_flows_to_keep(
    document: Mapping[str, Any], case: Any, settings
) -> frozenset[str]:
    """The steps whose written steam volumetric flow is a statement to check.

    As for the geometry, a written value older than the inputs it follows is stale
    (a frozen case whose steam mass flow a setting changed) and is recomputed; a
    newer or equal one is the user's own statement and must agree with them.
    """
    setting_paths = [
        _document_path(setting_path_parts(raw)) for raw in resolve_settings(settings)
    ]
    keep = set()
    for name in document["simulation"]["cycle"]["steps"]:
        step = _STEPS_PREFIX + (name,)
        newest = max(
            _field_layer(step + field, case, setting_paths) for field in _STEAM_FLOW_INPUTS
        )
        written = _field_layer(step + ("flow", "volumetric_flow_m3_s"), case, setting_paths)
        if written >= newest:
            keep.add(name)
    return frozenset(keep)


def _resolve_derivations(
    document: dict[str, Any],
    *,
    keep_geometry: frozenset[str] | None = None,
    keep_steam_flows: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Recompute every field that is a function of other public fields.

    This stage is a pure function of the merged document, which is what makes
    ``load_experiment`` idempotent: re-resolving a resolved document, or
    re-loading one with a new aspect ratio, always recomputes the same
    derived values instead of carrying stale ones forward.
    """
    result = deepcopy(document)
    simulation = result["simulation"]

    # The column size is given by one pair, named by `specified_by`; the other
    # pair is computed. `keep_geometry` (from the layers) says which written
    # computed values are statements to check rather than stale values to replace.
    derive_geometry_pair(simulation["geometry"], keep_geometry)

    # Wherever a step is fed by steam, its volumetric flow follows the steam
    # mass flow rather than being stated twice. `keep_steam_flows` (from the
    # layers) says which written values are statements to check, as above.
    for name, step in simulation["cycle"]["steps"].items():
        flow = step["flow"]
        mass_flow = flow["steam_mass_flow_kg_s"]
        if mass_flow is None or mass_flow <= 0.0:
            continue
        try:
            value = derive_steam_volumetric_flow(
                mass_flow,
                feed_pressure_Pa=step["pressure"]["feed_pressure_Pa"],
                feed_temperature_K=step["thermal"]["feed_temperature_K"],
            )
        except ConfigurationError as exc:
            raise ConfigurationError(f"Step `{name}`: {exc}") from exc
        written = flow.get("volumetric_flow_m3_s")
        if written is not None and (keep_steam_flows is None or name in keep_steam_flows):
            if not math.isclose(float(written), value, rel_tol=DERIVED_RTOL):
                raise ConfigurationError(
                    f"`steps.{name}.flow.volumetric_flow_m3_s` is computed from the step's "
                    f"`steam_mass_flow_kg_s` at its feed pressure and temperature: {written} "
                    f"differs from {value:.12g}. Leave it null, or change "
                    f"`steps.{name}.flow.steam_mass_flow_kg_s` instead."
                )
        flow["volumetric_flow_m3_s"] = value

    return result


# --------------------------------------------------------------------------
# Typed construction and public API
# --------------------------------------------------------------------------


def _construct_config(
    document: Mapping[str, Any],
    *,
    vocabulary: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    # The reference profile defines the public vocabulary accepted at runtime.
    if vocabulary is None:
        vocabulary = reference_profile()
    try:
        config = experiment_config_from_mapping(document, vocabulary)
        from pydac.config.model_validation import validate_model_configuration

        validate_model_configuration(config)
        return config
    except ConfigurationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"Invalid resolved experiment configuration: {exc}") from exc


def _resolve_document(
    source: Any = None,
    *,
    settings: Mapping[str, Any] | list[str] | tuple[str, ...] | None = None,
    default_profile: Any = None,
) -> dict[str, Any]:
    # Start from the selected complete profile: default, JSON, or supplied mapping.
    if default_profile is None:
        profile = reference_profile()
    else:
        profile = _read_source(default_profile, label="default profile")
    profile = upgrade_experiment_document(profile, label="default profile")

    # Merge optional preset/case-like source, fill canonical nulls, apply settings.
    case = (
        {}
        if source is None
        else upgrade_experiment_document(
            _read_source(source, label="experiment"), label="experiment"
        )
    )
    merged = _resolve_bed_profile_alias(merge_documents(profile, case))
    merged = _resolve_owned_defaults(merged)
    merged = apply_settings(merged, settings)
    if merged.get("schema_version") != profile.get("schema_version"):
        raise ConfigurationError(
            "Unsupported `schema_version`; the loader supports only the profile version "
            f"{profile.get('schema_version')}."
        )
    keep = _computed_geometry_to_keep(
        merged["simulation"]["geometry"], _geometry_layers(case, settings)
    )
    return _resolve_derivations(
        merged,
        keep_geometry=keep,
        keep_steam_flows=_steam_flows_to_keep(merged, case, settings),
    )


# --------------------------------------------------------------------------
# Settings as a preset
# --------------------------------------------------------------------------


_OPAQUE_SETTING_PATHS = tuple(setting_path_parts(path) for path in OPAQUE_MAPPING_PATHS)


def _block_owner(canonical: tuple[str, ...], path: tuple[str, ...]) -> tuple[str, ...] | None:
    """The document path of the block a setting changes one field of, or ``None``.

    A preset replaces an opaque block as one, and gives a block whose profile
    value is a `null` directive (the bed) as one, so such a block is written whole.
    """
    # The bed holds the opaque `bed.design`, so the outer block is looked up first.
    for owner in canonical_values():
        if len(path) > len(owner) and path[: len(owner)] == owner:
            return owner
    for owner in _OPAQUE_SETTING_PATHS:
        if len(canonical) > len(owner) and canonical[: len(owner)] == owner:
            return _document_path(owner)
    return None


def _has_path(document: Mapping[str, Any], path: tuple[str, ...]) -> bool:
    node: Any = document
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return False
        node = node[key]
    return True


def _put_path(document: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    # A preset may lack any branch, so create it; a bed alias is expanded first
    # so that a bed setting can walk into it.
    _resolve_bed_profile_alias(document)
    node = document
    for key in path[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[path[-1]] = deepcopy(value)


def _drop_path(document: dict[str, Any], path: tuple[str, ...]) -> None:
    if _has_path(document, path):
        del _get_path(document, path[:-1])[path[-1]]


def settings_to_preset(
    settings: Mapping[str, Any] | list[str] | tuple[str, ...] | None,
    *,
    base: Any = None,
) -> dict[str, Any]:
    """Write ``base`` (a preset) and ``settings`` on top of it as one preset document.

    Loading the result alone gives the case ``load_experiment(base, settings=settings)``
    gives. A preset only states fields, so three setting meanings are written out:
    a field of an opaque block or of the bed writes the whole resolved block (a
    preset gives those blocks as one); an inheriting ``null`` keeps the base value; a computed
    geometry field or steam flow of ``base`` that the settings make stale is left
    out, so that it is recomputed as it would be.
    """
    case = (
        {}
        if base is None
        else upgrade_experiment_document(_read_source(base, label="base preset"), label="base preset")
    )
    preset = _resolve_bed_profile_alias(deepcopy(case))
    resolved = _resolve_document(base, settings=settings)
    for raw_path, value in resolve_settings(settings).items():
        canonical = setting_path_parts(raw_path)
        path = _document_path(canonical)
        owner = _block_owner(canonical, path)
        if owner is not None:
            _put_path(preset, owner, _get_path(resolved, owner))
            continue
        if value is None and null_policy(path) == INHERIT and _has_path(preset, path):
            continue
        _put_path(preset, path, value)

    geometry = resolved["simulation"]["geometry"]
    layers = _geometry_layers(case, settings)
    stale = (
        set(_GEOMETRY_FIELDS[1:])
        - set(GEOMETRY_INPUT_PAIRS.get(geometry.get("specified_by"), ()))
        - _computed_geometry_to_keep(geometry, layers)
    )
    for name in stale:
        if layers[name] == 1:
            _drop_path(preset, _GEOMETRY_PATH + (name,))
    # Only a steam-fed step recomputes its volumetric flow (_resolve_derivations).
    kept_flows = _steam_flows_to_keep(resolved, case, settings)
    for name, step in resolved["simulation"]["cycle"]["steps"].items():
        mass_flow = step["flow"]["steam_mass_flow_kg_s"]
        if name not in kept_flows and mass_flow is not None and mass_flow > 0.0:
            _drop_path(preset, _STEPS_PREFIX + (name, "flow", "volumetric_flow_m3_s"))

    preset.pop("schema_version", None)
    return {"schema_version": resolved["schema_version"], **preset}


def load_experiment(
    source: Any = None,
    *,
    settings: Mapping[str, Any] | list[str] | tuple[str, ...] | None = None,
    default_profile: Any = None,
) -> ExperimentConfig:
    """Load a path/mapping/config with profile inheritance and settings."""
    # Build the vocabulary once, then use it for both resolution and validation.
    profile = (
        reference_profile()
        if default_profile is None
        else _read_source(default_profile, label="default profile")
    )
    profile = upgrade_experiment_document(profile, label="default profile")
    return _construct_config(
        _resolve_document(source, settings=settings, default_profile=profile),
        vocabulary=profile,
    )


def default_experiment_config() -> ExperimentConfig:
    """Return the resolved reference experiment carried by the default profile."""
    # Convenience entry point for callers that want the canonical default case.
    return load_experiment()


def write_resolved_config(config: ExperimentConfig, path: str | Path) -> Path:
    """Write a resolved configuration with strict standard JSON semantics."""
    # Persist the resolved document beside fingerprint metadata for replay.
    output = Path(path)
    output.write_text(
        json.dumps(experiment_to_document(config), indent=2, sort_keys=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    from pydac.inputs.case_io import write_case_metadata
    write_case_metadata(config, output)
    return output


__all__ = [
    "apply_settings",
    "default_experiment_config",
    "load_experiment",
    "merge_documents",
    "settings_to_preset",
    "write_resolved_config",
]
