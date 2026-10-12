"""Generate editor JSON Schema from editable defaults and colocated rules."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydac.inputs.defaults import default_profile, default_rules
from pydac.inputs.defaults._rules import Rule
from pydac.config.migrations import CURRENT_SCHEMA_VERSION
from pydac.utils.version import __version__


SCHEMA_REPOSITORY_URL = "https://github.com/process-intensification-lab/pyDAC"


PACKAGED_SCHEMA_PATH = Path(__file__).with_name("json_schema") / "experiment.schema.json"


def _kind(default: Any, rule: Rule) -> str:
    if rule.kind is not None:
        return rule.kind
    if isinstance(default, bool):
        return "boolean"
    if isinstance(default, int):
        return "integer"
    if isinstance(default, float):
        return "number"
    if isinstance(default, str):
        return "string"
    if isinstance(default, Mapping):
        return "object"
    if isinstance(default, Sequence) and not isinstance(default, (str, bytes)):
        return "array"
    raise TypeError("A null default needs an explicit JSON Schema kind rule.")


def _scalar_schema(default: Any, rule: Rule) -> dict[str, Any]:
    kind = _kind(default, rule)
    schema: dict[str, Any] = {"type": kind}
    if rule.choices:
        schema.pop("type")
        schema["enum"] = list(rule.choices)
    keyword_values = {
        "minimum": rule.minimum,
        "maximum": rule.maximum,
        "exclusiveMinimum": rule.exclusive_minimum,
        "exclusiveMaximum": rule.exclusive_maximum,
        "minLength": rule.min_length,
        "minItems": rule.min_items,
    }
    schema.update({name: value for name, value in keyword_values.items() if value is not None})
    if rule.item_kind is not None:
        schema["items"] = {"type": rule.item_kind}
    return schema


def _node_schema(default: Any, path: tuple[str, ...], rules) -> dict[str, Any]:
    rule = rules.get(path, Rule())
    if rule.schema is not None:
        schema = deepcopy(dict(rule.schema))
    elif isinstance(default, Mapping) and not rule.opaque:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                name: _node_schema(value, path + (name,), rules)
                for name, value in default.items()
            },
        }
    elif rule.opaque:
        schema = {"type": "object", "additionalProperties": True}
    else:
        schema = _scalar_schema(default, rule)
    if rule.nullable and rule.schema is None:
        schema = {"anyOf": [schema, {"type": "null"}]}
    if rule.read_only:
        schema["readOnly"] = True
    if rule.description:
        schema["description"] = rule.description
    return schema


def _simulation_schema(profile: Mapping[str, Any], rules) -> tuple[dict, dict]:
    """Factor the repeated step contract while deriving it from all steps."""
    steps = profile["cycle"]["steps"]
    step_schemas = [
        _node_schema(step, ("simulation", "cycle", "steps", name), rules)
        for name, step in steps.items()
    ]
    if any(schema != step_schemas[0] for schema in step_schemas[1:]):
        raise ValueError("Every default cycle step must declare the same group vocabulary.")
    cycle_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "steps": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    name: {"$ref": "#/$defs/step"} for name in steps
                },
            }
        },
    }
    properties = {
        name: _node_schema(value, ("simulation", name), rules)
        for name, value in profile.items()
        if name != "cycle"
    }
    properties["cycle"] = {"$ref": "#/$defs/cycle"}
    simulation_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    return simulation_schema, {"cycle": cycle_schema, "step": step_schemas[0]}


def generate_json_schema() -> dict[str, Any]:
    """Return a deterministic Draft 2020-12 schema for public input."""
    profile = default_profile()
    rules = default_rules()
    simulation, shared = _simulation_schema(profile["simulation"], rules)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": (
            f"{SCHEMA_REPOSITORY_URL.replace('https://github.com/', 'https://raw.githubusercontent.com/')}"
            f"/v{__version__}/src/pydac/config/json_schema/experiment.schema.json"
        ),
        "title": "pyDAC experiment configuration",
        "description": (
            "Public pyDAC input. Fields are optional in development presets; "
            "frozen-case completeness is enforced by inputs.case_io."
        ),
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": CURRENT_SCHEMA_VERSION},
            "simulation": {"$ref": "#/$defs/simulation"},
            "execution": {"$ref": "#/$defs/execution"},
        },
        "$defs": {
            **shared,
            "simulation": simulation,
            "execution": _node_schema(profile["execution"], ("execution",), rules),
        },
    }


def render_json_schema() -> str:
    return json.dumps(generate_json_schema(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_packaged_schema(path: str | Path = PACKAGED_SCHEMA_PATH) -> Path:
    output = Path(path)
    output.write_text(render_json_schema(), encoding="utf-8")
    return output


if __name__ == "__main__":
    write_packaged_schema()


__all__ = [
    "PACKAGED_SCHEMA_PATH",
    "generate_json_schema",
    "render_json_schema",
    "write_packaged_schema",
]
