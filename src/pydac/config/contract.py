"""Default-inferred type validation strengthened by colocated path rules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real
from typing import Any

from pydac.inputs.defaults._rules import Rule


def path_text(path: tuple[str, ...]) -> str:
    return ".".join(path) or "<root>"


def _infer_kind(default: Any, rule: Rule) -> str:
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
    raise TypeError(
        "A null default needs an explicit non-null type rule; "
        f"received {default!r}."
    )


def _validate_kind(value: Any, kind: str, name: str) -> Any:
    if kind == "boolean":
        if not isinstance(value, bool):
            raise TypeError(f"`{name}` must be bool.")
        return value
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"`{name}` must be an integer.")
        return int(value)
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"`{name}` must be a real number.")
        result = float(value)
        if not isfinite(result):
            raise ValueError(f"`{name}` must be finite.")
        return result
    if kind == "string":
        if not isinstance(value, str):
            raise TypeError(f"`{name}` must be a string.")
        return value.strip()
    if kind == "object":
        if not isinstance(value, Mapping):
            raise TypeError(f"`{name}` must be a mapping.")
        return dict(value)
    if kind == "array":
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"`{name}` must be a sequence.")
        return tuple(value)
    raise TypeError(f"Unsupported default rule kind {kind!r} for `{name}`.")


def _validate_mapping(result, default, path, rules):
    unknown = sorted(set(result) - set(default))
    missing = sorted(set(default) - set(result))
    if unknown or missing:
        key = unknown[0] if unknown else missing[0]
        issue = "Unknown" if unknown else "Missing"
        raise TypeError(f"{issue} configuration field `{path_text(path + (key,))}`.")
    return {
        key: validate_leaf(
            result[key],
            child_default,
            rules.get(path + (key,), Rule()),
            path + (key,),
            rules=rules,
        )
        for key, child_default in default.items()
    }


def _validate_sequence(result, default, path, rules):
    name = path_text(path)
    if len(result) != len(default):
        raise ValueError(f"`{name}` must contain exactly {len(default)} item(s).")
    return tuple(
        validate_leaf(
            item,
            child_default,
            rules.get(path + (str(index),), Rule()),
            path + (str(index),),
            rules=rules,
        )
        for index, (item, child_default) in enumerate(zip(result, default))
    )


def validate_leaf(
    value: Any,
    default: Any,
    rule: Rule,
    path: tuple[str, ...],
    *,
    rules: Mapping[tuple[str, ...], Rule] | None = None,
) -> Any:
    """Validate one value using its default type and optional stronger rule."""
    name = path_text(path)
    if value is None:
        if rule.nullable:
            return None
        raise TypeError(f"`{name}` must not be null.")
    if rule.normalize is not None:
        value = rule.normalize(value, name)
    result = _validate_kind(value, _infer_kind(default, rule), name)
    declared_rules = {} if rules is None else rules
    if isinstance(default, Mapping) and not rule.opaque:
        result = _validate_mapping(result, default, path, declared_rules)
    if (
        isinstance(default, Sequence)
        and not isinstance(default, (str, bytes))
        and rule.normalize is None
    ):
        result = _validate_sequence(result, default, path, declared_rules)
    if rule.min_length is not None and len(result) < rule.min_length:
        raise ValueError(f"`{name}` must not be empty.")
    if rule.min_items is not None and len(result) < rule.min_items:
        raise ValueError(f"`{name}` must contain at least {rule.min_items} item(s).")
    if rule.choices and result not in rule.choices:
        raise ValueError(f"`{name}` must be one of {list(rule.choices)!r}.")
    for bound, relation, valid in (
        (rule.minimum, "greater than or equal to", lambda item, edge: item >= edge),
        (rule.maximum, "less than or equal to", lambda item, edge: item <= edge),
        (rule.exclusive_minimum, "greater than", lambda item, edge: item > edge),
        (rule.exclusive_maximum, "less than", lambda item, edge: item < edge),
    ):
        if bound is not None and not valid(result, bound):
            raise ValueError(f"`{name}` must be {relation} {bound}.")
    return result


__all__ = ["path_text", "validate_leaf"]
