"""Default-driven immutable mappings used by public runtime sections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import Any

from pydac.inputs.defaults._rules import Rule

from pydac.config.contract import path_text, validate_leaf


def _path_value(document: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = document
    for component in path:
        value = value[component]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_thaw(item) for item in value)
    return value


def _restore_section(section_type, values, defaults, path):
    return section_type.from_mapping(values, defaults=defaults, path=path)


class FrozenMapping(Mapping):
    """A recursively immutable, pickle-safe mapping."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[Any, Any]):
        items = tuple((key, _freeze(value)) for key, value in values.items())
        object.__setattr__(self, "_items", items)

    def __getitem__(self, key: Any) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise FrozenInstanceError("immutable mapping")

    def __repr__(self) -> str:
        return repr(dict(self))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Mapping) and dict(self) == dict(other)

    def __reduce__(self):
        return type(self), (_thaw(self),)


class FrozenSection(Mapping):
    """Immutable public section with mapping and attribute access."""

    PATH: tuple[str, ...] = ()
    __slots__ = ("_values", "_defaults", "_path")

    def __init__(self, **values: Any):
        from pydac.inputs.defaults.profile import default_profile

        defaults = _path_value(default_profile(), self.PATH)
        self._initialize(values, defaults, self.PATH)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        defaults: Mapping[str, Any],
        path: tuple[str, ...] | None = None,
    ):
        instance = cls.__new__(cls)
        instance._initialize(values, defaults, cls.PATH if path is None else path)
        return instance

    def _initialize(
        self,
        values: Mapping[str, Any],
        defaults: Mapping[str, Any],
        path: tuple[str, ...],
    ) -> None:
        from pydac.inputs.defaults.profile import default_rules

        if not isinstance(values, Mapping) or not isinstance(defaults, Mapping):
            raise TypeError(f"`{path_text(path)}` must be a mapping.")
        unknown = sorted(set(values) - set(defaults))
        missing = sorted(set(defaults) - set(values))
        if unknown:
            raise TypeError(f"Unknown configuration field `{path_text(path + (unknown[0],))}`.")
        if missing:
            raise TypeError(f"Missing configuration field `{path_text(path + (missing[0],))}`.")
        rules = default_rules()
        resolved = []
        for name, default in defaults.items():
            leaf_path = path + (name,)
            rule = rules.get(leaf_path, Rule())
            value = validate_leaf(
                values[name],
                default,
                rule,
                leaf_path,
                rules=rules,
            )
            resolved.append((name, _freeze(value)))
        object.__setattr__(self, "_values", tuple(resolved))
        object.__setattr__(self, "_defaults", FrozenMapping(defaults))
        object.__setattr__(self, "_path", path)

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._values:
            if candidate == key:
                return value
        raise KeyError(key)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __iter__(self):
        return (key for key, _value in self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise FrozenInstanceError("immutable configuration section")

    def __repr__(self) -> str:
        body = ", ".join(f"{key}={value!r}" for key, value in self._values)
        return f"{type(self).__name__}({body})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Mapping) and dict(self) == dict(other)

    def __reduce__(self):
        return _restore_section, (
            type(self),
            _thaw(self),
            _thaw(self._defaults),
            self._path,
        )

    def with_overrides(self, **values: Any):
        merged = _thaw(self)
        merged.update(values)
        return type(self).from_mapping(
            merged,
            defaults=_thaw(self._defaults),
            path=self._path,
        )


__all__ = ["FrozenMapping", "FrozenSection"]
