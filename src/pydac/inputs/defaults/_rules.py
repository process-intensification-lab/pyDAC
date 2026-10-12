"""Declarative constraints that strengthen types inferred from defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


Normalizer = Callable[[Any, str], Any]


@dataclass(frozen=True, slots=True)
class Rule:
    """Optional constraints for one public path.

    ``kind`` is needed only when a ``None`` or empty-container default cannot
    reveal its non-null type. Ordinary new leaves need no rule at all.
    """

    kind: str | None = None
    nullable: bool = False
    minimum: float | int | None = None
    maximum: float | int | None = None
    exclusive_minimum: float | int | None = None
    exclusive_maximum: float | int | None = None
    choices: tuple[Any, ...] = ()
    min_length: int | None = None
    min_items: int | None = None
    item_kind: str | None = None
    read_only: bool = False
    opaque: bool = False
    description: str = ""
    schema: Mapping[str, Any] | None = None
    normalize: Normalizer | None = field(default=None, repr=False, compare=False)


def species_keys(value: Any, name: str) -> Any:
    """Key a species mapping by upper-case names, as the mixture lists are.

    Species names do not depend on case, so a lower- and an upper-case spelling
    give one case and one fingerprint; two keys that name one species are refused. A value
    that is not a mapping is left to the type check.
    """
    if not isinstance(value, Mapping):
        return value
    keyed: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"`{name}` keys must be non-empty species names.")
        species = key.strip().upper()
        if species in keyed:
            raise ValueError(
                f"`{name}` names {species} twice; species names do not depend on case."
            )
        keyed[species] = item
    return keyed


def positive_number(description: str = "", **options: Any) -> Rule:
    return Rule(
        kind="number",
        exclusive_minimum=0,
        description=description,
        **options,
    )


def nonnegative_number(description: str = "", **options: Any) -> Rule:
    return Rule(kind="number", minimum=0, description=description, **options)


def fraction(description: str = "", *, strict: bool = False, **options: Any) -> Rule:
    bounds = (
        {"exclusive_minimum": 0, "exclusive_maximum": 1}
        if strict
        else {"minimum": 0, "maximum": 1}
    )
    return Rule(kind="number", description=description, **bounds, **options)


def positive_integer(description: str = "", **options: Any) -> Rule:
    return Rule(kind="integer", minimum=1, description=description, **options)


def nonempty_string(description: str = "", **options: Any) -> Rule:
    return Rule(kind="string", min_length=1, description=description, **options)


__all__ = [
    "Rule",
    "fraction",
    "nonempty_string",
    "nonnegative_number",
    "positive_integer",
    "positive_number",
    "species_keys",
]
