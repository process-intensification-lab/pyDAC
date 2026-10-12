"""Small validation primitives shared by the public configuration schema.

Every ``__post_init__`` in this package states its rule as "these fields obey
that validator"; the mechanism of writing back to a frozen record lives here,
in :func:`coerce`, so the records stay readable as contracts.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : stage 5 — invoked by the typed records as they are constructed.
# Owns     : the field-level rules and their error messages.
# Not here : which field obeys which rule — that is stated by each record.
# Entry    : coerce()  ·  require_bools()  ·  the individual validators
# Guide    : docs/manual/01_inputs.md §9
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import Any, Callable, Iterable


def finite_float(value: Real, name: str) -> float:
    """Return a finite real as ``float``; reject booleans and non-reals."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"`{name}` must be finite.")
    return result


def positive_float(value: Real, name: str) -> float:
    result = finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return result


def nonnegative_float(value: Real, name: str) -> float:
    result = finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"`{name}` must be non-negative.")
    return result


def fraction(value: Real, name: str, *, strict: bool = False) -> float:
    result = finite_float(value, name)
    if strict:
        valid = 0.0 < result < 1.0
    else:
        valid = 0.0 <= result <= 1.0
    if not valid:
        interval = "strictly between 0 and 1" if strict else "between 0 and 1"
        raise ValueError(f"`{name}` must be {interval}.")
    return result


def optional_instance(value: Any, name: str, *, expected: type) -> Any:
    """Accept a typed record or ``None``, rejecting any other type by name.

    Records that travel as optional runtime inputs (a reduced grid, explicit
    solver tolerances) are checked at the boundary that receives them, so a
    wrong type is reported there instead of surfacing as a missing attribute
    while the model is being built.
    """
    if value is not None and not isinstance(value, expected):
        raise TypeError(f"`{name}` must be a {expected.__name__} instance or None.")
    return value


def nonempty_string(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"`{name}` must be a string.")
    result = value.strip()
    if not result:
        raise ValueError(f"`{name}` must not be empty.")
    return result


def coerce(
    record: Any,
    names: Iterable[str],
    validator: Callable[..., Any],
    **options: Any,
) -> None:
    """Validate and normalize several fields of a frozen dataclass in place.

    Frozen records must assign through ``object.__setattr__``; routing every
    field through this helper keeps ``__post_init__`` bodies stating the rule
    (which fields obey which validator) instead of the mechanism.
    """
    for name in names:
        object.__setattr__(record, name, validator(getattr(record, name), name, **options))


def require_bools(record: Any, names: Iterable[str]) -> None:
    """Reject non-``bool`` values for fields that must be true flags."""
    for name in names:
        if not isinstance(getattr(record, name), bool):
            raise TypeError(f"`{name}` must be bool.")


__all__ = [
    "coerce",
    "finite_float",
    "fraction",
    "nonempty_string",
    "nonnegative_float",
    "optional_instance",
    "positive_float",
    "require_bools",
]
