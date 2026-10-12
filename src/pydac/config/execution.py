"""The default-driven public execution section for one experiment."""

from __future__ import annotations

from pydac.config.sections import FrozenSection


class ExecutionConfig(FrozenSection):
    """Immutable solve, CSS, output, plotting, and diagnostic settings.

    Field names and basic types come from ``execution_defaults()``. Stronger
    rules and canonicalizers remain beside those editable values in
    ``inputs.defaults.execution``.
    """

    PATH = ("execution",)


__all__ = ["ExecutionConfig"]
