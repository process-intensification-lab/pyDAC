"""Boolean runtime switches read from the environment.

Runtime switches keep optimizations behind A/B controls so a regression
can be bisected without editing code.
Each switch is read once at import time and bound to a module-level name, so a
test can override it by patching that name instead of the environment.
"""

from __future__ import annotations

import os


FALSE_TOKENS = frozenset({"0", "off", "false", "no"})


def env_flag(variable, *, default=True):
    """Return one boolean switch read from `variable`.

    Anything in `FALSE_TOKENS` disables the switch; every other value, and an
    unset variable, leaves it at `default`.
    """
    raw = os.environ.get(variable)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in FALSE_TOKENS
