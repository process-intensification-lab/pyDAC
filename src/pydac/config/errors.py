"""Public exceptions raised by the experiment configuration API.

Catching :class:`ConfigurationError` catches every public configuration
failure, including the settings ones. Both derive from ``ValueError``, so
existing callers that guard on ``ValueError`` keep working.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : raised from every stage; caught by `pydac.cli` (exit 2).
# Guide    : docs/manual/01_inputs.md §9
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations


class ConfigurationError(ValueError):
    """Raised when a document cannot be resolved into typed configuration."""


class SettingsError(ConfigurationError):
    """Raised when a setting path or a CLI ``path=value`` expression is invalid.

    It derives from :class:`ConfigurationError` so a caller can catch one
    exception type for every public configuration failure.
    """


class RunRefused(ConfigurationError):
    """Raised when a valid request would continue or overwrite a run it must not.

    The request itself is well formed; the saved run it targets refuses it
    (for example, it was written by another pyDAC version).
    """


__all__ = ["ConfigurationError", "RunRefused", "SettingsError"]
