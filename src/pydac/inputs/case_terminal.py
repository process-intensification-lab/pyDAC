"""Terminal bridges for development presets and exact frozen cases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydac.config.errors import ConfigurationError
from pydac.config.loader import load_experiment
from pydac.config.schema import ExperimentConfig

from pydac.inputs.case_io import case_fingerprint, export_case, load_case


def resolve_terminal_experiment(args: Any, *, settings: Mapping[str, Any] | None = None):
    """Resolve terminal arguments in development or explicit-case mode."""
    # Frozen case: load the exact standalone document and reject all overrides.
    if getattr(args, "case", None) is not None:
        if args.config is not None or args.settings:
            raise ConfigurationError("Frozen-case mode does not accept --config or --set.")
        if getattr(args, "export_case", None) is not None:
            raise ConfigurationError("--case cannot be combined with --export-case.")
        if getattr(args, "overwrite_case", False):
            raise ConfigurationError("--overwrite-case requires --export-case.")
        return load_case(args.case)

    # Development mode: defaults < preset < --set in one pass, as `pydac.load_case`
    # does, so the terminal and Python build the same case. A runner's own
    # settings sit between the preset and --set as a layer of their own.
    if getattr(args, "overwrite_case", False) and getattr(args, "export_case", None) is None:
        raise ConfigurationError("--overwrite-case requires --export-case.")
    if not settings:
        return load_experiment(args.config, settings=args.settings or None)
    config = load_experiment(args.config, settings=settings)
    return load_experiment(config, settings=args.settings) if args.settings else config


def resolve_runner_experiment(
    args: Any,
    *,
    case_path: str | None,
    development_settings: Mapping[str, Any] | None,
):
    """Resolve a runner in development or exact-case mode."""
    # A runner can choose a case in the file or from CLI, but never both.
    explicit = getattr(args, "case", None)
    if case_path is not None and explicit is not None:
        raise ConfigurationError("A runner CASE_PATH cannot be combined with --case.")
    selected = case_path if case_path is not None else explicit

    # No case selected: apply this runner's DEVELOPMENT_SETTINGS.
    if selected is None:
        return (
            resolve_terminal_experiment(args, settings=development_settings),
            "case from built-in defaults, preset and settings",
        )

    # Case selected: run the frozen document exactly, with no other input.
    if args.config is not None or args.settings:
        raise ConfigurationError("Frozen-case mode does not accept --config or --set.")
    if development_settings:
        raise ConfigurationError(
            "Frozen-case mode does not accept the runner's DEVELOPMENT_SETTINGS; "
            "empty them, or run without a case."
        )
    if getattr(args, "export_case", None) is not None:
        raise ConfigurationError("--case cannot be combined with --export-case.")
    if getattr(args, "overwrite_case", False):
        raise ConfigurationError("--overwrite-case requires --export-case.")
    return load_case(selected), f"frozen case {selected}"


def export_case_if_requested(args: Any, config: ExperimentConfig):
    """Export a requested case and return its path, or ``None``."""
    # No export flag: caller should continue with dry-run or simulation.
    target = getattr(args, "export_case", None)
    if target is None:
        return None

    # Export the already-resolved development document as a complete frozen case.
    output = export_case(config, target, overwrite=getattr(args, "overwrite_case", False))
    print(f"Exported frozen case: {output}")
    print(f"Case fingerprint: {case_fingerprint(config)}")
    return output


__all__ = [
    "export_case_if_requested",
    "resolve_runner_experiment",
    "resolve_terminal_experiment",
]
