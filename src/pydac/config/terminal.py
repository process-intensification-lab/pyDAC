"""Shared terminal vocabulary and runner-mode resolution."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

try:
    import argcomplete
except ModuleNotFoundError:  # Optional for environments without shell completion.
    argcomplete = None

from pydac.config.input_help import print_settings_catalog
from pydac.inputs.case_terminal import (
    export_case_if_requested,
    resolve_runner_experiment,
    resolve_terminal_experiment,
)

from pydac.config.schema import ExperimentConfig
from pydac.config.serialization import experiment_to_document


class _ShowSettingsAction(argparse.Action):
    """Print the live catalog and stop before resolving or running a case."""

    def __call__(self, parser, namespace, values, option_string=None):
        print_settings_catalog()
        parser.exit()


def _normalize_set_arguments(args: Sequence[str] | None) -> list[str]:
    """Join shell-separated ``--set path = value`` tokens before parsing."""
    tokens = list(sys.argv[1:] if args is None else args)
    normalized: list[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if token != "--set" and not token.startswith("--set="):
            normalized.append(token)
            index += 1
            continue

        normalized.append("--set")
        if token.startswith("--set="):
            setting = token[len("--set=") :]
        else:
            index += 1
            if index >= len(tokens):
                normalized.append("")
                break
            setting = tokens[index]
            index += 1

        if "=" not in setting and index < len(tokens):
            if tokens[index] == "=":
                setting += "="
                index += 1
                if index < len(tokens) and not tokens[index].startswith("--"):
                    setting += tokens[index]
                    index += 1
            elif tokens[index].startswith("="):
                setting += tokens[index]
                index += 1
        elif setting.endswith("=") and index < len(tokens):
            if not tokens[index].startswith("--"):
                setting += tokens[index]
                index += 1

        if "=" in setting:
            path, value = setting.split("=", 1)
            setting = f"{path.strip()}={value.strip()}"
        normalized.append(setting)

    return normalized


class _FlexibleArgumentParser(argparse.ArgumentParser):
    """Argument parser accepting optional shell spaces around ``--set =``."""

    def parse_args(self, args=None, namespace=None):
        return super().parse_args(_normalize_set_arguments(args), namespace)

    def parse_known_args(self, args=None, namespace=None):
        return super().parse_known_args(_normalize_set_arguments(args), namespace)


FlexibleArgumentParser = _FlexibleArgumentParser


def experiment_parser(
    description: str,
    *,
    config_path: str | None = None,
    complete: bool = True,
    resumable: bool = False,
    resume_path: str | None = None,
) -> argparse.ArgumentParser:
    """Build the parser shared by the runners and standalone validator.

    ``resumable`` adds ``--resume FOLDER`` (default ``resume_path``) for the
    runners whose saved results can be continued or replayed in place.
    """
    parser = _FlexibleArgumentParser(description=description)
    add_experiment_arguments(
        parser, config_path=config_path, resumable=resumable, resume_path=resume_path
    )
    if complete:
        enable_argcomplete(parser)
    return parser


def add_experiment_arguments(
    parser: argparse.ArgumentParser,
    *,
    config_path: str | None = None,
    resumable: bool = False,
    resume_path: str | None = None,
) -> argparse.ArgumentParser:
    """Add the shared case flags to ``parser`` (also a ``pydac`` subcommand)."""
    parser.add_argument(
        "--config",
        default=config_path,
        metavar="PATH",
        help="Partial preset JSON; omitted fields take the built-in defaults.",
    )
    parser.add_argument(
        "--case",
        default=None,
        metavar="PATH",
        help="Complete frozen-case JSON; run it exactly without default inheritance.",
    )
    parser.add_argument(
        "--export-case",
        default=None,
        metavar="PATH",
        help="Save the complete current configuration to PATH as a reusable case; do not run.",
    )
    parser.add_argument(
        "--overwrite-case",
        action="store_true",
        help="Allow --export-case to replace an existing target explicitly.",
    )
    parser.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Apply a canonical setting, for example --set grid.nz=25.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the resolved configuration without starting a simulation.",
    )
    parser.add_argument(
        "--print-resolved",
        action="store_true",
        help="Print the final configuration as JSON; do not run.",
    )
    parser.add_argument(
        "--show-settings",
        action=_ShowSettingsAction,
        nargs=0,
        help="List every accepted setting, its default, type, and constraints.",
    )
    if resumable:
        parser.add_argument(
            "--resume",
            default=resume_path,
            metavar="FOLDER",
            help="Continue or replay the saved run in FOLDER with its saved configuration; "
            "--set may change only operational settings.",
        )
    return parser


def enable_argcomplete(parser: argparse.ArgumentParser) -> None:
    """Enable shell completion when argcomplete is installed and active."""
    if argcomplete is not None:
        argcomplete.autocomplete(parser)


def resolve_experiment(
    args: argparse.Namespace, *, settings=None
) -> ExperimentConfig:
    return resolve_terminal_experiment(args, settings=settings)


def print_mode_summary(mode: str) -> None:
    """Print the short mode line every dry-run starts with."""
    print(f"Mode: {mode}")


def print_resolved_document(config: ExperimentConfig, *, dry_run: bool) -> None:
    """Print the resolved document; ``--dry-run`` also states that nothing ran."""
    print(json.dumps(experiment_to_document(config), indent=2, allow_nan=False))
    if dry_run:
        print("Dry run: no simulation started.")


__all__ = [
    "FlexibleArgumentParser",
    "add_experiment_arguments",
    "experiment_parser",
    "enable_argcomplete",
    "export_case_if_requested",
    "print_mode_summary",
    "print_resolved_document",
    "resolve_experiment",
    "resolve_runner_experiment",
]
