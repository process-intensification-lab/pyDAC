"""Terminal validation and dry-run entry point for experiment configuration.

This is the entry point with no campaign attached: it resolves a document and
reports on it, so it is the honest way to see what a case JSON says before a
runner adds its own ``SETTINGS`` on top.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : the shared terminal flags plus `--validate`/`--write-resolved`/`--export-preset`.
# Owns     : exit codes and what is printed. One resolution serves every flag.
# Not here : the flag vocabulary itself (terminal.py); solving anything.
# Entry    : run() — reached as `pydac validate` and `python runners/validate_config.py`
# Guide    : docs/manual/01_inputs.md §3.1
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydac.config.errors import ConfigurationError
from pydac.config.loader import write_resolved_config
from pydac.config.serialization import experiment_to_document
from pydac.config.terminal import (
    enable_argcomplete,
    export_case_if_requested,
    experiment_parser,
    resolve_experiment,
)


def add_validate_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The flags this entry point adds to the shared experiment flags."""
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the resolved document without starting a simulation.",
    )
    parser.add_argument(
        "--write-resolved",
        type=Path,
        metavar="PATH",
        help="Save the resolved configuration to PATH.",
    )
    parser.add_argument(
        "--export-preset",
        type=Path,
        metavar="PATH",
        help="Save --config and --set as one preset file at PATH: only the fields they state.",
    )
    parser.add_argument(
        "--overwrite-preset",
        action="store_true",
        help="Allow --export-preset to replace an existing target explicitly.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    """The shared experiment flags plus the ones this entry point adds."""
    parser = experiment_parser(
        "Validate and resolve a pyDAC experiment without solving it.",
        complete=False,
    )
    add_validate_arguments(parser)
    enable_argcomplete(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    """Validate the parsed request; shared with ``pydac validate``."""
    try:
        # One resolution serves every flag; the document is the serialized
        # form of the very config that was just validated.
        config = resolve_experiment(args)
        document = experiment_to_document(config)
        # Every check of every requested file runs before the first one is written.
        preset = _checked_preset(args, config)
        export_case_if_requested(args, config)
        if preset is not None:
            _write_preset(args, preset)
        if args.write_resolved is not None:
            write_resolved_config(config, args.write_resolved)
        if args.dry_run:
            print("Dry run validation passed: configuration is valid; no simulation started.")
        if args.validate:
            print("Configuration valid.")
        if args.print_resolved:
            print(json.dumps(document, indent=2, allow_nan=False))
        if not (args.validate or args.dry_run or args.print_resolved):
            print("Configuration valid; no simulation started.")
        return 0
    except (ConfigurationError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


def _checked_preset(args: argparse.Namespace, config) -> str | None:
    """The text --export-preset will write, refused now if it cannot be written."""
    if args.export_preset is None:
        if args.overwrite_preset:
            raise ConfigurationError("--overwrite-preset requires --export-preset.")
        return None
    if args.case is not None:
        raise ConfigurationError(
            "--export-preset writes --config and --set as a preset; a --case is already "
            "complete: copy it, or load it with --config."
        )
    from pydac import api

    text = api._preset_text(args.settings, args.config, case=config)
    api._refuse_existing_preset(args.export_preset, overwrite=args.overwrite_preset)
    if args.export_case is not None:
        from pydac.inputs.case_io import refuse_existing_case

        refuse_existing_case(args.export_case, overwrite=args.overwrite_case)
    return text


def _write_preset(args: argparse.Namespace, text: str) -> None:
    from pydac import api

    output = api._write_preset(text, args.export_preset, overwrite=args.overwrite_preset)
    print(f"Exported preset: {output}")


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["add_validate_arguments", "build_parser", "main", "run"]
