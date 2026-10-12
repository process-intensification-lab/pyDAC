"""Dual CLI/VS Code launcher for a local NSGA-II optimization (`pydac nsga2`)."""
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import sys

from pydac.cli import LauncherSettings, main as pydac_main

# Guide: docs/manual/07_nsga2_optimization.md
# Terminal: python runners/run_NSGA2.py --bounds bounds.json --pop-size 20 --dry-run
#           (same as `pydac nsga2 ...`; --report CAMPAIGN regenerates the Pareto outputs).
# Base-case edits use the ordinary case/config/--set interface.
CASE_PATH = None
CONFIG_PATH = None
DEVELOPMENT_SETTINGS = {}
# Set RESUME_PATH to a campaign folder to continue it (same as --resume).
RESUME_PATH = None


def main(argv=None):
    """Run `pydac nsga2` with this file's constants (the terminal flags still apply)."""
    return pydac_main(
        ["nsga2", *(sys.argv[1:] if argv is None else argv)],
        launcher=LauncherSettings(
            case_path=CASE_PATH,
            config_path=CONFIG_PATH,
            resume_path=RESUME_PATH,
            development_settings=DEVELOPMENT_SETTINGS,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
