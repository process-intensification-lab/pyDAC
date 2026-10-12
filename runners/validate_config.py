"""Validate or dry-run a pyDAC JSON experiment configuration (`pydac validate`)."""
# PYTHON_ARGCOMPLETE_OK

from __future__ import annotations

import sys

from pydac.cli import main as pydac_main


def main(argv=None):
    return pydac_main(["validate", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
