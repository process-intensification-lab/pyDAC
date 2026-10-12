"""Shared utilities grouped by current-framework responsibility.

The bed-architecture models live in ``src/``, which the runners and workflows
put on the import path before importing anything else. A process that reaches
these utilities directly from the repository root — the terminal validator
resolving a document, a focused test — gets the same path here, so every
module below can state ``from bed_architectures import ...`` flatly and only
once.
"""

from pathlib import Path

