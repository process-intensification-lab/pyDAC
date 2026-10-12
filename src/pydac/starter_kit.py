"""``pydac get-started FOLDER``: copy the starter kit (README, licence, citation, guide, tutorials).

```python
import pydac

pydac.get_started("my_pydac")   # my_pydac/README.md, docs/guide/, examples/, ...
```
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : find the kit -> refuse a non-empty FOLDER -> copy -> point the links that
#            leave the kit (manual, CONTRIBUTING, ...) to the public repository.
# Owns     : what the kit holds (KIT_MEMBERS) and where a copied link points.
# Not here : how the kit gets into the wheel (pyproject.toml, wheel `sources`).
# Contract : FOLDER keeps the repository layout, so the tutorials' relative paths
#            (../cases, ../data) and the guide's links work as in a clone.
# Guide    : docs/guide/01_installation.md
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from pydac.config.errors import ConfigurationError
from pydac.utils.version import __version__

# Kept in step with [project.urls] Repository (a test checks it).
REPOSITORY_URL = "https://github.com/process-intensification-lab/pyDAC"

# Repository paths copied, in the repository layout. The wheel carries them under pydac/_kit.
KIT_MEMBERS = ("README.md", "LICENSE", "CITATION.cff", "docs/guide", "examples")

# Never copied, also from a checkout that has run the tutorials.
_SKIPPED = {"simulation_results", "output_cases", "__pycache__", ".ipynb_checkpoints"}

_LINK_RE = re.compile(r"(\]\()(?!https?:|mailto:|file:|#)([^)\s\"\\]+)(\))")


def repository_ref() -> str:
    """The Git ref the kit's outside links name: the release tag, ``main`` for a dev build."""
    return "main" if "dev" in __version__ else f"v{__version__}"


def kit_source() -> Traversable:
    """The kit of the installed wheel, or the repository root of a checkout."""
    installed = resources.files("pydac") / "_kit"
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2]
    if all((checkout / member).exists() for member in KIT_MEMBERS):
        return checkout
    raise ConfigurationError(
        "this pyDAC installation has no starter kit (neither pydac/_kit nor a repository "
        f"checkout); download it from {REPOSITORY_URL}."
    )


def _copy(source: Traversable, target: Path, written: list[Path]) -> None:
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in sorted(source.iterdir(), key=lambda item: item.name):
            if child.name not in _SKIPPED:
                _copy(child, target / child.name, written)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        written.append(target)


def _outside_link(match: re.Match, file: Path, dest: Path) -> str:
    """A relative link that leaves the kit points to the same file in the public repository."""
    link = match.group(2)
    path, _, anchor = link.partition("#")
    target = (file.parent / path).resolve()
    if target.exists():
        return match.group(0)
    try:
        repo_path = target.relative_to(dest.resolve()).as_posix()
    except ValueError:
        return match.group(0)
    url = f"{REPOSITORY_URL}/blob/{repository_ref()}/{repo_path}"
    return f"{match.group(1)}{url}{'#' + anchor if anchor else ''}{match.group(3)}"


def get_started(folder) -> Path:
    """Copy README, LICENSE, CITATION.cff, the guide and the tutorials into ``folder``.

    ``folder`` must not exist or be empty: nothing is ever overwritten. Links to
    files outside the kit (the developer manual, CONTRIBUTING) point to the public
    repository at this version. Returns ``folder`` as a resolved path.
    """
    dest = Path(folder).expanduser()
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise ConfigurationError(f"{dest} already exists and is not an empty folder; "
                                 "choose a new folder.")
    source = kit_source()
    written: list[Path] = []
    for member in KIT_MEMBERS:
        item = source
        for part in member.split("/"):
            item = item / part
        _copy(item, dest / member, written)
    for file in written:
        if file.suffix in {".md", ".ipynb"}:
            text = file.read_text(encoding="utf-8")
            fixed = _LINK_RE.sub(lambda match: _outside_link(match, file, dest), text)
            if fixed != text:
                file.write_text(fixed, encoding="utf-8")
    return dest.resolve()


__all__ = ["KIT_MEMBERS", "REPOSITORY_URL", "get_started", "kit_source", "repository_ref"]
