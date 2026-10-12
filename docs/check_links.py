"""Verify that documentation links and code section references still resolve.

Two failure modes are cheap to introduce and expensive to notice: a relative
link that survives a file move as a dead path, and a ``# Guide: … §4.3`` header
in the code that points at a section number the guide no longer has. Both are
mechanical to check, so they are checked rather than reviewed.

Run from the repository root::

    python docs/check_links.py

Exit status is 0 when every link and every cited section resolves.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Owns     : link resolution for `docs/**/*.md` + the two READMEs, and the
#            `§` contract between code headers and the guide they cite.
# Not here : prose quality, or whether a link points at the *right* target.
# Entry    : main()
# Guide    : docs/manual/00_framework_map.md
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Markdown inline links, excluding pure-anchor and external targets.
LINK_RE = re.compile(r"\[[^\]]*\]\((?!https?:|mailto:|file:|#)([^)\s]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
# A code header such as `# Guide    : docs/manual/01_inputs.md §3.4, §8.5`
GUIDE_REF_RE = re.compile(r"(docs/[\w/]+\.md)((?:\s*§[\d.]+,?)*)")
SECTION_RE = re.compile(r"§([\d.]+?)\.?(?=[,\s]|$)")

SEARCHED_CODE = ("config", "runners", "workflows", "utils", "src")


def slugify(heading: str) -> str:
    """Convert heading text to a GitHub/Quarto-style anchor.

    Backticks and emphasis markers are dropped, but an underscore is kept: it
    is part of identifiers such as ``CONFIG_PATH`` that headings cite verbatim.
    """
    text = re.sub(r"[`*]", "", heading).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "-", text)


def anchors_of(path: Path) -> set[str]:
    return {slugify(title) for _, title in HEADING_RE.findall(path.read_text())}


def section_numbers_of(path: Path) -> set[str]:
    """Section numbers a document defines, e.g. {"2", "2.1", "3.1.1"}."""
    numbers = set()
    for _, title in HEADING_RE.findall(path.read_text()):
        match = re.match(r"([\d]+(?:\.[\d]+)*)\.?\s", title)
        if match:
            numbers.add(match.group(1))
    return numbers


def markdown_sources() -> list[Path]:
    """Return the maintained documentation plus the project README and CONTRIBUTING.

    `docs/development` and `docs/reports` are left out, as in the sdist: internal
    roadmaps and reports that cite files as they were when written.
    """
    files = sorted(REPO_ROOT.joinpath("docs").rglob("*.md"))
    files += [REPO_ROOT / "README.md", REPO_ROOT / "CONTRIBUTING.md"]
    return [
        f
        for f in files
        if ".quarto" not in f.parts
        and not {"reports", "development"} & set(f.relative_to(REPO_ROOT).parts)
    ]


def check_links() -> list[str]:
    """Every relative link resolves; `.html` targets resolve via their source."""
    problems = []
    for source in markdown_sources():
        for raw in LINK_RE.findall(source.read_text()):
            target, _, anchor = raw.partition("#")
            if not target:
                continue
            resolved = (source.parent / target).resolve()
            # Quarto renders `X.md` to `X.html`; check the source instead.
            if not resolved.exists() and resolved.suffix == ".html":
                resolved = resolved.with_suffix(".md")
            if not resolved.exists():
                problems.append(f"{source.relative_to(REPO_ROOT)}: dead link -> {raw}")
                continue
            if anchor and resolved.suffix == ".md" and anchor not in anchors_of(resolved):
                problems.append(
                    f"{source.relative_to(REPO_ROOT)}: unknown anchor -> {raw}"
                )
    return problems


def check_guide_sections() -> list[str]:
    """Every `docs/<file>.md §N` cited by the code exists in that file."""
    problems = []
    sources = [REPO_ROOT / "README.md"]
    for directory in SEARCHED_CODE:
        sources.extend(sorted(REPO_ROOT.joinpath(directory).rglob("*.py")))

    for source in sources:
        if not source.exists():
            continue
        for guide, sections in GUIDE_REF_RE.findall(source.read_text()):
            guide_path = REPO_ROOT / guide
            where = source.relative_to(REPO_ROOT)
            if not guide_path.exists():
                problems.append(f"{where}: cites missing guide -> {guide}")
                continue
            defined = section_numbers_of(guide_path)
            for number in SECTION_RE.findall(sections):
                if number not in defined:
                    problems.append(f"{where}: {guide} has no §{number}")
    return problems


def main() -> int:
    problems = check_links() + check_guide_sections()
    if problems:
        print(f"{len(problems)} problem(s):\n")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("All documentation links and cited guide sections resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
