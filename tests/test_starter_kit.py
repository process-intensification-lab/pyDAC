"""`pydac get-started FOLDER` copies the starter kit in the repository layout, links included."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import pydac
from pydac import starter_kit
from pydac.cli import main
from pydac.config.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"\]\((?!https?:|mailto:|#)([^)\s\"\\]+)\)")


def _wheel_config():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["build"][
        "targets"]["wheel"]


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    return pydac.get_started(tmp_path_factory.mktemp("kit") / "my_pydac")


def test_the_kit_keeps_the_repository_layout(kit):
    copied = {p.relative_to(kit).as_posix() for p in kit.rglob("*") if p.is_file()}
    tracked = set()
    for member in starter_kit.KIT_MEMBERS:
        source = ROOT / member
        files = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
        tracked |= {p.relative_to(ROOT).as_posix() for p in files
                    if not set(p.relative_to(ROOT).parts) & starter_kit._SKIPPED}
    assert copied == tracked
    assert {"README.md", "LICENSE", "CITATION.cff", "docs/guide/01_installation.md",
            "examples/cases/01_first_steps.json",
            "examples/notebooks/01_first_steps.ipynb"} <= copied
    assert not any(part in starter_kit._SKIPPED for name in copied for part in name.split("/"))
    assert not (kit / "docs" / "manual").exists()


def test_every_link_of_the_kit_resolves_or_names_the_public_repository(kit):
    outside = []
    for file in [*kit.rglob("*.md"), *kit.rglob("*.ipynb")]:
        text = file.read_text(encoding="utf-8")
        for link in LINK_RE.findall(text):
            assert (file.parent / link.partition("#")[0]).exists(), (file, link)
        outside += re.findall(r"\]\((https://github\.com/[^)\s\"\\]+)\)", text)
    prefix = f"{starter_kit.REPOSITORY_URL}/blob/{starter_kit.repository_ref()}/"
    manual = [url for url in outside if url.startswith(prefix + "docs/manual/")]
    assert manual and all(
        (ROOT / url[len(prefix):].partition("#")[0]).is_file() for url in manual)
    # Only what is missing from the kit leaves it: the guide and the tutorials stay local.
    assert not [url for url in outside
                if url.startswith((prefix + "docs/guide/", prefix + "examples/"))]


def test_the_kit_refuses_a_folder_that_is_not_empty(tmp_path):
    (tmp_path / "old.txt").write_text("keep me")
    with pytest.raises(ConfigurationError, match="not an empty folder"):
        pydac.get_started(tmp_path)
    assert (tmp_path / "old.txt").read_text() == "keep me"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert (pydac.get_started(empty) / "README.md").is_file()


def test_pydac_get_started_command(tmp_path, capsys):
    dest = tmp_path / "kit"
    assert main(["get-started", str(dest)]) == 0
    assert "Starter kit copied to" in capsys.readouterr().out
    assert (dest / "examples" / "README.md").is_file()
    assert main(["get-started", str(dest)]) == 2
    assert "Configuration error:" in capsys.readouterr().err


def test_the_wheel_carries_every_kit_member_under_pydac_kit():
    wheel = _wheel_config()
    mapped = {**wheel["sources"], **wheel["force-include"]}
    assert mapped.pop("src/pydac") == "pydac"
    assert mapped == {member: f"pydac/_kit/{member}" for member in starter_kit.KIT_MEMBERS}
    assert set(wheel["only-include"]) == {"src/pydac"} | {
        member for member in starter_kit.KIT_MEMBERS if (ROOT / member).is_dir()}
    assert {f"**/{name}" for name in starter_kit._SKIPPED} == set(wheel["exclude"])


def test_the_repository_url_is_the_project_url():
    urls = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["urls"]
    assert urls["Repository"] == starter_kit.REPOSITORY_URL


def test_the_pypi_page_links_the_release_on_github():
    hook = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["metadata"][
        "hooks"]["fancy-pypi-readme"]
    [rewrite] = hook["substitutions"]
    release = re.sub(r"\.dev\d+$", "", pydac.__version__)
    assert rewrite["replacement"].startswith(f"]({starter_kit.REPOSITORY_URL}/blob/v{release}/")
    page = re.sub(rewrite["pattern"], rewrite["replacement"], (ROOT / "README.md").read_text())
    assert LINK_RE.findall((ROOT / "README.md").read_text()) and not LINK_RE.findall(page)


def test_outside_links_name_the_release_tag(monkeypatch):
    monkeypatch.setattr(starter_kit, "__version__", "3.0.0")
    assert starter_kit.repository_ref() == "v3.0.0"
    monkeypatch.setattr(starter_kit, "__version__", "3.0.1.dev0")
    assert starter_kit.repository_ref() == "main"
