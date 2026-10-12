"""Human-readable catalog of the settings accepted by pyDAC runners."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Iterator

from pydac.inputs.defaults import default_profile, default_rules
from pydac.inputs.defaults._rules import Rule


def _kind(value: Any, rule: Rule) -> str:
    if rule.kind is not None:
        return rule.kind
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    return "array"


def _public_path(path: tuple[str, ...]) -> str:
    if path[:3] == ("simulation", "cycle", "steps"):
        return ".".join(("steps",) + path[3:])
    if path[:1] == ("simulation",):
        return ".".join(path[1:])
    return ".".join(path)


def _walk(
    value: Any,
    path: tuple[str, ...],
    rules: Mapping[tuple[str, ...], Rule],
) -> Iterator[tuple[str, Any, Rule]]:
    rule = rules.get(path, Rule())
    if not isinstance(value, Mapping) or rule.opaque:
        yield _public_path(path), value, rule
        return
    for name, child in value.items():
        if path == () and name == "schema_version":
            continue
        yield from _walk(child, path + (name,), rules)


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, Mapping) and not value:
        return "{}"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _constraint_text(value: Any, rule: Rule) -> str:
    details = [_kind(value, rule)]
    if rule.nullable:
        details.append("nullable")
    if rule.read_only:
        details.append("derived/read-only")
    if rule.opaque:
        details.append("replace as one block")
    if rule.choices:
        details.append("one of " + ", ".join(repr(item) for item in rule.choices))
    if rule.exclusive_minimum is not None:
        details.append(f"> {rule.exclusive_minimum:g}")
    elif rule.minimum is not None:
        details.append(f">= {rule.minimum:g}")
    if rule.exclusive_maximum is not None:
        details.append(f"< {rule.exclusive_maximum:g}")
    elif rule.maximum is not None:
        details.append(f"<= {rule.maximum:g}")
    if rule.min_items is not None:
        details.append(f"at least {rule.min_items} item(s)")
    return "; ".join(details)


def settings_catalog() -> str:
    """Return the live catalog derived from the editable default profile."""
    profile = default_profile()
    rules = default_rules()
    rows = list(_walk(profile, (), rules))
    lines = [
        "pyDAC settings catalog",
        "=" * 58,
        "",
        "Use any path below with --set PATH=VALUE or in pydac.load_case(settings=...).",
        "The optional prefixes `simulation.` and `cycle.` are not needed.",
        "Values shown are the built-in defaults.",
        "",
    ]
    current_section = None
    for path, value, rule in rows:
        section = path.split(".", 1)[0]
        if path.startswith("steps."):
            section = ".".join(path.split(".")[:2])
        if section != current_section:
            if current_section is not None:
                lines.append("")
            lines.append(f"[{section}]")
            current_section = section
        lines.append(
            f"{path:<68} default={_format_value(value):<18} "
            f"{_constraint_text(value, rule)}"
        )
    lines.extend(
        [
            "",
            "Special blocks:",
            "  bed: set a complete mapping or use packed_bed_default.",
            "  bed.design / bed.material: opaque architecture-owned mappings.",
            "  adsorption.equilibrium_systems: replace the complete model-name keyed mapping.",
            "  adsorption.kinetics: replace the complete adsorbate-name keyed mapping.",
            "  Existing nested parameter leaves may also be changed directly",
            "  with a dotted --set path.",
            "  execution.step_options / execution.cycle_options: opaque override mappings.",
            "",
            "Computed fields are displayed for reference; a value written there must",
            "equal the computed one:",
            "  geometry.length_m and geometry.radius_m (or geometry.aspect_ratio and",
            "  geometry.volume_m3 with geometry.specified_by=length_radius), and the",
            "  volumetric flow of every step fed with steam.",
        ]
    )
    return "\n".join(lines)


def print_settings_catalog() -> None:
    """Print the live settings catalog to the terminal."""
    print(settings_catalog())


__all__ = ["print_settings_catalog", "settings_catalog"]
