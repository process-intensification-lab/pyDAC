"""Column, bed-selection, geometry, and wall defaults."""

from __future__ import annotations

from typing import Any

from pydac.inputs.defaults._rules import Rule, nonempty_string, positive_number

BED_PROFILE_ALIASES = ("packed_bed_default",)


def bed_defaults() -> dict[str, Any]:
    """Return the public bed directive.

    ``None`` deliberately delegates the default packed-bed design to the
    architecture resolver, so the bed recipe is not copied into this layer.
    """
    return {"bed": None}


def geometry_defaults() -> dict[str, Any]:
    """Return the geometry: one input pair, named by `specified_by`, and the other computed."""
    return {
        "specified_by": "volume_aspect_ratio",
        "length_m": None,
        "radius_m": None,
        "aspect_ratio": 0.5,
        "volume_m3": 1e-3,
    }


def wall_defaults() -> dict[str, Any]:
    """Return wall material and the two optional wall-coefficient overrides.

    ``overall_htc_override_W_m2K`` replaces the lumped (Dixon overall) 1D bed-to-wall
    coefficient ``U_w``.
    ``bed_wall_htc_override_W_m2K`` replaces the bed-side wall coefficient
    ``h_w`` in 2D, where no lumped coefficient exists; it is the lever for
    selecting a wall-HTC correlation other than the resolved one.
    """
    return {
        "thickness_m": 0.01,
        "specific_heat_J_kgK": 500.0,
        "density_kg_m3": 8000.0,
        "conductivity_W_mK": 15.0,
        "overall_htc_override_W_m2K": None,
        "bed_wall_htc_override_W_m2K": None,
    }


def column_defaults() -> dict[str, Any]:
    """Return every column-domain default section."""
    return {
        **bed_defaults(),
        "geometry": geometry_defaults(),
        "wall": wall_defaults(),
    }


def column_rules():
    """Return geometry, wall, and delegated-bed constraints."""
    rules = {
        ("simulation", "bed"): Rule(
            kind="object",
            nullable=True,
            schema={
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "architecture": {
                                "enum": ["packed_bed"],
                            },
                            "design": {
                                "type": "object",
                                "additionalProperties": True,
                            },
                            "material": {
                                "type": "null",
                            },
                        },
                    },
                    {
                        "type": "string",
                        "enum": list(BED_PROFILE_ALIASES),
                    },
                ]
            },
        ),
        ("simulation", "bed", "architecture"): nonempty_string(),
        ("simulation", "bed", "design"): Rule(kind="object", opaque=True),
        ("simulation", "bed", "material"): Rule(
            kind="object", nullable=True, opaque=True
        ),
        ("simulation", "geometry", "specified_by"): Rule(
            kind="string",
            choices=("volume_aspect_ratio", "length_radius"),
            description=(
                "Which pair gives the column size: volume_m3 and aspect_ratio, or "
                "length_m and radius_m. The other pair is computed; leave it null."
            ),
        ),
        ("simulation", "geometry", "length_m"): positive_number(
            "Bed length; an input with specified_by=length_radius.", nullable=True
        ),
        ("simulation", "geometry", "radius_m"): positive_number(
            "Bed radius; an input with specified_by=length_radius.", nullable=True
        ),
        ("simulation", "geometry", "aspect_ratio"): positive_number(
            "Length over diameter, L/D; an input with specified_by=volume_aspect_ratio.",
            nullable=True,
        ),
        ("simulation", "geometry", "volume_m3"): positive_number(
            "Bed volume; an input with specified_by=volume_aspect_ratio.", nullable=True
        ),
        ("simulation", "wall", "overall_htc_override_W_m2K"): positive_number(
            nullable=True
        ),
        ("simulation", "wall", "bed_wall_htc_override_W_m2K"): positive_number(
            nullable=True
        ),
    }
    for name in wall_defaults():
        path = ("simulation", "wall", name)
        rules.setdefault(path, positive_number())
    return rules


__all__ = [
    "BED_PROFILE_ALIASES",
    "bed_defaults",
    "column_defaults",
    "column_rules",
    "geometry_defaults",
    "wall_defaults",
]
