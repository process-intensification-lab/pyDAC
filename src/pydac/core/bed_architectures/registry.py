"""Bed architectures registered at run time, next to the built-in packed bed.

A user package adds a bed architecture without editing pyDAC: it registers a
``BedArchitectureEntry`` under a name with ``pydac.register_bed_architecture``
and a case selects that name in ``simulation.bed.architecture``.  The registry
lives in one Python session, like the equilibrium and kinetic registries.

``packed_bed`` is built in and keeps its own code path in the column.  The
column reaches a registered architecture only through ``_RegisteredArchitecture``,
a pyDAC-owned snapshot of the entry that checks every result of the entry's
callables before the column binds it.  A model keeps the snapshot it was built
with, so registering a name again changes only the models built afterwards.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : stage 1 (case load: build_design/to_config through
#            build_bed_config) and model build (structure, closures).
# Owns     : the session registry, the entry snapshot, the result checks.
# Not here : the public registration call (pydac/extensions.py), the column
#            binding of the results (core/column_model.py).
# Guide    : docs/guide/08_extending.md "A bed architecture"
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from pydac.core.bed_architectures.base import (
    ARCHITECTURE_NAMES,
    ClosureInputs,
    ParticleProperties,
    RegisteredDesign,
    ResolvedStructure,
    ResolvedTransport,
    validate_architecture_name,
)
from pydac.core.bed_architectures.packed_bed import solve_quadratic_drag_velocity

if TYPE_CHECKING:
    from pydac.utils.postprocessing.context_schema import ContextParameter

_SPATIAL_MODES = ("1d", "2d")
_TOKEN = re.compile(r"[a-z][a-z0-9_]*\Z")
_CONTEXT_ID = re.compile(r"bed\.(design|material|provenance)\.([a-z][a-z0-9_]*)\Z")
_CONTEXT_VALUE_TYPES = {"design": "number", "material": "number", "provenance": "text"}
_DISPERSION_ARRAYS = ("Disp_ax", "Disp_rad", "inv_Pe_m_ax", "inv_Pe_m_rad")
_CALLBACKS = (
    "build_design",
    "to_config",
    "resolve_structure",
    "resolve_pressure_drop",
    "resolve_dispersion",
    "resolve_thermal_transport",
)
# Probe densities [kg/m3] for the pressure-law agreement check.
_PROBE_DENSITIES = (0.3, 1.2, 5.0)


class _ResultTypeError(TypeError):
    """A checked result failed; the message already names architecture and callable."""


class _ResultValueError(ValueError):
    """A checked result failed; the message already names architecture and callable."""


@dataclass(frozen=True)
class QuadraticPressureDropLaw:
    """The pressure law the column binds for a registered architecture.

    pyDAC rebuilds it from the provider law's two scalar coefficients, so the
    column, the 2D vector drag and the saved results all use the same
    quadratic law ``A*v + B*rho*v*|v| = -grad``, whatever object the provider
    returned.  Same operations, in the same order, as the packed-bed Ergun law.
    """

    name: str
    linear_coefficient: float
    inertial_coefficient_per_density: float

    def velocity_from_gradient(self, pressure_gradient: Any, local_density: Any) -> Any:
        """Return dimensionless superficial velocity from pressure gradient."""
        driving_force = -np.asarray(pressure_gradient, dtype=float)
        inertial_coefficient = (
            self.inertial_coefficient_per_density
            * np.asarray(local_density, dtype=float)
        )
        return solve_quadratic_drag_velocity(
            driving_force,
            self.linear_coefficient,
            inertial_coefficient,
        )


@dataclass(frozen=True)
class BedArchitectureEntry:
    """What a bed architecture gives the column, registered under a name.

    See guide 08 "A bed architecture" for the contract of each callable.
    """

    spatial_modes: tuple[str, ...]
    build_design: Callable[[Mapping[str, Any], Mapping[str, Any] | None], Any]
    to_config: Callable[[Any], tuple[dict, dict | None]]
    resolve_structure: Callable[[Any, ParticleProperties], ResolvedStructure]
    resolve_pressure_drop: Callable[[Any, ClosureInputs], Any]
    resolve_dispersion: Callable[[Any, ClosureInputs], Mapping[str, Any]]
    resolve_thermal_transport: Callable[[Any, ClosureInputs], ResolvedTransport]
    wall_htc_model: str
    provenance: Mapping[str, str] = field(default_factory=dict)
    context_parameters: tuple["ContextParameter", ...] = ()


# Session registry: name -> immutable snapshot of the registered entry.
_REGISTERED_BED_ARCHITECTURES: dict[str, "_RegisteredArchitecture"] = {}


def registered_bed_architecture_names() -> tuple[str, ...]:
    """Return the names registered in this session, sorted."""
    return tuple(sorted(_REGISTERED_BED_ARCHITECTURES))


def registered_bed_architecture(name: str) -> "_RegisteredArchitecture | None":
    """Return the registered snapshot for ``name``, or ``None`` for a built-in."""
    if name in ARCHITECTURE_NAMES:
        return None
    provider = _REGISTERED_BED_ARCHITECTURES.get(name)
    if provider is None:
        validate_architecture_name(name)  # raises the "Unknown architecture" error
    return provider


def register_bed_architecture_entry(name: str, entry: BedArchitectureEntry) -> None:
    """Check ``entry`` and store its snapshot as ``name`` (replacing any earlier one)."""
    if not isinstance(name, str) or not _TOKEN.match(name):
        raise ValueError(
            "A bed architecture name is one lowercase token "
            f"([a-z][a-z0-9_]*), got {name!r}."
        )
    if name in ARCHITECTURE_NAMES:
        raise ValueError(
            f"{name!r} is a built-in bed architecture; register yours under another name."
        )
    if not isinstance(entry, BedArchitectureEntry):
        raise TypeError(
            "A bed architecture is registered as a pydac.BedArchitectureEntry, "
            f"got {type(entry).__name__}."
        )
    _REGISTERED_BED_ARCHITECTURES[name] = _RegisteredArchitecture(name, entry)


def _number_mapping(value: Any, *, label: str) -> dict[str, float]:
    """Return a flat mapping of canonical keys to finite floats, or raise."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping, got {type(value).__name__}.")
    result = {}
    for key, number in value.items():
        if not isinstance(key, str) or not _TOKEN.match(key):
            raise ValueError(f"{label} key {key!r} is not one lowercase token.")
        if isinstance(number, (bool, np.bool_)) or not isinstance(number, Real):
            raise TypeError(
                f"{label}[{key!r}] must be a number (3.1.0 supports flat numeric "
                f"mappings only), got {type(number).__name__}."
            )
        number = float(number)
        if not math.isfinite(number):
            raise ValueError(f"{label}[{key!r}] must be finite.")
        result[key] = number
    return result


def _same_numbers(left: Mapping[str, float] | None, right: Mapping[str, float] | None) -> bool:
    """Bitwise equality of two canonical mappings, key order included."""
    if left is None or right is None:
        return left is right
    return [(k, v.hex()) for k, v in left.items()] == [(k, v.hex()) for k, v in right.items()]


class _RegisteredArchitecture:
    """Immutable snapshot of a registered entry; checks every callable result."""

    __slots__ = (
        "name",
        "spatial_modes",
        "wall_htc_model",
        "provenance",
        "context_parameters",
        "_callbacks",
    )

    def __init__(self, name: str, entry: BedArchitectureEntry) -> None:
        def fail(field_name, message, error=ValueError):
            raise error(f"Bed architecture {name!r}, entry field `{field_name}`: {message}")

        modes = entry.spatial_modes
        if isinstance(modes, str) or not isinstance(modes, (tuple, list)):
            fail("spatial_modes", "must be a tuple such as ('1d',) or ('1d', '2d').", TypeError)
        modes = tuple(modes)
        if not modes or len(set(modes)) != len(modes) or not set(modes) <= set(_SPATIAL_MODES):
            fail("spatial_modes", f"must be a nonempty subset of {_SPATIAL_MODES}, got {modes}.")
        callbacks = {}
        for callback in _CALLBACKS:
            function = getattr(entry, callback)
            if not callable(function):
                fail(callback, "must be callable.", TypeError)
            callbacks[callback] = function
        label = entry.wall_htc_model
        if not isinstance(label, str) or not label.strip():
            fail("wall_htc_model", "must be a nonempty string.", TypeError)
        if not isinstance(entry.provenance, Mapping):
            fail("provenance", "must be a mapping of text.", TypeError)
        provenance = {}
        for key, text in entry.provenance.items():
            if not isinstance(key, str) or not _TOKEN.match(key):
                fail("provenance", f"key {key!r} is not one lowercase token.")
            if not isinstance(text, str) or not text.strip():
                fail("provenance", f"[{key!r}] must be nonempty text.", TypeError)
            provenance[key] = text
        rows = self._context_rows(entry.context_parameters, provenance, fail)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "spatial_modes", modes)
        object.__setattr__(self, "wall_htc_model", label.strip())
        object.__setattr__(self, "provenance", MappingProxyType(provenance))
        object.__setattr__(self, "context_parameters", rows)
        object.__setattr__(self, "_callbacks", MappingProxyType(callbacks))

    def __setattr__(self, key, value):
        raise AttributeError("A registered bed architecture snapshot is read-only.")

    def __delattr__(self, key):
        raise AttributeError("A registered bed architecture snapshot is read-only.")

    @staticmethod
    def _context_rows(parameters, provenance, fail):
        from pydac.utils.postprocessing.context_schema import ContextParameter

        if isinstance(parameters, (str, Mapping)) or not isinstance(parameters, (tuple, list)):
            fail("context_parameters", "must be a tuple of ContextParameter.", TypeError)
        seen = set()
        rows = []
        for row in parameters:
            if not isinstance(row, ContextParameter):
                fail("context_parameters", f"{row!r} is not a ContextParameter.", TypeError)
            match = _CONTEXT_ID.match(row.parameter_id)
            if match is None:
                fail(
                    "context_parameters",
                    f"ID {row.parameter_id!r} must be bed.design.<key>, "
                    "bed.material.<key> or bed.provenance.<key> (one lowercase token).",
                )
            category, key = match.groups()
            if row.category != category:
                fail("context_parameters", f"{row.parameter_id!r} needs category {category!r}.")
            if row.value_type != _CONTEXT_VALUE_TYPES[category]:
                fail(
                    "context_parameters",
                    f"{row.parameter_id!r} needs value_type "
                    f"{_CONTEXT_VALUE_TYPES[category]!r}.",
                )
            if not isinstance(row.unit, str) or not row.unit.strip():
                fail("context_parameters", f"{row.parameter_id!r} needs a nonempty unit.")
            if row.parameter_id in seen:
                fail("context_parameters", f"duplicate ID {row.parameter_id!r}.")
            if category == "provenance" and key not in provenance:
                fail("context_parameters", f"{row.parameter_id!r} has no `provenance` text.")
            seen.add(row.parameter_id)
            rows.append(
                ContextParameter(row.category, row.parameter_id, row.unit, row.value_type)
            )
        return tuple(rows)

    def _declared_keys(self, category: str) -> list[str]:
        prefix = f"bed.{category}."
        return [
            row.parameter_id.removeprefix(prefix)
            for row in self.context_parameters
            if row.parameter_id.startswith(prefix)
        ]

    def _call(self, callback: str, *args):
        try:
            return self._callbacks[callback](*args)
        except Exception as exc:
            error = TypeError if isinstance(exc, TypeError) else ValueError
            raise error(f"Bed architecture {self.name!r}, `{callback}`: {exc}") from exc

    def _fail(self, callback: str, message: str, error=ValueError):
        error = _ResultTypeError if issubclass(error, TypeError) else _ResultValueError
        raise error(f"Bed architecture {self.name!r}, `{callback}` result: {message}")

    def _checked(self, callback: str, check, result):
        """Run ``check(result)``; any failure, of any kind, names the architecture
        and the callable and keeps the original error as its cause."""
        try:
            return check(result)
        except (_ResultTypeError, _ResultValueError):
            raise
        except Exception as exc:
            error = TypeError if isinstance(exc, TypeError) else ValueError
            raise error(f"Bed architecture {self.name!r}, `{callback}` result: {exc}") from exc

    # -- case load --------------------------------------------------------

    def _canonical(self, design_obj) -> tuple[dict[str, float], dict[str, float] | None]:
        return self._checked("to_config", self._check_config, self._call("to_config", design_obj))

    def _check_config(self, result):
        if type(result) is not tuple or len(result) != 2:
            self._fail("to_config", "must be a (design, material) tuple.", TypeError)
        design, material = result
        design = _number_mapping(design, label="design")
        if material is not None:
            material = _number_mapping(material, label="material")
        for category, values in (("design", design), ("material", material or {})):
            declared = self._declared_keys(category)
            if sorted(declared) != sorted(values):
                self._fail(
                    "to_config",
                    f"{category} keys {sorted(values)} must match the declared "
                    f"bed.{category}.* context rows {sorted(declared)} one to one.",
                )
        return design, material

    def build_design(self, design: Mapping[str, Any], material: Mapping[str, Any] | None):
        """Validate a case's design/material and return a ``RegisteredDesign``."""
        if not isinstance(design, Mapping):
            raise TypeError(f"Bed architecture {self.name!r}: `design` must be a mapping.")
        if material is not None and not isinstance(material, Mapping):
            raise TypeError(
                f"Bed architecture {self.name!r}: `material` must be a mapping or null."
            )
        copy = (lambda m: None if m is None else dict(m))
        obj = self._call("build_design", dict(design), copy(material))
        canonical_design, canonical_material = self._canonical(obj)
        again = self._call("build_design", dict(canonical_design), copy(canonical_material))
        design_again, material_again = self._canonical(again)
        if not (
            _same_numbers(canonical_design, design_again)
            and _same_numbers(canonical_material, material_again)
        ):
            self._fail(
                "to_config",
                "does not round-trip through build_design (the case fingerprint "
                "would not be stable).",
            )
        return RegisteredDesign(self.name, obj, canonical_design, canonical_material)

    # -- model build ------------------------------------------------------
    # Every result is read once and rebuilt as a new pyDAC-owned value, so the
    # column never binds the provider's object itself.

    def resolve_structure(self, design: RegisteredDesign, particle: ParticleProperties):
        result = self._call("resolve_structure", design.obj, particle)
        return self._checked("resolve_structure", self._check_structure, result)

    def _check_structure(self, result):
        if not isinstance(result, ResolvedStructure):
            self._fail("resolve_structure", "must be a ResolvedStructure.", TypeError)
        structure = ResolvedStructure(
            eps_flow=result.eps_flow,
            eps_total=result.eps_total,
            rho_sorbent=result.rho_sorbent,
            acc_structure=result.acc_structure,
        )
        if structure.rho_sorbent <= 0.0:
            self._fail("resolve_structure", "`rho_sorbent` must be positive.")
        if structure.eps_flow > structure.eps_total:
            self._fail("resolve_structure", "`eps_flow` must not exceed `eps_total`.")
        return structure

    def resolve_pressure_drop(self, design: RegisteredDesign, inputs: ClosureInputs):
        law = self._call("resolve_pressure_drop", design.obj, inputs)
        return self._checked("resolve_pressure_drop", self._check_pressure_drop, law)

    def _check_pressure_drop(self, law):
        name = getattr(law, "name", None)
        if not isinstance(name, str) or not name.strip():
            self._fail("resolve_pressure_drop", "`name` must be nonempty text.")
        coefficients = []
        for attribute in ("linear_coefficient", "inertial_coefficient_per_density"):
            if not hasattr(law, attribute):
                self._fail("resolve_pressure_drop", f"has no `{attribute}`.", TypeError)
            value = getattr(law, attribute)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                self._fail(
                    "resolve_pressure_drop",
                    f"`{attribute}` must be a real scalar (scalar quadratic laws only), "
                    f"got {type(value).__name__}.",
                    TypeError,
                )
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                self._fail("resolve_pressure_drop", f"`{attribute}` must be finite and positive.")
            coefficients.append(value)
        if not callable(getattr(law, "velocity_from_gradient", None)):
            self._fail("resolve_pressure_drop", "`velocity_from_gradient` must be callable.", TypeError)
        # The column binds the quadratic law of the two coefficients (below);
        # the provider's own law must be that law, at every probe density and
        # for both flow directions, or the provider meant something else.
        linear, inertial = coefficients
        for density in _PROBE_DENSITIES:
            scale = linear + inertial * density
            gradients = scale * np.array([-1.0, -1.0e-3, 0.0, 1.0e-3, 1.0])
            velocity = np.asarray(law.velocity_from_gradient(gradients, density))
            expected = solve_quadratic_drag_velocity(-gradients, linear, inertial * density)
            if velocity.shape != gradients.shape or not (
                velocity[2] == 0.0 and np.allclose(velocity, expected, rtol=1e-12, atol=0.0)
            ):
                self._fail(
                    "resolve_pressure_drop",
                    "velocity_from_gradient must solve A*v + B*rho*v*|v| = -grad with its "
                    "own linear_coefficient (A) and inertial_coefficient_per_density (B), "
                    f"at every density (failed at rho = {density} kg/m3).",
                )
        return QuadraticPressureDropLaw(name.strip(), linear, inertial)

    def resolve_dispersion(self, design: RegisteredDesign, inputs: ClosureInputs):
        result = self._call("resolve_dispersion", design.obj, inputs)
        return self._checked(
            "resolve_dispersion", lambda r: self._check_dispersion(r, inputs), result
        )

    def _check_dispersion(self, result, inputs):
        if not isinstance(result, Mapping):
            self._fail("resolve_dispersion", "must be a mapping.", TypeError)
        missing = [key for key in ("model", *_DISPERSION_ARRAYS) if key not in result]
        if missing:
            self._fail("resolve_dispersion", f"missing keys {missing}.")
        unexpected = [key for key in result if key not in ("model", *_DISPERSION_ARRAYS)]
        if unexpected:  # refused, not dropped: nothing would carry them (Astra AF3-02)
            self._fail(
                "resolve_dispersion",
                f"unexpected keys {unexpected}: the result has exactly `model` and "
                f"{list(_DISPERSION_ARRAYS)}; give provenance text through "
                "`BedArchitectureEntry.provenance`.",
            )
        model = result["model"]
        if not isinstance(model, str) or not model.strip():
            self._fail("resolve_dispersion", "`model` must be nonempty text.")
        resolved = {"model": model.strip()}
        shape = np.shape(inputs.molecular_diffusivity)
        for key in _DISPERSION_ARRAYS:
            values = np.array(result[key], dtype=float, copy=True, subok=False)
            if values.shape != shape or values.size != len(inputs.gas_species):
                self._fail(
                    "resolve_dispersion",
                    f"`{key}` must have the shape {shape} of the molecular diffusivity "
                    f"(one value per gas species), got {values.shape}.",
                )
            if not (np.all(np.isfinite(values)) and np.all(values > 0.0)):
                self._fail("resolve_dispersion", f"`{key}` must be finite and positive.")
            resolved[key] = values
        return resolved

    def resolve_thermal_transport(self, design: RegisteredDesign, inputs: ClosureInputs):
        result = self._call("resolve_thermal_transport", design.obj, inputs)
        return self._checked(
            "resolve_thermal_transport", lambda r: self._check_transport(r, inputs), result
        )

    def _check_transport(self, result, inputs):
        if not isinstance(result, ResolvedTransport):
            self._fail("resolve_thermal_transport", "must be a ResolvedTransport.", TypeError)
        transport = ResolvedTransport(
            lambda_ax=result.lambda_ax,
            lambda_rad=result.lambda_rad,
            h_wall=result.h_wall,
            wall_bed_htc=result.wall_bed_htc,
            pressure_drop_model=result.pressure_drop_model,
            dispersion_model=result.dispersion_model,
            diagnostics=dict(result.diagnostics),
        )
        if transport.lambda_rad is None:
            self._fail("resolve_thermal_transport", "`lambda_rad` must not be None.")
        if not inputs.is_1d and transport.wall_bed_htc != transport.h_wall:
            self._fail(
                "resolve_thermal_transport",
                "in 2D `wall_bed_htc` must equal `h_wall` (the bed-side coefficient "
                "couples bed and wall).",
            )
        return transport


__all__ = [
    "BedArchitectureEntry",
    "QuadraticPressureDropLaw",
    "register_bed_architecture_entry",
    "registered_bed_architecture",
    "registered_bed_architecture_names",
]
