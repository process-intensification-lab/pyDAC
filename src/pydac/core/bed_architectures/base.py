"""Detached contracts for column bed architectures.

This module deliberately has no dependency on ``ColumnModelMoL`` or any
solver, runner, plotting, or persistence implementation.  It contains only
the immutable design/context records, small validation helpers, and protocols
that the later architecture implementations will satisfy.

The source follows the contract-resolution narrative:

1. validation primitives and registered names;
2. architecture designs and public configuration;
3. common geometry, particle, gas, and flow inputs;
4. structure and transport contexts/results;
5. complete resolved architecture and behavioral protocols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


ARCHITECTURE_NAMES = frozenset({"packed_bed"})
_SPATIAL_MODES = frozenset({"1d", "2d"})

# The framework default packed bed.  This module owns the values; the public
# input profile carries `null` for `bed` and delegates here rather than
# restating them.
DEFAULT_PACKED_BED_DESIGN: Mapping[str, float] = MappingProxyType({"eps_b": 0.38})

# Selectable stagnant (no-flow) bed conductivity closures.  The default is the
# modified Zehner-Schlunder law pyDAC has always used; `specchia_1980` is the
# correlation used by the Zanco et al. (2022) reference experiment, and is
# selected through the optional `bed.design.stagnant_conductivity_model` field.
STAGNANT_CONDUCTIVITY_MODELS = frozenset({"zehner_schlunder", "specchia_1980"})
DEFAULT_STAGNANT_CONDUCTIVITY_MODEL = "zehner_schlunder"

# Selectable packed-bed bed-to-wall heat-transfer closures.  Dixon remains the
# historical/default law; Specchia is the correlation used by Zanco et al.
WALL_HTC_MODELS = frozenset({"dixon", "specchia_1980"})
DEFAULT_WALL_HTC_MODEL = "dixon"

# Public bed flow models.  `uniform` is the historical native structure.
# `wall-channeling-reduced` resolves a radial voidage profile in storage and in
# the Ergun drag, with a prescribed inlet allocation derived from that same
# closure and no radial momentum equation.  `wall-channeling` additionally
# solves an axial Brinkman balance and stays unavailable until it is qualified.
FLOW_MODELS = ("uniform", "wall-channeling-reduced", "wall-channeling")
DEFAULT_FLOW_MODEL = "uniform"
CHANNELING_FLOW_MODELS = frozenset({"wall-channeling-reduced", "wall-channeling"})
UNAVAILABLE_FLOW_MODELS: Mapping[str, str] = MappingProxyType(
    {
        "wall-channeling": (
            "`wall-channeling` (full axial Brinkman) is not implemented yet; it "
            "becomes available only once its own qualification phase closes."
        )
    }
)

# Radial voidage correlations available to the channeling modes.  `de_klerk`
# is the near-wall oscillatory packing profile of de Klerk (2003), whose core
# value is solved so that the exact cylindrical mean equals the measured
# `eps_b`.  `giese` is the smooth exponential profile of Giese (1998), used by
# Gaeini et al. (2017) Eq. 2, normalised the same way; it is a published
# alternative to de Klerk, not a tuning of it, and at large `D/d_p` it gives a
# weaker wall excess.  `tabulated` consumes a user-supplied
# `radial_porosity_profile`.
CHANNELING_POROSITY_MODELS = frozenset({"de_klerk", "giese", "tabulated"})
DEFAULT_CHANNELING_POROSITY_MODEL = "de_klerk"

# Thickness of the fixed, wall-anchored physical layers on which the Ergun
# resistance is evaluated, in particle diameters.  It is a declared closure
# length, not a mesh parameter: the de Klerk profile reaches eps = 1 exactly at
# the wall, where both Ergun coefficients vanish, so a pointwise continuum
# limit does not exist.  The layers regularise that singularity at a physical
# scale and therefore belong to the case identity.  The qualified band is
# 0.5-2 d_p; 3 d_p smears the whole channelling region and is rejected.
DEFAULT_DRAG_LAYER_WIDTH_DP = 1.0
DRAG_LAYER_WIDTH_DP_BOUNDS = (0.5, 2.0)

# How the prescribed feed is allocated over the inlet rings of a channeling
# bed.  `uniform` imposes the same flux per unit area and lets the interior
# drag and radial convection develop the channeling profile over a short
# entrance length; `distributed` pre-allocates the feed with the
# reduced-closure streamtube weights.  Both conserve the total molar flow.
CHANNELING_INLET_DISTRIBUTIONS = ("uniform", "distributed")
DEFAULT_CHANNELING_INLET_DISTRIBUTION = "uniform"


# ---------------------------------------------------------------------------
# 1. Validation primitives and registered names
# ---------------------------------------------------------------------------

def _finite_real(value: Real, name: str) -> float:
    """Return a finite float, rejecting booleans and non-real values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"`{name}` must be finite.")
    return result


def _positive(value: Real, name: str) -> float:
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return result


def _fraction(value: Real, name: str) -> float:
    result = _finite_real(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"`{name}` must be strictly between 0 and 1.")
    return result


def _nonnegative(value: Real, name: str) -> float:
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"`{name}` must be non-negative.")
    return result


def _inlet_flux_profile(
    value: Any, name: str = "inlet_flux_profile"
) -> tuple[tuple[float, float], ...]:
    points = tuple(
        (_finite_real(x, f"{name} r/R"), _nonnegative(w, f"{name} flux"))
        for x, w in value
    )
    xs = [x for x, _ in points]
    if len(points) < 2 or xs[0] != 0.0 or xs[-1] != 1.0:
        raise ValueError(f"`{name}` needs >= 2 points spanning r/R = 0 to 1.")
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError(f"`{name}` r/R values must be strictly increasing.")
    if not any(w > 0.0 for _, w in points):
        raise ValueError(f"`{name}` must carry a positive flux somewhere.")
    return points


def _exact_mapping(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unknown {sorted(extra)}")
        raise ValueError(f"Invalid `{name}` fields: {', '.join(details)}.")
    return dict(value)


def validate_stagnant_conductivity_model(name: str) -> str:
    """Validate and return one of the registered stagnant-conductivity models.

    Matching ignores case and surrounding whitespace, and the canonical
    lower-case name is returned, so how a case file spells the selector never
    reaches the resolved configuration or the case fingerprint.
    """
    if not isinstance(name, str):
        raise TypeError("`stagnant_conductivity_model` must be a string.")
    canonical = name.strip().lower()
    if canonical not in STAGNANT_CONDUCTIVITY_MODELS:
        choices = ", ".join(sorted(STAGNANT_CONDUCTIVITY_MODELS))
        raise ValueError(
            f"Unknown stagnant conductivity model {name!r}. Choose one of: {choices}."
        )
    return canonical


def validate_wall_htc_model(name: str) -> str:
    """Validate and return one of the registered bed-to-wall HTC models."""
    if not isinstance(name, str):
        raise TypeError("`wall_htc_model` must be a string.")
    canonical = name.strip().lower()
    if canonical not in WALL_HTC_MODELS:
        choices = ", ".join(sorted(WALL_HTC_MODELS))
        raise ValueError(
            f"Unknown wall HTC model {name!r}. Choose one of: {choices}."
        )
    return canonical


def validate_flow_model(name: str) -> str:
    """Validate and return one of the registered public bed flow models.

    Matching ignores case and surrounding whitespace only: the three documented
    spellings are the only accepted ones, so a near-miss such as
    ``wall_channeling_reduced`` is an error rather than a silent uniform run.
    """
    if not isinstance(name, str):
        raise TypeError("`flow_model` must be a string.")
    canonical = name.strip().lower()
    if canonical not in FLOW_MODELS:
        choices = ", ".join(FLOW_MODELS)
        raise ValueError(
            f"Unknown flow model {name!r}. Choose one of: {choices}."
        )
    if canonical in UNAVAILABLE_FLOW_MODELS:
        raise ValueError(UNAVAILABLE_FLOW_MODELS[canonical])
    return canonical


def validate_channeling_porosity_model(name: str) -> str:
    """Validate and return one of the registered radial voidage correlations."""
    if not isinstance(name, str):
        raise TypeError("`channeling.porosity_model` must be a string.")
    canonical = name.strip().lower()
    if canonical not in CHANNELING_POROSITY_MODELS:
        choices = ", ".join(sorted(CHANNELING_POROSITY_MODELS))
        raise ValueError(
            f"Unknown channeling porosity model {name!r}. Choose one of: {choices}."
        )
    return canonical


@dataclass(frozen=True)
class ChannelingOptions:
    """Immutable options owned by the channeling bed flow models.

    ``drag_layer_width_dp`` is the declared closure length ``W`` in particle
    diameters.  It regularises a genuine singularity of the voidage
    correlation rather than a mesh, so it participates in the case identity and
    is always serialized once a channeling mode is selected.

    ``inlet_distribution`` selects how the prescribed feed enters the bed:
    ``"uniform"`` imposes the same flux per unit area on every ring and lets the
    interior drag and radial convection redistribute it; ``"distributed"``
    pre-allocates it with the reduced-closure streamtube weights.
    """

    porosity_model: str = DEFAULT_CHANNELING_POROSITY_MODEL
    drag_layer_width_dp: float = DEFAULT_DRAG_LAYER_WIDTH_DP
    inlet_distribution: str = DEFAULT_CHANNELING_INLET_DISTRIBUTION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "porosity_model",
            validate_channeling_porosity_model(self.porosity_model),
        )
        if self.inlet_distribution not in CHANNELING_INLET_DISTRIBUTIONS:
            choices = ", ".join(CHANNELING_INLET_DISTRIBUTIONS)
            raise ValueError(
                "`channeling.inlet_distribution` must be one of: "
                f"{choices}; received {self.inlet_distribution!r}."
            )
        width = _positive(self.drag_layer_width_dp, "channeling.drag_layer_width_dp")
        low, high = DRAG_LAYER_WIDTH_DP_BOUNDS
        if not low <= width <= high:
            raise ValueError(
                "`channeling.drag_layer_width_dp` must lie in the qualified band "
                f"[{low}, {high}] particle diameters; received {width}."
            )
        object.__setattr__(self, "drag_layer_width_dp", width)

    def to_config_dict(self) -> dict[str, Any]:
        # Resolved defaults are always emitted: the closure length is part of
        # the physics a channeling case declares, never an implicit constant.
        return {
            "porosity_model": self.porosity_model,
            "drag_layer_width_dp": self.drag_layer_width_dp,
            "inlet_distribution": self.inlet_distribution,
        }


def resolve_channeling_options(value: Any) -> ChannelingOptions:
    """Materialize the channeling option block, rejecting unknown fields."""
    if value is None:
        return ChannelingOptions()
    if isinstance(value, ChannelingOptions):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("`channeling` must be a mapping.")
    known = {"porosity_model", "drag_layer_width_dp", "inlet_distribution"}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"Unknown `channeling` fields: {unknown}.")
    return ChannelingOptions(
        **{key: value[key] for key in known if key in value}
    )


def validate_architecture_name(name: str) -> str:
    """Validate and return one of the registered architecture names."""
    if not isinstance(name, str):
        raise TypeError("`architecture` must be a string.")
    if name not in ARCHITECTURE_NAMES:
        choices = ", ".join(sorted(ARCHITECTURE_NAMES))
        raise ValueError(f"Unknown architecture {name!r}; choose one of {choices}.")
    return name


def validate_spatial_mode(mode: str, *, architecture: str | None = None) -> str:
    """Validate a supported spatial mode, optionally for one architecture."""
    if not isinstance(mode, str):
        raise TypeError("`spatial_mode` must be a string.")
    if mode not in _SPATIAL_MODES:
        choices = ", ".join(sorted(_SPATIAL_MODES))
        raise ValueError(f"Unsupported spatial mode {mode!r}; choose one of {choices}.")
    if architecture is not None:
        architecture = validate_architecture_name(architecture)
    return mode


# ---------------------------------------------------------------------------
# 2. Architecture designs and public configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PackedBedDesign:
    """Immutable design parameters for a conventional packed bed."""

    eps_b: float
    stagnant_conductivity_model: str = DEFAULT_STAGNANT_CONDUCTIVITY_MODEL
    wall_htc_model: str = DEFAULT_WALL_HTC_MODEL
    # (r/R, relative superficial flux) pairs; None = uniform inlet flux.
    inlet_flux_profile: tuple[tuple[float, float], ...] | None = None
    # (r/R, local bed voidage) pairs; area mean must match eps_b. None = uniform.
    radial_porosity_profile: tuple[tuple[float, float], ...] | None = None
    # Public bed flow model.  `None` means the case never mentioned it, which
    # is NOT the same as an explicit `uniform`: a case that omits the selector
    # but supplies `radial_porosity_profile` keeps the historical prescribed
    # prototype physics, whereas an explicit `uniform` rejects that profile.
    flow_model: str | None = None
    channeling: ChannelingOptions | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "eps_b", _fraction(self.eps_b, "eps_b"))
        if self.flow_model is not None:
            object.__setattr__(
                self, "flow_model", validate_flow_model(self.flow_model)
            )
        if self.inlet_flux_profile is not None:
            object.__setattr__(
                self,
                "inlet_flux_profile",
                _inlet_flux_profile(self.inlet_flux_profile),
            )
        if self.radial_porosity_profile is not None:
            profile = _inlet_flux_profile(
                self.radial_porosity_profile, name="radial_porosity_profile"
            )
            if any(not 0.0 < e <= 1.0 for _, e in profile):
                raise ValueError("`radial_porosity_profile` voidage must be in (0, 1].")
            object.__setattr__(self, "radial_porosity_profile", profile)
        object.__setattr__(
            self,
            "stagnant_conductivity_model",
            validate_stagnant_conductivity_model(self.stagnant_conductivity_model),
        )
        object.__setattr__(
            self,
            "wall_htc_model",
            validate_wall_htc_model(self.wall_htc_model),
        )
        self._validate_flow_model_combination()

    def _validate_flow_model_combination(self) -> None:
        """Reject input combinations whose intended physics is ambiguous."""
        mode = self.resolved_flow_model
        if mode not in CHANNELING_FLOW_MODELS:
            if self.channeling is not None:
                raise ValueError(
                    "`channeling` belongs to the channeling flow models; "
                    f"`flow_model={mode!r}` must not carry it."
                )
            if self.flow_model is not None and self.radial_porosity_profile is not None:
                raise ValueError(
                    "An explicit `flow_model='uniform'` rejects "
                    "`radial_porosity_profile`. Remove the profile, or select "
                    "`wall-channeling-reduced` with "
                    "`channeling.porosity_model='tabulated'`, which is a new "
                    "case identity and intentionally different physics."
                )
            return

        object.__setattr__(
            self, "channeling", resolve_channeling_options(self.channeling)
        )
        if self.inlet_flux_profile is not None:
            raise ValueError(
                f"`flow_model={mode!r}` derives its own inlet allocation from "
                "the resolved bed closure; `inlet_flux_profile` would compete "
                "with it. Use `flow_model='uniform'` to prescribe a measured "
                "or CFD inlet profile instead."
            )
        porosity_model = self.channeling.porosity_model
        if porosity_model in ("de_klerk", "giese") and self.radial_porosity_profile is not None:
            raise ValueError(
                f"`channeling.porosity_model={porosity_model!r}` generates the "
                "radial voidage itself; remove `radial_porosity_profile` or "
                "select `porosity_model='tabulated'`."
            )
        if porosity_model == "tabulated" and self.radial_porosity_profile is None:
            raise ValueError(
                "`channeling.porosity_model='tabulated'` requires "
                "`radial_porosity_profile`."
            )

    @property
    def resolved_flow_model(self) -> str:
        """Return the flow model actually in force, omitted meaning uniform."""
        return self.flow_model or DEFAULT_FLOW_MODEL

    @property
    def compatibility_status(self) -> str:
        """Name the internal replay status of a selector-free radial profile.

        A case that omits `flow_model` but carries `radial_porosity_profile`
        is the historical prescribed-profile prototype: local storage, the
        original scalar drag and the original inlet handling.  It is reported
        under its own name rather than silently relabelled as reduced.
        """
        if self.flow_model is None and self.radial_porosity_profile is not None:
            return "legacy-prescribed-profile"
        return "native"

    def to_config_dict(self) -> dict[str, Any]:
        # Selectors are emitted only when they depart from their historical laws,
        # so cases that never mention them keep their exact canonical mapping -
        # and therefore their case fingerprint.
        config: dict[str, Any] = {"eps_b": self.eps_b}
        if self.stagnant_conductivity_model != DEFAULT_STAGNANT_CONDUCTIVITY_MODEL:
            config["stagnant_conductivity_model"] = self.stagnant_conductivity_model
        if self.wall_htc_model != DEFAULT_WALL_HTC_MODEL:
            config["wall_htc_model"] = self.wall_htc_model
        if self.inlet_flux_profile is not None:
            config["inlet_flux_profile"] = [list(p) for p in self.inlet_flux_profile]
        if self.radial_porosity_profile is not None:
            config["radial_porosity_profile"] = [
                list(p) for p in self.radial_porosity_profile
            ]
        # An omitted or explicitly uniform selector serializes to nothing, so
        # ordinary cases keep their historical canonical mapping and case
        # fingerprint.  A channeling mode always serializes its resolved
        # options, because they are the physics it declares.
        if self.resolved_flow_model != DEFAULT_FLOW_MODEL:
            config["flow_model"] = self.resolved_flow_model
            config["channeling"] = self.channeling.to_config_dict()
        return config


def build_bed_config(
    *,
    architecture: str,
    design: Mapping[str, Real],
    material: Mapping[str, Real] | None = None,
) -> dict[str, Any]:
    """Validate and return the canonical JSON/pickle-safe bed configuration."""
    architecture = validate_architecture_name(architecture)

    if architecture == "packed_bed":
        # The closure selector is optional: a case that does not declare it keeps
        # the historical Zehner-Schlunder law, so existing inputs stay valid and
        # a canonical mapping round-trips through this function unchanged.
        design = dict(design)
        stagnant_model = design.pop(
            "stagnant_conductivity_model", DEFAULT_STAGNANT_CONDUCTIVITY_MODEL
        )
        wall_htc_model = design.pop("wall_htc_model", DEFAULT_WALL_HTC_MODEL)
        inlet_flux_profile = design.pop("inlet_flux_profile", None)
        radial_porosity_profile = design.pop("radial_porosity_profile", None)
        if "flow_model" in design and design["flow_model"] is None:
            raise ValueError(
                "`flow_model` must not be null; omit the key to keep the "
                "historical uniform bed structure."
            )
        flow_model = design.pop("flow_model", None)
        channeling = design.pop("channeling", None)
        fields = _exact_mapping(design, {"eps_b"}, "design")
        if material is not None:
            raise ValueError("Packed-bed configuration must use `material=None`.")
        parsed = PackedBedDesign(
            eps_b=fields["eps_b"],
            stagnant_conductivity_model=stagnant_model,
            wall_htc_model=wall_htc_model,
            inlet_flux_profile=inlet_flux_profile,
            radial_porosity_profile=radial_porosity_profile,
            flow_model=flow_model,
            channeling=channeling,
        )
        return {
            "architecture": architecture,
            "design": parsed.to_config_dict(),
            "material": None,
        }



def resolve_bed_config(
    bed_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a fresh canonical bed configuration, applying the PB default."""
    if bed_config is None:
        return build_bed_config(
            architecture="packed_bed",
            design=DEFAULT_PACKED_BED_DESIGN,
        )

    fields = _exact_mapping(
        bed_config,
        {"architecture", "design", "material"},
        "bed_config",
    )
    return build_bed_config(
        architecture=fields["architecture"],
        design=fields["design"],
        material=fields["material"],
    )


# ---------------------------------------------------------------------------
# 3. Common runtime input contexts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnGeometry:
    """Column geometry context, with all lengths in metres."""

    length: float
    radius: float
    volume: float
    aspect_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "length", _positive(self.length, "length"))
        object.__setattr__(self, "radius", _positive(self.radius, "radius"))
        object.__setattr__(self, "volume", _positive(self.volume, "volume"))
        object.__setattr__(
            self,
            "aspect_ratio",
            _positive(self.aspect_ratio, "aspect_ratio"),
        )

    def to_context_dict(self) -> dict[str, float]:
        return {
            "length": self.length,
            "radius": self.radius,
            "volume": self.volume,
            "aspect_ratio": self.aspect_ratio,
        }


@dataclass(frozen=True)
class ParticleProperties:
    """Common particle properties needed by architecture resolution."""

    density: float
    porosity: float
    radius: float
    thermal_conductivity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "density", _positive(self.density, "density"))
        object.__setattr__(self, "porosity", _fraction(self.porosity, "porosity"))
        object.__setattr__(self, "radius", _positive(self.radius, "radius"))
        object.__setattr__(
            self,
            "thermal_conductivity",
            _positive(self.thermal_conductivity, "thermal_conductivity"),
        )

    def to_context_dict(self) -> dict[str, float]:
        return {
            "density": self.density,
            "porosity": self.porosity,
            "radius": self.radius,
            "thermal_conductivity": self.thermal_conductivity,
        }


@dataclass(frozen=True)
class GasProperties:
    """Gas-mixture properties in SI units."""

    density: float
    viscosity: float
    thermal_conductivity: float
    heat_capacity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "density", _positive(self.density, "density"))
        object.__setattr__(self, "viscosity", _positive(self.viscosity, "viscosity"))
        object.__setattr__(
            self,
            "thermal_conductivity",
            _positive(self.thermal_conductivity, "thermal_conductivity"),
        )
        object.__setattr__(
            self,
            "heat_capacity",
            _positive(self.heat_capacity, "heat_capacity"),
        )

    def to_context_dict(self) -> dict[str, float]:
        return {
            "density": self.density,
            "viscosity": self.viscosity,
            "thermal_conductivity": self.thermal_conductivity,
            "heat_capacity": self.heat_capacity,
        }


@dataclass(frozen=True)
class FlowProperties:
    """Flow quantities passed explicitly to transport models."""

    superficial_velocity: float
    reference_velocity: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "superficial_velocity",
            _finite_real(self.superficial_velocity, "superficial_velocity"),
        )
        object.__setattr__(
            self,
            "reference_velocity",
            _positive(self.reference_velocity, "reference_velocity"),
        )

    def to_context_dict(self) -> dict[str, float]:
        return {
            "superficial_velocity": self.superficial_velocity,
            "reference_velocity": self.reference_velocity,
        }


@dataclass(frozen=True)
class StructureContext:
    """Read-only inputs for the first architecture-resolution stage."""

    column: ColumnGeometry
    particle: ParticleProperties

    def to_context_dict(self) -> dict[str, Any]:
        return {
            "column": self.column.to_context_dict(),
            "particle": self.particle.to_context_dict(),
        }


@dataclass(frozen=True)
class ResolvedStructure:
    """Semantic structure/inventory properties consumed by the column.

    Flow and total porosity, sorbent inventory, and additional structural heat
    capacity. The conventional packed bed resolves the latter to zero.
    """

    eps_flow: float
    eps_total: float
    rho_sorbent: float
    acc_structure: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "eps_flow", _fraction(self.eps_flow, "eps_flow"))
        object.__setattr__(self, "eps_total", _fraction(self.eps_total, "eps_total"))
        object.__setattr__(
            self,
            "rho_sorbent",
            _nonnegative(self.rho_sorbent, "rho_sorbent"),
        )
        object.__setattr__(
            self,
            "acc_structure",
            _nonnegative(self.acc_structure, "acc_structure"),
        )

    def to_context_dict(self) -> dict[str, float]:
        return {
            "eps_flow": self.eps_flow,
            "eps_total": self.eps_total,
            "rho_sorbent": self.rho_sorbent,
            "acc_structure": self.acc_structure,
        }


@dataclass(frozen=True)
class TransportContext:
    """Read-only inputs for transport/closure resolution."""

    column: ColumnGeometry
    particle: ParticleProperties
    gas: GasProperties
    flow: FlowProperties
    structure: ResolvedStructure
    spatial_mode: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "spatial_mode",
            validate_spatial_mode(self.spatial_mode),
        )

    def to_context_dict(self) -> dict[str, Any]:
        return {
            "column": self.column.to_context_dict(),
            "particle": self.particle.to_context_dict(),
            "gas": self.gas.to_context_dict(),
            "flow": self.flow.to_context_dict(),
            "structure": self.structure.to_context_dict(),
            "spatial_mode": self.spatial_mode,
        }


# ---------------------------------------------------------------------------
# 4. Resolved transport and architecture results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedTransport:
    """Serializable transport result shared by architecture implementations."""

    lambda_ax: float
    lambda_rad: float | None
    h_wall: float
    wall_bed_htc: float
    pressure_drop_model: str
    dispersion_model: str
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lambda_ax", _positive(self.lambda_ax, "lambda_ax"))
        if self.lambda_rad is not None:
            object.__setattr__(
                self,
                "lambda_rad",
                _positive(self.lambda_rad, "lambda_rad"),
            )
        object.__setattr__(self, "h_wall", _positive(self.h_wall, "h_wall"))
        object.__setattr__(
            self,
            "wall_bed_htc",
            _positive(self.wall_bed_htc, "wall_bed_htc"),
        )
        for key in ("pressure_drop_model", "dispersion_model"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`{key}` must be a non-empty string.")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("`diagnostics` must be a mapping of numeric values.")
        diagnostics = {
            str(key): _finite_real(value, f"diagnostics[{key!r}]")
            for key, value in self.diagnostics.items()
        }
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))

    def to_context_dict(self) -> dict[str, Any]:
        return {
            "lambda_ax": self.lambda_ax,
            "lambda_rad": self.lambda_rad,
            "h_wall": self.h_wall,
            "wall_bed_htc": self.wall_bed_htc,
            "pressure_drop_model": self.pressure_drop_model,
            "dispersion_model": self.dispersion_model,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class ResolvedArchitecture:
    """Complete serializable result of a future architecture resolution."""

    architecture: str
    design: PackedBedDesign
    structure: ResolvedStructure
    transport: ResolvedTransport

    def __post_init__(self) -> None:
        validate_architecture_name(self.architecture)
        if self.architecture == "packed_bed" and not isinstance(
            self.design,
            PackedBedDesign,
        ):
            raise TypeError("Packed-bed resolution requires PackedBedDesign.")

    def to_context_dict(self) -> dict[str, Any]:
        design = self.design.to_config_dict()
        material = None
        return {
            "architecture": self.architecture,
            "design": design,
            "material": material,
            "resolved": {
                "structure": self.structure.to_context_dict(),
                "transport": self.transport.to_context_dict(),
            },
        }


# ---------------------------------------------------------------------------
# 5. Behavioral protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class PressureDropLaw(Protocol):
    """Behavioral pressure-drop contract used by the common column."""

    name: str

    def velocity_from_gradient(
        self,
        pressure_gradient: Any,
        local_density: Any,
    ) -> Any:
        ...


@runtime_checkable
class HeatTransferModel(Protocol):
    """Behavioral heat-transfer resolution contract."""

    name: str

    def resolve(self, context: TransportContext) -> ResolvedTransport:
        ...


@runtime_checkable
class DispersionModel(Protocol):
    """Behavioral axial/radial dispersion resolution contract."""

    name: str

    def resolve(self, context: TransportContext) -> Mapping[str, Any]:
        ...


@runtime_checkable
class BedArchitecture(Protocol):
    """Composition boundary for a selected bed architecture."""

    name: str
    design: PackedBedDesign

    def resolve_structure(self, context: StructureContext) -> ResolvedStructure:
        ...

    def resolve_transport(self, context: TransportContext) -> ResolvedTransport:
        ...


__all__ = [
    "ARCHITECTURE_NAMES",
    "DEFAULT_PACKED_BED_DESIGN",
    "DEFAULT_STAGNANT_CONDUCTIVITY_MODEL",
    "DEFAULT_WALL_HTC_MODEL",
    "STAGNANT_CONDUCTIVITY_MODELS",
    "WALL_HTC_MODELS",
    "validate_stagnant_conductivity_model",
    "validate_wall_htc_model",
    "validate_architecture_name",
    "validate_spatial_mode",
    "PackedBedDesign",
    "build_bed_config",
    "resolve_bed_config",
    "ColumnGeometry",
    "ParticleProperties",
    "GasProperties",
    "FlowProperties",
    "StructureContext",
    "ResolvedStructure",
    "TransportContext",
    "ResolvedTransport",
    "ResolvedArchitecture",
    "PressureDropLaw",
    "DispersionModel",
    "HeatTransferModel",
    "BedArchitecture",
]
