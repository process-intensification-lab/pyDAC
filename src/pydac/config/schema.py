"""Immutable public experiment sections driven by editable Python defaults.

The named classes below preserve readable runtime access and ``isinstance``
boundaries. Ordinary scalar sections derive their fields from the editable
defaults; the variable-size mixture and model declarations have small explicit
records because their names and lengths are user-defined.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping

from pydac.config.execution import ExecutionConfig
from pydac.config.sections import FrozenSection
from pydac.config.steps import STEP_NAMES, CycleStep, StepSequenceConfig, step_config_from_mapping


class GridConfig(FrozenSection):
    """Spatial and output time-grid controls."""

    PATH = ("simulation", "grid")


class SolverConfig(FrozenSection):
    """CVODE and integrated-scalar tolerances."""

    PATH = ("simulation", "solver")


class GeometryConfig(FrozenSection):
    """Public and deterministically derived column geometry."""

    PATH = ("simulation", "geometry")


class WallConfig(FrozenSection):
    """Column-wall properties and the optional equivalent HTC."""

    PATH = ("simulation", "wall")


class SorbentConfig(FrozenSection):
    """Sorbent and particle properties, with per-adsorbate heats of adsorption."""

    PATH = ("simulation", "sorbent")


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{label}` must be a non-empty string.")
    return value.strip()


def _species_name(value: Any, label: str) -> str:
    return _name(value, label).upper()


def _unique_species(values: Any, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"`{label}` must be a sequence of species names.")
    names = tuple(_species_name(value, label) for value in values)
    if not names:
        raise ValueError(f"`{label}` must not be empty.")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"`{label}` contains duplicate species {duplicates}.")
    return names


def _exact_mapping(values: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"`{label}` must be a mapping.")
    unknown = sorted(set(values) - expected)
    missing = sorted(expected - set(values))
    if unknown:
        raise TypeError(f"Unknown configuration field `{label}.{unknown[0]}`.")
    if missing:
        raise TypeError(f"Missing configuration field `{label}.{missing[0]}`.")
    return dict(values)


def _freeze_json_value(value: Any, label: str) -> Any:
    """Freeze a finite JSON value used as a registry parameter."""
    if isinstance(value, Mapping):
        items = {}
        for key, item in value.items():
            name = _name(key, f"{label} key")
            items[name] = _freeze_json_value(item, f"{label}.{name}")
        return MappingProxyType(items)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Real):
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"`{label}` must be finite.")
        return number
    raise TypeError(f"`{label}` must contain only JSON-compatible values.")


def _parameter_mapping(values: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"`{label}` must be a mapping.")
    return _freeze_json_value(values, label)


@dataclass(frozen=True, slots=True)
class MixtureConfig:
    """Named gas mixture, closure species, and ordered adsorbates."""

    species: tuple[str, ...]
    closure_species: str
    adsorbates: tuple[str, ...]

    def __post_init__(self) -> None:
        species = _unique_species(self.species, "simulation.mixture.species")
        closure = _species_name(
            self.closure_species, "simulation.mixture.closure_species"
        )
        adsorbates = _unique_species(
            self.adsorbates, "simulation.mixture.adsorbates"
        )
        if len(species) < 2:
            raise ValueError("A mixture needs at least two gas species.")
        if closure not in species:
            raise ValueError(
                f"Closure species {closure!r} is absent from mixture species {species}."
            )
        missing = [name for name in adsorbates if name not in species]
        if missing:
            raise ValueError(f"Adsorbates absent from the mixture: {missing}.")
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "closure_species", closure)
        object.__setattr__(self, "adsorbates", adsorbates)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], **_options) -> "MixtureConfig":
        data = _exact_mapping(
            values,
            {"species", "closure_species", "adsorbates"},
            "simulation.mixture",
        )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EquilibriumSystemConfig:
    """Datasets and unit-bearing parameters for one named equilibrium system."""

    datasets: Mapping[str, str]
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.datasets, Mapping):
            raise TypeError("Equilibrium `datasets` must be a mapping.")
        datasets = {
            _name(name, "dataset name"): _name(value, f"dataset {name}")
            for name, value in self.datasets.items()
        }
        object.__setattr__(self, "datasets", MappingProxyType(datasets))
        object.__setattr__(
            self,
            "parameters",
            _parameter_mapping(self.parameters, "equilibrium parameters"),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], label: str):
        return cls(
            **_exact_mapping(values, {"datasets", "parameters"}, label)
        )


@dataclass(frozen=True, slots=True)
class KineticModelConfig:
    """Registry model name and unit-bearing parameters for one adsorbate."""

    model: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _name(self.model, "kinetic model"))
        object.__setattr__(
            self,
            "parameters",
            _parameter_mapping(self.parameters, "kinetic parameters"),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], label: str):
        return cls(**_exact_mapping(values, {"model", "parameters"}, label))


@dataclass(frozen=True, slots=True)
class AdsorptionConfig:
    """Named equilibrium systems and one kinetic declaration per adsorbate."""

    equilibrium_systems: Mapping[str, EquilibriumSystemConfig]
    kinetics: Mapping[str, KineticModelConfig]

    def __post_init__(self) -> None:
        if not isinstance(self.equilibrium_systems, Mapping):
            raise TypeError("`simulation.adsorption.equilibrium_systems` must be a mapping.")
        if not isinstance(self.kinetics, Mapping):
            raise TypeError("`simulation.adsorption.kinetics` must be a mapping.")
        if not self.equilibrium_systems:
            raise ValueError("At least one equilibrium system must be selected.")
        if not self.kinetics:
            raise ValueError("At least one kinetic model must be selected.")
        equilibrium = {
            _name(name, "equilibrium model"): (
                value
                if isinstance(value, EquilibriumSystemConfig)
                else EquilibriumSystemConfig.from_mapping(
                    value, f"simulation.adsorption.equilibrium_systems.{name}"
                )
            )
            for name, value in self.equilibrium_systems.items()
        }
        names = [_species_name(species, "kinetic species") for species in self.kinetics]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(
                f"`simulation.adsorption.kinetics` names {twice[0]} twice; species names "
                "do not depend on case."
            )
        kinetics = {
            _species_name(species, "kinetic species"): (
                value
                if isinstance(value, KineticModelConfig)
                else KineticModelConfig.from_mapping(
                    value, f"simulation.adsorption.kinetics.{species}"
                )
            )
            for species, value in self.kinetics.items()
        }
        object.__setattr__(self, "equilibrium_systems", MappingProxyType(equilibrium))
        object.__setattr__(self, "kinetics", MappingProxyType(kinetics))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], **_options) -> "AdsorptionConfig":
        return cls(
            **_exact_mapping(
                values,
                {"equilibrium_systems", "kinetics"},
                "simulation.adsorption",
            )
        )

    @property
    def co_adsorption(self) -> bool:
        """Compatibility flag consumed by the physical core."""
        return tuple(self.equilibrium_systems) == ("TothGABWADST",)


class ReferenceConfig(FrozenSection):
    """Reference state used for nondimensionalization."""

    PATH = ("simulation", "reference")


def _resolve_canonical_bed_config(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Import the detached resolver without making ``config`` path-dependent."""
    from pydac.core.bed_architectures import resolve_bed_config
    return resolve_bed_config(values)


def default_bed_mapping() -> dict[str, Any]:
    """Return the framework default bed owned by ``bed_architectures``."""
    return dict(_resolve_canonical_bed_config(None))


class BedConfig(FrozenSection):
    """Immutable view over the canonical architecture-owned bed mapping."""

    PATH = ("simulation", "bed")

    def __init__(
        self,
        architecture: str,
        design: Mapping[str, Any],
        material: Mapping[str, Any] | None,
    ) -> None:
        canonical = dict(
            _resolve_canonical_bed_config(
                {
                    "architecture": architecture,
                    "design": design,
                    "material": material,
                }
            )
        )
        self._initialize(canonical, canonical, self.PATH)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        defaults: Mapping[str, Any] | None = None,
        path: tuple[str, ...] | None = None,
    ) -> "BedConfig":
        canonical = dict(_resolve_canonical_bed_config(values))
        instance = cls.__new__(cls)
        instance._initialize(canonical, canonical, cls.PATH)
        return instance

    def to_mapping(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "design": dict(self.design),
            "material": None if self.material is None else dict(self.material),
        }


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Complete physical and numerical input for one experiment."""

    grid: GridConfig
    solver: SolverConfig
    bed: BedConfig
    geometry: GeometryConfig
    wall: WallConfig
    sorbent: SorbentConfig
    mixture: MixtureConfig
    adsorption: AdsorptionConfig
    reference: ReferenceConfig
    cycle: StepSequenceConfig

    def __post_init__(self) -> None:
        expected = {
            "grid": GridConfig,
            "solver": SolverConfig,
            "bed": BedConfig,
            "geometry": GeometryConfig,
            "wall": WallConfig,
            "sorbent": SorbentConfig,
            "mixture": MixtureConfig,
            "adsorption": AdsorptionConfig,
            "reference": ReferenceConfig,
            "cycle": StepSequenceConfig,
        }
        for name, record_type in expected.items():
            if not isinstance(getattr(self, name), record_type):
                raise TypeError(f"`{name}` must be a {record_type.__name__} instance.")


SIMULATION_SECTIONS = tuple(
    item.name for item in fields(SimulationConfig) if item.name != "cycle"
)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Top-level public experiment document."""

    simulation: SimulationConfig
    execution: ExecutionConfig
    schema_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, SimulationConfig):
            raise TypeError("`simulation` must be a SimulationConfig instance.")
        if not isinstance(self.execution, ExecutionConfig):
            raise TypeError("`execution` must be an ExecutionConfig instance.")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError("`schema_version` must be an integer.")
        if self.schema_version < 1:
            raise ValueError("`schema_version` must be a positive integer.")


def experiment_config_from_mapping(
    document: Mapping[str, Any],
    vocabulary: Mapping[str, Any],
) -> ExperimentConfig:
    """Construct named runtime sections using one declared default tree."""
    simulation = document["simulation"]
    declared = vocabulary["simulation"]
    steps = tuple(
        CycleStep(
            name,
            step_config_from_mapping(
                values,
                location=name,
                defaults=declared["cycle"]["steps"][name],
            ),
        )
        for name, values in simulation["cycle"]["steps"].items()
    )
    if tuple(step.name for step in steps) != STEP_NAMES:
        raise ValueError(
            "The resolved cycle must contain the reference steps in canonical order."
        )
    section_types = {
        "grid": GridConfig,
        "solver": SolverConfig,
        "geometry": GeometryConfig,
        "wall": WallConfig,
        "sorbent": SorbentConfig,
        "mixture": MixtureConfig,
        "adsorption": AdsorptionConfig,
        "reference": ReferenceConfig,
    }
    sections = {
        name: record.from_mapping(simulation[name], defaults=declared[name])
        for name, record in section_types.items()
    }
    return ExperimentConfig(
        simulation=SimulationConfig(
            bed=BedConfig.from_mapping(simulation["bed"]),
            cycle=StepSequenceConfig(steps=steps),
            **sections,
        ),
        execution=ExecutionConfig.from_mapping(
            document["execution"], defaults=vocabulary["execution"]
        ),
        schema_version=document["schema_version"],
    )


__all__ = [
    "AdsorptionConfig",
    "BedConfig",
    "ExperimentConfig",
    "GeometryConfig",
    "GridConfig",
    "EquilibriumSystemConfig",
    "KineticModelConfig",
    "MixtureConfig",
    "ReferenceConfig",
    "SIMULATION_SECTIONS",
    "SimulationConfig",
    "SorbentConfig",
    "SolverConfig",
    "WallConfig",
    "default_bed_mapping",
    "experiment_config_from_mapping",
]
