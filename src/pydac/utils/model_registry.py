"""Reusable construction helpers for named equilibrium and kinetic models.

Physical equations and the explicit supported-model registries stay in
``src/pydac/core/isotherm_models.py`` and ``src/pydac/core/kinetic_models.py``.  This module contains
only the generic bookkeeping needed to validate selections and combine them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class EquilibriumChoice:
    """One selected equilibrium name, datasets, and constructor parameters."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    datasets: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "model name"))
        object.__setattr__(
            self, "datasets", MappingProxyType(dict(self.datasets))
        )
        object.__setattr__(
            self, "parameters", MappingProxyType(dict(self.parameters))
        )


@dataclass(frozen=True, slots=True)
class KineticChoice:
    """One adsorbate, selected kinetic name, and parameter set."""

    species: str
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", _species(self.species))
        object.__setattr__(self, "name", _text(self.name, "model name"))
        object.__setattr__(
            self, "parameters", MappingProxyType(dict(self.parameters))
        )


@dataclass(frozen=True, slots=True)
class EquilibriumRegistryEntry:
    """Compatibility metadata and constructor stored in an equilibrium registry."""

    adsorbates: tuple[str, ...]
    required_gas_species: tuple[str, ...]
    required_current_loadings: tuple[str, ...]
    build: Callable[[Mapping[str, str], Mapping[str, Any]], Any]
    required_datasets: tuple[str, ...] = ()
    optional_datasets: tuple[str, ...] = ()
    required_parameters: tuple[str, ...] = ()
    optional_parameters: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class KineticRegistryEntry:
    """Compatibility metadata and constructor stored in a kinetic registry."""

    build: Callable[[str, Mapping[str, Any]], Any]
    supported_species: tuple[str, ...] | None = None
    required_parameters: tuple[str, ...] = ()
    optional_parameters: tuple[str, ...] = ()
    positive_parameters: tuple[str, ...] = ()


class CompositeEquilibriumSystem:
    """Present several selected systems through one name-based call."""

    def __init__(self, systems: Iterable[Any], model_names: Iterable[str]) -> None:
        self.systems = tuple(systems)
        self.model_names = tuple(model_names)
        if not self.systems or len(self.systems) != len(self.model_names):
            raise ValueError("Each selected equilibrium model needs one system.")

        self.species = tuple(
            species for system in self.systems for species in system.species
        )
        duplicates = _duplicates(self.species)
        if duplicates:
            raise ValueError(
                f"Equilibrium systems cover adsorbates more than once: {duplicates}."
            )

        component_pairs = [
            (species, model)
            for system in self.systems
            for species, model in getattr(system, "component_models", {}).items()
        ]
        duplicate_models = _duplicates(species for species, _ in component_pairs)
        if duplicate_models:
            raise ValueError(
                f"Equilibrium component models are duplicated: {duplicate_models}."
            )
        self.component_models = MappingProxyType(dict(component_pairs))

    def equilibrium_loadings(
        self,
        *,
        partial_pressures_Pa: Mapping[str, Any],
        temperature_K: Any,
        current_loadings_mol_per_kg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(partial_pressures_Pa, Mapping):
            raise TypeError("Partial pressures must be a species-keyed mapping.")
        if not isinstance(current_loadings_mol_per_kg, Mapping):
            raise TypeError("Current loadings must be a species-keyed mapping.")

        loadings: dict[str, Any] = {}
        for system in self.systems:
            result = dict(
                system.equilibrium_loadings(
                    partial_pressures_Pa=partial_pressures_Pa,
                    temperature_K=temperature_K,
                    current_loadings_mol_per_kg=current_loadings_mol_per_kg,
                )
            )
            if tuple(result) != tuple(system.species):
                raise ValueError(
                    f"{type(system).__name__} returned {tuple(result)}; "
                    f"expected {tuple(system.species)}."
                )
            loadings.update(result)
        return loadings


def build_equilibrium_system(
    *,
    mixture_species: Iterable[str],
    adsorbates: Iterable[str],
    choices: Iterable[EquilibriumChoice],
    registry: Mapping[str, EquilibriumRegistryEntry],
) -> CompositeEquilibriumSystem:
    """Validate physical coverage and build the selected equilibrium systems."""
    _mixture, _expected, selected, entries = _equilibrium_plan(
        mixture_species=mixture_species,
        adsorbates=adsorbates,
        choices=choices,
        registry=registry,
    )

    systems = []
    for choice, entry in zip(selected, entries, strict=True):
        system = entry.build(choice.datasets, choice.parameters)
        if tuple(getattr(system, "species", ())) != entry.adsorbates:
            raise ValueError(
                f"Registry {choice.name!r} declares {entry.adsorbates}, but "
                f"its system owns {getattr(system, 'species', ())}."
            )
        if not callable(getattr(system, "equilibrium_loadings", None)):
            raise TypeError(
                f"Equilibrium model {choice.name!r} has no equilibrium_loadings call."
            )
        systems.append(system)

    return CompositeEquilibriumSystem(
        systems, model_names=(choice.name for choice in selected)
    )


def build_kinetic_laws(
    *,
    adsorbates: Iterable[str],
    choices: Iterable[KineticChoice],
    equilibrium_system: Any,
    registry: Mapping[str, KineticRegistryEntry],
) -> Mapping[str, Any]:
    """Validate physical coverage and build one kinetic law per adsorbate."""
    expected, selected, entries = _kinetic_plan(
        adsorbates=adsorbates,
        choices=choices,
        registry=registry,
    )
    if tuple(equilibrium_system.species) != expected:
        raise ValueError(
            "Equilibrium and kinetic adsorbates differ "
            f"(equilibrium={tuple(equilibrium_system.species)}, kinetics={expected})."
        )

    laws: dict[str, Any] = {}
    for choice, entry in zip(selected, entries, strict=True):
        law = entry.build(choice.species, choice.parameters)
        if tuple(getattr(law, "species", ())) != (choice.species,):
            raise ValueError(
                f"Kinetic model {choice.name!r} was selected for "
                f"{choice.species!r}, but its law owns "
                f"{getattr(law, 'species', ())}."
            )
        if not callable(getattr(law, "loading_rates_mol_per_kg_s", None)):
            raise TypeError(
                f"Kinetic model {choice.name!r} has no physical-rate call."
            )

        laws[choice.species] = law

    return MappingProxyType(laws)


def validate_equilibrium_choices(
    *,
    mixture_species: Iterable[str],
    adsorbates: Iterable[str],
    choices: Iterable[EquilibriumChoice],
    registry: Mapping[str, EquilibriumRegistryEntry],
) -> None:
    """Validate named equilibrium declarations without constructing models."""
    _equilibrium_plan(
        mixture_species=mixture_species,
        adsorbates=adsorbates,
        choices=choices,
        registry=registry,
    )


def validate_kinetic_choices(
    *,
    adsorbates: Iterable[str],
    choices: Iterable[KineticChoice],
    registry: Mapping[str, KineticRegistryEntry],
) -> None:
    """Validate named kinetic declarations without constructing models."""
    _kinetic_plan(adsorbates=adsorbates, choices=choices, registry=registry)


def _equilibrium_plan(*, mixture_species, adsorbates, choices, registry):
    mixture = _species_names(mixture_species, "mixture species")
    expected = _species_names(adsorbates, "adsorbates")
    missing_adsorbates = set(expected) - set(mixture)
    if missing_adsorbates:
        raise ValueError(
            f"Adsorbates are absent from the mixture: {sorted(missing_adsorbates)}."
        )

    selected = tuple(choices)
    if not selected:
        raise ValueError("At least one equilibrium model must be selected.")
    entries = tuple(_registry_entry(choice.name, registry) for choice in selected)
    covered = tuple(species for entry in entries for species in entry.adsorbates)
    _validate_coverage(expected, covered, "Equilibrium models")

    mixture_set = set(mixture)
    adsorbate_set = set(expected)
    for choice, entry in zip(selected, entries, strict=True):
        missing_gas = set(entry.required_gas_species) - mixture_set
        if missing_gas:
            raise ValueError(
                f"Equilibrium model {choice.name!r} requires gas species "
                f"{sorted(missing_gas)} not present in the mixture."
            )
        missing_loadings = set(entry.required_current_loadings) - adsorbate_set
        if missing_loadings:
            raise ValueError(
                f"Equilibrium model {choice.name!r} requires current loadings "
                f"for undeclared adsorbates {sorted(missing_loadings)}."
            )
        _validate_named_inputs(
            choice.name,
            "datasets",
            choice.datasets,
            entry.required_datasets,
            entry.optional_datasets,
        )
        _validate_named_inputs(
            choice.name,
            "parameters",
            choice.parameters,
            entry.required_parameters,
            entry.optional_parameters,
        )
    return mixture, expected, selected, entries


def _kinetic_plan(*, adsorbates, choices, registry):
    expected = _species_names(adsorbates, "adsorbates")
    selected = tuple(choices)
    covered = tuple(choice.species for choice in selected)
    _validate_coverage(expected, covered, "Kinetic models")
    entries = tuple(_registry_entry(choice.name, registry) for choice in selected)
    for choice, entry in zip(selected, entries, strict=True):
        if (
            entry.supported_species is not None
            and choice.species not in entry.supported_species
        ):
            raise ValueError(
                f"Kinetic model {choice.name!r} does not support "
                f"{choice.species!r}; supported species are "
                f"{entry.supported_species}."
            )
        _validate_named_inputs(
            choice.name,
            "parameters",
            choice.parameters,
            entry.required_parameters,
            entry.optional_parameters,
        )
        for parameter in entry.positive_parameters:
            value = choice.parameters.get(parameter)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"Kinetic model {choice.name!r} parameter {parameter!r} "
                    "must be a finite number greater than zero."
                )
    return expected, selected, entries


def _validate_named_inputs(name, label, values, required, optional) -> None:
    received = set(values)
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - received)
    unknown = sorted(received - allowed)
    if missing or unknown:
        raise ValueError(
            f"Model {name!r} has invalid {label} "
            f"(missing={missing}, unknown={unknown}, allowed={sorted(allowed)})."
        )


def _registry_entry(name: str, registry: Mapping[str, Any]) -> Any:
    try:
        return registry[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {name!r}. Available models: {', '.join(registry)}."
        ) from exc


def _validate_coverage(
    expected: tuple[str, ...], covered: tuple[str, ...], owner: str
) -> None:
    counts = Counter(covered)
    duplicate = sorted(name for name, count in counts.items() if count > 1)
    missing = sorted(set(expected) - set(covered))
    extra = sorted(set(covered) - set(expected))
    wrong_order = not (duplicate or missing or extra) and covered != expected
    if duplicate or missing or extra or wrong_order:
        raise ValueError(
            f"{owner} must cover every adsorbate exactly once in declared order "
            f"(missing={missing}, duplicate={duplicate}, extra={extra}, "
            f"expected={expected}, covered={covered})."
        )


def _species_names(values: Iterable[str], label: str) -> tuple[str, ...]:
    names = tuple(_species(value) for value in values)
    if not names:
        raise ValueError(f"{label.capitalize()} must not be empty.")
    duplicates = _duplicates(names)
    if duplicates:
        raise ValueError(f"{label.capitalize()} contain duplicates: {duplicates}.")
    return names


def _species(value: str) -> str:
    return _text(value, "species name").upper()


def _text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label.capitalize()} must be a non-empty string.")
    return value.strip()


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(name for name, count in counts.items() if count > 1)


__all__ = [
    "CompositeEquilibriumSystem",
    "EquilibriumChoice",
    "EquilibriumRegistryEntry",
    "KineticChoice",
    "KineticRegistryEntry",
    "build_equilibrium_system",
    "build_kinetic_laws",
    "validate_equilibrium_choices",
    "validate_kinetic_choices",
]
