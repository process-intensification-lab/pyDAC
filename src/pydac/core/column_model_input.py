"""Runtime construction contracts exchanged with ``ColumnModelMoL``.

These two records are not configuration: they carry *live objects* — the
isotherm/kinetic instances the factory built and the previous step's field
arrays — so they stay next to the model rather than in ``config``. The typed
configuration they reference is owned by :mod:`config.cycle`.

The bundle exposes only small name lookups; all construction and physical
behavior remain in the registries and models.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from pydac.config.cycle import ColumnModelConfig


@dataclass(frozen=True, slots=True)
class AdsorptionModelBundle:
    """Named runtime equilibrium and kinetic dependencies shared by all steps."""

    equilibrium_system: Any
    kinetics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not callable(
            getattr(self.equilibrium_system, "equilibrium_loadings", None)
        ):
            raise TypeError("`equilibrium_system` needs an equilibrium_loadings call.")
        if not isinstance(self.kinetics, Mapping):
            raise TypeError("`kinetics` must be a species-keyed mapping.")
        kinetics = dict(self.kinetics)
        if set(kinetics) != set(self.equilibrium_system.species):
            raise ValueError(
                "Equilibrium and kinetic species differ "
                f"(equilibrium={self.equilibrium_system.species}, "
                f"kinetics={tuple(kinetics)})."
            )
        if any(
            not callable(getattr(law, "loading_rates_mol_per_kg_s", None))
            for law in kinetics.values()
        ):
            raise TypeError("Every kinetic model needs a physical-rate call.")
        object.__setattr__(self, "kinetics", MappingProxyType(kinetics))

    def equilibrium_model(self, species: str) -> Any:
        """Return one underlying correlation by physical species name."""
        normalized = _species_name(species)
        try:
            return self.equilibrium_system.component_models[normalized]
        except KeyError as exc:
            raise ValueError(
                f"No equilibrium component model exists for {normalized!r}."
            ) from exc

    def kinetic_law(self, species: str) -> Any:
        """Return one kinetic law by physical species name."""
        normalized = _species_name(species)
        try:
            return self.kinetics[normalized]
        except KeyError as exc:
            raise ValueError(f"No kinetic law exists for {normalized!r}.") from exc

@dataclass(frozen=True, slots=True)
class ColumnModelInput:
    """The only construction request accepted by ``ColumnModelMoL``."""

    config: ColumnModelConfig
    components: AdsorptionModelBundle
    initial_state: Mapping[str, Any] | None = None


def _species_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Species name must be a non-empty string.")
    return value.strip().upper()


__all__ = ["AdsorptionModelBundle", "ColumnModelInput"]
