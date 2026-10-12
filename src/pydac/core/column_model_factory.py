"""Composition root for typed column-model construction."""

from __future__ import annotations

from pydac.core.column_model import ColumnModelMoL
from pydac.core.column_model_input import AdsorptionModelBundle, ColumnModelInput
from pydac.config.cycle import CycleConfig, SolverOptions
from pydac.core.isotherm_models import equilibrium_registry
from pydac.core.kinetic_models import kinetic_registry
from pydac.utils.model_registry import (
    EquilibriumChoice,
    KineticChoice,
    build_equilibrium_system,
    build_kinetic_laws,
)


class ColumnModelFactory:
    """Create step-specific column models from one typed campaign config."""

    def __init__(self, config: CycleConfig):
        # Keep the cycle config that every step model will be built from.
        if not isinstance(config, CycleConfig):
            raise TypeError("`config` must be a CycleConfig instance.")
        self.config = config
        self._components = None

        print("\n" + "-" * 60)
        print("        Thermodynamic + Kinetic Models Setup         ".center(60))
        print("-" * 60)

    def _build_model_components(self) -> AdsorptionModelBundle:
        """Build the declared named mixture, equilibrium, and kinetic models."""
        mixture = self.config.simulation.mixture
        adsorption = self.config.simulation.adsorption
        equilibrium_choices = (
            EquilibriumChoice(
                name=name,
                datasets=selection.datasets,
                parameters=selection.parameters,
            )
            for name, selection in adsorption.equilibrium_systems.items()
        )
        equilibrium_system = build_equilibrium_system(
            mixture_species=mixture.species,
            adsorbates=mixture.adsorbates,
            choices=equilibrium_choices,
            registry=equilibrium_registry(),
        )

        kinetic_choices = (
            KineticChoice(
                species=species,
                name=selection.model,
                parameters=selection.parameters,
            )
            for species, selection in adsorption.kinetics.items()
        )

        kinetics = build_kinetic_laws(
            adsorbates=mixture.adsorbates,
            choices=kinetic_choices,
            equilibrium_system=equilibrium_system,
            registry=kinetic_registry(),
        )
        return AdsorptionModelBundle(
            equilibrium_system=equilibrium_system,
            kinetics=kinetics,
        )

    @property
    def components(self) -> AdsorptionModelBundle:
        """Return the cycle-shared runtime model components."""
        # Build components only once; every step reuses the same models.
        if self._components is None:
            self._components = self._build_model_components()
        return self._components

    def build_column_model_input(
        self,
        *,
        step: str,
        initial_state=None,
        solver_options: SolverOptions | None = None,
    ) -> ColumnModelInput:
        """Resolve one step and assemble its complete construction request."""
        # Convert one step name into the input object ColumnModelMoL expects.
        return ColumnModelInput(
            config=self.config.step_config(
                step,
                solver_options=solver_options,
            ),
            components=self.components,
            initial_state=initial_state,
        )

    def build_column_model(
        self,
        *,
        step: str,
        initial_state=None,
        solver_options: SolverOptions | None = None,
    ) -> ColumnModelMoL:
        """Construct one fully configured column model."""
        # Create the actual physical model for one TVSA step.
        model = ColumnModelMoL(
            self.build_column_model_input(
                step=step,
                initial_state=initial_state,
                solver_options=solver_options,
            )
        )
        # Retain the originating public document for execution provenance only.
        model.provenance_config = self.config.experiment
        return model


__all__ = ["ColumnModelFactory"]
