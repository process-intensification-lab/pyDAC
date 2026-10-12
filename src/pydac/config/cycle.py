"""Typed configuration of one running TVSA campaign.

``config.schema`` states the contract of a whole experiment *document*; this
module states the runtime view of one resolved campaign: the values a cycle
hands to every step it builds. Dictionary/JSON input is resolved only by
``config.loader``; past that boundary the factory and the column model
exchange the typed records defined here.

Nothing physical is restated. A campaign **is** a resolved
:class:`config.schema.SimulationConfig` — the same record the JSON and the
terminal produce — so its knobs travel through
:func:`config.loader.load_experiment` as settings rather than through a second
set of records. The only thing added here is what the public document has no
place for: the three per-step sparsity switches a caller may flip step by step.

The runtime-only construction contracts (``ColumnModelInput``,
``AdsorptionModelBundle``) are *not* configuration and live next to the model,
in ``src/pydac/core/column_model_input.py``.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : after resolution — the bridge from the DOCUMENT to the PHYSICS.
#              ExperimentConfig --from_experiment--> CycleConfig
#              CycleConfig --step_config("ads")--> ColumnModelConfig
#              ColumnModelConfig --factory--> ColumnModelInput --> the model
# Owns     : the runtime view of one campaign, and nothing physical.
# Not here : resolution of any kind — only `config.loader` resolves input.
# Entry    : CycleConfig.from_experiment()  ·  CycleConfig.step_config()
# Guide    : docs/manual/01_inputs.md §2.2
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from pydac.config.schema import (
    AdsorptionConfig,
    ExperimentConfig,
    GeometryConfig,
    GridConfig,
    MixtureConfig,
    ReferenceConfig,
    SimulationConfig,
    SolverConfig,
    SorbentConfig,
    WallConfig,
)
from pydac.config.steps import StepConfig, StepSequenceConfig
from pydac.config.validation import optional_instance


def _normalize_cvode_sparsity_mode(value: Any) -> str:
    """Normalize the solver option without importing ``utils`` at module load."""
    from pydac.utils.solver import normalize_cvode_sparsity_mode

    return normalize_cvode_sparsity_mode(value)


@dataclass(frozen=True, slots=True)
class SolverOptions:
    """Sparsity controls, separate from the numerical tolerances.

    Tolerances and the grid belong to :class:`config.schema.SolverConfig` and
    :class:`config.schema.GridConfig`; this record carries only the three
    per-step sparsity switches, which a caller may override step by step.
    """

    use_sparsity: bool
    compute_numerical_sparsity: bool
    cvode_sparsity_mode: str

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        use_sparsity: bool | None = None,
        compute_numerical_sparsity: bool | None = None,
    ) -> "SolverOptions":
        selected_use_sparsity = (
            values.get("use_sparsity", True)
            if use_sparsity is None
            else use_sparsity
        )
        selected_compute = (
            values.get("compute_numerical_sparsity", False)
            if compute_numerical_sparsity is None
            else compute_numerical_sparsity
        )
        if not isinstance(selected_use_sparsity, bool):
            raise TypeError("`use_sparsity` must be bool.")
        if not isinstance(selected_compute, bool):
            raise TypeError("`compute_numerical_sparsity` must be bool.")
        return cls(
            use_sparsity=selected_use_sparsity,
            compute_numerical_sparsity=selected_compute,
            cvode_sparsity_mode=_normalize_cvode_sparsity_mode(
                values.get("cvode_sparsity_mode")
            ),
        )

    def with_overrides(
        self,
        *,
        use_sparsity: bool | None = None,
        compute_numerical_sparsity: bool | None = None,
    ) -> "SolverOptions":
        values = {
            "use_sparsity": self.use_sparsity if use_sparsity is None else use_sparsity,
            "compute_numerical_sparsity": (
                self.compute_numerical_sparsity
                if compute_numerical_sparsity is None
                else compute_numerical_sparsity
            ),
        }
        if not all(isinstance(value, bool) for value in values.values()):
            raise TypeError("Solver overrides must be bool when provided.")
        return replace(self, **values)


@dataclass(frozen=True, slots=True)
class CycleConfig:
    """Complete typed configuration owned by one TVSA campaign."""

    simulation: SimulationConfig
    solver: SolverOptions
    experiment: ExperimentConfig | None = None

    @classmethod
    def from_experiment(
        cls,
        experiment_config: ExperimentConfig,
        *,
        grid_config: GridConfig | None = None,
        solver_config: SolverConfig | None = None,
    ) -> "CycleConfig":
        """The runtime view of a resolved document.

        The sparsity switches come from the document's execution section, so a
        campaign that wants to change them states a setting rather than
        passing a second copy through this boundary.
        """
        if not isinstance(experiment_config, ExperimentConfig):
            raise TypeError("`experiment_config` must be an ExperimentConfig instance.")
        optional_instance(grid_config, "grid_config", expected=GridConfig)
        optional_instance(solver_config, "solver_config", expected=SolverConfig)
        execution = experiment_config.execution
        solver_values = {
            "use_sparsity": execution.use_sparsity,
            "compute_numerical_sparsity": execution.compute_numerical_sparsity,
            "cvode_sparsity_mode": execution.cvode_sparsity_mode,
        }
        simulation = experiment_config.simulation
        explicit = {
            name: value
            for name, value in (("grid", grid_config), ("solver", solver_config))
            if value is not None
        }
        return cls(
            simulation=replace(simulation, **explicit) if explicit else simulation,
            solver=SolverOptions.from_mapping(solver_values),
            experiment=replace(experiment_config, simulation=replace(simulation, **explicit)),
        )

    @property
    def grid(self) -> GridConfig:
        return self.simulation.grid

    @property
    def solver_parameters(self) -> SolverConfig:
        return self.simulation.solver

    @property
    def bed(self) -> dict[str, Any]:
        """The canonical bed mapping the architecture resolvers consume."""
        return self.simulation.bed.to_mapping()

    @property
    def steps(self) -> StepSequenceConfig:
        return self.simulation.cycle

    def _step_settings(self, step: str):
        """Return one step's resolved settings, by canonical or alias name."""
        if not isinstance(step, str) or not step:
            raise ValueError("`step` must be a non-empty string.")
        from pydac.utils.tvsa import normalize_step_token

        step = normalize_step_token(step)
        settings = {item.name: item.config for item in self.steps.steps}.get(step)
        if settings is None:
            raise ValueError(f"Unsupported cycle step '{step}'.")
        return step, settings

    def step_config(
        self,
        step: str,
        *,
        solver_options: SolverOptions | None = None,
    ) -> "ColumnModelConfig":
        step, settings = self._step_settings(step)
        return ColumnModelConfig(
            step=step,
            step_settings=settings,
            bed=self.bed,
            geometry=self.simulation.geometry,
            wall=self.simulation.wall,
            sorbent=self.simulation.sorbent,
            mixture=self.simulation.mixture,
            reference=self.simulation.reference,
            grid=self.simulation.grid,
            solver_parameters=self.simulation.solver,
            adsorption=self.simulation.adsorption,
            solver=self.solver if solver_options is None else solver_options,
        )

    def solver_for_step(
        self,
        *,
        use_sparsity: bool | None = None,
        compute_numerical_sparsity: bool | None = None,
    ) -> SolverOptions:
        return self.solver.with_overrides(
            use_sparsity=use_sparsity,
            compute_numerical_sparsity=compute_numerical_sparsity,
        )


@dataclass(frozen=True, slots=True)
class ColumnModelConfig:
    """Resolved configuration for one concrete cycle step.

    Every public simulation input the model needs is here, already resolved:
    the model reads this record and nothing else.
    """

    step: str
    step_settings: StepConfig
    bed: Mapping[str, Any]
    geometry: GeometryConfig
    wall: WallConfig
    sorbent: SorbentConfig
    mixture: MixtureConfig
    reference: ReferenceConfig
    grid: GridConfig
    solver_parameters: SolverConfig
    adsorption: AdsorptionConfig
    solver: SolverOptions


__all__ = [
    "ColumnModelConfig",
    "CycleConfig",
    "SolverOptions",
]
