"""Public typed configuration API for pyDAC experiments.

A JSON file plus ``--set`` from the terminal and an editable runner mapping
travel through the same loader and produce the same immutable records:

    load_experiment(source, settings=...) -> ExperimentConfig

``config.cli`` is deliberately not imported here so that importing the public
API never drags in the terminal entry point.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# This file is a PURE FACADE: imports and `__all__`, never logic. Read it as
# the index of the package. Where to go from here:
#   what a field means  -> schema.py, steps.py, execution.py
#   how input resolves  -> loader.py  (the pipeline)  ·  settings.py, nulls.py
#   how a run is built  -> cycle.py, launch.py  ·  terminal.py, cli.py
#   how a campaign maps -> operating.py
# Guide: docs/manual/01_inputs.md
# ───────────────────────────────────────────────────────────────────────────

from pydac.config.errors import ConfigurationError, SettingsError
from pydac.config.execution import ExecutionConfig
from pydac.config.operating import OPERATING_SETTING_KEYS, operating_step_settings
from pydac.config.loader import (
    apply_settings,
    default_experiment_config,
    load_experiment,
    merge_documents,
    write_resolved_config,
)
from pydac.config.nulls import NULL_POLICY_BY_PATH, null_policy
from pydac.config.launch import resolve_experiment_config
from pydac.config.profile import reference_profile
from pydac.config.schema import (
    SIMULATION_SECTIONS,
    AdsorptionConfig,
    BedConfig,
    ExperimentConfig,
    EquilibriumSystemConfig,
    GeometryConfig,
    GridConfig,
    KineticModelConfig,
    MixtureConfig,
    ReferenceConfig,
    SimulationConfig,
    SorbentConfig,
    SolverConfig,
    WallConfig,
)
from pydac.config.serialization import experiment_to_document, to_jsonable
from pydac.config.sections import FrozenSection
from pydac.config.settings import resolve_settings
from pydac.config.steps import (
    NULLABLE_STEP_FIELDS,
    STEP_FIELD_ORDER,
    STEP_NAMES,
    CompositionConfig,
    CycleStep,
    FlowConfig,
    HorizonConfig,
    OutletConfig,
    PressureConfig,
    StepConfig,
    StepSequenceConfig,
    TerminationConfig,
    ThermalConfig,
)

__all__ = [
    "AdsorptionConfig",
    "BedConfig",
    "CompositionConfig",
    "ConfigurationError",
    "CycleStep",
    "ExecutionConfig",
    "ExperimentConfig",
    "EquilibriumSystemConfig",
    "FlowConfig",
    "FrozenSection",
    "GeometryConfig",
    "GridConfig",
    "KineticModelConfig",
    "MixtureConfig",
    "HorizonConfig",
    "NULLABLE_STEP_FIELDS",
    "NULL_POLICY_BY_PATH",
    "OPERATING_SETTING_KEYS",
    "OutletConfig",
    "PressureConfig",
    "ReferenceConfig",
    "SIMULATION_SECTIONS",
    "STEP_FIELD_ORDER",
    "STEP_NAMES",
    "SettingsError",
    "SimulationConfig",
    "SorbentConfig",
    "SolverConfig",
    "StepConfig",
    "StepSequenceConfig",
    "TerminationConfig",
    "ThermalConfig",
    "WallConfig",
    "apply_settings",
    "default_experiment_config",
    "experiment_to_document",
    "load_experiment",
    "merge_documents",
    "null_policy",
    "operating_step_settings",
    "reference_profile",
    "resolve_experiment_config",
    "resolve_settings",
    "to_jsonable",
    "write_resolved_config",
]
