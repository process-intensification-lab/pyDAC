"""Default-driven immutable configuration for the named TVSA cycle steps."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from pydac.inputs.defaults.profile import default_profile, default_rules
from pydac.inputs.defaults.steps import INITIAL_SOLID_LOADINGS_FIELD, cycle_defaults

from pydac.config.errors import ConfigurationError
from pydac.config.sections import FrozenSection


_DEFAULT_STEPS = cycle_defaults()["steps"]
STEP_NAMES = tuple(_DEFAULT_STEPS)
SUPPORTED_STEP_NAMES = frozenset(STEP_NAMES)
STEP_FIELD_ORDER = tuple(next(iter(_DEFAULT_STEPS.values())))


class CompositionConfig(FrozenSection):
    PATH = ("simulation", "cycle", "steps", STEP_NAMES[0], "composition")


class FlowConfig(FrozenSection):
    PATH = ("simulation", "cycle", "steps", STEP_NAMES[0], "flow")


class ThermalConfig(FrozenSection):
    PATH = ("simulation", "cycle", "steps", STEP_NAMES[0], "thermal")


class PressureConfig(FrozenSection):
    PATH = ("simulation", "cycle", "steps", STEP_NAMES[0], "pressure")


class HorizonConfig(FrozenSection):
    PATH = ("simulation", "cycle", "steps", STEP_NAMES[0], "horizon")


class TerminationConfig(FrozenSection):
    PATH = ("simulation", "cycle", "steps", STEP_NAMES[0], "termination")


class OutletConfig(FrozenSection):
    PATH = ("simulation", "cycle", "steps", STEP_NAMES[0], "outlet")


# The only step that builds its initial state from the case instead of the previous step.
STARTUP_STEP = "startup"

# Steps whose outlet cannot reach the product line: startup and adsorption discharge to
# the atmosphere, cooldown and pressurization run with a closed outlet.
NON_COLLECTING_STEPS = frozenset({"startup", "ads", "cool", "press"})

# Termination fields the solver reads on some steps only. Set on another step they
# would be accepted and never used, so they are refused there.
STEP_ONLY_TERMINATION_FIELDS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "stop_on_low_pressure": frozenset({"cool"}),
        "low_pressure_stop_Pa": frozenset({"cool"}),
        "stop_on_pressure_plateau": frozenset({"blow", "press"}),
        "pressure_plateau_rate_tolerance_nd": frozenset({"blow", "press"}),
        "pressure_plateau_arm_below_Pa": frozenset({"blow"}),
        "pressure_plateau_arm_above_Pa": frozenset({"press"}),
    }
)


STEP_RECORD_BY_FIELD: Mapping[str, type[FrozenSection]] = MappingProxyType(
    {
        "composition": CompositionConfig,
        "flow": FlowConfig,
        "thermal": ThermalConfig,
        "pressure": PressureConfig,
        "horizon": HorizonConfig,
        "termination": TerminationConfig,
        "outlet": OutletConfig,
    }
)


@dataclass(frozen=True, slots=True)
class StepConfig:
    """One resolved step with the fixed domain-group boundaries."""

    composition: CompositionConfig
    flow: FlowConfig
    thermal: ThermalConfig
    pressure: PressureConfig
    horizon: HorizonConfig
    termination: TerminationConfig
    outlet: OutletConfig
    FIELD_ORDER: ClassVar[tuple[str, ...]] = STEP_FIELD_ORDER

    def __post_init__(self) -> None:
        for name, record_type in STEP_RECORD_BY_FIELD.items():
            if not isinstance(getattr(self, name), record_type):
                raise TypeError(f"`{name}` must be a {record_type.__name__} instance.")
        if (
            self.horizon.event_max_time_s is not None
            and self.horizon.event_max_time_s < self.horizon.nominal_time_s
        ):
            raise ValueError("`event_max_time_s` cannot be shorter than nominal time.")
        termination = self.termination
        if (termination.threshold_value is None) == (termination.threshold_source is None):
            raise ValueError(
                "Set exactly one of `threshold_value` and `threshold_source`; the "
                "other must be null."
            )
        if termination.stop_on_low_pressure and termination.low_pressure_stop_Pa is None:
            raise ValueError(
                "`low_pressure_stop_Pa` is required when low-pressure stopping is enabled."
            )
        if termination.stop_on_pressure_plateau and (
            termination.pressure_plateau_rate_tolerance_nd is None
            or (
                termination.pressure_plateau_arm_below_Pa is None
                and termination.pressure_plateau_arm_above_Pa is None
            )
        ):
            raise ValueError(
                "Pressure-plateau stopping requires both its rate tolerance and "
                "arming pressure."
            )


@dataclass(frozen=True, slots=True)
class CycleStep:
    """A named step in a cycle plan."""

    name: str
    config: StepConfig

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("`name` must be a non-empty string.")
        object.__setattr__(self, "name", self.name.strip().lower())
        if self.name not in SUPPORTED_STEP_NAMES:
            raise ValueError(f"Unsupported cycle step {self.name!r}.")
        if not isinstance(self.config, StepConfig):
            raise TypeError("`config` must be a StepConfig instance.")


@dataclass(frozen=True, slots=True)
class StepSequenceConfig:
    """Immutable ordered collection of named step configurations."""

    steps: tuple[CycleStep, ...]

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        if not steps or not all(isinstance(step, CycleStep) for step in steps):
            raise TypeError("`steps` must contain CycleStep instances.")
        names = tuple(step.name for step in steps)
        if len(set(names)) != len(names):
            raise ValueError("Cycle step names must be unique.")
        object.__setattr__(self, "steps", steps)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(step.name for step in self.steps)


def _nullable_field_names(group: str) -> frozenset[str]:
    prefix = ("simulation", "cycle", "steps", STEP_NAMES[0], group)
    rules = default_rules()
    return frozenset(
        name
        for name in _DEFAULT_STEPS[STEP_NAMES[0]][group]
        if rules.get(prefix + (name,)) is not None
        and rules[prefix + (name,)].nullable
    )


NULLABLE_STEP_FIELDS: Mapping[str, frozenset[str]] = MappingProxyType(
    {group: _nullable_field_names(group) for group in STEP_FIELD_ORDER}
)


def step_config_from_mapping(
    values: Mapping[str, Any],
    *,
    location: str,
    defaults: Mapping[str, Any] | None = None,
) -> StepConfig:
    """Build one step using group vocabularies from its default branch."""
    declared = _DEFAULT_STEPS[location] if defaults is None else defaults
    missing = sorted(set(declared) - set(values))
    unknown = sorted(set(values) - set(declared))
    if missing or unknown:
        issue = f"missing {missing}" if missing else f"unknown {unknown}"
        raise ConfigurationError(f"Invalid configuration for step `{location}`: {issue}.")
    try:
        groups = {}
        for group in STEP_FIELD_ORDER:
            path = ("simulation", "cycle", "steps", location, group)
            groups[group] = STEP_RECORD_BY_FIELD[group].from_mapping(
                values[group], defaults=declared[group], path=path
            )
        for name, steps in STEP_ONLY_TERMINATION_FIELDS.items():
            if location not in steps and getattr(groups["termination"], name) not in (None, False):
                raise ValueError(
                    f"`termination.{name}` acts only on {', '.join(sorted(steps))}; "
                    "on this step it would be ignored: set it to null (or false)."
                )
        if (
            location != STARTUP_STEP
            and getattr(groups["composition"], INITIAL_SOLID_LOADINGS_FIELD) is not None
        ):
            raise ValueError(
                f"`composition.{INITIAL_SOLID_LOADINGS_FIELD}` acts only on "
                f"{STARTUP_STEP}: the other steps start from the previous step's solid; "
                "set it to null."
            )
        config = StepConfig(**groups)
        outlet = config.outlet
        if outlet.collect_product and location in NON_COLLECTING_STEPS:
            raise ValueError(
                "`outlet.collect_product` must be false: this step's outlet goes to "
                "the atmosphere or is closed."
            )
        if outlet.condenser and not outlet.collect_product:
            raise ValueError(
                "`outlet.condenser` requires `outlet.collect_product`: only a "
                "collected step has a product line."
            )
        termination = config.termination
        if termination.stop_on_pressure_plateau:
            if location == "blow" and termination.pressure_plateau_arm_below_Pa is None:
                raise ValueError(
                    "Blowdown pressure-plateau stopping requires "
                    "`pressure_plateau_arm_below_Pa`."
                )
            if location == "press" and termination.pressure_plateau_arm_above_Pa is None:
                raise ValueError(
                    "Pressurization pressure-plateau stopping requires "
                    "`pressure_plateau_arm_above_Pa`."
                )
        return config
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid configuration for step `{location}`: {exc}") from exc


__all__ = [
    "CompositionConfig",
    "CycleStep",
    "FlowConfig",
    "HorizonConfig",
    "NON_COLLECTING_STEPS",
    "NULLABLE_STEP_FIELDS",
    "OutletConfig",
    "PressureConfig",
    "STEP_FIELD_ORDER",
    "STEP_NAMES",
    "STEP_RECORD_BY_FIELD",
    "STEP_ONLY_TERMINATION_FIELDS",
    "SUPPORTED_STEP_NAMES",
    "StepConfig",
    "StepSequenceConfig",
    "TerminationConfig",
    "ThermalConfig",
    "step_config_from_mapping",
]
