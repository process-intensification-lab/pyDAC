"""Early physical compatibility checks for named model declarations."""

from __future__ import annotations

from collections.abc import Mapping

from pydac.config.schema import ExperimentConfig


def validate_model_configuration(config: ExperimentConfig) -> None:
    """Validate catalog and registry compatibility without building models."""
    from pydac.core.isotherm_models import equilibrium_registry
    from pydac.core.kinetic_models import kinetic_registry
    from pydac.core.species_catalog import get_species_properties
    from pydac.utils.model_registry import (
        EquilibriumChoice,
        KineticChoice,
        validate_equilibrium_choices,
        validate_kinetic_choices,
    )

    mixture = config.simulation.mixture
    adsorption = config.simulation.adsorption

    for species in mixture.species:
        get_species_properties(species)

    validate_equilibrium_choices(
        mixture_species=mixture.species,
        adsorbates=mixture.adsorbates,
        choices=(
            EquilibriumChoice(
                name=name,
                datasets=selection.datasets,
                parameters=selection.parameters,
            )
            for name, selection in adsorption.equilibrium_systems.items()
        ),
        registry=equilibrium_registry(),
    )
    validate_kinetic_choices(
        adsorbates=mixture.adsorbates,
        choices=(
            KineticChoice(
                species=species,
                name=selection.model,
                parameters=selection.parameters,
            )
            for species, selection in adsorption.kinetics.items()
        ),
        registry=kinetic_registry(),
    )
    _validate_sorbent_declarations(config)
    _validate_initial_solid_loadings(config)
    _validate_flow_model_grid(config)
    _validate_registered_architecture(config)
    _validate_wall_override_grid(config)
    _validate_vacuum_outlets(config)
    _validate_feed_blowers(config)
    _validate_output_schedule_and_outlet(config)


def _validate_sorbent_declarations(config: ExperimentConfig) -> None:
    """Check the sorbent inputs the model build would refuse, before a run.

    Every number in them must be a JSON number, the heats must cover every
    adsorbate (others are ignored), the fixed loadings must be well formed, and
    the solid cp must name an implemented model with its exact parameters and
    stay positive at every step's feed temperature (where the model evaluates it).
    """
    from pydac.utils.column.validation import (
        resolve_fixed_adsorbates,
        resolve_heats_of_adsorption,
        resolve_solid_heat_capacity,
    )

    sorbent = config.simulation.sorbent
    _require_json_numbers(sorbent)
    resolve_heats_of_adsorption(
        sorbent.heats_of_adsorption_J_mol, config.simulation.mixture.adsorbates
    )
    resolve_fixed_adsorbates(sorbent.fixed_adsorbates)
    for step in config.simulation.cycle.steps:
        temperature = step.config.thermal.feed_temperature_K
        try:
            resolve_solid_heat_capacity(sorbent.heat_capacity, temperature)
        except ValueError as exc:
            raise ValueError(
                f"simulation.sorbent.heat_capacity (step {step.name!r}, feed "
                f"temperature {temperature} K): {exc}"
            ) from exc


def _validate_initial_solid_loadings(config: ExperimentConfig) -> None:
    """Check the declared startup solid loadings as the model build reads them.

    They must be JSON numbers, cover every adsorbate and nothing else, and be
    finite and non-negative. Only startup can declare them (the step reader
    refuses them elsewhere), so only startup is read.
    """
    from pydac.inputs.defaults.steps import INITIAL_SOLID_LOADINGS_FIELD
    from pydac.utils.column.validation import resolve_initial_solid_loadings

    for step in config.simulation.cycle.steps:
        loadings = getattr(step.config.composition, INITIAL_SOLID_LOADINGS_FIELD)
        if loadings is None:
            continue
        path = f"simulation.cycle.steps.{step.name}.composition.{INITIAL_SOLID_LOADINGS_FIELD}"
        for species, value in loadings.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{path}.{species} must be a number, got {value!r}.")
            try:
                float(value)
            except OverflowError:
                raise ValueError(
                    f"{path}.{species} is outside the floating-point range."
                ) from None
        try:
            resolve_initial_solid_loadings(loadings, config.simulation.mixture.adsorbates)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc


def _require_json_numbers(sorbent) -> None:
    """Refuse booleans, strings and out-of-range integers where the sorbent takes numbers.

    The resolvers convert with float(), which accepts True and "1550" and
    overflows on an integer beyond the float range.
    """
    members = []
    heats = sorbent.heats_of_adsorption_J_mol
    if isinstance(heats, Mapping):
        members += [(f"heats_of_adsorption_J_mol.{name}", value) for name, value in heats.items()]
    parameters = (
        sorbent.heat_capacity.get("parameters")
        if isinstance(sorbent.heat_capacity, Mapping) else None
    )
    if isinstance(parameters, Mapping):
        members += [(f"heat_capacity.parameters.{name}", value) for name, value in parameters.items()]
    fixed = sorbent.fixed_adsorbates
    if isinstance(fixed, Mapping):
        for species, entry in fixed.items():
            if isinstance(entry, Mapping):
                members += [(f"fixed_adsorbates.{species}.{name}", value) for name, value in entry.items()]
    for path, value in members:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"simulation.sorbent.{path} must be a number, got {value!r}.")
        try:
            float(value)
        except OverflowError:
            raise ValueError(
                f"simulation.sorbent.{path} is outside the floating-point range."
            ) from None


def _validate_registered_architecture(config: ExperimentConfig) -> None:
    """Refuse a registered architecture on a grid it does not declare, or with a wall override.

    The wall heat-transfer overrides replace the packed-bed Dixon/h_w coefficients;
    a registered architecture resolves its own bed-to-wall coefficient, which an
    override would silently remove. The model build refuses both cases too.
    """
    from pydac.config.errors import ConfigurationError
    from pydac.core.bed_architectures import ARCHITECTURE_NAMES, validate_spatial_mode

    architecture = config.simulation.bed["architecture"]
    if architecture in ARCHITECTURE_NAMES:
        return
    nr = int(config.simulation.grid.nr)
    validate_spatial_mode("1d" if nr == 1 else "2d", architecture=architecture)
    wall = config.simulation.wall
    for field in ("overall_htc_override_W_m2K", "bed_wall_htc_override_W_m2K"):
        if getattr(wall, field) is not None:
            raise ConfigurationError(
                f"`wall.{field}` applies only to the packed bed; bed architecture "
                f"{architecture!r} resolves its own bed-to-wall coefficient. Set it to null."
            )


def _validate_wall_override_grid(config: ExperimentConfig) -> None:
    """Refuse a wall heat-transfer override the grid would never read.

    The 1D model lumps the bed-to-wall exchange into the Dixon overall coefficient
    U_w; the 2D model resolves the radial profile and only takes the bed-side h_w.
    Each override replaces one of them, so the other dimension would ignore it.
    """
    from pydac.config.errors import ConfigurationError

    wall = config.simulation.wall
    is_1d = int(config.simulation.grid.nr) == 1
    field, needs = (
        ("bed_wall_htc_override_W_m2K", "a 2D grid (grid.nr > 1)") if is_1d
        else ("overall_htc_override_W_m2K", "a 1D grid (grid.nr = 1)")
    )
    if getattr(wall, field) is not None:
        raise ConfigurationError(
            f"`wall.{field}` is read only on {needs}; this case has "
            f"grid.nr = {int(config.simulation.grid.nr)}. Remove it, or run it on {needs}."
        )


def _validate_flow_model_grid(config: ExperimentConfig) -> None:
    """Refuse a channeling flow model on a grid it does not cover (1D, short Nz).

    The model build refuses the same case; checking here makes `--validate` and
    `--dry-run` say so before a run is planned.
    """
    from pydac.core.bed_architectures import PackedBedDesign
    from pydac.core.bed_architectures.channeling import validate_channeling_support
    from pydac.core.bed_architectures.base import CHANNELING_FLOW_MODELS

    bed = config.simulation.bed
    if bed["architecture"] != "packed_bed":
        return
    flow_model = PackedBedDesign(**bed["design"]).resolved_flow_model
    if flow_model not in CHANNELING_FLOW_MODELS:
        return
    grid = config.simulation.grid
    validate_channeling_support(
        flow_model,
        architecture=bed["architecture"],
        is_1d=int(grid.nr) == 1,
        Nr=int(grid.nr),
        Nz=int(grid.nz),
    )


def _validate_vacuum_outlets(config: ExperimentConfig) -> None:
    """Refuse a vacuum-outlet declaration the step or the grid would not read.

    The model build repeats the check; doing it here makes `--validate` and
    `--dry-run` refuse the case before a run is planned. A pump step that runs in
    event mode in some cycle (the same overrides as the run preflight) with its
    pressure target inside the throttle band gets a warning.
    """
    import warnings

    from pydac.core.vacuum_manifold import (
        VACUUM_PUMP_OUTLET_MODEL,
        check_vacuum_outlet,
        target_inside_band_warning,
    )
    from pydac.utils.run.validation import _event_mode_steps

    is_1d = int(config.simulation.grid.nr) == 1
    for step in config.simulation.cycle.steps:
        flow = step.config.flow
        check_vacuum_outlet(
            path=f"simulation.cycle.steps.{step.name}.flow",
            step=step.name,
            is_1d=is_1d,
            model=flow.vacuum_outlet_model,
            speed_m3_s=flow.vacuum_pump_speed_m3_s,
            ultimate_pressure_Pa=flow.vacuum_pump_ultimate_pressure_Pa,
            band_fraction=flow.vacuum_setpoint_band_fraction,
            setpoint_Pa=step.config.pressure.outlet_pressure_Pa,
            idle_mode=flow.vacuum_pump_idle_mode,
            idle_power_W=flow.vacuum_pump_idle_power_W,
        )
    pump_steps = {
        step.name: step for step in config.simulation.cycle.steps
        if step.config.flow.vacuum_outlet_model == VACUUM_PUMP_OUTLET_MODEL
    }
    for name in _event_mode_steps(config, _executed_steps(config) & set(pump_steps)):
        step = pump_steps[name]
        message = target_inside_band_warning(
            path=f"simulation.cycle.steps.{name}.termination",
            termination=step.config.termination,
            setpoint_Pa=step.config.pressure.outlet_pressure_Pa,
            band_fraction=step.config.flow.vacuum_setpoint_band_fraction,
            feed_pressure_Pa=step.config.pressure.feed_pressure_Pa,
        )
        if message is not None:
            warnings.warn(message, UserWarning, stacklevel=2)


def _executed_steps(config: ExperimentConfig) -> set[str]:
    """Steps a step or CSS run of the case executes (both default sequences when unset)."""
    from pydac.utils.tvsa import build_cycle_steps
    from pydac.utils.tvsa.cycle_steps import build_repeating_cycle_steps

    tokens = config.execution.cycle_steps
    if tokens is None:
        return set(build_cycle_steps(None)) | set(build_repeating_cycle_steps(None))
    try:
        return set(build_cycle_steps(tokens))
    except ValueError:      # an unknown token is refused by the run preflight
        return {step.name for step in config.simulation.cycle.steps}


__all__ = ["validate_model_configuration"]


def _validate_feed_blowers(config: ExperimentConfig) -> None:
    import warnings
    from pydac.core.feed_blower import check_feed_blower, target_inside_band_warning
    from pydac.utils.run.validation import _event_mode_steps

    blowers = {}
    for step in config.simulation.cycle.steps:
        flow = step.config.flow
        check_feed_blower(path=f"simulation.cycle.steps.{step.name}.flow", step=step.name,
                          is_1d=int(config.simulation.grid.nr) == 1, model=flow.press_inlet_model,
                          flow_m3_s=flow.feed_blower_flow_m3_s,
                          band_fraction=flow.feed_blower_band_fraction,
                          setpoint_Pa=step.config.pressure.feed_pressure_Pa)
        if flow.press_inlet_model == "feed_blower":
            blowers[step.name] = step
    for name in _event_mode_steps(config, _executed_steps(config) & set(blowers)):
        step = blowers[name]
        message = target_inside_band_warning(
            path=f"simulation.cycle.steps.{name}.termination", termination=step.config.termination,
            setpoint_Pa=step.config.pressure.feed_pressure_Pa,
            band_fraction=step.config.flow.feed_blower_band_fraction,
            outlet_pressure_Pa=step.config.pressure.outlet_pressure_Pa)
        if message is not None:
            warnings.warn(message, UserWarning, stacklevel=2)


def _validate_output_schedule_and_outlet(config: ExperimentConfig) -> None:
    from pydac.utils.output_schedule import validate_schedule_spec
    from pydac.utils.run.validation import reachable_termination_modes
    from pydac.core.ads_outlet import check_ads_outlet
    grid = config.simulation.grid
    # Every mode a run can give the step (own, global, cycle and step overrides):
    # the schedule must fit each horizon, or the case would load and fail in the run.
    modes = reachable_termination_modes(config)
    for step in config.simulation.cycle.steps:
        settings = step.config
        for mode in sorted(modes[step.name]):
            horizon = (settings.horizon.event_max_time_s
                       if mode == "event" and settings.horizon.event_max_time_s is not None
                       else settings.horizon.nominal_time_s)
            try:
                validate_schedule_spec(grid.output_schedule, horizon_s=horizon, nt=grid.nt)
            except ValueError as exc:
                raise ValueError(
                    f"Step `{step.name}` in {mode} mode (horizon {horizon:g} s): {exc}"
                ) from exc
        check_ads_outlet(step=step.name, is_1d=grid.nr == 1,
                         overpressure_Pa=settings.flow.ads_outlet_overpressure_Pa,
                         feed_flow_m3_s=settings.flow.volumetric_flow_m3_s)
