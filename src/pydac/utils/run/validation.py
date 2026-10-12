"""Validation for run flags, step options, and solver diagnostics."""

from collections.abc import Mapping
import warnings

from pydac.utils.solver import (
    DEFAULT_CVODE_SPARSITY_MODE,
    normalize_cvode_sparsity_mode,
)
from pydac.utils.tvsa.steps import SUPPORTED_STEP_TOKENS, normalize_step_token_text
from pydac.utils.tvsa.termination import SUPPORTED_TERMINATION_MODES, normalize_termination_mode

SUPPORTED_STEP_OPTION_KEYS = {
    "solve",
    "plot_profiles",
    "use_sparsity",
    "compute_numerical_sparsity",
    "termination_mode",
}


def validate_output_options(
    *,
    plot_profiles,
    plot_final_cycle_profiles,
    keep_only_final_cycle,
):
    """Validate mutually exclusive CSS profile-output settings."""
    if plot_profiles and keep_only_final_cycle:
        raise ValueError(
            "`plot_profiles=True` cannot be combined with `keep_only_final_cycle=True`."
        )
    if plot_profiles and plot_final_cycle_profiles:
        raise ValueError("Choose profiles for every cycle or for the final cycle only.")


def validate_step_options_structure(step_options):
    """
    Validate schema and content of step_options.

    Expected structure:
    {
        "<STEP_TOKEN>": {
            "solve": bool,
            "plot_profiles": bool,
            "use_sparsity": bool,
            "compute_numerical_sparsity": bool,
            "termination_mode": "time" | "event",
        },
        ...
    }
    """
    if step_options is None:
        return

    # `Mapping`, not `dict`: a frozen case reloads these as `FrozenMapping`, so a
    # `dict` check makes `execution.step_options` unusable on exactly the replay
    # path frozen cases exist for.
    if not isinstance(step_options, Mapping):
        raise TypeError("step_options must be a mapping of step -> options, or None.")

    for step_token, opts in step_options.items():
        if not isinstance(step_token, str) or not step_token.strip():
            raise TypeError("Each step_options key must be a non-empty string step token.")

        token = normalize_step_token_text(step_token)
        if token not in SUPPORTED_STEP_TOKENS:
            valid = ", ".join(sorted(SUPPORTED_STEP_TOKENS))
            raise ValueError(
                f"Unsupported step token '{step_token}' in step_options. "
                f"Valid tokens: {valid}."
            )

        if not isinstance(opts, Mapping):
            raise TypeError(
                f"step_options['{step_token}'] must be a mapping of option overrides."
            )

        unknown_keys = set(opts.keys()) - SUPPORTED_STEP_OPTION_KEYS
        if unknown_keys:
            supported = ", ".join(sorted(SUPPORTED_STEP_OPTION_KEYS))
            bad = ", ".join(sorted(unknown_keys))
            raise ValueError(
                f"Unsupported option key(s) for step '{step_token}': {bad}. "
                f"Supported keys: {supported}."
            )

        for key, value in opts.items():
            if key == "termination_mode":
                normalize_termination_mode(
                    value,
                    option_name=f"step_options['{step_token}']['termination_mode']",
                    allow_none=False,
                )
                continue

            if not isinstance(value, bool):
                raise TypeError(
                    f"step_options['{step_token}']['{key}'] must be bool. "
                    f"Received type: {type(value).__name__}."
                )

def validate_flags(
    solve: bool,
    plot_profiles: bool = False,
    use_sparsity: bool = True,
    compute_numerical_sparsity: bool = False,
    termination_mode=None,
    step_options=None,
):
    """
    Validate run flags and step_options schema.
    """
    validate_step_options_structure(step_options)
    normalize_termination_mode(termination_mode, allow_none=True)
    if not isinstance(solve, bool):
        raise TypeError(f"`solve` must be bool. Received type: {type(solve).__name__}.")
    if not isinstance(plot_profiles, bool):
        raise TypeError(
            f"`plot_profiles` must be bool. Received type: {type(plot_profiles).__name__}."
        )
    if not isinstance(use_sparsity, bool):
        raise TypeError(
            f"`use_sparsity` must be bool. Received type: {type(use_sparsity).__name__}."
        )
    if not isinstance(compute_numerical_sparsity, bool):
        raise TypeError(
            "`compute_numerical_sparsity` must be bool. "
            f"Received type: {type(compute_numerical_sparsity).__name__}."
        )
    return True


def validate_steps_against_mixture(case, executed_steps):
    """Refuse executed steps the mixture cannot feed or whose product it cannot hold.

    A steam step injects H2O, so it needs H2O in the mixture; a step collected as
    product reports recovered CO2, so it needs CO2.
    """
    species = set(case.simulation.mixture.species)
    executed = set(executed_steps)
    if "H2O" not in species:
        steam = sorted(executed & {"s_purge", "s_des"})
        if steam:
            raise ValueError(
                f"Steps {steam} inject steam, but the mixture has no H2O; remove them "
                "from execution.cycle_steps or add H2O to mixture.species."
            )
    if "CO2" not in species:
        outlets = {step.name: step.config.outlet for step in case.simulation.cycle.steps}
        collected = sorted(name for name in executed if outlets[name].collect_product)
        if collected:
            raise ValueError(
                f"Steps {collected} collect CO2 product, but the mixture has no CO2; set "
                "their outlet.collect_product=false or add CO2 to mixture.species."
            )
    validate_event_metrics_against_mixture(case, executed_steps)


def _event_mode_steps(case, executed_steps):
    """Executed steps that run in event mode in at least one cycle of the case.

    A step's mode is its own `termination.mode` unless an execution override
    sets one: the global `termination_mode`, then a cycle's, then a step's.
    """
    execution = case.execution
    overrides = [(execution.termination_mode, execution.step_options)]
    for options in execution.cycle_options.values():
        overrides.append((
            options.get("termination_mode", execution.termination_mode),
            options.get("step_options", {}),
        ))
    own = {step.name: step.config.termination.mode for step in case.simulation.cycle.steps}
    return sorted(
        name
        for name in set(executed_steps)
        if any(
            (step_options.get(name, {}).get("termination_mode") or mode or own[name])
            == "event"
            for mode, step_options in overrides
        )
    )


def validate_event_metrics_against_mixture(case, executed_steps):
    """Refuse an event-mode step whose stop metric the mixture cannot evaluate."""
    from pydac.utils.tvsa.termination import EVENT_METRIC_REQUIREMENTS

    mixture = case.simulation.mixture
    available = {"species": set(mixture.species), "adsorbate": set(mixture.adsorbates)}
    metrics = {step.name: step.config.termination.metric for step in case.simulation.cycle.steps}
    for name in _event_mode_steps(case, executed_steps):
        requirement = EVENT_METRIC_REQUIREMENTS[metrics[name]]
        if requirement is not None and requirement[1] not in available[requirement[0]]:
            kind, needed = requirement
            raise ValueError(
                f"Step `{name}` runs in event mode on `{metrics[name]}`, which needs "
                f"{needed} as a mixture {kind}; choose another "
                f"`steps.{name}.termination.metric` or run the step in time mode."
            )


def validate_step_case(case):
    """Validate a step run: its runner settings and the steps against the mixture."""
    from pydac.utils.tvsa import build_cycle_steps

    validate_step_runner_configuration(case.execution)
    validate_steps_against_mixture(case, build_cycle_steps(cycle_steps=case.execution.cycle_steps))
    return True


def validate_step_runner_configuration(execution):
    """Validate runner-level settings needed before a standalone step run."""
    validate_flags(
        solve=execution.solve,
        plot_profiles=execution.plot_profiles,
        use_sparsity=execution.use_sparsity,
        compute_numerical_sparsity=execution.compute_numerical_sparsity,
        termination_mode=execution.termination_mode,
        step_options=dict(execution.step_options) or None,
    )
    from pydac.utils.tvsa import build_cycle_steps

    build_cycle_steps(cycle_steps=execution.cycle_steps)
    return True


def warn_advanced_sparsity_options(
    *,
    use_sparsity,
    compute_numerical_sparsity,
    cvode_sparsity_mode,
    stacklevel=2,
):
    """
    Warn when a run departs from the recommended analytical sparse CVODE path.
    """
    if not use_sparsity:
        warnings.warn(
            "`execution.use_sparsity=false` disables the recommended analytical sparse "
            "CVODE path and should only be used for solver diagnostics.",
            UserWarning,
            stacklevel=stacklevel,
        )

    if compute_numerical_sparsity:
        warnings.warn(
            "`execution.compute_numerical_sparsity=true` is a slow diagnostic mode. "
            "Production runs should use the analytical sparsity pattern, which "
            "is generated in memory automatically. The finite-difference "
            "Jacobians are saved into each step's `numerical_jacobian/` folder.",
            UserWarning,
            stacklevel=stacklevel,
        )

    if normalize_cvode_sparsity_mode(cvode_sparsity_mode) != DEFAULT_CVODE_SPARSITY_MODE:
        warnings.warn(
            f"`execution.cvode_sparsity_mode={cvode_sparsity_mode}` is intended only "
            f"for diagnostics. The recommended default is "
            f"`{DEFAULT_CVODE_SPARSITY_MODE}`.",
            UserWarning,
            stacklevel=stacklevel,
        )
