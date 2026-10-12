"""Public Python layer of pyDAC: load a case, run it, continue it, report it.

Everything here is reached as ``pydac.<name>``; the ``pydac`` command calls the
same functions, so a case run from the terminal and from Python lands in the
same run folder with the same fingerprint.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : load_case -> run_steps / run_css / run_nsga2 -> resume / report.
# Owns     : the run folder (RUN/CAMPAIGN), occupied-folder refusal and
#            --overwrite, the CSS post-solve outputs, RunResult.
# Not here : exit codes and printing errors (cli.py); the workflows themselves
#            (workflows/); what a case means (config/).
# Contract : public case loading, run outcomes, output folders, reporting,
#            continuation and refusal rules.
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import contextlib
import copy
import functools
import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydac.config.errors import ConfigurationError, RunRefused
from pydac.config.schema import ExperimentConfig

STEPS, CSS, NSGA2 = "steps", "css", "nsga2"

# What `run_nsga2_optimization` keeps in `CAMPAIGN/results`: either file marks
# a folder as an NSGA-II campaign.
_CAMPAIGN_MARKERS = ("checkpoint_state.json", "campaign_identity.json")


@dataclass(frozen=True)
class RunResult:
    """Where a run lives and how it ended.

    ``folder`` is the RUN or CAMPAIGN folder that ``resume`` and ``report``
    take; ``results_folder`` is its spatial folder (``1D_mode``/``2D_mode``)
    or ``results/``. ``css_reached`` and ``cycle`` are ``None`` for step runs
    and NSGA-II campaigns.
    """

    folder: Path
    results_folder: Path
    workflow: str
    css_reached: bool | None = None
    cycle: int | None = None


def _figures_stay_on_disk(run):
    """pydac saves and closes every figure it draws: none may pop up meanwhile in
    an interactive session (IPython with a window backend).

    The backend is selected before ``ioff``, as pyplot does at the first figure: in
    Jupyter that selection turns interactive mode on (matplotlib-inline), and inside
    ``ioff`` it would be undone on exit, so no later notebook figure would show."""
    @functools.wraps(run)
    def quiet(*args, **kwargs):
        import matplotlib.pyplot as plt

        plt._get_backend_mod()
        with plt.ioff():
            return run(*args, **kwargs)
    return quiet


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def load_case(path=None, *, settings=None, exact: bool = False) -> ExperimentConfig:
    """Load a case.

    ``exact=False``: built-in defaults < ``path`` (a partial or
    complete JSON document, or ``None`` for the defaults) < ``settings`` (a
    mapping, or ``["path=value", ...]`` as ``--set``). ``exact=True`` loads a
    complete frozen case exactly and refuses ``settings``.
    """
    if exact:
        if path is None:
            raise ConfigurationError("exact=True loads a complete case: give its path.")
        if settings:
            raise ConfigurationError(
                "exact=True runs the case exactly and accepts no settings; load it with "
                "exact=False to make a variant."
            )
        from pydac.inputs.case_io import load_case as load_frozen_case

        return load_frozen_case(path)
    from pydac.config.loader import load_experiment

    return load_experiment(path, settings=_settings(settings))


def export_case(case: ExperimentConfig, path, *, overwrite: bool = False) -> Path:
    """Write ``case`` as a complete frozen case and return ``path``."""
    _require_case(case)
    from pydac.inputs.case_io import export_case as write_case

    return write_case(case, path, overwrite=overwrite)


def export_settings(settings, path, *, base=None, overwrite: bool = False) -> Path:
    """Write ``settings`` as a preset file and return ``path``.

    The file states only what ``settings`` (and the optional preset ``base``
    under them) state; every other field keeps the built-in defaults, so
    ``load_case(path)`` gives the case ``load_case(base, settings=settings)``
    gives. Invalid settings are refused before anything is written.
    """
    text = _preset_text(settings, base)
    return _write_preset(text, path, overwrite=overwrite)


def _preset_text(settings, base, *, case: ExperimentConfig | None = None) -> str:
    """The preset file of ``settings`` over ``base``, checked to load ``case``.

    ``case`` defaults to ``load_case(base, settings=settings)``. The check reads
    the rendered text back, so the file itself is what loads the same case.
    """
    from pydac.config.loader import load_experiment, settings_to_preset
    from pydac.config.serialization import to_jsonable
    from pydac.inputs.case_io import case_fingerprint

    settings = _settings(settings)
    if case is None:
        case = load_experiment(base, settings=settings)
    try:
        text = _preset_json(to_jsonable(settings_to_preset(settings, base=base))) + "\n"
    except ValueError as exc:  # a non-standard JSON number
        raise ConfigurationError(f"These settings cannot be written as JSON: {exc}") from exc
    if case_fingerprint(load_experiment(json.loads(text))) != case_fingerprint(case):
        raise ConfigurationError(
            "These settings cannot be written as a preset that loads the same case; "
            "export the complete case with export_case instead."
        )
    return text


def _refuse_existing_preset(path, *, overwrite: bool) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise ConfigurationError(
            f"Refusing to overwrite existing preset `{target}`; use --overwrite-preset "
            "(overwrite=True in Python) to replace it."
        )
    return target


def _write_preset(text: str, path, *, overwrite: bool) -> Path:
    target = _refuse_existing_preset(path, overwrite=overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _preset_json(value, indent: str = "") -> str:
    # Indented JSON, with a list of plain values on one line as people write it.
    inner = indent + "  "
    if isinstance(value, Mapping) and value:
        items = (f"{inner}{json.dumps(key)}: {_preset_json(item, inner)}"
                 for key, item in value.items())
        return "{\n" + ",\n".join(items) + f"\n{indent}}}"
    if isinstance(value, list) and any(isinstance(item, (Mapping, list)) for item in value):
        items = (inner + _preset_json(item, inner) for item in value)
        return "[\n" + ",\n".join(items) + f"\n{indent}]"
    return json.dumps(value, allow_nan=False)


def _settings(settings):
    if settings is None or isinstance(settings, Mapping):
        return settings
    if isinstance(settings, str):
        return [settings]
    return list(settings)


def _require_case(case) -> None:
    # Run functions take a loaded case so that frozen-case loading and preset resolution are explicit.
    if not isinstance(case, ExperimentConfig):
        raise ConfigurationError(
            f"Expected a case loaded with pydac.load_case(path), got {type(case).__name__}."
        )


def _preflight(validate, case: ExperimentConfig) -> None:
    """Run a workflow's settings check; it solves and writes nothing."""
    try:
        validate(case)
    except ConfigurationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc
    # The diagnostic solver modes are valid but slow or nonstandard: say so once
    # per run instead of letting them pass silently.
    from pydac.utils.run.validation import warn_advanced_sparsity_options

    execution = case.execution
    warn_advanced_sparsity_options(
        use_sparsity=execution.use_sparsity,
        compute_numerical_sparsity=execution.compute_numerical_sparsity,
        cvode_sparsity_mode=execution.cvode_sparsity_mode,
        stacklevel=3,
    )


def _validator(workflow: str):
    if workflow == STEPS:
        from pydac.utils.run.validation import validate_step_case

        return validate_step_case
    from pydac.utils.css.runner import validate_css_case

    return validate_css_case


# --------------------------------------------------------------------------
# Run folders and occupied-folder rules
# --------------------------------------------------------------------------

def run_folder(case: ExperimentConfig, workflow: str) -> str:
    """The run folder a fresh step or CSS run of ``case`` writes."""
    from pydac.utils.run import make_base_folder

    return make_base_folder(
        output_root=case.execution.output_root,
        cycle=workflow == CSS,
        extra_tag=case.execution.extra_tag,
        experiment_config=case,
    )


def occupied(folder) -> bool:
    """A fresh run refuses a RUN folder that exists and is not empty."""
    path = Path(folder)
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


def _refuse_occupied(folder) -> None:
    path = Path(folder)
    if occupied(path):
        raise RunRefused(
            f"`{path}` already holds a run. Continue it with --resume {path} "
            "(pydac.resume), replace it with --overwrite (overwrite=True), or set "
            "another execution.extra_tag."
        )


def _claim(folder, *, overwrite: bool) -> None:
    """Refuse an occupied RUN folder, or delete exactly it with ``overwrite``."""
    path = Path(folder)
    if not occupied(path):
        return
    if not overwrite:
        _refuse_occupied(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"Overwrite: deleted {_absolute(path)}")


def _absolute(folder) -> Path:
    return Path(folder).absolute()


def _announce(label: str, folder) -> None:
    print(f"{label}: {_absolute(folder)}", flush=True)


def _spatial(case: ExperimentConfig) -> str:
    return "1D_mode" if case.simulation.grid.nr == 1 else "2D_mode"


def _require_dict(outcome, workflow: str) -> dict:
    if not isinstance(outcome, dict):
        raise RuntimeError(f"The {workflow} workflow did not return run details.")
    return outcome


# --------------------------------------------------------------------------
# Step runs and CSS runs
# --------------------------------------------------------------------------

@_figures_stay_on_disk
def run_steps(case: ExperimentConfig, *, overwrite: bool = False,
              snapshot_times=None) -> RunResult:
    """Run the case's step sequence in a fresh RUN folder.

    ``snapshot_times`` chooses the times of the step plots, for example
    ``{"heat": {"T_bed": {"times": [1, 10, 30, 60, 90, 150], "unit": "min"}}}``:
    per step and field (``T_bed``, ``p_tot``, ``c_tot``, ``y_<gas>``, ``q_<adsorbate>``),
    1 to 6 times >= 0 in ``"s"`` (default), ``"min"`` or ``"h"``, used by the 2D map and
    the 1D profiles of that field (the profiles the step draws). It changes only the figures, not the case (same
    fingerprint, not saved with the run).
    """
    folder = _check_fresh_run(case, STEPS)
    times = _snapshot_times(case, STEPS, snapshot_times)
    _claim(folder, overwrite=overwrite)
    with _snapshot_overrides(times):
        return _solve_steps(case, folder)


@_figures_stay_on_disk
def run_css(case: ExperimentConfig, *, overwrite: bool = False,
            snapshot_times=None) -> RunResult:
    """Run cycles to cyclic steady state (or ``max_cycles``) in a fresh RUN folder.

    Not reaching CSS is not an error: the result says ``css_reached=False``.
    ``snapshot_times`` chooses the times of the step plots, for example
    ``{"heat": {"T_bed": {"times": [1, 10, 30, 60, 90, 150], "unit": "min"}}}``:
    per step and field (``T_bed``, ``p_tot``, ``c_tot``, ``y_<gas>``, ``q_<adsorbate>``),
    1 to 6 times >= 0 in ``"s"`` (default), ``"min"`` or ``"h"``, used by the 2D map and
    the 1D profiles of that field (the profiles the step draws). It changes only the figures, not the case (same
    fingerprint, not saved with the run).
    """
    folder = _check_fresh_run(case, CSS)
    times = _snapshot_times(case, CSS, snapshot_times)
    _claim(folder, overwrite=overwrite)
    with _snapshot_overrides(times):
        result, results = _solve_css(case, folder)
        _finish_css(case, result, results)
    return result


# The 2D map of a field is a 2 x 3 grid of snapshots: at most six times.
_SNAPSHOT_PANELS = 6
_SNAPSHOT_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0}


def _drawn_steps(case: ExperimentConfig, workflow: str) -> list[str]:
    """The steps whose plots a fresh run of ``case`` draws, in run order: the same
    switches as the runners (step_options over plot_profiles; for CSS also
    plot_final_cycle_profiles and the plot switches of cycle_options)."""
    from pydac.utils.tvsa.cycle_steps import (
        build_cycle_steps, build_cyclic_cycle_steps, build_repeating_cycle_steps,
    )
    from pydac.utils.tvsa.steps import normalize_step_options

    execution = case.execution
    if workflow != CSS:
        options = normalize_step_options(execution.step_options)
        return [step for step in build_cycle_steps(execution.cycle_steps)
                if options.get(step, {}).get("plot_profiles", execution.plot_profiles)]
    repeating = build_repeating_cycle_steps(execution.cycle_steps)
    drawn = set()
    if execution.plot_profiles:  # live plots in every cycle not overridden below
        drawn.update(["startup", *repeating])
    if execution.plot_final_cycle_profiles:  # replay of the last cycle (not cycle 1)
        drawn.update(repeating)
    for cycle, override in execution.cycle_options.items():
        cycle_plots = override.get("plot_profiles", execution.plot_profiles)
        options = normalize_step_options(override.get("step_options"))
        drawn.update(step for step in build_cyclic_cycle_steps(int(cycle), execution.cycle_steps)
                     if options.get(step, {}).get("plot_profiles", cycle_plots))
    return [step for step in ["startup", *repeating] if step in drawn]


def _snapshot_times(case: ExperimentConfig, workflow: str, snapshot_times, *, steps=None):
    """Check the ``snapshot_times`` of a run (see ``run_css``); return them by canonical
    step, or None. A field left out keeps pyDAC's choice. ``steps`` (a report) names the
    steps that will be drawn; a run instead needs its own step plots on. Writes nothing."""
    if snapshot_times is None:
        return None
    usage = ('snapshot_times must be a mapping {step: {field: {"times": [...], '
             '"unit": "min"}}}')
    if not isinstance(snapshot_times, Mapping) or not snapshot_times:
        raise ConfigurationError(f"{usage}; got {snapshot_times!r}.")
    execution = case.execution
    from pydac.utils.tvsa.cycle_steps import build_cycle_steps, build_repeating_cycle_steps
    from pydac.utils.tvsa.steps import normalize_step_token

    # A report replays its cycle and draws every step; a run draws the steps whose
    # plots its switches turn on.
    switches = "execution.plot_profiles or execution.step_options" if workflow != CSS else (
        "execution.plot_profiles, execution.plot_final_cycle_profiles or execution.cycle_options")
    if steps is not None:
        steps = drawn = list(steps)
    else:
        drawn = _drawn_steps(case, workflow)
        if not drawn:
            raise ConfigurationError(
                f"snapshot_times sets the times of the step plots, but this run draws none: "
                f"turn them on with {switches}."
            )
        steps = (["startup", *build_repeating_cycle_steps(execution.cycle_steps)]
                 if workflow == CSS else build_cycle_steps(execution.cycle_steps))
    mixture = case.simulation.mixture
    fields = [*(f"y_{name}" for name in mixture.species),
              *(f"q_{name}" for name in mixture.adsorbates), "c_tot", "T_bed", "p_tot"]
    checked: dict[str, dict[str, dict]] = {}
    for step, by_field in snapshot_times.items():
        try:
            token = normalize_step_token(step)
        except ValueError as exc:
            raise ConfigurationError(f"snapshot_times: {exc}") from exc
        if token not in steps:
            raise ConfigurationError(
                f"snapshot_times: step {step!r} is not in this run; its steps are "
                f"{', '.join(steps)}."
            )
        if token not in drawn:
            raise ConfigurationError(
                f"snapshot_times: this run draws no plots of step {token!r}; it draws "
                f"{', '.join(drawn)} (switches: {switches})."
            )
        if not isinstance(by_field, Mapping) or not by_field:
            raise ConfigurationError(f"{usage}; step {step!r} has {by_field!r}.")
        for field, spec in by_field.items():
            label = f"snapshot_times[{step!r}][{field!r}]"
            if field not in fields:
                raise ConfigurationError(
                    f"{label}: no step plot has this field; the fields are {', '.join(fields)}."
                )
            if field in checked.get(token, {}):
                raise ConfigurationError(f"{label}: given twice for step {token!r}.")
            if not isinstance(spec, Mapping) or "times" not in spec or set(spec) - {"times", "unit"}:
                raise ConfigurationError(f'{label} must be {{"times": [...], "unit": "min"}}.')
            unit = spec.get("unit", "s")
            if unit not in _SNAPSHOT_UNITS:
                raise ConfigurationError(
                    f"{label}: unit must be one of {', '.join(_SNAPSHOT_UNITS)}; got {unit!r}."
                )
            times = spec["times"]
            if (isinstance(times, (str, bytes)) or not isinstance(times, (list, tuple))
                    or not 1 <= len(times) <= _SNAPSHOT_PANELS
                    or any(isinstance(time, bool) or not isinstance(time, (int, float))
                           or not 0 <= _seconds(time, unit) < math.inf for time in times)):
                raise ConfigurationError(
                    f"{label}: times must be 1 to {_SNAPSHOT_PANELS} numbers >= 0, finite also in "
                    f"seconds (one per panel of the 2 x 3 map); got {times!r}."
                )
            if len(set(times)) != len(times):
                raise ConfigurationError(f"{label}: a time is repeated; got {times!r}.")
            checked.setdefault(token, {})[field] = {
                "times": tuple(float(time) for time in times), "unit": unit,
            }
    return checked


def _seconds(time, unit: str) -> float:
    """``time`` in seconds; inf when it does not fit a float (NaN stays NaN)."""
    try:
        return float(time) * _SNAPSHOT_UNITS[unit]
    except OverflowError:
        return math.inf


@contextlib.contextmanager
def _snapshot_overrides(times):
    """Lay ``times`` over pyDAC's snapshot choice for the duration of one run."""
    if not times:
        yield
        return
    from pydac.utils.plot_helpers import SNAPSHOT_TIME_OVERRIDES as table

    saved = copy.deepcopy(table)
    try:
        for step, by_field in times.items():
            table.setdefault(step, {}).update(by_field)
        yield
    finally:
        table.clear()
        table.update(saved)


def _check_fresh_run(case: ExperimentConfig, workflow: str) -> str:
    """Validate a fresh step/CSS run completely; return its RUN folder. Writes nothing."""
    _require_case(case)
    _preflight(_validator(workflow), case)
    return run_folder(case, workflow)


def _solve_steps(case: ExperimentConfig, folder) -> RunResult:
    from pydac.workflows.orchestrator_TVSA_steps import run_steps_simulation

    _announce("Run folder", folder)
    _require_dict(run_steps_simulation(experiment_config=case, base_folder=folder), STEPS)
    _announce("Run folder", folder)
    return RunResult(
        folder=_absolute(folder),
        results_folder=_absolute(folder) / _spatial(case),
        workflow=STEPS,
    )


def _solve_css(case: ExperimentConfig, folder) -> tuple[RunResult, str]:
    """Solve the cycles; return the result and the spatial folder as the workflow named it."""
    from pydac.workflows.orchestrator_TVSA_cycle_css import run_css_simulation

    _announce("Run folder", folder)
    outcome = _require_dict(run_css_simulation(experiment_config=case, base_folder=folder), CSS)
    result = RunResult(
        folder=_absolute(folder),
        results_folder=_absolute(outcome["results_base_folder"]),
        workflow=CSS,
        css_reached=bool(outcome["css_reached"]),
        # Absolute cycle index, also after a resume.
        cycle=outcome["css_cycle_idx"] or outcome["cycles_completed"],
    )
    return result, outcome["results_base_folder"]


def _finish_css(case: ExperimentConfig, result: RunResult, results) -> None:
    """After a CSS solve: the requested plots/KPIs, the O2 audit, then pruning."""
    execution = case.execution
    _css_outputs(
        results,
        result.cycle,
        css_reached=result.css_reached,
        fmt=execution.plot_format,
        profiles=execution.plot_cyclic_profiles,
        kpis=execution.generate_kpi_outputs,
    )
    if execution.keep_only_final_cycle:
        from pydac.utils.css.pruning import prune_cycle_directories

        removed = prune_cycle_directories(Path(results), keep_cycle_idx=result.cycle)
        print(f"Retained cycle_{result.cycle}; removed intermediate cycles: {removed or 'none'}")
    _announce("Run folder", result.folder)


def _css_outputs(results, cycle: int, *, css_reached, fmt, profiles: bool, kpis: bool) -> None:
    label = "" if css_reached else " (CSS not reached: last completed cycle)"
    print(f"Post-processing cycle {cycle}{label}.")
    if profiles:
        from pydac.post_processing.profiles.cyclic_plotting import CyclicSteadyStateProfiles

        CyclicSteadyStateProfiles(base_folder=results, css_cycle_idx=cycle).plot_all_css_fields(
            fmt=fmt
        )
    if kpis and not (Path(results) / f"cycle_{cycle}" / "ads").is_dir():
        # Cycle 1 runs startup in place of adsorption; the cycle KPIs are
        # normalized by the adsorption step, so they exist from cycle 2 on.
        print(f"KPIs skipped: cycle {cycle} has no adsorption step (cycle 1 starts with "
              "startup); KPIs are reported from cycle 2 on.")
        kpis = False
    if kpis:
        from pydac.post_processing.kpi.cycle_kpis import CycleKPIs
        from pydac.post_processing.kpi.plotting import KPIPlotting
        from pydac.post_processing.kpi.reporting import KPIReporting
        from pydac.post_processing.metrics.cycle_data_loader import CycleDataLoader

        cycle_kpis = CycleKPIs(CycleDataLoader(results, cycle_index=cycle).load())
        kpis_folder = Path(results) / "kpis"
        KPIPlotting(cycle_kpis, output_folder=kpis_folder).plot_all_kpi_figures(fmt=fmt)
        KPIReporting(cycle_kpis, output_folder=kpis_folder).save_css_summary_report(print_report=True)
    # Compact report-only check, after the regular KPI post-processing.
    from pydac.utils.solver.oxygen_hot_bed_audit import run_oxygen_hot_bed_check

    run_oxygen_hot_bed_check(results, cycle)


# --------------------------------------------------------------------------
# NSGA-II campaigns
# --------------------------------------------------------------------------

def _nsga2_workflow():
    """The NSGA-II orchestrator, imported on first use (pymoo is heavy to import)."""
    from pydac.workflows import orchestrator_NSGA2

    return orchestrator_NSGA2


@dataclass(frozen=True)
class _CampaignPlan:
    run_config: Any
    folder: str
    pop_size: int
    generations: int
    seed: int
    termination: Any


@_figures_stay_on_disk
def run_nsga2(
    case: ExperimentConfig,
    bounds,
    *,
    pop_size: int = 100,
    generations: int = 100,
    workers: int = 1,
    seed: int = 1,
    purity_min: float | None = None,
    oxygen_ppm_max: float | None = None,
    eq_work_max: float | None = None,
    steam_condensation: bool = False,
    termination: str = "fixed_generations",
    ftol: float | None = None,
    period: int | None = None,
    n_skip: int | None = None,
    max_evaluations: int | None = None,
) -> RunResult:
    """Run a local NSGA-II campaign over ``bounds`` (a mapping or a JSON file of
    ``variable: [lower, upper]``). The options mirror the ``pydac nsga2`` flags;
    ``ftol``/``period``/``n_skip`` (defaults 5e-3, 10, 0) apply only with
    ``termination="robust_pareto"``.
    """
    plan = _plan_campaign(
        case, bounds, pop_size=pop_size, generations=generations, workers=workers,
        seed=seed, purity_min=purity_min, oxygen_ppm_max=oxygen_ppm_max,
        eq_work_max=eq_work_max, steam_condensation=steam_condensation,
        termination=termination, ftol=ftol, period=period, n_skip=n_skip,
        max_evaluations=max_evaluations,
    )
    return _run_campaign(plan, workers=workers)


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"`{name}` must be a positive integer, got {value!r}.")
    return value


def _load_bounds(bounds) -> dict:
    if isinstance(bounds, (str, os.PathLike)):
        try:
            bounds = json.loads(Path(bounds).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"Cannot read the bounds file `{bounds}`: {exc}") from exc
    if not isinstance(bounds, Mapping) or not bounds:
        raise ConfigurationError("Bounds must be a nonempty JSON object of variable: [lower, upper].")
    loaded = {}
    for name, pair in bounds.items():
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in pair)
                or pair[0] >= pair[1]):
            raise ConfigurationError(
                f"Bounds for {name!r} must be two finite numbers with lower < upper."
            )
        loaded[name] = list(pair)
    return loaded


def _threshold(name: str, value) -> float | None:
    # A float, as the CLI parses it: 60 and 60.0 must name the same campaign.
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(f"`{name}` must be a finite number or None, got {value!r}.")
    return float(value)


def _plan_campaign(case, bounds, *, pop_size, generations, workers, seed, purity_min,
                   oxygen_ppm_max, eq_work_max, steam_condensation, termination, ftol,
                   period, n_skip, max_evaluations) -> _CampaignPlan:
    """Validate a fresh campaign completely; return its plan. Writes nothing."""
    _require_case(case)
    purity_min = _threshold("purity_min", purity_min)
    oxygen_ppm_max = _threshold("oxygen_ppm_max", oxygen_ppm_max)
    eq_work_max = _threshold("eq_work_max", eq_work_max)
    _preflight(_validator(CSS), case)
    for name, value in (("pop_size", pop_size), ("generations", generations),
                        ("workers", workers)):
        _positive_int(name, value)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigurationError(f"`seed` must be an integer, got {seed!r}.")
    if max_evaluations is not None:
        _positive_int("max_evaluations", max_evaluations)
    if steam_condensation and "H2O" not in case.simulation.mixture.species:
        raise ConfigurationError(
            "steam_condensation constrains the preheating against steam condensation; "
            "the case's mixture has no H2O."
        )
    # The campaign drops this constraint on a cycle without s_des: refuse instead of ignoring it.
    if steam_condensation:
        from pydac.utils.run.steps import infer_desorption_step
        from pydac.utils.tvsa.cycle_steps import DEFAULT_CYCLIC_SEQUENCE

        cycle = case.execution.cycle_steps
        steps = DEFAULT_CYCLIC_SEQUENCE if cycle is None else cycle
        if infer_desorption_step(cycle_steps=steps, default=None) != "s_des":
            raise ConfigurationError(
                "steam_condensation constrains the preheating before steam-assisted "
                f"desorption (s_des); the case's cycle {list(steps)} has no s_des step."
            )
    # Without O2 the product has no O2 content: the constraint would be NaN.
    if oxygen_ppm_max is not None and "O2" not in case.simulation.mixture.species:
        raise ConfigurationError(
            "oxygen_ppm_max constrains the O2 in the product; the case's mixture has no O2."
        )
    robust = {name: value for name, value in
              (("ftol", ftol), ("period", period), ("n_skip", n_skip)) if value is not None}
    if termination != "robust_pareto" and robust:
        raise ConfigurationError(
            f"{', '.join(robust)} only apply with termination='robust_pareto'."
        )
    loaded = _load_bounds(bounds)

    from pydac.utils.run import current_grid_tag, make_base_folder

    workflow = _nsga2_workflow()
    try:
        termination_config = workflow.NSGA2TerminationConfig(
            mode=termination, max_evaluations=max_evaluations, **robust
        )
        run_config = workflow.NSGA2RunConfig(
            base_folder="", bounds=loaded, experiment=case,
            constraints=workflow.NSGA2Constraints(
                purity_min=purity_min, oxygen_ppm_max=oxygen_ppm_max,
                eq_work_max=eq_work_max, steam_condensation=steam_condensation,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc
    extra = set(loaded) - set(run_config.var_names)
    if extra:
        raise ConfigurationError(f"Unknown or inactive decision variables: {sorted(extra)}")
    # A base case that disagrees with the bounds would fail every candidate.
    workflow.preflight_search_space(run_config)
    # The folder is named by the whole campaign, not only the base case: a
    # different search space, constraint set or optimizer is a new campaign.
    identity = workflow.nsga2_campaign_identity(
        run_config, pop_size=pop_size, n_generations=generations, seed=seed,
        termination_config=termination_config,
    )
    # The leaf is [TAG_]GRID_packed_bed_HASH: execution.extra_tag first, as in a
    # step or CSS run folder (the identity already holds it, so the hash is unchanged).
    grid_tag = current_grid_tag(case.simulation.grid)
    tag = (case.execution.extra_tag or "").strip()
    folder = make_base_folder(
        mode_override="nsga2", extra_tag=f"{tag}_{grid_tag}" if tag else grid_tag,
        output_root=case.execution.output_root, experiment_config=case,
        identity_hash=identity["sha256"],
    )
    workflow.guard_campaign_folder(folder, identity, continuation=False)
    run_config.base_folder = folder
    return _CampaignPlan(run_config, folder, pop_size, generations, seed, termination_config)


def _limit_worker_threads() -> None:
    # One numerical thread per local worker (as the NSGA-II runner sets it).
    os.environ["MKL_NUM_THREADS"] = "1"


def _run_campaign(plan: _CampaignPlan, *, workers: int) -> RunResult:
    from pydac.config.loader import write_resolved_config

    workflow = _nsga2_workflow()
    _limit_worker_threads()
    _announce("Campaign folder", plan.folder)
    Path(plan.folder).mkdir(parents=True, exist_ok=True)
    write_resolved_config(plan.run_config.experiment,
                          os.path.join(plan.folder, "resolved_config.json"))
    workflow.run_nsga2_optimization(
        plan.run_config, pop_size=plan.pop_size, n_generations=plan.generations,
        n_procs=workers, seed=plan.seed, termination_config=plan.termination,
        delete_generation_folders=True, run_postprocessing=True,
    )
    _announce("Campaign folder", plan.folder)
    return _campaign_result(plan.folder)


def _campaign_result(folder) -> RunResult:
    return RunResult(
        folder=_absolute(folder),
        results_folder=_absolute(folder) / "results",
        workflow=NSGA2,
    )


def _is_campaign(path: Path) -> bool:
    return any((path / "results" / name).is_file() for name in _CAMPAIGN_MARKERS)


def _campaign_state(folder) -> dict:
    """The committed checkpoint of a campaign a continuation reads."""
    state_path = Path(folder) / "results" / "checkpoint_state.json"
    if not state_path.is_file():
        raise ConfigurationError(
            f"{folder} has no committed NSGA-II checkpoint ({state_path})."
        )
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"Unreadable NSGA-II checkpoint {state_path}: {exc}") from exc


def _check_campaign_resume(folder) -> dict:
    """Check every continuation refusal before writing and return the committed state."""
    _campaign_state(folder)
    return _nsga2_workflow().preflight_nsga2_resume(folder)


def _resume_campaign(folder, *, workers) -> RunResult:
    workers = 1 if workers is None else _positive_int("workers", workers)
    _check_campaign_resume(folder)
    workflow = _nsga2_workflow()
    _limit_worker_threads()
    _announce("Campaign folder", folder)
    workflow.resume_nsga2_campaign(folder, n_procs=workers)
    _announce("Campaign folder", folder)
    return _campaign_result(folder)


# --------------------------------------------------------------------------
# Continue and report a saved run
# --------------------------------------------------------------------------

def saved_workflow(folder, label: str = "Folder") -> str:
    """``"steps"``, ``"css"`` or ``"nsga2"``: what the saved run in ``folder`` is."""
    path = Path(folder)
    if not path.is_dir():
        raise ConfigurationError(f"{label} `{path}` does not exist.")
    if _is_campaign(path):
        return NSGA2
    if any((path / name).is_file() for name in _CAMPAIGN_MARKERS):
        raise ConfigurationError(
            f"`{path}` is the results/ folder of an NSGA-II campaign; give the campaign "
            f"folder `{path.parent}`."
        )
    from pydac.utils.run.resume import saved_run

    return saved_run(path, label)[2]


@_figures_stay_on_disk
def resume(folder, *, settings=None, workers: int | None = None) -> RunResult:
    """Continue or replay the saved run in ``folder`` in place.

    Steps/CSS: ``settings`` may change only the operational settings of
    ``RESUMABLE_SETTINGS`` (mapping or ``["path=value"]``). NSGA-II: the campaign
    continues exactly as defined; only ``workers`` may be chosen.
    """
    workflow = saved_workflow(folder, "--resume folder")
    if workflow == NSGA2:
        if settings:
            raise ConfigurationError(
                "An NSGA-II campaign continues exactly as it was defined: settings are "
                "not accepted (only workers)."
            )
        return _resume_campaign(folder, workers=workers)
    if workers is not None:
        raise ConfigurationError("workers applies only to NSGA-II campaigns.")
    plan = _plan_resume(folder, workflow, settings)
    return _run_resume(plan, workflow, finish=True)


def _plan_resume(folder, workflow: str, settings):
    """Validate a step/CSS resume completely; return its plan. Writes nothing."""
    from pydac.utils.run.resume import resolve_resume

    plan = resolve_resume(folder, workflow=workflow, settings=_settings(settings))
    _preflight(_validator(workflow), plan.config)
    return plan


def _run_resume(plan, workflow: str, *, finish: bool):
    print(plan.summary)
    if workflow == STEPS:
        return _solve_steps(plan.config, plan.base_folder)
    result, results = _solve_css(plan.config, plan.base_folder)
    if not finish:
        return result, results
    _finish_css(plan.config, result, results)
    return result


@_figures_stay_on_disk
def report(folder, *, cycle: int | None = None, fmt: str | None = None,
           snapshot_times=None) -> RunResult:
    """Regenerate the report of a saved CSS run or NSGA-II campaign.

    CSS: the CSS plots (``plots_css/``), KPI figures and summary and the O2 audit of
    ``cycle`` (default: the CSS cycle, else the last completed one), whatever the saved
    output flags say; nothing is solved, pruned or rewritten. ``fmt`` defaults to the
    saved ``execution.plot_format``. With ``snapshot_times`` (as in ``run_css``) the step
    plots of ``cycle`` are drawn again too, with those times (2D maps and 1D profiles): each step
    is reloaded from its saved solution, not solved, which needs the ``sol.pkl`` files of
    the cycle and the final state of the cycle before it. NSGA-II: the Pareto plots and
    CSV (PDF figures; ``cycle``, ``fmt`` and ``snapshot_times`` do not apply).
    """
    return _execute_report(_plan_report(folder, cycle=cycle, fmt=fmt,
                                        snapshot_times=snapshot_times))


@dataclass(frozen=True)
class _ReportPlan:
    workflow: str
    folder: Path
    results: Path
    cycle: int | None = None
    css_reached: bool | None = None
    fmt: str | None = None
    case: ExperimentConfig | None = None
    snapshot_times: dict | None = None


def _plan_report(folder, *, cycle=None, fmt=None, workflow: str | None = None,
                 snapshot_times=None) -> _ReportPlan:
    """Validate a report request completely; writes nothing."""
    found = saved_workflow(folder, "--report folder")
    if workflow is not None and found != workflow and found != STEPS:
        raise ConfigurationError(
            f"`{folder}` holds a {found} run; report it with pydac {found} --report {folder}."
        )
    if found == NSGA2:
        if snapshot_times is not None:
            raise ConfigurationError("snapshot_times applies to CSS reports, not to NSGA-II campaigns.")
        return _plan_campaign_report(folder, cycle=cycle, fmt=fmt)
    if found == STEPS:
        raise ConfigurationError(
            f"`{folder}` holds a step run, which has no report; replay it instead: "
            f"pydac steps --resume {folder} --set execution.solve=false "
            "--set execution.plot_profiles=true."
        )
    if fmt is not None and (not isinstance(fmt, str) or not fmt.strip()):
        raise ConfigurationError(f"fmt must be a nonempty format name such as 'png', got {fmt!r}.")
    from pydac.utils.run.resume import completed_cycles, saved_run

    base, results, _ = saved_run(folder, "--report folder")
    available = completed_cycles(results)
    if not available:
        raise ConfigurationError(f"The CSS run in `{base}` has no completed cycle to report.")
    if cycle is None:
        # The loop stops at CSS, so the last completed cycle is the CSS cycle when reached.
        cycle = available[-1]
    elif isinstance(cycle, bool) or not isinstance(cycle, int) or cycle not in available:
        raise ConfigurationError(
            f"Cycle {cycle!r} is not a completed cycle of `{base}`; available: "
            f"{', '.join(str(index) for index in available)}."
        )
    if fmt is None:
        saved = json.loads((results / "resolved_config.json").read_text(encoding="utf-8"))
        fmt = saved["execution"]["plot_format"]
    diagnostics = results / f"cycle_{cycle}" / "css_diagnostics.json"
    reached = diagnostics.is_file() and bool(
        json.loads(diagnostics.read_text(encoding="utf-8")).get("reached")
    )
    case = times = None
    if snapshot_times is not None:
        from pydac.utils.tvsa.cycle_steps import build_cyclic_cycle_steps

        # The saved case, drawing in the report's format.
        case = load_case(results / "resolved_config.json",
                         settings={"execution.plot_format": fmt})
        steps = build_cyclic_cycle_steps(cycle, cycle_steps=case.execution.cycle_steps)
        times = _snapshot_times(case, CSS, snapshot_times, steps=steps)
        _require_saved_cycle(base, results, cycle, steps, case.execution.cycle_steps)
    return _ReportPlan(CSS, base, results, cycle, reached, fmt, case, times)


def _require_saved_cycle(base, results, cycle: int, steps, cycle_steps) -> None:
    """The step plots of ``cycle`` are drawn from its saved solutions, starting from the
    final state of the cycle before it: refuse before writing if any is gone."""
    from pydac.utils.tvsa.cycle_steps import build_repeating_cycle_steps

    needed = [results / f"cycle_{cycle}" / step / "sol.pkl" for step in steps]
    if cycle > 1:
        last = build_repeating_cycle_steps(cycle_steps=cycle_steps)[-1]
        needed.append(results / f"cycle_{cycle - 1}" / last / "sol.pkl")
    missing = [path for path in needed if not path.is_file()]
    if missing:
        raise ConfigurationError(
            f"The step plots of cycle {cycle} of `{base}` cannot be drawn again: "
            f"`{missing[0]}` is missing. They need the saved solutions of that cycle and the "
            "final state of the cycle before it (a run with execution.keep_only_final_cycle "
            "deletes the earlier cycles)."
        )


def _plan_campaign_report(folder, *, cycle, fmt) -> _ReportPlan:
    if cycle is not None:
        raise ConfigurationError("cycle applies only to CSS runs, not to NSGA-II campaigns.")
    if fmt is not None:
        raise ConfigurationError("NSGA-II figures are written as PDF; fmt applies to CSS reports.")
    _nsga2_workflow()
    results = Path(folder) / "results"
    if not (results / "last_pareto_front.npz").is_file():
        raise ConfigurationError(
            f"The NSGA-II campaign in `{folder}` has no saved Pareto front "
            f"({results / 'last_pareto_front.npz'})."
        )
    return _ReportPlan(NSGA2, Path(folder), results)


def _execute_report(plan: _ReportPlan) -> RunResult:
    if plan.workflow == NSGA2:
        workflow = _nsga2_workflow()
        _announce("Campaign folder", plan.folder)
        workflow.run_pareto_postprocessing(str(plan.results))
        _announce("Campaign folder", plan.folder)
        return _campaign_result(plan.folder)
    _announce("Run folder", plan.folder)
    _css_outputs(str(plan.results), plan.cycle, css_reached=plan.css_reached, fmt=plan.fmt,
                 profiles=True, kpis=True)
    if plan.snapshot_times:
        from pydac.workflows.orchestrator_TVSA_cycle_css import replot_cycle_profiles

        with _snapshot_overrides(plan.snapshot_times):
            replot_cycle_profiles(experiment_config=plan.case,
                                  results_base_folder=str(plan.results), cycle_idx=plan.cycle)
    _announce("Run folder", plan.folder)
    return RunResult(
        folder=_absolute(plan.folder),
        results_folder=_absolute(plan.results),
        workflow=CSS,
        css_reached=plan.css_reached,
        cycle=plan.cycle,
    )


__all__ = [
    "RunResult",
    "export_case",
    "load_case",
    "report",
    "resume",
    "run_css",
    "run_nsga2",
    "run_steps",
]
