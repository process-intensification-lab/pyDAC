"""The ``pydac`` command: ``pydac steps|css|nsga2|validate|get-started`` and ``pydac --version``."""
# PYTHON_ARGCOMPLETE_OK

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : parse -> check the request (nothing written) -> run -> exit code.
# Owns     : the flags of each subcommand, which combinations are refused, and
#            the exit codes: 0 success, 2 configuration error or refusal
#            (one stderr line), 3 CSS not reached with require_css, 1 anything
#            that fails after the run started (traceback).
# Not here : what a run does (api.py); the shared case flags (config/terminal.py).
# Contract : input resolution, reporting, exit outcomes, output folders and
#            overwrite rules.
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydac.config.errors import ConfigurationError, RunRefused
from pydac.utils.version import __version__

# NSGA-II flags that define a campaign: a continuation reads them from the checkpoint.
_CAMPAIGN_FLAGS = (
    "bounds", "pop_size", "generations", "seed", "purity_min", "oxygen_ppm_max",
    "eq_work_max", "termination", "ftol", "period", "n_skip", "max_evaluations",
)


@dataclass(frozen=True)
class LauncherSettings:
    """Settings a checkout launcher (``runners/``) explicitly passes to ``pydac``.

    ``pydac`` never reads a runner file: a launcher passes its constants here.
    ``config_path``/``resume_path`` are defaults that ``--config``/``--resume``
    override; ``development_settings`` apply between the config and ``--set``;
    ``case_path`` is exclusive with ``--case``; ``overwrite`` means ``--overwrite``.
    """

    case_path: str | None = None
    config_path: str | None = None
    resume_path: str | None = None
    development_settings: Mapping = field(default_factory=dict)
    overwrite: bool = False


_NO_LAUNCHER = LauncherSettings()


class _Stop(Exception):
    """A request refused before anything was written: exit 2."""

    def __init__(self, error: ConfigurationError):
        super().__init__(str(error))
        self.error = error


def environment_warning(prefix=None, base_prefix=None, environ=None) -> str | None:
    """The one-line warning when pydac runs outside a dedicated environment.

    Dedicated: a virtual environment (``sys.prefix != sys.base_prefix``) or a
    conda environment other than base (``conda-meta/`` without ``condabin/``).
    ``PYDAC_NO_ENV_WARNING=1`` silences it; it never blocks a run.
    """
    prefix = Path(sys.prefix if prefix is None else prefix)
    base_prefix = Path(sys.base_prefix if base_prefix is None else base_prefix)
    environ = os.environ if environ is None else environ
    if environ.get("PYDAC_NO_ENV_WARNING") == "1" or prefix != base_prefix:
        return None
    if (prefix / "conda-meta").is_dir() and not (prefix / "condabin").is_dir():
        return None
    return (
        f"Warning: pydac is running outside a dedicated environment ({prefix}). pyDAC pins "
        "some dependencies, and pip replaces installed versions without asking: install it "
        "in its own venv or conda environment (README, Installation). "
        "PYDAC_NO_ENV_WARNING=1 silences this."
    )


def _error(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _non_negative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def _add_overwrite(parser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete this case's existing run folder (and only it) before a fresh run.",
    )


def _add_snapshot_times(parser) -> None:
    parser.add_argument(
        "--snapshot-times", metavar="JSON",
        help='Times of the step plots, 2D maps and 1D profiles (figures only; with --report RUN the step '
             'plots of the cycle are drawn again from the saved run), e.g. '
             '\'{"heat": {"T_bed": {"times": [1, 10, 30, 60, 90, 150], "unit": "min"}}}\'.',
    )


def _snapshot_times(args):
    """The --snapshot-times JSON of a fresh run or a report, parsed (checked by the API), or None."""
    text = getattr(args, "snapshot_times", None)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"--snapshot-times is not valid JSON: {exc}.") from exc


def build_parser() -> argparse.ArgumentParser:
    from pydac.config.cli import add_validate_arguments
    from pydac.config.terminal import FlexibleArgumentParser, add_experiment_arguments

    parser = FlexibleArgumentParser(
        prog="pydac",
        description="Simulate adsorption-based direct air capture processes.",
    )
    parser.add_argument("--version", action="version", version=f"pydac {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    steps = commands.add_parser(
        "steps", help="Run a step sequence (startup + the listed steps).",
        description="Run a step sequence: startup, then execution.cycle_steps.",
    )
    add_experiment_arguments(steps, resumable=True)
    _add_overwrite(steps)
    _add_snapshot_times(steps)

    css = commands.add_parser(
        "css", help="Run cycles to cyclic steady state, or report a saved run.",
        description="Run cycles until cyclic steady state or execution.max_cycles.",
    )
    add_experiment_arguments(css, resumable=True)
    _add_overwrite(css)
    _add_snapshot_times(css)
    css.add_argument("--report", metavar="RUN",
                     help="Regenerate plots, KPIs and the O2 audit of the saved run RUN.")
    css.add_argument("--cycle", type=_positive_int, metavar="N",
                     help="With --report: the cycle to report (default: CSS or last cycle).")

    nsga2 = commands.add_parser(
        "nsga2", help="Run, continue or report a local NSGA-II campaign.",
        description="Run a local NSGA-II optimization with explicit search bounds.",
    )
    add_experiment_arguments(nsga2)
    nsga2.add_argument("--bounds", metavar="PATH",
                       help="JSON object mapping decision variables to [lower, upper].")
    nsga2.add_argument("--pop-size", type=_positive_int, default=None,
                       help="Population size (default 100).")
    nsga2.add_argument("--generations", type=_positive_int, default=None,
                       help="Generations, or the hard cap with robust_pareto (default 100).")
    nsga2.add_argument("--workers", type=_positive_int, default=None,
                       help="Local worker processes (default 1).")
    nsga2.add_argument("--seed", type=int, default=None, help="Random seed (default 1).")
    nsga2.add_argument("--purity-min", type=float, default=None,
                       help="Constraint: minimum CO2 mole fraction of the recovered product, "
                            "water included, for example 0.95 (default: off).")
    nsga2.add_argument("--oxygen-ppm-max", type=float, default=None,
                       help="Constraint: maximum O2 in the recovered product, ppm (molar) "
                            "(default: off).")
    nsga2.add_argument("--eq-work-max", type=float, default=None,
                       help="Constraint: maximum equivalent work, MJ per kg of recovered CO2 "
                            "(default: off).")
    nsga2.add_argument("--steam-condensation", action="store_true",
                       help="Constraint: at the end of heating the coldest bed point must be "
                            "at or above the water saturation temperature at the heating "
                            "pressure (steam-assisted cycles; needs H2O).")
    nsga2.add_argument("--termination", choices=("fixed_generations", "robust_pareto"),
                       default=None,
                       help="Stop after --generations, or when the feasible front is stable "
                            "(default fixed_generations).")
    nsga2.add_argument("--ftol", type=float, default=None,
                       help="robust_pareto front tolerance (default 5e-3).")
    nsga2.add_argument("--period", type=_positive_int, default=None,
                       help="robust_pareto window in generations (default 10).")
    nsga2.add_argument("--n-skip", type=_non_negative_int, default=None,
                       help="robust_pareto generations skipped between checks (default 0).")
    nsga2.add_argument("--max-evaluations", type=_positive_int, default=None,
                       help="Optional cap on candidate evaluations.")
    nsga2.add_argument("--resume", default=None, metavar="CAMPAIGN",
                       help="Continue the campaign in CAMPAIGN from its last committed "
                            "generation (only --workers and --dry-run apply).")
    nsga2.add_argument("--report", metavar="CAMPAIGN",
                       help="Regenerate the Pareto plots and CSV of the campaign CAMPAIGN.")

    validate = commands.add_parser(
        "validate", help="Validate, print or export a case without solving it.",
        description="Validate and resolve a pyDAC case without solving it.",
    )
    add_experiment_arguments(validate)
    add_validate_arguments(validate)

    get_started = commands.add_parser(
        "get-started", help="Copy the tutorials, the user guide and the README into a new folder.",
        description="Copy README, LICENSE, CITATION.cff, the user guide (docs/guide) and the "
                    "tutorials (examples) into FOLDER, in the repository layout.",
    )
    get_started.add_argument("folder", metavar="FOLDER", help="A new or empty folder.")

    from pydac.config.terminal import enable_argcomplete

    enable_argcomplete(parser)
    return parser


# --------------------------------------------------------------------------
# Flag combinations (checked before anything is resolved or written)
# --------------------------------------------------------------------------

def _refuse(used: list[str], context: str) -> None:
    if used:
        raise ConfigurationError(f"{context}; remove {', '.join(used)}.")


def _report_format(args) -> str | None:
    """--report accepts only --set execution.plot_format=…; return that format."""
    from pydac.config.settings import resolve_settings

    settings = resolve_settings(args.settings)
    other = sorted(set(settings) - {"execution.plot_format"})
    if other:
        raise ConfigurationError(
            f"--report accepts only --set execution.plot_format=...; remove --set {', '.join(other)}."
        )
    return settings.get("execution.plot_format")


def _check_report_flags(args, campaign_flags=()) -> None:
    used = [flag for flag, on in (
        ("--resume", args.resume is not None), ("--case", _case_given(args)),
        ("--config", args.config is not None), ("--export-case", args.export_case is not None),
        ("--overwrite-case", args.overwrite_case),
        ("--overwrite", getattr(args, "overwrite", False)), ("--dry-run", args.dry_run),
        ("--print-resolved", args.print_resolved),
    ) if on]
    used += [_flag(name) for name in campaign_flags if getattr(args, name, None) is not None]
    if getattr(args, "steam_condensation", False):
        used.append("--steam-condensation")
    if getattr(args, "workers", None) is not None:
        used.append("--workers")
    _refuse(used, "--report only regenerates the report of a saved run")


def _case_given(args) -> bool:
    return args.case is not None or args.launcher.case_path is not None


def _check_resume_flags(args) -> None:
    used = [flag for flag, on in (
        ("--config", args.config is not None), ("--case", _case_given(args)),
        ("--export-case", args.export_case is not None),
        ("--overwrite", getattr(args, "overwrite", False)),
        ("--snapshot-times", getattr(args, "snapshot_times", None) is not None),
    ) if on]
    _refuse(used, "--resume uses the saved run's configuration")


def _check_fresh_flags(args) -> None:
    if args.overwrite and args.export_case is not None:
        raise ConfigurationError(
            "--overwrite replaces a run folder, not a case file; use --overwrite-case with "
            "--export-case."
        )
    if getattr(args, "snapshot_times", None) is not None:
        _refuse([flag for flag, on in (("--export-case", args.export_case is not None),
                                       ("--print-resolved", args.print_resolved)) if on],
                "--snapshot-times only changes the figures of a run")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def _preflight(check):
    """Run ``check`` (writes nothing); turn its configuration errors into exit 2."""
    try:
        return check()
    except ConfigurationError as exc:
        raise _Stop(exc) from exc


def _resolve(args):
    """The case of a fresh run, and the mode line the dry run prints."""
    from pydac.config.terminal import resolve_runner_experiment

    launcher = args.launcher
    return resolve_runner_experiment(
        args,
        case_path=launcher.case_path,
        development_settings=launcher.development_settings or None,
    )


def _finish_css(case, result, results) -> int:
    from pydac import api

    try:
        api._finish_css(case, result, results)
    except Exception as exc:  # the solved run stays valid
        traceback.print_exc()
        _error(f"Run completed; reporting failed: {exc}. "
               f"Retry with `pydac css --report {result.folder}`.")
        return 1
    if not result.css_reached and case.execution.require_css:
        _error(f"CSS not reached after cycle {result.cycle} (execution.require_css=true); "
               f"the last cycle is reported in {result.folder}.")
        return 3
    return 0


def _run_fresh(args, workflow: str) -> int:
    from pydac import api
    from pydac.config.terminal import (
        export_case_if_requested,
        print_mode_summary,
        print_resolved_document,
    )

    def check():
        _check_fresh_flags(args)
        case, mode = _resolve(args)
        if export_case_if_requested(args, case) is not None:
            return None
        if args.print_resolved:
            print_mode_summary(mode)
            print_resolved_document(case, dry_run=False)
            return None
        folder = api._check_fresh_run(case, workflow)
        times = api._snapshot_times(case, workflow, _snapshot_times(args))
        if args.dry_run:
            print_mode_summary(mode)
            if args.overwrite and api.occupied(folder):
                print(f"Dry run: --overwrite would delete {api._absolute(folder)}")
            elif not args.overwrite:
                api._refuse_occupied(folder)
            print(f"Run folder: {api._absolute(folder)}")
            print("Dry run validation passed: configuration is valid; simulation can start.")
            return None
        api._claim(folder, overwrite=args.overwrite)
        return case, folder, times

    planned = _preflight(check)
    if planned is None:
        return 0
    case, folder, times = planned
    with api._snapshot_overrides(times):
        if workflow == api.STEPS:
            api._solve_steps(case, folder)
            return 0
        result, results = api._solve_css(case, folder)
        return _finish_css(case, result, results)


def _run_resume(args, workflow: str) -> int:
    from pydac import api

    def check():
        _check_resume_flags(args)
        if args.launcher.development_settings:
            print("Note: DEVELOPMENT_SETTINGS are ignored by --resume; use --set for "
                  "resumable settings.")
        plan = api._plan_resume(args.resume, workflow, args.settings)
        if args.print_resolved:
            from pydac.config.serialization import experiment_to_document

            print(plan.summary)
            print(json.dumps(experiment_to_document(plan.config), indent=2, allow_nan=False))
            return None
        if args.dry_run:
            print(plan.summary)
            print("Dry run validation passed: the run can be resumed; no simulation started.")
            return None
        return plan

    plan = _preflight(check)
    if plan is None:
        return 0
    if workflow == api.STEPS:
        api._run_resume(plan, workflow, finish=True)
        return 0
    result, results = api._run_resume(plan, workflow, finish=False)
    return _finish_css(plan.config, result, results)


def _run_report(args, workflow: str) -> int:
    from pydac import api

    def check():
        _check_report_flags(args, _CAMPAIGN_FLAGS if workflow == api.NSGA2 else ())
        fmt = _report_format(args)
        if workflow == api.NSGA2 and fmt is not None:
            raise ConfigurationError("NSGA-II figures are written as PDF; remove --set.")
        return api._plan_report(args.report, cycle=getattr(args, "cycle", None), fmt=fmt,
                                workflow=workflow, snapshot_times=_snapshot_times(args))

    api._execute_report(_preflight(check))
    return 0


def _steps(args) -> int:
    if args.resume is not None:
        return _run_resume(args, "steps")
    return _run_fresh(args, "steps")


def _css(args) -> int:
    if args.report is not None:
        return _run_report(args, "css")
    if args.cycle is not None:
        raise _Stop(ConfigurationError("--cycle applies only with --report RUN."))
    if args.resume is not None:
        return _run_resume(args, "css")
    return _run_fresh(args, "css")


def _nsga2(args) -> int:
    from pydac import api

    if args.report is not None:
        return _run_report(args, "nsga2")
    if args.resume is not None:
        return _nsga2_resume(args)

    def check():
        from pydac.config.terminal import (
            export_case_if_requested,
            print_mode_summary,
            print_resolved_document,
        )

        case, mode = _resolve(args)
        if export_case_if_requested(args, case) is not None:
            return None
        if args.print_resolved:
            print_mode_summary(mode)
            print_resolved_document(case, dry_run=False)
            return None
        if not args.bounds:
            raise ConfigurationError(
                "--bounds is required to run an optimization; choose a search space for your case."
            )
        options = {name: getattr(args, name) for name in _CAMPAIGN_FLAGS if name != "bounds"}
        options = {name: value for name, value in options.items() if value is not None}
        plan = api._plan_campaign(
            case, args.bounds, workers=args.workers or 1,
            steam_condensation=args.steam_condensation,
            **{"pop_size": 100, "generations": 100, "seed": 1,
               "termination": "fixed_generations", "purity_min": None,
               "oxygen_ppm_max": None, "eq_work_max": None, "ftol": None, "period": None,
               "n_skip": None, "max_evaluations": None, **options},
        )
        if args.dry_run:
            print_mode_summary(mode)
            print(f"Campaign folder: {api._absolute(plan.folder)}")
            print("Dry run validation passed: case, bounds and campaign folder are valid; "
                  "no optimization started.")
            return None
        return plan

    plan = _preflight(check)
    if plan is None:
        return 0
    api._run_campaign(plan, workers=args.workers or 1)
    return 0


def _nsga2_resume(args) -> int:
    from pydac import api

    def check():
        used = [_flag(name) for name in _CAMPAIGN_FLAGS if getattr(args, name) is not None]
        used += [flag for flag, on in (
            ("--steam-condensation", args.steam_condensation),
            ("--config", args.config is not None), ("--case", _case_given(args)),
            ("--set", bool(args.settings)), ("--export-case", args.export_case is not None),
            ("--print-resolved", args.print_resolved),
        ) if on]
        _refuse(used, "--resume continues the campaign exactly as it was defined "
                      "(only --workers and --dry-run apply)")
        if api.saved_workflow(args.resume, "--resume folder") != api.NSGA2:
            raise ConfigurationError(f"`{args.resume}` is not an NSGA-II campaign.")
        state = api._check_campaign_resume(args.resume)
        if args.dry_run:
            optimizer = state["campaign_identity"]["definition"]["optimizer"]
            print(
                f"Dry run validation passed: campaign {args.resume} can be resumed after "
                f"generation {state['completed_generations']} of "
                f"{optimizer['total_generations']} ({optimizer['termination']['mode']}); "
                "no optimization started."
            )
            return None
        return state

    if _preflight(check) is None:
        return 0
    api._resume_campaign(args.resume, workers=args.workers)
    return 0


def _validate(args) -> int:
    from pydac.config.cli import run

    return run(args)


def _get_started(args) -> int:
    from pydac.starter_kit import get_started

    try:
        dest = get_started(args.folder)
    except ConfigurationError as error:
        raise _Stop(error) from error
    print(f"Starter kit copied to {dest}")
    print(f"Next: open {dest / 'examples' / 'README.md'}, or run "
          f"'pydac steps --config examples/cases/01_first_steps.json' from {dest}.")
    return 0


_COMMANDS = {"steps": _steps, "css": _css, "nsga2": _nsga2, "validate": _validate,
             "get-started": _get_started}


_PACKAGE_DIR = str(Path(__file__).resolve().parent)


def _format_warning(message, category, filename, lineno, line=None):
    """pyDAC's own warnings print as one plain line; others keep Python's format."""
    if issubclass(category, UserWarning) and str(filename).startswith(_PACKAGE_DIR):
        return f"Warning: {message}\n"
    return _PYTHON_FORMATWARNING(message, category, filename, lineno, line)


_PYTHON_FORMATWARNING = warnings.formatwarning


def main(argv: Sequence[str] | None = None, *, launcher: LauncherSettings | None = None) -> int:
    """Run ``pydac``; ``launcher`` carries the constants of a checkout runner."""
    # The command writes its figures to files and never opens a window; an
    # MPLBACKEND set by the user still wins.
    os.environ.setdefault("MPLBACKEND", "Agg")
    warnings.formatwarning = _format_warning
    args = build_parser().parse_args(argv)
    warning = environment_warning()
    if warning:
        _error(warning)
    args.launcher = launcher = launcher or _NO_LAUNCHER
    if getattr(args, "config", None) is None and launcher.config_path is not None:
        args.config = launcher.config_path
    if getattr(args, "resume", None) is None and launcher.resume_path is not None:
        args.resume = launcher.resume_path
    if launcher.overwrite and hasattr(args, "overwrite"):
        args.overwrite = True
    try:
        return _COMMANDS[args.command](args)
    except _Stop as stop:
        label = "Refused" if isinstance(stop.error, RunRefused) else "Configuration error"
        _error(f"{label}: {stop.error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LauncherSettings", "build_parser", "environment_warning", "main"]
