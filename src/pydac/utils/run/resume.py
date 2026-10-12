"""Continue or replay a saved step/CSS run in its own folder (``--resume``).

A result folder is named by the fingerprint of the complete document, so a
run started again with a changed execution flag lands in a new folder. The
explicit ``--resume <folder>`` instead reuses the saved folder and its saved
``resolved_config.json``: the physics, grid, solver and cycle stay exactly
those of the saved run, and only the operational settings listed below may
change.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from pydac.config.errors import ConfigurationError, RunRefused
from pydac.config.loader import load_experiment
from pydac.config.schema import ExperimentConfig
from pydac.config.serialization import experiment_to_document
from pydac.utils.provenance import require_same_version

SPATIAL_FOLDERS = ("1D_mode", "2D_mode")

_SHARED_SETTINGS = (
    "execution.plot_profiles",
    "execution.plot_format",
    "execution.print_to_terminal",
    "execution.print_cycle_build",
)

#: Document paths a resumed run may change, per workflow. Everything else is
#: part of the saved run's identity and must stay as saved.
RESUMABLE_SETTINGS: dict[str, tuple[str, ...]] = {
    "steps": _SHARED_SETTINGS + (
        "execution.solve",
        "execution.step_options.*.solve",
        "execution.step_options.*.plot_profiles",
    ),
    "css": _SHARED_SETTINGS + (
        "execution.max_cycles",
        "execution.css_tolerance",
        "execution.require_css",
        "execution.plot_final_cycle_profiles",
        "execution.plot_cyclic_profiles",
        "execution.generate_kpi_outputs",
        "execution.keep_only_final_cycle",
    ),
}


@dataclass(frozen=True)
class ResumePlan:
    """What a resumed run executes and where it writes."""

    config: ExperimentConfig
    base_folder: str
    results_folder: str
    start_cycle: int | None
    summary: str


def _changed_paths(saved, requested, prefix=()):
    """Leaf paths whose value differs between two documents."""
    if isinstance(saved, Mapping) or isinstance(requested, Mapping):
        saved = saved if isinstance(saved, Mapping) else {}
        requested = requested if isinstance(requested, Mapping) else {}
        for key in sorted(set(saved) | set(requested), key=str):
            yield from _changed_paths(saved.get(key), requested.get(key), prefix + (str(key),))
    elif saved != requested:
        yield ".".join(prefix)


def _locate(folder, label="--resume folder") -> tuple[Path, Path]:
    """Return ``(base_folder, spatial results folder)`` for a saved run."""
    path = Path(folder)
    if not path.is_dir():
        raise ConfigurationError(f"{label} `{path}` does not exist.")
    if path.name in SPATIAL_FOLDERS and (path / "resolved_config.json").is_file():
        return path.parent, path
    found = [path / name for name in SPATIAL_FOLDERS if (path / name / "resolved_config.json").is_file()]
    if len(found) != 1:
        raise ConfigurationError(
            f"{label} `{path}` is not a saved pyDAC run: expected exactly one "
            "1D_mode/ or 2D_mode/ subfolder holding resolved_config.json."
        )
    return path, found[0]


def _saved_workflows(results: Path) -> set[str]:
    workflows = set()
    for record in sorted((results / "provenance").glob("*.json")):
        try:
            payload = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        workflow = payload.get("execution", {}).get("workflow")
        if payload.get("activity") == "run" and workflow:
            workflows.add(workflow)
    return workflows


def _workflows_of(results: Path) -> set[str]:
    saved = _saved_workflows(results)
    if not saved:
        # Runs saved before provenance recorded the workflow.
        saved = {"css" if any(results.glob("cycle_*")) else "steps"}
    return saved


def saved_run(folder, label="--resume folder") -> tuple[Path, Path, str]:
    """``(base_folder, spatial results folder, workflow)`` of a saved step/CSS run."""
    base, results = _locate(folder, label)
    saved = _workflows_of(results)
    if len(saved) != 1:
        raise ConfigurationError(
            f"{label} `{base}` holds runs of several workflows ({', '.join(sorted(saved))})."
        )
    return base, results, saved.pop()


def completed_cycles(results) -> list[int]:
    """Indices of the cycles of a CSS run that saved their metadata."""
    return _completed_cycles(Path(results))


def _check_workflow(results: Path, workflow: str) -> None:
    saved = _workflows_of(results)
    if saved != {workflow}:
        raise ConfigurationError(
            f"--resume folder `{results.parent}` holds a {'/'.join(sorted(saved))} run; "
            f"it cannot be resumed as a {workflow} run."
        )


def _completed_cycles(results: Path) -> list[int]:
    cycles = []
    for folder in results.glob("cycle_*"):
        suffix = folder.name.split("_", 1)[1]
        if suffix.isdigit() and (folder / "cycle_metadata.json").is_file():
            cycles.append(int(suffix))
    return sorted(cycles)


def _step_plan(requested: ExperimentConfig, base: Path, results: Path) -> ResumePlan:
    """A step resume must reuse saved solutions of steps the run executes."""
    from pydac.utils.tvsa import build_cycle_steps
    from pydac.utils.tvsa.steps import normalize_step_token

    execution = requested.execution
    executed = build_cycle_steps(cycle_steps=execution.cycle_steps)
    options = {
        normalize_step_token(step): dict(values)
        for step, values in dict(execution.step_options).items()
    }
    not_executed = sorted(set(options) - set(executed))
    if not_executed:
        raise ConfigurationError(
            f"--resume step options name steps this run does not execute: {', '.join(not_executed)} "
            f"(executed: {', '.join(executed)})."
        )
    reused = [step for step in executed if options.get(step, {}).get("solve", execution.solve) is False]
    if not reused:
        raise ConfigurationError(
            "--resume on a step run reuses its saved steps: set execution.solve=false to "
            "replay them all, or give the steps to keep as "
            'execution.step_options={"STEP": {"solve": false}, ...}. To solve everything '
            "again, start a fresh run instead."
        )
    # Checked up front: a missing solution found mid-run would fail only after
    # the earlier re-solved steps had already overwritten their saved results.
    # A reused step after a re-solved one would reload results computed from
    # the old state of the step before it: reuse a leading block only.
    first_solved = next(index for index, step in enumerate([*executed, None]) if step not in reused)
    stale = [step for step in executed[first_solved:] if step in reused]
    if stale:
        raise ConfigurationError(
            f"--resume cannot reuse {stale[0]} after re-solving {executed[first_solved]}: its "
            "saved solution was computed from the old state. Reuse only a leading block of "
            f"steps ({', '.join(executed)}), or solve {stale[0]} again."
        )
    missing = [step for step in reused if not (results / step / "sol.pkl").is_file()]
    if missing:
        raise ConfigurationError(
            f"--resume cannot reuse {', '.join(missing)}: no saved solution in `{results}`."
        )
    if len(reused) < len(executed):
        # Re-solving some steps next to saved ones would mix two model versions.
        require_same_version(results, "run", f"The step run in `{base}`",
                             still_allowed="Reporting it or replaying all its steps "
                                           "(execution.solve=false)")
    what = (
        "replaying all saved steps"
        if len(reused) == len(executed)
        else f"reusing saved {', '.join(reused)}, solving {', '.join(s for s in executed if s not in reused)}"
    )
    return ResumePlan(requested, str(base), str(results), None, f"Resuming step run {base}: {what}.")


def resolve_resume(
    folder,
    *,
    workflow: str,
    settings: Mapping | Sequence[str] | None = None,
) -> ResumePlan:
    """Resolve a ``--resume`` request into the exact run it continues."""
    if workflow not in RESUMABLE_SETTINGS:
        raise ValueError(f"Unsupported resume workflow {workflow!r}.")
    base, results = _locate(folder)
    _check_workflow(results, workflow)
    if workflow == "css":
        # Before the saved case is loaded: a later version may not read it.
        require_same_version(results, "run", f"The CSS run in `{base}`")

    from pydac.inputs.case_io import load_case

    saved = load_case(results / "resolved_config.json")
    if settings and not isinstance(settings, Mapping):
        settings = list(settings)
    requested = load_experiment(saved, settings=settings) if settings else saved
    allowed = RESUMABLE_SETTINGS[workflow]
    changed = list(_changed_paths(experiment_to_document(saved), experiment_to_document(requested)))
    refused = [path for path in changed if not any(fnmatchcase(path, rule) for rule in allowed)]
    if refused:
        raise ConfigurationError(
            "--resume keeps the saved run's case; these settings cannot change: "
            f"{', '.join(refused)}. A resumed {workflow} run may change only: "
            f"{', '.join(allowed)}."
        )

    if workflow == "steps":
        return _step_plan(requested, base, results)

    completed = _completed_cycles(results)
    if not completed:
        raise ConfigurationError(
            f"--resume folder `{base}` has no completed cycle; start a fresh run instead."
        )
    last = completed[-1]
    diagnostics = results / f"cycle_{last}" / "css_diagnostics.json"
    if diagnostics.is_file() and json.loads(diagnostics.read_text(encoding="utf-8")).get("reached"):
        raise RunRefused(
            f"The CSS run in `{base}` already reached cyclic steady state at cycle {last}."
        )
    max_cycles = requested.execution.max_cycles
    if last >= max_cycles:
        raise ConfigurationError(
            f"The CSS run in `{base}` stopped at max_cycles={max_cycles} (cycle {last} "
            "completed); raise it to continue, e.g. --set execution.max_cycles="
            f"{last + max(1, max_cycles // 2)}."
        )
    start = last + 1
    requested = load_experiment(requested, settings=[f"execution.resume_from_cycle={start}"])
    return ResumePlan(
        requested,
        str(base),
        str(results),
        start,
        f"Resuming CSS run {base} at cycle {start} (up to max_cycles={max_cycles}).",
    )


__all__ = [
    "RESUMABLE_SETTINGS",
    "SPATIAL_FOLDERS",
    "ResumePlan",
    "completed_cycles",
    "resolve_resume",
    "saved_run",
]
