---
title: "Quick start"
toc: true
toc-depth: 3
embed-resources: true
---

> **New to pyDAC? Do the [tutorials](../../examples/README.md) first, starting with
> [01 First steps](../../examples/notebooks/01_first_steps.ipynb).** They teach pyDAC step by
> step on real runs; this guide is the reference behind them and goes into more technical detail.

You can drive pyDAC in three ways: from the terminal with the `pydac` command, from a
Python script, and from a notebook. All three go through the same layer, so one case gives
the same results and the same run folder whichever way you start it. This page runs one
small case each way, then lists the Python API.

It assumes an installed and active environment ([Installation](01_installation.md)).

Each page of this guide starts from scratch: run its commands in a new, empty folder. pyDAC
does not replace what is already there. Exporting a case to an existing file is refused
(exit code `2`; `--overwrite-case` replaces it), and so is a fresh run into a run folder that
already holds a run (`--overwrite` replaces it, `--resume` continues it).

## Words used on this page

- A **case** is the complete set of inputs of one simulation.
- The **built-in defaults** are the values a field takes when nothing else sets it.
- A **setting** changes one field: a dotted path and a value, such as `grid.nz=25`.
- A **preset** is a JSON file with some fields; the others take the built-in defaults.
- A **frozen case** is a complete case saved as JSON; it runs exactly as saved.
- A **run folder** holds everything one run writes. Its name ends with the start of the
  case **fingerprint**, a hash of the complete case.

## From the terminal

Write the built-in case to a file, with a coarse 1D grid and at most three cycles:

```bash
pydac validate --set grid.nz=25 --set grid.nr=1 --set execution.max_cycles=3 --set execution.require_css=false --export-case my_case.json
```

`my_case.json` holds every input of the simulation, including fields computed from others
(the bed length from its volume, for example). To change it, prefer settings over editing
the file, so those fields are computed again ([Cases and settings](03_cases.md)). Run it
exactly as saved. A CSS run repeats the cycle until cyclic steady state (CSS); this one
runs at most three cycles, and this coarse case reaches CSS at cycle 3:

```bash
pydac css --case my_case.json
```

The run takes one to two minutes and writes
`simulation_results/cyclic_mode/packed_bed_XXXXXXXX/`. Its last output line names that run
folder.

Once per pressurization step the run prints `RuntimeWarning: H2O GAB isotherm evaluated at
relative humidity 0.90…, above the fitted data (RH <= 0.9)`. This is expected with the default
case: pressurization compresses the humid gas at the closed end of the bed, where the relative
humidity briefly reaches up to about 0.98. Nothing needs to be changed or silenced; the
warning only says that the H2O loading there extrapolates the GAB fit
([Governing equations](../manual/04_governing_equations.md)).

### Check a case without running it

`pydac validate` resolves a case and stops before solving. `--show-settings`,
`--dry-run`, `--print-resolved` and `--export-case` also exist on `pydac steps`,
`pydac css` and `pydac nsga2`; `--validate` and `--write-resolved` exist only on
`pydac validate`:

| Flag | Effect |
|---|---|
| `--show-settings` | list every setting with its type, built-in default and constraints |
| `--validate` | check the case and report "Configuration valid." |
| `--dry-run` | check the case; on `pydac steps` and `pydac css`, also print the run folder it would write |
| `--print-resolved` | print the complete resolved case as JSON |
| `--write-resolved PATH` | save the complete resolved case to `PATH` (and its fingerprint next to it) |
| `--export-case PATH` | save the case as a frozen case; `--overwrite-case` replaces an existing file |

A valid case exits with code `0`; an invalid one prints one line on stderr and exits with
code `2`. An unknown option also exits with code `2`, after printing the usage.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | failure after the run started (a traceback is printed) |
| `2` | invalid command-line arguments, configuration error or refused request; no run starts and no run folder is written |
| `3` | `pydac css` stopped at `execution.max_cycles` without reaching CSS while `execution.require_css` is true |

## From a Python script

Save this as `my_run.py`:

```python
# my_run.py
import pydac


def main():
    case = pydac.load_case("my_case.json", settings={"execution.max_cycles": 4})
    result = pydac.run_css(case)
    print(result.folder, result.css_reached, result.cycle)

    kpis = pydac.CycleKPIs(pydac.CycleDataLoader(result.folder).load())
    print(f"CO2 recovery: {100 * kpis.recovery('CO2'):.1f} %")


if __name__ == "__main__":
    main()
```

Run it in the environment:

```bash
python my_run.py
```

`load_case` with a file and `settings` builds a new case: the file, then each setting on
top. Because `execution.max_cycles` differs, this is a different case from the one the
terminal ran, with its own run folder. Keep the `if __name__ == "__main__":` guard: an
NSGA-II campaign with several workers starts new Python processes, which import your
script again.

## From a notebook

Select the kernel of your pyDAC environment ([Use pyDAC in
Jupyter](01_installation.md#use-pydac-in-jupyter)) and run the same calls in a cell. This
cell runs the saved case as a step run (startup, then each step of the cycle once) and
lists what it wrote:

```python
import pydac

case = pydac.load_case("my_case.json", exact=True)
result = pydac.run_steps(case)
print(result.workflow, result.folder.name)
print(sorted(p.name for p in result.results_folder.iterdir()))
```

`exact=True` runs the frozen case as saved, like `--case`. Relative paths, including the
`simulation_results` folder, are taken from the notebook's working folder, normally the
folder of the notebook file. Restart the kernel after you update pyDAC.

## The same case from both sides

A setting given with `--set` and the same setting given to `load_case` produce the same
case, down to the byte:

```bash
pydac validate --config my_case.json --set steps.s_des.flow.steam_mass_flow_kg_s=0.002 --export-case steam_cli.json
```

```python
import pydac

case = pydac.load_case("my_case.json", settings={"steps.s_des.flow.steam_mass_flow_kg_s": 0.002})
pydac.export_case(case, "steam_py.json")
```

```bash
cmp steam_cli.json steam_py.json
```

`cmp` prints nothing: the files are identical. Both cases therefore have the same
fingerprint and the same run folder, and running the second one is refused while the
first one's results exist.

## The Python API

The supported API is the set of names in `pydac.__all__`. Call them through the package
(`pydac.run_css(...)`).

### Functions

| Function | Returns | What it does |
|---|---|---|
| `load_case(path=None, *, settings=None, exact=False)` | case | Builds a case: built-in defaults, then the JSON file `path` (a preset or a complete case), then `settings`. `settings` is a mapping (`{"grid.nz": 25}`) or a list of `"path=value"` strings as `--set` takes them. `exact=True` loads a frozen case as saved and refuses `settings`. |
| `export_case(case, path, *, overwrite=False)` | `Path` | Saves `case` as a frozen case. Refuses an existing file unless `overwrite=True`. |
| `export_settings(settings, path, *, base=None, overwrite=False)` | `Path` | Saves `settings` (on top of the optional preset `base`) as a preset: only the fields they state, so `load_case(path)` gives the case `load_case(base, settings=settings)` gives. Refuses invalid settings before writing, and an existing file unless `overwrite=True`. |
| `run_steps(case, *, overwrite=False, snapshot_times=None)` | `RunResult` | Startup, then each step of `execution.cycle_steps` once. |
| `run_css(case, *, overwrite=False, snapshot_times=None)` | `RunResult` | Cycles until CSS or `execution.max_cycles`, then the KPI outputs the case asks for. |
| `run_nsga2(case, bounds, *, pop_size=100, generations=100, workers=1, seed=1, ...)` | `RunResult` | An NSGA-II campaign over `bounds` (a mapping or a JSON file of `variable: [lower, upper]`). The other options mirror the `pydac nsga2` flags. |
| `resume(folder, *, settings=None, workers=None)` | `RunResult` | Continues or replays a saved run in place. For a step or CSS run, `settings` may change only operational settings; a campaign continues exactly as defined, and only `workers` may change. |
| `report(folder, *, cycle=None, fmt=None, snapshot_times=None)` | `RunResult` | Regenerates the CSS plots (`plots_css/`) and KPI files of a saved CSS run (with `snapshot_times`, also the step plots of that cycle) (default cycle: the CSS cycle, else the last) or the Pareto files of a campaign. Solves nothing; reads the step KPIs the run saved (`kpis.pkl`), does not recompute them. |
| `get_started(folder)` | `Path` | Copies the README, licence, citation, this guide and the tutorials into `folder`, a new or empty folder, as `pydac get-started` does ([Get the tutorials](01_installation.md#get-the-tutorials-and-this-guide)). |

`overwrite=True` deletes this case's existing run folder, and only that folder, before a
fresh run. `snapshot_times` picks the times of the step plots (2D maps and 1D profiles); it
changes only the figures ([Times of the step plots](04_running.md#times-of-the-step-plots)).

### Classes

| Name | Use |
|---|---|
| `RunResult` | What a run returns (below). |
| `CycleDataLoader(folder)` | `.load()` reads one cycle of a saved CSS run. |
| `CycleKPIs(data)` | KPIs of that cycle: `recovery`, `productivity`, `purity`, `equivalent_work`, ... |
| `ExperimentConfig` | The type of a case, for type hints and `isinstance` only. Obtain cases with `load_case`; its constructor and inner records are not part of the API. |
| `ConfigurationError` | Raised for an invalid case, file or option. |
| `RunRefused` | Subclass of `ConfigurationError`: a valid request that the saved run refuses. |

`RunResult` is read-only:

| Field | Meaning |
|---|---|
| `folder` | absolute path of the run folder (campaign folder for NSGA-II); give it to `resume` and `report` |
| `results_folder` | the folder inside it with the results: `1D_mode` or `2D_mode`, `results` for a campaign |
| `workflow` | `"steps"`, `"css"` or `"nsga2"` |
| `css_reached` | `True` or `False` for a CSS run, `None` otherwise |
| `cycle` | for a CSS run, the CSS cycle or the last completed one; `None` otherwise |

### Errors

The API never exits Python: it returns or raises.

- **Before anything is written**, an invalid request raises `ConfigurationError`: a bad
  setting or file, a missing input, a damaged campaign checkpoint, or something other than
  a case passed to a `run_` function. A valid request that the saved state refuses raises
  `RunRefused`: a run folder that already holds a run, a campaign that is complete or was
  defined differently, a CSS run that already reached CSS, a run written by another pyDAC
  version.
- **After the solve has started**, errors propagate unchanged; the files written so far
  stay on disk.
- **Not reaching CSS is not an error** for `run_css`: it returns `css_reached=False`. Only
  the `pydac css` command turns it into exit code `3`, when `execution.require_css` is
  true.

```python
import pydac

try:
    pydac.run_css(pydac.load_case("my_case.json", exact=True))
except pydac.RunRefused as error:
    print(error)
```

This prints why the run was refused (the terminal already ran this case) and how to
continue, replace or rename it.

## What is supported

| Level | What | Promise |
|---|---|---|
| Command | `pydac steps`, `pydac css`, `pydac nsga2`, `pydac validate`, `pydac get-started` and their flags | documented; changes announced in the release notes |
| API | the names in `pydac.__all__` | documented; changes announced in the release notes |
| Internal | every other module under `pydac` | shipped and importable, no stability promise |

The user guide shows only commands and `pydac.__all__` names. Internal functions can
change in any release, and some write their results to other places than the API does.

## Related pages

- [Installation](01_installation.md)
- [README](../../README.md)
