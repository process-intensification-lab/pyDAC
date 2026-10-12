---
title: "Running steps and cycles"
toc: true
toc-depth: 3
embed-resources: true
---

> **New to pyDAC? Do the [tutorials](../../examples/README.md) first, starting with
> [01 First steps](../../examples/notebooks/01_first_steps.ipynb).** They teach pyDAC step by
> step on real runs; this guide is the reference behind them and goes into more technical detail.

pyDAC runs a case in one of three ways: a step run (`pydac steps`), which solves startup
and then each listed step once; a CSS run (`pydac css`), which repeats the cycle until
cyclic steady state (CSS); and an NSGA-II campaign (`pydac nsga2`, on the
[optimization page](05_optimization.md)). This page covers the first two: which steps run, when
a step stops, and how to continue, replay, report or replace a run.

The examples use the small 1D case of the [quick start](02_quickstart.md), in a new, empty
folder:

```bash
pydac validate --set grid.nz=25 --set grid.nr=1 --set execution.max_cycles=3 --set execution.require_css=false --export-case my_case.json
```

## Which steps run

`execution.cycle_steps` lists the steps in order. The step names are `startup`, `ads`
(adsorption), `blow` (blowdown), `heat` (heating), `w_purge` (wall purge), `s_purge`
(steam purge), `s_des` (steam-assisted desorption), `w_des` (wall-heated desorption),
`cool` (cooling) and `press` (pressurization).

**Step runs.** With `execution.cycle_steps = null` (the default) a step run solves
`startup`, `blow`, `heat`, `s_des`, `cool`, `press`, `ads`. A list runs startup first, then
each listed step once; `startup` is added at the front if the list lacks it, so `["heat"]`
runs `startup` then `heat`, and `["startup"]` alone is valid.

**CSS runs.** The list is the repeating cycle: it must start with `ads` and must not contain
`startup`. The default is `ads`, `blow`, `heat`, `s_des`, `cool`, `press`. Cycle 1 runs
`startup` in place of its adsorption step; from cycle 2 on, the cycle starts with `ads`.

Step names are case-insensitive, and the long names are accepted too (`PRE-HEATING` is
`heat`). A plain `des` is refused: choose `s_des` or `w_des`.

Every step starts from the state the previous step left; the first step starts from the
initial values of `startup`: its temperature, pressure, gas composition and sorbent
loadings, which you can set, uniform over the bed. pyDAC 3.0.0 has no way to start a run
from your own spatial profile.

## When a step stops

Each step has a stop rule in `steps.STEP.termination`:

- `mode = "time"`: the step runs its `horizon.nominal_time_s`.
- `mode = "event"`: the step stops when `metric` crosses its threshold in `direction`,
  and at the latest after `horizon.event_max_time_s`.

The mode can be overridden, from the weakest to the strongest:

1. the step's own `steps.STEP.termination.mode`;
2. `execution.termination_mode`, for every step of the run at once;
3. the step options: `execution.step_options` in a step run, and
   `execution.cycle_options` (per cycle) in a CSS run.

A run-wide `execution.termination_mode` replaces the mode of every step without naming the
steps it changed. The run prints the mode each step actually uses before it starts, in its
setup table. For example, every step of a step run on time except blowdown:

```bash
pydac validate --config my_case.json --set execution.termination_mode=time --set 'execution.step_options={"blow":{"termination_mode":"event"}}' --validate
```

A CSS run refuses a top-level `execution.step_options` (exit code `2`): it takes the same
options per cycle, for example `execution.cycle_options={"2":{"step_options":{"blow":{"termination_mode":"event"}}}}`
for cycle 2.

Cooling (`cool`) also has a low-pressure safety stop
(`steps.cool.termination.stop_on_low_pressure`): it applies in every mode, including
`"time"`, and the setup table shows it as `+ low-pressure safety event`. Blowdown and
pressurization can also stop on a pressure plateau (`stop_on_pressure_plateau`), in
`"event"` mode only. The stop
rule of each step is described in [05 — Cycle steps](../manual/05_cycle_steps.md).

## A step run

```bash
pydac steps --case my_case.json
```

The run prints its run folder when it starts and when it ends. Each step has a folder named
after it, here under `simulation_results/step_mode/packed_bed_7a4930a5/1D_mode/`, with its
solution (`sol.pkl`) and its KPI table (`kpis.csv`). Step profile plots are written when
`execution.plot_profiles` is true (default false), in `execution.plot_format` (default
`pdf`; `png` for figures to show in a notebook).

### Replay and partial re-solve

`--resume RUN` reruns a saved step run in place. It must reuse at least one saved step.

To replay every step from its saved solution, for example to draw the profile plots
afterwards, set `execution.solve` to false:

```bash
pydac steps --resume simulation_results/step_mode/packed_bed_7a4930a5 --set execution.solve=false --set execution.plot_profiles=true
```

To keep the first steps and solve the rest again, mark the steps to keep with
`"solve": false` in `execution.step_options`. A resumed run starts from the settings saved
in the run folder, so after the replay above `execution.solve` is still false: set it back
to true, or every step is replayed again:

```bash
pydac steps --resume simulation_results/step_mode/packed_bed_7a4930a5 --set execution.solve=true --set 'execution.step_options={"startup":{"solve":false},"blow":{"solve":false},"heat":{"solve":false}}'
```

The kept steps must form a leading block: a step after a re-solved one is refused, because
its saved solution started from the old state. A resumed run may change only these
settings: `execution.solve`, the `solve` and `plot_profiles` keys of
`execution.step_options`, `execution.plot_profiles`, `execution.plot_format`,
`execution.print_to_terminal` and `execution.print_cycle_build`.

## A CSS run

```bash
pydac css --config my_case.json --set execution.max_cycles=2 --set execution.require_css=true || echo "exit code $?"
```

After each cycle, pyDAC compares the state at the end of the probe step
(`execution.css_probe_step`, default the last step of the cycle) with the previous cycle.
CSS is reached when no compared value changed by more than both an absolute `1e-8` and
the relative `execution.css_tolerance` (default `0.005`), the relative change being
taken against the current cycle's value; the run then stops and reports that cycle. Otherwise it stops after
`execution.max_cycles` cycles (default 50) and reports the last one, labelled "CSS not
reached". With `execution.require_css` true (the default), that ending exits with code `3`,
as the command above does after two cycles; with false it exits with `0`.

### What a CSS run writes

At the end of a CSS run, pyDAC writes for the reported cycle:

- the CSS profile plots, when `execution.plot_cyclic_profiles` is true (default true);
- the KPI tables and figures, when `execution.generate_kpi_outputs` is true (default true),
  in `execution.plot_format` (default `pdf`);
- the oxygen and hot-bed check, always.

The KPIs are normalized by the adsorption step, so a run that reports cycle 1 (for example
`execution.max_cycles = 1`) skips them with a printed note; the plots and the check still
run. `pydac.run_css` writes the same files as `pydac css`.

### Continue a CSS run

A run that stopped at `execution.max_cycles` continues with `--resume` and a higher limit.
The folder is the one the CSS run above printed (`packed_bed_08773fa0` for that command;
any change to the case gives another name):

```bash
pydac css --resume simulation_results/cyclic_mode/packed_bed_08773fa0 --set execution.max_cycles=4
```

It starts after the last completed cycle, from that cycle's final state. A run that already
reached CSS is refused. A resumed CSS run may change only `execution.max_cycles`,
`execution.css_tolerance`, `execution.require_css`, the output settings
(`execution.plot_cyclic_profiles`, `execution.plot_final_cycle_profiles`,
`execution.generate_kpi_outputs`, `execution.keep_only_final_cycle`,
`execution.plot_profiles`, `execution.plot_format`) and the printing settings.

### Report a saved CSS run

`--report RUN` writes the CSS plots (`plots_css/`), KPIs and oxygen check of a saved cycle
again, whatever the saved output settings say; it does not redraw the step plots unless you
give `--snapshot-times` ([Times of the step plots](#times-of-the-step-plots)). It solves nothing, deletes nothing, and replaces the earlier
report files of that cycle. `--cycle N` picks the cycle (default: the CSS cycle, else the
last completed one); `--set execution.plot_format=FMT` is the only setting it accepts.

The report rebuilds only the cycle outputs, from what the run saved: the step KPIs
(`kpis.pkl` in each step folder, computed when the step was solved) and the step solutions
(`sol.pkl`). It does not compute the step KPIs again: if one is missing, the report stops
and names the file.

```bash
pydac css --report simulation_results/cyclic_mode/packed_bed_08773fa0 --cycle 3 --set execution.plot_format=png
```

A step run has no report; replay it with `--resume` as shown above.

If the solve succeeds but writing the report fails, the run is kept and the command prints
`Run completed; reporting failed: ... Retry with pydac css --report RUN.` and exits with
code `1`. Fix the cause and run that report command.

## Times of the step plots

With step plots on (`execution.plot_profiles`, or `execution.plot_final_cycle_profiles` in
a CSS run), every step draws its fields at a few times:

- **2D maps** over the length and the radius: six snapshots in a 2 x 3 grid;
- **1D profiles** along the length (and, in 2D, along the radius): one line per time.

pyDAC picks the times per field: fixed times for some steps and fields (for example 1 to
60 min for the CO2 loading of `s_des`), and then the map and the profiles of that field show
the same times. Otherwise the map takes six times and the profiles ten, evenly spaced over
the step. `--snapshot-times` picks them yourself, per step and field, for the map and the
profiles together:

```bash
pydac css --config my_case.json --set execution.plot_final_cycle_profiles=true --snapshot-times '{"heat": {"T_bed": {"times": [1, 10, 30, 60, 90, 150], "unit": "min"}}}' --overwrite
```

- A field is `T_bed`, `p_tot`, `c_tot`, `y_<gas>` or `q_<adsorbate>` (for example `y_CO2`);
  the fields you leave out keep pyDAC's times.
- The profiles follow the times only where the step draws them: with a mixture other than the
  built-in one (as in tutorials [02](../../examples/notebooks/02_custom_case.ipynb) and [06](../../examples/notebooks/06_breakthrough_validation.ipynb)), the 1D profiles show only `y_<gas>` and `q_<adsorbate>`.
- In a profile figure with two panels (for example `y_CO2` and `q_CO2`) each panel takes
  the times of its own field. The wall temperature follows `T_bed`; the gas velocity has no
  field and keeps ten evenly spaced times.
- Give 1 to 6 different times >= 0, in `"s"` (the default), `"min"` or `"h"`. A time past
  the end of the step shows the nearest saved state, with a warning; two times that fall on
  the same saved state give one panel, with a warning.
- It changes only the figures: not the case, its fingerprint or its run folder. It is not
  saved with the run, so `--resume` draws with pyDAC's times again.
- Only a step whose plots the run draws takes times: `execution.step_options` (step run)
  and `execution.cycle_options` (CSS run) count too. A step without plots is refused, as are
  `--resume` and `--export-case`.

Already ran, and the maps miss the moment you care about? `--report` draws the step plots of
the reported cycle again with new times, from the saved run. It redraws every step of that
cycle: steps and fields left out of `--snapshot-times` get pyDAC's default times, even if the
run used its own. Give those schedules again to keep them, as
[tutorial 03](../../examples/notebooks/03_css_tvsa.ipynb) does.

```bash
pydac css --report simulation_results/cyclic_mode/packed_bed_08773fa0 --snapshot-times '{"cool": {"T_bed": {"times": [0, 2, 5, 10, 20, 55], "unit": "min"}}}'
```

Nothing is solved: every step of the cycle is reloaded from its `sol.pkl` (about a minute
on a 50 x 25 grid) and only the step plots are written: the step logs and step KPIs stay as
the run wrote them. It needs the saved solutions of that cycle and the final state of the
cycle before it, so a run with `execution.keep_only_final_cycle` refuses it for any cycle
after the first.

In Python: `pydac.run_css(case, snapshot_times={"heat": {"T_bed": {...}}})` (the same for
`pydac.run_steps`), and `pydac.report(folder, snapshot_times={...})`.

## Run folders

A simulation run writes its results into its run folder, named from the case fingerprint
([Cases and settings](03_cases.md#fingerprint-and-run-folder)). Both commands print it when
they start and when they end, and `RunResult.folder` is the same path.

A fresh run into a folder that already holds a run is refused before anything is written,
with exit code `2` and three ways out: continue it with `--resume`, replace it with
`--overwrite`, or set another `execution.extra_tag`. `--overwrite` deletes exactly that run
folder, after the whole case has been checked. With `--dry-run` it only says what it would
delete:

```bash
pydac css --config my_case.json --set execution.max_cycles=2 --set execution.require_css=true --overwrite --dry-run
```

`--overwrite` cannot be combined with `--resume`, `--report` or `--export-case`
(`--overwrite-case` is the separate flag that replaces a case file). In Python, pass
`overwrite=True` to `run_steps` or `run_css`.

## Check before running

`--dry-run` checks the case, the step sequence, the run options and the run folder, prints
the run folder, and stops before solving; `--print-resolved` prints the complete case. With
`--resume RUN`, `--dry-run` checks the whole continuation (saved run, version, allowed
settings) and prints what would run:

```bash
pydac steps --resume simulation_results/step_mode/packed_bed_7a4930a5 --set execution.solve=false --dry-run
```

Every refusal exits with code `2` before anything is written, with or without `--dry-run`.

## Continuing needs the same pyDAC version

Continuing a CSS run, or re-solving steps of a step run, needs the pyDAC version that wrote
the run; another version is refused and names both versions. Reporting a CSS run and
replaying all steps of a step run work with any later release of the same major version
([Reproducibility](07_reproducibility.md#across-pydac-versions)).

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | failure after the run started: a traceback is printed; the files written so far stay |
| `2` | configuration error or refused request, one line on stderr starting `Configuration error:` or `Refused:`; or invalid command-line arguments, after the usage. No run starts and no run folder is written |
| `3` | `pydac css` (also with `--resume`) stopped at `execution.max_cycles` without CSS while `execution.require_css` is true |

## The same from Python

| Terminal | Python |
|---|---|
| `pydac steps --case PATH` | `pydac.run_steps(pydac.load_case(PATH, exact=True))` |
| `pydac css --case PATH` | `pydac.run_css(pydac.load_case(PATH, exact=True))` |
| `--overwrite` | `overwrite=True` |
| `--resume RUN --set PATH=VALUE` | `pydac.resume(RUN, settings={PATH: VALUE})` |
| `pydac css --report RUN --cycle N` | `pydac.report(RUN, cycle=N)` |
| `--snapshot-times JSON` | `snapshot_times={...}` in `run_steps` / `run_css` / `report` |

The Python calls raise instead of returning an exit code, and `run_css` returns
`css_reached=False` instead of exit code `3` ([Quick start](02_quickstart.md#errors)).

## Related pages

- [Optimization campaigns](05_optimization.md)
- [Results](06_results.md): the files a run writes and how to read them
- [Cases and settings](03_cases.md)
- [Quick start](02_quickstart.md)
- [05 — Cycle steps](../manual/05_cycle_steps.md): what each step does and its stop rule
