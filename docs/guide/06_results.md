---
title: "Results"
toc: true
toc-depth: 3
embed-resources: true
---

> **New to pyDAC? Do the [tutorials](../../examples/README.md) first, starting with
> [01 First steps](../../examples/notebooks/01_first_steps.ipynb).** They teach pyDAC step by
> step on real runs; this guide is the reference behind them and goes into more technical detail.

Every run writes its results into one folder, and you read them from that folder, not from
the value a command returns. This page shows where a step run, a cyclic steady state (CSS)
run and an NSGA-II campaign write, and how to read the results from Python.

The examples use the small 1D case of the [quick start](02_quickstart.md), run to three
cycles, in a new, empty folder:

```bash
pydac validate --set grid.nz=25 --set grid.nr=1 --set execution.max_cycles=3 --set execution.require_css=false --export-case my_case.json
pydac css --case my_case.json
```

## Where pyDAC writes

Simulation results are written inside the run folder or campaign folder, under the output
root `execution.output_root` (default `simulation_results`, relative to the folder you run
from). Case and preset exports are written to the path you give them; the tutorials use
`output_cases/`. pyDAC never writes results into the installed package. Libraries keep their own caches where they
always do, for example the matplotlib font cache in your home folder.

The output root is part of the case, so changing it changes the fingerprint and the folder
name ([Cases and settings](03_cases.md#fingerprint-and-run-folder)). The folder names are:

| Run | Folder |
|---|---|
| step run | `OUTPUT_ROOT/step_mode/[TAG_]packed_bed_HASH/` |
| CSS run | `OUTPUT_ROOT/cyclic_mode/[TAG_]packed_bed_HASH/` |
| campaign | `OUTPUT_ROOT/nsga2/[TAG_]GRID_packed_bed_HASH/` |

`HASH` is the first eight characters of the case fingerprint (for a campaign, of the whole
campaign definition, see [Optimization campaigns](05_optimization.md#the-campaign-folder)),
`TAG_` is `execution.extra_tag` when you set one, and `GRID` is the grid, for example
`1D_25z`. Inside a step or CSS run folder, everything sits in the results folder `1D_mode/`
(`grid.nr = 1`) or `2D_mode/` (`grid.nr > 1`). `RunResult.folder` is the run folder and
`RunResult.results_folder` the results folder.

## A CSS run

The CSS run above writes `simulation_results/cyclic_mode/packed_bed_7a4930a5/`:

```text
packed_bed_7a4930a5/
└── 1D_mode/
    ├── resolved_config.json      the complete case of the run
    ├── case_metadata.json
    ├── log.txt
    ├── provenance/               who wrote the run, with which versions
    ├── cycle_1/                  one folder per completed cycle
    │   ├── startup/              one folder per step (cycle 1 starts with startup)
    │   ├── blow/  heat/  s_des/  cool/  press/
    │   ├── css_diagnostics.json  the CSS check of this cycle
    │   ├── cycle_metadata.json
    │   ├── resolved_runtime.json the runtime properties and models the steps used
    │   └── simulation_context.pkl
    ├── cycle_2/  cycle_3/        from cycle 2 on, the cycle starts with ads/
    ├── plots_css/cycle_3/        CSS profile plots of the reported cycle
    └── kpis/                     KPI tables and figures of the reported cycle
```

Each step folder holds the step's solution (`sol.pkl`), its step metrics (`kpis.csv`,
`kpis.pkl`, `metrics_provenance.json`), its log, its provenance and a `profiles/` folder
(three empty subfolders unless the step profiles are plotted); with `execution.compute_numerical_sparsity` true
(a solver diagnostic, default false) also `numerical_jacobian/`. `plots_css/cycle_N/` exists for
the reported cycle `N` when `execution.plot_cyclic_profiles` is true (default true), and for
every cycle reported with `pydac css --report`.
`execution.generate_kpi_outputs` (default true) writes the KPI tables and figures into
`kpis/`, from cycle 2 on (the KPIs need an adsorption step). The oxygen and hot-bed check
writes `kpis/csv/process/` in any case, so a `kpis/` folder alone does not mean the KPIs
were written. In `kpis/`:

- `cycle_N_css_kpi_summary.txt` and `.json`: the main KPIs of cycle `N`;
- `csv/`: the tables, by topic (`process/`, `energy/`, `mass/`, `closures/`), and
  `process/oxygen_hot_bed_check.csv`;
- `pdf/` (or the folder of `execution.plot_format`): the figures.

The file names carry `css` whether or not the run reached CSS; the run's last lines and
`cycle_N/css_diagnostics.json` say whether it did.

### Keep only the reported cycle

Every completed cycle keeps its folder: about 7 MB per cycle for this coarse 1D case, much
more on a fine 2D grid. With `execution.keep_only_final_cycle` true, a CSS run deletes
every `cycle_N/` except the reported one when it ends. It cannot be combined with
`execution.plot_profiles`. A report (`pydac css --report`) never deletes anything.

## A step run

`pydac steps --case my_case.json` writes `simulation_results/step_mode/packed_bed_7a4930a5/`,
with the same fingerprint as the CSS run of the same case:

```text
packed_bed_7a4930a5/
└── 1D_mode/
    ├── resolved_config.json
    ├── case_metadata.json
    ├── provenance/
    ├── resolved_runtime.json
    ├── simulation_context.pkl
    └── startup/  blow/  heat/  s_des/  cool/  press/  ads/
```

The step folders hold the same files as in a CSS run. A step run has no KPI report: to see
the step profiles, set `execution.plot_profiles` to true, or replay the saved run with plots
([Running steps and cycles](04_running.md#replay-and-partial-re-solve)).

## Figures in a notebook

pyDAC saves every figure it draws to its run folder and shows none, in a notebook too, so a
run does not fill the notebook with dozens of plots. Show the ones you want from their files.
A notebook displays PNG, not PDF: set `execution.plot_format` to `"png"` in the case first.

```python
from IPython.display import Image, display

display(Image(filename="simulation_results/cyclic_mode/RUN/1D_mode/plots_css/cycle_3/css_outlet_T_bed.png"))
```

Your own matplotlib figures, for example of the KPIs read in the next section, show as usual.

## Read a CSS cycle from Python

`pydac.CycleDataLoader` finds a cycle in a run folder and `pydac.CycleKPIs` computes the
KPIs from it:

```python
import pydac

run = "simulation_results/cyclic_mode/packed_bed_7a4930a5"
loader = pydac.CycleDataLoader(run)
kpis = pydac.CycleKPIs(loader.load())

print(loader.cycle_label)
print(f"productivity    {kpis.productivity():.2f} kg CO2 per m3 of bed per day")
print(f"equivalent work {kpis.equivalent_work():.1f} MJ per kg CO2")
print(f"CO2 recovery    {kpis.recovery():.3f}")
print(f"CO2 purity      {kpis.purity()['purity_wet']:.3f} (wet), {kpis.purity()['purity_dry']:.3f} (dry)")
print(kpis.cycle_time_breakdown())
```

```text
cycle_3
productivity    33.23 kg CO2 per m3 of bed per day
equivalent work 77.1 MJ per kg CO2
CO2 recovery    0.746
CO2 purity      0.968 (wet), 1.000 (dry)
             Step  Duration [s]  Fraction [-]  Cumulative time [s]
0      adsorption  39000.000000      0.716837         39000.000000
1        blowdown    106.588379      0.001959         39106.588379
2      preheating   9000.000000      0.165424         48106.588379
3      desorption   3600.000000      0.066170         51706.588379
4        cooldown   2673.463051      0.049139         54380.051431
5  pressurization     25.656984      0.000472         54405.708415
```

This coarse run reaches CSS at cycle 3 with the default `execution.css_tolerance` (0.005);
the numbers only show the calls.

`CycleDataLoader(FOLDER, *, cycle_index=None, spatial_mode=None)` (keyword arguments after
`FOLDER`):

- `FOLDER` is the run folder, its results folder, or one `cycle_N/` folder.
- `cycle_index=None` takes the highest-numbered `cycle_N/` folder, which is the CSS cycle
  when the run stopped there; the loader does not check that the folder is complete, so
  name the cycle of an interrupted run. `cycle_index=2` takes `cycle_2/`; negative values count from the end
  (`-1` is the last).
- `spatial_mode` (`"1D_mode"` or `"2D_mode"`) is needed only if a folder holds both `1D_mode/` and
  `2D_mode/`.
- `.load()` returns the cycle data that `CycleKPIs` takes.
- It raises `FileNotFoundError` when a folder or file is missing and `ValueError` when the
  cycle does not exist (the message lists the cycles that do) or the spatial mode is
  ambiguous. These are not `pydac.ConfigurationError`.

`CycleKPIs` returns plain Python and pandas values:

| Call | Returns | Unit |
|---|---|---|
| `productivity()` | `float` | kg CO2 per m³ of bed per day |
| `equivalent_work()` | `float` | MJ per kg of recovered CO2 |
| `recovery(species="CO2")` | `float` | recovered over adsorbed mass, fraction |
| `cyclic_working_capacity(species="CO2")` | `float` | mol per kg of sorbent |
| `recovered_mass(species)` | `float` | kg per cycle |
| `cycle_time()` | `float` | s |
| `steam_to_co2_ratio()` | `float`, or `None` without injected steam (`inf` with steam but no recovered CO2) | kg steam per kg CO2 |
| `purity()` | `dict`: `purity_wet`, `purity_dry` (fractions), `oxygen_ppm`, `nitrogen_ppm` (ppm molar), `masses_kg`, `moles` | |
| `cycle_time_breakdown()` | `DataFrame`, one row per step | s and fraction |
| `energy_breakdown()` | `DataFrame`, one row per step, columns Compression, Vacuum, Jacket, Steam, Condenser | MJ per cycle; `specific=True`: MJ per kg of recovered CO2 |
| `equivalent_work_breakdown()` | `DataFrame`, the same rows and columns, as equivalent work | kWh per kg of recovered CO2; `specific=False`: kWh per cycle |
| `product_composition()` | `DataFrame`, one row per species | |

The table gives the units; `help(pydac.CycleKPIs.productivity)` shows a method's options. How each KPI
is computed is in [06 — Post-processing](../manual/06_post_processing.md). A value that needs a
species the mixture does not have is NaN (for example `oxygen_ppm` without O2) or raises
`ValueError`.

## Read a campaign

A campaign's results are files in `CAMPAIGN/results/`. The example campaign of
[Optimization campaigns](05_optimization.md) needs its bounds file. Save it as
`my_bounds.json`:

```json
{
  "t_ads": [3600, 39000],
  "t_heat": [1800, 9000],
  "t_des": [600, 3600],
  "P_vac": [10000, 50000],
  "v_feed": [0.02, 0.1],
  "v_steam": [0.1, 0.6]
}
```

```bash
pydac validate --set grid.nz=25 --set grid.nr=1 --set execution.max_cycles=2 --set execution.require_css=false --export-case campaign_case.json
pydac nsga2 --case campaign_case.json --bounds my_bounds.json --pop-size 4 --generations 2 --workers 4 --purity-min 0.5 --oxygen-ppm-max 1000 --steam-condensation
```

The Pareto front is in `pareto_front.csv`: one row per point, the decision variables in their
units, the model flows they give, and the two objectives `eq_work` (MJ per kg CO2) and
`productivity` (kg CO2 per m³ of bed per day). `last_pareto_front.npz` holds the same points
as arrays: `X` (decision variables, in the order of `var_names`), `F` (the objectives as the
optimizer minimizes them: equivalent work and minus productivity) and `G` (the constraint
values, feasible when at most 0, in the order of `constraint_names`).

```python
import numpy as np
import pandas as pd

results = "simulation_results/nsga2/1D_25z_packed_bed_0b76eb17/results"
front = pd.read_csv(f"{results}/pareto_front.csv")
print(front[["t_ads", "P_vac", "eq_work", "productivity"]])

arrays = np.load(f"{results}/last_pareto_front.npz")
print(arrays["var_names"], arrays["F"].shape)
```

The other files of `results/`: `last_population.npz` (the final population with its
feasibility and rank), `population_and_extras.csv` (the feasible, non-dominated candidates
of the last generation with their purity, O2 and energy terms),
`candidate_records/generation_NNNN.jsonl` (one JSON record per line for every evaluated
candidate, including its error if it failed), `optimization_summary.json`, `figures/` (PDF),
`checkpoints/` and `checkpoint_state.json` (what `--resume` reads).

The candidate folders keep no simulation results: a candidate's complete case is in its record,
under `provenance.input.resolved_config` (a failed candidate may have none). That object
alone, not the whole record, is a frozen case. Its `execution.output_root` is `"."`, so it
writes under the folder you run it from:

```python
import json

results = "simulation_results/nsga2/1D_25z_packed_bed_0b76eb17/results"
records = f"{results}/candidate_records/generation_0001.jsonl"
with open(records) as handle:
    record = json.loads(handle.readline())          # the first candidate of generation 1
with open("candidate_case.json", "w") as handle:
    json.dump(record["provenance"]["input"]["resolved_config"], handle, indent=2)
```

```bash
pydac css --case candidate_case.json --dry-run
```

## Open only folders you trust

Reading a run folder (`pydac css --report`, `--resume`, `CycleDataLoader`, a step replay)
loads its `.pkl` files with Python's `pickle`, which can run code stored in the file. Open
only run folders you or someone you trust wrote, as you would with `pandas.read_pickle`.
A campaign's checkpoint and result files are JSON, CSV and NumPy `.npz` files, read without
`pickle`.

## Related pages

- [Running steps and cycles](04_running.md): what each run writes, report, replay
- [Optimization campaigns](05_optimization.md)
- [Reproducibility](07_reproducibility.md): sharing a case, provenance records, versions
- [06 — Post-processing](../manual/06_post_processing.md): KPI definitions and result files
