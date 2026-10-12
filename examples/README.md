# pyDAC tutorials

Each tutorial is one case file in [`cases/`](cases/), run two ways: a notebook in
[`notebooks/`](notebooks/) and the terminal commands below (tutorial 07 only as a notebook: it
registers its own isotherm from Python). Both give the same cases, so they
write the same run folders. [`cases/quick_1D.json`](cases/quick_1D.json) is not a tutorial: it is
the partial preset of the quick start in the [README](../README.md#quick-start-from-the-terminal).

The tutorials are in the repository, in the source distribution and in the installed
package: Use `pydac get-started my_pydac` to copy the tutorials with the user guide into a new folder of the
same layout ([Get the tutorials](../docs/guide/01_installation.md#get-the-tutorials-and-this-guide)).
They need only an installed pyDAC
([Installation](../docs/guide/01_installation.md)); for the notebooks, select the kernel
of your pyDAC environment ([Use pyDAC in Jupyter](../docs/guide/01_installation.md#use-pydac-in-jupyter)).

The runs take from a few minutes to about 40 minutes; each tutorial states its runtime. Results go
under `simulation_results/` in the folder you run from (for a notebook, its own folder); the
case files a tutorial saves (complete cases, presets) go under `output_cases/`, next to it.
Running a tutorial again replaces its earlier results; from the terminal, a campaign (05) is
never replaced: delete its folder first.

| Tutorial | What it shows | Notebook |
|---|---|---|
| 01 First steps | a step run, the KPI table of each step, step profile plots, time- or event-based stop | [01_first_steps.ipynb](notebooks/01_first_steps.ipynb) |
| 02 Build your own case | replace the defaults block by block (column, gas, sorbent, competitive Sips isotherm on 13X, kinetics, reference state, steps), a 2D TSA step run (load, then heat at 1 atm), one setting changed at a time, `export_settings` and `export_case` | [02_custom_case.ipynb](notebooks/02_custom_case.ipynb) |
| 03 A cycle at its periodic state | a CSS run of the built-in TVSA cycle on a 2D grid, how the cycle settles, `CycleKPIs` (productivity, equivalent work, recovery, purity), where the CO2 and the energy go, the KPI report, the times of the step plots (`snapshot_times`), `report` and `resume` | [03_css_tvsa.ipynb](notebooks/03_css_tvsa.ipynb) |
| 04 1D or 2D | the same cycle in 1D, in 2D and in 2D with wall channeling, on the same axial grid; where the 1D model departs from the 2D one (wall heating) and what channeling changes | [04_1d_vs_2d.ipynb](notebooks/04_1d_vs_2d.ipynb) |
| 05 An optimization campaign | a trial NSGA-II campaign on 8 workers (how many cores you need and how to choose the workers), variables and bounds, constraints, the checks before running, every candidate and the Pareto front, why a campaign is not run twice, a front point run again as a normal case | [05_nsga2_trial.ipynb](notebooks/05_nsga2_trial.ipynb) |
| 06 Reproduce the breakthrough validation | the 2D model of the pyDAC paper on a 13X breakthrough experiment (200 × 40 grid, wall channeling, a declared initial solid) run from the release, compared with the digitised measurements of Zanco et al. (outlet CO2, three bed temperatures), compared curve by curve in figures | [06_breakthrough_validation.ipynb](notebooks/06_breakthrough_validation.ipynb) |
| 07 Your own isotherm | a new sorbent from the literature (APDES-NFC of Stampi-Bombelli et al., 2020, CO2/H2O co-adsorption): the isotherm written as an equilibrium system, tested, registered with `pydac.register_equilibrium`, given with the sorbent's properties as a sorbent file and run from humid and dry air | [07_new_isotherm.ipynb](notebooks/07_new_isotherm.ipynb) |

## From the terminal

Run these from the repository root, or from the folder that `pydac get-started` made.

### 01 First steps

```bash
pydac steps --config examples/cases/01_first_steps.json --overwrite
pydac steps --config examples/cases/01_first_steps.json --set execution.termination_mode=event --overwrite
```

Each run prints its run folder; every step folder in it holds `kpis.csv` and, under
`profiles/`, the step plots as PNG.

### 02 Build your own case

```bash
pydac steps --config examples/cases/02_custom_case.json --overwrite
pydac steps --config examples/cases/02_custom_case.json --set steps.w_des.thermal.jacket_temperature_K=473.15 --overwrite
pydac steps --config examples/cases/02_custom_case.json --export-case output_cases/02_full_case.json --overwrite-case --dry-run
```

The first command runs the case (2D grid, regeneration at 150 °C, about two and a half
minutes); the second regenerates at 200 °C (use `373.15` for 100 °C). The third saves the
complete case (every field), as `pydac.export_case` does in the notebook.
`pydac validate --show-settings` lists every field path.

### 03 A cycle at its periodic state

```bash
pydac css --config examples/cases/03_css_tvsa.json --snapshot-times '{"heat": {"T_bed": {"times": [1, 5, 10, 20, 30, 150], "unit": "min"}, "q_CO2": {"times": [5, 15, 30, 60, 90, 150], "unit": "min"}}}' --overwrite
pydac css --report simulation_results/cyclic_mode/RUN --set execution.plot_format=pdf
```

The first command runs the cycle to its periodic state (2D grid, about ten minutes,
about 0.8 GB on disk) and prints its run folder; `RUN` in the second is that folder's
name (`tutorial_03_packed_bed_…`). The second writes the plots and the KPI report again as
PDF, without solving.

### 04 1D or 2D

```bash
pydac css --config examples/cases/04_1d_vs_2d.json --snapshot-times '{"heat": {"T_bed": {"times": [1, 5, 10, 20, 30, 150], "unit": "min"}}}' --overwrite
pydac css --config examples/cases/04_1d_vs_2d.json --set grid.nr=1 --snapshot-times '{"heat": {"T_bed": {"times": [1, 5, 10, 20, 30, 150], "unit": "min"}}}' --overwrite
pydac css --config examples/cases/04_1d_vs_2d.json --set bed.design.flow_model=wall-channeling-reduced --snapshot-times '{"heat": {"T_bed": {"times": [1, 5, 10, 20, 30, 150], "unit": "min"}}}' --overwrite
```

The same cycle in 2D, in 1D and in 2D with wall channeling (about 20, 2 and 20 minutes);
each run keeps only its reported cycle and writes its KPI report into its own `kpis/`.

### 05 An optimization campaign

```bash
pydac nsga2 --config examples/cases/05_nsga2_trial.json --bounds examples/cases/05_nsga2_bounds.json --pop-size 8 --generations 4 --purity-min 0.95 --oxygen-ppm-max 1000 --eq-work-max 60 --workers 8 --dry-run
pydac nsga2 --config examples/cases/05_nsga2_trial.json --bounds examples/cases/05_nsga2_bounds.json --pop-size 8 --generations 4 --purity-min 0.95 --oxygen-ppm-max 1000 --eq-work-max 60 --workers 8
```

The first command checks the campaign and prints its folder
(`simulation_results/nsga2/tutorial_05_1D_25z_packed_bed_ac074b54`); the second runs it: 32 CSS runs on
a coarse 1D grid, about 3 minutes with 8 workers. It needs at least 8 free cores (the notebook
explains how to count them); with fewer, lower `--workers`: same result, longer run. Set
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` in the shell first.
This is a trial, not a converged front.

### 06 Reproduce the breakthrough validation

```bash
pydac steps --config examples/cases/06_breakthrough_validation.json --overwrite
```

One 2D run of about 40 minutes. The comparison with the measurements in
[`data/zanco2022_13x/`](data/zanco2022_13x/README.md) is Python: run the notebook.

### 07 Your own isotherm

No terminal version: a registered model exists only in the Python session that registered it.
Run the notebook ([Extending pyDAC](../docs/guide/08_extending.md)).

## More

The user guide explains each part in full: [Quick start](../docs/guide/02_quickstart.md),
[Cases and settings](../docs/guide/03_cases.md),
[Running steps and cycles](../docs/guide/04_running.md),
[Optimization campaigns](../docs/guide/05_optimization.md),
[Results](../docs/guide/06_results.md) and
[Extending pyDAC](../docs/guide/08_extending.md).
