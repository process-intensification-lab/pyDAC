---
title: "Optimization campaigns"
toc: true
toc-depth: 3
embed-resources: true
---

> **New to pyDAC? Do the [tutorials](../../examples/README.md) first, starting with
> [01 First steps](../../examples/notebooks/01_first_steps.ipynb).** They teach pyDAC step by
> step on real runs; this guide is the reference behind them and goes into more technical detail.

An NSGA-II campaign (`pydac nsga2`) searches the operating conditions of the case's cycle (by
default the steam-assisted temperature-vacuum swing adsorption, TVSA, cycle) for the best
trade-off between energy and productivity. You give a case and bounds on a few operating
variables. Each candidate is your case with those variables changed, run to cyclic steady state
(CSS). The result is the Pareto front of the two objectives, which are fixed in pyDAC 3.0.0:

| Objective | Direction | Unit |
|---|---|---|
| equivalent work | minimized | MJ per kg of recovered CO2 |
| productivity | maximized | kg of recovered CO2 per m³ of bed per day |

Every candidate is a full CSS run, so a campaign costs about population size × generations
CSS runs. Try the commands on a small case first.

## The base case

The campaign takes everything except its decision variables from the base case: grid,
geometry, bed, sorbent, cycle steps, CSS settings. Every candidate solves, prints nothing
and draws no step profiles, whatever the base case says, and keeps a compact record
instead of its simulation results ([Results](06_results.md#read-a-campaign)). This page uses a coarse 1D case
that stops after two cycles, so the example campaign below finishes in a few minutes. Run
this page in a new, empty folder ([Quick start](02_quickstart.md)):

```bash
pydac validate --set grid.nz=25 --set grid.nr=1 --set execution.max_cycles=2 --set execution.require_css=false --export-case my_case.json
```

For a real campaign keep `execution.require_css` true (the default) and enough
`execution.max_cycles` for your candidates to reach CSS: with `require_css` true, a candidate
that stops at `execution.max_cycles` without CSS counts as failed.

## Decision variables and bounds

The decision variables are set by the desorption step of the cycle:

| Variable | Unit | What it sets |
|---|---|---|
| `t_ads` | s | the duration of adsorption (`ads`) and startup |
| `t_heat` | s | the duration of heating (`heat`) |
| `t_des` | s | the duration of desorption (`s_des` or `w_des`), also its event fallback limit |
| `P_vac` | Pa | the low pressure of the vacuum steps (blowdown to cooling) |
| `v_feed` | m/s | the superficial velocity of the air feed in adsorption and startup |
| `v_steam` | m/s | the superficial velocity of the steam in steam purge and steam-assisted desorption; only for a cycle with `s_des` |

A cycle with steam-assisted desorption (`s_des`) has all six variables; a cycle with wall-heated
desorption (`w_des`) has the first five. Before each candidate runs, the velocities become the
case's flows: `v_feed` times the bed cross-section gives the feed flow in m³/s, and `v_steam`
times the cross-section and the steam density at 373.15 K and `P_vac` gives the steam mass
flow in kg/s (`P_vac` capped at the water saturation pressure at 373.15 K, about 1.013 bar). The complete case of each candidate is kept in its record in
`results/candidate_records/` ([Results](06_results.md#read-a-campaign)).

The bounds file gives `[lower, upper]` for every variable of your cycle. Save it as
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

These bounds show the syntax; they are not recommended process settings. pyDAC has no built-in
search space: `--bounds` is required. A missing variable, a variable the cycle does not have,
or a pair that is not two finite numbers with lower < upper is refused.

## Constraints

Without constraints, every candidate that runs is feasible. Each flag adds one constraint:

| Flag | Constraint | Unit |
|---|---|---|
| `--purity-min X` | CO2 mole fraction of the recovered product, water included, at least `X` | fraction, for example `0.95` |
| `--oxygen-ppm-max X` | O2 in the recovered product at most `X` | ppm (molar) |
| `--eq-work-max X` | equivalent work at most `X` | MJ per kg of recovered CO2 |
| `--steam-condensation` | at the end of heating, the coldest point of the bed is at or above the water saturation temperature at the heating pressure, so steam does not condense when it enters | |

One condition is always active: a candidate whose run fails, or that does not reach CSS while
`execution.require_css` is true, is infeasible. `--steam-condensation` applies only to cycles
with steam-assisted desorption and needs H2O in the mixture; `--oxygen-ppm-max` needs O2 in
the mixture. Otherwise the campaign is refused before it starts.

## Check before running

`--dry-run` checks the case, the bounds, the variables of the cycle and the campaign folder,
prints the campaign folder, and stops:

```bash
pydac nsga2 --case my_case.json --bounds my_bounds.json --pop-size 4 --generations 2 --purity-min 0.5 --oxygen-ppm-max 1000 --steam-condensation --dry-run
```

It also builds the candidate case at the two corners of the search space, all variables at
their lower bounds and all at their upper bounds, and refuses bounds that make a corner
impossible, naming it. For example, an upper `t_ads` longer than the startup step's
`event_max_time_s` (43200 s in the default case) is refused. Save it as `wide_bounds.json`:

```json
{
  "t_ads": [3600, 50000],
  "t_heat": [1800, 9000],
  "t_des": [600, 3600],
  "P_vac": [10000, 50000],
  "v_feed": [0.02, 0.1],
  "v_steam": [0.1, 0.6]
}
```

```bash
pydac nsga2 --case my_case.json --bounds wide_bounds.json --pop-size 4 --generations 2 --dry-run || echo "exit code $?"
```

```text
Configuration error: Candidates at the upper bounds (t_ads=50000, t_heat=9000, t_des=3600, P_vac=50000, v_feed=0.1, v_steam=0.6) cannot run: Invalid configuration for step `startup`: `event_max_time_s` cannot be shorter than nominal time. Adjust the bounds or the base case.
exit code 2
```

Only the two corners are built: a combination that fails only inside the box is found when
a candidate runs.

## Run a campaign

```bash
pydac nsga2 --case my_case.json --bounds my_bounds.json --pop-size 4 --generations 2 --workers 4 --purity-min 0.5 --oxygen-ppm-max 1000 --steam-condensation
```

The constraint values are loose so that this two-cycle case has feasible candidates. The
campaign prints its campaign folder, a summary of the setup, one line per generation, and at
the end (the timings depend on the machine):

```text
Stop reason: max_generations at generation 2 after 8 evaluations.
Timing: optimizer wall=0.96 min | candidate mean/median/p90=0.43/0.41/0.48 min | candidate sum=3.41 min.
Candidates evaluated 8, failed 0.
```

The options:

| Option | Default | Meaning |
|---|---|---|
| `--pop-size N` | `100` | candidates per generation |
| `--generations N` | `100` | generations, or the upper limit with `robust_pareto` |
| `--workers N` | `1` | local processes that evaluate candidates in parallel; to keep each at one numerical thread, set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` in the shell before starting |
| `--seed N` | `1` | random seed of the optimizer |
| `--termination MODE` | `fixed_generations` | the stop rule, see below |
| `--max-evaluations N` | none | an upper limit on candidate evaluations, in either stop rule |

`--workers` changes only the speed: the same campaign with another number of workers gives
the same result.

### When a campaign stops

With `--termination fixed_generations` the campaign runs `--generations` generations. With
`--termination robust_pareto` it stops earlier once the feasible Pareto front stops moving:
the front changes by less than `--ftol` (default `5e-3`) over a window of `--period`
generations (default 10), checked every generation, or less often with `--n-skip` (default 0,
the number of generations skipped between checks). `--generations` remains the upper limit.

```bash
pydac nsga2 --case my_case.json --bounds my_bounds.json --termination robust_pareto --ftol 5e-3 --period 10 --generations 100 --dry-run
```

`--ftol`, `--period` and `--n-skip` are refused with `fixed_generations`.

### Failed candidates

A candidate whose run fails is kept as infeasible and the campaign goes on. The exception
is an error that would make every candidate's KPIs unreliable, such as a missing condenser
entry in the energy ledger: it stops the campaign (exit code `1`). The final
`Candidates evaluated N, failed F.` line lists below it the five most frequent errors, and
`results/optimization_summary.json` keeps all of them under `evaluations`.

If every candidate of the first generation fails, the campaign stops with an error (exit
code `1`) that shows the same summary and the file with the candidate records. Nothing is
committed: fix the cause and run the same command again.

## The campaign folder

The campaign folder is `OUTPUT_ROOT/nsga2/` followed by `execution.extra_tag` (when you set
one), the grid, the bed and a hash. This example sets no tag, so its folder is
`simulation_results/nsga2/1D_25z_packed_bed_0b76eb17`. The hash is computed from the whole
campaign definition: the base case, the variables and bounds, the constraints, the population
size, the generations, the seed and the stop rule (not the workers). Changing any of them
starts a new campaign in a new folder. The commands below use the folder of the example
campaign; for your own campaign, use the folder that `--dry-run` or the run prints.

The main files of the campaign folder:

- `resolved_config.json`, the base case;
- `results/`: the Pareto front (`pareto_front.csv`), the final population, the candidate
  records of every generation (`candidate_records/`), the figures (`figures/`, PDF),
  `optimization_summary.json`, and the checkpoint;
- `generation_log.txt` and `evaluations_log.txt`, the progress logs;
- `gen_NN/`, the candidate folders of the last generation. When a generation finishes,
  the candidate folders of the generation before it are deleted, just before the
  checkpoint is written.

In `pareto_front.csv`, the columns `eq_work` and `productivity` are the two objectives in the
units of the table at the top of this page.

A campaign never writes into another campaign's folder. Running the same command again is
refused, because the folder already holds this campaign:

```bash
pydac nsga2 --case my_case.json --bounds my_bounds.json --pop-size 4 --generations 2 --workers 4 --purity-min 0.5 --oxygen-ppm-max 1000 --steam-condensation || echo "exit code $?"
```

```text
Refused: simulation_results/nsga2/1D_25z_packed_bed_0b76eb17 already holds committed generations of this campaign; continue it with --resume, or change the seed or output root for a new one.
exit code 2
```

`pydac nsga2` has no `--overwrite`. To start a campaign from scratch in the same folder,
delete the folder yourself. A folder with NSGA-II results but no campaign identity, for
example files copied in by hand, is refused too.

## Continue a campaign

A campaign that was interrupted, for example with Ctrl+C or because the machine stopped,
continues from its last committed generation:

```bash
pydac nsga2 --resume simulation_results/nsga2/1D_25z_packed_bed_0b76eb17 --dry-run || echo "exit code $?"
```

The whole campaign definition is read from the folder, so `--resume` accepts only
`--workers` and `--dry-run`; any other option is refused. With `--dry-run` it runs every
check of a real continuation and says after which generation the campaign would continue.
The target stays the original number of generations. The example campaign is complete, so
it is refused:

```text
Refused: Campaign already completed 2 of 2 requested generations.
exit code 2
```

To run more generations, start a new campaign with a larger `--generations`; it gets a new
folder and starts from the first generation. A campaign interrupted during its first
generation has nothing committed: start it again with the same command.

A campaign continues only with the same pyDAC version and the same pymoo version that wrote
its checkpoint. Upgrading either in the middle of a campaign stops its continuation; the
refusal names both versions. The same seed gives the same campaign only with the same pymoo
version.

## Report a campaign

`--report CAMPAIGN` writes the Pareto figures (PDF) and `pareto_front.csv` again from the
saved results. It solves nothing and accepts no other option:

```bash
pydac nsga2 --report simulation_results/nsga2/1D_25z_packed_bed_0b76eb17
```

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | failure after the campaign started, including a first generation where every candidate failed; a traceback is printed |
| `2` | configuration error or refused request, one line on stderr; or invalid command-line arguments, after the usage. No campaign starts and no campaign folder is written |

## The same from Python

| Terminal | Python |
|---|---|
| `pydac nsga2 --case PATH --bounds BOUNDS` | `pydac.run_nsga2(pydac.load_case(PATH, exact=True), BOUNDS)` |
| `--pop-size N`, `--generations N`, `--workers N`, `--seed N` | `pop_size=N`, `generations=N`, `workers=N`, `seed=N` |
| `--purity-min X`, `--oxygen-ppm-max X`, `--eq-work-max X` | `purity_min=X`, `oxygen_ppm_max=X`, `eq_work_max=X` |
| `--steam-condensation` | `steam_condensation=True` |
| `--termination MODE --ftol X --period N --n-skip N` | `termination=MODE`, `ftol=X`, `period=N`, `n_skip=N` |
| `--max-evaluations N` | `max_evaluations=N` |
| `--resume CAMPAIGN --workers N` | `pydac.resume(CAMPAIGN, workers=N)` |
| `--report CAMPAIGN` | `pydac.report(CAMPAIGN)` |

`BOUNDS` is the path of a bounds file or a dictionary such as
`{"t_ads": [3600, 39000], ...}`. The calls return a `RunResult` whose `folder` is the campaign
folder, and raise instead of returning an exit code. With `workers` above 1 in a script, put
the call under `if __name__ == "__main__":`: on Windows and macOS the worker processes import
the script again.

The objectives, the decision variables of the table above and the options on this page are
what pyDAC 3.0.0 supports. The code has further options (other decision variables, warm
starts from an earlier campaign, crossover and mutation parameters); they are internal and
may change without notice.

## Related pages

- [Results](06_results.md): reading `pareto_front.csv` and the other campaign files
- [Running steps and cycles](04_running.md): CSS runs, which every candidate is
- [Cases and settings](03_cases.md)
- [06 — Post-processing](../manual/06_post_processing.md): how equivalent work, productivity and
  purity are computed
