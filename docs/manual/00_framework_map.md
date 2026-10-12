---
title: "00 — Framework Map"
toc: true
toc-depth: 3
embed-resources: true
---

> **This manual is the detailed technical reference: how pyDAC works inside, for contributors
> and for checking the model.** To learn to use pyDAC, do the [tutorials](../../examples/README.md)
> first, then read the [user guide](../guide/01_installation.md).

This chapter is the map of the pyDAC code for contributors: how the package is laid out, which
layer calls which, the files one run visits and what it writes. It stays at the level of files
and functions. To run pyDAC you do not need it: start with the
[user guide](../guide/02_quickstart.md).

- What an input means, and how a case is written and resolved:
  [01 — Inputs](01_inputs.md).
- The same run stage by stage, with the code of each stage:
  [02 — pyDAC Core](02_pydac_core.md).

## 1. Repository layout

The package source in a checkout is `src/pydac/`:

| Path | Role |
|---|---|
| `api.py` | The Python API: `load_case`, `export_case`, `export_settings`, `run_steps`, `run_css`, `run_nsga2`, `resume`, `report`, `RunResult`. The names exported by `pydac` (`src/pydac/__init__.py`) are the supported surface. |
| `cli.py` | The `pydac` command (`steps`, `css`, `nsga2`, `validate`, `get-started`). It parses flags, prints messages and maps errors to exit codes; the run commands delegate to `api.py`, `validate` to `config/cli.py`, and `get-started` to `starter_kit.py`. |
| `starter_kit.py` | `pydac.get_started(folder)` / `pydac get-started FOLDER` copy `README.md`, `LICENSE`, `CITATION.cff`, `docs/guide/` and `examples/` from the installed wheel or a checkout into a new or empty folder. Generated run output and caches are skipped; Markdown and notebook links to files outside the kit point to the public repository at `v<version>` (`main` for a development build). |
| `config/` | Case resolution and validation: settings, the immutable `ExperimentConfig`, the runtime records `CycleConfig` and `ColumnModelConfig`, serialization, the generated JSON Schema, the shared case flags. It does not import the solver. |
| `inputs/` | `inputs/defaults/` holds the default values, one module per domain; `case_io.py` reads and writes frozen cases and their fingerprint; `case_terminal.py` chooses between a frozen case and the defaults. |
| `workflows/` | The orchestrators: `orchestrator_TVSA_steps.py` (one step sequence), `orchestrator_TVSA_cycle_css.py` (cycles to cyclic steady state), `orchestrator_NSGA2.py` (a campaign). Internal: reachable, not supported API. |
| `core/` | The simulation engine: `TVSA_cycle.py`, `column_model_factory.py`, `column_model_input.py`, `column_model.py` (the PDE model), `solver.py` (CVODE), `state_layout.py`, isotherm and kinetic models, the species catalog, `bed_architectures/`, `sparsity_patterns/`. |
| `post_processing/` | Simulation context, step metrics, cycle KPIs, profile plots and campaign post-processing. |
| `utils/` | Support packages: `run/` (run folders, validation, resume), `tvsa/` (step sequences, termination, paths), `css/` (CSS loop and report), `nsga2/` (campaign evaluation, checkpoints), `solver/` (events, linear-solver options, outcomes), `postprocessing/`, `column/`, plus `provenance.py`, `logger.py`, `version.py`. |

A clone also has:

| Path | Role |
|---|---|
| `runners/` | One launcher per subcommand, for running and debugging from an editor ([CONTRIBUTING](../../CONTRIBUTING.md#running-from-vs-code)). |
| `tests/` | The test suite, its small cases and baselines. |
| `examples/` | The tutorials: notebooks in `notebooks/`, their case files in `cases/` (with `quick_1D.json`, the preset of the README quick start), measured data in `data/`. |
| `inputs/cases/` | Where you keep your own frozen cases; empty in the repository. |
| `docs/` | This documentation and `check_links.py`. |
| `ci/` | The qualification lock file used by continuous integration. |

## 2. Layer map

Execution flows downward. The case is resolved once and then passed down as immutable records.

```text
pydac steps | css | nsga2          (cli.py; a runner calls the same main())
        │
        ▼
pydac.api                          load_case, run_steps, run_css, run_nsga2
        │   preflight: validate the case, refuse an occupied run folder
        ▼
workflows/                         run_steps_simulation / run_css_simulation / NSGA-II campaign
        │   ExperimentConfig ──resolve_experiment_config()──▶ CycleConfig
        ▼
core/TVSA_cycle.py                 one TVSA_Cycle per sequence (per cycle in CSS)
        │
        ▼
core/column_model_factory.py       ColumnModelInput, one per step
        │
        ▼
core/column_model.py               ColumnModelMoL._bind_inputs()      ⚠ maintainer review
        │
        ▼
core/solver.py → saved results → post_processing/                     ⚠ maintainer review
```

Boundaries:

- `config/` owns how values are checked, resolved, frozen and serialized. It does not import
  the physical model.
- `ColumnModelMoL._bind_inputs()` binds the public scalar inputs to model attributes, each one
  explicitly, with no generic `setattr` loop; a few other construction methods
  (`set_simulation_parameters()` for the mixture, `_configured_mole_fractions()` for the
  reference and step compositions) also read the typed case.
- `src/pydac/core/column_model.py` and `src/pydac/post_processing/` define what pyDAC computes
  and reports; changes to them need the maintainer's review first
  ([CONTRIBUTING](../../CONTRIBUTING.md#physics-core-maintainer-review)).

## 3. Input paths

A run starts from one of two inputs: the defaults, optionally changed by a preset
(`--config`) and by settings (`--set`), or a frozen case (`--case`), loaded exactly. The
precedence and the case document are in
[01 — Inputs §1–§2](01_inputs.md#1-one-pipeline-two-ways-in).

## 4. Entry points

The `pydac` command and the Python API are the supported entry points. The four case-aware
subcommands (`steps`, `css`, `nsga2`, `validate`) share one set of case flags; `get-started FOLDER`
copies the starter kit. Both, the launchers in `runners/` and the internal workflows are
described in [01 — Inputs §3](01_inputs.md#3-entry-points).

## 5. Runtime Configuration Boundary

The resolved case, `ExperimentConfig`, is the last object a user writes. Everything below it in
the layer map of §2 narrows that one document (one campaign, then one step, then one model);
nothing downstream decides a value again or changes it. The document that was resolved is the
document that is saved and the document that runs.

A new public value starts in `src/pydac/inputs/defaults/`: the default tree drives the
accepted paths, basic types, serialization, frozen cases and the generated JSON Schema. If the
model must read the new value, its binding in `src/pydac/core/column_model.py` goes through
maintainer review. The whole chain, with the file and function at every step, is in
[02 — pyDAC Core §7](02_pydac_core.md#7-where-one-public-input-reaches-the-physics) and the
procedure in [01 — Inputs §10.1](01_inputs.md#101-adding-a-public-input).

## 6. What A Run Actually Does

### 6.1 The whole picture

A fresh step run, `pydac steps --case my_case.json`:

```text
 pydac steps --case my_case.json           (or Run on runners/run_steps_simulation.py)
  │
  ├─ 1. RESOLVE THE CASE ─────────────────────────────────────────────────────
  │    cli.py :: _run_fresh() → inputs/case_terminal.py :: resolve_runner_experiment()
  │    frozen case: inputs/case_io.py · defaults + --config + --set: config/loader.py
  │        └─▶ ExperimentConfig                 (immutable, validated, complete)
  │
  ├─ 2. SHORT-CIRCUITS (stop here, exit 0) ───────────────────────────────────
  │    --export-case     → write a frozen case
  │    --print-resolved  → print the case
  │
  ├─ 3. PREFLIGHT (writes nothing; a configuration error or refusal is exit 2) ─
  │    api.py :: _check_fresh_run()
  │    ├─ utils/run/validation.py :: validate_step_case()   (CSS: validate_css_case)
  │    ├─ utils/run/validation.py :: warn_advanced_sparsity_options()
  │    └─ api.py :: run_folder() → utils/run/config.py :: make_base_folder()
  │           └─▶ "simulation_results/step_mode/[TAG_]packed_bed_HASH"
  │    --dry-run  → print the run folder, STOP (exit 0)
  │    api.py :: _claim()   refuse an occupied folder, or delete it with --overwrite
  │
  ├─ 4. ENTER THE WORKFLOW ───────────────────────────────────────────────────
  │    api.py :: _solve_steps()
  │    └─ workflows/orchestrator_TVSA_steps.py :: run_steps_simulation()
  │       ├─ config/launch.py :: resolve_experiment_config()
  │       │      └─▶ (ExperimentConfig, CycleConfig)        ← the runtime view
  │       ├─ utils/tvsa/paths.py :: resolve_spatial_base_folder()
  │       │      └─▶ "<run folder>/1D_mode" or "<run folder>/2D_mode"
  │       ├─ config/loader.py :: write_resolved_config()
  │       │      └─▶ resolved_config.json + case_metadata.json   (before anything runs)
  │       ├─ utils/provenance.py :: start_activity()  → provenance/<id>.json
  │       └─ core/TVSA_cycle.py :: TVSA_Cycle(cycle_config, ...)
  │              ├─ __init__  →  ColumnModelFactory(cycle_config)    (built once)
  │              └─ .run_cycle(...)
  │                     ├─ utils/tvsa/cycle_steps.py :: build_cycle_steps()
  │                     │      └─▶ ["startup", "blow", "heat", "s_des", "cool", "press", "ads"]
  │                     └─ for step in steps: ._run_step(step)  ── REPEATS PER STEP ──
  │                            │
  │                            │ 5. BUILD THE MODEL
  │                            ├─ column_factory.build_column_model(step=..., initial_state=..., solver_options=...)
  │                            │      └─▶ ColumnModelInput → ColumnModelMoL      ⚠
  │                            │
  │                            │ 6. SOLVE (solve=true)
  │                            ├─ core/solver.py :: Solver(adsorber, run_base_folder)
  │                            ├─ SimulationContext.from_model(), runtime report     ⚠
  │                            ├─ solver.solve_cvode()   CVODE (BDF), direct sparse solver (dense if use_sparsity=false)
  │                            │      └─▶ <step>/provenance/   (the solve record)
  │                            ├─ solver.save_results()  → <step>/sol.pkl
  │                            └─ context and runtime report saved after a successful solve
  │                            │
  │                            │ 7. POST-PROCESS THE STEP                            ⚠
  │                            ├─ StepPostProcessor(adsorber, solver).compute_step_metrics()
  │                            ├─ StepMetricsStore.save()  → <step>/kpis.csv, kpis.pkl
  │                            └─ plotter.plot_all_profiles()  only with plot_profiles
  │                            │
  │                            └─▶ final fields become the initial state of the NEXT step
  │
  └─ 8. RETURN ───────────────────────────────────────────────────────────────
       RunResult(folder, results_folder, workflow); the last line prints the run folder
```

Two things to keep in mind before the detail:

1. **The case is resolved, validated and written to disk before any model is built.**
   Stage 3 checks the execution settings and the mixture requirements and writes nothing;
   `resolved_config.json` is written in stage 4, before stage 5. The checks made while the
   model is built and the saved-context compatibility checks come later, still before any
   integration. Whatever fails later, the exact input of the run is already on disk.
2. **The per-step loop (stages 5–7) is the same for every run.** A CSS run and an NSGA-II
   campaign do not reimplement solving, context capture, step metrics or plotting: they wrap
   `TVSA_Cycle` in an outer loop (§8).

Each stage, with what it reads, decides and writes, is in
[02 — pyDAC Core §2–§3](02_pydac_core.md#2-from-the-command-or-api-to-the-cycle-runner).

### 6.2 The files a run visits

Grouped by role, roughly in the order a fresh step run first reaches them (imports and calls
revisit several). All paths are under `src/pydac/`.

| # | File | Role in a run |
|---|---|---|
| 1 | [`cli.py`](../../src/pydac/cli.py) | `main()` parses the flags, prints the environment warning, runs the short-circuits, the preflight and `--dry-run`, maps a configuration error or a refusal to exit 2. A runner in `runners/` calls the same `main()` with its `LauncherSettings`. |
| 2 | [`config/terminal.py`](../../src/pydac/config/terminal.py) | The shared case flags and the helpers that export or print a case. |
| 3 | [`inputs/case_terminal.py`](../../src/pydac/inputs/case_terminal.py) | `resolve_runner_experiment()`: frozen case or defaults, then `inputs.case_io.load_case` or `config.loader.load_experiment`. |
| 4 | [`config/loader.py`](../../src/pydac/config/loader.py) | Resolves defaults, preset and settings into the immutable `ExperimentConfig`; also `write_resolved_config()`. |
| 5 | [`api.py`](../../src/pydac/api.py) | `_check_fresh_run()` (validation, sparsity warning, run folder), `_claim()` (occupied folder, `--overwrite`), `_solve_steps()` / `_solve_css()`, `RunResult`. |
| 6 | [`utils/run/validation.py`](../../src/pydac/utils/run/validation.py) | `validate_step_case()` and `warn_advanced_sparsity_options()`; the CSS check `validate_css_case()` is in `utils/css/runner.py`. |
| 7 | [`utils/run/config.py`](../../src/pydac/utils/run/config.py) | `make_base_folder()`: the run folder from the output root, the run mode, `extra_tag` and the first eight characters of the case fingerprint. |
| 8 | [`workflows/orchestrator_TVSA_steps.py`](../../src/pydac/workflows/orchestrator_TVSA_steps.py) | `run_steps_simulation()`: the runtime view, the results folder, the resolved case on disk, one `TVSA_Cycle`. It also sets the numerical libraries to one thread. |
| 9 | [`config/launch.py`](../../src/pydac/config/launch.py) | `resolve_experiment_config()`: the one place `ExperimentConfig` becomes `CycleConfig`, for every workflow. |
| 10 | [`config/cycle.py`](../../src/pydac/config/cycle.py) | `CycleConfig` (one campaign) and `ColumnModelConfig` (one step): the runtime records the factory reads. |
| 11 | [`utils/tvsa/paths.py`](../../src/pydac/utils/tvsa/paths.py) | `resolve_spatial_base_folder()`: `1D_mode` for `grid.nr = 1`, else `2D_mode`. |
| 12 | [`utils/provenance.py`](../../src/pydac/utils/provenance.py) | `start_activity()`: the run's provenance record. |
| 13 | [`utils/tvsa/cycle_steps.py`](../../src/pydac/utils/tvsa/cycle_steps.py) | `build_cycle_steps()` / `build_repeating_cycle_steps()`: `execution.cycle_steps` as the ordered list of steps. |
| 14 | [`core/TVSA_cycle.py`](../../src/pydac/core/TVSA_cycle.py) | `TVSA_Cycle`: the per-step loop, the step log, context capture, the hand-off of fields between steps. |
| 15 | [`core/column_model_factory.py`](../../src/pydac/core/column_model_factory.py) | `ColumnModelFactory`: builds the isotherm and kinetic objects once per `TVSA_Cycle`, then one `ColumnModelMoL` per step. |
| 16 | [`core/column_model_input.py`](../../src/pydac/core/column_model_input.py) | `ColumnModelInput`, `AdsorptionModelBundle`: the plain data passed into the model. |
| 17 | [`core/column_model.py`](../../src/pydac/core/column_model.py) | `ColumnModelMoL`: the PDE model. ⚠ Maintainer review. |
| 18 | [`core/solver.py`](../../src/pydac/core/solver.py) | `Solver`: runs CVODE with its events, writes the step's solve record, saves and reloads `sol.pkl`, returns dimensional fields. |
| 19 | [`post_processing/context/`](../../src/pydac/post_processing/context/) | `SimulationContext`, `SimulationContextStore`, `RuntimeReportStore`: `simulation_context.pkl` and `resolved_runtime.json`. ⚠ |
| 20 | [`post_processing/profiles/step_post_processing.py`](../../src/pydac/post_processing/profiles/step_post_processing.py) | `StepPostProcessor`: uses the prepared results in memory when present, else loads the saved step; computes `StepMetrics`. ⚠ |
| 21 | [`post_processing/metrics/step_metrics_store.py`](../../src/pydac/post_processing/metrics/step_metrics_store.py) | `StepMetricsStore`: prints and saves `kpis.csv`, `kpis.pkl` and `metrics_provenance.json`. ⚠ |
| 22 | [`post_processing/profiles/step_plotting.py`](../../src/pydac/post_processing/profiles/step_plotting.py) | `ProfilesPlotter.plot_all_profiles()`: the profile plots under `<step>/profiles/`. ⚠ |
| 23 | [`utils/logger.py`](../../src/pydac/utils/logger.py) | `log_to()`: copies everything printed during one step into `<step>/log.txt`. |

Files 1–16 are configuration and orchestration. Files 17 and 19–22 are the physics and
reporting core marked ⚠: read them freely, change them only after maintainer review. The rule
covers the whole `post_processing/` folder, including the files that only save or plot.

### 6.3 The condensed call stack

```text
cli.main()                                       cli.py
 └─ _steps() → _run_fresh()
     ├─ resolve_runner_experiment()              inputs/case_terminal.py
     │   └─ load_case() | load_experiment()      inputs/case_io.py | config/loader.py
     ├─ api._check_fresh_run()                   api.py
     │   ├─ validate_step_case()                 utils/run/validation.py
     │   ├─ warn_advanced_sparsity_options()     utils/run/validation.py
     │   └─ make_base_folder()                   utils/run/config.py
     ├─ api._claim()                             api.py
     └─ api._solve_steps()                       api.py
         └─ run_steps_simulation()               workflows/orchestrator_TVSA_steps.py
             ├─ resolve_experiment_config()      config/launch.py
             ├─ resolve_spatial_base_folder()    utils/tvsa/paths.py
             ├─ write_resolved_config()          config/loader.py
             ├─ start_activity()                 utils/provenance.py
             └─ TVSA_Cycle(...)                  core/TVSA_cycle.py
                 ├─ ColumnModelFactory(cycle_config)   core/column_model_factory.py
                 └─ .run_cycle()
                     ├─ build_cycle_steps()      utils/tvsa/cycle_steps.py
                     └─ for step in steps: ._run_step()
                         ├─ log_to()             utils/logger.py
                         ├─ column_factory.build_column_model()
                         │   └─ ColumnModelMoL(...)          core/column_model.py      ⚠
                         ├─ Solver(adsorber, ...)            core/solver.py
                         ├─ SimulationContext.from_model()   post_processing/context/  ⚠
                         ├─ RuntimeReportStore.from_model()  post_processing/context/  ⚠
                         ├─ solver.solve_cvode()
                         ├─ solver.save_results()
                         ├─ StepPostProcessor(...)           post_processing/profiles/ ⚠
                         │   ├─ .describe()
                         │   └─ .compute_step_metrics()
                         ├─ StepMetricsStore.save()          post_processing/metrics/  ⚠
                         └─ plotter.plot_all_profiles()      post_processing/profiles/ ⚠
```

## 7. Artifacts On Disk

The step run of §6, on the 1D case `pydac validate --set grid.nz=25 --set grid.nr=1
--export-case my_case.json`, writes:

```text
simulation_results/
└── step_mode/                          ← run mode
    └── packed_bed_10b76906/            ← run folder: [TAG_]packed_bed_ + case fingerprint
        └── 1D_mode/                    ← results folder: grid.nr = 1
            ├── resolved_config.json    ← written first, before any model is built
            ├── case_metadata.json      ← fingerprint of the case
            ├── provenance/             ← the run's provenance record
            ├── simulation_context.pkl  ← shared by all steps
            ├── resolved_runtime.json   ← runtime report, merged step by step
            ├── startup/
            │   ├── log.txt
            │   ├── sol.pkl
            │   ├── provenance/
            │   ├── kpis.csv
            │   ├── kpis.pkl
            │   ├── metrics_provenance.json
            │   └── profiles/           ← always created; plots only with execution.plot_profiles
            └── blow/  heat/  s_des/  cool/  press/  ads/   (same entries)
```

Its `case_metadata.json`:

```json
{
  "canonicalization": "json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + newline",
  "case_fingerprint": "10b769060a287d421bb2b1f6a9eea0661a40ac32206e8ba9293b0c369bd59576",
  "schema_version": 8
}
```

Who writes what, and when:

| File | Written by | When |
|---|---|---|
| `resolved_config.json` | `config.loader.write_resolved_config()` | once, before `TVSA_Cycle` is built |
| `case_metadata.json` | `inputs.case_io.write_case_metadata()` | with `resolved_config.json` |
| `provenance/<id>.json` | `utils.provenance.start_activity()` | when the run starts |
| `simulation_context.pkl` | `SimulationContextStore.save()` | after each solved step, checked against the saved one |
| `resolved_runtime.json` | `RuntimeReportStore.save()` | after each solved step, checked and merged |
| `<step>/log.txt` | `utils.logger.log_to()` | while the step runs |
| `<step>/provenance/<id>.json` | `Solver.solve_cvode()` | when the step's solve starts |
| `<step>/sol.pkl` | `Solver.save_results()` | for a step that is solved |
| `<step>/kpis.csv`, `kpis.pkl`, `metrics_provenance.json` | `StepMetricsStore.save()` | every step, also when a saved step is reused; not in the snapshot replay of `report`, which only draws plots |
| `<step>/profiles/**` | the three subfolders: `StepPostProcessor`, always; the figures: `ProfilesPlotter.plot_all_profiles()` | figures only when the step plots (§8) |

Feeding `resolved_config.json` back with `--case` runs the same case again. The layout of a
CSS run and of a campaign, and how to read them, are in the user guide,
[Results](../guide/06_results.md); the step files in
[06 — Post-Processing §21](06_post_processing.md#21-file-outputs).

## 8. How CSS And NSGA-II Differ

`TVSA_Cycle._run_step()` is the same in every run; only what wraps it changes:

| | `pydac steps` | `pydac css` | `pydac nsga2` |
|---|---|---|---|
| Workflow | `run_steps_simulation()` | `run_css_simulation()` | `orchestrator_NSGA2.py`, one `run_css_simulation()` per candidate |
| `TVSA_Cycle` | one, `cyclic_mode=False` | one per cycle, `cyclic_mode=True`, until CSS or `execution.max_cycles` | one per cycle of each candidate |
| Step sequence | `build_cycle_steps()`: `startup` + your steps | `build_repeating_cycle_steps()`: must start with `ads`; cycle 1 replaces it with `startup`, later cycles repeat `ads → … → press` | as CSS, for each candidate |
| Profile plots | every step, with `execution.plot_profiles` | `plot_profiles`: step profiles in every cycle where it is true (globally or in `execution.cycle_options`); `plot_final_cycle_profiles`: the reported cycle, from saved solutions; `plot_cyclic_profiles` and `--report`: the CSS figures in `plots_css/` | candidates run with `plot_profiles` false; campaign figures in `results/figures/` |
| `resolved_config.json` | one, in the results folder | one, in the results folder | one per candidate folder while it runs; the candidate folder is then emptied and the case is kept in its record in `results/candidate_records/` |

The campaign is described in [07 — NSGA-II Optimization](07_nsga2_optimization.md), the step
sequences in [05 — Cycle Steps §1](05_cycle_steps.md#1-cycle-construction).

## 9. Where Do I Put My Hands?

| I want to… | Touch |
|---|---|
| Change a value for one run | `--set`, a preset with `--config`, or `DEVELOPMENT_SETTINGS` in a runner |
| Change a default value of the package | `src/pydac/inputs/defaults/<domain>.py` |
| Run a saved case again | `--case <file>.json` |
| Save a case to run again | `--export-case <file>.json` |
| Add a public input | [01 — Inputs §10.1](01_inputs.md#101-adding-a-public-input) |
| Add a species, an isotherm or a kinetic model | [CONTRIBUTING](../../CONTRIBUTING.md#extending-the-model) and [01 — Inputs §10.2](01_inputs.md#102-adding-or-changing-an-isotherm-or-kinetic-model); an isotherm or kinetic law from user code: [Extending pyDAC](../guide/08_extending.md) |
| Change the optimizer bounds, workers or seed | the `pydac nsga2` flags ([user guide](../guide/05_optimization.md)) |
| Add a bed architecture | `src/pydac/core/bed_architectures/` and its registry ([01 — Inputs §10.3](01_inputs.md#103-packed-bed-architecture-ownership)) |
| Change a PDE term, closure or boundary condition | ⚠ `src/pydac/core/column_model.py`, maintainer review |
| Change a KPI formula, the context schema or a post-processing output | ⚠ `src/pydac/post_processing/`, maintainer review |

When something went wrong:

| Symptom or question | Look at |
|---|---|
| A setting I changed had no effect | [01 — Inputs §2](01_inputs.md#2-the-mental-map), precedence |
| Which folder did my run write to | the `Run folder:` line printed at start and end, or §7 |
| A step says "no saved solution exists" | `solve=false` for a step without `<step>/sol.pkl` in that results folder ([02 §4](02_pydac_core.md#4-solve-vs-reuse-per-step)) |
| The exact case that ran, without running again | `resolved_config.json` in the results folder, or `pydac validate --case <that file> --print-resolved` |
| A KPI looks wrong | [06 — Post-Processing](06_post_processing.md) for the formula; `src/pydac/post_processing/` for the code ⚠ |
| The model must read a new input | [01 — Inputs §10](01_inputs.md#10-extending-the-contract); `_bind_inputs()` is under maintainer review |
| No profile plots for an intermediate CSS cycle | `plot_profiles` was false for that cycle; `plot_final_cycle_profiles` draws only the reported cycle (§8) |
| Where is the PDE itself | mapped to the code in [04 — Governing Equations](04_governing_equations.md); assembled by the RHS of [02 — pyDAC Core](02_pydac_core.md) |

## 10. Compactness And Ownership Guards

A new responsibility gets one cohesive owner instead of growing the command, the API or the
configuration engine. `cli.py` and `api.py` stay thin: they decide and report, the workflows
and `core/` do the work. The runners in `runners/` only hold a launcher's settings.
