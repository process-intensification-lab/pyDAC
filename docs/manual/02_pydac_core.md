---
title: "02 — pyDAC Core"
toc: true
toc-depth: 3
embed-resources: true
---

> **This manual is the detailed technical reference: how pyDAC works inside, for contributors
> and for checking the model.** To learn to use pyDAC, do the [tutorials](../../examples/README.md)
> first, then read the [user guide](../guide/01_installation.md).

This chapter traces a step run, `pydac steps --case PATH` or `pydac.run_steps(case)`, stage by
stage: case resolution, the preflight, the workflow, model construction, the per-step loop and
the solver. It is the detailed counterpart of the diagram in
[00 — Framework Map §6](00_framework_map.md#6-what-a-run-actually-does). All paths are under
`src/pydac/` unless stated otherwise.

What an input means is in [01 — Inputs](01_inputs.md); how to run a case is in the user guide,
[Running steps and cycles](../guide/04_running.md).

## 1. Runtime Construction

The resolved case is narrowed before the core sees it:

```text
ExperimentConfig
  └─ CycleConfig.from_experiment(...)
       └─ ColumnModelFactory(CycleConfig)
            └─ ColumnModelInput
                 └─ ColumnModelMoL
```

`CycleConfig` is the runtime view of one resolved case: it holds the simulation sections,
the per-step sparsity controls and the case it came from. The factory builds the equilibrium
and kinetic components once, then one `ColumnModelMoL` per step.

## 2. From the command or API to the cycle runner

### 2.1 The command and the launchers

`pydac` runs `cli.main()` ([`cli.py`](../../src/pydac/cli.py)). It installs a warning
format (pyDAC's own `UserWarning`s print as one `Warning: …` line), parses the subcommand and
its flags, prints the dedicated-environment warning when pyDAC runs outside a virtual or conda
environment, and dispatches: `steps` → `_steps()` → `_run_fresh(args, "steps")` for a fresh
run, or `_run_resume()` with `--resume`.

The launchers in `runners/` of a clone call the same `main()`. They pass their constants
(`CASE_PATH`, `CONFIG_PATH`, `DEVELOPMENT_SETTINGS`, and the resume and overwrite constants)
as one `LauncherSettings`; `main()` fills an unset `--config` or `--resume` from it. With
`argv=None` the launcher forwards the command-line arguments, so a launcher run from an editor
without arguments uses the flag defaults and the constants as written
([01 — Inputs §3.2](01_inputs.md#32-the-launchers-in-runners-contributors)).

`pydac.run_steps(case)` ([`api.py`](../../src/pydac/api.py)) enters at §2.3 with a case
already loaded by `pydac.load_case()`; it has no short-circuits.

### 2.2 Resolving the case

`cli._resolve()` calls `resolve_runner_experiment()`
([`inputs/case_terminal.py`](../../src/pydac/inputs/case_terminal.py), re-exported by
`pydac.config.terminal`):

- without `--case` (and without a launcher `CASE_PATH`), the case is built from the built-in
  defaults, an optional `--config` preset, the launcher settings when present, and then the
  `--set` settings, by `config.loader.load_experiment()`;
- with a case, `inputs.case_io.load_case()` loads the complete frozen case without inheriting
  any default. Frozen mode refuses `--config`, `--set` and non-empty launcher settings.

Inside `load_experiment()` ([`config/loader.py`](../../src/pydac/config/loader.py)) the
input schema is upgraded, the preset is merged into the reference profile (the defaults of
`inputs/defaults/`), bed aliases and owned defaults are resolved, the settings are applied, the
schema version is checked, the derived fields are resolved, and the typed case is built and
validated. The stages and their rules are in
[01 — Inputs §2.1](01_inputs.md#21-resolution-how-values-become-a-typed-configuration).
Either way the result is one immutable `ExperimentConfig`; nothing downstream changes it.

### 2.3 The short-circuits and the preflight

`cli._run_fresh()`, condensed:

```python
def check():
    _check_fresh_flags(args)
    case, mode = _resolve(args)                       # §2.2
    if export_case_if_requested(args, case) is not None:
        return None                                   # --export-case: exit 0
    if args.print_resolved:
        print_mode_summary(mode)
        print_resolved_document(case, dry_run=False)
        return None                                   # --print-resolved: exit 0
    folder = api._check_fresh_run(case, workflow)     # validation; writes nothing
    if args.dry_run:
        ...                                           # print the run folder
        print("Dry run validation passed: configuration is valid; simulation can start.")
        return None                                   # --dry-run: exit 0
    api._claim(folder, overwrite=args.overwrite)      # occupied folder: refuse or delete
    return case, folder

planned = _preflight(check)   # ConfigurationError / RunRefused -> exit 2
```

`api._check_fresh_run()` is the preflight shared by the command and the API:

1. `api._validator()` selects `validate_step_case()`
   ([`utils/run/validation.py`](../../src/pydac/utils/run/validation.py)) for a step run, or
   `validate_css_case()` ([`utils/css/runner.py`](../../src/pydac/utils/css/runner.py)) for
   a CSS run. They check the execution settings and the executed steps against the mixture;
   the CSS check also requires a step that collects product.
2. `warn_advanced_sparsity_options()` warns once for each diagnostic solver setting:
   `execution.use_sparsity=false`, `execution.compute_numerical_sparsity=true`, or a
   non-default `execution.cvode_sparsity_mode`.
3. `api.run_folder()` names the run folder (§2.4).

So an invalid combination (for CSS, `plot_profiles=true` with `keep_only_final_cycle=true`)
fails before a folder is created or the workflow is entered, both under `--dry-run` and in a
real run. `--export-case` and `--print-resolved` skip the preflight: they inspect or export,
they do not launch. Every configuration error and refusal of the preflight prints
`Configuration error: …` or `Refused: …` and exits with code `2`.

### 2.4 Naming the run folder

```python
make_base_folder(
    output_root=case.execution.output_root,   # "simulation_results"
    cycle=workflow == CSS,
    extra_tag=case.execution.extra_tag,       # e.g. "TEST"
    experiment_config=case,
)
```

`make_base_folder()` ([`utils/run/config.py`](../../src/pydac/utils/run/config.py)) returns
`<output_root>/<mode>/<leaf>` without creating it:

- `mode` is `step_mode` for a step run and `cyclic_mode` for a CSS run;
- `leaf` is `[<extra_tag>_]packed_bed_<hash8>`, where `hash8` is the first eight characters of
  the case fingerprint. The full SHA-256 fingerprint is written to `case_metadata.json` (§2.7);
  no setting is repeated in the name.

With `extra_tag = "TEST"`:

```text
simulation_results/step_mode/TEST_packed_bed_<hash8>
```

A fresh run refuses a run folder that exists and is not empty. `--overwrite`
(`overwrite=True`) deletes exactly that folder first; `--resume` (`pydac.resume()`) continues
or replays the saved run ([user guide](../guide/04_running.md#run-folders)). The name does not
contain `1D_mode`/`2D_mode` yet: that split depends on the grid and is made in the workflow.

### 2.5 Entering the workflow: `run_steps_simulation()`

`api._solve_steps()` imports the workflow, prints the run folder, calls the workflow and prints
the run folder again when it returns:

```python
run_steps_simulation(experiment_config=case, base_folder=folder)
```

[`workflows/orchestrator_TVSA_steps.py`](../../src/pydac/workflows/orchestrator_TVSA_steps.py)
sets the numerical libraries to one thread when it is imported (`OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, … = `1`), so every step run sets them, a replay with
`execution.solve=false` included; validation and `--dry-run` never import it. It then
derives the runtime view and reads the run options **from the case**, not from keyword
arguments:

```python
resolved, cycle_config = resolve_experiment_config(experiment_config=experiment_config)
execution = resolved.execution
solve, plot_profiles, use_sparsity, ... = execution.solve, execution.plot_profiles, ...
```

`resolve_experiment_config()` ([`config/launch.py`](../../src/pydac/config/launch.py))
returns the case and `CycleConfig.from_experiment(case)` together, so the document written to
disk and the runtime record that drives the solve cannot disagree.

`validate_flags()` (`utils/run/validation.py`) then checks the option structure, the
termination-mode values and the flag types. It returns `True` or raises; there is no silent
early return. `compute_numerical_sparsity=true` with `use_sparsity=false` is accepted, as a
diagnostic combination.

The internal workflow can be called from a script, but it does not run the preflight of §2.3
or the occupied-folder check; use `pydac.run_steps(case)`
([01 — Inputs §3.3](01_inputs.md#33-the-internal-workflows)).

### 2.6 Deciding `1D_mode` vs. `2D_mode`

```python
results_base_folder = resolve_spatial_base_folder(base_folder, grid_config=cycle_config.grid)
```

`resolve_spatial_base_folder()` ([`utils/tvsa/paths.py`](../../src/pydac/utils/tvsa/paths.py))
appends `1D_mode` when `grid.nr == 1`, else `2D_mode`, and creates the folder. For
`grid.nr = 10` the folder of §2.4 becomes:

```text
simulation_results/step_mode/TEST_packed_bed_<hash8>/2D_mode
```

This is the results folder, `RunResult.results_folder`.

### 2.7 Writing the case to disk, before anything runs

```python
write_resolved_config(resolved, os.path.join(results_base_folder, "resolved_config.json"))
start_activity(results_base_folder, "run", config=resolved,
               execution={"workflow": "steps", "solve": solve})
```

`write_resolved_config()` (`config/loader.py`) writes the case as strict, indented JSON and
calls `inputs.case_io.write_case_metadata()`, which writes `case_metadata.json` next to it;
the fingerprint is computed on a separately canonicalized form of the case.
`utils.provenance.start_activity()` then writes the run's provenance record into
`provenance/`. Both happen before `TVSA_Cycle` is built: if a later step fails, the case that
was attempted is on disk. Load it with `--case` to run the same inputs in a fresh folder, or
continue the saved run with `--resume`. The full list of files is in
[00 — Framework Map §7](00_framework_map.md#7-artifacts-on-disk).

### 2.8 Building the cycle runner

```python
tvsa_cycle = TVSA_Cycle(cycle_config, base_folder=base_folder,
                        print_to_terminal=print_to_terminal, cyclic_mode=False)
```

`TVSA_Cycle.__init__` ([`core/TVSA_cycle.py`](../../src/pydac/core/TVSA_cycle.py)) checks
the runtime record, creates the `ColumnModelFactory`, makes sure `base_folder` exists, and
stores the cycle position (`cyclic_mode=False`, `cycle_idx=None` for a step run), the
logging, spatial-root, composition-guard and persistence settings, and the bookkeeping of the
final profiles. It builds no step model.

`ColumnModelFactory.__init__` checks and stores the `CycleConfig` and prints its setup banner.
Its `components` property builds the equilibrium and kinetic models selected by the case, from
the equilibrium and kinetic registries, on first access (inside the first `_run_step()`), and
reuses the same `AdsorptionModelBundle` for every later step.

`base_folder` here is the run folder without `1D_mode`: every `_run_step()` derives the
spatial split again (`resolve_spatial_base_folder_from_solver_settings`), so `TVSA_Cycle`
does not need the grid at construction. Sequence, path, reporting and CSS helpers live in
`utils/tvsa/` and `utils/css/`; the cycle class owns only the lifecycle.

### 2.9 Building the step sequence

```python
final_fields, step_records, cycle_total_time = tvsa_cycle.run_cycle(
    solve=solve, plot_profiles=plot_profiles, cycle_steps=cycle_steps,
    step_options=step_options, termination_mode=termination_mode,
    use_sparsity=use_sparsity, compute_numerical_sparsity=compute_numerical_sparsity,
    print_cycle_build=print_cycle_build,
)
```

`TVSA_Cycle.run_cycle()` is internal: its keyword arguments carry the options the workflow read
from the case. It turns `execution.cycle_steps` (a list of step names, or `null`) into the
ordered list with `build_cycle_steps()`
([`utils/tvsa/cycle_steps.py`](../../src/pydac/utils/tvsa/cycle_steps.py)):

```python
DEFAULT_STEP_MODE_SEQUENCE = (
    "startup", "blow", "heat", "s_des", "cool", "press", "ads"
)
```

`build_cycle_steps()` always puts `startup` first and drops a `startup` you listed yourself.
`execution.cycle_steps = null` means this default; an explicit list may use the accepted
aliases (for example `ADSORPTION` for `ads`), each normalized to its short name by
`normalize_step_token`
([01 — Inputs §2.3](01_inputs.md#23-the-one-distinction-that-causes-most-of-the-confusion)).
With `print_cycle_build` true (the default), `_print_cycle_build_summary()` prints the sequence
and the per-step overrides before anything solves: the block printed right after "Starting
Adsorption Cycle".

## 3. The Per-Step Loop — `_run_step()`

`run_cycle()` calls `self._run_step(step, ...)` once per step, the same way for every step
name, with the previous step's final fields as `prev_fields`.

**a) Where does this step write?**

```python
run_base_folder = _resolve_spatial_base_folder_from_solver_settings(
    self.base_folder, grid_config=self.column_factory.config.grid, ...
)   # → .../2D_mode  (the split of §2.6, derived again)
```

**b) Everything printed during the step is copied to a file:**

```python
with log_to(step_name, run_base_folder, also_stdout=self.print_to_terminal, append=not solve):
    ...
```

`log_to()` ([`utils/logger.py`](../../src/pydac/utils/logger.py)) writes
`<run_base_folder>/<step>/log.txt` for the duration of the block, and still prints to the
terminal when `also_stdout` is true (the default). A step that reuses its saved solution
appends to its log instead of replacing it.

**c) Reuse check**, only for `solve=false`: if `<step>/sol.pkl` does not exist, `_run_step()`
raises `FileNotFoundError` at once instead of skipping the step.

**d) Build this step's model:**

```python
solver_options = self.column_factory.config.solver_for_step(
    use_sparsity=use_sparsity, compute_numerical_sparsity=compute_numerical_sparsity,
)
adsorber = self.column_factory.build_column_model(
    step=step_name, initial_state=prev_fields, solver_options=solver_options,
)
```

`ColumnModelFactory.build_column_model()`
([`core/column_model_factory.py`](../../src/pydac/core/column_model_factory.py)):

```text
CycleConfig.step_config(step)        → ColumnModelConfig      (this step's slice)
   +  self.components                → AdsorptionModelBundle  (built once, cached)
   +  initial_state = prev_fields
        ▼
ColumnModelInput(config, components, initial_state)   (core/column_model_input.py, plain data)
        ▼
ColumnModelMoL(column_input)                          ⚠ core/column_model.py, maintainer review
```

`CycleConfig.step_config(step)` ([`config/cycle.py`](../../src/pydac/config/cycle.py))
assembles the seven groups of the step (`composition, flow, thermal, pressure, horizon,
termination, outlet`, see
[01 — Inputs §4.2](01_inputs.md#42-a-step-ten-names-seven-groups-fixed-order)) with the
geometry, wall, sorbent, reference, grid, adsorption and solver settings into the one object
the model accepts. `ColumnModelMoL.__init__` is §6.

If a termination mode was set for the step, `adsorber.termination_mode` is set right after
construction; then one line describing the stop criterion the model will use is printed
(`_describe_termination_setup_from_model`).

**e) Wrap it in a solver:**

```python
solver = Solver(adsorber, base_folder=run_base_folder, composition_domain_guard=...)
```

`Solver.__init__` ([`core/solver.py`](../../src/pydac/core/solver.py)) does not solve: it
reads grid dimensions, reference scales and tolerances, and builds `self.event_specs` from the
model's termination settings (`build_event_specs_from_model`), so an event step (for example
one that stops when `avg_outlet_temperature` crosses its threshold) is set up before
`solve_cvode()` is called.

With event termination and `stop_on_pressure_plateau=true`, blowdown and pressurization add a
pressure-rate plateau fallback to their configured primary event, from
[`utils/solver/pressure_plateau.py`](../../src/pydac/utils/solver/pressure_plateau.py).
The blowdown fallback evaluates `-d(mean(p/p_ref))/d(t/tau_hyd)` and the pressurization
fallback evaluates `d(mean(p/p_ref))/d(t/tau_hyd)` from the RHS. Their `50000 Pa` settings are
stored and printed separately as arming conditions; the terminal threshold stored in
`sol.pkl` remains the rate tolerance (`1e-5`), never the arming pressure.

**f) If `solve=true`, run it** (condensed; the full run persistence of a step or CSS run):

```python
if solve:
    simulation_context = SimulationContext.from_model(adsorber)
    context_store = SimulationContextStore(run_base_folder)
    runtime_store = RuntimeReportStore(run_base_folder)
    runtime_report = runtime_store.from_model(adsorber)
    if context_store.exists():
        context_store.validate_compatible(simulation_context)
    if runtime_store.exists():
        runtime_store.validate_compatible(runtime_report)
    solver.solve_cvode()
    solver.save_results()
    require_final_solver_success(solver.sol, step=step_name)
    context_store.save(simulation_context)
    runtime_store.save(runtime_report)
```

- `SimulationContext.from_model()` captures the resolved static parameters and this step's
  operating parameters in one canonical table. Post-processing (cycle KPIs, NSGA-II reports)
  reads it back later, independently of `resolved_config.json`.
- `validate_compatible(...)` is why the steps of one cycle can share a folder (their contexts
  merge and must agree), while reusing a folder for a different case fails loudly instead of
  overwriting it.
- The runtime report records the generated state layout, the selected model implementations,
  catalog data, the evaluation state, and unit-bearing scalar, per-species and pairwise
  properties as readable JSON. Spatial histories stay in `<step>/sol.pkl`.
- `solver.solve_cvode()` writes the step's solve record into `<step>/provenance/`, builds the
  CVODE options (BDF, positivity constraints, per-block tolerances, the event callback if
  any), integrates, and keeps the trajectory in `solver.sol`.
- `solver.save_results()` reshapes and re-dimensionalizes `sol["y"]` into
  `y_2d`/`y_1d`/`y_scalars`, converts time back to seconds, and saves everything (with
  geometry and pressure-drop metadata) to `<step>/sol.pkl`.
- `require_final_solver_success()` stops the run if the final CVODE attempt failed. With full
  persistence the failed result is saved first, as evidence; a composition-domain failure is
  rejected before anything is saved, and compact persistence keeps only the failure summary.
  A failed result never reaches the step metrics or the next step.
- `context_store.save(...)` merges into `<run_base_folder>/simulation_context.pkl`, **one
  file per results folder, shared by every step**; `runtime_store.save(...)` merges the same
  way into `<run_base_folder>/resolved_runtime.json`.

NSGA-II candidates run with a compact persistence policy: a successful step saves only its
terminal state (`<step>/terminal.pkl`), and a failed one only its solver and termination
summary.

With `solve=false` this block is skipped: `_run_step()` loads the saved result through
`solver.load_results()` while it builds the metrics provenance, then prints the saved CVODE
outcome (success, status, message, `nfev`/`njev`) when it is available.

**g) Post-process the step, whether or not it was solved now:**

This describes ordinary step execution and `solve=false` replay, which persist the step
outputs. The snapshot replay used by `report` sets `solve=false` and
`persist_step_outputs=false`: it loads and describes the saved step and draws its plots, but
skips the step log, the typed metrics and their provenance. `persist_step_outputs=false` is
refused with `solve=true`.

```python
step_postprocessor = StepPostProcessor(adsorber, solver)
step_postprocessor.describe()
step_metrics = step_postprocessor.compute_step_metrics(step_name)
metrics_store = StepMetricsStore(step_postprocessor.base_folder)
metrics_store.show(step_metrics)
metrics_store.save(step_metrics, step_folder=step_name, source_provenance=...)
```

`StepPostProcessor`
([`post_processing/profiles/step_post_processing.py`](../../src/pydac/post_processing/profiles/step_post_processing.py))
uses the prepared results in memory when `y_2d`, `y_1d` and `y_scalars` are there, and
otherwise loads the saved result through `solver.load_results()`; it extracts the dimensional
fields and derives the reporting quantities from them. `.describe()` prints the console
summary (bed length and diameter, loading and temperature ranges, termination reason, net mass
change). `.compute_step_metrics(step_name)` computes the typed `StepMetrics` of the step; the
formulas are in [06 — Post-Processing](06_post_processing.md). `StepMetricsStore` prints a
sorted table and writes `<step>/kpis.pkl`, `<step>/kpis.csv` and `metrics_provenance.json`.

**h) Plot, if this step should be plotted:**

```python
if self._should_plot_profiles(plot_profiles):
    step_postprocessor.plotter.plot_all_profiles(save=True, show=False, fmt=self.plot_format)
```

A step run plots every step when `plot_profiles` is true. In a CSS run, `plot_profiles` true
(for the run, or for one cycle in `execution.cycle_options`) draws the step profiles of every
cycle where it applies: the CSS workflow then makes the current cycle the plotted one. The
alternative, `plot_final_cycle_profiles`, draws the step profiles of the reported cycle once,
after the loop, from the saved solutions. `plot_all_profiles()`
([`post_processing/profiles/step_plotting.py`](../../src/pydac/post_processing/profiles/step_plotting.py))
runs four suites in order (internal 2D fields, temperature with the wall, internal 1D
profiles, outlet profiles), writing figures in the case's `execution.plot_format` under
`<step>/profiles/{internal_2D_profiles,internal_1D_profiles,outlet_profiles}/`; for other
declared mixtures it uses generic 1D and outlet routines.

**i) Hand off to the next step:**

```python
final_fields = solver.get_final_fields()
step_time = solver.sol["t"][-1]
```

`get_final_fields()` returns the last time slice of `ci`, `qi`, `c_tot`, `T`, `T_wall`. Back in
`run_cycle()`, it becomes `prev_fields`, the initial state of the next step's
`ColumnModelInput`: this is the physical thread connecting
`startup → blow → heat → s_des → cool → press → ads` into one continuous cycle rather than
seven independent solves.

**End of the sequence.** After the loop, `run_cycle()` prints the completion banner and a
per-step timing table (`_print_cycle_completion`, `_print_cycle_timing_summary`) and returns
`(final_fields, step_records, cycle_total_time_s)`. The workflow returns them as a dictionary;
`api._solve_steps()` prints the run folder again and returns a `RunResult` with the run folder
and the results folder, and the command exits with code `0`.

## 4. Solve vs. Reuse, Per Step

`solve`, `plot_profiles`, `use_sparsity`, `compute_numerical_sparsity` and `termination_mode`
have a value for the whole sequence (`execution.*`) that `execution.step_options` can change
for single steps:

```json
"execution": {
  "solve": true,
  "step_options": {"startup": {"solve": false}}
}
```

`normalize_step_options()` (`utils/tvsa/steps.py`) accepts the short names (`ads`) and the aliases accepted in
`cycle_steps` (`ADSORPTION`). A step with `solve=false` needs its saved `<step>/sol.pkl` in
the same results folder; since a fresh run refuses an occupied folder, reusing saved steps goes
through `--resume` ([user guide](../guide/04_running.md#replay-and-partial-re-solve)).
Post-processing (§3.g) runs for every step either way.

In a CSS run there is no top-level `step_options`: per-step control lives in
`execution.cycle_options[n]["step_options"]`.

## 5. Log Files vs. Terminal

Every step writes its own `log.txt` (§3.b). With `print_to_terminal=true` (the default) you see
everything live and it is also written to disk; the log is a byproduct of a normal run, useful
for comparing what two runs printed without running them again. A CSS run also keeps one
overall `log.txt` in the results folder, covering the whole cycle loop.

## 6. Column Model Initialization

`ColumnModelMoL` accepts exactly one `ColumnModelInput`. Public inputs become model attributes
in `ColumnModelMoL._bind_inputs()` through explicit named assignments. Changes there are under
maintainer review: a new physical consumer comes with a proposal that states the default path,
the binding, the formula or closure using it, and the expected numerical effect
([CONTRIBUTING](../../CONTRIBUTING.md#physics-core-maintainer-review)).

The constructor stores `column_input.config` as `self.column_config` and then runs, in order:

```text
__init__(column_input)
  ├── _bind_inputs()
  ├── _set_options(column_input)
  ├── set_simulation_parameters()
  ├── store_prev_fields(initial_state)        (every step except startup)
  ├── validate_architecture_support(...)
  ├── _resolve_bed_structure()
  ├── init_radial_structure_factors()
  ├── set_reference_conditions()
  ├── set_ic()
  ├── set_feed()
  ├── init_gas_mix()
  ├── init_solid_thermo()
  ├── calculate_system_parameters()
  ├── set_disp()
  ├── set_cond()
  ├── set_bc()
  ├── init_Jac()
  ├── init_inlet_fluxes()
  ├── _build_runtime_properties()
  └── _print_runtime_property_summary()
```

## 7. Where One Public Input Reaches The Physics

The full narrowing chain for one value, `steps.s_des.flow.steam_mass_flow_kg_s`:

```text
inputs/defaults/steps.py            (the default, or --set / a preset / a frozen case)
  → ExperimentConfig.simulation.cycle.steps: the CycleStep named "s_des",
    .config.flow.steam_mass_flow_kg_s
  → CycleConfig.from_experiment(...)   (config/cycle.py)    same value, runtime view
  → CycleConfig.step_config("s_des")                       ColumnModelConfig for this step
  → ColumnModelInput.config                                (core/column_model_input.py)
  → ColumnModelMoL._bind_inputs()                          ⚠ core/column_model.py
  → self.m_dot_steam
  → _resolve_inlet_velocity(): self.vol_flowrate = self.m_dot_steam / self.rho_steam_inlet,
    then the inlet velocity, at model setup
```

This is the chain of [00 — Framework Map §5](00_framework_map.md#5-runtime-configuration-boundary)
with the file and function at each arrow.

## 8. State Vector Layout

`StateLayout` ([`core/state_layout.py`](../../src/pydac/core/state_layout.py)) generates
the complete order from the declared mixture and grid:

1. one `(Nz, Nr)` concentration for every non-closure gas species;
2. one `(Nz, Nr)` loading for every adsorbate;
3. `(Nz, Nr)` total concentration and bed temperature;
4. one `(Nz,)` wall-temperature field;
5. named scalar mass and energy states.

Each field owns its generated slice, physical name, unit, storage group, scaling source,
constraint policy and tolerance group. Solver packing, dimensional export, post-processing,
events and sparsity all use that same layout and its grid-sensitive signature; none of them
computes offsets on its own.

## 9. RHS Evaluation

At each RHS call, `pde_system_kpis()`:

1. unpacks the current fields;
2. reconstructs pressure from total concentration and temperature;
3. recovers the axial face velocities by scalar Ergun inversion in 1D (radial velocity zero),
   and both axial and radial face velocities by isotropic vector inversion in 2D;
4. updates pressure-driven inlet fluxes when required;
5. evaluates adsorption kinetics;
6. assembles gas, loading, total-concentration, bed-energy, and wall-energy balances;
7. assembles integrated KPI derivatives;
8. fills and returns the flat derivative vector in `StateLayout` order (a reusable output
   buffer, or the buffer the caller passes).

The model uses `core/bed_architectures/` for packed-bed structure, momentum, dispersion, and
thermal closures in both 1D and 2D.

The 2D momentum closure uses isotropic vector drag. Axial and radial convection use
conservative face fluxes; radial convective and dispersive mass fluxes vanish at the axis
and wall, while bed-wall heat transfer follows the radial thermal Robin condition. Channeling resolves
separate radial storage and drag fields; neither field replaces the other. The drag
averaging width is independent of the finite-volume ring width. Cooldown inherits the
preceding state without flattening or hydraulic relaxation.

## 10. Solver

[`core/solver.py`](../../src/pydac/core/solver.py) wraps `sksundae.cvode.CVODE` and owns:

- time-grid generation in natural nondimensional time;
- event callback assembly;
- positivity constraints for physically non-negative variables;
- sparse Jacobian-mask selection;
- the direct sparse linear solver (dense only with `use_sparsity=false`, for diagnostics);
- `sol.pkl` persistence.

### 10.1 The linear solver

By default every step is solved with CVODE's direct sparse solver on the analytical mask
(`configure_cvode_linear_solver_options` in
[`utils/solver/sparsity.py`](../../src/pydac/utils/solver/sparsity.py));
`execution.use_sparsity=false` selects the dense diagnostic solver. The solver
that ran is recorded as `sol["cvode"]["linsolver"]` (`sparse` or `dense`). The iterative GMRES
path was removed before the public release; the schema migration refuses a saved case that
asks for it.

The default mask is the analytical `pde_only` mask; `execution.cvode_sparsity_mode=full` keeps
all rows of the analytical mask, for diagnostics.

The mask is generated in memory from the state layout and cached by layout (an LRU cache of 64
entries; every request returns a copy), so the steps of one grid reuse it. No mask is shipped
with the package, read from disk or written. To look at it, draw it with
`python -m pydac.core.sparsity_patterns.analytical_sparsity --nz <nz> --nr <nr> --out <folder>`.

The finite-difference diagnostic `execution.compute_numerical_sparsity=true` (default `false`;
slow, dense Jacobians) saves snapshots into the step folder,
`<step>/numerical_jacobian/jacobian_sparsity_<tick>.npy`.
`python -m pydac.core.sparsity_patterns.numerical_sparsity <step folder> --nz <nz> --nr <nr>`
unions them, draws the union and fails if a numerically found coupling is missing from the
analytical mask.

## 11. Physics core: maintainer review

[`core/column_model.py`](../../src/pydac/core/column_model.py) and everything under
[`post_processing/`](../../src/pydac/post_processing/) define what pyDAC computes and
reports. Changes to them need the maintainer's review before they are written: state the
functions, the physical or numerical reasoning and the expected effect
([CONTRIBUTING](../../CONTRIBUTING.md#physics-core-maintainer-review)). Configuration and
orchestration work normally belongs in `inputs/`, `config/`, `workflows/`, the launchers in
`runners/`, the tests or the documentation.
