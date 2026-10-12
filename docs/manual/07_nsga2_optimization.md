---
title: "07 — NSGA-II Optimization"
toc: true
toc-depth: 3
embed-resources: true
---

> **This manual is the detailed technical reference: how pyDAC works inside, for contributors
> and for checking the model.** To learn to use pyDAC, do the [tutorials](../../examples/README.md)
> first, then read the [user guide](../guide/01_installation.md).

This chapter describes how an NSGA-II campaign works inside pyDAC: which modules take part,
how a decision vector becomes a case, how candidates are evaluated and recorded, and how a
campaign is checkpointed and continued. To run a campaign, read the user guide page
[Optimization campaigns](../guide/05_optimization.md); this chapter does not repeat its
commands and options.

A campaign uses the same case vocabulary and the same CSS workflow as an ordinary run. It adds
bounds, population settings, constraints and parallelism; it does not add a second set of
physical inputs.

## 1. Architecture

| Layer | Module | Responsibility |
|---|---|---|
| Command and API | [`cli.py`](../../src/pydac/cli.py), [`api.py`](../../src/pydac/api.py) | `pydac nsga2` / `pydac.run_nsga2`: validate the whole campaign, name and guard its folder, write the base case; `pydac.resume`: check and continue a saved campaign; `pydac.report`: regenerate its figures and tables from the saved results |
| Orchestrator | [`workflows/orchestrator_NSGA2.py`](../../src/pydac/workflows/orchestrator_NSGA2.py) | `NSGA2RunConfig`, `NSGA2Constraints`, `NSGA2TerminationConfig`, candidate evaluation, the pymoo problem, the optimizer loop, resume |
| Campaign utilities | [`utils/nsga2/`](../../src/pydac/utils/nsga2/) | decision vector (`decisions.py`), candidate records and pruning (`evaluation.py`), checkpoints (`checkpoint.py`), warm starts (`warm_start.py`), logs (`logging.py`), campaign metadata files (`artifacts.py`) |
| CSS workflow | [`workflows/orchestrator_TVSA_cycle_css.py`](../../src/pydac/workflows/orchestrator_TVSA_cycle_css.py) | runs each candidate until CSS, `execution.max_cycles` or failure; reaching the cap without CSS rejects the candidate only with `execution.require_css=true` |
| Case engine | [`config/`](../../src/pydac/config/) | resolves each candidate's case; `config/operating.py` owns the operating knobs (§2) |
| KPI layer | [`post_processing/metrics/`](../../src/pydac/post_processing/metrics/), [`post_processing/kpi/`](../../src/pydac/post_processing/kpi/) | loads the CSS cycle (or, with `require_css=false` and no CSS, the highest-numbered saved cycle) and computes objectives and constraint values |
| Campaign figures | [`post_processing/optimization/`](../../src/pydac/post_processing/optimization/) | Pareto and distribution figures, `pareto_front.csv` |

`pydac.run_nsga2` checks the campaign before it writes anything: the base case passes the same
preflight as a CSS run, the bounds name exactly the decision variables of the cycle, the
constraints fit the mixture and the cycle (an O2 limit needs O2; `steam_condensation` needs H2O
and a cycle whose desorption step is `s_des`, otherwise the campaign is refused), and the candidate cases at the lower and the upper corner of the bounds are resolved
without solving, so a base case that disagrees with the bounds is refused before the first
generation (`preflight_search_space`). A coupling that fails only inside the box is not
caught there: that candidate fails when it runs. The campaign folder is then named by
`execution.extra_tag` (when set), the grid and a hash of the whole campaign definition
(`nsga2_campaign_identity`), and refused if it already
holds a different or an unidentified campaign, or committed generations of this same
campaign, which `pydac nsga2 --resume CAMPAIGN` continues (`guard_campaign_folder`). Only then is the base
case written as `resolved_config.json` and `run_nsga2_optimization` started.

## 2. Decision variables and candidate settings

`NSGA2RunConfig.experiment` is the base case. It supplies the grid, geometry, bed, cycle steps,
CSS criteria and outputs; `tau_ar`, `bed_volume`, `cycle_steps` and `require_css` are read
from it, never stated twice. The decision vector is set by the desorption step of the cycle
([`utils/nsga2/decisions.py`](../../src/pydac/utils/nsga2/decisions.py)):

| Variable | Operating knob it sets |
|---|---|
| `t_ads` | `adsorption_time_s` |
| `t_heat` | `heating_time_s` |
| `t_des` | `desorption_time_s` |
| `P_vac` | `low_pressure_Pa` |
| `v_feed` | `volumetric_flow_m3_s`, from the superficial velocity and the bed cross-section |
| `v_steam` | `steam_mass_flow_kg_s`, from the superficial velocity, the cross-section and the steam density at 373.15 K and `P_vac`; only for steam-assisted desorption (`s_des`) |
| `T_cool_target` | `cooling_target_temperature_K`; optional, internal (§6) |

The order is fixed: the five base variables, then `v_steam` for a steam-assisted cycle, then
the optional `T_cool_target`. Optional variables are appended, never interleaved, so the saved
decision columns of a campaign keep their meaning. `validate_var_names` refuses any other
vector.

Each knob stands for the step fields it drives; the table is written once, in
[`config/operating.py`](../../src/pydac/config/operating.py), and listed in
[01 — Inputs §8.1](01_inputs.md#81-operating-knobs-and-their-fan-out). A candidate is the base case plus `candidate_settings(...)`: the settings
`operating_step_settings(...)` returns for its knobs, together with `execution.solve=true`,
`execution.print_to_terminal=false`, `execution.print_cycle_build=false` and
`execution.plot_profiles=false`. The ordinary case loader resolves them, so every candidate is
described in the ordinary case vocabulary.

## 3. Objectives and constraints

The pymoo problem (`TVSAOptimizationProblem`) minimizes two objectives:

| Objective | Meaning |
|---|---|
| `F[0]` | equivalent work, MJ per kg of recovered CO2 |
| `F[1]` | minus the productivity, kg of recovered CO2 per m³ of bed per day (negative, so that minimizing it maximizes productivity) |

The constraints follow the pymoo convention `g <= 0` is feasible. The first is always active:
the evaluation must have finished without failure and, when `execution.require_css` is true,
reached CSS. The others come from `NSGA2Constraints` and are active only when set:

| Constraint | `g` |
|---|---|
| `purity_min` | `purity_min - purity` |
| `oxygen_ppm_max` | `oxygen_ppm - oxygen_ppm_max` |
| `steam_condensation` | `T_steam_condensation - T_bed_min_preheat`; enabling it without H2O or without `s_des` is refused before the campaign starts |
| `eq_work_max` | `eq_work - eq_work_max` |

A failed candidate gets large objective and constraint values, so it is infeasible and never
reaches the front. How equivalent work, productivity and purity are computed is in
[06 — Post-Processing](06_post_processing.md).

## 4. Candidate evaluation and the campaign folder

`evaluate_pydac_candidate` evaluates one decision vector in its own folder,
`gen_<NN>/pid_<PID>_eval_<ID>/`: it builds the candidate case, writes it as
`resolved_config.json`, runs the CSS workflow, and computes the objectives and constraint
values from the CSS cycle or, when CSS is optional and not reached, from the highest-numbered
saved cycle. It then writes two compact files, `candidate_record.json` and
`evaluation_summary.json`, and deletes everything else in the folder, including the
trajectories and the `resolved_config.json` (`prune_candidate_artifacts`). An ordinary
candidate therefore keeps no simulation results on disk.

At the end of each generation its candidate records are staged into
`results/candidate_records/generation_<NNNN>.jsonl`, one JSON record per line
(`commit_generation_candidate_records`); they count as committed once the checkpoint state
below is written. With `delete_generation_folders=True`, the setting `pydac.run_nsga2` uses,
the callback removes the previous generation folder before writing the new checkpoint state,
so after a successful callback only the newest `gen_<NN>/` is on disk. The consolidated records are the durable result of every
candidate.

A record holds the decision vector, objectives, constraint values, KPIs, CSS history, solver
outcomes and a provenance record. The candidate's complete case is in the provenance, at
`provenance.input.resolved_config`. Saved as a JSON file, it is a frozen case and runs again
with `pydac css --case`.

The campaign folder holds:

| Path | Content |
|---|---|
| `resolved_config.json`, `case_metadata.json` | the base case and its fingerprint |
| `pareto_bounds.csv`, `pareto_flow_transform.json` | the bounds, and how the velocity variables became flows |
| `generation_log.txt`, `evaluations_log.txt`, `nsga2_log.txt`, `optimization_log.csv` | progress: a table with one row per generation, one line per candidate, the full campaign output, the per-generation numbers as CSV |
| `gen_<NN>/` | the candidates of the newest generation (compact files only) |
| `results/campaign_identity.json` | the campaign definition the folder belongs to |
| `results/checkpoint_state.json`, `results/checkpoints/` | the checkpoint (§5) |
| `results/candidate_records/` | the consolidated candidate records |
| `results/pareto_front.csv`, `pareto_front.npz`, `last_population.npz`, `last_pareto_front.npz`, `population_and_extras.csv`, `decision_variable_modes.csv` | the final front and population |
| `results/optimization_summary.json`, `results/provenance/` | the run summary and provenance records |
| `results/figures/` | Pareto and distribution figures (PDF) |

The final and report post-processing write `pareto_front.csv` before drawing any figure, so a
figure that fails does not lose the front table. `decision_variable_modes.csv` uses a KDE on the
scaled values when the front has spread; for a single front point, or a value identical at every
front point, that value is the mode and no KDE is fitted.

With `n_procs > 1` the pool initializer `utils.nsga2.logging.share_worker_state` installs the
parent's shared generation counter and stdout lock in each worker. Keep it when changing how the
pool is created: under `spawn` or `forkserver` the workers re-import the module globals and would
otherwise count generations on their own, filing evaluations under `gen_01`.

## 5. Checkpoint and continuation

After each generation the optimizer callback writes the population and the front of that
generation (`results/checkpoints/generation_<NNNN>_population.npz` and `_pareto.npz`), then
`results/checkpoint_state.json`
([`utils/nsga2/checkpoint.py`](../../src/pydac/utils/nsga2/checkpoint.py)). The state file
holds the campaign identity, the completed generations and evaluations, the population size,
the pymoo version, the random-generator state, the state of the stop rule, and the SHA-256
checksums of the checkpoint's population, Pareto front and (when present) consolidated
candidate records. A generation counts as committed only when its state file
is written; writes are atomic.

`pydac.resume` reads the campaign definition back from the folder, so a continuation cannot
differ from the original campaign. Before it touches anything it checks the campaign identity,
the pyDAC and pymoo versions, and the checksums; a missing or corrupt committed file is
refused. It then discards uncommitted generation folders and records, restores the population,
the front, the counters and the random-generator state, and continues without evaluating the
saved parents again (`WarmStartSpec(mode="continuation")`). The generation target stays the
original total. Because the state of the `robust_pareto` stop rule is saved too, a continued
campaign stops at the same generation as an uninterrupted one. A campaign that already met its
stop rule is refused before anything is written (`RunRefused`, exit code 2), with a message
that it is complete.

## 6. Internal options

`run_nsga2_optimization` and `NSGA2RunConfig` have options that `pydac nsga2` and
`pydac.run_nsga2` do not expose. They are internal: shipped, tested, and subject to change
without notice.

| Option | Default | Effect |
|---|---|---|
| `NSGA2RunConfig.decision_variables` | `None` (the vector of §2) | declares the vector explicitly, the only way to add `T_cool_target` |
| `NSGA2RunConfig.sbx_prob`, `sbx_eta`, `pm_eta` | `0.9`, `15`, `20` | simulated binary crossover and polynomial mutation parameters |
| `NSGA2RunConfig.composition_domain_guard` | `CompositionDomainGuard()` | stops a candidate's run as failed when the closure-species concentration falls below zero by more than an allowance; it never changes the state |
| `warm_start` | `None` | `WarmStartSpec(mode="continuation")` resumes (§5); `mode="refining"` seeds a new campaign from a saved front or population |
| `delete_generation_folders` | `True` | `False` keeps every generation folder (compact files only) |
| `run_postprocessing` | `True` | writes the figures and `pareto_front.csv` at the end |

## 7. Where to change what

| Goal | Module |
|---|---|
| Change how a decision variable maps to case settings | `utils/nsga2/decisions.py`, `config/operating.py` |
| Change which step fields an operating knob writes | `config/operating.py` |
| Change objectives or constraints | `workflows/orchestrator_NSGA2.py` (`TVSAOptimizationProblem`, `NSGA2Constraints`); a change to the KPI layer needs maintainer review |
| Change what a candidate record holds | `workflows/orchestrator_NSGA2.py` (`evaluate_pydac_candidate` builds it); `utils/nsga2/evaluation.py` writes and consolidates it |
| Change the checkpoint content | `utils/nsga2/checkpoint.py`; a changed shape bumps `CHECKPOINT_SCHEMA_VERSION`; the supported older schemas must still load, although a continuation can still be refused (pyDAC or pymoo version, missing saved state) |
| Change the campaign figures | `post_processing/optimization/`, maintainer review |
| Change the physical model | `core/column_model.py`, maintainer review |

Maintainer review is described in [`CONTRIBUTING.md`](../../CONTRIBUTING.md#physics-core-maintainer-review):
`src/pydac/core/column_model.py` and everything under `src/pydac/post_processing/` are
changed only after an issue that states the change and its reasoning.
