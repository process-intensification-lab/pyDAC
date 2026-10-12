# Documentation Index

The repository landing page lives at the repo root in [`../README.md`](../README.md).

**New to pyDAC? Start with the tutorials** in [`../examples/`](../examples/README.md), beginning
with [01 First steps](../examples/notebooks/01_first_steps.ipynb). Read in this order: the
tutorials teach pyDAC on real runs; the guide below is the reference for using it, in more
technical detail; the manual describes how pyDAC works inside, in full technical detail, for
contributors and for checking the model.

## Guide: how to use pyDAC

- [01 Installation](guide/01_installation.md): environments, Jupyter, shell completion, exact
  reproduction, troubleshooting.
- [02 Quick start](guide/02_quickstart.md): one case from the terminal, a script and a notebook;
  the Python API and its errors.
- [03 Cases and settings](guide/03_cases.md): presets, settings, frozen cases, model choices,
  `null`, fingerprints.
- [04 Running steps and cycles](guide/04_running.md): step sequence, stop rules, resume, replay,
  report, run folders, exit codes.
- [05 Optimization campaigns](guide/05_optimization.md): NSGA-II campaigns, bounds, constraints,
  stop rules, continuing and reporting a campaign.
- [06 Results](guide/06_results.md): run folders, KPI files, reading results from Python, which
  folders to trust.
- [07 Reproducibility](guide/07_reproducibility.md): sharing a case, provenance, the exact
  environment, compatibility across versions.
- [08 Extending pyDAC](guide/08_extending.md): your own isotherms and kinetic laws, registered
  from Python; what needs a change to the package.

## Manual: how pyDAC works

| Document | Contents |
|----------|----------|
| [00 — Framework Map](manual/00_framework_map.md) | **Start here for the code.** Layer map, dependency rules, entry points, the three pipelines, typed configuration boundary, artifact flow, ownership contract, and a "where do I change X" cookbook |
| [01 — Inputs](manual/01_inputs.md) | **Start here for anything about inputs.** The two input mechanisms, resolution order, the document contract, frozen cases, ownership map, and a "where do I change X" cookbook |
| [02 — pyDAC Core](manual/02_pydac_core.md) | What happens when you press Run, stage by stage; factory, PDE model, solver internals, cycle orchestration |
| [03 — pymrm Discretization](manual/03_pymrm_discretization.md) | Sparse FVM operators, boundary conditions, caching |
| [04 — Governing Equations](manual/04_governing_equations.md) | The equation→code map: where balances and closures are assembled (reference state, bed-wall coupling, architectures, isotherms and kinetics, positivity). **Not the derivation** — that is the manual |
| [05 — Cycle Steps](manual/05_cycle_steps.md) | Per-step physics, BCs, event-based and time-based stopping rules |
| [06 — Post-Processing](manual/06_post_processing.md) | KPIs, plotting, CSS diagnostics, result-file structure |
| [07 — NSGA-II Optimization](manual/07_nsga2_optimization.md) | Campaign setup, decision vector, objectives and constraints, sweeps, output layout |

Every subject has exactly one owner chapter above; the others link to it rather
than restating it.

## Notes

- New to the codebase? Read [00 — Framework Map](manual/00_framework_map.md) first;
  it explains the layers, the entry points, and which document to open next.
- The markdown files under `guide/` and `manual/` are the maintained sources.
- Operational case scripts live in [`../runners`](../runners).
- Reusable high-level orchestration functions live in [`src/pydac/workflows/`](../src/pydac/workflows/).
- User-editable process defaults live in [`src/pydac/inputs/defaults/`](../src/pydac/inputs/defaults/); complete replayable cases live in [`../inputs/cases/`](../inputs/cases/).
- [`src/pydac/config/`](../src/pydac/config/) is the validation/resolution engine and typed runtime contract, not the routine value-editing surface.
- Step execution flags, `termination_mode`, operating conditions, event metrics, and event thresholds are documented in [01 — Inputs](manual/01_inputs.md).
- Every step uses the direct sparse linear solver; how schema-v2 cases with a per-step `linear_solver` are upgraded is [01 — Inputs §4.5](manual/01_inputs.md#45-the-linear-solver), the internals are [02 — pyDAC Core §10](manual/02_pydac_core.md#10-solver).
- An event threshold is also a cost decision for the *next* step: see [05 — Cycle Steps §3.3](manual/05_cycle_steps.md#33-event-stop-rules) for the state passed from one step to the next.
