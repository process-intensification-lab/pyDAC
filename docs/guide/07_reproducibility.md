---
title: "Reproducibility"
toc: true
toc-depth: 3
embed-resources: true
---

> **New to pyDAC? Do the [tutorials](../../examples/README.md) first, starting with
> [01 First steps](../../examples/notebooks/01_first_steps.ipynb).** They teach pyDAC step by
> step on real runs; this guide is the reference behind them and goes into more technical detail.

A result is reproducible when someone else can run the same case with the same software and
get the same numbers. pyDAC helps in four ways: the frozen case fixes every input, the
fingerprint names it, every run records the software that produced it, and the
qualification lock fixes the whole environment. This page shows how to use each, and what
pyDAC 3.0.0 promises across versions.

The examples use the small 1D case of the [quick start](02_quickstart.md), in a new, empty
folder:

```bash
pydac validate --set grid.nz=25 --set grid.nr=1 --set execution.max_cycles=3 --set execution.require_css=false --export-case my_case.json
pydac css --case my_case.json
```

## Share a case

Share the frozen case, not the preset and settings you built it from. A frozen case holds
every value of the case, so it resolves to the same case, and the same fingerprint, on any
machine. A case built from a preset recomputes derived values (for example the steam
volumetric flow from the steam mass flow) with the local numerical libraries, and can get a
slightly different fingerprint elsewhere ([Cases and settings](03_cases.md#fingerprint-and-run-folder)).

Every run folder holds its frozen case as `resolved_config.json`. Running it again gives the
same fingerprint and the same run folder, which the folder refusal shows:

```bash
pydac css --case simulation_results/cyclic_mode/packed_bed_7a4930a5/1D_mode/resolved_config.json --dry-run || echo "exit code $?"
```

```text
Mode: frozen case simulation_results/cyclic_mode/packed_bed_7a4930a5/1D_mode/resolved_config.json
Refused: `simulation_results/cyclic_mode/packed_bed_7a4930a5` already holds a run. Continue it with --resume simulation_results/cyclic_mode/packed_bed_7a4930a5 (pydac.resume), replace it with --overwrite (overwrite=True), or set another execution.extra_tag.
exit code 2
```

On another machine, or in another folder, the same command runs the case into
`simulation_results/cyclic_mode/packed_bed_7a4930a5`. An NSGA-II campaign is shared the
same way: its base case is `CAMPAIGN/resolved_config.json`, and the bounds, constraints and
optimizer options are the command line (`results/campaign_identity.json` records them all).

## What every run records

Every run, report, continuation and campaign writes a provenance record, a JSON file in a
`provenance/` folder of the run: in the results folder, in each step folder, in `kpis/`,
and in `CAMPAIGN/results/`. A record holds:

- the pyDAC version (`software.version`);
- the resolved case and its fingerprint (`input`); the stored copy has
  `execution.output_root` set to `"."` so it can be moved, and the record keeps both
  fingerprints, the original one under `source_case_metadata`;
- the Python version, the operating system and the machine;
- the versions of numpy, scipy, pandas, numba, scikit-sundae and pymoo, the BLAS and LAPACK
  libraries and SUNDIALS, the thread settings (`OMP_NUM_THREADS` and the like), and the
  two solver switches `PYDAC_RHS_CSR_OPERATORS` and `PYDAC_CACHE_SLICE_WRITE`
  (`solver_environment`; both on unless set to `0`, `off`, `false` or `no`);
- the source: run from a Git checkout of pyDAC, whether it had uncommitted changes (the
  commit itself is not recorded: note it yourself for an unreleased checkout); installed
  from a package, `"no_checkout"`. The version names the source then: every release has a
  Git tag that is never moved.

Abridged, from the run above (installed from a package):

```json
{
  "activity": "run",
  "execution": {"workflow": "css", "solve": true, "persistence_policy": "run_full"},
  "input": {
    "source_case_metadata": {"case_fingerprint": "7a4930a5459081a9bb61e998f196b3064a0483b3a467c2ad540b93ff9a91f585"},
    "relocated_fields": ["execution.output_root"]
  },
  "software": {
    "version": "3.0.1",
    "python": {"implementation": "CPython", "version": "3.12.2"},
    "dependencies": {"numpy": {"version": "2.5.3"}, "scipy": {"version": "1.18.1"}, "pymoo": {"version": "0.6.2"}},
    "native_libraries": {"sundials_version": "7.5.0"},
    "git": {"status": "no_checkout"}
  }
}
```

Recording never stops a run: if Git is missing or slow, the record says so and the run goes
on.

## The exact environment

The package ranges pyDAC declares (`numpy`, `scipy`, `pandas`, `pymoo` and the others in
`pyproject.toml`) are the versions it accepts. Different versions in that range can change
results in the last digits. The qualification lock `ci/qualification-lock-py3.12.txt` is the
environment that produced the qualified results: CPython 3.12, every package at one exact
version with its hash. It changes only by a declared decision of the maintainers. To
reproduce a published number exactly, install from the lock
([Installation](01_installation.md#exact-reproduction)) and compare the provenance record of
your run with the published one.

## Across pyDAC versions

- **Reading and reporting** a run (`pydac css --report`, `CycleDataLoader`, a step replay
  with `execution.solve` false) works with any later release of the same major version:
  pyDAC 3.x reads runs written by pyDAC 3.0.0.
- **Continuing** a run needs the version that wrote it: a CSS continuation, a step resume
  that solves steps again, and an NSGA-II continuation are refused by another version, and
  the refusal names both versions. Install that version to continue, or start a new run.

## NSGA-II campaigns and pymoo

A campaign depends on pymoo as well as on pyDAC:

- The same case, bounds, options and seed give the same campaign only with the same pymoo
  version: the seed drives pymoo's random generator, whose use can change between pymoo
  versions.
- A campaign continues only with the pymoo version that wrote its checkpoint. Upgrading pymoo
  in the middle of a campaign stops its continuation (the refusal names both versions);
  finish the campaign first.

The number of workers does not change a campaign's result.

## Related pages

- [Installation](01_installation.md#exact-reproduction): the exact environment from the lock
- [Cases and settings](03_cases.md): frozen cases and fingerprints
- [Running steps and cycles](04_running.md): resume, replay and report
- [Optimization campaigns](05_optimization.md)
