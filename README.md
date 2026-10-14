<p align="center">
  <img alt="pyDAC" src="https://raw.githubusercontent.com/process-intensification-lab/pyDAC/main/docs/assets/pydac_header.svg" width="420">
</p>

<p align="center">
  <a href="https://pypi.org/project/pydac3/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pydac3"></a>
  <a href="https://pypi.org/project/pydac3/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/pydac3"></a>
  <a href="https://github.com/process-intensification-lab/pyDAC/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/process-intensification-lab/pyDAC/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://pepy.tech/projects/pydac3"><img alt="Downloads" src="https://img.shields.io/pepy/dt/pydac3"></a>
  <a href="https://doi.org/10.5281/zenodo.23319859"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.23319859.svg"></a>
</p>

pyDAC simulates adsorption-based direct air capture (DAC) processes. It solves the
coupled mass, energy and momentum balances of a sorbent column in 1D or 2D
(axisymmetric) and runs adsorption cycles step by step or to cyclic steady state (CSS).
A cycle is composed from configurable steps (adsorption, blowdown, heating, purge, steam-
or wall-heated desorption, cooling, pressurization); the default is a steam-assisted
temperature-vacuum swing adsorption (TVSA) cycle. pyDAC reports process key performance
indicators (KPIs) and optimizes operating conditions with the NSGA-II genetic algorithm
(non-dominated sorting genetic algorithm II).

pyDAC 3.0.1, development status Beta. MIT license.

https://github.com/user-attachments/assets/b471d7f8-2b83-4b18-a132-149970f4f62b

## Installation

### 1. Create a dedicated environment

Install pyDAC in an environment of its own: never in the system Python, a conda `base`
environment, or an environment shared with other projects. pyDAC fixes the versions of
some dependencies (`pymrm==2.1.10`, `pandas<3`, `scikit-sundae` 1.1.x), and pip silently
replaces versions that are already installed, so a shared environment can break pyDAC
or the other project without any message.

Linux, with `venv`:

```bash
python3 -m venv pydac-env
source pydac-env/bin/activate
```

Windows (PowerShell), with `venv`:

```powershell
py -m venv pydac-env
pydac-env\Scripts\Activate.ps1
```

With conda instead:

```bash
conda create -n pydac python=3.12
conda activate pydac
```

With uv instead (`uv` environments have no `pip`; write `uv pip` where this page writes
`python -m pip`):

```bash
uv venv pydac-env
source pydac-env/bin/activate
```

### 2. Install pyDAC

```bash
python -m pip install pydac3
```

The distribution is called `pydac3`; the package you import and the command you run are
both `pydac`. `pip install pydac` installs an unrelated project.

One install contains everything: step runs, CSS runs, case validation and NSGA-II
optimization (pymoo is a regular dependency). `pydac get-started my_pydac` copies the tutorials
and the user guide into a new folder
([Get the tutorials](docs/guide/01_installation.md#get-the-tutorials-and-this-guide)). The optional extra `completion` adds shell
completion for the `pydac` command (`python -m pip install "pydac3[completion]"`).

Python 3.11 to 3.14 on Linux (x86_64) is supported. Windows is being tested; macOS is
neither tested nor supported.

**Want to adapt pyDAC to your own needs, or help improve it?** Install it from a clone in
editable mode instead; then an edit in `src/pydac/` is used at the next import:

```bash
git clone https://github.com/process-intensification-lab/pyDAC.git
cd pyDAC
python -m pip install -e .
```

[`CONTRIBUTING.md`](CONTRIBUTING.md) gives the full procedure (dedicated environment, light or
exact dependencies, tests) and how to propose a change.

### 3. Check the environment

```bash
python -m pip check
pydac --version
python -c "import pydac; print(pydac.__file__)"
```

`pip check` must report no broken requirements, and the printed path must lie inside
the environment you created. A path in the user site instead (for example under
`~/.local/lib/`) means a user-site installation shadows the environment; set
`PYTHONNOUSERSITE=1` or remove that installation.

When the `pydac` command runs outside a virtual environment or a conda environment
other than `base`, it prints a one-line warning on stderr and continues. Set
`PYDAC_NO_ENV_WARNING=1` to silence it, for example in a container that uses the system
Python on purpose. `import pydac` never prints it.

### Exact reproduction

To reproduce the qualified environment exactly (every package pinned, with hashes;
Python 3.12), install from the lock file into a second fresh environment, from the
root of a clone:

```bash
python3.12 -m venv pydac-exact
source pydac-exact/bin/activate
python -m pip install --require-hashes -r ci/qualification-lock-py3.12.txt
python -m pip install --no-deps .
```

## Tutorials

**The best way to learn pyDAC is the seven tutorials. Start with
[01 First steps](examples/notebooks/01_first_steps.ipynb)**, then follow them in order.
`pydac get-started my_pydac` copies them, with the user guide, into a new folder. Each
tutorial is a notebook and states its runtime (from a few minutes to about 40). Tutorials
01-06 also run from the terminal, with the commands in [`examples/README.md`](examples/README.md);
tutorial 07 needs Python, because it registers its own isotherm.

| Tutorial | What it shows |
|---|---|
| [01 First steps](examples/notebooks/01_first_steps.ipynb) | a step run, the KPI table of each step, step profile plots, time- or event-based stop |
| [02 Build your own case](examples/notebooks/02_custom_case.ipynb) | replace the defaults block by block (column, gas, sorbent, isotherm, kinetics, steps); a 2D TSA step run |
| [03 A cycle at its periodic state](examples/notebooks/03_css_tvsa.ipynb) | a CSS run of the built-in TVSA cycle in 2D, the cycle KPIs, where the CO2 and the energy go |
| [04 1D or 2D](examples/notebooks/04_1d_vs_2d.ipynb) | the same cycle in 1D, in 2D and in 2D with wall channeling |
| [05 An optimization campaign](examples/notebooks/05_nsga2_trial.ipynb) | a trial NSGA-II campaign: variables, bounds, constraints, the Pareto front |
| [06 Reproduce the breakthrough validation](examples/notebooks/06_breakthrough_validation.ipynb) | the 2D model of the [pyDAC paper](https://doi.org/10.26434/chemrxiv.15010382/v1) against the measurements of Zanco et al. |
| [07 Your own isotherm](examples/notebooks/07_new_isotherm.ipynb) | a new sorbent from the literature, its isotherm registered from Python and run |

## Quick start from the terminal

A case is the complete set of inputs of one simulation. Write the built-in case to a
file, here with a coarse 1D grid and at most three cycles:

```bash
pydac validate --set grid.nz=25 --set grid.nr=1 --set execution.max_cycles=3 --set execution.require_css=false --export-case my_case.json
```

`my_case.json` holds every input of the simulation. To change it, prefer settings over
editing the file: they recompute the fields that depend on others, such as the bed length
from its volume ([Cases and settings](docs/guide/03_cases.md)). Then run it exactly as
saved; it repeats the cycle until cyclic steady state, here for at most three cycles;
this coarse case reaches CSS at cycle 3:

```bash
pydac css --case my_case.json
```

The run takes one to two minutes. It writes everything under
`simulation_results/cyclic_mode/packed_bed_XXXXXXXX/`, where `XXXXXXXX` is the start of
the case fingerprint, and prints that run folder as its last line. Inside,
`1D_mode/cycle_N/` holds each cycle and `1D_mode/kpis/` the KPI tables of the final
cycle. Running the same case again is refused while that folder exists: continue it with
`--resume`, replace it with `--overwrite`, or set another `execution.extra_tag`.

The other commands are `pydac steps` (startup, then each listed step once),
`pydac nsga2` (an optimization campaign) and `pydac validate`. Each lists its options
with `--help`; `pydac validate --show-settings` lists the settings with their built-in
defaults. Instead of a complete case, `--config` takes a partial preset whose omitted
fields take the built-in defaults, for example
[`examples/cases/quick_1D.json`](examples/cases/quick_1D.json) from the repository:

```bash
pydac css --config examples/cases/quick_1D.json --dry-run
```

Exit codes: `0` success, `1` failure after the run started, `2` invalid command-line
arguments, configuration error or refused request, `3` CSS not reached while
`execution.require_css` is true.

## Quick start from Python

The same steps through the Python API, in a script or a notebook. This loads the saved
case, changes one setting, runs it and reads two KPIs:

```python
import pydac

case = pydac.load_case("my_case.json", settings={"execution.max_cycles": 4})
result = pydac.run_css(case)
print(result.folder, result.css_reached, result.cycle)

kpis = pydac.CycleKPIs(pydac.CycleDataLoader(result.folder).load())
print(f"CO2 recovery: {100 * kpis.recovery('CO2'):.1f} %")
print(f"Productivity: {kpis.productivity():.2f} kg CO2 per m3 bed per day")
```

`pydac.load_case(path, exact=True)` runs a complete case exactly, as `--case` does. The
supported API is the set of names in `pydac.__all__`: `load_case`, `export_case`,
`export_settings`, `run_steps`, `run_css`, `run_nsga2`, `resume`, `report`, `get_started`,
`RunResult`, `ConfigurationError`, `RunRefused`, `ExperimentConfig`, `CycleDataLoader`,
`CycleKPIs`, `register_equilibrium`, `register_kinetics`, `EquilibriumSystem`, `KineticLaw`,
`EquilibriumRegistryEntry`, `KineticRegistryEntry` and the version string `__version__`.

## Reproducibility

Every run folder contains the resolved case (`resolved_config.json`), its fingerprint
(`case_metadata.json`) and a provenance record: the pyDAC version, the Python version,
the platform and the versions of the numerical dependencies. Run from a clone, the
record also says whether the checkout had uncommitted changes; it does not record the Git
revision, so keep the commit of an unreleased checkout yourself. To
rerun a result, run its `resolved_config.json` with `--case` (in another output root, or
with `--overwrite`); to rebuild the environment, use the lock file above.

## Documentation

Read in this order: the tutorials (above) teach pyDAC on real runs; the user guide is
the reference for using it, in more technical detail; the reference manual describes how pyDAC
works inside, in full technical detail, for contributors and for checking the model. The
documentation index is [`docs/README.md`](docs/README.md). The user guide explains how to use
pyDAC:

- [Installation](docs/guide/01_installation.md): environments, Jupyter, shell
  completion, exact reproduction, troubleshooting.
- [Quick start](docs/guide/02_quickstart.md): one case from the terminal, a script
  and a notebook; the Python API and its errors.
- [Cases and settings](docs/guide/03_cases.md): presets, settings, frozen cases,
  model choices, `null`, fingerprints.
- [Running steps and cycles](docs/guide/04_running.md): step sequence, stop rules,
  resume, replay, report, run folders, exit codes.
- [Optimization campaigns](docs/guide/05_optimization.md): NSGA-II campaigns, bounds,
  constraints, stop rules, continuing and reporting a campaign.
- [Results](docs/guide/06_results.md): run folders, KPI files, reading results from
  Python, which folders to trust.
- [Reproducibility](docs/guide/07_reproducibility.md): sharing a case, provenance,
  the exact environment, compatibility across versions.
- [Extending pyDAC](docs/guide/08_extending.md): your own isotherms and kinetic laws,
  registered from Python; what needs a change to the package.

The reference chapters describe the model and the code:

| Chapter | Contents |
|---|---|
| [`00_framework_map.md`](docs/manual/00_framework_map.md) | Layers, entry points and what a run does end to end |
| [`01_inputs.md`](docs/manual/01_inputs.md) | Cases, presets, settings and the case document |
| [`02_pydac_core.md`](docs/manual/02_pydac_core.md) | Model construction, state vector, solver and cycle orchestration |
| [`03_pymrm_discretization.md`](docs/manual/03_pymrm_discretization.md) | Finite-volume operators and boundary conditions |
| [`04_governing_equations.md`](docs/manual/04_governing_equations.md) | Balances and closures, and where the code assembles them |
| [`05_cycle_steps.md`](docs/manual/05_cycle_steps.md) | Each cycle step: physics, boundary conditions, stop rules |
| [`06_post_processing.md`](docs/manual/06_post_processing.md) | KPIs, separators, plots and result files |
| [`07_nsga2_optimization.md`](docs/manual/07_nsga2_optimization.md) | NSGA-II campaigns |

To work on pyDAC itself, read [`CONTRIBUTING.md`](CONTRIBUTING.md).

## How to cite

If you use pyDAC in your research, please cite the pyDAC paper, available as a preprint on
ChemRxiv:

> M. Galanti, I. Roghair, M. van Sint Annaland. pyDAC: Open-Source two-dimensional modeling of s-TVSA for direct air capture: experimental assessment and validity of one-dimensional approximations.
> ChemRxiv (2026). [doi:10.26434/chemrxiv.15010382/v1](https://doi.org/10.26434/chemrxiv.15010382/v1)

and the software itself with the metadata in [`CITATION.cff`](CITATION.cff). Each GitHub release
has its own DOI on Zenodo; [10.5281/zenodo.23319859](https://doi.org/10.5281/zenodo.23319859)
always resolves to the latest version.

## License and authors

pyDAC is released under the MIT license ([`LICENSE`](LICENSE)). Authors: Mattia
Galanti, Ivo Roghair and Martin van Sint Annaland (Eindhoven University of Technology).
Maintainer: Mattia Galanti.

## Acknowledgements

We thank Shell Global Solutions International BV for financial support.
