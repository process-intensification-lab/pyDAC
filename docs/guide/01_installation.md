---
title: "Installation"
toc: true
toc-depth: 3
embed-resources: true
---

> **New to pyDAC? Do the [tutorials](../../examples/README.md) first, starting with
> [01 First steps](../../examples/notebooks/01_first_steps.ipynb).** They teach pyDAC step by
> step on real runs; this guide is the reference behind them and goes into more technical detail.

This page installs pyDAC in an environment of its own, checks it, and connects it to
Jupyter and to your shell. The [README](../../README.md) gives the short version.

## What you need

- Python 3.11, 3.12, 3.13 or 3.14.
- Linux (x86_64) is the supported platform. Windows is being tested; macOS is neither
  tested nor supported.
- About 1 GB of disk space for the environment (see
  [What gets installed](#what-gets-installed)).

## Why pyDAC needs its own environment

pyDAC fixes the versions of some dependencies: `pymrm==2.1.10`, `pandas<3` and
`scikit-sundae` 1.1.x. pip silently replaces versions that are already installed: in an
environment that already had pandas 3.0.6, installing pyDAC downgraded it to 2.3.3
without a question. In a shared environment, installing pyDAC can therefore break another
project, and installing another project can break pyDAC.

Install pyDAC only in a new environment created for it. Never use:

- the system Python;
- the conda `base` environment;
- an environment shared with another project.

## Create the environment

Pick one tool. Each block creates an environment and activates it; later commands on this
page run inside the active environment.

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

With conda:

```bash
conda create -n pydac python=3.12
conda activate pydac
```

With uv:

```bash
uv venv pydac-env
source pydac-env/bin/activate
```

A uv environment has no `pip`: wherever this page writes `python -m pip`, write `uv pip`
instead.

Activate the environment again in every new terminal before you use pyDAC.

## Install pyDAC

```bash
python -m pip install pydac3
```

Or install it from a clone of the repository:

```bash
git clone https://github.com/process-intensification-lab/pyDAC.git
cd pyDAC
python -m pip install .
```

One install contains every workflow: step runs, cyclic steady state (CSS) runs, case
validation and NSGA-II optimization. Shell completion is an optional extra, see
[Shell completion](#shell-completion).

### Names

| Name | What it is |
|---|---|
| `pydac3` | the distribution: the name you give to `pip install` |
| `pydac` | the package you import (`import pydac`) and the command you run (`pydac css`) |
| pyDAC | the project |

`pip install pydac` installs an unrelated project with the same name. If you installed it
by mistake, remove it with `python -m pip uninstall pydac`.

### What gets installed

The environment holds about 100 packages and 0.9 GB. pyDAC's own dependencies are numpy,
scipy, pandas, matplotlib, seaborn, pymoo, scikit-sundae and pymrm; pymrm 2.1.10 in turn
requires its documentation, notebook and test tools (sphinx, ipython, ipykernel, nbclient,
pytest, numba and others). They are installed even though pyDAC does not call them. One
useful consequence: the environment already holds a Jupyter kernel (ipykernel), so VS Code
or an existing Jupyter installation can run notebooks on it
([Use pyDAC in Jupyter](#use-pydac-in-jupyter)). The browser interface itself (JupyterLab
or Notebook) is not included: install it separately if you want it.

### Get the tutorials and this guide

The installed package also holds a starter kit: the README, the licence, the citation
file, this guide and the tutorials (cases, notebooks and data). Copy it into a new
folder:

```bash
pydac get-started my_pydac
cd my_pydac
```

(`pydac.get_started("my_pydac")` does the same from Python.) The folder has the layout
of the repository, so the tutorials run as written in
[`examples/README.md`](../../examples/README.md) from inside it. The kit has no source code,
tests or developer manual: its links to the manual open it in the public repository, at
the version you installed. `pydac get-started` refuses a folder that already holds files. A clone
of the repository already has everything; it needs no kit.

## Check the environment

```bash
python -m pip check
pydac --version
python -c "import pydac; print(pydac.__file__)"
```

A correct environment prints:

- `No broken requirements found.`;
- `pydac 3.0.0` (the installed version);
- a path inside the environment you created, ending in `site-packages/pydac/__init__.py`.

If the path lies in the user site instead (for example under `~/.local/lib/`), a
package installed with `pip install --user` shadows the environment's own. A venv
ignores the user site, but a conda environment searches it before its own packages. Set
`PYTHONNOUSERSITE=1` in the shell before running pyDAC, or remove the user-site
installation.

### The environment warning

When the `pydac` command runs outside a dedicated environment, it prints one line on
stderr and continues:

```text
Warning: pydac is running outside a dedicated environment (PREFIX). pyDAC pins some dependencies, and pip replaces installed versions without asking: install it in its own venv or conda environment (README, Installation). PYDAC_NO_ENV_WARNING=1 silences this.
```

The command counts as dedicated when it runs in a virtual environment (venv, uv) or in a
conda environment other than `base`. The warning never stops a run. `import pydac` never
prints it, so in scripts and notebooks the check above is your only safeguard. To silence
it on purpose, for example in a container that uses the system Python by design, set
`PYDAC_NO_ENV_WARNING=1`.

## Use pyDAC in Jupyter

Register the environment as a Jupyter kernel, once, with the environment active:

```bash
python -m ipykernel install --user --name pydac-env --display-name "Python (pyDAC)"
```

In Jupyter (or VS Code), select the kernel "Python (pyDAC)". A notebook running on another
kernel imports another environment's packages, or none. To check from a notebook cell:

```python
import sys
import pydac

print(sys.executable)
print(pydac.__file__)
```

Both paths must lie inside the pyDAC environment. To remove the kernel later:

```bash
jupyter kernelspec uninstall pydac-env
```

## Shell completion

The `completion` extra adds tab completion of the `pydac` subcommands and flags in bash
and zsh:

```bash
python -m pip install "pydac3[completion]"
eval "$(register-python-argcomplete pydac)"
```

The second line enables completion in the current shell; run it again in each new shell
after activating the environment. Then `pydac cs` + Tab completes to `pydac css`, and
`pydac css --ov` + Tab lists `--overwrite` and `--overwrite-case`.

## Exact reproduction

To rebuild the environment in which pyDAC was qualified, every package at the exact
version and checked against its hash, use the lock file of the repository. It is written
for Python 3.12, and the same file serves Linux and Windows. From the root of a clone, in a
second fresh environment; Linux, with `venv`:

```bash
python3.12 -m venv pydac-exact
source pydac-exact/bin/activate
python -m pip install --require-hashes -r ci/qualification-lock-py3.12.txt
python -m pip install --no-deps .
```

`--no-deps` installs pyDAC itself without letting pip change any version the lock fixed.

Windows, in PowerShell, with the `py` launcher (the same lock file):

```powershell
py -3.12 -m venv pydac-exact
pydac-exact\Scripts\Activate.ps1
python -m pip install --require-hashes -r ci/qualification-lock-py3.12.txt
python -m pip install --no-deps .
```

## Update or remove

To update pyDAC, run `python -m pip install --upgrade pydac3` in its environment, then
`python -m pip check`. A new pyDAC version may require other dependency versions; if
`pip check` reports a conflict, create a new environment instead of repairing the old one.

To remove pyDAC, delete the environment: the `pydac-env` folder, or
`conda env remove -n pydac`. Remove the Jupyter kernel as shown above.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pydac: command not found` | The environment is not active. Activate it (`source pydac-env/bin/activate`). |
| `pydac.__file__` points into `~/.local` | A user-site installation shadows the environment. Set `PYTHONNOUSERSITE=1`, or uninstall the user-site copy. |
| `pip check` reports a broken requirement | Another install changed a version pyDAC fixes. Create a new environment and install pyDAC alone in it. |
| `import pydac` fails in a notebook | The notebook runs another kernel. Select "Python (pyDAC)" (see [Use pyDAC in Jupyter](#use-pydac-in-jupyter)). |
| `pip install pydac` installed version 0.0.x | That is an unrelated project. Uninstall it and install `pydac3`. |

## Related pages

- [README](../../README.md): overview and quick start.
- [CONTRIBUTING](../../CONTRIBUTING.md): the environment for working on pyDAC itself.
