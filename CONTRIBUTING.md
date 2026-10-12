# Contributing to pyDAC

This page is for developers working in a clone of the repository. To use pyDAC, read
[`README.md`](README.md) instead. Rules come first, each with its reason in one line.

## Using pyDAC or changing it

| You want to | Install | Which code runs |
|---|---|---|
| use pyDAC | `python -m pip install pydac3` ([README](README.md#2-install-pydac)) | a copy fixed at install time; files you edit are not used |
| change pyDAC | a clone, installed editable (`python -m pip install -e .`) | the clone's `src/pydac/`: an edit is used at the next import |

An editable install copies no code: it points the environment at the clone. Everything
below is about the second row.

## Environment

- **One dedicated environment per clone, never shared between clones.** An editable
  install points the environment at one clone; a second clone in the same environment
  silently runs the first one's code.
- **Disable the user site.** Packages under `~/.local` shadow the environment's own
  numpy, scipy and pandas: set `PYTHONNOUSERSITE=1` in the shell that runs pyDAC.
- **Pick the light or the exact environment.** The light one takes the newest allowed
  version of every dependency and works with any supported Python; the exact one is the
  qualification lock, the only one in which your numbers match the qualified ones digit
  for digit and the `qualification` tests hold.

Get the code: fork the repository on GitHub (if you plan to propose changes) and clone
your fork:

```bash
git clone https://github.com/<your-user>/pyDAC.git
cd pyDAC
```

**Light environment**, from the root of the clone, with Python 3.11 to 3.14:

```bash
export PYTHONNOUSERSITE=1
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install --group test
python -m pip check
python -c "import site, pydac; print(site.ENABLE_USER_SITE, pydac.__file__)"
```

**Exact environment**, from the root of the clone, with Python 3.12 (the lock is written
for it):

```bash
export PYTHONNOUSERSITE=1
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r ci/qualification-lock-py3.12.txt
python -m pip install --no-deps -e .
python -m pip check
python -c "import site, pydac; print(site.ENABLE_USER_SITE, pydac.__file__)"
```

In both, the last line must print `False` and a path inside this clone's `src/pydac/`.
The name `.venv` is the one VS Code detects on its own. The lock already holds pytest.

- **Restart running Python sessions after editing sources.** An editable install reads
  the sources when a module is first imported; a running interpreter or notebook kernel
  keeps the old code until it restarts.
- **Reinstall after changing `pyproject.toml`** (dependencies, entry points, metadata):
  run `python -m pip install --no-deps -e .` again. Until then the installed metadata is
  stale, which is why `importlib.metadata.version("pydac3")` is never used as the
  version.

## What is supported

| Level | What | Promise |
|---|---|---|
| Command | `pydac steps`, `pydac css`, `pydac nsga2`, `pydac validate` and their flags | documented; changes announced in the release notes |
| API | the names in `pydac.__all__` | documented; changes announced in the release notes |
| Internal | every other module under `src/pydac/` | shipped and importable, no stability promise |
| Clone tools | `runners/`, `tests/`, `docs/`, `ci/` | for working in a clone; not in the wheel |

User documentation shows only commands and `pydac.__all__` names.

## Running from VS Code

The files in [`runners/`](runners/) are launchers for this clone; they are not shipped in
the wheel. The step, CSS and campaign runners call the `pydac` command with the constants
at their top; `validate_config.py` has no constants and passes its command-line arguments
to `pydac validate`:

| Runner | Same as |
|---|---|
| `runners/run_steps_simulation.py` | `pydac steps` |
| `runners/run_css_simulation.py` | `pydac css` |
| `runners/run_NSGA2.py` | `pydac nsga2` |
| `runners/validate_config.py` | `pydac validate` |

| Constant | Effect |
|---|---|
| `CONFIG_PATH` | a partial preset, as `--config` |
| `DEVELOPMENT_SETTINGS` | settings applied after the preset and before `--set` |
| `CASE_PATH` | a complete case run exactly, as `--case` |
| `RESUME_PATH` | a saved run to continue, as `--resume` |
| `OVERWRITE` | replace this case's existing run folder, as `--overwrite` (step and CSS runners only) |

- Precedence: built-in defaults < `CONFIG_PATH` (or `--config`) < `DEVELOPMENT_SETTINGS`
  < `--set`. `pydac` itself never reads these constants; only the runner passes them.
- `CASE_PATH` refuses `CONFIG_PATH`, `--config`, `--set` and non-empty
  `DEVELOPMENT_SETTINGS`: a frozen case runs exactly as saved.
- Running the same case twice is refused (exit code `2`) while its run folder exists:
  set `OVERWRITE = True`, resume it, or change `execution.extra_tag`. A campaign has no
  `OVERWRITE`: continue it with `RESUME_PATH`, or delete its folder yourself.
- A CSS (cyclic steady state) run that stops at `execution.max_cycles` without reaching CSS exits with code
  `3` when `execution.require_css` is true. Errors go to stderr.
- Select this clone's `.venv` as the VS Code interpreter.

## Tests

```bash
python -m pytest -q -m "not qualification"
```

The light environment installs pytest with `--group test`, which needs pip 25.1 or
newer. Tests marked `qualification` compare fingerprints and numbers that hold only in
the exact environment; inside it, run the full suite with `python -m pytest -q`.

## Proposing a change

- **Open an issue before the code** for anything beyond a small fix: say what you want to
  change and why. A change to the physics core needs the maintainer's agreement first
  ([below](#physics-core-maintainer-review)).
- **One topic per pull request, on its own branch** of your fork, started from `main`:
  a reviewer can then accept or refuse each change on its own.
- **Run the checks in your clone before opening the pull request:**

  ```bash
  python -m pytest -q -m "not qualification"
  python docs/check_links.py
  ```

  New behaviour comes with a test that fails without it.
- **Open the pull request against `main`.** CI builds the wheel, installs it on Linux
  (Python 3.11 to 3.14) and on Windows, runs the tests, a short run of the `pydac`
  command and the link check. A pull request is merged with CI green and the
  maintainer's review.
- **Say in the pull request whether results change, and which cases you compared.** The
  `qualification` job of CI fails when numbers move; a change that moves them on purpose
  needs the reason, and the maintainer regenerates the baseline.
- **Keep the documentation in step.** A change to a command, a setting or a
  `pydac.__all__` name updates the guide or the manual in the same pull request, and its
  description says what users will notice, for the release notes.

## Code rules

- **Absolute imports only:** `from pydac.config import ...`. No `sys.path` edits and no
  imports of `src.` or of top-level folders; with the `src/` layout, `import pydac` works
  from a clone only through the editable install.
- **No formatter or linter pass.** A formatting pass rewrites every line of the physics
  files, which would hide the physical changes under review; do not reformat code you do
  not change.
- **Code headers cite reference sections** (`# Guide: docs/manual/01_inputs.md §3`), and
  `docs/check_links.py` checks them. When you renumber or move a section, update every
  header that cites it.
- **Plain words in code and messages.** Write what a rule does, not an internal decision
  label.

## Physics core: maintainer review

Changes to [`src/pydac/core/column_model.py`](src/pydac/core/column_model.py) and to
anything under [`src/pydac/post_processing/`](src/pydac/post_processing/) need the
maintainer's review before they are written: open an issue that states what you intend
to change, why (the physical or numerical reasoning) and the expected effect on results.
These files define what pyDAC computes and reports.

## Extending the model

An isotherm or a kinetic law of your own can be used without changing pyDAC: register it from
Python ([Extending pyDAC](docs/guide/08_extending.md)). Adding one to the package, so that
every user and the `pydac` command know it, or adding a species, is code work in a clone;
pull requests are welcome. The contract each addition must meet is in
[`docs/manual/01_inputs.md` §10](docs/manual/01_inputs.md#10-extending-the-contract):

- **Species:** a `SpeciesProperties` entry in
  [`src/pydac/core/species_catalog.py`](src/pydac/core/species_catalog.py) (molar mass,
  Lennard-Jones parameters, Fuller volume, conductivity and Shomate heat-capacity
  coefficients).
- **Isotherm:** an entry in `EQUILIBRIUM_REGISTRY` in
  [`src/pydac/core/isotherm_models.py`](src/pydac/core/isotherm_models.py).
- **Kinetic model:** an entry in `KINETIC_REGISTRY` in
  [`src/pydac/core/kinetic_models.py`](src/pydac/core/kinetic_models.py).

The built-in registries, the species catalog and the bed-architecture registry are read-only
at run time. `pydac.register_equilibrium` and `pydac.register_kinetics` add models for one
Python session only, beside the built-in ones, and refuse a built-in name.

## Saved files and versions

- A change to the shape of a saved file (case, run folder records, campaign checkpoint)
  bumps that file's version number.
- Readers keep accepting every older version; a newer version than the reader knows is
  refused with a message naming the file.
- A new case field needs a case-schema version bump and an upgrade step that gives
  older cases the value reproducing what they computed
  ([`src/pydac/config/migrations.py`](src/pydac/config/migrations.py)).
- From a later release on, each release will add a small set of frozen saved files under
  `tests/data/`, with a test that the installed package reads, reports, replays and
  continues them.

## Dependency pins

`pymrm` is pinned exactly (`pymrm==2.1.10`) because pyDAC calls internals of it that
changed after that version; `scikit-sundae` is held to 1.1.x and `pandas` below 3.
Raising any of these is a release decision: rerun the qualified cases in the new
environment and compare them with the current results first.

## Tools

| Task | Command |
|---|---|
| Regenerate the JSON schema of the case document after changing a default or rule | `python -m pydac.config.schema_generation` |
| Draw the analytical Jacobian sparsity mask | `python -m pydac.core.sparsity_patterns.analytical_sparsity --nz 10 --nr 5 --out FOLDER` |
| Check a step's recorded Jacobians against the analytical mask | `python -m pydac.core.sparsity_patterns.numerical_sparsity FOLDER --nz 10 --nr 5` |
| Check documentation links and code `# Guide:` headers | `python docs/check_links.py` |
| Regenerate the smoke-test baseline (qualification environment only) | `python tests/tools/generate_v3_smoke_baseline.py` |

Environment variables for developers (do not set them for production runs). The two
solver switches are recorded in each run's provenance (`software.solver_environment`); the
other two are not:

| Variable | Effect |
|---|---|
| `PYDAC_NO_ENV_WARNING=1` | silences the dedicated-environment warning of the `pydac` command |
| `PYDAC_PROFILE=PATH` | replaces the built-in defaults with a complete JSON profile |
| `PYDAC_RHS_CSR_OPERATORS=0` | applies the fixed operators in their original sparse format (A/B checks) |
| `PYDAC_CACHE_SLICE_WRITE=0` | disables the contiguous-slice writes of the operator cache (A/B checks) |

## Releases

- The version lives in one place, `src/pydac/utils/version.py`.
- A published tag is never moved or re-created; a fix is a new version.
