# Changelog

## 3.1.0

### Added
- `pydac.register_bed_architecture(name, entry)`: a bed architecture besides the packed bed can be
  registered at run time, the same way as an isotherm or a kinetic law. The `BedArchitectureEntry`
  gives the design, the structure, and the pressure-drop, dispersion and thermal closures of the bed;
  the closures receive the step's values in a `ClosureInputs`. pyDAC checks every result and copies
  it into its own values before the column uses it. `ResolvedStructure`, `ResolvedTransport`,
  `ContextParameter` and `solve_quadratic_drag_velocity` are importable from `pydac`. See guide 08,
  "A bed architecture".
- A saved `simulation_context.pkl` of a registered architecture reads back without the registration;
  it carries the row `bed.registered_architecture` (the architecture's name), without which a
  context with a name that is not built in is refused.

### Changed
- A registered architecture runs only in the spatial modes it declares, and refuses both wall
  heat-transfer overrides (`wall.overall_htc_override_W_m2K`, `wall.bed_wall_htc_override_W_m2K`):
  it resolves its own bed-to-wall coefficient.
- The `SimulationContext` (schema 3) names the two wall coefficients apart: `bed.models.wall_htc`
  is the law of the bed-side coefficient h_w (`dixon`, `specchia_1980`, or `override`), and
  `bed.models.overall_wall_htc` the lumped 1D coefficient (`dixon_1996` or `override`; `none` in
  2D, which has no lumped coefficient). They replace `bed.models.heat_transfer`, which read
  `dixon_1996` in 2D too. A context saved by pyDAC 3.0.x (schema 2) is read in the new rows:
  `overall_wall_htc` from its old row (`none` for a 2D run), `wall_htc` as `unrecorded`. A
  registered architecture names its law with `BedArchitectureEntry.wall_htc_model`.
- An NSGA-II campaign started without any product constraint (purity, oxygen, steam condensation,
  equivalent work) warns that its Pareto front may hold products of any purity; every constraint
  is off by default.
- The PyPI page links the changelog (GitHub releases), the Zenodo concept DOI and the preprint;
  `CITATION.cff` carries the concept DOI, so "Cite this repository" on GitHub shows it.
- `.zenodo.json` gives Zenodo the full record of each new version (creators with affiliation and
  ORCID, the preprint as the describing publication, the funding note), so it no longer needs
  manual edits after a release.
- Terminal messages print a path inside the working folder relative to it (`Run folder:
  simulation_results/...`, `[Saved] ...`); a path elsewhere is still printed absolute. The files
  are written where they were, and `RunResult.folder` stays absolute.

### Removed
- `NSGA2_pp.pandas_postprocess` (low-W_eq tables, `low_energy_pareto.pdf`, `KDE_plots.pdf`): no
  part of pyDAC called it, and the campaign report already holds the Pareto front and its
  decision-variable plots.

### Fixed
- `external_htc_W_m2K = 0` is accepted: it is an adiabatic outer wall (no exchange with the
  jacket; the wall still exchanges heat with the bed). Before, every step `thermal` field had to
  be greater than zero, so the adiabatic setting was refused; negative values are still refused.
- The outlet species figures of a step run draw the stream that leaves the outlet face (net-flow
  mixing cup), the value the KPIs and the CSS figures use, instead of the last cell center; the
  cell value stays where no net stream leaves (closed outlet). On a coarse grid the two differ by
  up to one cell of the outlet gradient.
- Module headers no longer carry an e-mail address or a hand-written "Last updated" date.
- A terminal setting with an object value merges like the same block in Python:
  `--set 'sorbent={"density_kg_m3": 800}'` changes that field, where it used to replace the whole
  `sorbent` block and fail on the missing fields. Blocks marked "replace as one block" are still
  replaced whole, in both forms.
- A nested `{"cycle": {"w_des": ...}}` setting is accepted like its dotted form `cycle.w_des....`;
  only `{"cycle": {"steps": {...}}}` worked before.
- A dotted setting inside a species-keyed block reaches the species whatever its case:
  `adsorption.kinetics.CO2.parameters.k_LDF_s_inv=0.004` works on a case that writes `co2`, as
  species keys already did everywhere else. Only the species key is matched this way; model
  parameter names and every other key stay exact, and two spellings of one field in the same
  settings are refused as one setting supplied twice.
- A frozen case of an older schema written by an early build, before one of that schema's fields
  existed, is refused with the way to bring it up to date: load it as a preset (missing fields take
  the built-in defaults) and save it again with `pydac.export_case` or `--export-case`. Before, the
  refusal only named the missing field.
- An empty block setting (`--set 'grid={}'`, `settings={"grid": {}}`) is refused: it set nothing
  and skipped the check of its path, so a misspelled field passed unnoticed. An empty block on a
  "replace as one block" path is still a whole, empty replacement.
- An NSGA-II campaign on a cycle without a desorption step (`s_des` or `w_des` in
  `execution.cycle_steps`) is refused; it was taken as steam-assisted, and `t_des` and `v_steam`
  acted on no step. Resuming a checkpoint that the campaign settings no longer accept is a
  configuration refusal (exit code 2), not a traceback.
- Tutorial 05 lists its candidates in a fixed order (generation, then the decision variables), so
  the table is the same at every run; tutorial 01 says that a preset gives the same case with the
  same pyDAC version, while the numbers can differ in the last digits on another machine.
- CI and release workflows run every JavaScript action on Node 24 (`checkout@v5`,
  `setup-python@v6`, `upload-artifact@v6`, `download-artifact@v7`, `setup-uv@v7`); GitHub retires
  Node 20.
- `python -m pytest` from the repository root collects `tests/` only; before, it also walked into
  `simulation_results/` and failed on files found there. Guide 06 says how to keep the results on
  another disk through a `simulation_results` link.
- The qualification lock `ci/qualification-lock-py3.12.txt` installs on Windows too: it is one
  universal lock for Linux and Windows (platform-only packages carry markers: `colorama` on
  Windows). Every version and hash it had on Linux is unchanged; the unused PARDISO/MKL packages
  (`pypardiso`, `mkl`, `intel-openmp`, `intel-cmplr-lib-ur`, `onemkl-license`, `tbb`, `tcmlib`,
  `umf`) are gone. CI runs the qualification job on Windows as well; README, guide 01 and
  CONTRIBUTING give the Windows commands.
- `.gitattributes`: a clone checks out text files with LF on every platform. On Windows, Git
  converted them to CRLF, which changed the hash of the case file the qualification test checks.

The packed bed is unchanged: its computed results (trajectories, KPIs, CSS status, NSGA-II
fronts) are bit-identical to 3.0.1.
