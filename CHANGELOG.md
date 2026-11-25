# Changelog

## 3.2.0

### Added
- An optional vacuum pump of finite speed on the outlet of `blow`, `heat`, `w_purge` and `w_des`,
  selected per step with `flow.vacuum_outlet_model = "vacuum_pump"` on a 1D grid. The default
  outlet stays `"valve"`. `vacuum_pump_speed_m3_s` is the whole pump's asymptotic inlet speed and
  `vacuum_pump_ultimate_pressure_Pa` its ultimate pressure (default 0 Pa). A pressure control
  reduces the removal near the step's `outlet_pressure_Pa`, over a band
  `vacuum_setpoint_band_fraction` (default 1e-3, range [1e-6, 0.1]). When the bed releases gas
  faster than the pump can remove it, the pressure can stay above that band; the pump never
  supplies gas back. A null step setting inherits the lower layer, so it cannot clear a pump value
  set there. See guide 03 and manual 05 §2.7.
- Standby electricity of that pump: `vacuum_pump_idle_mode` is `"off"` by default, or
  `"minimum_speed"` / `"fixed_speed"` with a required positive vendor value
  `vacuum_pump_idle_power_W`. The added power is `P_idle * (1 - g)`, with `g` the control's
  throughput fraction; with `"off"` nothing is added. It is an averaged energy model, not a
  vendor part-load curve. See manual 06 §8.2.
- An optional feed blower for 1D pressurization, `flow.press_inlet_model = "feed_blower"`: a
  supply of finite capacity `feed_blower_flow_m3_s` (positive, at feed conditions) with a smooth
  closing band `feed_blower_band_fraction` (default 1e-3, range [1e-6, 0.1]). It never takes gas back.
  With the blower, the press valve conductance is ignored, even when it is zero. Its supply work,
  from ambient to the feed header (efficiency 0.7, suction at the feed temperature, zero at or
  below ambient), is the compression work of `press`, in place of the valve pressure-drop
  estimate. Saved profiles rebuild the inlet
  with the same law.
- An optional one-way outlet for 1D `ads` and `startup`:
  `flow.ads_outlet_overpressure_Pa` is the outlet-cell overpressure at the nominal feed velocity
  (bed-referred), and fixes a linear outlet law `u_out = K max(p - p_out, 0)` that never lets gas
  back in. Null (the default) keeps the fixed-pressure outlet. Steps with this outlet run CVODE
  BDF up to order 2, which is A-stable; every other step keeps order 3.
- `simulation.grid.output_schedule`: denser saved outputs at the start of each step (a geometric
  window, `dense_start_s` with `dense_start_points` and `first_output_s`) and a maximum spacing
  between saved outputs (`max_output_spacing_s`). They are added to the usual `nt` outputs; all
  four null (the default) saves exactly the outputs of 3.1.0. Saved outputs are observations:
  CVODE does not stop at each of them, but asking for them can shift its adaptive steps, so
  results can change at the level of the solver tolerances. A denser output schedule can catch a
  brief outlet reflux that coarser saved outputs miss; condenser reporting then refuses the run,
  because the product-line model does not describe reflux.
- Validation warnings for a pump step that stops on a falling pressure target at or below the
  upper edge of its control band, and for a blower step that stops on a rising target at or above
  the lower edge of its band: the step can stop on the pressure plateau or at the time cap before
  reaching the target. Place the target outside the band.

### Changed
- Case schema 11 (3.1.0 used schema 8). A document of pyDAC 3.1.0 or older (schema 8 or lower) is
  migrated: every step keeps the default valve outlet and valve pressurization, the output
  schedule is off and the one-way outlet is off, so a migrated case describes the same plant as
  before. A field introduced in 3.2.0 is refused in such a document, even when it is null; set
  `schema_version` to 11 to use it.
- On a step with a closed inlet (`blow`, `heat`, `w_purge`, `w_des` with zero inlet flow), the
  vented amount of each species is the decrease of the bed's inventory of that species over the
  step (plus the injected steam for H2O). In a mixture without H2O, the collected product of CO2
  and of each inert gas is that decrease too. Before, the CO2 product was the time integral of the
  saved outlet stream, which depends on how finely the outputs are saved, and the inert gases used
  the absolute value of their inventory change. For those steps the integral and the solver's
  accumulator are still saved as diagnostics (`mass.outlet.saved.simpson`,
  `mass.outlet.accumulator`); KPIs do not use them. In a mixture with H2O the collected product
  is unchanged: CO2 and water come from the condenser and vacuum-discharge flashes of the saved
  outlet stream, and the inert gases from their inventory change.
- The compression and vacuum work no longer give energy back: the flow and the pressure-ratio
  factor are each clamped at zero, so a moment of reversed flow or of expansion adds no negative
  work. The same holds for the pump work behind the condenser. Feed-blower supply work and pump
  standby power keep their own laws.
- These two changes and the fixes below move the energy and product figures of existing cases,
  and the integrated energy states take part in the solver's error control, so other outputs can
  move slightly too, even with every new option off.

### Fixed
- Maruyama et al. (2020), Eq. 10, is evaluated with pressure in atm: 101325 Pa instead of 1e5 Pa. At
  the same gas state, the vacuum gas-work term increases by approximately 0.064 % at 1 atm, 0.448 %
  at 10 kPa and 1.108 % at 1 kPa, for valve and pump outlets alike. A whole step's change depends on
  its pressure history.
- The condenser and vacuum-discharge CO2/H2O flashes conserve CO2 in steam-rich feeds. The old
  separator shortcut for `z_CO2 < 1e-3` returned pure liquid water and discarded that feed's CO2
  whenever the separator was called. The corrected flash keeps all feed in the liquid below the
  bubble point and splits it between liquid and vapor in the two-phase region. At 278.15 K and 20
  kPa the equilibrium liquid CO2 fraction is about 2.2e-4. Recovery, productivity and specific
  energy of steam cases change: the built-in DAC case with `grid.nr=1` and `grid.nz=25`, converged
  at cycle 3, changes from 74.6 % to 77.6 % recovery, with essentially unchanged purity.
- Every product-line flash uses the same Henry/Raoult K values for its liquid, vapor and two-phase
  tests. This removes a mismatch between the former ideal dew-point test and the split; two
  near-dew examples at 278.15 K/20 kPa and 298.15 K/1 atm lost about 6.1e-5 and 1.9e-4 mol CO2 per
  mol of feed. The cooling-water duty uses the same checked flash. Invalid inputs, failed fugacity
  calculations and non-finite or unbalanced results raise errors instead of silently becoming zero
  product or zero condensate. A feed CO2 fraction slightly outside [0, 1], which the outlet-face
  extrapolation can give next to a steep outlet gradient, is flashed as given so the balances
  close; more than 1e-3 outside gives a warning.
- Recovery is referred to the bed's net CO2 uptake from the feed over adsorption and
  pressurization, no longer over adsorption alone; the vented percentage uses the same reference,
  and the CSS summary reports the feed uptake. Pressurization takes feed gas through a closed
  outlet, and that CO2 was missing from the denominator: a cycle pressurized with 12 % CO2 after a
  5 kPa vacuum reported 104 % recovery (now 97.3 %). DAC cycles pressurized with air change by
  about -0.001 percentage points.
- The CSS summary report, the KPI outputs of a run and the NSGA-II candidate KPIs no longer fail
  on a cycle without `heat` or `w_purge` (e.g. `ads -> blow -> w_des -> press`):
  `preheating_temperatures()` returns `None` for both temperatures, the report omits the
  preheating rows, and an NSGA-II campaign with the `steam_condensation` constraint on such a
  steam cycle is refused before it starts (no step preheats the bed, so the constraint cannot hold).
- A gas of a single species (e.g. a pure-N2 start) no longer gives NaN in Blanc's mixture
  diffusivity. Where the law is 0/0, the species takes the equal-share limit over the other
  species: the binary diffusivity for two gases, the harmonic mean of the binary diffusivities for
  more. Every value the law can evaluate is unchanged.
- A `ValueError` from the model equations at a non-physical trial state of the solver is rejected,
  and CVODE retries, up to 1000 rejections per solve. A `ValueError` at a
  physical state is still raised. Non-finite saved states, times or event outputs mark the solve
  as failed even if the solver reports success.
- A refused CLI `--set` value names its setting and keeps the original error, for example
  "Setting `steps.blow.flow.vacuum_pump_idle_power_W`: Non-standard JSON constant 'NaN' is not
  allowed."
- `execution.cycle_options[n].termination_mode` is normalized and checked like the global and the
  per-step modes (e.g. `"EVENT"` is `event`, an unknown mode is refused). Before, it was used as
  written, so a value spelled otherwise did not select the mode it named.
- NSGA-II imports `StarmapParallelization` from `pymoo.parallelization`, its supported location,
  so pymoo no longer warns that the old import path is deprecated.

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
