---
title: "06 — Post-processing, step metrics and KPI methodology"
toc: true
toc-depth: 3
embed-resources: true
---

> **This manual is the detailed technical reference: how pyDAC works inside, for contributors
> and for checking the model.** To learn to use pyDAC, do the [tutorials](../../examples/README.md)
> first, then read the [user guide](../guide/01_installation.md).

This chapter is the methodology reference for the pyDAC post-processing layer.
It describes what is extracted from each simulated step of the cycle
(by default the temperature-vacuum swing adsorption, TVSA, cycle), how typed step metrics are
computed, how product and separator quantities are reconstructed, and how the
cycle-level key performance indicators (KPIs) of a cyclic steady state (CSS)
run are assembled.

The intent is physical and thermodynamic documentation rather than numerical
solver documentation. Implementation details are included only where they define
the meaning of a reported KPI. To read results from a run, see the user guide
page [Results](../guide/06_results.md); to regenerate the report of a saved CSS
run, [Report a saved CSS run](../guide/04_running.md#report-a-saved-css-run). The
folders a step run writes are drawn in
[00 — Framework Map §7](00_framework_map.md#7-artifacts-on-disk).

Main source files, under `src/pydac/`:

- [`post_processing/profiles/step_post_processing.py`](../../src/pydac/post_processing/profiles/step_post_processing.py)
- [`post_processing/profiles/step_plotting.py`](../../src/pydac/post_processing/profiles/step_plotting.py)
- [`post_processing/metrics/step_metrics.py`](../../src/pydac/post_processing/metrics/step_metrics.py)
- [`post_processing/metrics/step_metrics_store.py`](../../src/pydac/post_processing/metrics/step_metrics_store.py)
- [`post_processing/context/simulation_context.py`](../../src/pydac/post_processing/context/simulation_context.py)
- [`post_processing/context/store.py`](../../src/pydac/post_processing/context/store.py)
- [`post_processing/context/runtime_store.py`](../../src/pydac/post_processing/context/runtime_store.py)
- [`post_processing/metrics/cycle_data.py`](../../src/pydac/post_processing/metrics/cycle_data.py)
- [`post_processing/metrics/cycle_data_loader.py`](../../src/pydac/post_processing/metrics/cycle_data_loader.py)
- [`post_processing/metrics/cycle_metrics.py`](../../src/pydac/post_processing/metrics/cycle_metrics.py)
- [`post_processing/kpi/cycle_kpis.py`](../../src/pydac/post_processing/kpi/cycle_kpis.py)
- [`post_processing/kpi/reporting.py`](../../src/pydac/post_processing/kpi/reporting.py)
- [`post_processing/kpi/plotting.py`](../../src/pydac/post_processing/kpi/plotting.py)
- [`post_processing/profiles/cyclic_plotting.py`](../../src/pydac/post_processing/profiles/cyclic_plotting.py)
- [`utils/postprocessing/`](../../src/pydac/utils/postprocessing/)
- [`core/column_model.py`](../../src/pydac/core/column_model.py) for the
  integrated KPI states solved together with the PDEs

`core/column_model.py` and everything under `post_processing/` define what pyDAC
computes and reports; changes to them need maintainer review
([CONTRIBUTING](../../CONTRIBUTING.md#physics-core-maintainer-review)).

---

## 1. Scope and data flow

### Execution provenance (schema 1)

Each solve records the pyDAC version, originating case,
Python and numerical-library versions, available SUNDIALS/BLAS information,
thread/backend settings, step configuration and a hash of its initial state.
[`utils/version.py`](../../src/pydac/utils/version.py) is the single version
source; a version ending in `.dev0` identifies unreleased code.
Unknown native build details remain null, so this record is not an environment
lockfile or a claim of bitwise portability across machines.

Records use `schema_version`, `record_id`, `created_at`, `activity`, `software`,
`input`, `execution`, `sources` and `artifacts`. Each activity gets a UUID and
UTC timestamp. JSON records are atomically saved under `provenance/<UUID>.json`;
an existing identity cannot be replaced with different content. Solve records
are also embedded as `solve_provenance` in `sol.pkl` and in `terminal.pkl`, the
compact end state an NSGA-II candidate keeps instead of `sol.pkl`.
An activity record establishes origin, not successful completion: consult the
existing solver outcome or optimization summary for success and termination.
`software.solver_environment` records the boolean values of `PYDAC_RHS_CSR_OPERATORS` and
`PYDAC_CACHE_SLICE_WRITE`, the two selectable arithmetic paths. When the owning module is
loaded, provenance reports its bound flag (the value the arithmetic actually used) rather than
rereading an environment variable that may have changed; otherwise it reads the environment.

The input block carries a complete portable case snapshot with
`execution.output_root` replaced by `"."`. Its `case_metadata` fingerprints that
snapshot; `source_case_metadata` retains the original case fingerprint and
`relocated_fields` declares the difference. Physical/numerical inputs and the
fingerprint in the original `case_metadata.json` are unchanged. Low-level callers
that construct a model without an originating experiment record unknown case
provenance and retain the available resolved column configuration.

Git is optional. Source-local inspection records `dirty` and one of `available`,
`no_checkout`, `git_unavailable` or `inspection_failed`, without blocking an
ordinary run when Git fails. Runtime revisions, branches, remotes, hostnames and
installation paths are not collected. A local revision is marked
`unverified_or_non_public` and withheld. Separately, `software.public_source`
is null unless a qualified release build supplies its public commit URL and SHA;
the presence of a remote alone does not qualify a public source.

Recomputing step metrics writes a new `metrics_provenance.json`, linking the
current processing configuration/software to the original solve. Reloading a
trajectory never assigns today's software to its solve. `CycleDataLoader`
captures step-metric provenance plus hashes of the loaded metric/context files,
checks for changes while loading, and carries that snapshot into `CycleData`.
Consequently, `KPIReporting.save_css_summary_report()` can write reports elsewhere
after the source directory has moved. Its `<cycle>_report_provenance.json` records
the processing software, sources, output hashes and the requested species,
temperatures and efficiencies. Numerical table columns and report formulas are
unchanged. UUID records retain earlier processing identities when reports are
regenerated; they do not archive the contents of overwritten reports.

Supported older results without provenance load with an explicit unknown origin.
Malformed or newer provenance schemas are rejected. This compatibility does not
extend support to older physical result formats that the existing loaders reject.
Original readers that enforce exact artifact keys may not read the new optional
terminal provenance field; use the current reader for new results.

NSGA-II candidates have separate records; compact candidate JSON retains
provenance before transient folders are pruned. Continuation records its checkpoint
hash and current software without changing existing checkpoint compatibility
checks. Loading an NSGA-II front/population records a separate analysis activity
and hashes of those NPZ files. That record identifies the loaded data; it does not
archive subsequent plot/export options or every generated figure.

After each cycle step, the solver writes a dimensional `sol.pkl`. The
post-processing layer then performs these operations:

1. Reconstruct physical fields and auxiliary profiles for one step.
2. Compute an immutable `StepMetrics` collection from those fields.
3. Materialize the collection as the legacy four-column table saved in `kpis.pkl`.
4. Save one versioned, architecture-aware `simulation_context.pkl` DataFrame
   for each newly solved case/cycle.
5. Save or extend the validated, human-readable `resolved_runtime.json` report.
6. Load the step tables and required context into `CycleData` for cycle-level KPIs.

The step-level object is:

```python
StepPostProcessor(model, solver)
```

The pure cycle-level calculation path starts from the two public names
`pydac.CycleDataLoader` and `pydac.CycleKPIs`:

```python
import pydac

cycle_data = pydac.CycleDataLoader("simulation_results/cyclic_mode/packed_bed_7a4930a5").load()
cycle_kpis = pydac.CycleKPIs(cycle_data)
```

The folder is a run folder, its results folder or one `cycle_N/` folder;
`cycle_index=` selects a cycle (default: the highest-numbered `cycle_N/` folder,
which is the CSS cycle when the run stopped there; the loader does not check that
the folder is complete, so name the cycle of an interrupted run). [Results](../guide/06_results.md#read-a-css-cycle-from-python)
shows these calls on a real run.

At the end of a CSS run, and when a saved CSS run is reported
(`pydac css --report RUN`, `pydac.report(RUN)`), `api._css_outputs()` runs this
chain on the reported cycle of the results folder:

```python
CyclicSteadyStateProfiles(base_folder=results_folder, css_cycle_idx=cycle).plot_all_css_fields(fmt=fmt)
cycle_kpis = CycleKPIs(CycleDataLoader(results_folder, cycle_index=cycle).load())
KPIPlotting(cycle_kpis, output_folder=results_folder / "kpis").plot_all_kpi_figures(fmt=fmt)
KPIReporting(cycle_kpis, output_folder=results_folder / "kpis").save_css_summary_report(print_report=True)
run_oxygen_hot_bed_check(results_folder, cycle)
```

At the end of a run the profile figures follow `execution.plot_cyclic_profiles`
and the KPI figures and summary follow `execution.generate_kpi_outputs`; a report
draws both whatever the saved settings say. `fmt` is `execution.plot_format`
(default `pdf`). KPIs need an adsorption step (`ads`) in the reported cycle:
cycle 1 starts with `startup` instead, so KPIs are reported from cycle 2 on. The
oxygen and hot-bed check (`utils/solver/oxygen_hot_bed_audit.py`) is report-only
and always runs. A step run has no report.

Ordinary cycle calculations, reporting, and KPI plots consume only the typed
step/context DataFrames. Full transient CSS field and outlet-profile plots are
separate: `CyclicSteadyStateProfiles` reads the retained `sol.pkl` files because
the solver trajectory is the quantity being visualized.

The two layers have different roles:

| Layer | File (under `src/pydac/`) | Main role |
|---|---|---|
| Step reconstruction and physics | `post_processing/profiles/step_post_processing.py` | Reload `sol.pkl`, recover dimensional fields, and compute the complete scalar output for one step |
| Typed step contract | `post_processing/metrics/step_metrics.py` | Define stable metric identities, units, basis, and the immutable `StepMetrics` payload |
| DataFrame and persistence adapter | `post_processing/metrics/step_metrics_store.py` | Display and save step metrics using the historical `Step`, `Species`, `KPI`, `Value` schema and `kpis.pkl` filename |
| Simulation-context contract | `post_processing/context/simulation_context.py` | Validate static geometry/design/structure plus per-step transport, pressure, dispersion, closure inputs, and provenance |
| Simulation-context persistence | `post_processing/context/store.py` | Save and incrementally merge compatible step blocks in `simulation_context.pkl`, rejecting mixed designs or repeated-step mismatches |
| Runtime-report persistence | `post_processing/context/runtime_store.py` | Save and incrementally merge immutable step blocks in `resolved_runtime.json` |
| Cycle data loading | `post_processing/metrics/cycle_data_loader.py` | Assemble ordered step metrics and the required current-schema context without reading solver results |
| Pure cycle calculations | `post_processing/kpi/cycle_kpis.py` | Calculate mass, product, and energy KPIs from the validated `CycleData` bundle |
| KPI reporting | `post_processing/kpi/reporting.py` | Assemble summary values and tables, format text, and persist CSV/JSON/TXT outputs |
| KPI plotting | `post_processing/kpi/plotting.py` | Plot mass changes, residual gas, and energy breakdowns from `CycleKPIs` |
| Step profile plotting | `post_processing/profiles/step_plotting.py` | Plot one step's outlet traces and internal profiles, only with `execution.plot_profiles` (§19) |
| Transient CSS plotting | `post_processing/profiles/cyclic_plotting.py` | Plot saved full-cycle fields and outlet profiles from step solutions |
| Typed access utilities | `utils/postprocessing/` | Normalize species/basis aliases and query stable metric identities |

---

## Saved analysis data

Each newly solved case or cycle stores one architecture-aware context table
beside its step folders. A cycle of a CSS run:

```text
cycle_2/
├── simulation_context.pkl
├── resolved_runtime.json
├── cycle_metadata.json
├── css_diagnostics.json
├── ads/kpis.pkl
├── blow/kpis.pkl
└── ...
```

Schema version 2 uses the columns `Category`, `Parameter ID`, `Step`, `Value`,
and `Unit`. Rows with an empty `Step` contain the static grid, geometry,
particle, architecture, design, material, resolved structure, and provenance.
Rows carrying a step key contain the actual resolved heat-transfer,
pressure-drop and dispersion models; thermal and species-dispersion
coefficients; dimensional quadratic pressure-law coefficients; and the
temperature, pressure, velocity, and composition sources used to resolve that
step.

Packed adsorbent mass is derived from the architecture-neutral saved inventory:

$$
m_{adsorbent} = V_{bed}\rho_{sorbent}.
$$

`cycle_metadata.json` remains the execution record: step order, termination,
timing, and CSS probe step. `css_diagnostics.json` is the CSS check of that cycle,
written by the CSS workflow. Dynamic quantities such as species inventories remain
per-step metrics. The current post-processing requires a complete schema-v2
`simulation_context.pkl`; a results folder without one is not adapted or
reconstructed from current defaults and must be regenerated.

`resolved_runtime.json` schema version 1 is a separate, readable initialization
report. Its top-level sections are `provenance`, `state_layout`, `models`,
`species_catalog`, `static_context`, `spatial_values_policy`, and `steps`.
Every step block contains:

- the temperature, pressure, gas composition, and dispersion evaluation state;
- evaluated scalar, per-species, and pairwise properties, each with units;
- the resolved step context;
- a field-by-field reference to `<step>/sol.pkl`, including layout signature,
  storage group, shape, and unit.

Spatial histories are deliberately not copied into JSON. They remain in
`sol.pkl`, where the saved `StateLayout` gives their physical names. Saving a
new step merges it in canonical cycle order. Existing completed blocks and all
static sections are immutable: a mismatch raises instead of overwriting the
earlier report. `resolved_config.json` remains the independently reloadable
input document; the runtime report is evaluated output and is not an input.

---

## 2. Saved solution layout

The solver state is split into dimensional storage groups before it is saved:

```python
{
    "t": np.array,          # dimensional time [s]
    "y_2d": np.array,      # (n_2d_fields, Nz, Nr, Nt)
    "y_1d": np.array,      # (n_1d_fields, Nz, Nt)
    "y_scalars": np.array, # (n_scalar_states, Nt)
    "extra": {
        "state_layout": {...}, # schema, names, shapes, units, offsets, signature
        ...                     # grid and geometry metadata
    },
}
```

The authoritative order is generated from the declaration: concentrations for
all non-closure gas species, loadings for all adsorbates, `c_tot`, `T_bed`,
`T_wall`, then named mass and energy scalars. `StepPostProcessor`, CSS helpers,
and cyclic plotting deserialize and verify that saved layout before reading any
field. A layout-less solution is accepted only when its storage counts are the
exact historical DAC bundle (7 two-dimensional, 1 one-dimensional, and 5
scalar fields); every other ambiguous legacy file is rejected.

For the current DAC declaration, this generation reproduces the historical
fields `c_CO2`, `c_H2O`, `c_O2`, `q_CO2`, `q_H2O`, `c_tot`, `T_bed`, `T_wall`,
`m_CO2`, `m_H2O`, `E_compression`, `E_vacuum`, and `E_jacket` exactly. A
CO2/N2 breakthrough declaration instead saves `c_CO2`, `q_CO2`, `c_tot`,
`T_bed`, `T_wall`, `m_CO2`, and the same named energy scalars—without a new
post-processing branch.

The scalar states are already dimensional when loaded by `StepPostProcessor`.
The time vector is also dimensional, in seconds.

---

## 3. Reconstructed physical fields

For every declared non-closure species and every time point/grid cell, the
post-processor reconstructs gas mole fractions as:

$$
y_i = \frac{c_i}{c_{\mathrm{tot}}+\epsilon}.
$$

The species declared as the closure is not an independent solved
concentration. It is reconstructed by name:

$$
c_{\mathrm{closure}}
= c_{\mathrm{tot}} - \sum_{i\ne\mathrm{closure}}c_i,
$$

and:

$$
y_{\mathrm{closure}} = \frac{c_{\mathrm{closure}}}{c_{\mathrm{tot}}+\epsilon}.
$$

Small negative concentrations and mole fractions are clipped after loading.
This is a post-processing safeguard against tiny numerical undershoots from the
transport discretization. It should not be interpreted as an additional physical
model.

The pressure field is reconstructed from the ideal-gas relation:

$$
p = c_{\mathrm{tot}} R T_b.
$$

The mixture density used for velocity reconstruction and mixed-cup averaging
uses catalog molecular weights for every declared gas species:

$$
\rho_g = \sum_i c_i MW_i.
$$

---

## 4. Geometry, integrals and averages

The saved geometry provides axial cell volumes and radial ring areas:

$$
V_{jr} = \Delta z_j A_r,
$$

where $j$ is the axial cell index and $r$ is the radial ring index.

The gas holdup used for gas inventories is:

$$
\epsilon_g
= \epsilon_b + (1-\epsilon_b)\epsilon_p.
$$

The solid mass in a cell is:

$$
m_{s,jr}
= \rho_p (1-\epsilon_b) V_{jr}.
$$

Area-weighted outlet averages use:

$$
\langle \phi \rangle_A(t)
= \frac{\sum_r \phi_{L,r}(t) A_r}{\sum_r A_r}.
$$

For native saved outlet observations, composition is the net molar-flow ratio:
`y_i = sum_r(n_dot_i,r) / sum_r,sum_j(n_dot_j,r)`. A mass-weighted average of
cell mole fractions is not the same quantity. `StepPostProcessor.get_mixed_cups()`
also computes temperature from saved sensible enthalpy and species heat-capacity
flows relative to the 298.15 K datum. Nonpositive total net flow has no mixing cup
and is represented by NaN.

Cyclic y/ppm curves use this native composition for every open-outlet step, including
closed-inlet venting steps. Closed outlets retain the bed cross-section reading and
zero displayed velocity. The cyclic `T_bed` trace is an average of last-cell
values (rule below); it is not the thermodynamic `T_out` mixed-cup temperature.
Saved solutions without face histories (`outlet_face`; every current solve saves
it) apply the per-step rule below to the last-cell mole fractions instead. A
vector-drag artifact without native outlet observations is rejected.

### 4.1 Which average each outlet figure uses

An outlet figure reads either the outlet face (native composition, above) or the
last axial cell, reduced over the radial rings by the area-weighted mean
$\langle \phi \rangle_A$ above or by the mass-flux-weighted mean

$$
\langle \phi \rangle_{\dot m}(t)
= \frac{\sum_r \phi_{L,r}(t)\,\max\!\left(\rho_{L,r}u_{L,r},0\right) A_r}
       {\sum_r \max\!\left(\rho_{L,r}u_{L,r},0\right) A_r},
$$

which falls back to $\langle \phi \rangle_A$ wherever its denominator is zero. In 1D
(`grid.nr = 1`) both means are the single cell value.

CSS figures in `plots_css/` (`CyclicSteadyStateProfiles`). The step classes are
`CLOSED_INLET_STEPS` and `CLOSED_OUTLET_STEPS` of
[`cyclic_plotting.py`](../../src/pydac/post_processing/profiles/cyclic_plotting.py):

| Figure | Open-flow steps: `startup`, `ads`, `s_purge`, `s_des` | Closed-inlet steps: `blow`, `heat`, `w_purge`, `w_des` | Closed-outlet steps: `cool`, `press` |
|---|---|---|---|
| `css_outlet_y_<species>`, `css_outlet_ppm_<species>` | native composition (NaN without a net outgoing stream) | native composition (NaN without a net outgoing stream) | $\langle y \rangle_A$, last cell |
| `css_outlet_T_bed` | $\langle T_b \rangle_{\dot m}$, last cell | $\langle T_b \rangle_A$, last cell | $\langle T_b \rangle_A$, last cell |
| `css_outlet_q_<adsorbate>` | $\langle q \rangle_A$, last cell | $\langle q \rangle_A$, last cell | $\langle q \rangle_A$, last cell |
| `css_outlet_pressure` | $\langle c_{\mathrm{tot}}RT_b \rangle_A$, last cell | $\langle c_{\mathrm{tot}}RT_b \rangle_A$, last cell | $\langle c_{\mathrm{tot}}RT_b \rangle_A$, last cell |
| `css_outlet_T_wall` | wall temperature of the last axial cell | wall temperature of the last axial cell | wall temperature of the last axial cell |

A report writes the `y`, `q`, `T_bed`, `T_wall` and pressure figures (§21); the
`ppm` figure of O2 comes only from `plot_cycle_impurity_outlet()`.

Step figures in `<step>/profiles/outlet_profiles/` (`ProfilesPlotter`, only with
`execution.plot_profiles`) reduce last-cell values with the same step classes, but
never read the outlet face for the DAC layout:

| Mixture | Figure | Average |
|---|---|---|
| CO2, H2O, O2, N2 (N2 closure; CO2 and H2O adsorbed) | `<step>_outlet_species`, `<step>_outlet_species_O2_N2`, `<step>_outlet_temperature` | $\langle \phi \rangle_{\dot m}$ of last-cell values on the open-flow steps, $\langle \phi \rangle_A$ on the closed-inlet and closed-outlet steps (the 2D title names the average) |
| CO2, H2O, O2, N2 (N2 closure; CO2 and H2O adsorbed) | `<step>_outlet_pressure_velocity` | pressure: $\langle p \rangle_A$, last cell; velocity: $\langle u \rangle_A$ at the outlet face |
| any other mixture | `<step>_outlet_species` | native composition (`get_mixed_cups()`); while no stream leaves the column (a closed step), the mixed cup is undefined and the dotted trace is the gas in the last cell, $\langle y \rangle_A$ |
| any other mixture | `<step>_outlet_state` | temperature: the bed in the last cell, $\langle T \rangle_A$, and `T_out` from `get_mixed_cups()` where a stream leaves; pressure: $\langle p \rangle_A$, last cell; velocity: $\langle u \rangle_A$ at the outlet face |

---

## 5. Velocity reconstruction

New solutions persist `outlet_face`, containing per-ring species molar flows,
sensible enthalpy flow and dimensional face velocity. Cyclic velocity is the
area-weighted native face velocity; it is forced to zero for closed outlets.
Internal velocity profiles use the saved pressure-law coefficients; vector-drag
reconstruction includes transverse gradients. Scalar Ergun reconstruction is a
legacy fallback, not a substitute for the native vector outlet observations.

Condenser processing checks finite species and total flows before applying a mask.
A small initial inward sample is tolerated only at t=0 within the defined relative
threshold; later inward flow is rejected. Streamless samples remain on the time axis.

---

## 6. Species inventories

For species $i$, gas-phase mass in the bed is:

$$
m_i^{g}(t)
= MW_i \sum_{j,r} c_i(j,r,t) V_{jr}\epsilon_g.
$$

For every declared adsorbate, solid-phase adsorbed mass is:

$$
m_i^{s}(t)
= MW_i \rho_{\mathrm{sorbent}}
   \sum_{j,r} q_i(j,r,t) V_{jr}.
$$

For channeling, multiply each cell gas inventory by `ring_gas_holdup_factor` and
each sorbent inventory by `ring_sorbent_factor` before summing. Loading events and
post-processing use these same local weights. Their area means equal one.

Here $\epsilon_g=\varepsilon_{\mathrm{total}}$ and
$\rho_{\mathrm{sorbent}}$ are loaded from the resolved architecture context.
For a conventional packed bed they reduce to
$\varepsilon_b+(1-\varepsilon_b)\varepsilon_p$ and
$(1-\varepsilon_b)\rho_p$.

The total bed inventory for each declared gas species is:

$$
m_i^{\mathrm{bed}}(t) = m_i^g(t)
+ \mathbf{1}_{i\in\mathrm{adsorbates}}m_i^s(t).
$$

Non-adsorbates have no solid-phase contribution.

The state-based mass change over one step is:

$$
\Delta m_i^{\mathrm{state}}
= m_i^{\mathrm{bed}}(t_f) - m_i^{\mathrm{bed}}(0).
$$

For every adsorbate represented by a named `m_<species>` scalar, the solver
also integrates a mass balance:

$$
\Delta m_i^{\mathrm{int}}
= \int_0^{t_f}
   \left(\dot m_{i,\mathrm{in}} - \dot m_{i,\mathrm{out}}\right) dt.
$$

The sign convention is therefore:

- positive $\Delta m_i$: net accumulation in the bed
- negative $\Delta m_i$: net removal from the bed

For those species, the post-processor reports both integrated and state-based
changes:

$$
\mathrm{gap}_i
= \Delta m_i^{\mathrm{int}} - \Delta m_i^{\mathrm{state}},
$$

and:

$$
\mathrm{error}_i[\%]
= 100
  \frac{|\mathrm{gap}_i|}
       {|\Delta m_i^{\mathrm{state}}|}.
$$

If the denominator is effectively zero, the error is reported as zero only when
the gap is also effectively zero; otherwise it is `NaN`.

### 6.1 Reading the gap

The gap is a genuine consistency check between two independent estimates, but
its *relative* form is only meaningful when the net change is a decent fraction
of what passed through the column. On the steam steps it is not: in cycle 3 of
the CSS run of [Results](../guide/06_results.md), `s_des` feeds 3.6 kg of steam
for a net water change of 9.3 g, a ratio near 390, so a 0.03 % error on either
flux shows up as a 12 % "error" on the net. Read $\mathrm{gap}_i$ against the
throughput before reading $\mathrm{error}_i$.

### 6.2 O2 and N2 residual inventories

For the exact DAC layout (gas order CO2/H2O/O2/N2, N2 closure, adsorbates CO2/H2O), step
post-processing checks the O2 and N2 residual inventories. O2 may rise in `startup`, `ads` and
`press`; N2 may also rise in `cool`. On the other steps a rise measured from
$\max(m_i^{\mathrm{initial}}, 0)$ is warned only above

$$
B_i = 10^{-12}\,\mathrm{mol\,m^{-3}} \times M_i \times \sum_{\mathrm{cells}} V \varepsilon_g \times (1 + 10^{-9}),
$$

the mass of the reporting concentration floor over the gas volume; a final inventory below
$-B_i$ also warns. The check accounts for that floor and never changes the fields or the
returned inventory.

---

## 7. Outlet flow rates

Outlet molar flowrates are reconstructed from the outlet face velocity and the
outlet **face** concentration:

$$
\dot n_{i,\mathrm{out}}(t)
= \sum_r u_{L,r}(t)\,c_{i,L,r}^{\mathrm{face}}(t)\,A_r,
\qquad
c^{\mathrm{face}} = w_{-1}c_{[-1]} + w_{-2}c_{[-2]}.
$$

The weights come from the convective boundary operator itself rather than being
written out here, so a change of boundary condition or a non-uniform axial grid
follows automatically; on a uniform grid with the shipped zero-gradient outlet
they are $1.125$ and $-0.125$. Using the last cell value $c_{[-1]}$ instead would
bias every outlet flow by one cell of the axial gradient, and the bias would
propagate into the separator recovery and purity figures below. `column_model` and
`_get_flowrates()` use the same face value.

The corresponding mass flowrate is:

$$
\dot m_{i,\mathrm{out}}(t)
= MW_i \dot n_{i,\mathrm{out}}(t).
$$

The current `_get_flowrates()` implementation uses the signed outlet velocity.
For product/separator calculations the outlet stream of every collected step
(`outlet.collect_product`, §9) is used; a net inward stream after $t = 0$ is
refused.

---

## 8. Integrated energy states

The solver integrates three energy scalars together with the PDE state. The
stored step energy is:

$$
E_k(t_f) = \int_0^{t_f} P_k(t) dt,
\qquad
k \in \{\mathrm{comp},\mathrm{vac},\mathrm{jacket}\}.
$$

Post-processing converts joules to megajoules:

$$
E_k[\mathrm{MJ}] = \frac{E_k[\mathrm{J}]}{10^6}.
$$

### 8.1 Compression work

The compression power is modeled as polytropic compression:

$$
P_{\mathrm{comp}}
=
\frac{1}{\eta_{\mathrm{comp}}}
\frac{\gamma}{\gamma-1}
p_{\mathrm{in}}Q_{\mathrm{in}}
\left[
\left(\frac{p_{\mathrm{in}}}{p_{\mathrm{out}}}\right)^{(\gamma-1)/\gamma}
-1
\right].
$$

In the code:

$$
\eta_{\mathrm{comp}} = 0.7.
$$

Here $p_{\mathrm{in}}$ and $p_{\mathrm{out}}$ are area-averaged inlet/outlet
pressures and $Q_{\mathrm{in}}$ is the inlet volumetric flowrate:

$$
Q_{\mathrm{in}} = \sum_r u_{\mathrm{in},r} A_r.
$$

### 8.2 Vacuum pump work

The vacuum pump is modeled with the same polytropic structure, discharging to
ambient pressure:

$$
P_{\mathrm{vac}}
=
\frac{1}{\eta_{\mathrm{vac}}}
\frac{\gamma}{\gamma-1}
p_{\mathrm{out}}Q_{\mathrm{out}}
\left[
\left(\frac{p_{\mathrm{amb}}}{p_{\mathrm{out}}}\right)^{(\gamma-1)/\gamma}
-1
\right].
$$

The vacuum pump efficiency is pressure-dependent:

$$
\eta_{\mathrm{vac}}
=
\frac{
0.8 \, 19.55 \, (p_{\mathrm{out}}/10^5)
}{
1 + 19.55 \, (p_{\mathrm{out}}/10^5)
}.
$$

The outlet volumetric flowrate is:

$$
Q_{\mathrm{out}} = \sum_r u_{\mathrm{out},r} A_r.
$$

### 8.3 Jacket energy

The jacket duty is based on convective heat exchange between the jacket and the
wall:

$$
P_{\mathrm{jacket}}
= h_{\mathrm{ext}}
  \sum_j 2\pi R_{b,\mathrm{ext}}\,\Delta z_j
  \left(T_{\mathrm{jacket}} - T_{w,j}\right),
$$

with $R_{b,\mathrm{ext}} = R_b + t_b$ the outer wall radius and $\Delta z_j$ the
cells of the wall equation. The code integrates over the outer wall surface with
the wall equation's own finite-volume weights, so $E_{\mathrm{jacket}}$ is the
energy the wall balance exchanges with the jacket. The sign convention is:

- positive: heat delivered from jacket to wall/bed
- negative: heat removed from wall/bed by the jacket

---

## 9. Separator and product treatment

A step whose outlet is collected (`steps.<step>.outlet.collect_product`; by
default `s_purge`, `w_des` and `s_des`) applies a two-stage downstream treatment
to the outlet stream:

1. A condenser flash on the CO2/H2O outlet stream (when `outlet.condenser`).
2. A vacuum-discharge flash on the condenser vapor outlet, or on the outlet
   stream itself when the step has no condenser.

Without a condenser the vacuum pump moves the whole outlet stream, so the step
reports the solver's vacuum energy (§9.4 applies only behind a condenser), and a
sample drier than its dew point at the discharge leaves as vapor. Every species
other than CO2 and H2O passes both stages unchanged (O2 and N2 in the DAC
mixture). A mixture without H2O has nothing to condense: a collected step skips
the condenser and the flash and records its outlet as the product, and the
heating steps record no steam condensation temperature. A step whose
outlet is not collected records `mass.vented`: its bed outlet before any
separation, injected steam included; that steam still counts in the cycle's
steam use.

O2 and N2 are not included in the binary flash calculation. In the desorption
KPI table they are treated as non-condensing gases and their recovered mass is
approximated by the absolute bed mass loss over that collected step.

### 9.1 Condenser flash

The condenser feed is:

$$
\dot n_{\mathrm{feed}}
=
\dot n_{\mathrm{CO2,out}}
+
\dot n_{\mathrm{H2O,out}},
$$

with feed composition:

$$
z_{\mathrm{CO2}}
=
\frac{\dot n_{\mathrm{CO2,out}}}
     {\dot n_{\mathrm{feed}}+\epsilon}.
$$

The condenser pressure is the area-weighted outlet pressure:

$$
p_{\mathrm{cond}}(t)
=
\frac{\sum_r p_{L,r}(t)A_r}{\sum_r A_r}.
$$

The binary flash separator returns vapor flow $n_v$ and vapor composition
$y_i$. The reported mass after condenser is:

$$
m_i^{\mathrm{cond,vap}}
= MW_i \int_0^{t_f} n_v(t)y_i(t)dt,
\qquad
i \in \{\mathrm{CO2},\mathrm{H2O}\}.
$$

The flash uses `T_cond = steps.<step>.outlet.condenser_temperature_K`, by
default 278.15 K (5 °C). **The condenser temperature sets the product**: the
flash split here, the condenser duty (§9.3) and the post-condenser vacuum-pump
update (§9.4) all use it, and the step records it as `temperature.condenser`.
The chiller of §17 is priced at that recorded temperature, step by step; a result
without the record (written before the setting existed) keeps
`EquivalentWorkConditions.T_condensation` (278.15 K), the value it was computed with.

A **second** temperature enters the chain, and only the pricing: the
cooling-water limit `T_COOLING_WATER_LIMIT = 303.15 K` of §9.3.1, which decides
how much of the duty ambient carries for free. It changes no mass and no duty —
it splits one duty into two prices.

### 9.2 Vacuum discharge flash

With a condenser, the vacuum-discharge feed is its vapor outlet; without a condenser it is the
untreated bed outlet. A collected mixture without H2O skips both flashes and records the bed
outlet directly as product. When a vacuum-discharge flash is performed, it is evaluated at:

$$
T = T_{\mathrm{amb}},
\qquad
p = p_{\mathrm{amb}}.
$$

The reported product mass after vacuum discharge is:

$$
m_i^{\mathrm{vac,vap}}
= MW_i \int_0^{t_f} n_{v,\mathrm{vac}}(t)y_{i,\mathrm{vac}}(t)dt.
$$

This is the default recovered mass used by the cycle-level KPI analyzer, which
sums it over every product step (a step that recorded it). A product step
without it falls back to its own absolute state mass change, and a result
written before `outlet.collect_product` existed counts the desorption steps
`w_des`, `s_purge` and `s_des` (§13).

### 9.3 Condenser duty

The condenser duty is computed from an enthalpy balance. The gas sensible term
is:

$$
Q_{\mathrm{sens}}
=
\dot n_{\mathrm{feed}}
\left(\sum_i z_i C_{p,i}^{g}\right)
\left(T_{\mathrm{feed}}-T_{\mathrm{cond}}\right).
$$

CO2 dissolution heat is:

$$
Q_{\mathrm{diss}}
=
n_l x_{\mathrm{CO2}}\Delta h_{\mathrm{sol,CO2}},
$$

with:

$$
\Delta h_{\mathrm{sol,CO2}}
=
1000
\left(
106.56
- \frac{6.2634\times 10^4}{T_{\mathrm{cond}}}
+ \frac{7.475\times 10^6}{T_{\mathrm{cond}}^2}
\right)
\quad [\mathrm{J\,mol^{-1}}].
$$

Water boiling temperature and latent heat use the Ward-style correlations:

$$
T_{\mathrm{bp}}
=
197
\left(\frac{p_{\mathrm{cond}}}{10^5}\right)^{0.1367}
-97.97
+273.15,
$$

and:

$$
\Delta h_{\mathrm{vap}}
=
\left[
-446.2
\left(\frac{p_{\mathrm{cond}}}{10^5}\right)^{0.1508}
+2707
\right] 10^3 MW_{\mathrm{H2O}}.
$$

The water condensation contribution is:

$$
Q_{\mathrm{cond,H2O}}
=
n_l x_{\mathrm{H2O}}
\left[
C_{p,\mathrm{H2O}}^g(T_{\mathrm{bp}}-T_{\mathrm{cond}})
-\Delta h_{\mathrm{vap}}
+C_{p,\mathrm{H2O}}^l(T_{\mathrm{cond}}-T_{\mathrm{bp}})
\right].
$$

The instantaneous condenser heat balance is:

$$
Q_{\mathrm{cond}}
=
Q_{\mathrm{sens}}
-Q_{\mathrm{diss}}
-Q_{\mathrm{cond,H2O}}.
$$

The integrated condenser energy is:

$$
E_{\mathrm{cond}}
=
\frac{1}{10^6}
\int_0^{t_f} Q_{\mathrm{cond}}(t)dt.
$$

$E_{\mathrm{cond}}$ is **positive when the condenser removes heat**: every term
of $Q_{\mathrm{cond}}$ is written as heat that has to be taken out of the
stream. Cooling the gas from $T_{\mathrm{feed}}$ to $T_{\mathrm{cond}}$ gives
$Q_{\mathrm{sens}}>0$; CO2 dissolution is exothermic
($\Delta h_{\mathrm{sol,CO2}}\approx-22\ \mathrm{kJ\,mol^{-1}}$ at 278.15 K), so
$-Q_{\mathrm{diss}}>0$; and condensing water releases $\Delta h_{\mathrm{vap}}$,
which makes $Q_{\mathrm{cond,H2O}}<0$ and hence $-Q_{\mathrm{cond,H2O}}>0$.

The KPI named `Condenser Duty [MJ]` stores `-E_cond`, so **the reported value is
negative**. That is deliberate: it places the condenser on the same sign axis as
every other energy KPI in this layer.

| Sign | Meaning | Examples |
|---|---|---|
| positive | energy supplied to the process | `Steam Energy` (§10), jacket heating (§8.3), compression, vacuum |
| negative | energy removed from the process | `Condenser Duty`, jacket cooling during `cool` |

Physically the condenser undoes most of what the steam generator did, so for a
steam-assisted step the two KPIs should come out close in magnitude and opposite
in sign. In cycle 3 of the CSS run of [Results](../guide/06_results.md),
`cycle_3/s_des/kpis.csv` holds, among other rows:

```text
steam assisted desorption,system,Condenser Duty [MJ],-9.534295942524672
steam assisted desorption,system,Condenser Duty Rejected To Ambient [MJ],-9.151572033017317
steam assisted desorption,system,Steam Energy [MJ],9.284432940102237
```

The ~0.25 MJ that does not cancel is what the condenser must reject *beyond* the
steam enthalpy: sensible heat carried out of the hot bed by the desorbed stream,
plus the exothermic CO2 dissolution. This near-cancellation is a cheap sanity
check on any new `s_des` run — if the two drift apart, suspect the steam flow
or the outlet reconstruction rather than the sign.

§17 is insensitive to the sign: the equivalent-work conversion uses
$|E_{\mathrm{cond}}|$, netted as in §9.3.1 and divided by a COP.

Condenser duty and the post-condenser vacuum-pump update use the same
$T_{\mathrm{cond}}$ as the flash in §9.1
(`steps.<step>.outlet.condenser_temperature_K`, default 278.15 K).

### 9.3.1 Where the duty is rejected

$E_{\mathrm{cond}}$ is **one number for a duty released over a temperature
range**, and that range straddles ambient. The stream leaves the bed at
88–92 °C, de-superheats, and condenses along a dew line that starts near 60 °C
at 20 kPa; with the product CO2 as the only non-condensable (~1 mol %), most of
the water is gone before the stream reaches ambient at all. Charging the whole
duty at the product temperature therefore prices heat that cooling water takes
for nothing — the same credit §17 already gives negative jacket duty.

The step post-processor measures the split rather than assuming a fraction. The
enthalpy path of §9.3 is a state-function difference between the inlet and an
end temperature, so evaluating it twice is enough:

$$
E_{\mathrm{cond}}^{\mathrm{amb}} = E_{\mathrm{cond}}(T_{\mathrm{cw}}),
\qquad
E_{\mathrm{cond}}^{\mathrm{chiller}}
= E_{\mathrm{cond}}(T_{\mathrm{cond}}) - E_{\mathrm{cond}}(T_{\mathrm{cw}}),
$$

with $T_{\mathrm{cw}} = $ `T_COOLING_WATER_LIMIT` $ = 303.15$ K, the coldest
stream temperature cooling water can reach, equal to
`EquivalentWorkConditions.T_low + cooling_water_approach`. Only the second term
is charged (§17.1).

`compute_condenser_duty_above_ambient()` computes the first term.
`compute_condenser_duty()` reads the flash cached by
`run_condensation_separator()` at the step's condenser temperature. A condenser
set above the cooling-water limit is cooled by water alone: the split is
evaluated at $\max(T_{\mathrm{cw}}, T_{\mathrm{cond}})$, so nothing is left for a
chiller. Both share
`_condenser_duty_from_flash()`, which is the single home for the enthalpy
constants of §9.3, so the two duties cannot drift onto different models.

The second evaluation needs its own flash, and it is gated. **`FlashSeparator.solve`
assumes two phases exist**: it reads the equilibrium $y$ and $x$ off $(T,p)$
alone and splits the feed with the lever rule, without checking the dew point.
Above the dew point the denominator changes sign, $y_{\mathrm{CO2}}$ goes
negative, and the lever rule returns *more liquid than there was feed* — with no
exception raised. Every product-line CO2/H2O flash therefore first tests
$(1-z_{\mathrm{CO2}})p \le p_{\mathrm{sat,H2O}}(T)$ (`_binary_flash()`): a sample that
satisfies it leaves entirely as unchanged vapor ($n_v = n_{\mathrm{feed}}$, $n_l = 0$);
otherwise the separator is called and the returned vapor flow is held within
$0 \le n_v \le n_{\mathrm{feed}}$. The condenser and vacuum-discharge splits use this guard,
and `_condenser_flash_at()` applies the same dew-point test when it evaluates the
cooling-water credit. The configured condenser temperature, like the cooling-water limit, can
place a flash above its dew point.

The split is stored as its own step KPI, `Condenser Duty Rejected To Ambient
[MJ]` (`energy.condenser.ambient`), on the same negative sign axis as the total.
It is a **credit, not a sixth energy term**: it never becomes a column of the
energy breakdown, and `Condenser Duty [MJ]` still carries the whole duty,
because the duty was never the thing that was wrong.

In the `s_des` step of §9.3, 96 % of the duty is rejected above ambient
(−9.15 of −9.53 MJ).

### 9.4 Vacuum pump update after condensation

After the condenser, the vacuum pump energy is recomputed using the condenser
vapor flow plus every species other than CO2 and H2O (O2 and N2 in the DAC
mixture):

$$
\dot n_{\mathrm{vac,in}}
= n_{v,\mathrm{cond}} + \dot n_{\mathrm{O2,out}} + \dot n_{\mathrm{N2,out}}.
$$

The ideal-gas volumetric flow at condenser temperature is:

$$
Q_{\mathrm{cond,out}}
=
\frac{\dot n_{\mathrm{vac,in}}RT_{\mathrm{cond}}}{p_{\mathrm{out}}}.
$$

The same polytropic vacuum model from Section 8.2 is then integrated over the
step. Desorption KPI tables use this updated value rather than the raw in-bed
vacuum energy scalar.

---

## 10. Steam-assisted desorption

For `s_purge` and `s_des`, the inlet flow can be defined by a steam mass
flowrate; both record the steam mass and energy below, and the cycle steam use
sums every step:

$$
\dot m_{\mathrm{steam}}.
$$

The reported steam mass is:

$$
m_{\mathrm{steam}}
=
\dot m_{\mathrm{steam}} t_f.
$$

The steam energy model uses:

$$
T_{\mathrm{vap}}
=
197
\left(\frac{p_{\mathrm{feed}}}{10^5}\right)^{0.1367}
-97.97
+273.15.
$$

The latent heat correlation is:

$$
\Delta h_{\mathrm{vap}}
=
-446.2
\left(\frac{p_{\mathrm{feed}}}{10^5}\right)^{0.1508}
+2707
\quad [\mathrm{kJ\,kg^{-1}}].
$$

The liquid sensible contribution is:

$$
\Delta h_1
=
C_{p,\mathrm{water}}
\left(T_{\mathrm{ref}}-T_{\mathrm{vap}}\right).
$$

The vapor sensible contribution is represented by the polynomial integral:

$$
\Delta h_2
=
\frac{a}{3}
\left(T_{\mathrm{vap}}^3-T_{\mathrm{feed}}^3\right)
+
\frac{b}{2}
\left(T_{\mathrm{vap}}^2-T_{\mathrm{feed}}^2\right)
+
c
\left(T_{\mathrm{vap}}-T_{\mathrm{feed}}\right),
$$

with:

$$
a = 1.012\times10^{-6},
\qquad
b = -3.594\times10^{-4},
\qquad
c = 1.882.
$$

The reported steam energy is:

$$
E_{\mathrm{steam}}
=
-
\frac{
m_{\mathrm{steam}}
\left(\Delta h_1-\Delta h_{\mathrm{vap}}+\Delta h_2\right)
}{1000}
\quad [\mathrm{MJ}].
$$

This convention makes steam supply appear as a positive heat input for the
current operating conditions, which is the same sign axis the condenser duty
uses with the opposite sign — see the sign table in §9.3, and the
near-cancellation of the two for a steam-assisted step.

---

## 11. Step-level KPI catalog

Each step first reports common grid and mass KPIs, then step-specific energy,
thermal, pressure-drop, and product KPIs.

Step folders on disk use the code name. The `Step` column of `kpis.csv` carries
the record label of `StepPostProcessor.compute_step_metrics()`, and the cycle
table of §12 a shorter label (`STEP_KEY_TO_CYCLE_STEP`); neither is a folder name:

| Code (folder) | Prose name | `kpis.csv` `Step` | Cycle table `Step` |
|---|---|---|---|
| `startup` | startup | `startup` | `startup` |
| `ads` | adsorption | `adsorption` | `adsorption` |
| `blow` | blowdown | `blowdown` | `blowdown` |
| `heat` | heating | `preheating` | `preheating` |
| `w_purge` | wall purge | `wall purge` | `wall purge` |
| `s_purge` | steam purge | `steam purge` | `steam purge` |
| `w_des` | wall-heated desorption | `wall heated desorption` | `desorption` |
| `s_des` | steam-assisted desorption | `steam assisted desorption` | `desorption` |
| `cool` | cooling | `cooldown` | `cooldown` |
| `press` | pressurization | `pressurization` | `pressurization` |

### 11.1 Common KPIs

For every post-processed step:

| KPI | Species | Formula or source |
|---|---|---|
| `Nz [-]` | system | saved axial grid size |
| `Nr [-]` | system | saved radial grid size |
| `initial mass [kg]` | every declared gas species | $m_i^{\mathrm{bed}}(0)$ |
| `mass change [kg]` | species with `m_<species>` | integrated scalar $\Delta m_i^{\mathrm{int}}$ |
| `mass change [kg]` | species without that scalar | state change $\Delta m_i^{\mathrm{state}}$ |
| `left-over mass [kg]` | every declared gas species | integrated result when available, otherwise final state inventory |
| `state-based mass change [kg]` | species with `m_<species>` | $\Delta m_i^{\mathrm{state}}$ |
| `integrated-state mass gap [kg]` | species with `m_<species>` | $\Delta m_i^{\mathrm{int}}-\Delta m_i^{\mathrm{state}}$ |
| `integrated-state error [%]` | species with `m_<species>` | $100|\mathrm{gap}|/|\Delta m_i^{\mathrm{state}}|$ |
| `state-based left-over mass [kg]` | species with `m_<species>` | $m_i^{\mathrm{bed}}(t_f)$ |
| `final gas-phase mass [kg]` | every declared gas species | $MW_i\sum c_i^g(t_f)V_{jr}\epsilon_{tot}$ |
| `Saturation [%]` | CO2 adsorbate only | guarded DAC KPI relative to $q_{\mathrm{CO2,ref}}$ |
| `Step duration [s]` | system | final time $t_f$ |

The CO2 saturation KPI is:

$$
S_{\mathrm{CO2}}
=
100
\frac{
\sum_{j,r} q_{\mathrm{CO2}}(j,r,t_f)\,\rho_{\mathrm{sorbent},jr}V_{jr}
}{
q_{\mathrm{CO2,ref}}
\sum_{j,r}\rho_{\mathrm{sorbent},jr}V_{jr}
},
$$

weighted by the local sorbent density of the saved context, which for a uniform
packed bed is $\rho_p(1-\epsilon_b)$.

### 11.2 Startup (`startup`) and adsorption (`ads`)

Additional KPIs:

| KPI | Formula or source |
|---|---|
| `Average pressure drop [mbar]` | time-average of inlet minus outlet pressure |
| `Compression Energy [MJ]` | integrated compression scalar |
| `Jacket Energy [MJ]` | integrated jacket scalar |

The pressure-drop KPI is:

$$
\Delta p_{\mathrm{avg}}[\mathrm{mbar}]
=
\frac{1}{100t_f}
\int_0^{t_f}
\left(
\langle p_{\mathrm{in}}\rangle_A
-
\langle p_{\mathrm{out}}\rangle_A
\right)dt.
$$

### 11.3 Blowdown (`blow`)

Additional KPIs:

| KPI | Species | Formula or source |
|---|---|---|
| `Vacuum Pump Energy [MJ]` | system | post-condenser updated vacuum work when the outlet is collected through an active condenser; otherwise the integrated vacuum scalar |
| `Jacket Energy [MJ]` | system | integrated jacket scalar |
| `vented mass [kg]` | every declared gas species | bed outlet before separation, signed, positive when the species leaves (§9); only when the outlet is not collected |

The outlet of `blow` is not collected by default. With
`outlet.collect_product` true the product rows of §11.5 replace `vented mass`;
this holds for every step that has an outlet (§11.3–§11.6).

### 11.4 Heating (`heat`)

Additional KPIs:

| KPI | Species | Formula or source |
|---|---|---|
| `T_bed_min [K]` | system | $\min_{j,r} T_b(j,r,t_f)$ |
| `T_steam_condensation [K]` | system | steam condensation temperature from the model; only when the mixture has H2O |
| `Vacuum Pump Energy [MJ]` | system | post-condenser updated vacuum work when the outlet is collected through an active condenser; otherwise the integrated vacuum scalar |
| `Jacket Energy [MJ]` | system | integrated jacket scalar |
| `vented mass [kg]` | every declared gas species | bed outlet before separation, signed, positive when the species leaves (§9); only when the outlet is not collected |

The temperature pair is used as a physical check for steam-assisted operation:
the bed should be warm enough to avoid steam condensation inside the column.

### 11.5 Wall-heated desorption (`w_des`)

Additional KPIs. The product rows need `outlet.collect_product` (default true on
`s_purge`, `w_des` and `s_des`), the condenser rows also `outlet.condenser`; a
step whose outlet is not collected records `vented mass [kg]` (§11.3) instead of
the product rows. Without H2O in the mixture there is no condenser and no flash
(§9).

| KPI | Species | Formula or source |
|---|---|---|
| `mass after condenser [kg]` | CO2, H2O | condenser vapor mass; with a condenser |
| `mass after condenser [kg]` | every other species (O2, N2 in the DAC mixture) | absolute bed mass loss, treated as non-condensing gas; with a condenser |
| `mass after vacuum discharge [kg]` | CO2, H2O | vacuum-discharge vapor mass |
| `mass after vacuum discharge [kg]` | every other species | absolute bed mass loss |
| `Jacket Energy [MJ]` | system | integrated jacket scalar |
| `Vacuum Pump Energy [MJ]` | system | post-condenser updated vacuum work with a condenser, else the integrated vacuum scalar |
| `Condenser Duty [MJ]` | system | reported as $-E_{\mathrm{cond}}$; with a condenser |
| `Condenser Duty Rejected To Ambient [MJ]` | system | the free share of the same duty, §9.3.1; with a condenser |
| `T_condenser [K]` | system | condenser temperature of the step (§9.1); with a condenser |

Wall purge (`w_purge`) instead follows the heating KPI branch of §11.4: it
records the rows of §11.4, `vented mass` included, or the product rows of this
section when its outlet is collected; it has no steam terms.

### 11.6 Steam purge (`s_purge`) and steam-assisted desorption (`s_des`)

`s_purge` and `s_des` use the same step-level KPI branch because both are
pure-steam, fixed-feed, low-pressure operations. They remain distinct cycle
steps (`steam purge` and `desorption` in the cycle table). Both are product steps
by default; recovery sums every product step, and the working capacity is read at
the end of the last desorption step (§16).

Additional KPIs, with the same conditions as §11.5:

| KPI | Species | Formula or source |
|---|---|---|
| `mass after condenser [kg]` | CO2, H2O | condenser vapor mass; with a condenser |
| `mass after condenser [kg]` | every other species | absolute bed mass loss; with a condenser |
| `mass after vacuum discharge [kg]` | CO2, H2O | vacuum-discharge vapor mass |
| `mass after vacuum discharge [kg]` | every other species | absolute bed mass loss |
| `Average pressure drop [mbar]` | system | time-average inlet minus outlet pressure |
| `Compression Energy [MJ]` | system | integrated compression scalar |
| `Vacuum Pump Energy [MJ]` | system | post-condenser updated vacuum work with a condenser, else the integrated vacuum scalar |
| `Jacket Energy [MJ]` | system | integrated jacket scalar |
| `Condenser Duty [MJ]` | system | reported as $-E_{\mathrm{cond}}$; with a condenser |
| `Condenser Duty Rejected To Ambient [MJ]` | system | the free share of the same duty, §9.3.1; with a condenser |
| `T_condenser [K]` | system | condenser temperature of the step (§9.1); with a condenser |
| `Steam Mass [kg]` | H2O | $\dot m_{\mathrm{steam}}t_f$ |
| `Steam Energy [MJ]` | system | steam enthalpy model |

### 11.7 Cooling (`cool`)

Additional KPIs:

| KPI | Formula or source |
|---|---|
| `Vacuum Pump Energy [MJ]` | integrated vacuum scalar |
| `Jacket Energy [MJ]` | integrated jacket scalar |

### 11.8 Pressurization (`press`)

Additional KPIs:

| KPI | Formula or source |
|---|---|
| `Average pressure drop [mbar]` | time-average inlet minus outlet pressure |
| `Compression Energy [MJ]` | integrated compression scalar |
| `Jacket Energy [MJ]` | integrated jacket scalar |

---

## 12. Cycle-level KPI analysis

`CycleDataLoader` (public as `pydac.CycleDataLoader`) loads each ordered
`kpis.pkl` through `StepMetricsStore`, combines the typed step records into
`CycleMetrics`, and loads the static `SimulationContext`. The resulting cycle
table has the canonical columns:

```text
Cycle, Step Index, Step Key, Step, Species, Metric ID, Value, Unit, Basis
```

`Step Key` is the code name of the step and `Step` its cycle-table label (§11).

The canonical step order is inferred from `cycle_metadata.json` when available;
otherwise it is inferred from the folders present in the result directory.

The desorption folder can be either:

- `w_des` for wall-heated desorption
- `s_des` for steam-assisted desorption

Both are canonicalized to the analysis step name `desorption`.
The separate `s_purge` folder is canonicalized to `steam purge`; like `w_des`
and `s_des` it is a product step when its outlet is collected (default), so its
product enters recovery and purity. A product step is one that recorded
`mass after vacuum discharge`.
The optional `w_purge` folder is canonicalized to `wall purge` and likewise
contributes its heating-style energy without becoming a desorption step.

---

## 13. Preferred mass basis

For cycle analysis there are three possible mass-change bases:

| Basis | Meaning |
|---|---|
| `integrated` | use the integrated `mass change [kg]` KPI |
| `state` | use final-minus-initial bed inventories |
| `preferred` | alias resolved unconditionally to `state` |

`signed_mass_change()` rewrites `basis="preferred"` to `basis="state"` before
querying the cycle table; there is no fallback to the integrated basis. The
preferred signed mass change is therefore:

$$
\Delta m_i^{\mathrm{pref}}
=
\Delta m_i^{\mathrm{state}}.
$$

The absolute mass change used in many summary metrics is:

$$
|\Delta m_i| = |\Delta m_i^{\mathrm{basis}}|.
$$

Recovered mass is:

$$
m_i^{\mathrm{rec}}
=
\sum_{s\,\in\,\mathrm{product\ steps}}
\begin{cases}
|m_{i,s}^{\mathrm{vac,vap}}|, & \text{if step } s \text{ recorded `mass after vacuum discharge`},\\
|\Delta m_{i,s}^{\mathrm{state}}|, & \text{otherwise}.
\end{cases}
$$

---

## 14. Bed closure and desorption closure

### 14.1 Step closure

For CO2 and H2O, per-step closure compares the integrated scalar to the
state-based inventory change:

$$
\mathrm{gap}_{i,s}
=
\Delta m_{i,s}^{\mathrm{int}}
-
\Delta m_{i,s}^{\mathrm{state}}.
$$

The relative error reported in analysis tables is:

$$
\mathrm{relerr}_{i,s}[\%]
=
\begin{cases}
0, & M_{i,s} < 10^{-9},\\
100\,\mathrm{gap}_{i,s}/M_{i,s}, & \text{otherwise},
\end{cases}
\qquad
M_{i,s} = \max(|\Delta m_{i,s}^{\mathrm{int}}|, |\Delta m_{i,s}^{\mathrm{state}}|).
$$

### 14.2 Cycle closure

Over a full executed cycle:

$$
\Delta m_i^{\mathrm{int,cycle}}
=
\sum_s \Delta m_{i,s}^{\mathrm{int}},
$$

and:

$$
\Delta m_i^{\mathrm{state,cycle}}
=
m_i^{\mathrm{bed}}(t_{\mathrm{last}})
-
m_i^{\mathrm{bed}}(t_{\mathrm{first}}).
$$

The cycle closure gap is:

$$
\mathrm{gap}_i^{\mathrm{cycle}}
=
\Delta m_i^{\mathrm{int,cycle}}
-
\Delta m_i^{\mathrm{state,cycle}}.
$$

### 14.3 Desorption process closure

For each product step, the analyzer reconstructs the downstream process balance
(one closure block per product step):

$$
m_{\mathrm{out,bed}}
=
m_{\mathrm{external}}
-
\Delta m_i^{\mathrm{bed,int}}.
$$

The external mass is the steam that step recorded (`s_purge` or `s_des`):

$$
m_{\mathrm{external}}
=
\begin{cases}
m_{\mathrm{steam}}, & i=\mathrm{H2O},\\
0, & \text{otherwise}.
\end{cases}
$$

For a step without steam (`w_des`), $m_{\mathrm{external}}=0$ for all species.

The condenser liquid removal is:

$$
m_{\mathrm{cond,liq}}
=
m_{\mathrm{out,bed}}
-
m_{\mathrm{cond,vap}}.
$$

The vacuum liquid removal is:

$$
m_{\mathrm{vac,liq}}
=
m_{\mathrm{cond,vap}}
-
m_{\mathrm{vac,vap}}.
$$

If `mass.after.condenser` is absent, `desorption_process_check()` uses the bed outlet as the
condenser vapor, so the condenser liquid removal is zero. If `mass.after.vacuum` is absent, it
uses that condenser vapor as the vacuum vapor, so the vacuum liquid removal is zero. The same
closure therefore covers bypassed stages and mixtures with nothing to condense.

The desorption closure gap is:

$$
\mathrm{gap}_i^{\mathrm{des}}
=
m_i^{\mathrm{bed}}(0)
+
m_{\mathrm{external}}
-
m_{\mathrm{cond,liq}}
-
m_{\mathrm{vac,liq}}
-
m_{\mathrm{vac,vap}}
-
m_i^{\mathrm{bed}}(t_f).
$$

---

## 15. Product metrics

### 15.1 Recovery

For species $i$:

$$
\mathrm{Recovery}_i
=
\frac{m_i^{\mathrm{rec}}}
     {|\Delta m_{i,\mathrm{adsorption}}|}.
$$

For CO2 this is interpreted as the fraction of adsorbed CO2 recovered after the
desorption and downstream separation train. $m_i^{\mathrm{rec}}$ is summed over
every product step (§13), and the recovery is 0 when adsorption took up at most
$10^{-12}$ kg.

### 15.2 Purity

The sums run over every declared species (the DAC four first). Recovered masses
are converted to moles:

$$
n_i^{\mathrm{rec}}
=
\frac{m_i^{\mathrm{rec}}}{MW_i}.
$$

Wet CO2 purity is:

$$
x_{\mathrm{CO2}}^{\mathrm{wet}}
=
\frac{n_{\mathrm{CO2}}^{\mathrm{rec}}}
     {\sum_i n_i^{\mathrm{rec}}}.
$$

Dry CO2 purity is:

$$
x_{\mathrm{CO2}}^{\mathrm{dry}}
=
\frac{n_{\mathrm{CO2}}^{\mathrm{rec}}}
     {\sum_i n_i^{\mathrm{rec}} - n_{\mathrm{H2O}}^{\mathrm{rec}}}.
$$

Residual O2 and N2 in the product are:

$$
\mathrm{ppm}_i
=
10^6
\frac{n_i^{\mathrm{rec}}}
     {\sum_j n_j^{\mathrm{rec}}},
\qquad i \in \{\mathrm{O2},\mathrm{N2}\},
$$

and are NaN when the mixture lacks the species.

Purity requires CO2. In the dry-purity denominator an absent H2O contributes zero, so wet and dry
purity agree without H2O. A nonpositive total or dry denominator gives purity zero (and ppm zero
for a present impurity with a nonpositive total).

### 15.3 H2O/CO2 co-desorption ratio

Mass basis:

$$
R_{\mathrm{H2O/CO2}}^{m}
=
\frac{m_{\mathrm{H2O}}^{\mathrm{rec}}}
     {m_{\mathrm{CO2}}^{\mathrm{rec}}}.
$$

Mole basis:

$$
R_{\mathrm{H2O/CO2}}^{n}
=
\frac{n_{\mathrm{H2O}}^{\mathrm{rec}}}
     {n_{\mathrm{CO2}}^{\mathrm{rec}}}.
$$

### 15.4 Residual bed gas after each step

Residual gas-phase content is computed from final gas inventories:

$$
\mathrm{ppm}_{i,s}^{\mathrm{bed}}
=
10^6
\frac{n_{i,s}^{g}(t_f)}
     {\sum_j n_{j,s}^{g}(t_f)}.
$$

The inventories are read from the per-step `mass.gas.final` records. The closure
species (N2 in the DAC mixture) is reconstructed as $c_{tot}$ minus the other
species and clipped only at zero;
trace species are not raised to a numerical concentration floor. Cycle analysis
therefore does not reopen `sol.pkl` for this diagnostic.

The default diagnostic tracks O2 and N2, with an O2 reference line at
40 ppm (molar) in the plotting layer.

---

## 16. Productivity and working capacity

Cycle time is the sum of executed step durations:

$$
t_{\mathrm{cycle}} = \sum_s t_s.
$$

CO2 productivity is:

$$
\mathrm{Productivity}
=
\frac{
m_{\mathrm{CO2}}^{\mathrm{rec}}
}{
V_{\mathrm{bed}} t_{\mathrm{cycle}}
}
86400
\quad
[\mathrm{kg_{CO2}\,m^{-3}\,day^{-1}}].
$$

The bed solid mass is the adsorbent mass of the saved context:

$$
m_s
=
V_{\mathrm{bed}}\rho_{\mathrm{sorbent}},
$$

which for a packed bed is $V_{\mathrm{bed}}(1-\epsilon_b)\rho_p$.

Cyclic working capacity for species $i$ is:

$$
\Delta q_i
=
\frac{
\left(m_i^{\mathrm{bed,ads,end}} - m_i^{\mathrm{bed,des,end}}\right)/MW_i
}{
m_s
}
\quad
[\mathrm{mol\,kg^{-1}}].
$$

For CO2, this is the working swing between the end of adsorption and the end of
the last desorption step (`w_des`, `s_purge` or `s_des`, whether its outlet is
collected or not).

---

## 17. Energy breakdown and equivalent work

The cycle-level energy table is built from these per-step KPI names:

| KPI name | Display column |
|---|---|
| `Compression Energy [MJ]` | `Compression` |
| `Vacuum Pump Energy [MJ]` | `Vacuum` |
| `Jacket Energy [MJ]` | `Jacket` |
| `Steam Energy [MJ]` | `Steam` |
| `Condenser Duty [MJ]` | `Condenser` |

Specific energy is normalized by recovered CO2:

$$
e_k
=
\frac{E_k}{m_{\mathrm{CO2}}^{\mathrm{rec}}}
\quad
[\mathrm{MJ\,kg_{CO2}^{-1}}].
$$

If recovered CO2 is at most $10^{-12}$ kg, specific energies and the equivalent
work return the penalty value $10^6$.

### 17.1 Thermal-to-work conversion

For equivalent work, heat duties are converted with a simple thermodynamic
availability model. The key distinction is between:

- heat supplied to the process;
- active refrigeration below ambient;
- passive heat rejection to ambient cooling water.

The heat-engine factor is:

$$
\eta_{\mathrm{Carnot,eff}}
=
\eta_{\mathrm{turb}}
\max\left(0,1-\frac{T_L}{T_H}\right).
$$

The condenser is charged on a **two-stage** scheme: cooling water carries the
duty down to its approach limit at no work cost, and one chiller carries the
rest. The free share is measured per step in §9.3.1, so only

$$
E_{\mathrm{cond}}^{\mathrm{chiller}}
=
E_{\mathrm{cond}} - E_{\mathrm{cond}}^{\mathrm{amb}}
$$

reaches the conversion. The chiller has to evaporate an approach *colder* than
the product it delivers, and that is the temperature which sets its COP:

$$
T_{\mathrm{evap}} = T_{\mathrm{cond}} - \Delta T_{\mathrm{chiller}},
\qquad
\mathrm{COP}_{\mathrm{chiller}}
=
\eta_{\mathrm{refr}}
\frac{T_{\mathrm{evap}}}{T_L-T_{\mathrm{evap}}},
\qquad T_{\mathrm{evap}}<T_L.
$$

If $T_{\mathrm{evap}}\ge T_L$, or $\eta_{\mathrm{refr}}\le0$, the code sets
$\mathrm{COP}_{\mathrm{chiller}}=1$. `CycleKPIs.chiller_cop()` returns it.

`thermodynamic_conversion_factors()` still returns the older COP evaluated at
$T_{\mathrm{cond}}$ itself, and it is still what a **legacy** result is charged
at — see the fallback below. The two differ by the chiller approach: 5.463
against 6.954 at the defaults.

**Legacy results.** A result recorded before §9.3.1 existed has no
`Condenser Duty Rejected To Ambient [MJ]`, and its duty cannot be split after
the fact without the saved solution. By default `CycleKPIs` refuses such a
result: `condenser_duty_above_ambient()` raises `LedgerMetricMissing`. Read with
`pydac.CycleKPIs(cycle_data, allow_legacy_ledger=True)`, it returns `None` instead
— distinct from `0.0`, which is a step whose whole duty genuinely needs a
chiller — and the conversion reproduces the **old** charge exactly,
$|E_{\mathrm{cond}}|/\mathrm{COP}$ at $T_{\mathrm{cond}}$. Charging the full duty at the new, lower COP would make an
un-backfilled result cost *more* than it used to, silently.

For negative jacket duty during the cooling step (`cool`), the current
implementation assumes ambient cooling-water heat rejection. Since this is heat
flowing in its natural direction from a hot bed to a colder sink, the ideal
thermodynamic work is taken as zero.

Conversion rules:

| Energy term | Equivalent-work conversion |
|---|---|
| Compression | unchanged |
| Vacuum | unchanged |
| Condenser cooling, split recorded | $|E_{\mathrm{cond}}-E_{\mathrm{cond}}^{\mathrm{amb}}|/\mathrm{COP}_{\mathrm{chiller}}$ |
| Condenser cooling, legacy result | $|E_{\mathrm{cond}}|/\mathrm{COP}$ |
| Condenser duty rejected above ambient | $0$ |
| Positive jacket or steam heat | $|E|\eta_{\mathrm{Carnot,eff}}$ |
| Negative jacket heat | $0$ |
| Negative steam energy | $|E|/\mathrm{COP}$ (the COP at $T_{\mathrm{cond}}$ above) |

The total equivalent work is:

$$
W_{\mathrm{eq}}
=
\frac{
\sum_{s,k} |E_{s,k}^{\mathrm{eq}}|
}{
m_{\mathrm{CO2}}^{\mathrm{rec}}
}
\quad
[\mathrm{MJ\,kg_{CO2}^{-1}}].
$$

In words: the condenser contributes through a refrigeration penalty on the part
of its duty that is actually below ambient, while heat rejected to ambient water
— by the condenser above $T_{\mathrm{cw}}$, or by the bed during cooling — does
not contribute to ideal equivalent work unless parasitic utility costs are
modeled separately.

The netting happens once, in `CycleKPIs.energy_breakdown()`, which is the only
call site of `convert_energy_to_equivalent_work()`; every figure, CSV and
summary downstream inherits it. **Only the equivalent-work view is netted.** The
raw and thermal tables are duties, and they keep the whole condenser number.

The numerical KPI layer retains this quantity in MJ/kgCO2 for optimization
compatibility. User-facing CSS reports divide by 3.6 and display it as
`kWh_eq/kgCO2`. This is a work-equivalent metric, not direct electricity use.
Direct electrical demand contains compression and vacuum only.
The thermal-utility export separates jacket heating, steam heating, jacket
cooling, and condenser cooling. Steam is an input heating utility, so no
`Steam cooling` column is emitted.

Default parameters in the analyzer are:

$$
T_L=298.15\ \mathrm{K},
\quad
T_H=373.15\ \mathrm{K},
\quad
T_{\mathrm{cond}}=278.15\ \mathrm{K},
\quad
\eta_{\mathrm{turb}}=0.75,
\quad
\eta_{\mathrm{refr}}=0.5,
$$

$$
\Delta T_{\mathrm{cw}}=5\ \mathrm{K},
\qquad
\Delta T_{\mathrm{chiller}}=5\ \mathrm{K}.
$$

$\Delta T_{\mathrm{cw}}$ is `cooling_water_approach` and must agree with
`T_COOLING_WATER_LIMIT` in the step post-processor, which is where the split is
actually measured: the KPI layer cannot re-derive it from the stored scalar.
$T_L$ is a **site** parameter, not a choice — cold cooling water is a chiller.

---

## 18. CSS summary metrics

The CSS summary combines the previous definitions into a compact table. The
main reported quantities are:

| Metric | Definition |
|---|---|
| Cycle time | $\sum_s t_s$ |
| Bed volume | saved `SimulationContext` geometry |
| Solid adsorbent mass | $m_s$ of §16 |
| CO2/H2O adsorbed | adsorption absolute state mass change |
| CO2/H2O desorbed | absolute state mass change summed over the product steps |
| CO2/H2O recovered | $m_i^{\mathrm{rec}}$ of §13 |
| CO2 vented | sum of `vented mass` over the steps whose outlet is not collected, and its percent of adsorbed CO2 (NaN for a result without the record) |
| Saturation after adsorption | final CO2 bed loading fraction of `ads` |
| Saturation after desorption | final CO2 bed loading fraction of the last desorption step |
| CO2/H2O working capacity | end-adsorption minus end-desorption inventory per solid mass |
| CO2/H2O recovery | recovered divided by adsorbed |
| Equivalent work | equivalent energy per recovered CO2 |
| Productivity | recovered CO2 per bed volume per day |
| Wet/dry purity | recovered product mole fractions |
| O2/N2 content | recovered product ppm (molar) |
| H2O/CO2 ratio | recovered H2O over recovered CO2 |
| Preheating temperature check | final bed minimum vs steam-condensation temperature |
| Steam/CO2 ratio | steam injected in every step divided by recovered CO2 (absent without steam) |

Without H2O, the H2O transfer, working-capacity and recovery entries and the H2O/CO2 ratio are
not applicable and are reported as NaN, not as zero water KPIs. A missing O2 or N2 likewise
gives NaN for that species' product ppm. Without injected steam the steam/CO2 ratio is `None`
and its summary row is omitted. CSS report suites require CO2.

---

## 19. Plotting outputs

The post-processing and KPI analysis layers generate plots from the same
quantities documented above.

Per-step profile plots are handled by `ProfilesPlotter` in
[`post_processing/profiles/step_plotting.py`](../../src/pydac/post_processing/profiles/step_plotting.py),
following `execution.plot_profiles` with its overrides in `execution.step_options` (step run).
A CSS run applies `execution.cycle_options[cycle].plot_profiles` and that cycle's `step_options`,
plus `execution.plot_final_cycle_profiles` for the reported cycle; a run-wide
`execution.step_options` is refused for CSS. A snapshot report replay draws every step of the
selected cycle. The plots are:

- outlet composition, pressure, velocity, and temperature traces (averages in §4.1)
- internal axial/radial profiles
- 2D field snapshots
- temperature snapshots with wall temperature

The schedule of a 2D snapshot plot comes, in order, from: the `snapshot_times` of the run or
report (`run_steps` / `run_css` / `report` / `--snapshot-times`; `api._snapshot_overrides`
lays them over the table below for that call only and restores it afterwards; a report
replays the cycle through `workflows.orchestrator_TVSA_cycle_css.replot_cycle_profiles`, which writes plots only: `TVSA_Cycle._run_step(persist_step_outputs=False)` leaves the step logs, step metrics and their provenance untouched); `SNAPSHOT_TIME_OVERRIDES` in
[`utils/plot_helpers.py`](../../src/pydac/utils/plot_helpers.py) (fixed times per step and
field); six times evenly spaced up to the field's `T_END_OVERRIDES` end, else the step end.

The 1D profiles (`internal_1D_profiles/*_alltimes`) use the same times, panel by panel:
`ProfilesPlotter._profile_schedule(field)` returns the schedule of the field's 2D map
whenever the field has times for the step (from the run or `SNAPSHOT_TIME_OVERRIDES`),
without printing the time mapping again; otherwise `n_times` (10) evenly spaced saved
times. `T_wall` follows `T_bed`; `u_z` has no field and keeps the evenly spaced times.

`snapshot_times` is `{step: {field: {"times": [...], "unit": "s"}}}`. Each field takes 1 to 6
distinct finite nonnegative times, in `s` (the default), `min` or `h`; the fields are
`y_<declared species>`, `q_<declared adsorbate>`, `c_tot`, `T_bed` and `p_tot`. A fresh run
refuses times for a step it does not execute or does not draw: the schedule chooses times, not
plot switches. Without an override a 2D map keeps six times. Each time maps to the nearest saved
state and repeated states are collapsed, so a map can have fewer panels than requested, down to
one for a single saved state.

A CSS report with `snapshot_times` reloads `resolved_config.json` and needs
`cycle_N/<step>/sol.pkl` for every step of the selected cycle; for `N > 1` it also needs the
`sol.pkl` of the previous cycle's last repeating step, which supplies the warm start. Missing
trajectories are refused before any report is written, so `execution.keep_only_final_cycle = true`
(which deletes the earlier cycles) can prevent this replay. A report without `snapshot_times`
does not replay the steps.

Cycle/CSS KPI plots are split by data source. `KPIPlotting` handles typed KPI
figures:

- mass change bars by step and species
- residual gas-phase O2/N2 in ppm (molar) by step
- direct electrical and thermal-utility breakdowns by step, where the condenser
  bar keeps its full duty and divides into solid (what a chiller carries) and
  hatched (what ambient takes for free) — `_plot_stacked_utility_table`'s
  `hatch_share`, fed by `CycleKPIs.condenser_cooling_split()`
- equivalent-work contributions in kWh_eq/kgCO2, where the condenser is only the
  part that is paid for: the free share costs no work and has no height
- state mass changes relative to adsorption uptake
- compact process-KPI dashboard
- recovered product composition and trace impurities
- equivalent-work conversion conditions as a compact table figure

This is the full DAC suite. `KPIPlotting.plot_all()` requires CO2. Without the full
CO2/H2O/O2/N2 set it uses generic mass-change plots; the residual-gas plots include only the
O2/N2 present and are skipped when neither is. The adsorption-normalized mass figure needs both
CO2 and H2O. The process overview and the recovered-product composition figures need all four
DAC species and are otherwise skipped with a reason, although the generic purity calculation
underneath is available.

`CyclicSteadyStateProfiles` handles transient solution plots over the cycle
(averages in §4.1), because those plots require the saved time- and
space-resolved solver trajectory. A report draws the outlet mole fraction of every
species, the loading of every adsorbate, `T_bed`, `T_wall` and the pressure; the
outlet O2 content in ppm (molar) is drawn only by `plot_cycle_impurity_outlet()`.
Its figures do not go into the `kpis/` tree: they are written to
`plots_css/cycle_N/` in the results folder, one folder per reported cycle (§21). Every figure is written in `execution.plot_format`.

The plotting layer should be interpreted as visualization of the KPI
definitions in this document, not as an independent source of definitions.

---

## 20. Current assumptions and caveats

The following assumptions are part of the current post-processing model:

1. The gas phase is ideal for pressure reconstruction and separator volumetric
   flow estimates.
2. The configured closure species is reconstructed from total concentration and
   all independently solved gas species.
3. Species in the configured adsorbate subset have gas and adsorbed phases;
   every other declared species is gas-only.
4. The gas inventory uses the total gas holdup
   $\epsilon_b+(1-\epsilon_b)\epsilon_p$.
5. The solid inventory uses $\rho_p(1-\epsilon_b)V_{jr}$.
6. A species with a generated `m_<species>` scalar has both integrated
   flow-balance and state-based mass changes; other species are state-based.
7. Generic fields, masses, closure checks, CSS extraction, and basic plots use
   the declared mixture and do not require DAC species.
8. The condenser and vacuum flash calculations are intentionally guarded
   binary CO2/H2O DAC calculations.
9. Product purity requires CO2 and sums the recovered moles over every declared gas species
   (§15.2); H2O is subtracted for dry purity only when present. Product O2/N2 ppm are computed
   when that species is present and are NaN when it is absent.
10. The condenser temperature is a per-step input,
    `steps.<step>.outlet.condenser_temperature_K` (default 278.15 K), used by the
    flash split, the condenser duty, the post-condenser vacuum energy and the
    product the chiller must deliver in §17. It sizes the downstream dryer
    rather than defending a pipeline specification — the product leaving at
    5 °C and 20 kPa is still far wetter than a transport spec allows.
11. The condenser is charged two-stage (§9.3.1, §17.1): cooling water free down
    to `T_low + cooling_water_approach`, then one chiller. The two approaches,
    5 K each, are design assumptions. Once the split at ambient is made the COP
    model stops mattering much — one chiller against infinitely staged
    refrigeration is a factor 2 on a term worth a few per cent of the bill —
    but the *split itself* changes the bill by a factor of ten.
12. Energy KPIs share one sign axis: positive is energy supplied to the process,
    negative is energy removed from it. `Condenser Duty [MJ]` is therefore
    stored as `-E_cond` and reads negative, mirroring `Steam Energy` — §9.3.
    `Condenser Duty Rejected To Ambient [MJ]` shares that axis and is bounded by
    it in magnitude.
13. In startup, CO2/H2O initial reference masses are determined by the startup
    model initialization; for cycle interpretation, the repeating CSS cycle is
    normally more meaningful than startup-only totals.
14. In 1D mode, radial averaging degenerates to the single radial cell.

These assumptions should be revisited before using the KPI layer for a
publication-quality techno-economic comparison, especially the binary separator
treatment of O2/N2 and the condenser temperature.

One further caveat is not a modeling assumption but a provenance one: results
produced before the ambient split existed carry no
`Condenser Duty Rejected To Ambient [MJ]`; `CycleKPIs` refuses them unless read
with `allow_legacy_ledger=True`, and then charges them the legacy way (§17.1).
A mixed comparison across backfilled and un-backfilled runs is therefore not
like-for-like, and the missing metric — not a differing number — is how to tell
them apart.

---

## 21. File outputs

The step-folder tree itself is drawn once, in
[00 — Framework Map §7](00_framework_map.md#7-artifacts-on-disk); this section
documents what each of those files *contains* and what the cycle/CSS layer adds
on top.

| Path (in a step folder) | Contents |
|---|---|
| `sol.pkl` | Raw + re-dimensionalized solver output: `t`, `y_2d`, `y_1d`, `y_scalars`, `termination`, and grid/geometry `extra`, including the verified `StateLayout`. |
| `kpis.csv` / `kpis.pkl` | The `StepMetrics` catalog as a flat table: `Step, Species, KPI, Value`. Species follows the declared mixture plus `system`; the catalog itself is §11. |
| `metrics_provenance.json` | The record of the step-metric computation, rewritten when the metrics are recomputed (§1). |
| `provenance/<id>.json` | The solve record of the step (§1). |
| `log.txt` | Everything printed while this step ran, including the termination setup and actual-stop lines. |
| `profiles/internal_2D_profiles/` | 2D contour/heatmap fields (concentration, loading, temperature) at several times. |
| `profiles/internal_1D_profiles/` | Radially-averaged / centerline line profiles at several times (species, T, p, velocities). |
| `profiles/outlet_profiles/` | Outlet traces vs. time; the average each figure uses is in §4.1. |

The three `profiles/` folders exist for every step; they stay empty unless the
step plots (`execution.plot_profiles`, §19).

In a **step run** the resolved document, fingerprint sidecar, shared
`simulation_context.pkl`, and `resolved_runtime.json` all sit one level up,
beside the step folders — see
[00 — Framework Map §7](00_framework_map.md#7-artifacts-on-disk).

In a **CSS run** the split is different, and it is the layout the cycle-level
analysis actually reads: `resolved_config.json` and `case_metadata.json` stay in
the results folder (`1D_mode/` or `2D_mode/`), while each `cycle_<n>/` carries
its own `cycle_metadata.json`, `css_diagnostics.json`, `simulation_context.pkl`
and `resolved_runtime.json` beside that cycle's step folders. The results folder
also holds the whole-run `log.txt`, the run's `provenance/` record, the transient
CSS plots and the KPI report. The CSS run of [Results](../guide/06_results.md)
(default case, 1D, reported cycle 3, `execution.plot_format` = `pdf`) writes:

```text
1D_mode/
├── resolved_config.json
├── case_metadata.json
├── log.txt                       ← one log for the whole cycle loop
├── provenance/
├── cycle_1/                      ← starts with startup/ in place of ads/
├── cycle_2/
├── cycle_3/
│   ├── css_diagnostics.json
│   ├── cycle_metadata.json
│   ├── resolved_runtime.json
│   ├── simulation_context.pkl
│   ├── ads/
│   ├── blow/
│   ├── heat/
│   ├── s_des/
│   ├── cool/
│   └── press/
├── plots_css/                    ← CyclicSteadyStateProfiles (§19)
│   └── cycle_3/                  one folder per reported cycle
│       ├── css_outlet_T_bed.pdf
│       ├── css_outlet_T_wall.pdf
│       ├── css_outlet_pressure.pdf
│       ├── css_outlet_q_CO2.pdf
│       ├── css_outlet_q_H2O.pdf
│       ├── css_outlet_y_CO2.pdf
│       ├── css_outlet_y_H2O.pdf
│       ├── css_outlet_y_N2.pdf
│       └── css_outlet_y_O2.pdf
└── kpis/
    ├── cycle_3_css_kpi_summary.txt
    ├── cycle_3_css_kpi_summary.json
    ├── cycle_3_report_provenance.json
    ├── provenance/
    ├── csv/
    │   ├── cycle_3_css_kpi_summary.csv
    │   ├── cycle_3_css_step_durations.csv
    │   ├── process/
    │   │   ├── cycle_3_css_process_kpis.csv
    │   │   ├── cycle_3_css_product_composition.csv
    │   │   └── oxygen_hot_bed_check.csv
    │   ├── energy/
    │   │   ├── cycle_3_css_electrical_energy_breakdown_kWh.csv
    │   │   ├── cycle_3_css_electrical_energy_breakdown_specific_kWh_per_kgCO2.csv
    │   │   ├── cycle_3_css_thermal_energy_breakdown_MJ.csv
    │   │   ├── cycle_3_css_thermal_energy_breakdown_specific_MJ_per_kgCO2.csv
    │   │   ├── cycle_3_css_equivalent_work_breakdown_specific_kWh_eq_per_kgCO2.csv
    │   │   └── cycle_3_css_equivalent_work_conditions.csv
    │   ├── mass/
    │   │   ├── cycle_3_css_mass_transfer_summary.csv
    │   │   ├── cycle_3_css_mass_changes_state.csv
    │   │   ├── cycle_3_css_mass_changes_state_percent_of_adsorption.csv
    │   │   ├── cycle_3_css_residual_gas_ppm_by_step.csv
    │   │   └── cycle_3_css_saturation_by_step.csv
    │   └── closures/
    │       ├── cycle_3_css_bed_closure_by_step.csv
    │       └── cycle_3_css_process_closure.csv
    └── pdf/
        ├── cycle_3_css_mass_changes_co2_h2o.pdf
        ├── cycle_3_css_mass_changes_n2_o2.pdf
        ├── cycle_3_css_residual_gas_ppm_n2_o2.pdf
        ├── cycle_3_css_equivalent_work_breakdown_specific.pdf
        ├── cycle_3_css_electrical_energy_breakdown.pdf
        ├── cycle_3_css_thermal_energy_breakdown.pdf
        ├── cycle_3_css_mass_changes_percent_of_adsorption.pdf
        ├── cycle_3_css_process_kpis_overview.pdf
        ├── cycle_3_css_product_composition.pdf
        └── cycle_3_css_equivalent_work_conditions.pdf
```

The `kpis/` file names carry the label of the reported cycle. A cycle folder holds
one folder per step of its sequence (here the default one); `w_purge/` or
`s_purge/` appear only when the sequence contains them. `plots_css/cycle_N/` holds,
for reported cycle `N`, one `css_outlet_y_<species>` figure per declared species and one
`css_outlet_q_<adsorbate>` figure per adsorbate; reporting another cycle
(`pydac css --report RUN --cycle M`) adds `plots_css/cycle_M/` and leaves the others
in place. `oxygen_hot_bed_check.csv` is
the report-only check of `run_oxygen_hot_bed_check()`, written at the end of
every CSS run and report: whether any cell of the reported cycle had $T_b$ above
90 °C while O2 exceeded $10^{-6}$ ppm (molar); columns `condition_met`, `step`,
`point`, `time_s`.

The run folder above the results folder is `OUTPUT_ROOT/step_mode/…` for a step
run and `OUTPUT_ROOT/cyclic_mode/…` for a CSS run
([Results](../guide/06_results.md#where-pydac-writes)); the results folder is
`1D_mode/` or `2D_mode/`. The figure folder in `kpis/` is named by
`execution.plot_format` (`pdf/`, `png/`, …). Homogeneous wide tables include an
explicit `Unit` column in addition to units in their filenames. Mixed-unit
diagnostic tables retain the unit in each value-column name. The compact
`process_kpis.csv` table uses `KPI|Value|Unit|Basis` so the reported product
location or normalization denominator is visible without opening the full CSS
summary. Its H2O/CO2 sequence distinguishes the absolute state-basis ratios
during adsorption and desorption from the recovered product ratio after vacuum
discharge. The mass percentage table reports
`100 * state delta m(step) / abs(state delta m(adsorption))` for CO2 and H2O;
positive values are uptake and negative values are release relative to the
adsorption uptake.
