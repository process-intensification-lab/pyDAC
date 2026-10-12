---
title: "01 — Inputs"
toc: true
toc-depth: 3
embed-resources: true
---

> **This manual is the detailed technical reference: how pyDAC works inside, for contributors
> and for checking the model.** To learn to use pyDAC, do the [tutorials](../../examples/README.md)
> first, then read the [user guide](../guide/01_installation.md).

This chapter describes how pyDAC turns its inputs into one typed case: the resolution
pipeline, the case document field by field, what an explicit `null` means, which module owns
what, and how to extend the contract. To write and run cases, read the user guide page
[Cases and settings](../guide/03_cases.md); this chapter does not repeat its tasks.

| If you want to | Go to |
|---|---|
| Understand how a case is built | §1, §2 |
| Know where the command, the API and the clone launchers enter | §3 |
| Know what a field means | §4 |
| Know what an explicit `null` does | §5 |
| Know which module owns what | §6 |
| Know how an optimization campaign maps to case settings | §8 |
| Decode an error message | §9 |
| Add a new public input, species, model, or bed architecture | §10 |

---

## 1. One pipeline, two ways in

There is one configuration pipeline. What changes is only where the values come from:

- **Built case.** The built-in defaults under
  [`src/pydac/inputs/defaults/`](../../src/pydac/inputs/defaults/), then an optional
  partial or complete JSON preset (`--config PATH`, `pydac.load_case(PATH)`), then settings,
  one dotted path each (`--set PATH=VALUE`, `pydac.load_case(..., settings={...})`). Each
  source overrides the one before.
- **Frozen case.** A complete JSON case (`--case PATH`, `pydac.load_case(PATH, exact=True)`)
  is used exactly as saved. It does not inherit the built-in defaults, and no other source
  applies: `--case` together with `--config` or `--set` is refused, and `exact=True` refuses
  `settings`.

The built route ends in

```python
config.loader.load_experiment(source, settings=...) -> ExperimentConfig
```

and the frozen route in `inputs.case_io.load_case(source)`, which never consults the
defaults. Both produce the same immutable typed case. A case JSON, the equivalent nested settings
mapping and the equivalent `--set` list resolve to byte-identical documents.

Two consequences:

1. **The built-in defaults live in `src/pydac/inputs/defaults/`,** not in the configuration
   engine or in the model. The resolved JSON document is the compiled snapshot of one case,
   not the source of the defaults.
2. **The document that ran is always saved.** A step or CSS run writes `resolved_config.json`
   and `case_metadata.json` (its fingerprint) in its results folder (`1D_mode/` or `2D_mode/`);
   a campaign writes its base case in the campaign folder. A candidate's case is written in its
   evaluation folder for the run and pruned afterwards; its compact record keeps the complete
   case under `provenance.input.resolved_config`
   ([07 §4](07_nsga2_optimization.md)). The resolved document is a complete case:
   `--case` runs it again exactly.

---

## 2. The mental map

### 2.1 Resolution: how values become a typed configuration

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. BUILT-IN DEFAULTS           src/pydac/inputs/defaults/*.py       │
  │    The complete reference input. Each domain owns its values in     │
  │    a fresh mapping function.                                        │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │  merge_documents()  ← unknown key = error
  ┌──────────────────────────────┴──────────────────────────────────────┐
  │ 2. PRESET DOCUMENT             --config / load_case(PATH)           │
  │    Optional partial or complete overlay. Omitted fields inherit     │
  │    the built-in defaults; an explicit `null` follows §5.            │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │  _resolve_owned_defaults()
                                 │  `null` directives get their canonical value
  ┌──────────────────────────────┴──────────────────────────────────────┐
  │ 3. SETTINGS                    --set path=value / settings={...}    │
  │    Canonical dotted paths: `grid.nz`, `steps.ads.horizon....`       │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │  _resolve_derivations()
  ┌──────────────────────────────┴──────────────────────────────────────┐
  │ 4. DERIVATIONS                 config/derivations.py                │
  │    the computed geometry pair  ← the given pair (§4.3)              │
  │    volumetric_flow_m3_s ← steam_mass_flow_kg_s (steam-fed steps)    │
  │    Recomputed on every resolution, so no derived value is stale.    │
  └──────────────────────────────┬──────────────────────────────────────┘
                                 │  _construct_config()
  ┌──────────────────────────────┴──────────────────────────────────────┐
  │ 5. TYPED, IMMUTABLE            ExperimentConfig                     │
  │      .simulation  → grid solver bed geometry wall sorbent mixture   │
  │                     adsorption reference cycle                      │
  │      .execution   → what the run does                               │
  │      .schema_version                                                │
  └─────────────────────────────────────────────────────────────────────┘
```

**Precedence, lowest to highest:**

```
  built-in defaults  <  preset (--config)  <  settings (--set)
```

The launchers in `runners/` add one layer for contributors working in a clone: their
`DEVELOPMENT_SETTINGS` sit between the preset and `--set` (§3.2). The environment variable
`PYDAC_PROFILE` can name a complete JSON document that replaces the built-in defaults as the
base of a built case; it is never consulted by `--case`.

`--config` and `--case` are deliberately separate: a preset is a starting point that the
settings change, a frozen case is a complete record that nothing changes.

### 2.2 From the document to the physics

```
   ExperimentConfig                              config/schema.py
        │
        │  CycleConfig.from_experiment(...)      config/cycle.py
        ▼
   CycleConfig            = the runtime view of one run
        │                   (.simulation, the solver options, the case)
        │  .step_config("ads")
        ▼
   ColumnModelConfig      = everything one step of the model needs
        │
        │  ColumnModelFactory(cycle_config)      core/column_model_factory.py
        │    .build_column_model_input(step=...)
        ▼
   ColumnModelInput       = config + isotherm and kinetic objects
        │                   + the previous step's state
        ▼
   ColumnModelMoL._bind_inputs()                 core/column_model.py
        │  a fixed list of named assignments: self.T_feed, self.p_feed, …
        ▼
   the PDE right-hand side
```

`_bind_inputs()` turns the public inputs into model attributes through a fixed list of named
assignments; there is no generic loop over the configuration. The model therefore owns its
attribute surface, and the physics reads the short names it always read.

### 2.3 The one distinction that causes most of the confusion

| | `simulation.cycle.steps` | `execution.cycle_steps` |
|---|---|---|
| Answers | *How is each step configured?* | *Which steps run, in which order?* |
| Contains | the **ten** step definitions, always all of them | a sequence of step tokens, or `null` |
| Example | `steps.heat.thermal.feed_temperature_K = 373.15` | `["ads", "blow", "heat", "s_des", "cool", "press"]` |
| Owner | `config/steps.py` | `utils/tvsa/cycle_steps.py` owns the two default sequences |

The ten steps are always defined. Choosing `s_des` over `w_des` in `execution.cycle_steps` is
what makes a cycle steam-assisted rather than wall-heated; it does not delete the other
step's configuration.

`execution.cycle_steps = null` means "use the default sequence for the mode being run":
`DEFAULT_STEP_MODE_SEQUENCE` for a step run, `DEFAULT_CYCLIC_SEQUENCE` for a CSS run.

**The tokens you may write.** Every entry of `execution.cycle_steps` is passed through
`normalize_step_token()`
([`src/pydac/utils/tvsa/steps.py`](../../src/pydac/utils/tvsa/steps.py)), which lowercases
the token and reads `-` and spaces as `_`. So `PRE-HEATING`, `pre heating` and `pre_heating`
are the same token:

| Canonical key | Accepted tokens |
|---|---|
| `startup` | `startup`, `start`, `initialization` |
| `ads` | `ads`, `adsorption` |
| `blow` | `blow`, `blowdown` |
| `heat` | `heat`, `preheat`, `preheating`, `pre_heating` |
| `w_purge` | `w_purge`, `wall_purge`, `wall_heated_purge`, `wall_purging` |
| `s_purge` | `s_purge`, `purge`, `steam_purge`, `steam_purging`, `pre_purge` |
| `w_des` | `w_des`, `wall_heated_desorption`, `wall_desorption` |
| `s_des` | `s_des`, `steam_assisted_desorption`, `steam_desorption` |
| `cool` | `cool`, `cooldown`, `cooling` |
| `press` | `press`, `pressurization`, `pressurisation` |

The canonical key on the left is what appears in folder names, `step_options` keys and
settings paths. The step KPI tables use display labels instead (`adsorption`, `preheating`,
`cooldown`, …, from `StepPostProcessor.compute_step_metrics()`). The generic tokens `des` and `desorption` are refused on purpose: the
desorption mode must be stated explicitly
([05 — Cycle Steps §1.2](05_cycle_steps.md#12-explicit-desorption-choice)).

### 2.4 The live settings catalog

Every subcommand prints the accepted vocabulary without resolving a case or starting a
simulation:

```bash
pydac validate --show-settings
```

It lists every accepted leaf path from `src/pydac/inputs/defaults/`, its built-in default, its
type and its limits, and ends with the blocks that are set as a whole. The catalog is
generated at run time, so a new default appears there without further work. Lines from it:

```text
grid.nz                                                              default=50                 integer; >= 1
geometry.length_m                                                    default=null               number; nullable; > 0
geometry.aspect_ratio                                                default=0.5                number; nullable; > 0
steps.s_purge.termination.threshold_value                            default=1e-07              number; nullable; > 0
steps.s_des.flow.steam_mass_flow_kg_s                                default=0.001              number; nullable; >= 0
```

The path before `default=` is what `--set` and `settings={...}` take. `bed` accepts either a
complete mapping or the named preset `packed_bed_default`; `bed.design`, `bed.material`,
`adsorption.equilibrium_systems`, `adsorption.kinetics`, `execution.step_options` and
`execution.cycle_options` are replaced as one block when they are given at their block path.
A dotted setting may still change a leaf that already exists in the resolved mapping, such as
`adsorption.kinetics.CO2.parameters.k_LDF_s_inv`; it cannot create a new species or model key:
supply the complete block to add keys. The same rule applies to the sorbent mappings and the
startup loading mapping.

---

## 3. Entry points

### 3.1 The command and the API

The `pydac` command ([`src/pydac/cli.py`](../../src/pydac/cli.py): `steps`, `css`,
`nsga2`, `validate`, `get-started`) and the Python API ([`src/pydac/api.py`](../../src/pydac/api.py), the
names in `pydac.__all__`) are the supported entry points. Their use is in the user guide:
[Quick start](../guide/02_quickstart.md), [Cases and settings](../guide/03_cases.md) and
[Running steps and cycles](../guide/04_running.md).

All four case-aware subcommands share the case flags `--config`, `--case`, `--export-case`,
`--overwrite-case`, `--set`, `--dry-run`, `--print-resolved` and `--show-settings`, defined once
in [`src/pydac/config/terminal.py`](../../src/pydac/config/terminal.py); `pydac validate`
adds `--validate`, `--write-resolved PATH` and `--export-preset PATH` (with
`--overwrite-preset`) ([`src/pydac/config/cli.py`](../../src/pydac/config/cli.py)).

`--export-preset` and `pydac.export_settings(settings, path, *, base=None)` write the
settings, on top of the optional preset `base` (`--config`), as one preset
(`config.loader.settings_to_preset`). The file states only what the two state, and loading
it alone must give the same case fingerprint as `load_case(base, settings=settings)`: the
export checks it and refuses to write otherwise. Three setting meanings have no
field-by-field form in a preset, so the export writes them out: a field inside an opaque
block, or inside the bed (its profile value is the `null` directive of §5), writes the
whole resolved block; an inheriting `null` over a `base` value keeps that value; a computed
geometry field of `base`, or the volumetric flow of a steam-fed step, that the settings make
stale (§4.3) is left out, so it is recomputed. The terminal builds a case in one pass, as
`load_case(base, settings=...)` does, so the frozen case, the resolved config and the preset
of one `pydac validate` call are the same case; every target is checked before the first
file is written. (A launcher with `DEVELOPMENT_SETTINGS` resolves the preset and those
settings first, then `--set` on top.)

A `--set` value is parsed as JSON, falling back to a plain string: `grid.nz=25` is an
integer, `execution.solve=false` a boolean, `execution.extra_tag=trial7` the string
`"trial7"`, and a list needs JSON quoting:
`--set 'execution.cycle_steps=["ads","blow","heat","s_des","cool","press"]'`. `NaN` and
`Infinity` are refused, so a configuration stays standard JSON.

### 3.2 The launchers in `runners/` (contributors)

A clone has one launcher per subcommand in `runners/`, for running and debugging from an
editor. Each has three blocks, `CASE_PATH`, `CONFIG_PATH` and `DEVELOPMENT_SETTINGS`; their
use and precedence are in [`CONTRIBUTING.md`](../../CONTRIBUTING.md). The
`DEVELOPMENT_SETTINGS` mapping accepts three equivalent spellings, resolved by
[`src/pydac/config/settings.py`](../../src/pydac/config/settings.py):

```python
DEVELOPMENT_SETTINGS = {"grid": {"nz": 25, "nr": 1}}          # nested block
DEVELOPMENT_SETTINGS = {"grid.nz": 25, "grid.nr": 1}          # dotted
DEVELOPMENT_SETTINGS = {"simulation": {"grid": {"nz": 25, "nr": 1}}}   # structural wrapper
```

### 3.3 The internal workflows

The API calls two workflows, one per run mode:

| Workflow | Function |
|---|---|
| [`workflows/orchestrator_TVSA_steps.py`](../../src/pydac/workflows/orchestrator_TVSA_steps.py) | `run_steps_simulation()`, one step sequence |
| [`workflows/orchestrator_TVSA_cycle_css.py`](../../src/pydac/workflows/orchestrator_TVSA_cycle_css.py) | `run_css_simulation()`, cycles to cyclic steady state |

They are internal: reachable, but not part of the supported API, and their keyword arguments
(for example the persistence and composition-guard options `run_css_simulation()` takes for
NSGA-II candidates) may change without notice. Both take the resolved case and an optional
`base_folder`, through
[`src/pydac/config/launch.py`](../../src/pydac/config/launch.py)
(`resolve_experiment_config`).

**Run behavior is stated in the document, not in keyword arguments.** The workflows take no
`solve=`, `max_cycles=`, `plot_profiles=`, `cycle_steps=` or `step_options=` arguments: they
read all of them from `execution.*` of the resolved case. That is what makes the saved
`resolved_config.json` a faithful record of what ran. `base_folder` only names where results
are written (`execution.output_root` when omitted).

The repeating CSS sequence must start with adsorption; in cycle 1 that leading step is
replaced by startup ([05 — Cycle Steps §1.1](05_cycle_steps.md#11-default-sequences-in-code)).
Per-cycle control in a CSS run lives in `execution.cycle_options[n]["step_options"]` (§4.4).

---

## 4. The document

### 4.1 `simulation` — what is modeled

| Section | Fields | Notes |
|---|---|---|
| `grid` | `nz`, `nr`, `nt`, `nr_plot_2d` | `nr = 1` is 1D, `nr > 1` is 2D |
| `solver` | `rtol`, `atol`, `atol_scalar_mass`, `atol_scalar_energy` | CVODE tolerances; defaults `1e-8`, `1e-10`, `1e-8`, `1e-6` |
| `bed` | `architecture`, `design`, `material` | `null` gives the default packed bed owned by `core/bed_architectures` (§4.6) |
| `geometry` | `specified_by`, `aspect_ratio`, `volume_m3`, `length_m`, `radius_m` | one pair is given, the other computed (§4.3) |
| `wall` | `thickness_m`, `specific_heat_J_kgK`, `density_kg_m3`, `conductivity_W_mK`, `overall_htc_override_W_m2K`, `bed_wall_htc_override_W_m2K` | the overrides bypass the heat-transfer correlations: `overall_htc_override_W_m2K` replaces the 1D overall (Dixon) coefficient `U_w`, `bed_wall_htc_override_W_m2K` the 2D bed-side `h_w`; a case that sets one on the other grid is refused |
| `sorbent` | `conductivity_W_mK`, `density_kg_m3`, `particle_radius_m`, `porosity`, `pore_radius_m`; species-keyed `heats_of_adsorption_J_mol`; `heat_capacity`; species-keyed `fixed_adsorbates` | a heat for every species of `mixture.adsorbates` (others are ignored), each **negative** (exothermic convention); `heat_capacity` is `{"model": "Low2023", "parameters": {a_J_K_per_kg, b_J_per_kgK2, c_J_per_kgK}}` (the default, Lewatit) or `{"model": "constant", "parameters": {"value_J_kgK": …}}`, evaluated at each step's feed temperature; `fixed_adsorbates` (default `{}`) are loadings that only add heat capacity, each `{loading_mol_per_kg, heat_capacity_J_molK}`. The three mappings are replaced wholesale, like `kinetics` |
| `mixture` | ordered `species`, one `closure_species`, ordered `adsorbates` | molecular constants come from the species catalog; a case never states state indices or duplicate constants |
| `adsorption` | model-name keyed `equilibrium_systems`; species-keyed `kinetics` | each selection states datasets and unit-bearing parameters, for example `k_LDF_s_inv` |
| `reference` | `temperature_K`, `pressure_Pa`, `relative_humidity`, `co2_mole_fraction`, `velocity_m_s` | nondimensionalization state |
| `cycle.steps` | the ten step definitions | §4.2 |

The solid declarations must be JSON numbers, not booleans or numeric strings. The resolved
dry-sorbent heat capacity must be finite and positive at the feed temperature of every declared
step, even a step outside the execution sequence; loading checks this before running. Each
`fixed_adsorbates` entry contains exactly `loading_mol_per_kg` (finite, ≥ 0) and
`heat_capacity_J_molK` (finite, > 0); these loadings add heat capacity and are not solved
adsorbates.

**Species names do not depend on case.** `co2` is `CO2` in the mixture lists and in every
species-keyed mapping (`kinetics`, `heats_of_adsorption_J_mol`, `fixed_adsorbates`, the
startup `initial_solid_loadings_mol_per_kg`). pyDAC stores them in upper case, so the two
spellings give one case, one fingerprint and one run folder, and an exported case writes
`CO2`. Two keys that name one species, such as `CO2` and `co2`, are refused. The parameters
inside an equilibrium system belong to its model and keep their own names.

#### Parameters of `CompetitiveSips`

`CompetitiveSips` (competitive Sips isotherm, used in tutorials 02 and 06) takes all its
numbers from the case. `datasets` is a free label for the source of the numbers; a block with
a missing or unknown parameter is refused, and the message lists the names it requires and
allows. With `x = T / reference_temperature_K` and the partial pressure `p` of each species
in Pa, for species `i`:

- `q_inf,i = qinf_ref * exp(qinf_temperature_coefficient * (x - 1))`
- `b_i = affinity_prefactor * exp(affinity_heat_J_per_mol / (R T))`
- `c_i = heterogeneity_ref + heterogeneity_temperature_coefficient * (x - 1)` (at least `1e-6`)
- `q_i = q_inf,i * (b_i p_i / pressure_unit_Pa)^c_i / (1 + sum_j (b_j p_j / pressure_unit_Pa)^c_j)`

| Parameter (under `parameters`) | Unit | Required | Meaning |
|---|---|---|---|
| `reference_temperature_K` | K | yes | the `T_ref` of `x` |
| `pressure_unit_Pa` | Pa | yes | pressure unit of the fit (`100000.0`: bar) |
| `component_parameters` | – | yes | one entry per adsorbate, species-keyed (see below) |
| `min_temperature_K`, `max_temperature_K` | K | no (150, 1000) | the temperature is clipped to this range |
| `modifier_loading_mol_per_kg` | mol/kg | no (0) | loading `q_m` of a modifying species (for example water) |
| `capacity_modifier_coefficient_kg_per_mol` | kg/mol | no (0) | `alpha`: every `q_inf` is divided by `1 + alpha q_m` |
| `affinity_modifier_coefficient_kg_per_mol` | kg/mol | no (0) | `beta`: every `b` is multiplied by `exp(-beta q_m)` |

Each entry of `component_parameters` has these six numbers, all required:

| Parameter | Unit | Meaning |
|---|---|---|
| `qinf_ref_mol_per_kg` | mol/kg | saturation capacity at `T_ref` |
| `qinf_temperature_coefficient` | – | how `q_inf` changes with `x` |
| `affinity_prefactor_inverse_pressure_unit` | 1 / pressure unit | prefactor of `b` |
| `affinity_heat_J_per_mol` | J/mol | heat in the exponent of `b` |
| `heterogeneity_ref` | – | exponent `c` at `T_ref` |
| `heterogeneity_temperature_coefficient` | – | how `c` changes with `x` |

A complete block, for CO2 and N2 on zeolite 13X, is in
[`examples/cases/02_custom_case.json`](../../examples/cases/02_custom_case.json).

Schema version 8 is the form written to `resolved_config.json`. Older documents are upgraded
while loading, deterministically: a schema-1 case gets the named mixture and model
declarations above, a schema-2 case loses its linear-solver fields (§4.5), and a schema-3 case
gets the per-step `outlet` group with the values that reproduce what it computed (product
collected through a 278.15 K condenser on `s_purge`, `w_des` and `s_des`, vented elsewhere,
§4.2). A schema-4 case (and every older one) gets the step valve conductances under their
unit-free names: `flow.blowdown_valve_conductance_m_s` becomes
`flow.blowdown_valve_conductance` and `flow.pressurization_valve_conductance_m_s` becomes
`flow.pressurization_valve_conductance`, with the same value. A schema-5 case (and every older
one) gets `wall.wall_htc_override_W_m2K` as `wall.overall_htc_override_W_m2K`, with the same
value: it replaces the 1D overall coefficient `U_w`. A schema-6 case (and every older one)
gets `sorbent.heat_of_adsorption_co2_J_mol` and `sorbent.heat_of_adsorption_h2o_J_mol` as the
mapping `sorbent.heats_of_adsorption_J_mol` (`CO2`, `H2O`; a field the document did not state
keeps its reference value), plus the `heat_capacity` and `fixed_adsorbates` it computed with
(Lewatit `Low2023`, none). A schema-7 case (and every older one) gets
`composition.initial_solid_loadings_mol_per_kg = null` in every step it states: the startup
solid in equilibrium with the initial gas, as it computed. In every rename or addition the new
field inside an older document is refused. Once resolved or exported, the document is schema 8.

### 4.2 A step: ten names, seven groups, fixed order

Step names: `startup`, `ads`, `blow`, `heat`, `w_purge`, `s_purge`, `s_des`, `w_des`, `cool`,
`press`.

The group order is part of the contract:

| Group | Fields |
|---|---|
| `composition` | `co2_feed_mole_fraction`, `relative_humidity`, `co2_initial_mole_fraction`, `initial_relative_humidity`, the optional `o2_`/`n2_` feed and initial mole fractions, and `initial_solid_loadings_mol_per_kg` (startup only, see below) |
| `flow` | `volumetric_flow_m3_s`, `steam_mass_flow_kg_s`, `blowdown_valve_conductance`, `pressurization_valve_conductance` (α in 1/m of dp/dn = α (p_set − p) at the outlet or inlet valve; null is an ideal valve) |
| `thermal` | `feed_temperature_K`, `initial_temperature_K`, `jacket_temperature_K`, `ambient_temperature_K`, `external_htc_W_m2K` |
| `pressure` | `initial_pressure_Pa`, `outlet_pressure_Pa`, `feed_pressure_Pa`, `ambient_pressure_Pa` |
| `horizon` | `nominal_time_s`, `event_max_time_s` |
| `termination` | `mode`, `metric`, `direction`, `threshold_value`, `threshold_source`, `threshold_offset`, `label`, `stop_on_low_pressure`, `low_pressure_stop_Pa`, `stop_on_pressure_plateau`, `pressure_plateau_rate_tolerance_nd`, `pressure_plateau_arm_below_Pa`, `pressure_plateau_arm_above_Pa` |
| `outlet` | `collect_product`, `condenser`, `condenser_temperature_K` |

`outlet` says where the step's outlet stream goes; it changes the post-processing and the
cycle KPIs, never the solved column. With `collect_product = true` the stream enters the
product line: a condenser at `condenser_temperature_K` when `condenser = true`, then the
vacuum pump and the vacuum-discharge flash
([06 — Post-Processing §9](06_post_processing.md#9-separator-and-product-treatment)). With
`collect_product = false` the stream is vented and reported as `mass.vented`. Defaults:
collected through a 278.15 K condenser on `s_purge`, `w_des` and `s_des`; vented on `blow`,
`heat` and `w_purge`. `startup` and `ads` discharge to the atmosphere and `cool` and `press`
have a closed outlet, so `collect_product` must stay `false` there; `condenser` requires
`collect_product`. A CSS run or campaign whose repeating steps collect nowhere is refused
before it starts.

Run preflight checks the executed steps against the mixture: an executed `s_purge` or `s_des`
requires H2O in `mixture.species`, and any executed step with `outlet.collect_product = true`
requires CO2. CSS preflight also checks `startup` and requires at least one collected outlet in
the repeating cycle. These checks run before any output folder is created; exporting or
resolving a case alone does not run them.

`composition.initial_solid_loadings_mol_per_kg` sets the solid the startup begins from. With
`null` (the default) the solid is in equilibrium with the startup initial gas
(`initial_*` mole fractions, `initial_temperature_K`, `initial_pressure_Pa`). A mapping gives
each species of `mixture.adsorbates` its own loading in mol/kg, and no other species, each
finite and ≥ 0, for example `{"CO2": 0.0, "N2": 0.357}` (the 13X bed of tutorial 06): CO2
starts at 0 and N2 at 0.357 mol/kg. Each value is the same in every cell of the bed. Species
names do not depend on case (§4.1). All species or none: a mapping that leaves one out is refused, because pyDAC would have to choose
its loading. Only `startup` reads it (every other
step starts from the solid the previous step left), so on any other step it must stay `null`.
Like the sorbent mappings it is replaced wholesale by a setting, never merged.

Two things about `pressure` that are easy to get wrong:

- **`feed_pressure_Pa` matters even when the inlet is closed.** The feed total concentration,
  the steam condensation temperature and the gas properties of the step (viscosity, molecular
  diffusivity, hence dispersion; with the feed temperature) are evaluated at it. The closed-inlet steps (`blow`,
  `w_des`, `cool`) carry a fixed 20000 Pa value that is a live evaluation state, not a vacuum
  setting.
- **There is no separate "vacuum level" field.** The vacuum level is the outlet pressure of a
  step and, for `heat`, `w_purge`, `s_purge` and `s_des`, whose supply sits at the vacuum
  level, also the feed pressure. See the fan-out table in §8.1.

A `mode: "event"` step stops when `metric` crosses its threshold in `direction`;
`event_max_time_s` is the fallback horizon. A `mode: "time"` step runs its `nominal_time_s`.
`metric` is one of the eight event metrics of
[05 — Cycle Steps §3.2](05_cycle_steps.md#32-supported-event-metrics), `threshold_source` one
of `p_out`, `p_feed` and `T_cond`, and exactly one of `threshold_value` and
`threshold_source` is set. Fields a step never reads are refused rather than ignored: the
low-pressure stop fields act only on `cool`, the plateau fields below only on `blow` and
`press`.

Blowdown and pressurization have additional pressure-rate stop rules controlled by
`stop_on_pressure_plateau`, `pressure_plateau_rate_tolerance_nd` and the direction-specific
arming field. Blowdown uses `pressure_plateau_arm_below_Pa`, and its event threshold is the
mean pressure descent rate, `-d(mean(p/p_ref))/d(t/tau_hyd)`. Pressurization uses
`pressure_plateau_arm_above_Pa` and the corresponding rise rate,
`d(mean(p/p_ref))/d(t/tau_hyd)`. The pressure value only arms each detector after the initial
fast transient; it is not the event threshold. The built-in defaults use a rate tolerance of
`1e-5` and a 50000 Pa arming value for both steps. Set the step's
`stop_on_pressure_plateau = false` to keep only its pressure-target event. See
[05 — Cycle Steps §4.3](05_cycle_steps.md#43-blowdown-blow).

### 4.3 Derived fields

| Derived | From | Rule |
|---|---|---|
| `geometry.length_m`, `geometry.radius_m` | `aspect_ratio` (L/D), `volume_m3` | when `geometry.specified_by = "volume_aspect_ratio"` (default) |
| `geometry.aspect_ratio`, `geometry.volume_m3` | `length_m`, `radius_m` | when `geometry.specified_by = "length_radius"` |
| `flow.volumetric_flow_m3_s` | `flow.steam_mass_flow_kg_s` | every step whose steam mass flow is positive |

The computed values appear in the resolved document. A value written in a computed field
must equal the computed one (relative tolerance `1e-9`), otherwise the case is refused: each
quantity is given one way only. A steam-fed step's volumetric flow is computed from its steam
mass flow at the step's feed pressure and temperature. A computed value inherited from an
older layer than its inputs (a frozen case used as a preset whose aspect ratio or steam mass
flow a setting changes, or a built-in default under the other `specified_by`) is recomputed
([`config/loader.py`](../../src/pydac/config/loader.py), `_computed_geometry_to_keep` and
`_steam_flows_to_keep`; [`config/derivations.py`](../../src/pydac/config/derivations.py),
`derive_geometry_pair` and `derive_steam_volumetric_flow`).

### 4.4 `execution` — what the run does

| Group | Fields |
|---|---|
| solve and report | `solve`, `print_to_terminal`, `print_cycle_build` |
| the run plan | `cycle_steps`, `termination_mode`, `step_options`, `cycle_options` |
| CSS | `max_cycles`, `css_tolerance`, `resume_from_cycle`, `css_probe_step`, `require_css` |
| numerics | `use_sparsity`, `compute_numerical_sparsity`, `cvode_sparsity_mode` |
| outputs | `output_root`, `extra_tag`, `plot_profiles`, `plot_final_cycle_profiles`, `plot_cyclic_profiles`, `plot_format`, `generate_kpi_outputs`, `keep_only_final_cycle` |

Because these fields are part of the document, a JSON case, a `--set` item and a launcher
block all describe the same run.

`step_options` and `cycle_options` are **opaque mappings**: the user owns their keys, so they
replace the default block as a whole instead of merging key by key. `cycle_options` keys are
cycle indices; `{"3": {...}}` in JSON and `{3: {...}}` in Python mean the same override.

`execution.extra_tag` is `null` or a portable folder tag. After stripping surrounding whitespace
and replacing spaces with `_`, it must match `[A-Za-z0-9_-][A-Za-z0-9._-]{0,63}`: 1 to 64
characters, no path separators, no leading dot. The folder uses the normalized tag; the case
fingerprint keeps the original text. The tag prefixes both run folders and NSGA-II campaign
folders.

### 4.5 The linear solver

CVODE solves every step with the **direct sparse factorization** on the analytical Jacobian
mask; it is not an input. `execution.use_sparsity = false` selects the dense solver instead,
for diagnostics only.

Case schema 2 had a per-step `steps.<step>.solver.linear_solver` (`auto`, `direct`, `gmres`)
and a run-wide `simulation.solver.krylov_min_states`. Schema 3 removed them together with the
iterative (GMRES) path. A schema-2 case is upgraded on loading: `direct` and `auto` are
dropped (with GMRES gone, both mean the direct solve), and a step that asked for `gmres` is refused
with a message naming it.

### 4.6 Packed-bed flow, transport and channeling

`bed.design.flow_model` defaults to `uniform`. In 2D, `wall-channeling-reduced` selects radial
storage and drag derived from a declared porosity profile; it includes radial convection.
`wall-channeling` is registered but refused as not implemented. Transport coefficients still
use the mean bed voidage and step bulk properties.

| Setting | Choices / default |
|---|---|
| `bed.design.wall_htc_model` | `dixon` (default), `specchia_1980` |
| `bed.design.stagnant_conductivity_model` | `zehner_schlunder` (default), `specchia_1980` |
| `bed.design.channeling.porosity_model` | `de_klerk` (default), `giese`, `tabulated` |
| `bed.design.channeling.drag_layer_width_dp` | 0.5–2.0, default 1.0 particle diameters |
| `bed.design.channeling.inlet_distribution` | `uniform` (default), `distributed` |

The drag averaging width is a physical closure parameter independent of the grid. `tabulated`
requires `radial_porosity_profile`. Explicit `uniform` flow refuses radial profiles and
channeling options; an omitted flow selection permits prescribed profiles. Channeling
generates the inlet weights and refuses a competing `inlet_flux_profile`. The uniform inlet
imposes equal feed flux per area; distributed inlet weights follow the reduced drag closure.
Both conserve the total prescribed feed. Closed-inlet steps keep zero feed, and
pressure-controlled inlets keep their pressure boundary.

For example: `--set bed.design.flow_model=wall-channeling-reduced
--set bed.design.channeling.inlet_distribution=distributed` (on one command line).
Neither inlet choice changes the conservation contract.

The channeling mode requires a 2D packed bed (`grid.nr >= 2`, `grid.nz >= 3`). Both 2D modes
use isotropic vector drag and radial gas convection. Channeling uses the regularized drag
voidage at cell centers for axial faces and its native linear interpolation for internal
radial faces, then evaluates the nonlinear Ergun law. Closed radial boundaries keep the
adjacent-cell drag voidage and zero velocity. This local isotropy is a modeling assumption,
not a measured radial permeability. Transport correlations keep their declared mean-porosity,
step-bulk policy.

Cooldown inherits the previous state directly in both modes: hydraulic reset and axial-profile
flattening are disabled.

A complete custom bed mapping is:

```json
{
  "architecture": "packed_bed",
  "design": {"eps_b": 0.4},
  "material": null
}
```

Set it with `--set 'bed={"architecture":"packed_bed","design":{"eps_b":0.4},"material":null}'`,
or change the design with `--set 'bed.design={"eps_b":0.4}'`. A design mapping replaces the
existing design as a whole; omitted optional closure selectors take their defaults. The
supported architecture is `packed_bed`, in 1D and 2D; unknown architecture names and extra
design fields are refused.

---

## 5. Omitted vs `null`

In a preset or a `--set` list, omitting an ordinary field means "inherit". Two exceptions: a
frozen case must state every field (an omission is an error), and an opaque mapping block
(`bed.design`, `bed.material`, …: `OPAQUE_MAPPING_PATHS` in `config/settings.py`) replaces the
whole default block, so a key left out inside it is not inherited. Writing an explicit `null`
is a different statement,
and each path that accepts one names its outcome
([`src/pydac/config/nulls.py`](../../src/pydac/config/nulls.py)):

| Policy | Meaning | Paths |
|---|---|---|
| `CANONICAL` | the module that owns the value supplies it | `simulation.bed`, the only one: the packed-bed design belongs to `core/bed_architectures` |
| `DERIVED` | the derivation stage computes it | `geometry.length_m`, `geometry.radius_m`, `geometry.aspect_ratio`, `geometry.volume_m3` (whichever pair `specified_by` computes) |
| `DISABLED` | stays `None`; the feature is off | `wall.overall_htc_override_W_m2K`, `wall.bed_wall_htc_override_W_m2K`, `bed.material`, `execution.resume_from_cycle`, `execution.cycle_steps`, `execution.termination_mode`, `execution.css_probe_step`, `execution.extra_tag` |
| `INHERIT` | same as omitting | every step field whose typed record admits `None` (`NULLABLE_STEP_FIELDS` in `config/steps.py`) |

Anywhere else, `null` is an error: ``null` is not allowed for `simulation.grid.nz`.``

---

## 6. Who owns what

`src/pydac/inputs/defaults/` owns the built-in values. `src/pydac/config/` is the engine that
validates, resolves, freezes, serializes and exports them. The configuration engine does not
import the physical model.

| Module | Owns | Key names |
|---|---|---|
| [`inputs/defaults/profile.py`](../../src/pydac/inputs/defaults/profile.py) | composing the built-in defaults | `default_profile()` |
| [`inputs/case_io.py`](../../src/pydac/inputs/case_io.py) | complete-case validation, exact load and export, canonical hash | `load_case()`, `export_case()`, `case_fingerprint()` |
| [`inputs/case_terminal.py`](../../src/pydac/inputs/case_terminal.py) | the terminal bridge for presets and frozen cases | `resolve_runner_experiment()` |
| [`config/profile.py`](../../src/pydac/config/profile.py) | the default profile, or an explicit `PYDAC_PROFILE` document | `reference_profile()`, `read_json_document()` |
| [`config/schema.py`](../../src/pydac/config/schema.py) | the immutable sections; no field defaults | `ExperimentConfig`, `SimulationConfig`, `GridConfig`, `SolverConfig`, `BedConfig`, `GeometryConfig`, `WallConfig`, `SorbentConfig`, `AdsorptionConfig`, `ReferenceConfig` |
| [`config/sections.py`](../../src/pydac/config/sections.py) | default-driven immutable mappings used by the sections | — |
| [`config/contract.py`](../../src/pydac/config/contract.py) | type validation inferred from the defaults, strengthened by the rules next to them | — |
| [`config/steps.py`](../../src/pydac/config/steps.py) | the seven-group step contract and the ten canonical names | `StepConfig`, `CycleStep`, `StepSequenceConfig`, `STEP_FIELD_ORDER`, `STEP_NAMES`, `NULLABLE_STEP_FIELDS` |
| [`config/execution.py`](../../src/pydac/config/execution.py) | the run contract | `ExecutionConfig` |
| [`config/nulls.py`](../../src/pydac/config/nulls.py) | what an explicit `null` means, path by path | `NULL_POLICY_BY_PATH`, `null_policy()`, `resolve_null()` |
| [`config/settings.py`](../../src/pydac/config/settings.py) | the dotted-path vocabulary; nested, dotted and command-line forms | `resolve_settings()`, `setting_path_parts()`, `OPAQUE_MAPPING_PATHS` |
| [`config/derivations.py`](../../src/pydac/config/derivations.py) | the derived quantities | `derive_geometry()`, `derive_geometry_pair()`, `derive_steam_volumetric_flow()` |
| [`config/loader.py`](../../src/pydac/config/loader.py) | the pipeline of §2.1, with no formula of its own | `load_experiment()`, `merge_documents()`, `apply_settings()`, `settings_to_preset()`, `default_experiment_config()`, `write_resolved_config()` |
| [`config/migrations.py`](../../src/pydac/config/migrations.py) | upgrading older schema versions | `upgrade_experiment_document()`, `CURRENT_SCHEMA_VERSION` |
| [`config/serialization.py`](../../src/pydac/config/serialization.py) | typed records to a JSON-safe document | `experiment_to_document()`, `to_jsonable()` |
| [`config/model_validation.py`](../../src/pydac/config/model_validation.py) | early compatibility checks of the named model declarations | — |
| [`config/bed_profiles.py`](../../src/pydac/config/bed_profiles.py) | named bed presets such as `packed_bed_default` | — |
| [`config/terminal.py`](../../src/pydac/config/terminal.py) | the shared case flags of every subcommand | `experiment_parser()`, `resolve_experiment()` |
| [`config/cli.py`](../../src/pydac/config/cli.py) | `pydac validate` | `add_validate_arguments()`, `run()` |
| [`config/input_help.py`](../../src/pydac/config/input_help.py) | the `--show-settings` catalog | — |
| [`config/launch.py`](../../src/pydac/config/launch.py) | the bridge a workflow uses: document and runtime view together | `resolve_experiment_config()` |
| [`config/cycle.py`](../../src/pydac/config/cycle.py) | the runtime view of one run | `CycleConfig`, `ColumnModelConfig`, `SolverOptions` |
| [`config/operating.py`](../../src/pydac/config/operating.py) | which step fields each operating knob writes | `OPERATING_SETTING_TARGETS`, `operating_step_settings()` |
| [`config/errors.py`](../../src/pydac/config/errors.py) | the public exceptions | `ConfigurationError`, `SettingsError` (a subclass), `RunRefused` |
| [`config/validation.py`](../../src/pydac/config/validation.py) | the reusable field validators | `coerce()`, `positive_float()`, `fraction()`, `optional_instance()`, … |
| [`config/schema_generation.py`](../../src/pydac/config/schema_generation.py) | generating the JSON Schema from the defaults and their rules | — |
| [`config/json_schema/experiment.schema.json`](../../src/pydac/config/json_schema/experiment.schema.json) | the JSON Schema for editor validation and completion | — |
| [`config/__init__.py`](../../src/pydac/config/__init__.py) | a facade: imports and `__all__`, no logic | — |

Inside [`src/pydac/inputs/defaults/`](../../src/pydac/inputs/defaults/), each file owns
one domain of the document; it is the file to edit to change a value for every case:

| File | Owns |
|---|---|
| [`solver.py`](../../src/pydac/inputs/defaults/solver.py) | `grid` and the CVODE tolerances of `solver` |
| [`column.py`](../../src/pydac/inputs/defaults/column.py) | `bed` (`None` gives the default packed bed), `geometry`, `wall` |
| [`particle.py`](../../src/pydac/inputs/defaults/particle.py) | `sorbent`: particle, porosity, conductivity, heats of adsorption |
| [`mixture.py`](../../src/pydac/inputs/defaults/mixture.py) | ordered species, closure species and adsorbates |
| [`adsorption.py`](../../src/pydac/inputs/defaults/adsorption.py) | named equilibrium systems, datasets, kinetics and unit-bearing parameters |
| [`reference.py`](../../src/pydac/inputs/defaults/reference.py) | the nondimensionalization reference state |
| [`steps.py`](../../src/pydac/inputs/defaults/steps.py) | all ten step definitions and their stop rules |
| [`execution.py`](../../src/pydac/inputs/defaults/execution.py) | solve, CSS and output behavior |
| [`profile.py`](../../src/pydac/inputs/defaults/profile.py) | composing the eight above into one profile |
| `_rules.py` | the reusable validation rules those defaults attach to themselves |

---

## 7. Editing JSON with assistance

pyDAC ships a JSON Schema,
[`src/pydac/config/json_schema/experiment.schema.json`](../../src/pydac/config/json_schema/experiment.schema.json),
generated from the defaults and their rules. A case file must not carry a `"$schema"` key:
the loader refuses unknown fields, and `$schema` is one. Connect the schema from the editor
side instead, as described in
[Cases and settings](../guide/03_cases.md#editing-a-case-with-schema-support).

---

## 8. Campaign inputs

An NSGA-II campaign does not introduce a second configuration vocabulary. A candidate is the
campaign's base case plus `candidate_settings(...)`: its operating settings and four fixed
execution settings (solve on, console and cycle-build printing off, profile plots off),
resolved by the same loader as `--set` items on the command line. How a campaign runs is in [07 — NSGA-II Optimization](07_nsga2_optimization.md).

### 8.1 Operating knobs and their fan-out

A campaign states a handful of named, unit-bearing knobs; each one stands for the step fields
it drives. The table is written once, in
[`src/pydac/config/operating.py`](../../src/pydac/config/operating.py):

| Knob | Writes |
|---|---|
| `adsorption_time_s` | `startup.horizon.nominal_time_s`, `ads.horizon.nominal_time_s` |
| `heating_time_s` | `heat.horizon.nominal_time_s` |
| `desorption_time_s` | `nominal_time_s` **and** `event_max_time_s` of both `s_des` and `w_des` |
| `cooling_time_s` | `cool.horizon.nominal_time_s` |
| `cooling_target_temperature_K` | `cool.termination.threshold_value` |
| `low_pressure_Pa` | `outlet_pressure_Pa` of `blow`, `heat`, `w_purge`, `s_purge`, `s_des`, `w_des`, `cool`, **plus the `feed_pressure_Pa` of `heat`, `w_purge`, `s_purge` and `s_des`** |
| `volumetric_flow_m3_s` | `startup.flow`, `ads.flow` |
| `steam_mass_flow_kg_s` | `s_purge.flow` and `s_des.flow` (each then derives its volumetric flow) |
| `adsorption_feed_temperature_K` | `startup.thermal`, `ads.thermal` |
| `adsorption_relative_humidity` | `startup.composition`, `ads.composition` |

`cooling_target_temperature_K` is the one knob that writes a *threshold* rather than a
horizon: it says how cold the bed must be before cooling ends, so the step still stops on
the physical event and its duration adapts to the column shape.

### 8.2 The NSGA-II base case

`NSGA2RunConfig` takes the base case as `experiment`. It reads the aspect ratio, the bed
volume, the bed, the cycle steps, `require_css` and the adsorption feed condition from it, so
they are never stated twice. Each candidate is that base case plus
`candidate_settings(...)` (its `operating_step_settings(...)` and the four execution settings
of 07 §2); the decision variables and their conversion to knobs are in
[07 — NSGA-II Optimization §2](07_nsga2_optimization.md#2-decision-variables-and-candidate-settings).

### 8.3 Where the resolved document is written

| Run | File |
|---|---|
| step run | `<results folder>/resolved_config.json` |
| CSS run | `<results folder>/resolved_config.json` |
| NSGA-II campaign | `<campaign folder>/resolved_config.json` (the base case) |
| NSGA-II candidate | in its record in `<campaign folder>/results/candidate_records/`, at `provenance.input.resolved_config` |

A complete one, saved as a JSON file, runs again with `--case`.

---

## 9. Error messages

| Message | Means |
|---|---|
| ``Unknown configuration field `simulation.geometry.tau`.`` | the JSON has a key the contract does not define: a typo or an old field name |
| ``Unknown setting path `simulation.geometry.tau`.`` | the same, for a `--set` item or a settings key |
| ``Unknown cycle step `adsorb` in `steps.adsorb.horizon.nominal_time_s`.`` | step names in settings paths are the ten canonical keys (`ads`, not `adsorb` or `ADSORPTION`) |
| ``null` is not allowed for `simulation.grid.nz`.`` | that field has no `null` policy: give it a value or omit it (§5) |
| ``CLI settings must use the form `path=value`, received 'grid.nz'.`` | a `--set` item without `=` |
| ``Invalid resolved experiment configuration: `simulation.grid.nz` must be greater than or equal to 1.`` | the value is well-formed JSON but outside the contract |
| ``Unsupported `schema_version`…`` | the document is neither schema 8 nor an upgradable older version |
| ``Invalid configuration for step `cool`: `outlet.collect_product` must be false…`` | `startup` and `ads` discharge to the atmosphere and `cool` and `press` have a closed outlet; none of them can feed the product line (§4.2) |
| ``…`outlet.condenser` requires `outlet.collect_product`…`` | a vented step has no product line; set `condenser` to `false` when `collect_product` is `false` |
| ``No step of the cycle … collects product…`` | every repeating step vents; collect the outlet of at least one (§4.2) |
| ``…`termination.metric` must be one of […]`` / ``…`threshold_source` must be one of […]`` | not an event metric or a threshold source the solver knows (§4.2) |
| ``…`termination.stop_on_low_pressure` acts only on cool…`` | a termination field the step never reads; set it to null or false (§4.2) |
| ``…`composition.initial_solid_loadings_mol_per_kg` acts only on startup…`` | only the startup builds its solid from the case; set it to null on the other steps (§4.2) |
| ``…Initial solid loadings must cover the declared adsorbates exactly (missing=[…], unknown=[…])`` | the initial loadings name every adsorbate and nothing else; none may be omitted (§4.2) |
| ``…names CO2 twice; species names do not depend on case.`` | a species-keyed mapping has two keys for one species, such as `CO2` and `co2`; keep one (§4.1) |
| ``…`composition.initial_solid_loadings_mol_per_kg` in frozen case exists only since case schema 8…`` | the field is new in schema 8; set `schema_version` to 8 (§4.1) |
| ``Step `w_purge` runs in event mode on `max_bed_oxygen_mole_fraction`, which needs O2…`` | the stop metric needs a species the mixture lacks; pick another metric or run the step in time mode (§4.2) |
| ``…`steps.s_des.flow.volumetric_flow_m3_s` is computed from the step's `steam_mass_flow_kg_s`…`` | a written steam-step volumetric flow differs from the one derived from its steam mass flow; leave it null (§4.3) |
| ``…`geometry.length_m` is computed from `geometry.aspect_ratio` and `geometry.volume_m3`…`` | a written computed geometry value differs from the computed one; leave it null, or switch `geometry.specified_by` (§4.3) |
| ``Non-standard JSON constant 'NaN' is not allowed.`` | configurations stay standard JSON |
| ``Incomplete case: missing field …`` | a frozen case without a field the contract now has; export it again with `--export-case`. Older schema versions are upgraded instead |
| ``…solver.linear_solver` is 'gmres'… GMRES was removed`` | a schema-2 case asked for GMRES on that step; set it to `direct` to run it with the direct solver (§4.5) |

Commands that help:

```bash
# is this case valid?
pydac validate --case my_case.json --validate
# can this run start?
pydac css --config my_preset.json --dry-run
# the complete resolved case
pydac validate --config my_preset.json --print-resolved
# what differs between two presets
diff <(pydac validate --config a.json --print-resolved) <(pydac validate --config b.json --print-resolved)
```

---

## 10. Extending the contract

Four kinds of extension, in increasing order of how much of the core they touch: a new
**public value** (§10.1), a new **catalog species** (§10.1.1), a new **isotherm or kinetic
model** (§10.2) and a new **bed architecture** (§10.3). The changes to
`src/pydac/core/column_model.py` and `src/pydac/post_processing/` that an extension may need
are reviewed by the maintainers before they are written
([`CONTRIBUTING.md`](../../CONTRIBUTING.md#physics-core-maintainer-review)).

### 10.1 Adding a public input

In this order:

1. **Add the value** to the matching function under `src/pydac/inputs/defaults/`; it is the
   owner of the default.
2. If the generic type validation is not enough, add a stronger rule next to that default.
3. If it may be `null`, add an explicit policy to `config/nulls.py`; `None` cannot be inferred
   safely.
4. If it is computed from other fields, add the formula to `config/derivations.py`, not to
   the loader.
5. Regenerate the JSON Schema if the default or rule tree changed; it is generated from the
   defaults and rules, not maintained by hand.
6. If the model must read it, add the assignment in `_bind_inputs()` of
   `src/pydac/core/column_model.py`, after maintainer review.
7. If a campaign should vary it, add the knob to `config/operating.py`.

A leaf added to the default tree is then accepted in settings and complete cases, readable
through the immutable sections, serialized into `resolved_config.json`, exportable as a frozen
case and included in the generated schema. A misspelled key is still refused with its full
dotted path.

Two cases need more than these steps:

- **A mapping whose keys the user owns** (species-keyed, like `adsorption.kinetics` or
  `sorbent.heats_of_adsorption_J_mol`): give its rule `opaque=True` and a JSON-schema fragment,
  and list its path in `config/settings.py` (`OPAQUE_MAPPING_PATHS`) and in
  `inputs/case_io.py` (`_OPAQUE_PATHS`). Otherwise a key that is not in the default, such as
  `N2`, is refused as an unknown field. If the model build would refuse a value, call the same
  check from `config/model_validation.py` so `--validate` and `--dry-run` refuse it first.
- **Renaming or replacing a field**: every saved case uses the old name, so raise the case
  schema in `config/migrations.py` (`CURRENT_SCHEMA_VERSION`, an upgrade that writes what the
  old case computed with, a `pre_…_vocabulary` for the structural check, refusal of the new
  name in an old document), add the branch in `inputs/case_io.validate_case_document`, and bump
  `schema_version` in `inputs/defaults/profile.py`. Every case fingerprint moves: update the
  pinned baseline and the fingerprints quoted in the documentation.

#### 10.1.1 Adding a species to the catalog

Species are added to the immutable catalog, not to a factory branch and not to any state
vector:

1. Add one `SpeciesProperties` record to
   [`src/pydac/core/species_catalog.py`](../../src/pydac/core/species_catalog.py), with the
   SI field names already used there: molecular weight, Lennard-Jones data, Fuller volume,
   conductivity reference data and Shomate coefficients.
2. Add exact-constant and immutability coverage in
   [`tests/test_species_catalog.py`](../../tests/test_species_catalog.py). Every property
   the generic initializer needs must be present and unit-labeled.
3. Put the name in `simulation.mixture.species`, choose its closure role, and list it in
   `adsorbates` only when it has a declared equilibrium and kinetic model. The generated
   `StateLayout` supplies every internal field and slice.
4. Add or select the required registry entries and a configuration, model-compatibility and
   numerical smoke test. Never write `c_3`, `q_2` or another positional index in a case.

If the new species needs a governing equation or a property correlation the generic core
does not expose, stop at the catalog stage and propose the change to the physical core for
maintainer review.

### 10.2 Adding or changing an isotherm or kinetic model

The document declares physical names, datasets and unit-bearing parameters; it never declares
a Python class or a state index. The built-in wet DAC choice is:

```json
{
  "mixture": {
    "species": ["CO2", "H2O", "O2", "N2"],
    "closure_species": "N2",
    "adsorbates": ["CO2", "H2O"]
  },
  "adsorption": {
    "equilibrium_systems": {
      "TothGABWADST": {
        "datasets": {"Toth": "Cleaned-Toth"},
        "parameters": {}
      }
    },
    "kinetics": {
      "CO2": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.003}},
      "H2O": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.00083}}
    }
  }
}
```

The equations and the model-name registries are next to each other:

| Change | File |
|---|---|
| equilibrium equations, requirements and model-name registry | [`src/pydac/core/isotherm_models.py`](../../src/pydac/core/isotherm_models.py): `EquilibriumSystem`, the Toth/GAB/WADST systems, `EQUILIBRIUM_REGISTRY` |
| kinetic equations, supported species and model-name registry | [`src/pydac/core/kinetic_models.py`](../../src/pydac/core/kinetic_models.py): `KineticLaw`, `LDFRate`, `KINETIC_REGISTRY` |
| registry construction and coverage checks | [`src/pydac/utils/model_registry.py`](../../src/pydac/utils/model_registry.py): `build_equilibrium_system()`, `build_kinetic_laws()` |
| configuration-to-registry composition | [`src/pydac/core/column_model_factory.py`](../../src/pydac/core/column_model_factory.py): `_build_model_components()` |

Loading checks that every named species exists in the catalog, every adsorbate belongs to the
mixture, the equilibrium and kinetic selections cover the adsorbates exactly once and in
declaration order, and every model requirement is met. These errors occur before
`ColumnModelFactory` builds a model.

To add a model:

1. Implement the name-based contract in the owning module. Equilibrium systems return
   loadings in mol/kg by species name; kinetic laws return loading rates in mol/(kg s) by
   species name.
2. Add one registry entry next to the equation: owned adsorbates, required gas and loadings,
   accepted datasets, required parameters and the constructor.
3. Select that registry name in the complete `adsorption.equilibrium_systems` or
   `adsorption.kinetics` block. Dimensional parameter names carry their units, as
   `k_LDF_s_inv` does.
4. Add focused equation, compatibility and configuration round-trip tests.

A minimal registry extension runs in
[`tests/test_phase7_extension_example.py`](../../tests/test_phase7_extension_example.py).
It copies the built-in mappings, adds a constant equilibrium system and a first-order kinetic
law for CO2, builds them through `build_equilibrium_system()` and `build_kinetic_laws()`, and
checks the named loading and rate. It is a test-only extension: no factory branch or
production registry is edited.

```bash
python -m pytest -q tests/test_phase7_extension_example.py
```

A registry entry needs no species branch in the factory. A model that changes the governing
equations or the numerical core still needs a change to `src/pydac/core/column_model.py`,
reviewed by the maintainers first.

A user can also add an entry without editing the package: `pydac.register_equilibrium` and
`pydac.register_kinetics` ([`src/pydac/extensions.py`](../../src/pydac/extensions.py)) keep
it for the current Python session, beside the built-in entries.
`config/model_validation.py` and `ColumnModelFactory` read `equilibrium_registry()` and
`kinetic_registry()`, the built-in mappings plus the registered ones
([Extending pyDAC](../guide/08_extending.md)).

`register_equilibrium(name, entry)` takes an `EquilibriumRegistryEntry`, `register_kinetics(name,
entry)` a `KineticRegistryEntry`. The name must be a non-empty string without surrounding
whitespace and `entry.build` must be callable. A built-in name is refused; registering an
existing custom name replaces its entry for the current session.

Register custom models before loading a case that selects them, and register them again before
`resume` or `report` in a new Python session. NSGA-II workers started with `spawn` or
`forkserver` (the default on Windows, macOS and, from Python 3.14, Linux) are new sessions and
do not inherit the parent's registrations. The saved case contains model names and selections,
not the custom Python code: keep the model implementation with the results. Reusing a
registered name with different code does not by itself change the case fingerprint.

### 10.3 Packed-bed architecture ownership

The registered architecture is `packed_bed`, in 1D and 2D. The typed configuration delegates
bed validation to `resolve_bed_config()` in
[`src/pydac/core/bed_architectures/`](../../src/pydac/core/bed_architectures/), which owns:

- the `ARCHITECTURE_REGISTRY` metadata, in `__init__.py`;
- `PackedBedDesign` and exact-field validation, in `base.py`, together with the contract a
  new architecture implements: the `typing.Protocol` classes `BedArchitecture`,
  `PressureDropLaw`, `HeatTransferModel` and `DispersionModel`;
- the structure, pressure-drop, dispersion and thermal closures, in `packed_bed.py`;
- the radial channeling structure of §4.6, in `channeling.py`.

Adding an architecture needs a coordinated change to the configuration, the solver, result
persistence and the tests, and the change to `src/pydac/core/column_model.py` is reviewed by
the maintainers first.

### 10.4 Two standing rules

- **Single ownership is about definitions.** If a number appears in two places, one of them
  is wrong, even when they currently agree.
- **Compactness never beats clarity.** Do not compress a guard onto a long line or unroll a
  declarative construct to fit a limit. A module that needs more room needs a new owner.

---

## Related documents

- [Cases and settings](../guide/03_cases.md): writing and checking cases.
- [00 — Framework Map](00_framework_map.md): layers and entry points.
- [05 — Cycle Steps](05_cycle_steps.md): the physics of each step.
- [07 — NSGA-II Optimization](07_nsga2_optimization.md): campaign internals.
