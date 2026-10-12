---
title: "Cases and settings"
toc: true
toc-depth: 3
embed-resources: true
---

> **New to pyDAC? Do the [tutorials](../../examples/README.md) first, starting with
> [01 First steps](../../examples/notebooks/01_first_steps.ipynb).** They teach pyDAC step by
> step on real runs; this guide is the reference behind them and goes into more technical detail.

A case is the complete set of inputs of one simulation: grid, column, sorbent, mixture,
adsorption models, every cycle step and the run options. This page shows how a case is
built, how to change it, and what decides its run folder. The meaning of each field is in
the reference chapter [01 — Inputs](../manual/01_inputs.md).

## Four sources, one case

| Source | Terminal | Python |
|---|---|---|
| built-in defaults | always the base | always the base |
| preset: a JSON file with some fields | `--config PATH` | `pydac.load_case(PATH)` |
| settings: one field each | `--set PATH=VALUE` (repeatable) | `pydac.load_case(..., settings={...})` |
| frozen case: a complete JSON case | `--case PATH` | `pydac.load_case(PATH, exact=True)` |

Settings tried in Python can be kept as a preset: `pydac.export_settings(settings, PATH)`
writes a preset with only the fields they state, and `pydac.load_case(PATH)` then gives
the same case. In the terminal, `pydac validate --config BASE --set ... --export-preset
PATH` writes the preset `BASE` with the settings on top (`--overwrite-preset` replaces an
existing file). A field of a block that a preset gives as one (the bed, or a block marked
"replace as one block") is written with the whole block.

Without a frozen case, the case is built in this order, each source overriding the one
before: built-in defaults, then the preset, then the settings. A frozen case is used
exactly as saved, and no other source applies: `--case` together with `--config` or
`--set` is refused with exit code `2`, and `exact=True` refuses `settings`.

To make a variant of a frozen case, give it as the preset and add settings, then save the
result as a new frozen case (in a new, empty folder, as on every page of this guide):

```bash
pydac validate --set grid.nz=25 --set grid.nr=1 --set execution.max_cycles=3 --set execution.require_css=false --export-case my_case.json
pydac validate --config my_case.json --set steps.ads.horizon.nominal_time_s=5400 --export-case my_case_long_ads.json
```

A complete case given as a preset, with no settings, is the same case as the frozen one:

```bash
pydac validate --config my_case.json --export-case my_case_copy.json
cmp my_case.json my_case_copy.json
```

## Writing settings

A setting is a dotted path and a JSON value:

```bash
pydac validate --set grid.nz=40 --set steps.s_des.flow.steam_mass_flow_kg_s=0.002 --set execution.plot_profiles=true --validate
```

- Paths are the ones `pydac validate --show-settings` lists, plus the keys already inside
  a listed block: a parameter of an adsorption model, or a field of the bed's architecture
  such as `bed.design.wall_htc_model`. The prefix `simulation.` is optional: `grid.nz` and
  `simulation.grid.nz` are the same path. For step fields, `cycle.` is an alias of
  `steps.` (`cycle.ads.horizon.nominal_time_s`); it is not a prefix of other paths.
- Step fields are `steps.STEP.GROUP.FIELD`, with the step names `startup`, `ads`, `blow`,
  `heat`, `w_purge`, `s_purge`, `s_des`, `w_des`, `cool`, `press`.
- Values are JSON: numbers, `true`/`false`, `null`, strings, lists and objects. A bare
  word that is not JSON, such as `length_radius`, is taken as a string. Quote a value with
  brackets or braces for the shell:
  `--set 'execution.cycle_steps=["ads","blow","heat","s_des","cool","press"]'`.
- Fields marked "replace as one block" in `--show-settings` (the adsorption models) take
  a whole JSON object, which replaces the block: use it to choose another model. One
  existing key inside the block can also be set alone, for example
  `adsorption.kinetics.CO2.parameters.k_LDF_s_inv=0.004`.
- In Python, `settings` is a mapping with the same paths and Python values, or a list of
  `"PATH=VALUE"` strings exactly as `--set` takes them.

A path that does not exist is refused, with exit code `2`:

```bash
pydac validate --set grid.nz=40 --set grid.nzz=40 --validate || echo "refused with exit code $?"
```

## What you can change without Python

Every input of the model is a field of the case, so a case JSON, a preset or `--set`
reaches all of them:

- the grid and solver tolerances (`grid.*`, `solver.*`);
- the column, wall and sorbent (`geometry.*`, `wall.*`, `sorbent.*`, `bed`);
- the gas mixture (`mixture.species`, `mixture.closure_species`, `mixture.adsorbates`);
- the adsorption models (`adsorption.equilibrium_systems`, `adsorption.kinetics`);
- every field of every step: composition, flow, temperatures, pressures, duration, stop
  rule and outlet (`steps.STEP.*`);
- the run options (`execution.*`).

Models are chosen by name among the ones pyDAC ships:

| Choice | Field | Names |
|---|---|---|
| Isotherm | `adsorption.equilibrium_systems` | `Toth`, `GAB`, `TothGABWADST` (default), `CompetitiveSips` |
| Uptake kinetics | `adsorption.kinetics` | `LDF` |
| Solid heat capacity | `sorbent.heat_capacity` | `Low2023` (default), `constant` |
| Bed flow | `bed.design.flow_model` | `uniform` (default), `wall-channeling-reduced` (2D only) |
| Radial voidage, with channeling | `bed.design.channeling.porosity_model` | `de_klerk`, `giese`, `tabulated` |
| Stagnant bed conductivity | `bed.design.stagnant_conductivity_model` | `zehner_schlunder` (default), `specchia_1980` |
| Bed-to-wall heat transfer | `bed.design.wall_htc_model` | `dixon` (default), `specchia_1980` |

A wrong name is refused with the list of available ones. The reference chapters give each
model's equations and source ([04 — Governing equations](../manual/04_governing_equations.md)).
You can add your own isotherms and kinetic laws from Python, without editing pyDAC
([Extending pyDAC](08_extending.md), [tutorial 07](../../examples/notebooks/07_new_isotherm.ipynb)).
Adding a new species or another kind of model is code work in a clone of the repository
([CONTRIBUTING](../../CONTRIBUTING.md)).

### Example: a dry CO2/N2 case

This preset replaces the default humid air (CO2, H2O, O2, N2 with CO2 and H2O adsorbing)
by a dry CO2/N2 mixture in which only CO2 adsorbs, with the Toth isotherm. Save it as
`dry_co2_n2.json`:

```json
{
  "simulation": {
    "grid": {"nz": 25, "nr": 1},
    "mixture": {
      "species": ["CO2", "N2"],
      "closure_species": "N2",
      "adsorbates": ["CO2"]
    },
    "adsorption": {
      "equilibrium_systems": {
        "Toth": {"datasets": {"Toth": "Cleaned-Toth"}, "parameters": {}}
      },
      "kinetics": {
        "CO2": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.003}}
      }
    }
  },
  "execution": {"cycle_steps": ["startup", "blow"]}
}
```

Run its startup and blowdown (`blow`) steps:

```bash
pydac steps --config dry_co2_n2.json
```

A reduced mixture changes what the steps and the KPIs can report:

- Without H2O, the steam steps (`s_purge`, `s_des`) cannot run: a run that lists them is
  refused before it starts. Heating and wall purge report no steam condensation
  temperature, and a collected step has nothing to condense: its outlet is the product as
  it leaves the bed, without condenser or vacuum-discharge flash (the run says so).
- Without O2 or N2, the species present pass the separators in their place.
- Without CO2 there is no product: a run whose listed steps collect product is refused.
- KPIs that name an absent species: without H2O the injected steam is `0` and the
  steam/CO2 ratio is `None` (as in any cycle without steam); the H2O/CO2 ratios and an
  unavailable oxygen content are `NaN`. The KPI figures that need an absent species are
  skipped with a printed note.

## Column size

The column size is given in one of two ways, chosen by `geometry.specified_by`:

| `specified_by` | You give | pyDAC computes |
|---|---|---|
| `volume_aspect_ratio` (default) | `volume_m3` and `aspect_ratio` (length over diameter, L/D) | `length_m` and `radius_m` |
| `length_radius` | `length_m` and `radius_m` | `volume_m3` and `aspect_ratio` |

```bash
pydac validate --set geometry.specified_by=length_radius --set geometry.length_m=0.2 --set geometry.radius_m=0.05 --print-resolved | grep -A6 '"geometry"'
```

A case states the size one way only. A value written in a computed field must equal the
computed one, otherwise the case is refused; a frozen case stores all four values, which
agree, and loads unchanged. When a setting changes an input (for example the aspect ratio
of a frozen case), pyDAC computes the other pair again from the new input.

## `null` and omitted fields

Leaving a field out of a preset always means "use the built-in default". Writing `null`
is a different statement, allowed only where the field accepts it:

- **Step fields:** `null` means "inherit": the field keeps the value it has from the layer
  below (the built-in default, or the preset), exactly as if it were left out. It does not
  mean "take the previous step's state": every step after `startup` starts from the state
  the previous step left anyway, and `startup` starts from its own initial values
  (`initial_temperature_K`, `initial_pressure_Pa`, initial mole fractions).
  `horizon.event_max_time_s = null` therefore keeps the current limit (43200 s for
  `startup` by default); it neither removes the limit nor shortens it. To stop an
  event-mode step no later than its nominal time, set `event_max_time_s` to that time.
- **The startup solid:** `steps.startup.composition.initial_solid_loadings_mol_per_kg` is
  `null` by default, and the startup solid is then in equilibrium with the startup initial
  gas. To start from another solid, give each adsorbate its own loading in mol/kg, for
  example `{"CO2": 0.0, "N2": 0.357}` (CO2 starts at 0, N2 at 0.357, the same in the whole
  bed; `co2` is the same name as `CO2`). Every adsorbate needs its value: a mapping that
  leaves one out or names one twice is refused, and so is the field on any other step
  ([manual 01 §4.2](../manual/01_inputs.md#42-a-step-ten-names-seven-groups-fixed-order)).
  Like every step field, a `null` setting inherits: it does not undo a mapping that the
  preset states. Remove it from the preset instead.
- **Switches:** `null` turns the feature off, for example `execution.extra_tag`,
  `execution.cycle_steps` (the default cycle), `wall.overall_htc_override_W_m2K` or
  `bed.material`.
- **The bed block:** `bed = null` stands for the built-in packed bed.
- **Computed geometry:** the pair of the column size that `geometry.specified_by` does
  not name is computed; leave it `null` ([Column size](#column-size)).
- **Anywhere else** `null` is refused.

`--show-settings` marks the fields that accept `null` as "nullable".

For NSGA-II campaigns this matters at startup: if the base case's
`steps.startup.horizon.event_max_time_s` is shorter than the adsorption times the search
explores, every candidate is invalid. `pydac nsga2` checks both corners of the bounds
before it starts and refuses such a campaign.

## Options for single steps

`execution.step_options` changes how individual steps run, without touching their
physics. It is one JSON object keyed by step name, set as a whole:

```bash
pydac validate --set 'execution.step_options={"cool":{"termination_mode":"time"},"ads":{"plot_profiles":true}}' --validate
```

The keys inside each step are `solve`, `plot_profiles`, `use_sparsity`,
`compute_numerical_sparsity` and `termination_mode` (`"time"` or `"event"`). The built-in
`execution.step_options` is empty, so a dotted path into it, such as
`execution.step_options.cool.solve`, is refused as an unknown setting: set the whole object.
A dotted path works only for a key the preset already has.

## The outlet of a step

Each step has an `outlet` group that decides where its outlet stream goes:
`collect_product` (collected as product, or vented), `condenser` and
`condenser_temperature_K`. By default the steam purge and both desorption steps are
collected through a condenser at 278.15 K; the other steps are vented. For example, to vent
the steam purge instead of collecting it:

```bash
pydac validate --set steps.s_purge.outlet.collect_product=false --set steps.s_purge.outlet.condenser=false --validate
```

The rules and their effect on the KPIs are in [01 — Inputs](../manual/01_inputs.md) and
[06 — Post-processing](../manual/06_post_processing.md).

## Fingerprint and run folder

The fingerprint is a hash of the complete resolved case. Its first eight characters name
the run folder, for example `simulation_results/cyclic_mode/packed_bed_7a4930a5` for the
quick-start case of [Quick start](02_quickstart.md).

- Every field counts, including `execution.output_root` and `execution.extra_tag`:
  changing the output folder or the tag gives another fingerprint and another run folder.
  `execution.extra_tag` also prefixes the folder name (`TAG_packed_bed_...`, and
  `TAG_1D_25z_packed_bed_...` for a campaign). So it must be one folder name: at most 64
  letters, digits, `_`, `-` or `.` (a space becomes `_`), not starting with `.`.
- A frozen case keeps the values stored in it, so its fingerprint is the same on every
  machine. A case built from a preset recomputes derived values (for example the steam
  volumetric flow from the steam mass flow) with the local numerical libraries, and can end
  up with a slightly different fingerprint elsewhere. To share a case or rerun it on
  another machine, use its frozen case: `--export-case`, `pydac.export_case`, or the
  `resolved_config.json` saved in every run folder.
- A run refuses a run folder that already holds results: continue the run with
  `--resume`, replace it with `--overwrite`, or choose another `execution.extra_tag`.

## Editing a case with schema support

The case format is described by a JSON schema, which lets an editor complete field names
and flag wrong types. Every installation contains it:

```bash
python -c "import pydac, pathlib; print(pathlib.Path(pydac.__file__).parent / 'config' / 'json_schema' / 'experiment.schema.json')"
```

From pyDAC 3.0.0 on, the schema of each release is also published at its release tag:
`https://raw.githubusercontent.com/process-intensification-lab/pyDAC/v3.0.0/src/pydac/config/json_schema/experiment.schema.json`.
Point your editor's JSON schema setting for your case files at either location. Do not
add a `$schema` key to a case: the case format has no such field, and the case is refused.

## Related pages

- [Quick start](02_quickstart.md)
- [Installation](01_installation.md)
- [01 — Inputs](../manual/01_inputs.md): every field and the case document in detail
