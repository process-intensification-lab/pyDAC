---
title: "Extending pyDAC"
toc: true
toc-depth: 3
embed-resources: true
---

> **New to pyDAC? Do the [tutorials](../../examples/README.md) first, starting with
> [01 First steps](../../examples/notebooks/01_first_steps.ipynb).** They teach pyDAC step by
> step on real runs; this guide is the reference behind them and goes into more technical detail.

A case chooses its isotherms and kinetic laws by name, from the models pyDAC knows. You can add
your own **equilibrium systems** (isotherms) and **kinetic laws** from your own Python code,
without editing pyDAC: register them, then select them by name in a case like a built-in
model. A sorbent of your own is then one JSON file with its properties and its models
([A sorbent of your own](#a-sorbent-of-your-own)).
[Tutorial 07](../../examples/notebooks/07_new_isotherm.ipynb) does it end to end for a
sorbent from the literature (Stampi-Bombelli et al., 2020), with a CO2/H2O co-adsorption isotherm.

What else you can change, and where:

| You want | How |
|---|---|
| other sorbent properties: density, heat capacity, heats of adsorption, a fixed adsorbate | inputs of the case, no code ([Cases and settings](03_cases.md), [tutorial 02](../../examples/notebooks/02_custom_case.ipynb); [tutorial 06](../../examples/notebooks/06_breakthrough_validation.ipynb) supplies a fixed H2O adsorbate) |
| another gas mixture made of known species | inputs of the case: `simulation.mixture` ([tutorial 02](../../examples/notebooks/02_custom_case.ipynb)) |
| your own isotherm or kinetic law | register it, this page |
| your own bed architecture (a monolith, a structured packing) with its closures | register it: [A bed architecture](#a-bed-architecture) |
| a new species, a new heat-capacity model, new equations | code in a clone of pyDAC ([CONTRIBUTING](../../CONTRIBUTING.md#extending-the-model), [01 — Inputs §10](../manual/01_inputs.md#10-extending-the-contract)) |

## An equilibrium system

An equilibrium system gives the equilibrium loading of the adsorbates it owns, by species name:

```python
import numpy as np
import pydac

R = 8.314462618  # J/(mol K)


class Langmuir(pydac.EquilibriumSystem):
    """CO2 Langmuir isotherm with a temperature-dependent affinity."""

    species = ("CO2",)  # the adsorbates this system owns

    def __init__(self, q_sat_mol_per_kg, b0_inverse_Pa, heat_J_per_mol):
        self.q_sat = q_sat_mol_per_kg
        self.b0 = b0_inverse_Pa
        self.heat = heat_J_per_mol

    def equilibrium_loadings(self, *, partial_pressures_Pa, temperature_K,
                             current_loadings_mol_per_kg):
        b = self.b0 * np.exp(self.heat / (R * temperature_K))
        p = partial_pressures_Pa["CO2"]
        return {"CO2": self.q_sat * b * p / (1 + b * p)}
```

- Partial pressures are in Pa, the temperature in K, loadings in mol per kg of sorbent. The
  model calls the system with NumPy arrays, one value per cell: write the equation with
  NumPy operations, not `math` or `if`.
- `current_loadings_mol_per_kg` holds the loadings of every adsorbate, for systems in which
  one species changes the equilibrium of another (co-adsorption).
- Return exactly the species in `species`, in that order.
- One system may own several adsorbates whose equilibria depend on each other, for example
  CO2 and H2O on a sorbent where the adsorbed water changes the CO2 capacity: list both in
  `species`, and H2O in `required_current_loadings` (below), as the built-in `TothGABWADST`
  does.

## Register it

The registry entry says what the system needs and how to build it from the case:

```python
pydac.register_equilibrium("Langmuir", pydac.EquilibriumRegistryEntry(
    adsorbates=("CO2",),
    required_gas_species=("CO2",),
    required_current_loadings=(),
    build=lambda datasets, parameters: Langmuir(**parameters),
    required_parameters=("q_sat_mol_per_kg", "b0_inverse_Pa", "heat_J_per_mol"),
))
```

| Field | Meaning |
|---|---|
| `adsorbates` | the species whose loading it gives |
| `required_gas_species` | the species that must be in the mixture |
| `required_current_loadings` | the loadings it reads from `current_loadings_mol_per_kg` |
| `build` | `build(datasets, parameters)` returns the system; both come from the case |
| `required_parameters`, `optional_parameters` | the parameter names the case must, or may, give; any other name is refused |
| `required_datasets`, `optional_datasets` | the same for named datasets |

A built-in name (`Toth`, `GAB`, `TothGABWADST`, `CompetitiveSips`) is refused. Registering a
name again replaces it, so a notebook cell can be run again after a change.

## Select it in a case

`simulation.adsorption.equilibrium_systems` names the systems of the case, each with its
datasets and parameters. It is replaced as a whole, so give every system the case needs:

```python
case = pydac.load_case(settings={
    "adsorption.equilibrium_systems": {
        "Langmuir": {"datasets": {}, "parameters": {
            "q_sat_mol_per_kg": 1.25, "b0_inverse_Pa": 3.8e-14, "heat_J_per_mol": 70300.0}},
        "GAB": {"datasets": {}, "parameters": {}},
    },
})
```

The systems together must cover the adsorbates of `simulation.mixture` exactly once: here
`Langmuir` covers CO2 and the built-in `GAB` covers H2O. Name the parameters with their units,
as the built-in models do (`k_LDF_s_inv`).

The relative humidity and the steam use water's vapor pressure. It is pyDAC's IAPWS-1995
correlation, the one `GAB` uses, unless the system that owns H2O gives its own: an object
with a method `EOS(T)` (Pa, from K) in its `component_models["H2O"]`.

When the case is loaded, pyDAC checks the names, the coverage of the adsorbates and the
parameter names. It does not check your equation: test it on its own first, for example
against the measured loadings of the paper you take the parameters from.

## A kinetic law

A kinetic law gives the loading rate of one or more adsorbates, in mol/(kg s):

```python
class HalfLDF(pydac.KineticLaw):
    species = ("CO2",)

    def __init__(self, k_s_inv):
        self.k = k_s_inv

    def loading_rates_mol_per_kg_s(self, *, partial_pressures_Pa, temperature_K,
                                   current_loadings_mol_per_kg,
                                   equilibrium_loadings_mol_per_kg):
        driving = equilibrium_loadings_mol_per_kg["CO2"] - current_loadings_mol_per_kg["CO2"]
        return {"CO2": 0.5 * self.k * driving}


pydac.register_kinetics("HalfLDF", pydac.KineticRegistryEntry(
    build=lambda species, parameters: HalfLDF(parameters["k_s_inv"]),
    supported_species=("CO2",),
    required_parameters=("k_s_inv",),
))
```

`build(species, parameters)` is called once per adsorbate that selects the law. Select it in
`simulation.adsorption.kinetics`, which is also replaced as a whole:

```python
settings = {"adsorption.kinetics": {
    "CO2": {"model": "HalfLDF", "parameters": {"k_s_inv": 0.003}},
    "H2O": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.00083}},
}}
```

## A sorbent of your own

A sorbent is two blocks of a case: `simulation.sorbent`, its properties, and
`simulation.adsorption`, its isotherms and kinetic laws with their parameters. Keep them in
one file, for example `my_mof.json`:

```json
{
  "sorbent": {
    "density_kg_m3": 650.0,
    "particle_radius_m": 0.0005,
    "conductivity_W_mK": 0.1,
    "heat_capacity": {"model": "constant", "parameters": {"value_J_kgK": 900.0}},
    "heats_of_adsorption_J_mol": {"CO2": -50000.0, "H2O": -48000.0}
  },
  "adsorption": {
    "equilibrium_systems": {
      "MyMOF": {"datasets": {}, "parameters": {"q_sat_mol_per_kg": 2.1}}
    },
    "kinetics": {
      "CO2": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.01}},
      "H2O": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.005}}
    }
  }
}
```

(The numbers only show the format; [`07_apdes_nfc_sorbent.json`](../../examples/cases/07_apdes_nfc_sorbent.json)
of [tutorial 07](../../examples/notebooks/07_new_isotherm.ipynb) is a worked sorbent file, from Stampi-Bombelli et al., 2020;
it leaves the conductivity and the pore radius at the case defaults.) Then put it on top of any
case, after registering your models:

```python
import json

with open("my_mof.json") as handle:
    mof = json.load(handle)
case = pydac.load_case("my_cycle.json", settings=mof)
```

The `sorbent` fields you give replace the default ones and the others keep their values; the
`equilibrium_systems` and `kinetics` mappings are replaced as a whole. The same file on top of
another case compares cycles on one sorbent; another file on top of the same case compares
sorbents. The heats of adsorption enter the energy balance: give values consistent with your
isotherm. The frozen case of a run holds every number of the file, so the run can be
reproduced without it.

From the terminal, write the two blocks in the case file itself, as
[tutorial 02](../../examples/notebooks/02_custom_case.ipynb) does for zeolite 13X; this works
only with the built-in models (below).

## A bed architecture

The built-in bed is `packed_bed`. A bed of another kind (a monolith, a structured packing) brings its
own structure and its own closures: pressure drop, dispersion and heat transport. Register it
under a name, then select the name in `simulation.bed.architecture`. The packed bed stays the
default and is not changed by it.

The architecture is a `pydac.BedArchitectureEntry` of callables plus some fixed text:

| Field | What it is |
|---|---|
| `spatial_modes` | where its closures hold: `("1d",)`, `("2d",)` or `("1d", "2d")`; a case on another grid is refused |
| `build_design(design, material)` | checks the case's `bed.design` and `bed.material` mappings (exact keys, physical ranges) and returns your design object; raise `ValueError` on a bad value |
| `to_config(design_obj)` | the canonical `(design, material)` mappings the case and its fingerprint record: flat, lowercase keys, finite numbers (`material` may be `None`); feeding them to `build_design` again must give the same mappings |
| `resolve_structure(design_obj, particle)` | a `ResolvedStructure`: flow and total porosity, sorbent density in the bed, heat capacity of the structure (J/m3/K) |
| `resolve_pressure_drop(design_obj, inputs)` | the pressure-drop law (below) |
| `resolve_dispersion(design_obj, inputs)` | a mapping with exactly `model`, `Disp_ax`, `Disp_rad` (m2/s) and `inv_Pe_m_ax`, `inv_Pe_m_rad`, one value per gas species; any other key is refused (provenance text goes in `provenance`) |
| `resolve_thermal_transport(design_obj, inputs)` | a `ResolvedTransport(lambda_ax, lambda_rad, h_wall, wall_bed_htc, pressure_drop_model, dispersion_model, diagnostics={})`: axial and radial effective conductivities, bed-side wall coefficient `h_wall`, the coefficient `wall_bed_htc` that couples bed and wall (below), the names of your pressure-drop and dispersion laws, and optional finite numbers for your own records |
| `wall_htc_model` | the name of your bed-to-wall law, written in the results as `bed.models.wall_htc` (and, in 1D, as `bed.models.overall_wall_htc`: your `wall_bed_htc` is the overall coefficient there) |
| `provenance` | fixed text, for example where the closures come from; written in the results |
| `context_parameters` | the rows your design, material and provenance add to the results' `SimulationContext` |

`inputs` is a `pydac.ClosureInputs`: the values pyDAC already resolved for the step, the same
ones the packed bed gets, by name. `eps_flow`, `viscosity` (Pa s), `molecular_diffusivity`
(m2/s) and `schmidt` (one value per gas species, in the order of `gas_species`), `reynolds`,
`prandtl`, `gas_conductivity` and `particle_conductivity` (W/m/K),
`superficial_inlet_velocity` (m/s), `particle_diameter`, `bed_length`, `bed_radius` and
`bed_diameter` (m), the reference velocity and pressure used to make the equations
dimensionless, and `is_1d`. The arrays are copies of the model's, flagged read-only: compute
new ones, do not change them.

What the column does with your results:

- **Pressure drop.** The law has a `name` and two coefficients, `linear_coefficient` (A) and
  `inertial_coefficient_per_density` (B), both scaled as the Ergun law of the packed bed
  (`pydac.core.bed_architectures.ErgunPressureDrop` shows the scales). Its
  `velocity_from_gradient(grad, rho)` returns the dimensionless velocity v solving
  `A*v + B*rho*v*|v| = -grad`, for both flow directions; `solve_quadratic_drag_velocity` does
  it for you (`pydac.solve_quadratic_drag_velocity`). Only this quadratic form, with scalar
  coefficients, is supported: pyDAC checks that your law agrees with its two coefficients, then
  the column and the reports use the quadratic law of those two numbers.
- **Heat.** The axial and radial Péclet numbers come from your two conductivities. In 1D,
  `wall_bed_htc` is the overall bed-to-wall coefficient the energy balance uses; in 2D it must
  equal `h_wall`, which sets the Robin condition at the wall. The wall heat-transfer overrides
  of the case (`wall.overall_htc_override_W_m2K`, `wall.bed_wall_htc_override_W_m2K`) replace
  packed-bed correlations and are refused for a registered architecture: your coefficient is
  the physics you are studying.
- **Structure.** Your bed is radially uniform: the radial porosity profiles and wall
  channeling of the packed bed do not apply to it.

The structure and closure results are checked before the column uses them and copied into
pyDAC's own values. The model keeps your design object and calls your functions with it; an
error names the architecture and the callable, with your original error as its cause. Units cannot be checked: give SI values as above. The checks
refuse:

- a name, a design or material key, or a provenance key that is not one lowercase token
  (`[a-z][a-z0-9_]*`); empty `wall_htc_model`, provenance text or model names;
- design and material values that are not finite numbers (`True` and `False` are not
  numbers), or a `to_config` whose mappings, fed back to `build_design`, do not give the same
  keys in the same order with the same values;
- a structure with `eps_flow` or `eps_total` outside (0, 1), `eps_flow` above `eps_total`,
  `rho_sorbent` not positive, or `acc_structure` negative (0 is allowed);
- pressure-drop coefficients A or B that are not positive finite scalars;
- dispersion arrays (all four, in 1D too) that are not positive and finite, one value per gas
  species;
- conductivities `lambda_ax` and `lambda_rad` (W/m/K; `lambda_rad` is required in 1D too)
  and coefficients `h_wall` and `wall_bed_htc` (W/m2/K) that are not positive and finite;
- context rows whose ID is not `bed.design.<key>`, `bed.material.<key>` or
  `bed.provenance.<key>`, whose category or value type does not match that ID, with an empty
  unit or a repeated ID, a provenance row without its text, or a design or material key
  without its row.

`ResolvedStructure`, `ResolvedTransport` and `ContextParameter` are importable from `pydac`.
pyDAC keeps a copy of the entry's fixed text and rows when you register it, but it calls your
functions themselves: state that they keep (a global, an attribute of a callable object) is your
code, and changing it changes the next models built.

### Results of a registered architecture

The `SimulationContext` of the results gets one row per entry of `context_parameters`, a
`pydac.ContextParameter` whose ID says where its value comes from:

- `bed.design.<key>` and `bed.material.<key>`: the numbers of the canonical mappings, with
  their unit; every key of the mappings needs its row;
- `bed.provenance.<key>`: a text of `provenance`, unit `-`.

pyDAC also writes the row `bed.registered_architecture`, which repeats the architecture's name.
Saved results can be read, plotted and compared in a session that has not registered the
architecture: the reader checks these rows by their form only, and reads a name that is not
built in only when that row repeats it, so a misspelled name is refused.

### An example

A packed bed with its own name, built from the packed-bed closures, shows every field (pyDAC's
tests run this clone and check that its results equal the built-in packed bed bit for bit). Only
because it is a clone does it import the packed-bed internals; your bed brings its own closures:

```python
import pydac
from pydac import ContextParameter
from pydac.core.bed_architectures import (
    PackedBedDesign, resolve_packed_bed_structure, resolve_packed_bed_pressure_drop,
    resolve_packed_bed_dispersion, resolve_packed_bed_thermal_transport,
)


def build_design(design, material):
    if material is not None or set(design) != {"eps_b"}:
        raise ValueError("my_bed takes design.eps_b and no material")
    return PackedBedDesign(eps_b=design["eps_b"])


def pressure_drop(design, x):
    return resolve_packed_bed_pressure_drop(
        eps_flow=x.eps_flow, viscosity=x.viscosity, particle_diameter=x.particle_diameter,
        bed_length=x.bed_length, reference_velocity=x.reference_velocity,
        reference_pressure=x.reference_pressure)


def dispersion(design, x):
    return resolve_packed_bed_dispersion(
        eps_flow=x.eps_flow, molecular_diffusivity=x.molecular_diffusivity,
        reynolds=x.reynolds, schmidt=x.schmidt,
        superficial_inlet_velocity=x.superficial_inlet_velocity,
        particle_diameter=x.particle_diameter, reference_velocity=x.reference_velocity,
        bed_length=x.bed_length, bed_radius=x.bed_radius)


def thermal(design, x):
    return resolve_packed_bed_thermal_transport(
        stagnant_conductivity_model=design.stagnant_conductivity_model,
        wall_htc_model=design.wall_htc_model, eps_flow=x.eps_flow,
        gas_conductivity=x.gas_conductivity, particle_conductivity=x.particle_conductivity,
        reynolds=x.reynolds, prandtl=x.prandtl, particle_diameter=x.particle_diameter,
        bed_diameter=x.bed_diameter, reference_velocity=x.reference_velocity,
        bed_length=x.bed_length, is_1d=x.is_1d)


pydac.register_bed_architecture("my_bed", pydac.BedArchitectureEntry(
    spatial_modes=("1d", "2d"),
    build_design=build_design,
    to_config=lambda design: ({"eps_b": design.eps_b}, None),
    resolve_structure=resolve_packed_bed_structure,
    resolve_pressure_drop=pressure_drop,
    resolve_dispersion=dispersion,
    resolve_thermal_transport=thermal,
    wall_htc_model="dixon",
    provenance={"source": "packed-bed closures under another name"},
    context_parameters=(
        ContextParameter("design", "bed.design.eps_b", "-"),
        ContextParameter("provenance", "bed.provenance.source", "-", "text"),
    ),
))

case = pydac.load_case("my_case.json", settings={
    "bed": {"architecture": "my_bed", "design": {"eps_b": 0.38}, "material": None},
})
```

The name is one lowercase word (`my_bed`, `my_monolith`); `packed_bed` is refused. Registering a
name again replaces it for the models built afterwards. The case file's JSON schema lists the
built-in architecture only, so an editor flags the name; pyDAC accepts it once it is
registered.

### Your own bed: a monolith

The clone above reuses the packed-bed closures. A bed of your own replaces each callable with
its physics. For a monolith (parallel straight channels, the sorbent in or on the channel
walls), this is what each field has to give; the correlations are yours to choose and cite in
`provenance`:

- **Spatial modes.** Declare `spatial_modes=("1d",)`. In 2D the column moves gas radially
  with the same drag law as axially, so gas would cross between channels that in a monolith
  are closed to each other; a small radial dispersion does not stop that flow.
- **Design.** `bed.design` holds the geometry, for example the channel width, the wall
  thickness and the channel shape; `bed.material` the wall material (density, heat capacity,
  conductivity). `build_design` checks the keys and their ranges; `to_config` returns them as
  flat lowercase keys with finite numbers, one `ContextParameter` row each.
- **Structure.** `eps_flow` is the open frontal area, the fraction of the cross-section the
  channels take; `eps_total` adds the pore volume of the walls; `rho_sorbent` is the sorbent
  mass per bed volume. The column already counts the heat capacity of that sorbent
  (`rho_sorbent` times the sorbent's heat capacity): `acc_structure` is only the heat
  capacity per bed volume (J/m3/K) of a support that is not sorbent, for example a ceramic
  or metal wall under a coating. For a wall made of the sorbent itself, it is 0.
- **Pressure drop.** Laminar flow in the channels gives the linear term A, from the
  friction factor of the channel shape (Hagen-Poiseuille; for a square channel, the
  pressure drop per length -dp/dz = 28.45 μ u / (eps_flow d_h²), with u the superficial
  velocity, u / eps_flow the channel velocity and d_h the hydraulic diameter), scaled as
  `ErgunPressureDrop` scales its own. B must be positive: give the inertial term of the
  entrance and exit losses, small next to A.
- **Dispersion.** The column takes the dispersion coefficient per bed cross-section, as the
  packed-bed correlation gives it. The Taylor-Aris coefficient of laminar channel flow is per
  channel cross-section (for a circular channel of diameter d, D_ch = D_m + u_ch² d² /
  (192 D_m), u_ch the channel velocity; other shapes have their own constant): return
  `Disp_ax` = eps_flow × D_ch and `inv_Pe_m_ax` = `Disp_ax` / (reference velocity × bed
  length), as `resolve_packed_bed_dispersion` does. A 1D run does not use the radial values,
  but they must still be positive: give the same form with D_m.
- **Heat.** The continuous walls carry heat: the axial conductivity is about the wall
  conductivity times the solid fraction; give `lambda_rad` too (it is required in 1D as
  well), the conductivity across the wall network. In 1D, `wall_bed_htc` is the overall coefficient
  from the monolith to the tube wall: the contact between them, often a gap resistance, in
  series with the radial conduction through the wall network (from a cell model of your
  channel shape). Name its law in `wall_htc_model`.
- **Sorbent.** The kinetic law and its parameters stay those of the case: a linear driving
  force coefficient for a coated wall comes from the wall or coating thickness, not from a
  particle diameter.

Check the closures on their own before a run (a pressure drop against a measured one, a
conductivity against your cell model), then run a short case. The results record the name,
the design and material rows and the provenance text, so another reader knows which
monolith you ran.

## Limits

- **One Python session.** A registered model exists in the session that registered it. A case
  that names it is refused (`Unknown model`) anywhere else: run the registration code again
  before `pydac.load_case` or `pydac.resume` in a new session. `pydac.report` and
  `CycleDataLoader` read only the saved results and need no registration (a `report` that
  redraws the step plots with `snapshot_times` loads the case again, and needs it).
- **Not from the terminal.** The `pydac` command knows only the built-in models; run cases
  with your models from Python.
- **NSGA-II workers.** With `workers` above 1, the candidates see your models only when the
  worker processes are forked from your session: on Linux with Python 3.13 or older. On macOS
  and Windows, and from Python 3.14 on Linux, run campaigns with your models with one worker.
- **The case names the model, not its code.** The frozen case, its fingerprint and the run
  folder record the model's name and parameters; they do not change when you change the code
  behind the name. Give a changed model a new name, and keep its code with your results.
- The same limits hold for a registered bed architecture. Saved results of one are read
  without the registration; `resume` and a `report` with `snapshot_times` rebuild the model
  and need it.

To make a model part of pyDAC, so that every user, the terminal and every worker know it,
add it to the package: [CONTRIBUTING](../../CONTRIBUTING.md#extending-the-model).

## Related pages

- [Cases and settings](03_cases.md): replacing a mapping such as `equilibrium_systems`
- [01 — Inputs §10](../manual/01_inputs.md#10-extending-the-contract): the contract each
  addition to pyDAC meets
- [Reproducibility](07_reproducibility.md): what a frozen case records
