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
| a new species, a new heat-capacity model, a new bed architecture, new equations | code in a clone of pyDAC ([CONTRIBUTING](../../CONTRIBUTING.md#extending-the-model), [01 — Inputs §10](../manual/01_inputs.md#10-extending-the-contract)) |

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

To make a model part of pyDAC, so that every user, the terminal and every worker know it,
add it to the package: [CONTRIBUTING](../../CONTRIBUTING.md#extending-the-model).

## Related pages

- [Cases and settings](03_cases.md): replacing a mapping such as `equilibrium_systems`
- [01 — Inputs §10](../manual/01_inputs.md#10-extending-the-contract): the contract each
  addition to pyDAC meets
- [Reproducibility](07_reproducibility.md): what a frozen case records
