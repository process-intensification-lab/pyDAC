---
title: "04 — Governing Equations: what pyDAC solves, and where"
toc: true
toc-depth: 3
embed-resources: true
---

> **This manual is the detailed technical reference: how pyDAC works inside, for contributors
> and for checking the model.** To learn to use pyDAC, do the [tutorials](../../examples/README.md)
> first, then read the [user guide](../guide/01_installation.md).

## 1. Scope and ownership

This chapter maps the integrated balances, transport closures, bed-wall coupling,
equilibrium and kinetics to their implementation. It documents the configuration
and runtime quantities needed to interpret a simulation.

For discretisation see [03 — pymrm discretisation](03_pymrm_discretization.md);
for boundary conditions see [05 — Cycle steps](05_cycle_steps.md); for model
construction see [02 — pyDAC core](02_pydac_core.md).

## 2. What is actually integrated

The solver advances the named `StateLayout` in nondimensional time. Independent gas
concentrations, adsorbate loadings, total concentration, bed temperature, wall temperature
and integrated mass/energy states are generated from the declared mixture and adsorbates.
Offsets are shared by initialization, RHS, solver, sparsity and saved-solution readers;
there is no pending fixed-layout migration.

For the default DAC mixture the independent gases are CO2, H2O and O2, the closure
species is N2, and the adsorbates are CO2 and H2O. Other declared mixtures may include
N2 adsorption. Closure concentration equals total concentration minus the independent
species. `ColumnModelMoL.pde_system_kpis` assembles the balances.

---

## 3. Equation → code map

### 3.1 Balances

| Equation | Assembled in |
|---|---|
| non-dim total mole balance | [`pde_system_kpis()`](../../src/pydac/core/column_model.py), stage 3 |
| non-dim species balance | [`pde_system_kpis()`](../../src/pydac/core/column_model.py), stage 4, with [`_corrected_species_dispersion_flux()`](../../src/pydac/core/column_model.py) |
| LDF solid-phase balance, with the low-pressure damping | [`_evaluate_adsorption_rates()`](../../src/pydac/core/column_model.py) → [`LDFRate.loading_rates_mol_per_kg_s()`](../../src/pydac/core/kinetic_models.py) |
| inversion of the pressure-drop law for the velocity | [`update_velocity_fields()`](../../src/pydac/core/column_model.py) → [`ErgunPressureDrop.velocity_from_gradient()`](../../src/pydac/core/bed_architectures/packed_bed.py), [`solve_quadratic_drag_velocity()`](../../src/pydac/core/bed_architectures/packed_bed.py) |
| 2D bed↔wall Robin coupling | [`pde_system_kpis()`](../../src/pydac/core/column_model.py), stage 5 |
| 1D bed↔wall source term | [`calc_nondim_pde_params()`](../../src/pydac/core/column_model.py) (`omega_wall`), [`pde_system_kpis()`](../../src/pydac/core/column_model.py), stage 5 |
| Ergun momentum closure | [`ErgunPressureDrop`](../../src/pydac/core/bed_architectures/packed_bed.py), [`RadialErgunPressureDrop`](../../src/pydac/core/bed_architectures/packed_bed.py), resolved in [`_resolve_packed_bed_pressure_drop()`](../../src/pydac/core/column_model.py) |
| non-dim energy balance | [`pde_system_kpis()`](../../src/pydac/core/column_model.py), stage 5 |
| non-dim wall balance | [`pde_system_kpis()`](../../src/pydac/core/column_model.py), stage 6 |

### 3.2 Dimensionless groups

All of them are computed once per step. The capacity, heat-of-adsorption, pressure and wall
groups are assembled in [`calc_nondim_pde_params()`](../../src/pydac/core/column_model.py); the material and thermal Péclet
groups come from the bed-architecture closures, resolved in
[`_resolve_packed_bed_dispersion()`](../../src/pydac/core/column_model.py) and [`_resolve_packed_bed_thermal_transport()`](../../src/pydac/core/column_model.py).
Index ``a`` runs over the declared adsorbates, and the `*_adsorbate` arrays follow that
order; in a channeling bed the RHS multiplies these mean-based arrays by the local
gas or solid factor.

| Symbol | Code attribute | Definition |
|---|---|---|
| φ_a | `phi_adsorbate` | ρ_b q_a⁰ / c_tot⁰ |
| ψ_a | `psi_adsorbate` | ρ_b q_a⁰ / c_a⁰ |
| c_tot⁰/c_i⁰ | `c_tot_i_ratio` | dispersion driving-force ratio |
| ⟨ρc_p⟩_eff | `acc_g`, `acc_sorbent`, `acc_fixed_adsorbates`, `acc_ads`, `acc_structure` | gas sensible storage, dry sorbent, any fixed adsorbate inventory, declared loadings; see §5.4 for `acc_structure` |
| Ω_a | `Omega_adsorbate` | −ρ_b q_a⁰ ΔH_a / T⁰ |
| ω_p | `omega_p` | ε_t c_tot⁰ R_g |
| β_w | `k_wall` | λ_w τ_H / (ρ_w Ĉ_p,w L_b²) |
| α_in, α_out | `alpha_in`, `alpha_out` | τ_H · 2 h r / (C_wall (r_out²−r_in²)) |
| 1/Pe_zm, 1/Pe_rm | `inv_Pe_m_ax`, `inv_Pe_m_rad` | D_disp / (v⁰ · L or R_b) |
| λ̄_ax, λ̄_rad | `inv_Pe_h_ax`, `inv_Pe_h_rad` | λ_ax / (v⁰ L_b), λ_rad / (v⁰ R_b) |
| ω_wall | `omega_wall` | (4/d_b) τ_H h_int — see §5.2 |

As in the paper (SI, scaled energy equation), the energy equation is made dimensionless in
its independent variables and in temperature only: ⟨ρc_p⟩_eff, Ω_a, ω_p, λ̄_ax, λ̄_rad and
ω_wall keep units J/(m³ K) and are not normalised by a reference heat capacity. The code
attribute names `inv_Pe_h_*` are historical.

The L/R_b geometric anisotropy factor of the radial material-dispersion term is
applied when the operators are built, not when the groups are computed
([`set_disp()`](../../src/pydac/core/column_model.py), factor `L_R_ratio`).

Here the subscript ``m`` denotes the material Péclet number. The molecular
diffusivity ``D_m`` enters the selected dispersion closure through the Schmidt
number, while ``D_disp`` denotes the resulting effective material-dispersion
coefficient used in the PDE groups.

### 3.3 Closures and properties

| Closure | Code |
|---|---|
| axial dispersion, Tsotsas–Schlünder | [`resolve_packed_bed_dispersion()`](../../src/pydac/core/bed_architectures/packed_bed.py) |
| radial dispersion, Delgado | [`resolve_packed_bed_dispersion()`](../../src/pydac/core/bed_architectures/packed_bed.py) |
| axial effective conductivity | [`axial_effective_conductivity()`](../../src/pydac/core/bed_architectures/packed_bed.py) |
| stagnant conductivity: modified Zehner–Schlünder (m = 0.9676) by default; `specchia_1980` selectable (`bed.design.stagnant_conductivity_model`) | [`zehner_schlunder_stagnant_conductivity()`](../../src/pydac/core/bed_architectures/packed_bed.py) |
| radial effective conductivity, Dixon Pe_∞ closure | [`radial_effective_conductivity()`](../../src/pydac/core/bed_architectures/packed_bed.py) |
| wall Nusselt number and h_in, default `dixon` form: Nu_w = Nu_w,0 + (1/Nu_w* + 1/Nu_m)⁻¹ — Yagi–Kunii (Nu_w, Nu_m), Martin–Nilles (Nu_w,0), Tobiś–Ziółkowski (Nu_w*); `specchia_1980` selectable (`bed.design.wall_htc_model`) | [`wall_heat_transfer_coefficient()`](../../src/pydac/core/bed_architectures/packed_bed.py) |
| solid heat capacity, once per step at `T_feed`: Low et al. (`Low2023`, default) or `constant` | [`init_solid_thermo()`](../../src/pydac/core/column_model.py) |
| Dixon overall wall-to-bed coefficient, and its 1D/2D selection | [`dixon_overall_coefficient()`](../../src/pydac/core/bed_architectures/packed_bed.py); selection in [`_resolve_packed_bed_thermal_transport()`](../../src/pydac/core/column_model.py) |
| Re, Sc, Pr | [`calc_adim_numbs()`](../../src/pydac/core/column_model.py) |
| mixture density | [`init_rho_mix()`](../../src/pydac/core/column_model.py) |
| viscosity: Chapman–Enskog + Neufeld + Wilke | [`init_visc_mix()`](../../src/pydac/core/column_model.py) |
| Shomate heat capacities — the **only** source of gas `cp`, per step at `T_feed` | [`init_cp_mix()`](../../src/pydac/core/column_model.py) |
| mixture thermal conductivity | [`init_lambda_mix()`](../../src/pydac/core/column_model.py) |
| Fuller–Schettler–Giddings + Wilke mixture diffusivity | [`calc_Diff_coeffs()`](../../src/pydac/core/column_model.py) |

All per-species arrays follow the declared mixture order. The terminal summary
and schema-v1 `resolved_runtime.json` read the same unit-bearing values stored
in `model.runtime_properties`. The JSON additionally records provenance, the
selected models, the species catalog, `StateLayout` and the evaluation state,
plus a reference to the step's spatial solution: the full trajectory in
`sol.pkl`, or only the terminal state in `terminal.pkl` under the compact
campaign persistence.

The following sections describe the reference state and transport configuration.

---

## 4. The reference state is an input, not the feed

The reference state is fixed and independent
of the feed, because a reference tied to the feed would rescale the whole
non-dimensional system at every step of the cycle. What follows is the concrete
input surface that implements it.

| Non-dim variable | Reference | Source |
|---|---|---|
| c̃_i, c̃_tot | c_tot⁰ = p_ref/(R_g T_ref), c_i⁰ = y_i^ref c_tot⁰ | `simulation.reference.pressure_Pa`, `.temperature_K`, `.co2_mole_fraction`, `.relative_humidity` |
| q̃_i | q_i⁰ = equilibrium loading **at the reference state** | equilibrium system evaluated in [`set_reference_conditions()`](../../src/pydac/core/column_model.py) |
| T̃ | T⁰ = `reference.temperature_K` | default 298.15 K |
| p̃ | p⁰ = `reference.pressure_Pa` — and p̃ = c̃_tot · T̃ | default 101325 Pa |
| ṽ_z | v⁰ = `reference.velocity_m_s` | default 0.1 m/s |
| τ | τ_H = L_b / v⁰ | `tau_hyd` |
| Z, R | L_b, R_b | bed geometry |

Defaults live in [`src/pydac/inputs/defaults/reference.py`](../../src/pydac/inputs/defaults/reference.py):

The temperature scale $T^0$ above is distinct from the thermodynamic datum
used for sensible enthalpy. pyDAC uses the fixed datum
$T_{h,\mathrm{ref}}=298.15\ \mathrm{K}$; as in the paper, their ratio is
$\theta=T_{h,\mathrm{ref}}/T^0$, so the sensible temperature is
$\widetilde T-\theta=(T-T_{h,\mathrm{ref}})/T^0$ (`theta_bed` in the code).
The temperature state itself remains the absolute ratio $\widetilde T=T/T^0$.
T = 298.15 K, p = 101325 Pa, RH = 0.5, y_CO₂ = 4·10⁻⁴, v⁰ = 0.1 m/s.

### 4.1 Two velocities that must not be confused

The reference velocity v⁰ and inlet velocity v_in are different quantities
and the code keeps them in different attributes:

| Quantity | Code | Used for |
|---|---|---|
| **Reference velocity** v⁰ | `u_ref` (a public input) | non-dimensionalisation only: τ_H, 1/Pe_zm, 1/Pe_rm, 1/Pe_zh, 1/Pe_rh |
| **Superficial inlet velocity** | `u_inlet` (resolved per step from the flow input) | the physics of the correlations: Re, and the convective term of D_rad |

The closure `D_rad = D_eff + v_in d_p/12` is therefore based on `u_inlet` in the code
([`resolve_packed_bed_dispersion()`](../../src/pydac/core/bed_architectures/packed_bed.py)) — and Re
likewise ([`calc_adim_numbs()`](../../src/pydac/core/column_model.py)). In a closed
step (`blow`, `heat`, `cool`) `u_inlet` is zero while `u_ref` stays finite.

---

## 5. The pyDAC-specific layer

### 5.1 Bed↔wall coupling in 2D

In 2D the bed-wall coupling uses the
**radial boundary condition of the bed energy equation** at r = R_b — the Robin
condition of the paper (SI, "Initial and Boundary conditions"), with the inverse wall Biot number
`inv_Bi_h = λ_rad / (h_w R_b)`
([`_resolve_packed_bed_thermal_transport()`](../../src/pydac/core/column_model.py); applied in [`pde_system_kpis()`](../../src/pydac/core/column_model.py), stage 5):

$$\left.\frac{1}{Bi_w}\frac{\partial \tilde T}{\partial R}\right|_{R=1} + \tilde T\big|_{R=1} = \tilde T_w$$

The wall balance must then debit exactly the flux the bed receives. It does not
use the outer *cell-centre* temperature but the reconstructed **face**
temperature, with the same second-order stencil pymrm applies to the Robin BC
(stencil weights `wall_face_c_*` in
[`calc_nondim_pde_params()`](../../src/pydac/core/column_model.py), used in [`pde_system_kpis()`](../../src/pydac/core/column_model.py), stage 5). Without this bed and wall
would exchange different amounts of heat (the sign depends on the heat-flow direction
and the radial profile) and the coupled discretisation would not be
energy-conserving.

### 5.2 Bed↔wall coupling in 1D

In 1D radial conduction is switched off and the coupling becomes the **explicit
volumetric source** of the paper (main text, bed-to-wall coupling of the energy balance;
SI, "Initial and Boundary conditions")
([`calc_nondim_pde_params()`](../../src/pydac/core/column_model.py) and
[`pde_system_kpis()`](../../src/pydac/core/column_model.py), stage 5):

$$\mathcal{S}_\text{wall} = \omega_\text{wall}\,(\tilde T_w - \tilde T), \qquad \omega_\text{wall} = \frac{4}{d_b}\,\tau_H\,h_\text{int}$$

**The transfer coefficient is not h_in.** In 1D the radial bed profile is
lumped, so `h_int` is the **Dixon (1996) overall wall-to-bed coefficient**
U_w of the paper (main text)
([`dixon_overall_coefficient()`](../../src/pydac/core/bed_architectures/packed_bed.py)).
Selection is in
[`_resolve_packed_bed_thermal_transport()`](../../src/pydac/core/column_model.py) —
`U_w` in 1D, `h_w` in 2D — and the same `h_int` also feeds α_in
in the wall balance.

An experimental bypass exists: `wall.overall_htc_override_W_m2K` replaces the
resolved `U_w` with a user constant in 1D (`wall.bed_wall_htc_override_W_m2K` is
the 2D one, for `h_w`; each is refused on the other grid), while the physically resolved value stays visible in
`resolved_transport.diagnostics`
([`_resolve_packed_bed_thermal_transport()`](../../src/pydac/core/column_model.py)).

### 5.3 Closures without through-flow

Most TVSA steps have no through-flow (Re = 0):

| Closure | At Re = 0 | Code |
|---|---|---|
| radial conductivity | λ_r^dyn = (Pe_h/Pe_∞) λ_g (Dixon 2012, Eq. 6) vanishes, so λ_rad = λ⁰. The molecular term 2ε/(3Pe_h) of the radial Péclet interpolation is not added: λ⁰ already accounts for conduction through the fluid | [`radial_effective_conductivity()`](../../src/pydac/core/bed_architectures/packed_bed.py) |
| wall Nusselt number | guarded for Re ≤ 10⁻¹⁶: Nu_w,conv = 0, so Nu_w = Nu_w,0 | [`wall_heat_transfer_coefficient()`](../../src/pydac/core/bed_architectures/packed_bed.py) |

### 5.4 Packed-bed structure and closures

The detached packed-bed contract resolves flow porosity ε_flow = ε_b,
total porosity ε_t = ε_b + (1−ε_b)ε_p, and sorbent density
ρ_b = (1−ε_b)ρ_p. The retained `acc_structure` field is zero.
Pressure drop follows Ergun; the thermal and dispersion closures are those
described above. Both 1D and axisymmetric 2D use this same structure contract.

### 5.5 Equilibrium and kinetics as implemented

The equilibrium and kinetic implementations are:

| Species | Equilibrium | Code |
|---|---|---|
| CO₂ | Toth, with optional WADST wet co-adsorption blend driven by the H₂O loading | [`TothIsotherm`](../../src/pydac/core/isotherm_models.py), [`TothGABWADSTEquilibriumSystem`](../../src/pydac/core/isotherm_models.py) |
| H₂O | GAB, with IAPWS-1995 saturation pressure; fitted up to RH = 0.9 (a `RuntimeWarning` reports the first evaluation above it), RH kept 1 % below the GAB pole 1/k | [`GABIsotherm`](../../src/pydac/core/isotherm_models.py), [`water_saturation_pressure()`](../../src/pydac/core/isotherm_models.py) |
| CO₂/N₂ | competitive Sips (not used by the DAC default) | [`CompetitiveSipsEquilibriumSystem`](../../src/pydac/core/isotherm_models.py) |

Water's vapor pressure is a property of water, not of the sorbent model. For the RH conversion
and the steam property calls, `require_water_component()`
([`utils/column/validation.py`](../../src/pydac/utils/column/validation.py)) uses the `EOS(T)` of
the selected H₂O model when it has one, and otherwise `water_saturation_pressure(T)`, the
IAPWS-1995 correlation the built-in GAB model also uses. H₂O therefore needs neither a GAB model
nor its own EOS for these calls. This fallback does not change `T_cond`, the steam condensation
temperature that `ColumnModel` computes from the feed pressure with its own correlation.

These are the entries of `EQUILIBRIUM_REGISTRY` (`Toth`, `GAB`, `TothGABWADST`,
`CompetitiveSips`). The kinetic registry (`KINETIC_REGISTRY`,
[`core/kinetic_models.py`](../../src/pydac/core/kinetic_models.py)) currently holds one
law, `LDF`, for every declared adsorbate.

The LDF rate is assembled in its damped, non-dimensional form
scaled by τ_H/q_i⁰ (physical rate in
[`LDFRate.loading_rates_mol_per_kg_s()`](../../src/pydac/core/kinetic_models.py), scaling in
[`_evaluate_adsorption_rates()`](../../src/pydac/core/column_model.py)):

$$\frac{\partial \tilde q_i}{\partial \tau} = k_{LDF,i}\,\frac{p_i}{1+p_i}\,\tau_H\,\frac{q_{i,eq}-q_i}{q_i^0}, \qquad p_i \text{ in Pa}$$

The kinetic law receives partial pressures, temperature, current loadings and equilibrium
loadings, and returns physical rates in mol/(kg s). The model applies τ_H/q_i⁰ after each
call; no reference scale is stored in the kinetic object.

### 5.6 Momentum: the velocity is obtained by inversion

The 2D isotropic drag law is `A*u + B*rho*|u|*u = -grad(p)` for the velocity
vector. Both normal and transverse pressure gradients enter the face inversion;
`utils/vector_drag.py` supplies the stable inversion and transverse gradients.
The 1D limit recovers the scalar Ergun law. Viscosity is frozen at the step property
state; density and velocities are updated from the local state at each RHS call.

Total gas, species and sensible enthalpy have axial and radial convective fluxes.
The radial divergence is cylindrical, with zero flux at the axis and bed wall.
Channeling uses separate local gas/sorbent storage factors and a drag porosity field
averaged on fixed physical layers. Their area means preserve the declared inventories.
Dispersion and thermal closures use mean bed voidage and step bulk properties.

---

## 6. Energy source terms: what the code actually assembles

The bed-temperature RHS assembles three named contributions:

- `source_q = Σ_a Ω_a ∂q̃_a/∂τ` — adsorption heat over the declared adsorbates;
- `source_p = ω_p T̃ ∂c̃_tot/∂τ` — the retained pressure correction;
- `source_extra` — the sensible-energy accumulation caused by changing gas
  concentrations and adsorbed loadings:

$$
\overline S_\text{extra}
=-\left(\widetilde T-\theta\right)\left[
\sum_i \varepsilon_t c_{p,i}c_i^0
\frac{\partial\widetilde c_i}{\partial\tau}
+\sum_a \rho_b c_{p,a}q_a^0
\frac{\partial\widetilde q_a}{\partial\tau}
\right].
$$

In the wall-channeling mode ε_t and ρ_b are ring-wise coefficients, and so are Ω_a and
ω_p built from them; the form of the three terms is unchanged (paper SI, "Local
coefficients"). The code keeps the mean-based groups and multiplies them by the local
factors: `source_q` and the loading part of `source_extra` by the solid factor
(`_solid_cell`), `source_p` and the gas part of `source_extra` by the gas factor
(`_gas_cell`). In the uniform mode both factors are one.

The gas sum covers the complete declared mixture. Because only the independent
gas concentrations are integrated, the closure-species rate is reconstructed
from `c_tot` before evaluating the sum. The implementation uses precomputed
reference ratios and reusable work buffers in
[`_sensible_energy_accumulation_source()`](../../src/pydac/core/column_model.py).
`wall_exch` is added separately in 1D (§5.2).

### Naming note: `source_p` is the compressibility term

`source_p` is the compressibility contribution of the paper (main text, table of the bed
energy source terms), $\mathcal{S}_\mathrm{p}=\varepsilon_t R_g T\,\partial c_\text{tot}/\partial t$,
scaled as $\omega_p\widetilde T\,\partial\widetilde c_\text{tot}/\partial\tau$ (SI, "Final
scaled energy equation"). It comes from the time derivative of p = c_tot R_g T on the
left-hand side and is not a sensible-enthalpy term: the sensible-inventory derivative
belongs to `source_extra`. The absolute $\widetilde T$ remains in `source_p` because the
ideal-gas pressure is $p=c_\text{tot}R_gT$ and is independent of the enthalpy datum.

---

## 7. Boundary conditions

Boundary conditions use the generic linear form of pymrm
([03 §3](03_pymrm_discretization.md#3-boundary-condition-notation)),
`a ∂φ/∂n + b φ = d`, and are set per step in
[`set_bc()`](../../src/pydac/core/column_model.py). The bed-temperature radial
pair is the 2D coupling of §5.1; the pressure pair is step-dependent (sealed bed
for `cool`, closed inlet / pressure-related outlet for `blow`, `heat`, `w_purge`, `w_des`,
an inlet valve to the upstream (feed) pressure for `press`, whose inflow follows
from the pressure difference, open inlet/outlet otherwise). The per-step
table, with the events that terminate each step, is owned by
[05 — Cycle steps](05_cycle_steps.md).

---

## 8. Time scaling

The solver integrates in natural non-dimensional time:

$$\tau = \frac{t}{\tau_H} \in \left[0,\; \frac{t_\text{end}}{\tau_H}\right], \qquad \tau_H = \frac{L_b}{u_\text{ref}}$$

`nondim_horizon = t_end / tau_hyd` is computed per step by `Solver.__init__` and stored on
the model for progress reporting. The RHS returns dY/dτ with no rescaling by the step
horizon, and CVODE integrates directly in τ.

---

## 9. Positivity handling

The coupled species-transport discretization is not positivity-preserving: its central
dispersion terms and the mixture correction do not keep every concentration in the
composition domain, so a depleting gas species can undershoot zero. The solver uses the
following constraints and nonlinear evaluation bounds:

| Mechanism | Scope | Details |
|---|---|---|
| **CVODE constraints** (y ≥ 0) | declared loadings, c_tot, T_bed, T_wall | via `constraints_idx` / `constraints_type`; these fields must never be negative |
| **RHS lower bound** (`np.maximum(·, 1e-32)`) | every declared independent gas concentration and every declared loading | protects non-linear terms (kinetics, mixture properties, mole fractions); the solver state itself is not projected |

Independent gas concentrations are **not** constrained by CVODE. Instead the RHS uses a
**split field** strategy:

- **lower-bounded fields** (`ci_ordered`, `y_full`) — species convection and the
  reconstructed property composition (mixture molar mass, density, partial
  pressures); loadings are lower-bounded before kinetics. The closure mole
  fraction is one minus the independent ones and is not clipped on its own;
- **raw, unclamped mole fractions** (`yi_ordered`) — the **dispersion flux only**, so
  that no clipping kink enters its gradients.

During `prepare_results()`, native outlet histories are computed from the raw
solver trajectory before dimensional gas and total concentrations are lower-clipped
to zero in the stored fields. Reconstructing gas properties uses lower bounds where required; an upper mole-fraction clip is not imposed on the RHS.
The optional composition-domain guard is a terminal failure event: it fires when the
closure-species concentration falls below −max(absolute allowance, relative allowance ·
c_tot). It does not change the RHS, clip species or make the discretization
positivity-preserving
([`utils/solver/composition_domain.py`](../../src/pydac/utils/solver/composition_domain.py)).

---

For a 2D explicit `bed_wall_htc_override_W_m2K`, both `h_w` and
`inv_Bi_h = lambda_rad/(R_b*h_w)` use the override. The bed Robin flux and wall
source therefore share the same interface coefficient. The original correlation
remains available in `resolved_transport`; 1D uses its lumped overall coefficient.
