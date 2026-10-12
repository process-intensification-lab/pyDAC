---
title: "05 — Cycle Steps"
toc: true
toc-depth: 3
embed-resources: true
---

> **This manual is the detailed technical reference: how pyDAC works inside, for contributors
> and for checking the model.** To learn to use pyDAC, do the [tutorials](../../examples/README.md)
> first, then read the [user guide](../guide/01_installation.md).

> **Configuration note (2026-08-03).** Step values and stopping metadata are
> public inputs under `simulation.cycle.steps` and are authored in
> [`inputs/defaults/steps.py`](../../src/pydac/inputs/defaults/steps.py). For resolution,
> presets, frozen cases, and terminal settings, see the
> [Input Configuration Guide](01_inputs.md).

This document describes the built-in settings, boundary conditions, and stopping rules of each cycle step.

---

## 1. Cycle Construction

pyDAC does not hard-code one single thermodynamic cycle. The runner builds a cycle from explicit step tokens.

### 1.1 Default Sequences In Code

The sequences and their construction rules are owned by
[`utils/tvsa/cycle_steps.py`](../../src/pydac/utils/tvsa/cycle_steps.py).
If `cycle_steps=None`, the shared cycle runner uses the **vanilla TVSA cycle** —
wall pre-heating followed directly by steam-assisted desorption:

- Step mode:
  `["startup", "blow", "heat", "s_des", "cool", "press", "ads"]`
- CSS repeating mode:
  `["ads", "blow", "heat", "s_des", "cool", "press"]`

Steam purge (`s_purge`) and wall-heated desorption (`w_des`) remain explicit
choices in the configured step sequence.

In CSS mode:

- the repeating sequence must start with `ads`
- cycle 1 replaces that leading `ads` step with `startup`

User scripts can still pass their own explicit `cycle_steps`, and those explicit
steps take precedence over the internal defaults — with **one rule that is not a
default and cannot be opted out of**: `build_cycle_steps()` always prepends
`startup`, and removes it from anywhere else in the list. Asking for
`["ads", "blow"]` in step mode gives `["startup", "ads", "blow"]`. The repeating
CSS builder is the mirror image: `build_repeating_cycle_steps()` *rejects*
`startup` in the list, because cycle 1 is the only cycle that gets one and it
gets it by substitution for the leading `ads`.

### 1.2 Explicit Desorption Choice

The code requires an explicit desorption mode:

- `w_des` for wall-heated desorption
- `s_des` for steam-assisted desorption

Generic `des` / `desorption` tokens are rejected on purpose. The full alias
table — every spelling accepted for each of the ten steps — is in
[01 — Inputs §2.3](01_inputs.md#23-the-one-distinction-that-causes-most-of-the-confusion).

### 1.3 Every Step Uses the Direct Linear Solver

Steps do not choose a linear solver: every integrated step uses a direct
solver, the sparse factorization by default; `execution.use_sparsity = false`
selects the dense diagnostic path. An older case asking for `direct` or `auto`
is upgraded and one asking for GMRES is refused; see
[01 — Inputs §4.5](01_inputs.md#45-the-linear-solver).

---

## 2. Boundary-Condition Framework

### 2.1 General BC Form

The operator-based BCs follow the generalized linear form:

$$a \frac{\partial \phi}{\partial n}\bigg|_{face} + b \phi\bigg|_{face} = d$$

This is the same convention used by the underlying `pymrm` operator builders.

### 2.2 Species And Energy Inlet Treatment

For species and bed temperature, the axial inlet operator BC is always created with a dummy zero form:

```python
bc_ax   = ({'a': 0, 'b': 0, 'd': 0}, {'a': 1, 'b': 0, 'd': 0})
bc_ax_T = ({'a': 0, 'b': 0, 'd': 0}, {'a': 1, 'b': 0, 'd': 0})
```

Actual inlet transport is injected separately through Danckwerts-style inlet flux terms:

- `flux_in_ctot`
- `flux_in_ci`
- `flux_in_T`

Those fluxes are:

- fixed at initialization for `startup`, `ads`, `s_purge`, and `s_des`
- zero for `blow`, `heat`, `w_purge`, `w_des`, and `cool`, because their built-in
  volumetric flow is zero: the fluxes are computed from that flow, so keep it zero
  for a closed inlet
- recomputed every RHS call for `press`

### 2.3 Pressure BCs By Step

The axial pressure BCs, assembled by `ColumnModelMoL.set_bc()` in
[`core/column_model.py`](../../src/pydac/core/column_model.py), are:

| Step(s) | Inlet (z=0) | Outlet (z=L) | Notes |
|---------|-------------|--------------|-------|
| `startup`, `ads`, `s_purge`, `s_des` | Dummy/open: `a=0, b=0, d=0` | Dirichlet: `a=0, b=1, d=p_out/p_ref` | Feed enters through inlet fluxes, not through a pressure Robin BC |
| `blow`, `heat`, `w_purge`, `w_des` | Zero gradient: `a=1, b=0, d=0` | Valve Robin: `a=1/(alpha_bd L), b=1, d=p_out/p_ref` | If `alpha_bd = inf`, this collapses to an outlet Dirichlet condition |
| `cool` | Zero gradient: `a=1, b=0, d=0` | Zero gradient: `a=1, b=0, d=0` | Sealed-bed pressure BC |
| `press` | Valve Robin: `a=1/(alpha_ps L), b=1, d=p_feed/p_ref` | Zero gradient: `a=1, b=0, d=0` | Inlet-driven pressurization |

Important implementation detail:

- `heat`, `w_purge`, and `w_des` default to `alpha_bd = inf`, so they still use the Robin form in the code, but the effective behavior is outlet Dirichlet at `p_out`.

### 2.4 Radial BCs

| Equation | Centre (`r=0`) | Wall (`r=R_b`) |
|----------|----------------|----------------|
| Species | Zero gradient | Zero gradient |
| Pressure | Zero gradient | Zero gradient |
| Bed temperature, 2D | Zero gradient | Robin to the wall temperature: $\lambda_{\mathrm{rad}}\,\partial T/\partial r = h_{\mathrm{in}}(T_w - T)$ |
| Bed temperature, 1D (`Nr=1`) | Not used explicitly | Wall coupling moved into a source term |
| Wall temperature | Axial zero gradient at both ends | Separate 1D wall equation, coupled to the jacket temperature $T_j$ |

### 2.5 Velocity Treatment

Axial and radial velocities are recovered algebraically from the selected
pressure law and then constrained according to the physical step:

| Step(s) | Inlet-face handling | Outlet-face handling |
|---------|---------------------|----------------------|
| `startup`, `ads`, `s_purge`, `s_des` | Inlet velocity extrapolated from interior faces | Outlet left free |
| `blow`, `heat`, `w_purge`, `w_des` | Inlet velocity forced to zero | Outlet left free |
| `cool` | Inlet velocity forced to zero | Outlet velocity forced to zero |
| `press` | Inlet velocity comes directly from the selected pressure law + Robin pressure BC | Outlet velocity forced to zero |

The radial velocity is zero everywhere in 1D. In 2D it is recovered in the
interior and set to zero at the centreline and at the impermeable wall, in every
step.

### 2.6 Inlet-Flux Modes

| Mode | Steps | How the inlet is handled |
|------|-------|--------------------------|
| Fixed-feed flux | `startup`, `ads`, `s_purge`, `s_des` | Inlet fluxes are computed once from the prescribed feed state and inlet flow input; in 2D the total may be distributed over the radial rings with area-normalised weights |
| Closed inlet | `blow`, `heat`, `w_purge`, `w_des`, `cool` | With the built-in zero flow the inlet fluxes are zero; the inlet velocity is clamped to zero by step |
| Pressure-driven inlet | `press` | Inlet fluxes are rebuilt every RHS call from the current, pressure-derived local inlet velocity; no ring weights are imposed |

---

## 3. Time-Based And Event-Based Stopping

### 3.1 General Mechanism

Every step defines a nominal horizon (`horizon.nominal_time_s`) and a maximum
horizon for event mode (`horizon.event_max_time_s`). The solver then follows one
of two modes:

- `termination_mode = "time"`
  the step runs to the nominal horizon; the primary process event is off, while
  the cooling pressure floor (§3.3) stays active
- `termination_mode = "event"`
  the step stops when the configured metric reaches its threshold in the
  configured direction, or at the maximum horizon

If the primary target is already met by the inherited state, the step is
recorded as initially satisfied at `t = 0` and CVODE is not run.

Event specifications are derived from the model by
[`utils/solver/event_config.py`](../../src/pydac/utils/solver/event_config.py) and
assembled into CVODE events by
[`utils/solver/events.py`](../../src/pydac/utils/solver/events.py).
The default mode of each step is `steps.<step>.termination.mode`, authored in
[`inputs/defaults/steps.py`](../../src/pydac/inputs/defaults/steps.py). A run overrides it
with `execution.termination_mode`; a CSS cycle with
`execution.cycle_options[n].termination_mode`, whose nested `step_options` take
precedence; a step run per step with `execution.step_options`.

### 3.2 Supported Event Metrics

The current event-capable metrics are:

| Metric key | Meaning |
|------------|---------|
| `avg_outlet_pressure` | Area-weighted outlet pressure |
| `max_internal_pressure` | Maximum pressure anywhere in the bed |
| `min_internal_pressure` | Minimum pressure anywhere in the bed |
| `avg_outlet_temperature` | Area-weighted outlet bed temperature |
| `max_bed_temperature` | Maximum bed temperature anywhere in the domain |
| `min_bed_temperature` | Minimum bed temperature anywhere in the domain |
| `max_bed_oxygen_mole_fraction` | Maximum gas-phase O₂ mole fraction anywhere in the bed |
| `avg_bed_loading_fraction_CO2` | Sorbent-mass-weighted mean of $q_{\mathrm{CO_2}}/q_{\mathrm{CO_2}}^0$ (weights: cell volume times the local solid-inventory factor) |

`steps.<step>.termination.metric` accepts only these eight names; `pydac validate`
refuses any other. `max_bed_oxygen_mole_fraction` needs O₂ in the mixture and
`avg_bed_loading_fraction_CO2` needs CO₂ as an adsorbate: the run preflight (also
`--dry-run`) refuses either on a step that runs in event mode in any cycle of the run
(the names are listed in
[`utils/tvsa/termination.py`](../../src/pydac/utils/tvsa/termination.py)).

Thresholds are built from exactly one of (validation refuses both or neither):

- `termination_threshold_source` — one of `p_out`, `p_feed` and `T_cond`, read as
  `getattr(model, source)`, so it means the value the resolved model actually holds;
- `termination_threshold_value` — a literal;

and then shifted by `termination_threshold_offset`. The offset is signed:
`blow` uses `p_out` with `+25`, `press` uses `p_feed` with `−25`.

#### The pressure-rate plateau fallback

This one is deliberately *not* one of the selectable metrics above. Instead of a
point value of the state, it reads the **rate** straight from the RHS, and it is
appended as a **second terminal event next to** the configured pressure target —
the target is never replaced, and whichever fires first ends the step. It exists
because a pressure that asymptotes slowly can approach its target without ever
crossing it.

Both directions come from one builder in
[`utils/solver/pressure_plateau.py`](../../src/pydac/utils/solver/pressure_plateau.py):

| | `blow` | `press` |
|---|---|---|
| Event name | `blowdown_pressure_rate_plateau` | `pressurization_pressure_rate_plateau` |
| Metric | `−d(mean(p/p_ref))/d(t/tau_hyd)` | `+d(mean(p/p_ref))/d(t/tau_hyd)` |
| Fires when | the rate falls below `pressure_plateau_rate_tolerance_nd` (built-in `1e-5`) | same |
| Arming field | `pressure_plateau_arm_below_Pa` (built-in `50 kPa`) | `pressure_plateau_arm_above_Pa` (built-in `50 kPa`) |
| Armed while | `max(p_internal) <= 50 kPa` | `min(p_internal) >= 50 kPa` |

`mean` is the arithmetic mean over the bed cells, with no volume weighting: the rate of
`p/p_ref = c_tot T_bed` (nondimensional) is averaged with equal weight per cell. In 1D the
cells have equal volume, so this is the volume average. In 2D the radial cells have equal
width but unequal annular volume, so the inner rings count more than their share of the bed
volume. The metric is only a stopping criterion compared against a tolerance; it does not
enter any balance.

**The 50 kPa is a guard, not the stopping threshold.** The guard is checked at
every evaluation; while the bed is not past it the detector returns
`tolerance + 1`, so it cannot fire during the fast initial transient — where the
rate is large but the pressure is nowhere near settled. The plateau event stops
the step at the rate tolerance; the primary pressure target can end it first.
When `stop_on_pressure_plateau` is true,
[`config/steps.py`](../../src/pydac/config/steps.py) requires the matching arming field.
It refuses every plateau field on a step that never reads it: the switch and the rate
tolerance off `blow` and `press`, `pressure_plateau_arm_above_Pa` on `blow`,
`pressure_plateau_arm_below_Pa` on `press`.

### 3.3 Event Stop Rules

Values below are the shipped defaults from
[`inputs/defaults/steps.py`](../../src/pydac/inputs/defaults/steps.py). The **default
mode** column matters: some steps ship in `time` mode, so their event rule
is defined but dormant until `termination_mode` is switched to `event` (§3.1).
The horizons are set with `steps.<step>.horizon.nominal_time_s` and
`steps.<step>.horizon.event_max_time_s`.

| Step | Default mode | Implemented event rule | Nominal horizon | Maximum horizon (event mode) |
|------|---|------------------------|------|------|
| `startup` | `time` | `avg_bed_loading_fraction_CO2` goes above `0.95` | `39000 s` (650 min) | `43200 s` (720 min) |
| `ads` | `time` | `avg_bed_loading_fraction_CO2` goes above `0.95` | `39000 s` (650 min) | `43200 s` (720 min) |
| `blow` | `event` | `max_internal_pressure` goes below `p_out + 25 Pa` **OR** the armed mean pressure descent rate goes below `1e-5` | `180 s` (3 min) | `900 s` (15 min) |
| `heat` | `time` | `min_bed_temperature` goes above `T_cond` (offset `0`) | `9000 s` (150 min) | `21600 s` (360 min) |
| `w_purge` | `event` | `max_bed_oxygen_mole_fraction` goes below `1e-7` (0.1 ppmmol) | `900 s` (15 min) | `9000 s` (150 min) |
| `s_purge` | `event` | `max_bed_oxygen_mole_fraction` goes below `1e-7` (0.1 ppmmol) | `300 s` (5 min) | `600 s` (10 min) |
| `w_des` | `event` | `avg_bed_loading_fraction_CO2` goes below `0.05` | `86400 s` (24 h) | `86400 s` (same as nominal) |
| `s_des` | `time` | `avg_bed_loading_fraction_CO2` goes below `0.05` | `3600 s` (60 min) | `10800 s` (180 min) |
| `cool` | `event` | `max_bed_temperature` goes below `323.15 K`; low-pressure floor can also terminate the step | `9000 s` (150 min) | `10800 s` (180 min) |
| `press` | `event` | `min_internal_pressure` goes above `p_feed − 25 Pa` **OR** the armed mean pressure rise rate goes below `1e-5` | `300 s` (5 min) | `900 s` (15 min) |

Extra cooldown safety event: by default (`termination.stop_on_low_pressure`,
`termination.low_pressure_stop_Pa`) cooling also stops when
`min_internal_pressure` falls to `100 Pa`, in both time and event modes. The two fields
act only on `cool`; validation refuses them on any other step, where they would be
ignored.

The event threshold determines the state passed to the next step. Changing it
can change both process performance and subsequent solver cost; choose it as a
process input. The primary process threshold is used only in event mode; the
cooling pressure floor is separate and stays active in time mode.

The model keeps the nominal duration as `t_end`. In event mode the solver
integrates toward `event_max_time_s` from the start and stops sooner if an
event fires; if `event_max_time_s` is null, it uses the nominal horizon.
Selected steps may be locally forced to `termination_mode = "time"` for
fixed-duration runs without editing the event
definitions.

---

## 4. Step-By-Step Details

### 4.1 Startup (`startup`)

**Purpose:** initialize uniform gas, solid and temperature fields from the startup case before the rest of the sequence.

Implementation summary:

- Feed is ambient air at `298.15 K`, `400 ppm CO2`, `50% RH`.
- `vol_flowrate` is user-supplied or defaults to `1e-3 m3/s`.
- `p_feed = p_out = 101325 Pa`.
- `alpha_bd = inf`, `alpha_ps = inf`.
- Built-in mode is time. In event mode, startup stops at a sorbent-mass-weighted
  mean CO₂ loading fraction of `0.95`.

Important notes:

- `startup` is not part of the repeating CSS cycle.
- In CSS mode it appears only in cycle 1.
- `startup` builds its state from the startup configuration rather than inheriting
  `ColumnModelInput.initial_state`. Its solid is in equilibrium with the configured initial gas
  when `composition.initial_solid_loadings_mol_per_kg` is `null`; otherwise every adsorbate starts
  at its declared uniform loading ([01 — Inputs §4.2](01_inputs.md#42-a-step-ten-names-seven-groups-fixed-order)).
  "Startup" does not imply an unloaded solid.

### 4.2 Adsorption (`ads`)

**Purpose:** capture CO2 and H2O from ambient air.

Implementation summary:

- Same feed composition and ambient jacket conditions as `startup`.
- Inherits the thermal and loading state from the previous step.
- Uses fixed inlet fluxes computed from the prescribed air flow rate.
- Pressure outlet is fixed at `p_out = 101325 Pa`.
- Built-in mode is time. In event mode, adsorption stops at a sorbent-mass-weighted
  mean CO₂ loading fraction of `0.95`.

Physical picture:

- a thermal adsorption front forms and propagates axially
- radial heat removal occurs through the wall and jacket
- the outlet pressure stays pinned to atmospheric conditions

### 4.3 Blowdown (`blow`)

**Purpose:** depressurize the loaded bed toward vacuum.

Implementation summary:

- No inlet feed: `vol_flowrate = 0`.
- Inlet is hydraulically closed.
- Outlet uses a finite valve conductance: `alpha_bd = 0.01 1/m`
  (`steps.blow.flow.blowdown_valve_conductance`).
- Built-in outlet pressure `20000 Pa` (`steps.blow.pressure.outlet_pressure_Pa`;
  the campaign knob `low_pressure_Pa` sets it with the other low-pressure steps,
  [01 — Inputs §8.1](01_inputs.md#81-operating-knobs-and-their-fan-out)).
- Primary event: maximum internal pressure reaches `p_out + 25 Pa`.
- Plateau fallback: the mean pressure **descent** rate falls below `1e-5`, armed
  only once `max(p_internal) <= 50 kPa` — mechanics, both directions, and why
  the 50 kPa is only a guard are in §3.2.

Implementation note (applies to `press` as well):

- `stop_on_pressure_plateau = false` disables only the fallback and preserves
  the original pressure-target event
- forcing the step to time mode disables both terminal events while preserving
  their configuration for later event-mode runs
- pressure-target and plateau behavior are both reported in the step log

### 4.4 Pre-Heating (`heat`)

**Purpose:** heat the bed before desorption.

Implementation summary:

- No inlet flow.
- Hot jacket at `373.15 K`.
- Built-in outlet and feed pressures both `20000 Pa`; the campaign knob
  `low_pressure_Pa` sets them together with the other low-pressure steps.
- `alpha_bd = inf`, so the outlet pressure relation behaves as a direct vacuum boundary.
- Built-in mode is time. In event mode, heating ends when the minimum bed
  temperature reaches the steam condensation temperature `T_cond` on ascent. The
  threshold offset is `0`, so there is no extra margin on top of it.

Important implementation detail:

- `T_cond` is a pressure-dependent condensation temperature evaluated from `p_feed`
- a fixed-duration run can force this step to time mode while keeping the temperature event definition available

### 4.5 Wall Purge (`w_purge`)

**Purpose:** pre-heat the evacuated bed with the hot wall until oxygen has
fallen below the purge target, without introducing steam.

Implementation summary:

- It uses the same composition, zero inlet flow, thermal conditions and pressure
  conditions as `heat`, with shorter horizons: nominal `900 s`, maximum `9000 s`.
- The hot jacket is held at `373.15 K`; the inlet is closed and the outlet
  remains tied to the low-pressure boundary.
- Unlike `heat`, it is event-driven by
  `max_bed_oxygen_mole_fraction < 1e-7` (0.1 ppmmol).
- If the oxygen event ends the step, every bed cell meets the target at that
  moment. Reaching the maximum horizon does not show that the target was met.

`w_purge` is a configured canonical step but is deliberately absent from both
default cycle sequences. A case requiring this step must include it
explicitly in `execution.cycle_steps`.

Post-processing follows the ordinary pre-heating branch: temperature,
vacuum-energy, and jacket-energy KPIs are retained, with no steam KPIs. By
default the outlet is vented (`mass.vented`); `outlet.collect_product` sends it
to the product line instead ([01 — Inputs §4.2](01_inputs.md#42-a-step-ten-names-seven-groups-fixed-order)).

### 4.6 Steam Purge (`s_purge`)

**Purpose:** remove oxygen from the hot evacuated bed before the main
desorption step.

Implementation summary:

- Feed is pure steam at `373.15 K`.
- Set `steps.s_purge.flow.steam_mass_flow_kg_s`; pyDAC derives
  `volumetric_flow_m3_s` from the inlet steam density exactly as for `s_des`
  ([01 — Inputs §4.3](01_inputs.md#43-derived-fields)).
- `p_out` and `p_feed` are both at the low-pressure level.
- The inlet uses fixed feed fluxes and the outlet pressure is Dirichlet, the
  same transport/pressure-BC family as `s_des`.
- The default event stops when the **maximum** O₂ mole fraction over every bed
  cell drops below `1e-7` (0.1 ppmmol).

The maximum is deliberate: an average criterion could stop while a local
  oxygen pocket remained above the safety target. The metric reconstructs the
physical mole fractions from the independent-species and total-concentration
states; if O₂ is the closure species, its fraction is one minus the sum of the
independent fractions. If the event is not reached, the solver uses the `600 s` fallback.

Post-processing uses the same KPI branch as `s_des`: it always records pressure
drop, compression, vacuum, jacket, injected steam mass and steam energy; the
product-line and condenser records follow its outlet settings. By default its
outlet is collected, so it adds to the cycle product. To vent it set both
`steps.s_purge.outlet.collect_product = false` and
`steps.s_purge.outlet.condenser = false` (a condenser requires a collected
outlet); its steam still counts in the steam use and the energy. It remains
a distinct cycle step; the regeneration (working capacity, saturation after
desorption) ends at the last desorption step executed, whatever its outlet.

### 4.7 Wall-Heated Desorption (`w_des`)

**Purpose:** continue desorption under vacuum without a purge gas.

Implementation summary:

- No inlet flow.
- Hot jacket at `373.15 K`.
- Outlet remains tied to the vacuum target.
- `alpha_bd = inf`, so the outlet behaves effectively as a direct vacuum boundary.
- Event stop rule: average bed CO2 loading fraction drops below `0.05`.

Compared with `heat`:

- boundary conditions are similar
- `w_des` defaults to a CO₂-loading event
- `heat` has a `T_cond` temperature event but defaults to fixed-duration (time) operation

### 4.8 Steam-Assisted Desorption (`s_des`)

**Purpose:** desorb CO2 using an explicit steam purge.

Implementation summary:

- Feed is steam at `373.15 K`.
- Set `steps.s_des.flow.steam_mass_flow_kg_s`; `volumetric_flow_m3_s` is derived
  from it and the inlet state. A written volumetric flow must agree with the
  derived one ([01 — Inputs §4.3](01_inputs.md#43-derived-fields)).
- `p_out` and `p_feed` are both set to the low-pressure level.
- Built-in mode is time. In event mode, it stops at a sorbent-mass-weighted mean
  CO₂ loading fraction of `0.05`.

Boundary-condition behavior:

- this step does not use the `blow` / `heat` / `w_des` pressure-BC branch
- instead, it behaves like a fixed-feed step with a dummy inlet pressure BC and outlet Dirichlet pressure

Practical consequence:

- for both `s_purge` and `s_des`, steam is imposed through inlet fluxes, not
  through a pressurization-style Robin inlet pressure condition

### 4.9 Cooldown (`cool`)

**Purpose:** cool the regenerated bed before returning to adsorption conditions.

Implementation summary:

- No inlet flow.
- Jacket at `298.15 K`.
- Pressure BC is zero-gradient at both inlet and outlet.
- Velocity is clamped to zero at both ends.
- Event stop rule: maximum bed temperature reaches the cooldown target, with an additional low-pressure safety event.

Cooldown preserves the inherited gas, adsorbed and temperature profiles.
`relax_cool_hydraulics` and `flatten_cool_axial_profiles` are internal flags set to
false; the step does not pre-flatten the state or replace it with uniform pressure.
Internal axial/radial redistribution is resolved by the PDE under sealed boundaries.
By default the step is event-driven: it ends when `max_bed_temperature` goes below
`323.15 K`, with a maximum horizon of `10800 s`. In time mode it runs for
`steps.cool.horizon.nominal_time_s` (`9000 s`).


### 4.10 Pressurization (`press`)

**Purpose:** raise the bed pressure back to atmospheric conditions.

Implementation summary:

- Feed is ambient air at `298.15 K`.
- The outlet is closed.
- The inlet uses a Robin pressure BC with `alpha_ps = 0.01 1/m`
  (`steps.press.flow.pressurization_valve_conductance`).
- Initial inlet fluxes are zero.
- Inlet fluxes are rebuilt every RHS call from the current inlet velocity.
- Primary event: minimum internal pressure reaches `p_feed − 25 Pa`.
- Plateau fallback: the mean pressure **rise** rate falls below `1e-5`, armed
  only once `min(p_internal) >= 50 kPa` — see §3.2, and the disabling note in
  §4.3, which applies here too.

Pressure-driven coupling sequence:

1. The inlet Robin BC creates a pressure gradient at `z = 0`.
2. The selected pressure-drop relation converts that gradient into an inlet velocity.
3. `update_inlet_fluxes()` rebuilds `flux_in_ctot`, `flux_in_ci`, and `flux_in_T`.
4. In event mode, the bed pressure rises until whichever of the two terminal
   events above fires first, or until the maximum horizon if neither fires. The plateau one exists precisely for this step's shape: a
   pressure-driven inlet makes the approach to `p_feed` asymptotic, so the point
   target can be approached for a long time without being crossed.

Important note:

- `vol_flowrate = 1e-3 m3/s` is only a dummy value for property correlations in this step
- the actual inlet transport rate is pressure-driven

---

## 5. Practical Implications

- A step run switches a step between `termination_mode = "time"` and `"event"`
  with `execution.step_options`; a CSS run uses
  `execution.cycle_options[n].step_options` (a top-level `step_options` is refused
  in a CSS run).
- CSS `cycle_options[n]["termination_mode"]` applies to every step in that cycle unless nested `step_options` override it.
- To change an event metric or threshold, set `steps.<step>.termination.metric`,
  `.direction`, `.threshold_value`, `.threshold_source` and `.threshold_offset` in
  the case; the built-in values are authored in
  [`inputs/defaults/steps.py`](../../src/pydac/inputs/defaults/steps.py). To change the
  event implementation itself, use the supervised core workflow if the change
  touches `src/pydac/core/column_model.py` or `src/pydac/post_processing/`.
- `cool` preserves its inherited state and solves redistribution with sealed boundaries.
