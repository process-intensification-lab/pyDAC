---
title: "05 — Cycle steps"
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

## 1. Cycle construction

pyDAC does not hard-code one single thermodynamic cycle. The runner builds a cycle from explicit step tokens.

### 1.1 Default sequences in code

The sequences and their construction rules are owned by
[`utils/tvsa/cycle_steps.py`](../../src/pydac/utils/tvsa/cycle_steps.py).
If `cycle_steps=None`, the shared cycle runner uses the **vanilla temperature-vacuum swing adsorption (TVSA) cycle** —
wall pre-heating followed directly by steam-assisted desorption:

- Step mode:
  `["startup", "blow", "heat", "s_des", "cool", "press", "ads"]`
- Cyclic steady state (CSS) repeating mode:
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

### 1.2 Explicit desorption choice

The code requires an explicit desorption mode:

- `w_des` for wall-heated desorption
- `s_des` for steam-assisted desorption

Generic `des` / `desorption` tokens are rejected on purpose. The full alias
table — every spelling accepted for each of the ten steps — is in
[01 — Inputs §2.3](01_inputs.md#23-the-one-distinction-that-causes-most-of-the-confusion).

### 1.3 Every step uses the direct linear solver

Steps do not choose a linear solver: every integrated step uses a direct
solver, the sparse factorization by default; `execution.use_sparsity = false`
selects the dense diagnostic path. An older case asking for `direct` or `auto`
is upgraded and one asking for GMRES is refused; see
[01 — Inputs §4.5](01_inputs.md#45-the-linear-solver).

### 1.4 Rejected trial states

CVODE's Newton iteration can propose a trial state that is physically impossible: a
non-positive total concentration or temperature, a negative species concentration or closure
species, or a non-finite value. When the right-hand side raises a `ValueError` at such a state
(the drag law cannot invert a negative pressure, for example), the solver returns NaN
derivatives instead; CVODE rejects the attempt and retries with a smaller step. A `ValueError` at a physical state is a genuine error and is
raised as before. After 1000 rejections in one solve (counted cumulatively, not in a row) the
solver gives up with an error that names the first one. Independently of rejections, a solve
that reports success but whose saved times, states or event arrays are not finite (also when
an initial event skips CVODE) is marked as failed, with status `nonfinite_output` in both
`sol["cvode"]` and the termination summary. One rejected attempt
usually brings a few dozen more: the NaN also reaches CVODE's difference-quotient Jacobian,
which is rebuilt before the step goes on. A step that needed any records them in its solution, `sol["cvode"]`:
`rhs_rejections` (the count) and `first_rhs_rejection` (time in s and message), and prints one
line; a step that needed none saves exactly as before. Rejections are expected on a vacuum
pump step whose pressure control reopens (§2.7).

---

## 2. Boundary-condition framework

### 2.1 General BC form

The operator-based boundary conditions (BCs) follow the generalized linear form:

$$a \frac{\partial \phi}{\partial n}\bigg|_{face} + b \phi\bigg|_{face} = d$$

This is the same convention used by the underlying `pymrm` operator builders.

### 2.2 Species and energy inlet treatment

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
- recomputed every right-hand side (RHS) call for `press`

### 2.3 Pressure BCs by step

The axial pressure BCs, assembled by `ColumnModelMoL.set_bc()` in
[`core/column_model.py`](../../src/pydac/core/column_model.py), are:

| Step(s) | Inlet (z=0) | Outlet (z=L) | Notes |
|---------|-------------|--------------|-------|
| `startup`, `ads`, `s_purge`, `s_des` | Dummy/open: `a=0, b=0, d=0` | Dirichlet: `a=0, b=1, d=p_out/p_ref` | Feed enters through inlet fluxes, not through a pressure Robin BC |
| `blow`, `heat`, `w_purge`, `w_des` | Zero gradient: `a=1, b=0, d=0` | Valve Robin: `a=1/(alpha_bd L), b=1, d=p_out/p_ref` | If `alpha_bd = inf`, this collapses to an outlet Dirichlet condition |
| `cool` | Zero gradient: `a=1, b=0, d=0` | Zero gradient: `a=1, b=0, d=0` | Sealed-bed pressure BC |
| `press` (valve) | Valve Robin: `a=1/(alpha_ps L), b=1, d=p_feed/p_ref` | Zero gradient: `a=1, b=0, d=0` | Inlet-driven pressurization |

Important implementation detail:

- `heat`, `w_purge`, and `w_des` default to `alpha_bd = inf`, so they still use the Robin form in the code, but the effective behavior is outlet Dirichlet at `p_out`.
- With `flow.vacuum_outlet_model = "vacuum_pump"` (1D) these four steps build the outlet
  pressure operator with `a = 0` (the Dirichlet form, whatever `alpha_bd`), and the
  outlet-face velocity is replaced by the pump law of §2.7, so `alpha_bd` is not read.

With `flow.press_inlet_model = "feed_blower"` (1D press only), the dormant inlet
pressure operator remains, with `ps_fact = 0`; the inlet-face velocity is replaced
before publishing fields and rebuilding transport fluxes. Conductance is ignored.
The outlet remains sealed. No cell pressure is prescribed by this law.

### 2.4 Radial BCs

| Equation | Center (`r=0`) | Wall (`r=R_b`) |
|----------|----------------|----------------|
| Species | Zero gradient | Zero gradient |
| Pressure | Zero gradient | Zero gradient |
| Bed temperature, 2D | Zero gradient | Robin to the wall temperature: $\lambda_{\mathrm{rad}}\,\partial T/\partial r = h_{\mathrm{in}}(T_w - T)$ |
| Bed temperature, 1D (`Nr=1`) | Not used explicitly | Wall coupling moved into a source term |
| Wall temperature | Axial zero gradient at both ends | Separate 1D wall equation, coupled to the jacket temperature $T_j$ |

### 2.5 Velocity treatment

Axial and radial velocities are recovered algebraically from the selected
pressure law and then constrained according to the physical step:

| Step(s) | Inlet-face handling | Outlet-face handling |
|---------|---------------------|----------------------|
| `startup`, `ads`, `s_purge`, `s_des` | Inlet velocity extrapolated from interior faces | Outlet left free; on 1D `ads`/`startup` with the one-way outlet, set by the check-valve law ([One-way outlet](#one-way-outlet-on-adsorption-schema-11-320)) |
| `blow`, `heat`, `w_purge`, `w_des` | Inlet velocity forced to zero | Outlet left free; with the vacuum pump, set by the pump law (§2.7) |
| `cool` | Inlet velocity forced to zero | Outlet velocity forced to zero |
| `press` | Inlet velocity comes directly from the selected pressure law + Robin pressure BC; with the 1D feed blower, set by the blower law (§4.10) | Outlet velocity forced to zero |

The radial velocity is zero everywhere in 1D. In 2D it is recovered in the
interior and set to zero at the centerline and at the impermeable wall, in every
step.

### 2.6 Inlet-flux modes

| Mode | Steps | How the inlet is handled |
|------|-------|--------------------------|
| Fixed-feed flux | `startup`, `ads`, `s_purge`, `s_des` | Inlet fluxes are computed once from the prescribed feed state and inlet flow input; in 2D the total may be distributed over the radial rings with area-normalized weights |
| Closed inlet | `blow`, `heat`, `w_purge`, `w_des`, `cool` | With the built-in zero flow the inlet fluxes are zero; the inlet velocity is clamped to zero by step |
| Pressure-driven inlet | `press` with valve | Inlet fluxes are rebuilt every RHS call from the current pressure-derived inlet velocity |
| Capacity-limited supply | 1D `press` with feed blower | Same flux rebuild from the supply law; metadata `inlet_flux_source = "feed_blower"` |

### 2.7 Vacuum pump outlet (1D, opt-in)

A vacuum-connected step (`blow`, `heat`, `w_purge`, `w_des`) has two outlet modes,
`flow.vacuum_outlet_model`:

| Mode | Physical picture | Outlet law | Vacuum energy |
|---|---|---|---|
| `"valve"` (default) | a throttling valve into a vacuum reservoir held at `p_out`, which can take any flow | the Robin relation of §2.3; `p_out` is reached asymptotically | isentropic work on the removed gas at the bed pressure (manual 06 §8.2) |
| `"vacuum_pump"` (opt-in, 1D) | a pump of finite speed with pressure control at the setpoint `P_set = p_out` | the law below | the same, plus the standby power $P_{idle}(1-g)$ |

This section is about the second. The fields are described in
[01 — Inputs §4.2](01_inputs.md#42-a-step-ten-names-seven-groups-fixed-order); the law is in
[`core/vacuum_manifold.py`](../../src/pydac/core/vacuum_manifold.py), its energy in
[06 — Post-processing §8.2](06_post_processing.md#82-vacuum-pump-work), and the tests are
[`tests/test_vacuum_manifold.py`](../../tests/test_vacuum_manifold.py) (law) and
[`tests/test_vacuum_pump_idle_power.py`](../../tests/test_vacuum_pump_idle_power.py) (energy).

The outlet superficial velocity on the last axial face is

$$u_{out} = \frac{S(p)}{A}\, g(x), \qquad S(p) = S_0\left(1 - \frac{p_{ult}}{p}\right), \qquad x = \frac{p - P_{set}}{\delta}, \quad \delta = \text{band}\cdot P_{set}$$

with $p$ the pressure of the last cell, $S_0$ = `vacuum_pump_speed_m3_s`, $p_{ult}$ =
`vacuum_pump_ultimate_pressure_Pa` (null is 0), band = `vacuum_setpoint_band_fraction` (null
is `1e-3`) and $A$ the cross-section. The control's throughput fraction $g$ is a C1 smoothstep: $g = 0$ for
$x \le 0$, $3x^2 - 2x^3$ in between, $1$ for $x \ge 1$. So:

- above $P_{set} + \delta$ the pump draws at full speed, at most $S_0/A$;
- inside the band $[P_{set}, P_{set} + \delta]$ the control throttles;
- at and below $P_{set}$ it is closed. The pump never supplies gas, so cooling or adsorption
  can still move the bed pressure below the setpoint.

The law states two things: at full opening the pump removes gas at its speed at the bed
pressure, $S(p)$, and a control reduces the removal to hold $P_{set}$. It does not say how
the control does it, and in this averaged description the bed cannot tell:

| How the control realises $g$ | Pump inlet pressure while removing |
|---|---|
| on/off with an isolation valve: the pump runs a fraction $g$ of the time | $p$ |
| variable-speed drive: the speed is turned down to $g\,S(p)$ | $p$ |
| a valve throttles a fixed-speed pump that keeps running | $p_s = p_{ult} + g\,(p - p_{ult})$ |

The removal is the same, $S_0 (p - p_{ult})\, g / (RT)$ for an ideal gas at one temperature and
no line storage. So $g$ is a throughput fraction, not a calibrated valve opening. It is a
smooth, averaged closure: a real relay switches with a finite period and ripples the
pressure, which the law does not resolve. The realisation matters only for the energy,
through the idle mode below.

What it means for a step:

- **Finite capacity.** A bed that releases gas faster than $S_0$ can take stays above the band
  (heating under vacuum, for example). A `blow` reaches the setpoint later than with the valve
  and, being capacity-limited, can extract less CO₂; as $S_0$ grows the result tends to the
  ideal valve (`blowdown_valve_conductance` very large).
- **Approach to the setpoint.** Without gas release the bed approaches $P_{set}$ algebraically,
  not exponentially ($g \approx 3x^2$ near the setpoint), and the last cell, not the outlet
  face, is held at it.
- **Reopening.** A closed control reopens when heating pushes the pressure back over
  $P_{set}$. The band acts on a millisecond time scale while CVODE may be taking steps of tens of
  seconds, so its first attempt can land on an impossible state; §1.4 rejects it and the step
  continues.
- **Outlet face.** With the pump, the drag-law inversion on the outlet face uses the last cell's
  density instead of the extrapolated face density (which turns negative at very high speed).
  The value is overwritten by the pump law anyway. Post-processing rebuilds the outlet velocity
  with the same law, so at each saved instant it equals the solver's. Dry closed-inlet product
  uses the signed bed inventory loss; integrated scalar states remain mass-balance diagnostics.
  An integral of saved samples needs output refinement (`grid.output_schedule` or `grid.nt`).
- **Vacuum energy.** The pump's electric power is

  $$P_{vac} = \frac{\dot n_{out}\, w_s(p \to p_{amb})}{\eta(p)} + P_{idle}\,(1 - g)$$

  The first term is the formula used with the valve (manual 06 §8.2): the isentropic work on
  the removed gas from the bed pressure to ambient, with the efficiency of Maruyama et al.
  (2020) at the bed pressure. For on/off operation it is the exact time average (on a fraction
  $g$ of the time at full load), if the switching is fast compared with the bed. For a
  variable-speed drive it also assumes that the efficiency does not depend on the speed, which
  is an extra assumption: a pump at reduced speed can be less efficient. A drive at its minimum
  speed may also still remove more than the bed releases, and then needs on/off cycling too.
  The second term is the standby power while the control throttles, set by
  `vacuum_pump_idle_mode`:

  | Idle mode | Plant | $P_{idle}$ = `vacuum_pump_idle_power_W` |
  |---|---|---|
  | `"off"` (default) | on/off with an isolation valve, the pump off while the setpoint holds | 0 (the field must stay null) |
  | `"minimum_speed"` | variable-speed drive at its minimum speed | vendor's absorbed power at minimum speed |
  | `"fixed_speed"` | fixed-speed pump running against the closed control valve | vendor's absorbed power at the ultimate pressure |

  The weight $1 - g$ does not count the pump twice: at $g = 1$ the first term is already its
  whole power at full load, at $g = 0$ only $P_{idle}$ remains. Between the two ends it is an
  interpolation, exact for on/off and approximate for the other two plants, whose real
  part-load power curve it does not resolve. It can dominate: in a heating step held at the
  setpoint $g$ is a few 1e-3, so a 2 kW standby power over 1800 s adds 3.6 MJ against a few kJ
  of isentropic work. A `blow` gets a much smaller absolute increment, because $g = 1$ for most
  of it, but not a negligible one: on the 13X test chain (pump 150 m3/h, 10 kPa) 2 kW add
  16 kJ to its 88 kJ, about 19 %. Start/stop transients, the line volume and the line losses
  at full opening are not modelled. With a positive $P_{idle}$ the
  energy is a larger state in CVODE's error norm, so the profiles move at the solver tolerance
  (about 3e-7) compared with `"off"`.
- **Pressure events.** A pressure target that is "below" and lies at or under the band's upper
  edge $P_{set} + \delta$ may not be reached: the flow throttles above $P_{set}$ and stops at
  or below it. Without cooling or adsorption the step then ends on the plateau (`blow`) or at
  `event_max_time_s`, and with the pump the plateau stop instant is ill-conditioned: it moves by about 1e-3 with the
  tolerances, against 1e-5 with the valve, while the masses and the energy agree to 3e-6. Put
  the target above the band, e.g. `p_out` + 0.5 %; validation warns otherwise (§3.2).

---

## 3. Time-based and event-based stopping

### 3.1 General mechanism

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

### 3.2 Supported event metrics

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

With the vacuum pump (§2.7) the plateau must not be the intended stop. The pressure control holds the
outlet inside the throttle band, where the descent rate crosses the tolerance at a shallow
angle, so the stop instant is sensitive to the solver tolerances. Validation
(`target_inside_band_warning`, called from `config/model_validation.py`) warns when a pump
step that may run in event mode stops on `max_internal_pressure`, `avg_outlet_pressure` or
`min_internal_pressure` going below a target of at most $P_{set} + \delta$ (target = `p_out`,
`p_feed` or the value, plus the offset). The mode is the step's own `termination.mode` unless
an execution override sets one: the global `execution.termination_mode`, then a cycle's
(`execution.cycle_options`, only cycles a CSS run reaches: from `resume_from_cycle`, default 1,
to `max_cycles`), then a step's (`step_options`). Solve flags, restarts and replays are not
resolved, so a step that is never solved can still be warned about.

### 3.3 Event stop rules

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

## 4. Step-by-step details

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
  (`steps.blow.flow.blowdown_valve_conductance`). With
  `flow.vacuum_outlet_model = "vacuum_pump"` (1D) a pump of finite speed replaces the valve and
  `alpha_bd` is not read; keep the pressure target above the throttle band (§2.7).
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

### 4.4 Heating (`heat`)

**Purpose:** heat the bed before desorption.

Implementation summary:

- No inlet flow.
- Hot jacket at `373.15 K`.
- Built-in outlet and feed pressures both `20000 Pa`; the campaign knob
  `low_pressure_Pa` sets them together with the other low-pressure steps.
- `alpha_bd = inf`, so the outlet pressure relation behaves as a direct vacuum boundary. With
  the vacuum pump (§2.7) the outlet is capacity-limited instead: while the bed releases gas
  faster than the pump takes it, the pressure rises above `p_out`.
- Built-in mode is time. In event mode, heating ends when the minimum bed
  temperature reaches the steam condensation temperature `T_cond` on ascent. The
  threshold offset is `0`, so there is no extra margin on top of it.

Important implementation detail:

- `T_cond` is a pressure-dependent condensation temperature evaluated from `p_feed`
- a fixed-duration run can force this step to time mode while keeping the temperature event definition available

### 4.5 Wall purge (`w_purge`)

**Purpose:** pre-heat the evacuated bed with the hot wall until oxygen has
fallen below the purge target, without introducing steam.

Implementation summary:

- It uses the same composition, zero inlet flow, thermal conditions and pressure
  conditions as `heat`, with shorter horizons: nominal `900 s`, maximum `9000 s`.
- The hot jacket is held at `373.15 K`; the inlet is closed and the outlet
  remains tied to the low-pressure boundary (or drawn by the vacuum pump, §2.7).
- Unlike `heat`, it is event-driven by
  `max_bed_oxygen_mole_fraction < 1e-7` (0.1 ppmmol).
- If the oxygen event ends the step, every bed cell meets the target at that
  moment. Reaching the maximum horizon does not show that the target was met.

`w_purge` is a configured canonical step but is deliberately absent from both
default cycle sequences. A case requiring this step must include it
explicitly in `execution.cycle_steps`.

Post-processing follows the ordinary pre-heating branch: temperature,
vacuum-energy, and jacket-energy key performance indicators (KPIs) are retained, with no steam KPIs. By
default the outlet is vented (`mass.vented`); `outlet.collect_product` sends it
to the product line instead ([01 — Inputs §4.2](01_inputs.md#42-a-step-ten-names-seven-groups-fixed-order)).

### 4.6 Steam purge (`s_purge`)

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

### 4.7 Wall-heated desorption (`w_des`)

**Purpose:** continue desorption under vacuum without a purge gas.

Implementation summary:

- No inlet flow.
- Hot jacket at `373.15 K`.
- Outlet remains tied to the vacuum target.
- `alpha_bd = inf`, so the outlet behaves effectively as a direct vacuum boundary; with the
  vacuum pump (§2.7) it is capacity-limited instead.
- Event stop rule: average bed CO2 loading fraction drops below `0.05`.

Compared with `heat`:

- boundary conditions are similar
- `w_des` defaults to a CO₂-loading event
- `heat` has a `T_cond` temperature event but defaults to fixed-duration (time) operation

### 4.8 Steam-assisted desorption (`s_des`)

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

### 4.9 Cooling (`cool`)

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
Internal axial/radial redistribution is resolved by the partial differential equation (PDE) system under sealed boundaries.
By default the step is event-driven: it ends when `max_bed_temperature` goes below
`323.15 K`, with a maximum horizon of `10800 s`. In time mode it runs for
`steps.cool.horizon.nominal_time_s` (`9000 s`), unless the low-pressure floor (§3.3) ends it
sooner.


### 4.10 Pressurization (`press`)

**Purpose:** raise the bed pressure toward the step's feed/header pressure.

Optional supply (`flow.press_inlet_model = "feed_blower"`, schema 11, 3.2.0):

```text
P_set = p_feed; delta = band * P_set; p = pressure of first axial cell
x = (P_set - p) / delta
 g(x) = 0                  x <= 0
      = 3*x*x - 2*x*x*x    0 < x < 1
      = 1                  x >= 1
u_supply = (Q_max / A) * g(x)
n_dot_in = Q_max * g(x) * p_feed / (R * T_feed)
```

The supply shuts at/above the setpoint without backflow. The band defaults to
`1e-3`; `Q_max` is `feed_blower_flow_m3_s` at this press step's feed conditions.
Header-referred velocity is capped; local bed velocity for the same molar flow
scales as `u_supply * (p_feed/p_bed) * (T_bed/T_feed)`. Internal velocities and
pressure gradients must be inspected before making hydrodynamic safety claims.
Closure does not prevent pressure rises from heating, desorption or redistribution.

For an inert isothermal bed with full opening, **volume-average** pressure rises at
`Q_max*p_feed/(eps_total*V_bed)`. A lumped fill-time estimate approximates a
minimum-pressure event only when gradients are small. As capacity grows, a positive
bounded gas demand gives first-cell error approximately
`delta * sqrt(n_required/(3*n_max))`, with `n_max=Q_max*p_feed/(R*T_feed)`.
It does not pin overpressured or net gas-generating beds. First-cell control differs
from an ideal valve's face Dirichlet condition at finite mesh spacing.
The largest lumped relaxation rate is `1.5*Q_max*p_feed/(eps_total*V_bed*delta)`;
use the first-cell volume for the local scale. The derivative vanishes at the exact
setpoint. Small bands/high capacity can be stiff; smoothness does not ensure positivity
or cheap integration.

The unchanged primary event reads **min p over the bed**; a 1% offset below feed
pressure does not prove `g=1`. Record `p_first`, `min p`, `x_first` and `g` at each
stop. Validation warns when an effective ascending pressure target is at or above
the band's lower edge (including above setpoint), honoring threshold source/value,
offset and executed event modes. The existing plateau fallback remains; sufficiently
small capacity can trigger it before target. Disable it explicitly only after checking
that the blower reaches the target. The following summary describes the default valve.

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
- the actual inlet transport rate follows the selected inlet law; the blower does not rescale property correlations

---

## 5. Practical implications

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


## One-way outlet on adsorption (schema 11, 3.2.0)

On 1D ads/startup an optional `flow.ads_outlet_overpressure_Pa = delta_p` replaces the
outlet-face velocity with `u_out = K max(p_last - p_out, 0)`, where `K=(Q_feed/A)/delta_p`
in m/(s Pa). This check valve closes below the outlet pressure; no inflow reservoir is required.
The pressure rise calibrates superficial velocity at the bed, not a compressible mass-flow capacity.
Null retains the Dirichlet outlet. RHS, saved outlet replay and reconstructed profiles use the same law.
Metadata saves version, K, delta_p, outlet/reference pressures and nominal feed velocity.
The linear law is continuous with a derivative kink at opening. A step with the valve runs CVODE BDF
up to order 2 (A-stable): at order 3 the valve adds a fast mode that can cap the step size (one
qualification case took 50 times more RHS evaluations). Steps without the valve keep order 3, so
leaving the field null changes neither the outlet nor the solver settings.
