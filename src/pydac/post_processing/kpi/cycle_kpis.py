"""Pure cycle-level KPI calculations from validated cycle metrics."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from pydac.post_processing.metrics.cycle_data import CycleData
from pydac.post_processing.metrics.cycle_metrics import STEP_KEY_TO_CYCLE_STEP
from pydac.utils.postprocessing.normalization import (
    CycleMetricsAccessor,
    normalize_mass_basis,
    normalize_species,
)
from pydac.core.species_catalog import get_species_properties

ENERGY_METRICS = (
    ("energy.compression", "Compression", "compression"),
    ("energy.vacuum", "Vacuum", "vacuum"),
    ("energy.jacket", "Jacket", "jacket"),
    ("energy.steam", "Steam", "steam"),
    ("energy.condenser", "Condenser", "condenser"),
)

# The condenser duty is released over a temperature range, not at one temperature, and
# most of it is rejected above ambient on cooling water.  `energy.condenser` stays the
# whole duty -- that number was always right -- and this companion carries the part that
# ambient takes for free, so that only the remainder is charged to a chiller.  It is a
# credit, not a sixth term: it never becomes a column of its own.
CONDENSER_ABOVE_AMBIENT_METRIC = "energy.condenser.ambient"
CONDENSER_TEMPERATURE_METRIC = "temperature.condenser"


class LedgerMetricMissing(RuntimeError):
    """Raised when a result predates the condenser split and is read as if current.

    `energy.condenser.ambient` is written by the step post-processor wherever
    `energy.condenser` is written, so a step carrying the duty without the split was
    produced before the 2026-09-08 re-pricing.  Charging it through the legacy branch
    would silently price it on the old ledger -- a 40-45 % difference on the axis an
    optimiser searches -- so reading such a result requires saying so explicitly.
    """

MJ_PER_KWH = 3.6

# The desorption steps.  Which steps discharge the cycle product is a per-step case
# setting (`steps.<step>.outlet.collect_product`), read back from what each step
# recorded; a result written before that setting collected exactly these steps.
# A cycle may execute more than one of them (wall-heated desorption followed by a
# steam pulse), in which case the cycle-level KPIs aggregate over the set rather
# than over one resolved step.
DESORPTION_STEP_KEYS = frozenset({"w_des", "s_des", "s_purge"})

# The direct-air-capture mixture. KPIs that name a species of it report NaN, or are
# skipped, when the case's mixture does not contain it.
DAC_SPECIES = ("CO2", "H2O", "O2", "N2")

STEP_SHORT_NAMES = {
    "startup": "startup",
    "adsorption": "ads",
    "blowdown": "bd",
    "preheating": "ph",
    "wall purge": "wp",
    "steam purge": "sp",
    "desorption": "des",
    "pressurization": "ps",
    "cooldown": "cool",
}


@dataclass(frozen=True, slots=True)
class EquivalentWorkConditions:
    """Temperatures and efficiencies used for equivalent-work conversion."""

    T_low: float = 298.15
    T_high: float = 373.15
    T_condensation: float = 278.15
    turbine_efficiency: float = 0.75
    refrigeration_efficiency: float = 0.5
    #: Minimum approach between the stream and the cooling water.  Sets how far ambient
    #: can carry the condenser duty before a chiller is needed; must match
    #: `T_COOLING_WATER_LIMIT` in the step post-processor, which measures the split.
    cooling_water_approach: float = 5.0
    #: Approach between the stream and the refrigerant, i.e. how much colder the
    #: evaporator runs than the product it delivers.
    chiller_approach: float = 5.0

    def __post_init__(self):
        for name in ("T_low", "T_high", "T_condensation"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")
            object.__setattr__(self, name, value)
        for name in (
            "turbine_efficiency",
            "refrigeration_efficiency",
            "cooling_water_approach",
            "chiller_approach",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)

    @property
    def T_cooling_water_limit(self):
        """Coldest stream temperature cooling water can still reach [K]."""
        return self.T_low + self.cooling_water_approach

    @property
    def T_evaporator(self):
        """Temperature the chiller has to evaporate at to deliver the product [K]."""
        return self.T_condensation - self.chiller_approach


class CycleKPIs:
    """Produce cycle-level physical KPIs without IO or input mutation."""

    def __init__(self, cycle_data, *, allow_legacy_ledger=False):
        """Read one cycle's metrics.

        `allow_legacy_ledger` opts out of the condenser-split gate and restores the
        pre-2026-09-08 charge for steps that never recorded `energy.condenser.ambient`.
        It exists for reading archived results, where the old charge is what those
        numbers actually were. Keep the default `False` for new simulations and
        comparisons so a missing condenser split is reported explicitly.
        """
        if not isinstance(cycle_data, CycleData):
            raise TypeError("cycle_data must be a CycleData instance.")
        self.cycle_data = cycle_data
        self.data = CycleMetricsAccessor(cycle_data.metrics)
        self.allow_legacy_ledger = bool(allow_legacy_ledger)

    @property
    def context(self):
        return self.cycle_data.context

    @property
    def available_species(self):
        return self.context.species

    @property
    def integrated_mass_species(self):
        table = self.cycle_data.metrics.to_dataframe()
        selected = table.loc[
            table["Metric ID"].eq("mass.change")
            & table["Basis"].eq("integrated"),
            "Species",
        ]
        return tuple(dict.fromkeys(selected))

    def _species(self, species, purpose="cycle KPI"):
        token = normalize_species(species)
        if token not in self.available_species:
            raise ValueError(
                f"{purpose} requires {token}; declared mixture={self.available_species}."
            )
        return token

    def present_species(self, species):
        """The given species that the mixture contains, in the given order."""
        return tuple(name for name in species if name in self.available_species)

    def product_species(self):
        """The species of the product: the DAC four first, then any other present."""
        dac = self.present_species(DAC_SPECIES)
        return dac + tuple(name for name in self.available_species if name not in dac)

    def _require_species(self, species, purpose):
        missing = [name for name in species if name not in self.available_species]
        if missing:
            raise ValueError(
                f"{purpose} requires species {tuple(species)}; missing={missing}, "
                f"declared mixture={self.available_species}."
            )

    @staticmethod
    def molecular_weight(species):
        return get_species_properties(species).molecular_weight_kg_per_mol

    # Gather cycle-level mass KPIs from saved metrics
    def initial_mass(self, species, step):
        return self.data.value(
            step=step,
            species=self._species(species, "initial mass"),
            metric_id="mass.initial",
        )

    def final_mass(self, species, step):
        return self.data.value(
            step=step,
            species=self._species(species, "final mass"),
            metric_id="mass.final",
            basis="state",
        )

    def signed_mass_change(self, species, step, basis="preferred"):
        species_token = self._species(species, "mass change")
        basis_token = normalize_mass_basis(basis)
        if basis_token == "preferred":
            basis_token = "state"
        return self.data.value(
            step=step,
            species=species_token,
            metric_id="mass.change",
            basis=basis_token,
        )

    def mass_change(self, species, step, basis="preferred"):
        return abs(self.signed_mass_change(species, step, basis=basis))

    def _has_step_record(self, step_key, metric_id):
        return any(
            self.data.value(
                step=step_key, species=species, metric_id=metric_id, default=None
            )
            is not None
            for species in self.available_species
        )

    def product_step_keys(self):
        """Return the executed steps that discharge product, in execution order.

        A product step is one that recorded what left its vacuum separator
        (``mass.after.vacuum``), so the case's ``outlet.collect_product`` switches
        are read from the data.  A result written before any step recorded it
        collected the desorption steps; a desorption step with a vent record is a
        current result, so it never falls back.

        Read from the cycle's own ``step_order`` and never from a default ordering:
        the step-mode loader fallback lists ``s_purge`` before ``w_des``, which is
        reversed against execution.
        """
        step_order = self.data.step_order
        keys = tuple(
            step_key
            for step_key in step_order
            if self._has_step_record(step_key, "mass.after.vacuum")
        )
        if not keys and not any(
            self._has_step_record(step_key, "mass.vented")
            for step_key in step_order
            if step_key in DESORPTION_STEP_KEYS
        ):
            keys = tuple(
                step_key for step_key in step_order if step_key in DESORPTION_STEP_KEYS
            )
        if not keys:
            raise ValueError(
                f"{self.data.cycle_label} collects product in no step of "
                f"{step_order}; set `outlet.collect_product` on at least one."
            )
        return keys

    def last_product_step_key(self):
        """Return the last executed step that discharges product."""
        return self.product_step_keys()[-1]

    def last_desorption_step_key(self):
        """Return the step whose end is the end of regeneration.

        A state reading, independent of where the step's outlet goes: the bed the
        cycle carries into cooldown is the one left by the last desorption step,
        whether its outlet was collected or vented.
        """
        keys = tuple(
            step_key
            for step_key in self.data.step_order
            if step_key in DESORPTION_STEP_KEYS
        )
        if not keys:
            raise ValueError(
                f"{self.data.cycle_label} executes no desorption step; expected at "
                f"least one of {sorted(DESORPTION_STEP_KEYS)} in "
                f"{self.data.step_order}."
            )
        return keys[-1]

    def recovered_mass_by_step(self, species="CO2"):
        """Return the vacuum-discharged mass of one species, per product step [kg]."""
        species_token = self._species(species, "recovered mass")
        recovered = {}
        for step_key in self.product_step_keys():
            value = self.data.value(
                step=step_key,
                species=species_token,
                metric_id="mass.after.vacuum",
                default=None,
            )
            if value is not None:
                recovered[step_key] = abs(float(value))
            else:
                recovered[step_key] = self.mass_change(
                    species_token, step_key, basis="preferred"
                )
        return recovered

    def recovered_mass(self, species="CO2"):
        """Return the cycle product of one species, summed over every product step."""
        return float(sum(self.recovered_mass_by_step(species).values()))

    def vented_mass_by_step(self, species="CO2"):
        """Return the mass vented by each step whose outlet is not collected [kg].

        A purge discharge is *lost*, not recovered, so it is never part of any product
        KPI -- but in a cycle whose preheating exists to sweep out the residual O2/N2
        left by blowdown it is the largest single lever on recovery, and reporting it
        is the whole point of recording it.  Positive means the species left the bed.

        Steps that recorded no vent are omitted, so a result written before the metric
        existed returns an empty mapping rather than a misleading zero.
        """
        species_token = self._species(species, "vented mass")
        vented = {}
        for step_key in self.data.step_order:
            value = self.data.value(
                step=step_key,
                species=species_token,
                metric_id="mass.vented",
                default=None,
            )
            if value is not None:
                vented[step_key] = float(value)
        return vented

    def vented_mass(self, species="CO2"):
        """Return the mass of one species vented over the whole cycle [kg]."""
        return float(sum(self.vented_mass_by_step(species).values()))

    def has_vent_record(self, species="CO2"):
        """Return whether any step actually measured a vent for this species.

        A result written before ``mass.vented`` existed reports no vent at all, which
        is not the same statement as a vent of zero; anything that displays the number
        must be able to tell the two apart.
        """
        return bool(self.vented_mass_by_step(species))

    def vented_fraction_of_adsorbed(self, species="CO2"):
        """Return the vented mass as a fraction of what adsorption took up [-]."""
        adsorbed = self.mass_change(species, "adsorption", basis="preferred")
        if adsorbed <= 1e-12:
            return 0.0
        return float(self.vented_mass(species) / adsorbed)

    def final_gas_mass(self, species, step):
        """Return one saved final gas-phase species inventory [kg]."""
        return self.data.value(
            step=step,
            species=self._species(species, "final gas mass"),
            metric_id="mass.gas.final",
            basis="state",
        )

    def residual_gas_ppm_by_step(self, species=None):
        """Return final gas-phase species fractions [ppmmol] for every step."""
        requested = self.available_species if species is None else species
        species_tokens = tuple(self._species(item, "residual gas ppm") for item in requested)
        rows = []

        for step_key in self.data.step_order:
            try:
                moles = {
                    species_name: self.final_gas_mass(species_name, step_key)
                    / self.molecular_weight(species_name)
                    for species_name in self.available_species
                }
            except KeyError as exc:
                raise ValueError(
                    "Residual gas ppm requires mass.gas.final for every species "
                    f"and step; missing data in '{step_key}'."
                ) from exc

            total_moles = float(sum(moles.values()))
            row = {"Step": STEP_KEY_TO_CYCLE_STEP[step_key]}
            for species_token in species_tokens:
                row[species_token] = (
                    1e6 * moles[species_token] / total_moles
                    if total_moles > 0.0
                    else 0.0
                )
            rows.append(row)

        return pd.DataFrame(rows).set_index("Step")[list(species_tokens)]

    def mass_change_table(
        self,
        species=None,
        basis="integrated",
    ):
        requested = self.available_species if species is None else species
        species_tokens = tuple(self._species(item, "mass-change table") for item in requested)
        basis_token = normalize_mass_basis(basis)
        rows = []

        for step_key in self.data.step_order:
            row = {"Step": STEP_KEY_TO_CYCLE_STEP[step_key]}
            for species_token in species_tokens:
                try:
                    value = self.signed_mass_change(
                        species_token,
                        step_key,
                        basis=basis_token,
                    )
                except KeyError:
                    value = 0.0
                row[species_token] = value
            rows.append(row)

        return pd.DataFrame(rows).set_index("Step")[list(species_tokens)]

    def mass_change_percent_of_adsorption(self, species=None):
        """Return signed state changes as percent of adsorption uptake."""
        requested = self.integrated_mass_species if species is None else species
        species_tokens = tuple(
            self._species(item, "adsorption-normalized mass change")
            for item in requested
        )

        references = {
            species_token: abs(
                self.signed_mass_change(
                    species_token,
                    "adsorption",
                    basis="state",
                )
            )
            for species_token in species_tokens
        }
        rows = []
        for step_key in self.data.step_order:
            row = {"Step": STEP_KEY_TO_CYCLE_STEP[step_key]}
            for species_token in species_tokens:
                reference = references[species_token]
                row[species_token] = (
                    100.0
                    * self.signed_mass_change(
                        species_token,
                        step_key,
                        basis="state",
                    )
                    / reference
                    if reference > 1e-12
                    else np.nan
                )
            rows.append(row)
        return pd.DataFrame(rows).set_index("Step")[list(species_tokens)]

    # Gather cycle-level time KPIs from saved metrics
    def cycle_time(self):
        return float(
            sum(
                self.data.value(
                    step=step_key,
                    species="system",
                    metric_id="step.duration",
                )
                for step_key in self.data.step_order
            )
        )

    def step_saturation(self, step):
        """Return final bed-average CO2 saturation for one step [%]."""
        self._require_species(("CO2",), "CO2 saturation KPI")
        return self.data.value(
            step=step,
            species="CO2",
            metric_id="bed.saturation.co2",
        )
        
    def cycle_time_breakdown(self):
        total_time = max(self.cycle_time(), 0.0)
        cumulative = 0.0
        rows = []
        for step_key in self.data.step_order:
            duration = self.data.value(
                step=step_key,
                species="system",
                metric_id="step.duration",
            )
            cumulative += duration
            rows.append(
                {
                    "Step": STEP_KEY_TO_CYCLE_STEP[step_key],
                    "Duration [s]": duration,
                    "Fraction [-]": duration / total_time if total_time > 0 else 0.0,
                    "Cumulative time [s]": cumulative,
                }
            )
        return pd.DataFrame(rows)

    # Constraint: preheating  must be above minimum bed and below steam condensation temperature
    def preheating_temperatures(self):
        """Return saved minimum-bed and steam-condensation temperatures [K]."""
        heating_step = next(
            (step for step in ("heat", "w_purge") if step in self.data.step_order),
            None,
        )
        if heating_step is None:
            raise KeyError(
                f"Neither 'heat' nor 'w_purge' is present in {self.data.cycle_label}."
            )
        return {
            "bed_minimum_K": self.data.value(
                step=heating_step,
                species="system",
                metric_id="temperature.bed.minimum",
            ),
            "steam_condensation_K": self.data.value(
                step=heating_step,
                species="system",
                metric_id="temperature.steam.condensation",
                default=None,
            ),
        }

    # Main cycle-level KPIs 
    def normalize_by_recovered_co2(self, value, penalty=1e6):
        self._require_species(("CO2",), "CO2-specific normalization")
        recovered_co2 = self.recovered_mass("CO2")
        if recovered_co2 <= 1e-12:
            return float(penalty)
        return float(value) / recovered_co2
    
    def productivity(self):
        """Return recovered-CO2 productivity [kgCO2/m3/day]."""
        self._require_species(("CO2",), "DAC productivity")
        bed_volume = self.context.value("bed.volume")
        cycle_time = self.cycle_time()
        if bed_volume <= 0.0 or cycle_time <= 0.0:
            raise ValueError("Bed volume and cycle time must be greater than zero.")
        return self.recovered_mass("CO2") / (bed_volume * cycle_time) * 86400.0

    def cyclic_working_capacity(self, species="CO2"):
        """Return end-adsorption to end-desorption swing [mol/kg adsorbent]."""
        species_token = self._species(species, "cyclic working capacity")
        adsorbent_mass = self.context.adsorbent_mass
        if adsorbent_mass <= 0.0:
            raise ValueError("Adsorbent mass must be greater than zero.")

        final_adsorption = self.final_mass(species_token, "adsorption")
        # State reading, not an extensive quantity: the bed loading that matters is
        # the one the bed carries into cooldown, i.e. after the *last* desorption step.
        final_desorption = self.final_mass(
            species_token, self.last_desorption_step_key()
        )
        return (
            (final_adsorption - final_desorption)
            / self.molecular_weight(species_token)
            / adsorbent_mass
        )
        
    def purity(self):
        self._require_species(("CO2",), "product purity")
        masses = {
            species: self.recovered_mass(species)
            for species in self.product_species()
        }
        moles = {
            species: masses[species] / self.molecular_weight(species)
            for species in masses
        }
        total_moles = sum(moles.values())
        dry_moles = total_moles - moles.get("H2O", 0.0)

        return {
            "masses_kg": masses,
            "moles": moles,
            "purity_wet": float(moles["CO2"] / total_moles) if total_moles > 0 else 0.0,
            "purity_dry": float(moles["CO2"] / dry_moles) if dry_moles > 0 else 0.0,
            "oxygen_ppm": (
                float(1e6 * moles["O2"] / total_moles) if total_moles > 0 else 0.0
            )
            if "O2" in moles
            else np.nan,
            "nitrogen_ppm": (
                float(1e6 * moles["N2"] / total_moles) if total_moles > 0 else 0.0
            )
            if "N2" in moles
            else np.nan,
        }

    def product_composition(self):
        """Return recovered-product composition after vacuum discharge."""
        product = self.purity()
        masses = product["masses_kg"]
        moles = product["moles"]
        total_mass = float(sum(masses.values()))
        total_moles = float(sum(moles.values()))
        dry_moles = total_moles - moles.get("H2O", 0.0)

        rows = []
        for species in masses:
            rows.append(
                {
                    "Species": species,
                    "Recovered mass [g]": 1000.0 * masses[species],
                    "Mass fraction [%]": (
                        100.0 * masses[species] / total_mass
                        if total_mass > 0.0
                        else 0.0
                    ),
                    "Wet mole fraction [%mol]": (
                        100.0 * moles[species] / total_moles
                        if total_moles > 0.0
                        else 0.0
                    ),
                    "Dry mole fraction [%mol]": (
                        np.nan
                        if species == "H2O"
                        else (
                            100.0 * moles[species] / dry_moles
                            if dry_moles > 0.0
                            else 0.0
                        )
                    ),
                    "Wet content [ppmmol]": (
                        1e6 * moles[species] / total_moles
                        if total_moles > 0.0
                        else 0.0
                    ),
                }
            )
        return pd.DataFrame(rows)

    def recovery(self, species="CO2"):
        species_token = self._species(species, "recovery")
        adsorbed = self.mass_change(
            species_token, "adsorption", basis="preferred"
        )
        recovered = self.recovered_mass(species_token)
        return float(recovered / adsorbed) if adsorbed > 1e-12 else 0.0
    
    def equivalent_work(self, conditions=None):
        """Return total equivalent work per recovered CO2 [MJ/kgCO2]."""
        equivalent = self.energy_breakdown(
            specific=False,
            equivalent=True,
            absolute=True,
            conditions=conditions,
        )
        total_equivalent_energy = float(equivalent.to_numpy(dtype=float).sum())
        return self.normalize_by_recovered_co2(total_equivalent_energy)

    # Steam mass and steam-to-CO2 ratio, over every executed step
    def steam_mass_used_by_step(self):
        """Return the injected steam mass per executed step [kg].

        Every step, not only the product steps: the steam of a vented steam purge
        was raised and paid for all the same.
        """
        if "H2O" not in self.available_species:
            # No H2O in the mixture: no step can have injected steam.
            return {step_key: 0.0 for step_key in self.data.step_order}
        return {
            step_key: float(
                self.data.value(
                    step=step_key,
                    species="H2O",
                    metric_id="steam.mass",
                    default=0.0,
                )
                or 0.0
            )
            for step_key in self.data.step_order
        }

    def steam_mass_used(self):
        """Return the steam injected over the whole cycle [kg]."""
        return float(sum(self.steam_mass_used_by_step().values()))

    def steam_to_co2_ratio(self):
        """Return supplied steam per recovered CO2 [kg/kgCO2]."""
        steam_mass = self.steam_mass_used()
        if steam_mass <= 0.0:
            return None
        recovered_co2 = self.recovered_mass("CO2")
        return float(steam_mass / recovered_co2) if recovered_co2 > 1e-12 else np.inf

    def desorption_process_check(self, species=None):
        requested = self.integrated_mass_species if species is None else species
        species_tokens = tuple(
            self._species(item, "desorption process check") for item in requested
        )
        # One block per product step: with a wall-heated desorption followed by a
        # steam pulse the product crosses the separator twice, and a closure written
        # on one of the two steps is not a closure of the regeneration.
        steam_by_step = (
            self.steam_mass_used_by_step()
            if "H2O" in self.available_species
            else {}
        )
        rows = []

        for step_key in self.product_step_keys():
            # External steam is credited to whichever step recorded it, rather than
            # only to `s_des`; otherwise a steam pulse on `s_purge` shows up as a
            # closure gap of exactly the injected steam mass.
            steam_mass = float(steam_by_step.get(step_key, 0.0))
            for species_token in species_tokens:
                initial_bed = self.initial_mass(species_token, step_key)
                final_bed = self.final_mass(species_token, step_key)
                delta_bed = self.signed_mass_change(
                    species_token, step_key, basis="integrated"
                )
                external_input = steam_mass if species_token == "H2O" else 0.0
                bed_outlet = external_input - delta_bed

                condenser_vapor = self.data.value(
                    step=step_key,
                    species=species_token,
                    metric_id="mass.after.condenser",
                    default=bed_outlet,
                )
                vacuum_vapor = self.data.value(
                    step=step_key,
                    species=species_token,
                    metric_id="mass.after.vacuum",
                    default=condenser_vapor,
                )
                condenser_vapor = float(condenser_vapor)
                vacuum_vapor = float(vacuum_vapor)
                condenser_liquid = bed_outlet - condenser_vapor
                vacuum_liquid = condenser_vapor - vacuum_vapor
                gap = (
                    initial_bed
                    + external_input
                    - condenser_liquid
                    - vacuum_liquid
                    - vacuum_vapor
                    - final_bed
                )
                reference = max(
                    initial_bed + external_input, final_bed + bed_outlet, 1e-12
                )

                rows.append(
                    {
                        "Step": STEP_KEY_TO_CYCLE_STEP[step_key],
                        "Species": species_token,
                        "Initial bed inventory [kg]": initial_bed,
                        "External input [kg]": external_input,
                        "Bed outlet before separation [kg]": bed_outlet,
                        "Condenser vapor outlet [kg]": condenser_vapor,
                        "Condenser liquid removed [kg]": condenser_liquid,
                        "Vacuum discharge vapor [kg]": vacuum_vapor,
                        "Vacuum liquid removed [kg]": vacuum_liquid,
                        "Final bed inventory [kg]": final_bed,
                        "Closure gap [kg]": gap,
                        "Relative error [%]": 100.0 * gap / reference,
                    }
                )
        return pd.DataFrame(rows)

    def h2o_to_co2_ratio(self, basis="mass"):
        self._require_species(("CO2",), "H2O/CO2 product ratio")
        basis_token = str(basis).strip().lower()
        if basis_token not in {"mass", "mole", "mol"}:
            raise ValueError("basis must be 'mass' or 'mole'.")
        if "H2O" not in self.available_species:
            return np.nan
        m_co2 = self.recovered_mass("CO2")
        m_h2o = self.recovered_mass("H2O")

        if basis_token == "mass":
            return float(m_h2o / m_co2) if m_co2 > 1e-12 else np.inf
        if basis_token in {"mole", "mol"}:
            n_co2 = m_co2 / self.molecular_weight("CO2")
            n_h2o = m_h2o / self.molecular_weight("H2O")
            return float(n_h2o / n_co2) if n_co2 > 1e-12 else np.inf
        raise ValueError("basis must be 'mass' or 'mole'.")

    def step_h2o_to_co2_mass_ratio(self, step):
        """Return |state delta H2O| / |state delta CO2| for one cycle step."""
        self._require_species(("CO2",), "step H2O/CO2 ratio")
        if "H2O" not in self.available_species:
            return np.nan
        m_co2 = self.mass_change("CO2", step, basis="state")
        m_h2o = self.mass_change("H2O", step, basis="state")
        return float(m_h2o / m_co2) if m_co2 > 1e-12 else np.inf

    def regeneration_h2o_to_co2_mass_ratio(self):
        """Return the H2O/CO2 state-mass ratio over the whole regeneration.

        A ratio is not additive, so this is the ratio of the sums rather than a sum
        of ratios: with a wall-heated desorption followed by a steam pulse the two
        steps move very different amounts of water, and the ratio of either one alone
        is not the ratio of the regeneration.  On a single-desorption cycle it
        collapses onto ``step_h2o_to_co2_mass_ratio`` of that step.
        """
        self._require_species(("CO2",), "regeneration H2O/CO2 ratio")
        if "H2O" not in self.available_species:
            return np.nan
        product_steps = self.product_step_keys()
        m_co2 = float(
            sum(
                self.mass_change("CO2", step_key, basis="state")
                for step_key in product_steps
            )
        )
        m_h2o = float(
            sum(
                self.mass_change("H2O", step_key, basis="state")
                for step_key in product_steps
            )
        )
        return float(m_h2o / m_co2) if m_co2 > 1e-12 else np.inf

    # Energy-to-equivalent-work conversion factors for cycle-level KPIs
    @staticmethod
    def thermodynamic_conversion_factors(conditions=None):
        """Return heat-to-work Carnot factor and refrigeration COP."""
        if conditions is None:
            conditions = EquivalentWorkConditions()
        if not isinstance(conditions, EquivalentWorkConditions):
            raise TypeError("conditions must be an EquivalentWorkConditions instance.")

        carnot_efficiency = conditions.turbine_efficiency * max(
            0.0,
            1.0 - conditions.T_low / conditions.T_high,
        )
        if (
            conditions.refrigeration_efficiency > 0.0
            and conditions.T_condensation < conditions.T_low
        ):
            refrigeration_cop = conditions.refrigeration_efficiency * (
                conditions.T_condensation
                / max(conditions.T_low - conditions.T_condensation, 1e-32)
            )
        else:
            refrigeration_cop = 1.0
        return float(carnot_efficiency), float(refrigeration_cop)

    @staticmethod
    def chiller_cop(conditions=None):
        """Return the COP of the chiller that delivers the product temperature.

        `thermodynamic_conversion_factors` prices refrigeration as if the evaporator sat
        at the product temperature itself.  A real machine has to run an approach colder
        than what it delivers, which is the temperature that sets its COP.
        """
        if conditions is None:
            conditions = EquivalentWorkConditions()
        if not isinstance(conditions, EquivalentWorkConditions):
            raise TypeError("conditions must be an EquivalentWorkConditions instance.")

        T_evaporator = conditions.T_evaporator
        if conditions.refrigeration_efficiency <= 0.0 or T_evaporator >= conditions.T_low:
            return 1.0
        lift = max(conditions.T_low - T_evaporator, 1e-32)
        return float(conditions.refrigeration_efficiency * T_evaporator / lift)

    def condenser_temperature(self, step):
        """Return the step's recorded condenser temperature [K], or None.

        `None` is a step without a condenser, or a result written before the condenser
        temperature was a step setting; that result was computed at 278.15 K, the
        default `EquivalentWorkConditions.T_condensation`.
        """
        value = self.data.value(
            step=step,
            species="system",
            metric_id=CONDENSER_TEMPERATURE_METRIC,
            default=None,
        )
        return None if value is None else float(value)

    def condenser_conditions(self, step, conditions=None):
        """Return `conditions` with the chiller at this step's condenser temperature.

        Each condenser is priced at the temperature it delivers; a step that recorded
        none keeps `conditions.T_condensation`.
        """
        if conditions is None:
            conditions = EquivalentWorkConditions()
        T_condenser = self.condenser_temperature(step)
        if T_condenser is None:
            return conditions
        return replace(conditions, T_condensation=T_condenser)

    def condenser_pricing(self, conditions=None):
        """Return `{step key: (condenser temperature [K], chiller COP)}` per condenser step."""
        return {
            step_key: (
                self.condenser_temperature(step_key),
                self.chiller_cop(self.condenser_conditions(step_key, conditions)),
            )
            for step_key in self.data.step_order
            if self.condenser_temperature(step_key) is not None
        }

    def energy_value(self, step, metric_id, default=0.0):
        supported_ids = {definition[0] for definition in ENERGY_METRICS}
        supported_ids.add(CONDENSER_ABOVE_AMBIENT_METRIC)
        if metric_id not in supported_ids:
            raise ValueError(f"Unsupported energy metric '{metric_id}'.")
        value = self.data.value(
            step=step,
            species="system",
            metric_id=metric_id,
            default=default,
        )
        return float(value)

    def condenser_duty_above_ambient(self, step):
        """Return the condenser duty [MJ] rejected above ambient, or None if unrecorded.

        `None` and `0.0` mean different things here: `None` is a result produced before
        the split existed, `0.0` is a step whose whole duty needs a chiller.

        A step that recorded a condenser duty but no split predates the re-pricing, and
        raises `LedgerMetricMissing` unless `allow_legacy_ledger` was set.  A step with
        no condenser duty at all has nothing to split and returns `None` either way.
        """
        value = self.data.value(
            step=step,
            species="system",
            metric_id=CONDENSER_ABOVE_AMBIENT_METRIC,
            default=None,
        )
        if value is None and not self.allow_legacy_ledger:
            duty = self.data.value(
                step=step,
                species="system",
                metric_id="energy.condenser",
                default=None,
            )
            if duty is not None:
                raise LedgerMetricMissing(
                    f"Step {step!r} records energy.condenser={float(duty):g} MJ but no "
                    f"{CONDENSER_ABOVE_AMBIENT_METRIC}, so it predates the "
                    "2026-09-08 condenser split and would be charged on the old "
                    "ledger.  Re-run the step with the current post-processor, or "
                    "pass allow_legacy_ledger=True to read it as the archive it is."
                )
        return None if value is None else float(value)

    def convert_energy_to_equivalent_work(
        self,
        metric_id,
        value,
        *,
        absolute=False,
        conditions=None,
        above_ambient=None,
    ):
        """Convert one saved energy [MJ] into equivalent work [MJ].

        `above_ambient` applies to `energy.condenser` only: it is the share of the duty
        rejected before the stream reaches the cooling-water limit, which costs no work.
        Passing it switches the charge onto the two-stage scheme -- cooling water down to
        ambient, then one chiller for the rest.  Leaving it `None` means the split was
        never measured for this result, and the legacy charge is reproduced unchanged
        rather than guessed at.
        """
        value = float(value)
        carnot_efficiency, refrigeration_cop = (
            self.thermodynamic_conversion_factors(conditions)
        )

        if metric_id in {"energy.compression", "energy.vacuum"}:
            return abs(value) if absolute else value

        if metric_id == "energy.condenser":
            if above_ambient is None:
                converted = abs(value) / max(refrigeration_cop, 1e-32)
                return converted if absolute else np.sign(value) * converted
            # Cooling water carries the duty down to the ambient approach at no work
            # cost -- the same credit the jacket's negative branch already gets below --
            # and one chiller carries what is left.
            chargeable = value - float(above_ambient)
            converted = abs(chargeable) / max(self.chiller_cop(conditions), 1e-32)
            return converted if absolute else np.sign(value) * converted

        if metric_id == "energy.jacket":
            if value < 0.0:
                # Ambient cooling-water heat rejection has no work penalty.
                return 0.0
            converted = abs(value) * carnot_efficiency
            return converted if absolute else np.sign(value) * converted

        if metric_id == "energy.steam":
            if value >= 0.0:
                converted = abs(value) * carnot_efficiency
                return converted if absolute else np.sign(value) * converted
            converted = abs(value) / max(refrigeration_cop, 1e-32)
            return converted if absolute else -converted

        raise ValueError(f"Unsupported energy metric '{metric_id}'.")

    def energy_breakdown(
        self,
        *,
        specific=False,
        equivalent=False,
        absolute=False,
        conditions=None,
    ):
        """Return the per-step energy contribution table."""
        recovered_co2 = self.recovered_mass("CO2")
        specific_penalty = 1e6 if specific else 0.0
        rows = []

        for step_key in self.data.step_order:
            row = {"Step": STEP_KEY_TO_CYCLE_STEP[step_key]}
            for metric_id, display_name, _ in ENERGY_METRICS:
                value = self.energy_value(step_key, metric_id, default=0.0)
                if equivalent:
                    # The raw table keeps the whole condenser duty -- it is a duty, and
                    # it is right.  Only the equivalent-work view splits it, and only
                    # where the split was measured.
                    above_ambient = (
                        self.condenser_duty_above_ambient(step_key)
                        if metric_id == "energy.condenser"
                        else None
                    )
                    value = self.convert_energy_to_equivalent_work(
                        metric_id,
                        value,
                        absolute=absolute,
                        conditions=(
                            self.condenser_conditions(step_key, conditions)
                            if metric_id == "energy.condenser"
                            else conditions
                        ),
                        above_ambient=above_ambient,
                    )
                elif absolute:
                    value = abs(value)

                if specific:
                    value = (
                        specific_penalty
                        if recovered_co2 <= 1e-12
                        else value / recovered_co2
                    )
                row[display_name] = float(value)
            rows.append(row)

        columns = [definition[1] for definition in ENERGY_METRICS]
        return pd.DataFrame(rows).set_index("Step")[columns]

    def electrical_energy_breakdown(self, *, specific=False):
        """Return direct compression/vacuum demand [kWh or kWh/kgCO2]."""
        table = self.energy_breakdown(specific=specific, equivalent=False)
        return table[["Compression", "Vacuum"]] / MJ_PER_KWH

    def thermal_energy_breakdown(self, *, specific=False):
        """Return non-negative heating/cooling utility duties [MJ or MJ/kgCO2]."""
        table = self.energy_breakdown(specific=specific, equivalent=False)
        thermal = pd.DataFrame(index=table.index)
        thermal["Jacket heating"] = table["Jacket"].clip(lower=0.0)
        thermal["Steam heating"] = table["Steam"].clip(lower=0.0)
        thermal["Jacket cooling"] = (-table["Jacket"]).clip(lower=0.0).abs()
        thermal["Condenser cooling"] = table["Condenser"].abs()
        return thermal

    def condenser_cooling_split(self, *, specific=False):
        """Split the condenser cooling duty into what ambient takes and what it costs.

        Same per-step total as the "Condenser cooling" column of
        `thermal_energy_breakdown`; this only says where the duty goes.  Under
        `allow_legacy_ledger` a step recorded before the split existed reports its whole
        duty as chiller duty, which is what the ledger charged it for; without it such a
        step raises rather than being read on the old ledger.
        """
        recovered_co2 = self.recovered_mass("CO2")
        rows = []
        for step_key in self.data.step_order:
            duty = abs(self.energy_value(step_key, "energy.condenser", default=0.0))
            ambient = self.condenser_duty_above_ambient(step_key)
            ambient = 0.0 if ambient is None else min(abs(ambient), duty)
            values = {"Condenser to chiller": duty - ambient,
                      "Condenser to ambient": ambient}
            if specific:
                if recovered_co2 <= 1e-12:
                    # Same penalty `energy_breakdown` applies, carried by the charged
                    # part alone so the two still sum to the column they split.
                    values = {"Condenser to chiller": 1e6, "Condenser to ambient": 0.0}
                else:
                    values = {
                        name: value / recovered_co2 for name, value in values.items()
                    }
            rows.append({"Step": STEP_KEY_TO_CYCLE_STEP[step_key], **values})
        return pd.DataFrame(rows).set_index("Step")

    def equivalent_work_breakdown(self, *, specific=True, conditions=None):
        """Return absolute work-equivalent contributions [kWh_eq or kWh_eq/kgCO2]."""
        table = self.energy_breakdown(
            specific=specific,
            equivalent=True,
            absolute=True,
            conditions=conditions,
        )
        return table / MJ_PER_KWH

    def specific_energy_breakdown(self):
        """Return the historical flat keys for raw specific energies."""
        table = self.energy_breakdown(specific=True)
        flat = {}
        for step in table.index:
            step_short = STEP_SHORT_NAMES[step]
            for _, display_name, energy_short in ENERGY_METRICS:
                flat[f"{step_short}_{energy_short}"] = float(
                    table.loc[step, display_name]
                )
        return flat
    
    # Mass balance closure checks for each step and over the full cycle
    @staticmethod
    def _relative_gap_pct(gap, *references, floor=1e-9):
        reference = max((abs(float(value)) for value in references), default=0.0)
        if reference < floor:
            return 0.0
        return 100.0 * float(gap) / reference

    def bed_closure_by_step(self, species=None):
        requested = self.integrated_mass_species if species is None else species
        species_tokens = tuple(
            self._species(item, "integrated bed closure") for item in requested
        )
        unsupported = set(species_tokens) - set(self.integrated_mass_species)
        if unsupported:
            raise ValueError(
                "Integrated mass closure is unavailable for species without saved "
                "integrated mass metrics; "
                f"received {sorted(unsupported)}."
            )

        rows = []
        for step_key in self.data.step_order:
            step = STEP_KEY_TO_CYCLE_STEP[step_key]
            for species_token in species_tokens:
                initial = self.initial_mass(species_token, step_key)
                final = self.final_mass(species_token, step_key)
                delta_integrated = self.signed_mass_change(
                    species_token, step_key, basis="integrated"
                )
                delta_state = self.signed_mass_change(
                    species_token, step_key, basis="state"
                )
                gap = delta_integrated - delta_state
                rows.append(
                    {
                        "Step": step,
                        "Species": species_token,
                        "Initial bed inventory [kg]": initial,
                        "Final bed inventory [kg]": final,
                        "Delta m integrated [kg]": delta_integrated,
                        "Delta m state [kg]": delta_state,
                        "Closure gap [kg]": gap,
                        "Relative error [%]": self._relative_gap_pct(
                            gap, delta_state, delta_integrated
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def bed_closure_over_cycle(self, species=None):
        requested = self.integrated_mass_species if species is None else species
        species_tokens = tuple(
            self._species(item, "integrated cycle closure") for item in requested
        )
        unsupported = set(species_tokens) - set(self.integrated_mass_species)
        if unsupported:
            raise ValueError(
                "Integrated mass closure is unavailable for species without saved "
                "integrated mass metrics; "
                f"received {sorted(unsupported)}."
            )

        first_step = self.data.step_order[0]
        last_step = self.data.step_order[-1]
        rows = []
        for species_token in species_tokens:
            initial = self.initial_mass(species_token, first_step)
            final = self.final_mass(species_token, last_step)
            delta_integrated = float(
                sum(
                    self.signed_mass_change(
                        species_token, step_key, basis="integrated"
                    )
                    for step_key in self.data.step_order
                )
            )
            delta_state = final - initial
            gap = delta_integrated - delta_state
            rows.append(
                {
                    "Species": species_token,
                    "Initial bed inventory [kg]": initial,
                    "Final bed inventory [kg]": final,
                    "Delta m integrated [kg]": delta_integrated,
                    "Delta m state [kg]": delta_state,
                    "Closure gap [kg]": gap,
                    "Relative error [%]": self._relative_gap_pct(
                        gap, delta_integrated, delta_state
                    ),
                }
            )
        return pd.DataFrame(rows)

    def mass_transfer_summary(self, species=None):
        requested = self.available_species if species is None else species
        rows = []
        for item in requested:
            species_token = self._species(item, "mass-transfer summary")
            rows.append(
                {
                    "Species": species_token,
                    "Adsorbed in adsorption [kg]": self.mass_change(
                        species_token, "adsorption", basis="preferred"
                    ),
                    "Desorbed in desorption [kg]": float(
                        sum(
                            self.mass_change(
                                species_token, step_key, basis="preferred"
                            )
                            for step_key in self.product_step_keys()
                        )
                    ),
                    "Recovered after vacuum discharge [kg]": self.recovered_mass(
                        species_token
                    ),
                }
            )
        return pd.DataFrame(rows)
