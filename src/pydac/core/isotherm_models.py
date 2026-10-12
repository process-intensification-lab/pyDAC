"""
pyDAC - Adsorption equilibrium models.

This module defines the thermodynamic isotherm interfaces and concrete
equilibrium correlations used by the packed-bed model, including dry,
co-adsorption, and parameterized competitive-Sips formulations.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.constants import gas_constant as R_gas
from pydac.utils.model_registry import EquilibriumRegistryEntry


class Thermodynamics(ABC):
    """
    Base interface for adsorption equilibrium models.

    All subclasses must implement `isotherm`, which returns at least the
    equilibrium loading plus optional model-specific auxiliary terms.
    """

    @abstractmethod
    def isotherm(self, p, T, q_co=None):
        """
        Evaluate adsorption equilibrium.

        Parameters
        ----------
        p : float or np.ndarray
            Partial pressure [Pa].
        T : float or np.ndarray
            Temperature [K].
        q_co : float or np.ndarray, optional
            Co-adsorbed loading [mol/kg], used by multicomponent models.

        Returns
        -------
        tuple
            `(q_eq, aux_1, aux_2)` where `q_eq` is the equilibrium loading [mol/kg].
        """
        raise NotImplementedError


# Public name-based interface used by every independent or coupled model.
class EquilibriumSystem:
    """Named-species contract shared by independent and coupled equilibria.

    Pressures are partial pressures in Pa, temperature is in K, and loadings
    are in mol/kg.  Coupled systems may inspect any declared current loading;
    every implementation returns only the adsorbates it owns.
    """

    species: tuple[str, ...] = ()
    component_models: Mapping[str, Thermodynamics] = MappingProxyType({})

    def equilibrium_loadings(
        self,
        *,
        partial_pressures_Pa: Mapping[str, Any],
        temperature_K: Any,
        current_loadings_mol_per_kg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return equilibrium loadings by species name."""
        raise NotImplementedError


class TothIsotherm(Thermodynamics):
    """
    CO2 Toth isotherm with optional WADST co-adsorption correction.

    The class keeps both formulations available:
    - dry Toth (`dry_isotherm`)
    - WADST blend (`co_isotherm_wadst`)

    The active routine (`dry_isotherm` vs `co_isotherm_wadst`) is selected
    once during model setup in `ColumnModelMoL._set_options`.
    """

    def __init__(self, data="Cleaned-Toth"):
        """
        Initialize model parameters for the selected Toth dataset.

        Parameters
        ----------
        data : str, optional
            Name of the parameter set. Currently supported: `"Cleaned-Toth"`.
        """
        print("Selected CO2 isotherm equation: Toth")

        if data != "Cleaned-Toth":
            raise ValueError("Invalid data source for Toth isotherm.")

        print("Selected CO2 isotherm dataset: Cleaned-Toth")

        # Dry Toth parameters
        self.qm = 4.077685      # [mol/kg]
        self.T0 = 288.0          # [K]
        self.b0 = 307.587735/100    # [1/Pa]
        self.DH0 = -104935.08    # [J/mol]
        self.th0 = 0.222508
        self.alpha = 0.548347
        self.Young_b_fact = 1.0

        # WADST parameters (used when co-adsorption is enabled)
        self.qm_wet = 4.586366      # [mol/kg]
        self.T0_wet = 288.0         # [K]
        self.b0_wet = 6500.42       # [1/Pa]
        self.DH0_wet = -130435.87   # [J/mol]
        self.th0_wet = 0.148360
        self.alpha_wet = 0.202008
        self.A_fact = 0.221114      # [mol/kg]
        self.Young_b_fact_wet = 1.0

        # Optional SB terms kept for compatibility/extension.
        self.gamma = -0.137
        self.beta = 5.612

    def isotherm(self, p, T, q_co=None):
        """
        Placeholder interface for CO2 equilibrium.

        Parameters
        ----------
        p : float or np.ndarray
            CO2 partial pressure [Pa].
        T : float or np.ndarray
            Temperature [K].
        q_co : float or np.ndarray, optional
            Co-adsorbed loading [mol/kg], used only by wet/co-adsorption variants.

        Returns
        -------
        tuple
            The tuple returned by the bound equilibrium method.

        Notes
        -----
        This method must be bound at model setup time to either:
        - `dry_isotherm`
        - `co_isotherm_wadst`
        in `ColumnModelMoL._set_options`.
        """
        raise RuntimeError(
            "TothIsotherm.isotherm is not bound. "
            "Bind it to `dry_isotherm` or `co_isotherm_wadst` during packed-bed initialization."
        )

    def _toth_terms(self, p, T, qm, b0, DH0, T0, th0, alpha, b_fact):
        """
        Evaluate generic Toth terms and loading.

        Parameters
        ----------
        p : float or np.ndarray
            CO2 partial pressure [Pa].
        T : float or np.ndarray
            Temperature [K].
        qm : float
            Saturation loading parameter [mol/kg].
        b0 : float
            Affinity pre-exponential factor [1/Pa].
        DH0 : float
            Isosteric heat parameter [J/mol].
        T0 : float
            Reference temperature [K].
        th0 : float
            Base heterogeneity exponent.
        alpha : float
            Temperature coefficient for heterogeneity exponent.
        b_fact : float
            Multiplicative factor applied to `b`.

        Returns
        -------
        tuple
            `(q_eq, b, th)` for the parameter set provided.
        """
        b, th = self._toth_bth(T, b0, DH0, T0, th0, alpha, b_fact)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            bp_th = (b * p) ** th
            bp_th = np.nan_to_num(bp_th, nan=0.0, posinf=np.finfo(float).max)
            q_eq = np.clip(qm * b * p / (1.0 + bp_th) ** (1.0 / th), 1e-32, None)
            q_eq = np.nan_to_num(q_eq, nan=1e-32, posinf=1e-32)
        return q_eq, b, th

    @staticmethod
    def _toth_bth(T, b0, DH0, T0, th0, alpha, b_fact):
        """Evaluate only the Toth affinity and heterogeneity exponent."""
        b = b0 * np.exp(DH0 / R_gas / T0 * (1.0 - T0 / T)) * b_fact
        th = th0 + alpha * (1.0 - T0 / T)
        return b, th

    def dry_isotherm(self, p, T, q_co=None):
        """
        Compute dry CO2 equilibrium with the Toth model.

        Parameters
        ----------
        p : float or np.ndarray
            CO2 partial pressure [Pa].
        T : float or np.ndarray
            Temperature [K].
        q_co : Any, optional
            Unused for dry Toth. Kept for interface compatibility.

        Returns
        -------
        tuple
            `(q_eq, b, th)` where `q_eq` is in [mol/kg].
        """
        return self._toth_terms(
            p=p,
            T=T,
            qm=self.qm,
            b0=self.b0,
            DH0=self.DH0,
            T0=self.T0,
            th0=self.th0,
            alpha=self.alpha,
            b_fact=self.Young_b_fact,
        )

    def co_isotherm_wadst(self, p, T, q_co=None):
        """
        Compute WADST-like wet correction for CO2 loading.

        The dry Toth loading is blended with a wet Toth loading through a
        water-loading-dependent weight.

        Parameters
        ----------
        p : float or np.ndarray
            CO2 partial pressure [Pa].
        T : float or np.ndarray
            Temperature [K].
        q_co : float or np.ndarray, optional
            Co-adsorbed H2O loading [mol/kg]. If None, defaults to zero.

        Returns
        -------
        tuple
            `(q_eq, b_wet, th_wet)` where `q_eq` is the blended loading [mol/kg].
        """
        q_co = 0.0 if q_co is None else q_co

        q_eq_dry, _, _ = self.dry_isotherm(p=p, T=T)
        q_eq_dry = np.clip(q_eq_dry, 1e-32, self.qm)

        # Only affinity and heterogeneity are consumed below. Avoid building
        # a wet loading that would be discarded before the log-form evaluation.
        b_wet, th_wet = self._toth_bth(
            T=T,
            b0=self.b0_wet,
            DH0=self.DH0_wet,
            T0=self.T0_wet,
            th0=self.th0_wet,
            alpha=self.alpha_wet,
            b_fact=self.Young_b_fact_wet,
        )

        # Numerically robust wet loading evaluation in log form.
        p_safe = np.maximum(p, 1e-32)
        log_num = np.log(self.qm_wet) + np.log(np.maximum(b_wet, 1e-32)) + np.log(p_safe)
        log_denom = (1.0 / th_wet) * np.log1p((b_wet * p_safe) ** th_wet)
        q_eq_wet = np.exp(log_num - log_denom)
        q_eq_wet = np.clip(q_eq_wet, 1e-32, None)

        weight_wet = np.exp(-self.A_fact / np.maximum(q_co, 1e-3))
        q_eq = (1.0 - weight_wet) * q_eq_dry + weight_wet * q_eq_wet

        return q_eq, b_wet, th_wet

WATER_CRITICAL_TEMPERATURE_K = 647.096
WATER_CRITICAL_PRESSURE_PA = 220.64 * 1e5
_IAPWS_1995_COEFFICIENTS = (
    -7.85951783,
    1.8440825,
    -11.7866497,
    22.6807411,
    -15.9618719,
    1.80122502,
)


def water_saturation_pressure(T):
    """Return the H2O saturation pressure [Pa] from the IAPWS-1995 correlation.

    This is the single owner of the relation in pyDAC. `GABIsotherm.EOS` — the
    saturation pressure the *simulation* actually uses — resolves to it, and so
    do the callers that need a vapor pressure before an equilibrium system
    exists (the steam flow transforms, the flash separators). Changing water's
    vapor pressure means changing it here, once.

    Parameters
    ----------
    T : float or np.ndarray
        Temperature [K].

    Returns
    -------
    float or np.ndarray
        Saturation pressure [Pa].
    """
    b1, b2, b3, b4, b5, b6 = _IAPWS_1995_COEFFICIENTS
    v = 1.0 - T / WATER_CRITICAL_TEMPERATURE_K
    ln_p_sat_p_crit = WATER_CRITICAL_TEMPERATURE_K / T * (
        b1 * v + b2 * v**1.5 + b3 * v**3 + b4 * v**3.5 + b5 * v**4 + b6 * v**7.5
    )
    return WATER_CRITICAL_PRESSURE_PA * np.exp(ln_p_sat_p_crit)


# The GAB parameters are fitted to adsorption data up to this relative humidity;
# above it the loading is an extrapolation, reported once per isotherm object.
GAB_FITTED_MAX_RH = 0.9
# The GAB denominator vanishes at rh = 1/k; rh is kept this fraction of the way
# to that pole so the loading stays finite.
GAB_POLE_MARGIN = 0.99


class GABIsotherm(Thermodynamics):
    """
    H2O GAB equilibrium model with saturation pressure from IAPWS-1995.
    """

    def __init__(self):
        """
        Initialize GAB and vapor-pressure parameters for H2O adsorption.
        """
        print("Selected H2O isotherm equation: GAB (overall ads - des)")
        self.qm = 3.1970616886895318        # [mol/kg]
        self.C = 46779.95249302809         # [J/mol]
        self.D = 0.02356068511609903      # [1/K]
        self.F = 56233.72466565154        # [J/mol]
        self.G = -42.738276947397196        # [J/mol/K]
        self.a_lin = 57220                 # [J/mol]
        self.b_lin = -44.38                # [J/mol/K]
        self._reported_extrapolation = False

        # IAPWS-1995 EOS parameters.
        self.T_crit = WATER_CRITICAL_TEMPERATURE_K   # [K]
        self.P_crit = WATER_CRITICAL_PRESSURE_PA     # [Pa]

    def IAPWS_1995_EOS(self, T):
        """
        Compute water saturation pressure from IAPWS-1995 correlation.

        Parameters
        ----------
        T : float or np.ndarray
            Temperature [K].

        Returns
        -------
        float or np.ndarray
            Saturation pressure [Pa].
        """
        return water_saturation_pressure(T)

    def Arden_Buck_EOS(self, T):
        """
        Alternative vapor-pressure relation (kept for reference).

        Parameters
        ----------
        T : float or np.ndarray
            Temperature [K].

        Returns
        -------
        float or np.ndarray
            Saturation pressure [Pa].
        """
        first = 18.678 - (T - 273.15) / 234.5
        second = (T - 273.15) / (T - 16.01)
        return 611.21 * np.exp(first * second)

    def EOS(self, T):
        """
        Return saturation pressure used by the GAB model.

        Parameters
        ----------
        T : float or np.ndarray
            Temperature [K].

        Returns
        -------
        float or np.ndarray
            Saturation pressure [Pa].
        """
        return self.IAPWS_1995_EOS(T)

    def isotherm(self, p, T, q_co=None):
        """
        Compute H2O loading with the GAB isotherm.

        Parameters
        ----------
        p : float or np.ndarray
            H2O partial pressure [Pa].
        T : float or np.ndarray
            Temperature [K].
        q_co : Any, optional
            Unused, kept for interface compatibility.

        Returns
        -------
        tuple
            `(q_eq, rh, None)` where:
            - `q_eq` is H2O loading [mol/kg]
            - `rh` is relative humidity [-]
        """
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            E10 = self.a_lin + self.b_lin * T
            E29 = self.F + self.G * T
            E1 = self.C - np.exp(self.D * T)

            k = np.exp((E29 - E10) / (R_gas * T))
            c = np.exp((E1 - E10) / (R_gas * T))
            rh = np.clip(p / self.EOS(T), 1e-32, GAB_POLE_MARGIN / k)
            denom = (1.0 - k * rh) * (1.0 + (c - 1.0) * k * rh)
            denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
            q_eq = np.clip((self.qm * k * c * rh) / denom, 1e-32, None)
            q_eq = np.nan_to_num(q_eq, nan=1e-32, posinf=1e-32)

        reported = getattr(self, "_reported_extrapolation", False)  # pickled models
        if not reported and np.any(rh > GAB_FITTED_MAX_RH):
            self._reported_extrapolation = True
            warnings.warn(
                f"H2O GAB isotherm evaluated at relative humidity {float(np.max(rh)):.3f}, "
                f"above the fitted data (RH <= {GAB_FITTED_MAX_RH}); the H2O loading "
                "there is an extrapolation.",
                RuntimeWarning,
                stacklevel=2,
            )
        return q_eq, rh, None


# Name-based systems. Their methods show the complete physical data flow.
class TothEquilibriumSystem(EquilibriumSystem):
    """Independent dry-CO2 Toth system."""

    species = ("CO2",)

    def __init__(self, *, data: str = "Cleaned-Toth") -> None:
        self.toth = TothIsotherm(data=data)
        self.component_models = MappingProxyType({"CO2": self.toth})

    def equilibrium_loadings(
        self,
        *,
        partial_pressures_Pa: Mapping[str, Any],
        temperature_K: Any,
        current_loadings_mol_per_kg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del current_loadings_mol_per_kg
        pressure = _required_value(partial_pressures_Pa, "CO2", "partial pressure")
        loading = self.toth.dry_isotherm(p=pressure, T=temperature_K)[0]
        return {"CO2": loading}


class GABEquilibriumSystem(EquilibriumSystem):
    """Independent H2O GAB system."""

    species = ("H2O",)

    def __init__(self) -> None:
        self.gab = GABIsotherm()
        self.component_models = MappingProxyType({"H2O": self.gab})

    def equilibrium_loadings(
        self,
        *,
        partial_pressures_Pa: Mapping[str, Any],
        temperature_K: Any,
        current_loadings_mol_per_kg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del current_loadings_mol_per_kg
        pressure = _required_value(partial_pressures_Pa, "H2O", "partial pressure")
        loading = self.gab.isotherm(p=pressure, T=temperature_K)[0]
        return {"H2O": loading}


class TothGABWADSTEquilibriumSystem(EquilibriumSystem):
    """Current coupled CO2/H2O system with water-dependent WADST blending."""

    species = ("CO2", "H2O")

    def __init__(self, *, data: str = "Cleaned-Toth") -> None:
        self.toth = TothIsotherm(data=data)
        self.gab = GABIsotherm()
        self.component_models = MappingProxyType(
            {"CO2": self.toth, "H2O": self.gab}
        )

    def equilibrium_loadings(
        self,
        *,
        partial_pressures_Pa: Mapping[str, Any],
        temperature_K: Any,
        current_loadings_mol_per_kg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        pressure_co2 = _required_value(
            partial_pressures_Pa, "CO2", "partial pressure"
        )
        pressure_h2o = _required_value(
            partial_pressures_Pa, "H2O", "partial pressure"
        )
        loading_h2o = _required_value(
            current_loadings_mol_per_kg, "H2O", "current loading"
        )
        equilibrium_h2o = self.gab.isotherm(p=pressure_h2o, T=temperature_K)[0]
        equilibrium_co2 = self.toth.co_isotherm_wadst(
            p=pressure_co2,
            T=temperature_K,
            q_co=loading_h2o,
        )[0]
        return {"CO2": equilibrium_co2, "H2O": equilibrium_h2o}


class CompetitiveSipsEquilibriumSystem(EquilibriumSystem):
    """Parameterized competitive Sips equilibrium for named components.

    The implementation contains the equation only; sorbent-specific fitted
    coefficients belong to the configuration selecting it.  Pressures enter
    in Pa and ``pressure_unit_Pa`` states the pressure unit used by the fitted
    affinity prefactors.

    An optional non-negative modifier loading can reduce both saturation
    capacities and affinities.  With loading ``q_m``, the factors are
    ``1 / (1 + alpha_m q_m)`` and ``exp(-beta_m q_m)`` respectively.  Setting
    both coefficients to zero gives the ordinary dry competitive Sips model.
    """

    COMPONENT_PARAMETER_NAMES = frozenset(
        {
            "qinf_ref_mol_per_kg",
            "qinf_temperature_coefficient",
            "affinity_prefactor_inverse_pressure_unit",
            "affinity_heat_J_per_mol",
            "heterogeneity_ref",
            "heterogeneity_temperature_coefficient",
        }
    )

    def __init__(
        self,
        *,
        dataset: str,
        component_parameters: Mapping[str, Mapping[str, Any]],
        reference_temperature_K: float,
        pressure_unit_Pa: float,
        min_temperature_K: float = 150.0,
        max_temperature_K: float = 1000.0,
        modifier_loading_mol_per_kg: float = 0.0,
        capacity_modifier_coefficient_kg_per_mol: float = 0.0,
        affinity_modifier_coefficient_kg_per_mol: float = 0.0,
    ) -> None:
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("Competitive Sips dataset must be a non-empty string.")
        if not isinstance(component_parameters, Mapping) or not component_parameters:
            raise TypeError("component_parameters must be a non-empty species mapping.")

        parameters = {
            _species_name(species): self._validate_component_parameters(
                species, values
            )
            for species, values in component_parameters.items()
        }
        if len(parameters) != len(component_parameters):
            raise ValueError("component_parameters contains duplicate species names.")

        self.reference_temperature_K = _positive_finite(
            reference_temperature_K, "reference_temperature_K"
        )
        self.pressure_unit_Pa = _positive_finite(
            pressure_unit_Pa, "pressure_unit_Pa"
        )
        self.min_temperature_K = _positive_finite(
            min_temperature_K, "min_temperature_K"
        )
        self.max_temperature_K = _positive_finite(
            max_temperature_K, "max_temperature_K"
        )
        if self.max_temperature_K < self.min_temperature_K:
            raise ValueError("max_temperature_K must not be below min_temperature_K.")

        self.modifier_loading_mol_per_kg = _nonnegative_finite(
            modifier_loading_mol_per_kg, "modifier_loading_mol_per_kg"
        )
        self.capacity_modifier_coefficient_kg_per_mol = _nonnegative_finite(
            capacity_modifier_coefficient_kg_per_mol,
            "capacity_modifier_coefficient_kg_per_mol",
        )
        self.affinity_modifier_coefficient_kg_per_mol = _nonnegative_finite(
            affinity_modifier_coefficient_kg_per_mol,
            "affinity_modifier_coefficient_kg_per_mol",
        )
        self.dataset = dataset.strip()
        self.parameters = MappingProxyType(parameters)
        self.species = tuple(parameters)
        self.component_models = MappingProxyType(
            {species: self for species in self.species}
        )
        print(
            "Selected equilibrium: competitive Sips | "
            f"Components: {', '.join(self.species)} | Dataset: {self.dataset}"
        )

    def equilibrium_loadings(
        self,
        *,
        partial_pressures_Pa: Mapping[str, Any],
        temperature_K: Any,
        current_loadings_mol_per_kg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del current_loadings_mol_per_kg
        modifier_loading = np.atleast_1d(self.modifier_loading_mol_per_kg)
        capacity_factor = 1.0 / (
            1.0
            + self.capacity_modifier_coefficient_kg_per_mol * modifier_loading
        )
        affinity_factor = np.exp(
            -self.affinity_modifier_coefficient_kg_per_mol * modifier_loading
        )

        saturation_capacities = {}
        competitive_terms = {}
        pressure_to_fit_unit = 1.0 / self.pressure_unit_Pa
        for species in self.species:
            pressure = np.maximum(
                np.nan_to_num(
                    np.atleast_1d(
                        _required_value(
                            partial_pressures_Pa, species, "partial pressure"
                        )
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                0.0,
            ) * pressure_to_fit_unit
            qinf, affinity, exponent = self._species_parameters(
                species, temperature_K
            )
            saturation_capacities[species] = qinf * capacity_factor
            competitive_terms[species] = np.power(
                affinity * affinity_factor * pressure,
                exponent,
            )

        denominator = 1.0 + sum(competitive_terms.values())
        loadings = {
            species: saturation_capacities[species]
            * competitive_terms[species]
            / denominator
            for species in self.species
        }
        scalar_state = np.asarray(temperature_K).ndim == 0 and all(
            np.asarray(partial_pressures_Pa[species]).ndim == 0
            for species in self.species
        )
        if scalar_state:
            return {species: value.item() for species, value in loadings.items()}
        return loadings

    def _species_parameters(self, species: str, temperature_K: Any):
        """Return ``qinf(T)``, ``b(T)`` and ``c(T)`` for one species."""
        temperature = np.asarray(temperature_K, dtype=float)
        temperature = np.nan_to_num(
            temperature,
            nan=self.reference_temperature_K,
            posinf=self.max_temperature_K,
            neginf=self.min_temperature_K,
        )
        temperature = np.clip(
            temperature, self.min_temperature_K, self.max_temperature_K
        )
        parameters = self.parameters[species]
        temperature_ratio = temperature / self.reference_temperature_K
        qinf = parameters["qinf_ref_mol_per_kg"] * np.exp(
            np.clip(
                parameters["qinf_temperature_coefficient"]
                * (temperature_ratio - 1.0),
                -700.0,
                700.0,
            )
        )
        affinity = parameters["affinity_prefactor_inverse_pressure_unit"] * np.exp(
            np.clip(
                parameters["affinity_heat_J_per_mol"] / (R_gas * temperature),
                -700.0,
                700.0,
            )
        )
        exponent = parameters["heterogeneity_ref"] + parameters[
            "heterogeneity_temperature_coefficient"
        ] * (temperature_ratio - 1.0)
        return qinf, affinity, np.clip(exponent, 1e-6, None)

    @classmethod
    def _validate_component_parameters(cls, species, values):
        name = _species_name(species)
        if not isinstance(values, Mapping):
            raise TypeError(
                f"Competitive Sips parameters for {name} must be a mapping."
            )
        missing = sorted(cls.COMPONENT_PARAMETER_NAMES - set(values))
        unknown = sorted(set(values) - cls.COMPONENT_PARAMETER_NAMES)
        if missing or unknown:
            raise ValueError(
                f"Invalid Competitive Sips parameters for {name} "
                f"(missing={missing}, unknown={unknown})."
            )
        result = {
            key: _finite_number(value, f"component_parameters.{name}.{key}")
            for key, value in values.items()
        }
        for key in (
            "qinf_ref_mol_per_kg",
            "affinity_prefactor_inverse_pressure_unit",
            "affinity_heat_J_per_mol",
            "heterogeneity_ref",
        ):
            if result[key] <= 0.0:
                raise ValueError(
                    f"component_parameters.{name}.{key} must be greater than zero."
                )
        return MappingProxyType(result)


def _build_toth(
    datasets: Mapping[str, str], parameters: Mapping[str, Any]
) -> EquilibriumSystem:
    _model_parameters(parameters, allowed=(), model="Toth")
    return TothEquilibriumSystem(
        data=datasets.get("Toth", "Cleaned-Toth")
    )


def _build_gab(
    datasets: Mapping[str, str], parameters: Mapping[str, Any]
) -> EquilibriumSystem:
    if datasets:
        raise ValueError("GAB does not accept a configurable dataset.")
    _model_parameters(parameters, allowed=(), model="GAB")
    return GABEquilibriumSystem()


def _build_toth_gab_wadst(
    datasets: Mapping[str, str], parameters: Mapping[str, Any]
) -> EquilibriumSystem:
    _model_parameters(parameters, allowed=(), model="TothGABWADST")
    return TothGABWADSTEquilibriumSystem(
        data=datasets.get("Toth", "Cleaned-Toth")
    )


def _build_competitive_sips(
    datasets: Mapping[str, str], parameters: Mapping[str, Any]
) -> EquilibriumSystem:
    values = _model_parameters(
        parameters,
        allowed=(
            "component_parameters",
            "reference_temperature_K",
            "pressure_unit_Pa",
            "min_temperature_K",
            "max_temperature_K",
            "modifier_loading_mol_per_kg",
            "capacity_modifier_coefficient_kg_per_mol",
            "affinity_modifier_coefficient_kg_per_mol",
        ),
        model="CompetitiveSips",
    )
    component_parameters = values["component_parameters"]
    expected_species = ("CO2", "N2")
    if (
        not isinstance(component_parameters, Mapping)
        or set(component_parameters) != set(expected_species)
    ):
        raise ValueError(
            "The built-in CompetitiveSips registry entry requires exactly "
            "CO2 and N2 component parameters."
        )
    ordered_parameters = {
        species: component_parameters[species] for species in expected_species
    }
    return CompetitiveSipsEquilibriumSystem(
        dataset=datasets["CompetitiveSips"],
        component_parameters=ordered_parameters,
        reference_temperature_K=values["reference_temperature_K"],
        pressure_unit_Pa=values["pressure_unit_Pa"],
        min_temperature_K=values.get("min_temperature_K", 150.0),
        max_temperature_K=values.get("max_temperature_K", 1000.0),
        modifier_loading_mol_per_kg=values.get(
            "modifier_loading_mol_per_kg", 0.0
        ),
        capacity_modifier_coefficient_kg_per_mol=values.get(
            "capacity_modifier_coefficient_kg_per_mol", 0.0
        ),
        affinity_modifier_coefficient_kg_per_mol=values.get(
            "affinity_modifier_coefficient_kg_per_mol", 0.0
        ),
    )


# Every supported equilibrium system and its physical requirements are here.
EQUILIBRIUM_REGISTRY: Mapping[str, EquilibriumRegistryEntry] = MappingProxyType(
    {
        "Toth": EquilibriumRegistryEntry(
            adsorbates=("CO2",),
            required_gas_species=("CO2",),
            required_current_loadings=(),
            build=_build_toth,
            optional_datasets=("Toth",),
        ),
        "GAB": EquilibriumRegistryEntry(
            adsorbates=("H2O",),
            required_gas_species=("H2O",),
            required_current_loadings=(),
            build=_build_gab,
        ),
        "TothGABWADST": EquilibriumRegistryEntry(
            adsorbates=("CO2", "H2O"),
            required_gas_species=("CO2", "H2O"),
            required_current_loadings=("H2O",),
            build=_build_toth_gab_wadst,
            optional_datasets=("Toth",),
        ),
        "CompetitiveSips": EquilibriumRegistryEntry(
            adsorbates=("CO2", "N2"),
            required_gas_species=("CO2", "N2"),
            required_current_loadings=(),
            build=_build_competitive_sips,
            required_datasets=("CompetitiveSips",),
            required_parameters=(
                "component_parameters",
                "reference_temperature_K",
                "pressure_unit_Pa",
            ),
            optional_parameters=(
                "min_temperature_K",
                "max_temperature_K",
                "modifier_loading_mol_per_kg",
                "capacity_modifier_coefficient_kg_per_mol",
                "affinity_modifier_coefficient_kg_per_mol",
            ),
        ),
    }
)


# Systems registered at run time with pydac.register_equilibrium: this Python session only.
_REGISTERED_EQUILIBRIA: dict[str, EquilibriumRegistryEntry] = {}


def equilibrium_registry() -> Mapping[str, EquilibriumRegistryEntry]:
    """The built-in equilibrium systems plus those registered in this session."""
    return MappingProxyType({**EQUILIBRIUM_REGISTRY, **_REGISTERED_EQUILIBRIA})


def _required_value(values: Mapping[str, Any], species: str, label: str) -> Any:
    if not isinstance(values, Mapping):
        raise TypeError(f"Named {label}s must be supplied as a mapping.")
    try:
        return values[species]
    except KeyError as exc:
        raise ValueError(f"Missing {label} for species {species!r}.") from exc


def _species_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Species names must be non-empty strings.")
    return value.strip().upper()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_finite(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _nonnegative_finite(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _model_parameters(
    parameters: Mapping[str, Any], *, allowed: tuple[str, ...], model: str
) -> dict[str, Any]:
    if not isinstance(parameters, Mapping):
        raise TypeError(f"Parameters for {model!r} must be a mapping.")
    values = dict(parameters)
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown parameters for {model!r}: {unknown}.")
    return values


__all__ = [
    "EQUILIBRIUM_REGISTRY",
    "equilibrium_registry",
    "EquilibriumSystem",
    "CompetitiveSipsEquilibriumSystem",
    "GABEquilibriumSystem",
    "GABIsotherm",
    "Thermodynamics",
    "TothEquilibriumSystem",
    "TothGABWADSTEquilibriumSystem",
    "TothIsotherm",
    "WATER_CRITICAL_PRESSURE_PA",
    "WATER_CRITICAL_TEMPERATURE_K",
    "water_saturation_pressure",
]
