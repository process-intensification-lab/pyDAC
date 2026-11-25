"""
pyDAC - Binary flash-separator utilities.

This module provides Redlich-Kwong-based component properties and CO2-H2O
flash calculations used for condenser and phase-equilibrium post-processing.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
"""

import warnings

import numpy as np
from scipy.constants import gas_constant as R_gas
from scipy.optimize import fsolve

from pydac.core.isotherm_models import water_saturation_pressure

class Component:
    def __init__(self, name, T_crit, p_crit_bar):
        self.name = name
        self.T_crit = T_crit
        self.p_crit = p_crit_bar * 1e5  # [Pa]
        self.a, self.b = self.compute_rk_parameters()

    def compute_rk_parameters(self):
        cuberoot2 = 2**(1/3)
        a = 1 / (9 * (cuberoot2 - 1)) * (R_gas**2 * self.T_crit**2.5 / self.p_crit)
        b = (cuberoot2 - 1) / 3 * (R_gas * self.T_crit) / self.p_crit
        return a, b

    def fugacity_coefficient(self, T, p):
        A = np.sqrt(self.a / (R_gas**2 * T**2.5))
        B = self.b / (R_gas * T)

        def Z_equation(Z):
            h = B * p / Z
            return Z - 1 / (1 - h) + (A**2 * h) / (B * (1 + h))

        Z_guess = 0.9
        Z, _, ier, message = fsolve(Z_equation, Z_guess, full_output=True)
        Z = float(Z[0])
        if ier != 1 or not np.isfinite(Z) or Z <= B * p:
            raise ValueError(
                f"Redlich-Kwong compressibility of {self.name} at T = {T} K, p = {p} Pa "
                f"did not converge to a gas root (Z = {Z}): {message}"
            )
        h = B * p / Z

        ln_phi = (Z - 1) - np.log(Z - B * p) - (A**2 / B) * np.log(1 + h)
        phi = float(np.exp(ln_phi))
        if not (np.isfinite(phi) and phi > 0.0):
            raise ValueError(
                f"Fugacity coefficient of {self.name} at T = {T} K, p = {p} Pa is {phi}."
            )
        return phi

class ThermoModels:
    @staticmethod
    def henry_CO2(T):
        H0 = -6.8346
        H1, H2, H3 = 1.2817e4, -3.7668e6, 2.997e8
        ln_H = H0 + H1 / T + H2 / T**2 + H3 / T**3
        return np.exp(ln_H) * 1e6  # Pa

    @staticmethod
    def p_sat_H2O(T):
        # Same vapor pressure the simulation's H2O model uses; see
        # `isotherm_models.water_saturation_pressure`.
        return water_saturation_pressure(T)

#: How far outside [0, 1] a feed CO2 fraction may sit before the flash warns. The
#: outlet-face extrapolation makes small excursions normal (see `FlashSeparator.solve`);
#: the built-in 1D DAC case stays inside [1e-4, 0.96] and never reaches this.
FEED_FRACTION_WARN_TOLERANCE = 1e-3


class FlashSeparator:
    def __init__(self, component_CO2, component_H2O):
        self.CO2 = component_CO2
        self.H2O = component_H2O

    def solve(self, n_feed, z_CO2_feed, T, p):
        """Flash `n_feed` mol (CO2 mole fraction `z_CO2_feed`) at (T, p).

        Henry's law for CO2 and Raoult's law for H2O, with Redlich-Kwong fugacity
        coefficients, give composition-independent K-values,
        K_CO2 = H / (p phi_CO2) and K_H2O = p_sat / (p phi_H2O), so the phase region
        follows from the feed alone (Rachford-Rice with constant K):
        sum z K <= 1 is at or below the bubble point (one liquid, x = z);
        sum z / K <= 1 is at or above the dew point (one vapour, y = z);
        otherwise two phases with the tie line x = (1 - K_H2O) / (K_CO2 - K_H2O),
        y = K_CO2 x and the lever rule. The three regions meet continuously and every
        branch conserves both species. The composition of an absent phase is the
        incipient one (bubble vapour, dew liquid); it carries no flow.

        `z_CO2_feed` may sit marginally outside [0, 1]: the outlet-face extrapolation
        can carry a species being flushed out slightly negative (see
        `StepPostProcessor._condenser_stream_mask`). It is flashed as given, so the
        balances still close; beyond `FEED_FRACTION_WARN_TOLERANCE` a RuntimeWarning
        says so, since such a gap is no longer marginal. A non-finite or negative
        feed, a non-finite fraction, a non-positive T or p, or an unusable K-value
        raises ValueError.
        """
        n_feed, z, T, p = float(n_feed), float(z_CO2_feed), float(T), float(p)
        where = f"(n_feed = {n_feed} mol, z_CO2 = {z}, T = {T} K, p = {p} Pa)"
        if not (np.isfinite(n_feed) and n_feed >= 0.0):
            raise ValueError(f"CO2/H2O flash: the feed must be finite and >= 0 {where}.")
        if not np.isfinite(z):
            raise ValueError(f"CO2/H2O flash: the CO2 fraction is not finite {where}.")
        if not (np.isfinite(T) and T > 0.0 and np.isfinite(p) and p > 0.0):
            raise ValueError(f"CO2/H2O flash: T and p must be finite and > 0 {where}.")
        tol = FEED_FRACTION_WARN_TOLERANCE
        if n_feed > 0.0 and not -tol <= z <= 1.0 + tol:
            # One fixed text, so Python shows it once and not at every output time.
            warnings.warn(
                f"CO2/H2O flash: a feed CO2 fraction lies outside [0, 1] by more than {tol:g}. "
                "It comes from the outlet-face extrapolation of a steep near-outlet gradient "
                "and is flashed as given, so the species balances close; a finer axial grid "
                "(grid.nz) reduces it.",
                RuntimeWarning,
                stacklevel=2,
            )

        H_CO2 = ThermoModels.henry_CO2(T)
        p_sat_H2O = ThermoModels.p_sat_H2O(T)
        phi_CO2 = self.CO2.fugacity_coefficient(T, p)
        phi_H2O = self.H2O.fugacity_coefficient(T, p)
        K_CO2 = float(H_CO2 / (p * phi_CO2))
        K_H2O = float(p_sat_H2O / (p * phi_H2O))
        if not all(np.isfinite(k) and k > 0.0 for k in (K_CO2, K_H2O)):
            raise ValueError(
                f"CO2/H2O flash: unusable K-values K_CO2 = {K_CO2}, K_H2O = {K_H2O} {where}."
            )

        bubble = z * K_CO2 + (1.0 - z) * K_H2O        # sum z K
        dew = z / K_CO2 + (1.0 - z) / K_H2O           # sum z / K
        if n_feed == 0.0 or bubble <= 1.0:
            # One liquid phase (or no feed): the whole feed condenses, CO2 included.
            # Incipient bubble vapour, clamped: a marginally negative z can make
            # sum z K tiny or negative, and this composition carries no flow.
            y_CO2 = min(max(z * K_CO2 / bubble, 0.0), 1.0) if bubble > 0.0 else 0.0
            return {'y_CO2': y_CO2, 'y_H2O': 1 - y_CO2, 'x_CO2': z, 'x_H2O': 1 - z,
                    'nv': 0.0, 'nl': n_feed}
        if dew <= 1.0:
            # One vapour phase: nothing condenses.
            x_CO2 = (z / K_CO2) / dew
            return {'y_CO2': z, 'y_H2O': 1 - z, 'x_CO2': x_CO2, 'x_H2O': 1 - x_CO2,
                    'nv': n_feed, 'nl': 0.0}

        # Two phases: here K_CO2 > 1 > K_H2O and x_CO2 < z < y_CO2.
        x_CO2 = (1.0 - K_H2O) / (K_CO2 - K_H2O)
        y_CO2 = K_CO2 * x_CO2
        nv = n_feed * (z - x_CO2) / (y_CO2 - x_CO2)
        return {
            'y_CO2': y_CO2,
            'y_H2O': 1 - y_CO2,
            'x_CO2': x_CO2,
            'x_H2O': 1 - x_CO2,
            'nv': nv,
            'nl': n_feed - nv
        }

    @staticmethod
    def pretty_print(results, title="Phase Equilibrium Results", n_feed_in=None):
        print(f"\n {title}")
        print("-" * 60)
        print(f"  Vapor mole fraction CO2:      {results['y_CO2']:.6f}")
        print(f"  Vapor mole fraction H2O:      {results['y_H2O']:.6f}")
        print(f"  Liquid mole fraction CO2:     {results['x_CO2']:.6f}")
        print(f"  Liquid mole fraction H2O:     {results['x_H2O']:.6f}")
        print()
        print(f"  Vapor molar flow (nv):        {results['nv']:.2e} mol/s")
        print(f"  Liquid molar flow (nl):       {results['nl']:.2e} mol/s")
        total_out = results['nv'] + results['nl']
        print(f"  Total outlet flow (nv + nl):  {total_out:.2e} mol/s")

        if n_feed_in is not None:
            error = total_out - n_feed_in
            rel_error = error / (n_feed_in + 1e-32) * 100
            print(f"  Total inlet flow:            {n_feed_in:.2e} mol/s")
            print(f"  Mass balance error:           {error:.4e} mol/s ({rel_error:.2f}%)")

        print("-" * 60)
