"""
pyDAC - Binary flash-separator utilities.

This module provides Redlich-Kwong-based component properties and CO2-H2O
flash calculations used for condenser and phase-equilibrium post-processing.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

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
        Z = fsolve(Z_equation, Z_guess)[0]
        h = B * p / Z

        ln_phi = (Z - 1) - np.log(Z - B * p) - (A**2 / B) * np.log(1 + h)
        return np.exp(ln_phi)

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

class FlashSeparator:
    def __init__(self, component_CO2, component_H2O):
        self.CO2 = component_CO2
        self.H2O = component_H2O

    def solve(self, n_feed, z_CO2_feed, T, p):
        if z_CO2_feed < 1e-3:
            return {
                'y_CO2': 0.0,
                'y_H2O': 0.0,
                'x_CO2': 0.0,
                'x_H2O': 1.0,
                'nv': 0.0,
                'nl': n_feed
            }
            
        H_CO2 = ThermoModels.henry_CO2(T)
        p_sat_H2O = ThermoModels.p_sat_H2O(T)

        phi_CO2 = self.CO2.fugacity_coefficient(T, p)
        phi_H2O = self.H2O.fugacity_coefficient(T, p)

        num = H_CO2 * (1 - p * phi_H2O / p_sat_H2O)
        den = (p * phi_CO2) - (H_CO2 * p * phi_H2O / p_sat_H2O)
        y_CO2 = num / den
        x_CO2 = 1 - (p * phi_H2O * (1 - y_CO2) / p_sat_H2O)

        nv = n_feed * (z_CO2_feed - x_CO2) / (y_CO2 - x_CO2)
        nl = n_feed - nv

        return {
            'y_CO2': y_CO2,
            'y_H2O': 1 - y_CO2,
            'x_CO2': x_CO2,
            'x_H2O': 1 - x_CO2,
            'nv': nv,
            'nl': nl
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
