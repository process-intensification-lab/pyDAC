"""
pyDAC - Ternary flash-separator utilities.

This module provides Redlich-Kwong-based component properties and ternary
CO2-H2O-O2 flash calculations used to estimate vapor-liquid splits in
oxygen-aware separator or condenser analyses.

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
    def henry_O2(T):
        A = 1700
        T0 = 298.15         # Reference temperature in K
        H0 = 1.2e-5         # [mol/m3/Pa]
        ln_H_hH0 = A * (1/T - 1/T0)
        H_cp = H0 * np.exp(ln_H_hH0)  # [mol/m3/Pa]
        H_xp = H_cp * 1.83089         # Convert to [1/atm]
        H_px = 1 / H_xp               # Convert to [atm]
        H = H_px * 101325  # Convert to [Pa]
        return H

    @staticmethod
    def p_sat_H2O(T):
        # Same vapor pressure the simulation's H2O model uses; see
        # `isotherm_models.water_saturation_pressure`.
        return water_saturation_pressure(T)

class FlashSeparator:
    def __init__(self, component_CO2, component_H2O, component_O2):
        self.CO2 = component_CO2
        self.H2O = component_H2O
        self.O2 = component_O2

    def solve_ternary(self, nf, z_CO2, z_O2, T, p):
        # feed fractions
        z_H2O = 1 - z_CO2 - z_O2

        # equilibrium constants
        H_CO2    = ThermoModels.henry_CO2(T)
        H_O2     = ThermoModels.henry_O2(T)
        p_sat_H2O = ThermoModels.p_sat_H2O(T)

        # fugacity coeffs
        phi_CO2 = self.CO2.fugacity_coefficient(T, p)
        phi_O2  = self.O2.fugacity_coefficient(T, p)
        phi_H2O = self.H2O.fugacity_coefficient(T, p)

        def F(u):
            nl, nv, xCO2, xO2, yCO2, yO2, xH2O, yH2O = u

            eq1 = nf - (nv + nl)
            eq2 = z_CO2*nf - (yCO2*nv + xCO2*nl)
            eq3 = z_O2* nf - (yO2 *nv + xO2 *nl)
            eq4 = xCO2*H_CO2 - yCO2*p*phi_CO2
            eq5 = xO2* H_O2  - yO2 *p*phi_O2
            eq6 = xH2O*p_sat_H2O - yH2O*p*phi_H2O
            eq7 = xH2O - (1 - xCO2 - xO2)
            eq8 = yH2O - (1 - yCO2 - yO2)

            return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]

        # initial guess  
        u0 = [
            nf*0.5,       # nl
            nf*0.5,       # nv
            z_CO2*0.1,    # xCO2
            z_O2*0.1,     # xO2
            z_CO2*0.9,    # yCO2
            z_O2*0.9,     # yO2
            1 - (z_CO2+z_O2)*0.1,  # xH2O
            1 - (z_CO2+z_O2)*0.9   # yH2O
        ]

        sol = fsolve(F, u0)
        nl, nv, xCO2, xO2, yCO2, yO2, xH2O, yH2O = sol
        return {
            'nl': nl, 'nv': nv,
            'x': dict(CO2=xCO2, O2=xO2, H2O=xH2O),
            'y': dict(CO2=yCO2, O2=yO2, H2O=yH2O)
        }

    @staticmethod
    def pretty_print_ternary(res, title="Ternary Flash Results"):
        print(f"\n {title}")
        print("-"*60)
        print(f"  Liquid flow n_l: {res['nl']:.4e} mol/s")
        print(f"  Vapor  flow n_v: {res['nv']:.4e} mol/s\n")
        print("  Liquid mole fractions:")
        for comp, xi in res['x'].items():
            print(f"    x_{comp}: {xi:.6f}")
        print("  Vapor mole fractions:")
        for comp, yi in res['y'].items():
            print(f"    y_{comp}: {yi:.6f}")
        print("-"*60)
