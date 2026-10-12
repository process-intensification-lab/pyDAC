"""Models registered at run time are selected by name in a case, like built-in ones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pydac
from pydac.core import isotherm_models, kinetic_models

CASE = str(Path(__file__).resolve().parent / "cases" / "v3_smoke_startup.json")
R = 8.314462618


class Langmuir(pydac.EquilibriumSystem):
    species = ("CO2",)

    def __init__(self, q_sat_mol_per_kg, b0_inverse_Pa, heat_J_per_mol):
        self.q_sat, self.b0, self.heat = q_sat_mol_per_kg, b0_inverse_Pa, heat_J_per_mol

    def equilibrium_loadings(self, *, partial_pressures_Pa, temperature_K,
                             current_loadings_mol_per_kg):
        b = self.b0 * np.exp(self.heat / (R * temperature_K))
        p = partial_pressures_Pa["CO2"]
        return {"CO2": self.q_sat * b * p / (1 + b * p)}


class HalfLDF(pydac.KineticLaw):
    species = ("CO2",)

    def __init__(self, k_s_inv):
        self.k = k_s_inv

    def loading_rates_mol_per_kg_s(self, *, partial_pressures_Pa, temperature_K,
                                   current_loadings_mol_per_kg,
                                   equilibrium_loadings_mol_per_kg):
        return {"CO2": 0.5 * self.k * (equilibrium_loadings_mol_per_kg["CO2"]
                                       - current_loadings_mol_per_kg["CO2"])}


LANGMUIR = pydac.EquilibriumRegistryEntry(
    adsorbates=("CO2",), required_gas_species=("CO2",), required_current_loadings=(),
    build=lambda datasets, parameters: Langmuir(**parameters),
    required_parameters=("q_sat_mol_per_kg", "b0_inverse_Pa", "heat_J_per_mol"),
)
PARAMETERS = {"q_sat_mol_per_kg": 1.25, "b0_inverse_Pa": 3.8e-14, "heat_J_per_mol": 70300.0}
SETTINGS = {
    "adsorption.equilibrium_systems": {
        "Langmuir": {"datasets": {}, "parameters": PARAMETERS},
        "GAB": {"datasets": {}, "parameters": {}},
    },
}


@pytest.fixture(autouse=True)
def session_registries(monkeypatch):
    monkeypatch.setattr(isotherm_models, "_REGISTERED_EQUILIBRIA", {})
    monkeypatch.setattr(kinetic_models, "_REGISTERED_KINETICS", {})


def test_a_registered_isotherm_and_kinetic_law_run_in_a_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(pydac.ConfigurationError, match="Unknown model 'Langmuir'"):
        pydac.load_case(CASE, settings=SETTINGS)

    pydac.register_equilibrium("Langmuir", LANGMUIR)
    pydac.register_kinetics("HalfLDF", pydac.KineticRegistryEntry(
        build=lambda species, parameters: HalfLDF(parameters["k_s_inv"]),
        supported_species=("CO2",), required_parameters=("k_s_inv",)))
    settings = dict(SETTINGS, **{"adsorption.kinetics": {
        "CO2": {"model": "HalfLDF", "parameters": {"k_s_inv": 0.003}},
        "H2O": {"model": "LDF", "parameters": {"k_LDF_s_inv": 0.00083}},
    }})
    result = pydac.run_steps(pydac.load_case(CASE, settings=settings))
    assert (result.results_folder / "startup" / "kpis.csv").is_file()


def test_registration_refusals_and_replacement():
    for name in ("Toth", "", " Langmuir"):
        with pytest.raises(pydac.ConfigurationError):
            pydac.register_equilibrium(name, LANGMUIR)
    with pytest.raises(pydac.ConfigurationError, match="built-in kinetic law"):
        pydac.register_kinetics("LDF", pydac.KineticRegistryEntry(build=lambda s, p: None))
    with pytest.raises(pydac.ConfigurationError, match="EquilibriumRegistryEntry"):
        pydac.register_equilibrium("Langmuir", {"build": LANGMUIR.build})
    pydac.register_equilibrium("Langmuir", LANGMUIR)
    again = pydac.EquilibriumRegistryEntry(**{**{f: getattr(LANGMUIR, f) for f in LANGMUIR.__slots__},
                                              "required_parameters": ()})
    pydac.register_equilibrium("Langmuir", again)
    assert isotherm_models.equilibrium_registry()["Langmuir"] is again
    assert "Toth" in isotherm_models.equilibrium_registry()


class WetLangmuir(pydac.EquilibriumSystem):
    """CO2 + H2O, the CO2 capacity rising with the adsorbed water; no water EOS of its own."""

    species = ("CO2", "H2O")

    def equilibrium_loadings(self, *, partial_pressures_Pa, temperature_K,
                             current_loadings_mol_per_kg):
        b = 3.8e-14 * np.exp(70300.0 / (R * temperature_K))
        p = partial_pressures_Pa["CO2"]
        q_sat = 1.25 * (1 + 0.1 * current_loadings_mol_per_kg["H2O"])
        p_w = partial_pressures_Pa["H2O"]
        return {"CO2": q_sat * b * p / (1 + b * p), "H2O": 8.0 * 1e-3 * p_w / (1 + 1e-3 * p_w)}


def test_a_coupled_user_system_runs_a_humid_case_without_a_water_eos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pydac.register_equilibrium("WetLangmuir", pydac.EquilibriumRegistryEntry(
        adsorbates=("CO2", "H2O"), required_gas_species=("CO2", "H2O"),
        required_current_loadings=("H2O",), build=lambda datasets, parameters: WetLangmuir()))
    case = pydac.load_case(CASE, settings={"adsorption.equilibrium_systems": {
        "WetLangmuir": {"datasets": {}, "parameters": {}}}})
    result = pydac.run_steps(case)
    assert (result.results_folder / "startup" / "kpis.csv").is_file()
