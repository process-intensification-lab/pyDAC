from dataclasses import FrozenInstanceError

import pytest

from pydac.core.species_catalog import (
    SPECIES_CATALOG,
    ShomateHeatCapacity,
    SpeciesProperties,
    available_species,
    get_species_properties,
)


EXPECTED = {
    "CO2": {
        "molecular_weight_kg_per_mol": 44.009 / 1000.0,
        "lennard_jones_sigma_m": 3.941e-10,
        "lennard_jones_epsilon_over_k_K": 195.2,
        "fuller_diffusion_volume_m3_per_mol": 26.9e-6,
        "thermal_conductivity_reference_W_per_mK": 0.0166,
        "thermal_conductivity_reference_temperature_K": 298.0,
        "shomate": (24.99735, 55.18696, -33.69137, 7.948387, -0.136638),
    },
    "H2O": {
        "molecular_weight_kg_per_mol": 18.015 / 1000.0,
        "lennard_jones_sigma_m": 2.641e-10,
        "lennard_jones_epsilon_over_k_K": 809.1,
        "fuller_diffusion_volume_m3_per_mol": 13.1e-6,
        "thermal_conductivity_reference_W_per_mK": 0.02457,
        "thermal_conductivity_reference_temperature_K": 398.0,
        "shomate": (30.09200, 6.832514, 6.793435, -2.534480, 0.082139),
    },
    "O2": {
        "molecular_weight_kg_per_mol": 31.999 / 1000.0,
        "lennard_jones_sigma_m": 3.467e-10,
        "lennard_jones_epsilon_over_k_K": 106.7,
        "fuller_diffusion_volume_m3_per_mol": 16.3e-6,
        "thermal_conductivity_reference_W_per_mK": 0.026,
        "thermal_conductivity_reference_temperature_K": 298.0,
        "shomate": (31.32230, -20.23531, 57.86644, -36.50616, -0.007374),
    },
    "N2": {
        "molecular_weight_kg_per_mol": 28.014 / 1000.0,
        "lennard_jones_sigma_m": 3.798e-10,
        "lennard_jones_epsilon_over_k_K": 71.4,
        "fuller_diffusion_volume_m3_per_mol": 18.5e-6,
        "thermal_conductivity_reference_W_per_mK": 0.02583,
        "thermal_conductivity_reference_temperature_K": 298.0,
        "shomate": (28.98641, 1.853978, -9.647459, 16.63537, 0.000117),
    },
}


def test_builtin_catalog_matches_pre_refactor_constants_exactly():
    assert available_species() == ("CO2", "H2O", "N2", "O2")
    assert set(SPECIES_CATALOG) == set(EXPECTED)

    for name, expected in EXPECTED.items():
        properties = get_species_properties(name)
        assert properties.name == name
        for field_name in (
            "molecular_weight_kg_per_mol",
            "lennard_jones_sigma_m",
            "lennard_jones_epsilon_over_k_K",
            "fuller_diffusion_volume_m3_per_mol",
            "thermal_conductivity_reference_W_per_mK",
            "thermal_conductivity_reference_temperature_K",
        ):
            assert getattr(properties, field_name) == expected[field_name]

        coefficients = properties.heat_capacity_shomate
        assert (
            coefficients.a_J_per_molK,
            coefficients.b_J_per_molK,
            coefficients.c_J_per_molK,
            coefficients.d_J_per_molK,
            coefficients.e_J_per_molK,
        ) == expected["shomate"]

        temperature_K = 473.15
        t = temperature_K / 1000.0
        expected_cp = (
            expected["shomate"][0]
            + expected["shomate"][1] * t
            + expected["shomate"][2] * t**2
            + expected["shomate"][3] * t**3
            + expected["shomate"][4] / t**2
        )
        assert coefficients.evaluate_J_per_molK(temperature_K) == expected_cp


def test_catalog_records_and_registry_are_immutable():
    co2 = get_species_properties(" co2 ")

    with pytest.raises(FrozenInstanceError):
        co2.molecular_weight_kg_per_mol = 0.0
    with pytest.raises(FrozenInstanceError):
        co2.heat_capacity_shomate.a_J_per_molK = 0.0
    with pytest.raises(TypeError):
        SPECIES_CATALOG["Ar"] = co2


def test_unknown_species_error_is_actionable():
    with pytest.raises(ValueError, match="Unknown species 'Ar'") as error:
        get_species_properties("Ar")
    assert "CO2, H2O, N2, O2" in str(error.value)


def test_catalog_records_validate_inputs():
    coefficients = ShomateHeatCapacity(1.0, 2.0, 3.0, 4.0, 5.0)
    with pytest.raises(ValueError, match="greater than zero"):
        SpeciesProperties(
            name="X",
            molecular_weight_kg_per_mol=0.0,
            lennard_jones_sigma_m=1.0,
            lennard_jones_epsilon_over_k_K=1.0,
            fuller_diffusion_volume_m3_per_mol=1.0,
            thermal_conductivity_reference_W_per_mK=1.0,
            thermal_conductivity_reference_temperature_K=1.0,
            heat_capacity_shomate=coefficients,
        )
