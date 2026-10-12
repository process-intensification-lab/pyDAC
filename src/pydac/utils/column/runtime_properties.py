"""Structured runtime-property record for one initialized column model.

`build_runtime_properties` serializes the initialization-time physics an
already-built model computed; `print_runtime_property_summary` formats that
same record for the terminal. Neither computes anything physical: the record
is the contract `utils.postprocessing.runtime_report` consumes downstream.
"""

from __future__ import annotations

import numpy as np


RUNTIME_PROPERTIES_SCHEMA_VERSION = 1


def _finite_or_none(value):
    """Return the value as a float, or `None` when it is not finite."""
    number = float(value)
    return number if np.isfinite(number) else None


def build_runtime_properties(model):
    """Collect every value shown by the terminal property summary."""
    property_composition = model._gas_property_composition().reshape(-1)

    binary_diffusivity = {}
    for i, species_i in enumerate(model.gas_species):
        for j in range(i + 1, model.Nc_gas):
            pair = f"{species_i}-{model.gas_species[j]}"
            binary_diffusivity[pair] = _finite_or_none(model.D_binary[i, j])

    runtime_properties = {
        "schema_version": RUNTIME_PROPERTIES_SCHEMA_VERSION,
        "step": model.step,
        "state_layout": model.state_layout.to_dict(),
        "evaluation": {
            "temperature_K": float(model.T_feed),
            "pressure_Pa": float(model.p_feed),
            "gas_composition_source": model.gas_property_composition_source,
            "gas_mole_fractions": {
                species: float(property_composition[model.gas_index[species]])
                for species in model.gas_species
            },
            "dispersion_composition_source": model.dispersion_composition_source,
            "dispersion_temperature_K": float(model.T_feed),
            "dispersion_pressure_Pa": float(model.p_feed),
        },
        "species": {
            species: {
                "heat_capacity_J_molK": _finite_or_none(model.cp_g[index]),
                "viscosity_Pa_s": _finite_or_none(model.mu_i[index]),
                "mixture_diffusivity_m2_s": _finite_or_none(model.D_m[index]),
                "schmidt_number": _finite_or_none(model.Sc[index]),
                "sherwood_number": _finite_or_none(model.Sh[index]),
            }
            for species, index in model.gas_index.items()
        },
        "binary_diffusivity_m2_s": binary_diffusivity,
        "mixture": {
            "density_kg_m3": _finite_or_none(model.rho_g_mix),
            "viscosity_Pa_s": _finite_or_none(model.mu_g_mix),
            "heat_capacity_J_molK": _finite_or_none(model.cp_g_mix),
            "thermal_conductivity_W_mK": _finite_or_none(model.lambda_g_mix),
        },
        "solid": {
            "heat_capacity_J_kgK": _finite_or_none(model.cp_p),
            "evaluation_temperature_K": float(model.T_feed),
        },
        "dimensionless": {
            "reynolds_number": _finite_or_none(model.Re),
            "prandtl_number": _finite_or_none(model.Pr),
        },
        "inlet": {
            "superficial_velocity_m_s": _finite_or_none(model.u_inlet),
            "total_concentration_flux_nd": _finite_or_none(
                np.mean(model.flux_in_ctot)
            ),
            "enthalpy_flux_nd": _finite_or_none(np.mean(model.flux_in_T)),
            "species_mass_flow_kg_s": {
                species: _finite_or_none(
                    model.mass_flow_in_i[model.gas_index[species]]
                )
                for species in model.gas_species
            },
        },
        "bed_architecture": model.bed_architecture_context,
    }

    if hasattr(model, "rho_steam_inlet"):
        runtime_properties["inlet"]["steam_density_kg_m3"] = _finite_or_none(
            model.rho_steam_inlet
        )

    return runtime_properties


def print_runtime_property_summary(runtime_properties, gas_species):
    """Print only values already present in `runtime_properties`."""
    report = runtime_properties
    evaluation = report["evaluation"]
    print("\nRuntime property summary...")
    print(
        f"Step={report['step']}, T={evaluation['temperature_K']:.2f} K, "
        f"p={evaluation['pressure_Pa']:.2f} Pa, "
        f"composition={evaluation['gas_composition_source']}."
    )

    for species in gas_species:
        values = report["species"][species]
        print(
            f"{species}: cp={values['heat_capacity_J_molK']:.2f} J/mol/K, "
            f"mu={values['viscosity_Pa_s']:.2E} Pa s, "
            f"D_mix={values['mixture_diffusivity_m2_s']:.2E} m2/s, "
            f"Sc={values['schmidt_number']:.2f}, "
            f"Sh={values['sherwood_number']:.2f}"
        )

    binary = ", ".join(
        f"{pair}={value:.2E}"
        for pair, value in report["binary_diffusivity_m2_s"].items()
    )
    print(f"Fuller binary diffusivity [m2/s]: {binary}")

    mixture = report["mixture"]
    dimensionless = report["dimensionless"]
    print(
        "Mixture: "
        f"rho={mixture['density_kg_m3']:.6f} kg/m3, "
        f"mu={mixture['viscosity_Pa_s']:.2E} Pa s, "
        f"cp={mixture['heat_capacity_J_molK']:.6f} J/mol/K, "
        f"lambda={mixture['thermal_conductivity_W_mK']:.6f} W/m/K"
    )
    print(
        f"Re={dimensionless['reynolds_number']:.2f}, "
        f"Pr={dimensionless['prandtl_number']:.2f}"
    )
    print(
        "Solid heat capacity: "
        f"{report['solid']['heat_capacity_J_kgK']:.2f} J/kg/K"
    )
    mass_flows = report["inlet"]["species_mass_flow_kg_s"]
    formatted = ", ".join(
        f"{species}={mass_flows[species]:.2E}"
        for species in gas_species
    )
    print(f"Inlet mass flow [kg/s]: {formatted}")
