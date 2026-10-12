"""Input validation and normalization for the column model.

Every function here is a pure check or a pure normalization lifted out of
``src/pydac/core/column_model.py``: it either returns the normalized value or raises the
same ``TypeError``/``ValueError`` the model used to raise inline. No model
state lives in this module. The one physical relation here is the solid
heat-capacity table: it is kept next to its parameter check so that
``--validate`` and the model accept exactly the same declarations.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


REQUIRED_FIXED_ADSORBATE_FIELDS = frozenset(
    {"loading_mol_per_kg", "heat_capacity_J_molK"}
)

#: Implemented solid heat-capacity models, keyed by lower-case name, with the
#: exact parameter names each one takes. Evaluated by `resolve_solid_heat_capacity`.
SOLID_HEAT_CAPACITY_MODELS = {
    "low2023": frozenset({"a_J_K_per_kg", "b_J_per_kgK2", "c_J_per_kgK"}),
    "constant": frozenset({"value_J_kgK"}),
}


def validate_flat_vector(value, *, size, label):
    """Return an array view after checking one packed-vector interface."""
    array = np.asarray(value)
    expected_shape = (int(size),)
    if array.shape != expected_shape:
        raise ValueError(
            f"{label} must have shape {expected_shape}; got {array.shape}."
        )
    return array


def validate_architecture_support(bed_architecture, *, is_1d, Nr):
    """Reject architecture/grid pairs the transport closures do not cover."""
    from pydac.core.bed_architectures import validate_architecture_name

    validate_architecture_name(bed_architecture)


def resolve_heats_of_adsorption(configured_heats, adsorbates):
    """Return the declared adsorbates' heats of adsorption [J/mol].

    Heats must be finite and negative: the model adds them as a released
    enthalpy, so a non-negative value would feed the bed energy on adsorption.
    """
    if not isinstance(configured_heats, Mapping):
        raise TypeError(
            "simulation.sorbent.heats_of_adsorption_J_mol must be a "
            "species-keyed mapping."
        )
    missing_heats = sorted(set(adsorbates) - set(configured_heats))
    if missing_heats:
        raise ValueError(
            "Sorbent heats of adsorption are missing declared adsorbates "
            f"{missing_heats}."
        )
    heats = {species: float(configured_heats[species]) for species in adsorbates}
    invalid_heats = [
        species
        for species, heat in heats.items()
        if not np.isfinite(heat) or heat >= 0.0
    ]
    if invalid_heats:
        raise ValueError(
            "Heats of adsorption must be finite and negative for "
            f"{invalid_heats}."
        )
    return heats


def resolve_solid_heat_capacity(declaration, temperature_K):
    """Return the solid heat capacity [J/kg/K] of a `{model, parameters}` declaration.

    ``Low2023`` is the Low et al. (2023) correlation cp = a/T^2 + b*T - c,
    evaluated at ``temperature_K``; ``constant`` returns ``value_J_kgK``.
    """
    if not isinstance(declaration, Mapping) or set(declaration) != {
        "model",
        "parameters",
    }:
        raise ValueError(
            "Solid heat_capacity must contain exactly `model` and `parameters`."
        )
    model = str(declaration["model"]).strip().lower()
    parameters = declaration["parameters"]
    if not isinstance(parameters, Mapping):
        raise TypeError("Solid heat-capacity parameters must be a mapping.")

    if model == "low2023":
        expected = SOLID_HEAT_CAPACITY_MODELS[model]
        if set(parameters) != expected:
            raise ValueError(
                "Low2023 heat capacity requires parameters "
                f"{sorted(expected)}."
            )
        a = float(parameters["a_J_K_per_kg"])
        b = float(parameters["b_J_per_kgK2"])
        c = float(parameters["c_J_per_kgK"])
        cp = a * temperature_K**(-2) + b * temperature_K - c
    elif model == "constant":
        if set(parameters) != SOLID_HEAT_CAPACITY_MODELS[model]:
            raise ValueError(
                "Constant solid heat capacity requires only `value_J_kgK`."
            )
        cp = float(parameters["value_J_kgK"])
    else:
        raise ValueError(
            f"Unknown solid heat-capacity model {declaration['model']!r}; "
            "available models are Low2023 and constant."
        )
    if not np.isfinite(cp) or cp <= 0.0:
        raise ValueError("Resolved solid heat capacity must be finite and positive.")
    return cp


def _species_keys(configured, what):
    """Map each key of `configured` to its species name (stripped, upper-case).

    Species names are matched without regard to case, as everywhere in a case;
    two keys naming one species are refused instead of one silently winning.
    """
    names = {}
    for key in configured:
        species = str(key).strip().upper()
        if species in names.values():
            twins = sorted(str(k) for k, name in names.items() if name == species)
            raise ValueError(
                f"{what} names {species} twice ({', '.join([*twins, str(key)])}); "
                "species names do not depend on case."
            )
        names[key] = species
    return names


def resolve_fixed_adsorbates(configured):
    """Return the normalized inert-loading inventory keyed by upper-case species."""
    if not isinstance(configured, Mapping):
        raise TypeError("simulation.sorbent.fixed_adsorbates must be a mapping.")

    resolved = {}
    names = _species_keys(configured, "simulation.sorbent.fixed_adsorbates")
    for raw_species, properties in configured.items():
        species = names[raw_species]
        if not species or not isinstance(properties, Mapping):
            raise TypeError(
                "Each fixed adsorbate must have a species name and a property mapping."
            )
        missing = sorted(REQUIRED_FIXED_ADSORBATE_FIELDS - set(properties))
        unknown = sorted(set(properties) - REQUIRED_FIXED_ADSORBATE_FIELDS)
        if missing or unknown:
            raise ValueError(
                f"Invalid fixed-adsorbate properties for {species} "
                f"(missing={missing}, unknown={unknown})."
            )
        loading = float(properties["loading_mol_per_kg"])
        heat_capacity = float(properties["heat_capacity_J_molK"])
        if not np.isfinite(loading) or loading < 0.0:
            raise ValueError(
                f"Fixed loading for {species} must be finite and non-negative."
            )
        if not np.isfinite(heat_capacity) or heat_capacity <= 0.0:
            raise ValueError(
                f"Fixed-adsorbate heat capacity for {species} must be "
                "finite and positive."
            )
        resolved[species] = {
            "loading_mol_per_kg": loading,
            "heat_capacity_J_molK": heat_capacity,
        }
    return resolved


def resolve_initial_solid_loadings(configured, adsorbates):
    """Return the declared initial loadings [mol/kg], or `None` when unset.

    `None` marks the value the model itself supplies by equilibrating the
    initial gas state, so it is a valid input rather than a missing one.
    """
    if configured is None:
        return None
    if not isinstance(configured, Mapping):
        raise TypeError(
            "initial_solid_loadings_mol_per_kg must be an "
            "adsorbate-keyed mapping."
        )

    names = _species_keys(configured, "initial_solid_loadings_mol_per_kg")
    normalized = {names[species]: value for species, value in configured.items()}
    expected_adsorbates = set(adsorbates)
    missing = sorted(expected_adsorbates - set(normalized))
    unknown = sorted(set(normalized) - expected_adsorbates)
    if missing or unknown:
        raise ValueError(
            "Initial solid loadings must cover the declared adsorbates "
            f"exactly (missing={missing}, unknown={unknown})."
        )

    loadings = {species: float(normalized[species]) for species in adsorbates}
    invalid_initial = [
        species
        for species, loading in loadings.items()
        if not np.isfinite(loading) or loading < 0.0
    ]
    if invalid_initial:
        raise ValueError(
            "Initial solid loadings must be finite and non-negative for "
            f"{invalid_initial}."
        )
    return loadings


class _IAPWSWater:
    """Water's vapor pressure when the H2O model of the case gives none."""

    def EOS(self, T):
        from pydac.core.isotherm_models import water_saturation_pressure

        return water_saturation_pressure(T)


def require_water_component(equilibrium_system):
    """Return the H2O component whose EOS gives water's vapor pressure (RH, steam).

    The vapor pressure is a property of water, not of the sorbent: an H2O model
    may give its own ``EOS(T)``; otherwise pyDAC's IAPWS-1995 correlation is
    used, the one the built-in GAB model returns.
    """
    water = getattr(equilibrium_system, "component_models", {}).get("H2O")
    if water is None or not callable(getattr(water, "EOS", None)):
        return _IAPWSWater()
    return water


def validate_warm_start_fields(
    prev_fields,
    *,
    step,
    layout_signature,
    Nz,
    Nr,
    Nc_ci,
    Nc_solid,
):
    """Check a saved previous-step state against the current layout and grid.

    Returns the previous grid `(prev_Nz, prev_Nr)` once every field shape, the
    state-layout signature, and the grid itself match what this step expects.
    """
    if prev_fields is None:
        raise ValueError(
            "Warm-start initialization requires `prev_fields` for non-startup steps. "
            f"Received None while building step '{step}'."
        )

    prev_Nz = int(prev_fields.get("Nz", prev_fields["c_tot"].shape[0]))
    prev_Nr = int(prev_fields.get("Nr", prev_fields["c_tot"].shape[1]))
    previous_layout_signature = prev_fields.get("layout_signature")
    if (
        previous_layout_signature is not None
        and previous_layout_signature != layout_signature
    ):
        raise ValueError(
            "Cannot initialize the next step from a different state layout. "
            f"Previous signature: {previous_layout_signature}; "
            f"current signature: {layout_signature}."
        )

    expected_ci_shape = (prev_Nz, prev_Nr, Nc_ci)
    expected_qi_shape = (prev_Nz, prev_Nr, Nc_solid)
    expected_scalar_shape = (prev_Nz, prev_Nr)
    expected_wall_shape = (prev_Nz,)

    if prev_fields["ci"].shape != expected_ci_shape:
        raise ValueError(
            "Cannot initialize the next step because the saved `ci` field shape is inconsistent "
            f"with the previous-step grid metadata. Expected {expected_ci_shape}, "
            f"found {prev_fields['ci'].shape}."
        )
    if prev_fields["qi"].shape != expected_qi_shape:
        raise ValueError(
            "Cannot initialize the next step because the saved `qi` field shape is inconsistent "
            f"with the previous-step grid metadata. Expected {expected_qi_shape}, "
            f"found {prev_fields['qi'].shape}."
        )
    if prev_fields["c_tot"].shape != expected_scalar_shape or prev_fields["T"].shape != expected_scalar_shape:
        raise ValueError(
            "Cannot initialize the next step because the saved scalar field shapes are inconsistent "
            f"with the previous-step grid metadata (Nz={prev_Nz}, Nr={prev_Nr})."
        )
    if prev_fields["T_wall"].shape != expected_wall_shape:
        raise ValueError(
            "Cannot initialize the next step because the saved `T_wall` field shape is inconsistent "
            f"with the previous-step grid metadata. Expected {expected_wall_shape}, "
            f"found {prev_fields['T_wall'].shape}."
        )

    if prev_Nz != Nz or prev_Nr != Nr:
        raise ValueError(
            "Cannot proceed with the next step because the previous step was saved on a different grid. "
            f"Previous step grid: Nz={prev_Nz}, Nr={prev_Nr}. "
            f"Current grid: Nz={Nz}, Nr={Nr}. "
            "Re-run the previous step on the requested grid or keep the same grid for the full sequence."
        )

    return prev_Nz, prev_Nr
