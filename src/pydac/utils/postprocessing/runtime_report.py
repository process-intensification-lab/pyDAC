"""Build and validate the bounded human-readable runtime report.

The JSON artifact describes initialization-time physics and identifies whether
spatial values live in a full ``sol.pkl`` trajectory or a compact
``terminal.pkl`` endpoint.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from numbers import Real
from typing import Mapping

from pydac.config.serialization import to_jsonable

from pydac.core.state_layout import StateLayout

from pydac.utils.postprocessing.context_model import simulation_context_dataframe_from_model
from pydac.utils.postprocessing.context_schema import STATIC_STEP, STEP_ORDER
from pydac.utils.solver.persistence import MOO_COMPACT, RUN_FULL, normalize_persistence_policy


RUNTIME_REPORT_SCHEMA_VERSION = 1
RUNTIME_REPORT_FILENAME = "resolved_runtime.json"
_TOP_LEVEL_KEYS = (
    "schema_version",
    "provenance",
    "state_layout",
    "models",
    "species_catalog",
    "static_context",
    "spatial_values_policy",
    "steps",
)


def runtime_report_from_model(
    model,
    *,
    persistence_policy: str = RUN_FULL,
) -> dict[str, object]:
    """Return one validated single-step report from an initialized model."""
    persistence_policy = normalize_persistence_policy(persistence_policy)
    properties = getattr(model, "runtime_properties", None)
    if not isinstance(properties, Mapping):
        raise AttributeError("Model has no structured `runtime_properties` mapping.")

    layout = getattr(model, "state_layout", None)
    if not isinstance(layout, StateLayout):
        raise AttributeError("Model has no generated StateLayout.")
    step = str(getattr(model, "step", "")).strip().lower()
    if step not in STEP_ORDER:
        raise ValueError(f"Unsupported runtime-report step {step!r}.")

    context_table = simulation_context_dataframe_from_model(model)
    static_context = _context_records(
        context_table.loc[context_table["Step"].eq(STATIC_STEP)]
    )
    step_context = _context_records(
        context_table.loc[context_table["Step"].eq(step)]
    )

    compact = persistence_policy == MOO_COMPACT
    artifact_filename = "terminal.pkl" if compact else "sol.pkl"
    report = {
        "schema_version": RUNTIME_REPORT_SCHEMA_VERSION,
        "provenance": {
            "producer": "pyDAC v3",
            "source": "initialized ColumnModelMoL",
            "resolved_config_artifact": "resolved_config.json",
            "spatial_solution_pattern": f"<step>/{artifact_filename}",
        },
        "state_layout": layout.to_dict(),
        "models": _model_report(model),
        "species_catalog": _species_catalog_report(model),
        "static_context": static_context,
        "spatial_values_policy": {
            "mode": (
                "external_compact_terminal"
                if compact
                else "external_solution_reference"
            ),
            "reason": (
                "Compact MOO runs retain only one dimensional terminal state per step; "
                "full histories are consumed in memory and released after metrics."
                if compact
                else "Spatial histories are not duplicated in JSON; dimensional values "
                "remain in each step sol.pkl and are interpreted through StateLayout."
            ),
        },
        "steps": {
            step: {
                "evaluation_state": _evaluation_report(properties),
                "properties": _evaluated_property_report(model, properties),
                "resolved_context": step_context,
                "spatial_values": {
                    "artifact": f"{step}/{artifact_filename}",
                    "layout_signature": layout.signature,
                    "fields": [
                        {
                            "name": field.name,
                            "unit": field.unit,
                            "shape": list(field.shape),
                            "storage": field.storage,
                        }
                        for field in layout.pde_fields
                    ],
                },
            }
        },
    }
    return validate_runtime_report(report)


def merge_runtime_reports(
    existing: Mapping[str, object], incoming: Mapping[str, object]
) -> dict[str, object]:
    """Merge new step blocks without altering prior completed information."""
    left = validate_runtime_report(existing)
    right = validate_runtime_report(incoming)

    for key in _TOP_LEVEL_KEYS:
        if key == "steps":
            continue
        if left[key] != right[key]:
            raise ValueError(
                f"Runtime report static section {key!r} differs from the existing "
                "result folder. Use a separate results folder."
            )

    merged_steps = dict(left["steps"])
    for step, block in right["steps"].items():
        if step in merged_steps and merged_steps[step] != block:
            raise ValueError(
                f"Runtime report block for completed step {step!r} differs from "
                "the existing result folder. Use a separate results folder."
            )
        merged_steps.setdefault(step, block)

    ordered_steps = {
        step: merged_steps[step] for step in STEP_ORDER if step in merged_steps
    }
    return validate_runtime_report({**left, "steps": ordered_steps})


def validate_runtime_report(document: Mapping[str, object]) -> dict[str, object]:
    """Return a normalized strict-JSON-compatible runtime report."""
    if not isinstance(document, Mapping):
        raise TypeError("Runtime report must be a mapping.")
    report = to_jsonable(document)
    _require_exact_keys(report, _TOP_LEVEL_KEYS, "runtime report")
    if report["schema_version"] != RUNTIME_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported runtime-report schema {report['schema_version']!r}; "
            f"expected {RUNTIME_REPORT_SCHEMA_VERSION}."
        )

    layout = StateLayout.from_dict(report["state_layout"])
    provenance = _mapping(report["provenance"], "provenance")
    _require_exact_keys(
        provenance,
        (
            "producer",
            "source",
            "resolved_config_artifact",
            "spatial_solution_pattern",
        ),
        "provenance",
    )
    _validate_text_mapping(provenance, "provenance")
    spatial_policy = _mapping(
        report["spatial_values_policy"], "spatial_values_policy"
    )
    _require_exact_keys(
        spatial_policy, ("mode", "reason"), "spatial_values_policy"
    )
    _validate_text_mapping(spatial_policy, "spatial_values_policy")
    pattern = provenance["spatial_solution_pattern"]
    mode = spatial_policy["mode"]
    expected_mode_by_pattern = {
        "<step>/sol.pkl": "external_solution_reference",
        "<step>/terminal.pkl": "external_compact_terminal",
    }
    if pattern not in expected_mode_by_pattern:
        raise ValueError(f"Unsupported spatial solution pattern {pattern!r}.")
    if mode != expected_mode_by_pattern[pattern]:
        raise ValueError(
            "Runtime spatial-values policy is inconsistent with its artifact pattern."
        )

    catalog = _mapping(report["species_catalog"], "species_catalog")
    if tuple(catalog) != layout.mixture.species:
        raise ValueError(
            "Runtime species catalog order must match StateLayout mixture order."
        )
    for species, values in catalog.items():
        _validate_catalog_species(species, values)

    _validate_models(report["models"], layout)
    _validate_context_records(report["static_context"], expected_step=None)

    steps = _mapping(report["steps"], "steps")
    if not steps:
        raise ValueError("Runtime report must contain at least one step block.")
    if tuple(steps) != tuple(step for step in STEP_ORDER if step in steps):
        raise ValueError("Runtime report steps must follow canonical cycle order.")
    for step, block in steps.items():
        if step not in STEP_ORDER:
            raise ValueError(f"Unsupported runtime-report step {step!r}.")
        _validate_step_block(step, block, layout, spatial_pattern=pattern)

    _reject_nonfinite(report, "runtime report")
    return report


def _model_report(model) -> dict[str, object]:
    adsorption = model.column_config.adsorption
    systems = tuple(model.equilibrium_system.systems)
    selections = tuple(adsorption.equilibrium_systems.items())
    if len(systems) != len(selections):
        raise ValueError("Runtime equilibrium implementations and selections differ.")

    equilibrium = []
    for (name, selection), system in zip(selections, systems, strict=True):
        equilibrium.append(
            {
                "name": name,
                "implementation": type(system).__name__,
                "adsorbates": list(system.species),
                "datasets": to_jsonable(selection.datasets),
                "parameters": to_jsonable(selection.parameters),
            }
        )

    kinetics = {}
    for species, selection in adsorption.kinetics.items():
        law = model.kinetics[species]
        kinetics[species] = {
            "name": selection.model,
            "implementation": type(law).__name__,
            "parameters": to_jsonable(selection.parameters),
        }
    return {"equilibrium_systems": equilibrium, "kinetics": kinetics}


def _species_catalog_report(model) -> dict[str, object]:
    report = {}
    for species in model.gas_species:
        record = model.species_properties[species]
        coefficients = asdict(record.heat_capacity_shomate)
        report[species] = {
            "molecular_weight": _value(record.molecular_weight_kg_per_mol, "kg mol^-1"),
            "lennard_jones_sigma": _value(record.lennard_jones_sigma_m, "m"),
            "lennard_jones_epsilon_over_k": _value(
                record.lennard_jones_epsilon_over_k_K, "K"
            ),
            "fuller_diffusion_volume": _value(
                record.fuller_diffusion_volume_m3_per_mol, "m^3 mol^-1"
            ),
            "thermal_conductivity_reference": {
                "value": record.thermal_conductivity_reference_W_per_mK,
                "unit": "W m^-1 K^-1",
                "temperature_K": record.thermal_conductivity_reference_temperature_K,
            },
            "heat_capacity_correlation": {
                "name": "Shomate",
                "temperature_unit": "K",
                "output_unit": "J mol^-1 K^-1",
                "coefficients": coefficients,
            },
        }
    return report


def _evaluation_report(properties: Mapping[str, object]) -> dict[str, object]:
    evaluation = _mapping(properties.get("evaluation"), "runtime_properties.evaluation")
    return {
        "temperature": _value(evaluation["temperature_K"], "K"),
        "pressure": _value(evaluation["pressure_Pa"], "Pa"),
        "gas_composition_source": evaluation["gas_composition_source"],
        "gas_mole_fractions": {
            "unit": "mol mol^-1",
            "values": dict(evaluation["gas_mole_fractions"]),
        },
        "dispersion_temperature": _value(evaluation["dispersion_temperature_K"], "K"),
        "dispersion_pressure": _value(evaluation["dispersion_pressure_Pa"], "Pa"),
        "dispersion_composition_source": evaluation["dispersion_composition_source"],
    }


def _evaluated_property_report(model, properties: Mapping[str, object]) -> dict[str, object]:
    species_properties = _mapping(properties.get("species"), "runtime_properties.species")
    inlet = _mapping(properties.get("inlet"), "runtime_properties.inlet")
    mixture = _mapping(properties.get("mixture"), "runtime_properties.mixture")
    solid = _mapping(properties.get("solid"), "runtime_properties.solid")
    dimensionless = _mapping(
        properties.get("dimensionless"), "runtime_properties.dimensionless"
    )

    per_species_specs = {
        "heat_capacity": ("heat_capacity_J_molK", "J mol^-1 K^-1"),
        "viscosity": ("viscosity_Pa_s", "Pa s"),
        "mixture_diffusivity": ("mixture_diffusivity_m2_s", "m^2 s^-1"),
        "schmidt_number": ("schmidt_number", "-"),
        "sherwood_number": ("sherwood_number", "-"),
    }
    per_species = {
        name: {
            "unit": unit,
            "values": {
                species: species_properties[species][source]
                for species in model.gas_species
            },
        }
        for name, (source, unit) in per_species_specs.items()
    }
    per_species["inlet_mass_flow"] = {
        "unit": "kg s^-1",
        "values": dict(inlet["species_mass_flow_kg_s"]),
    }

    pairs = []
    binary = _mapping(
        properties.get("binary_diffusivity_m2_s"),
        "runtime_properties.binary_diffusivity_m2_s",
    )
    for i, species_i in enumerate(model.gas_species):
        for species_j in model.gas_species[i + 1 :]:
            pairs.append(
                {
                    "species": [species_i, species_j],
                    "value": binary[f"{species_i}-{species_j}"],
                }
            )

    scalar = {
        "mixture_density": _value(mixture["density_kg_m3"], "kg m^-3"),
        "mixture_viscosity": _value(mixture["viscosity_Pa_s"], "Pa s"),
        "mixture_heat_capacity": _value(
            mixture["heat_capacity_J_molK"], "J mol^-1 K^-1"
        ),
        "mixture_thermal_conductivity": _value(
            mixture["thermal_conductivity_W_mK"], "W m^-1 K^-1"
        ),
        "solid_heat_capacity": _value(solid["heat_capacity_J_kgK"], "J kg^-1 K^-1"),
        "solid_evaluation_temperature": _value(
            solid["evaluation_temperature_K"], "K"
        ),
        "reynolds_number": _value(dimensionless["reynolds_number"], "-"),
        "prandtl_number": _value(dimensionless["prandtl_number"], "-"),
        "inlet_superficial_velocity": _value(inlet["superficial_velocity_m_s"], "m s^-1"),
        "inlet_total_concentration_flux_nd": _value(
            inlet["total_concentration_flux_nd"], "-"
        ),
        "inlet_enthalpy_flux_nd": _value(inlet["enthalpy_flux_nd"], "-"),
    }
    if "steam_density_kg_m3" in inlet:
        scalar["inlet_steam_density"] = _value(inlet["steam_density_kg_m3"], "kg m^-3")

    return {
        "scalar": scalar,
        "per_species": per_species,
        "pairwise": {
            "binary_diffusivity": {"unit": "m^2 s^-1", "pairs": pairs}
        },
    }


def _context_records(table) -> list[dict[str, object]]:
    return [
        {
            "category": row["Category"],
            "parameter_id": row["Parameter ID"],
            "value": row["Value"],
            "unit": row["Unit"],
        }
        for row in table.to_dict(orient="records")
    ]


def _validate_step_block(
    step: str,
    block: object,
    layout: StateLayout,
    *,
    spatial_pattern: str,
) -> None:
    values = _mapping(block, f"steps.{step}")
    _require_exact_keys(
        values,
        ("evaluation_state", "properties", "resolved_context", "spatial_values"),
        f"steps.{step}",
    )
    evaluation = _mapping(values["evaluation_state"], f"steps.{step}.evaluation_state")
    _require_exact_keys(
        evaluation,
        (
            "temperature",
            "pressure",
            "gas_composition_source",
            "gas_mole_fractions",
            "dispersion_temperature",
            "dispersion_pressure",
            "dispersion_composition_source",
        ),
        f"steps.{step}.evaluation_state",
    )
    for name in (
        "temperature",
        "pressure",
        "dispersion_temperature",
        "dispersion_pressure",
    ):
        _validate_value_record(evaluation[name], f"evaluation_state.{name}")
    for name in ("gas_composition_source", "dispersion_composition_source"):
        if not isinstance(evaluation[name], str) or not evaluation[name].strip():
            raise ValueError(f"Step {step!r} {name} must be non-empty text.")
    fractions = _mapping(
        evaluation.get("gas_mole_fractions"),
        f"steps.{step}.evaluation_state.gas_mole_fractions",
    )
    _require_exact_keys(
        fractions, ("unit", "values"), "gas_mole_fractions"
    )
    if not isinstance(fractions["unit"], str) or not fractions["unit"].strip():
        raise ValueError("gas_mole_fractions requires a non-empty unit.")
    fraction_values = _mapping(fractions.get("values"), "gas_mole_fractions.values")
    if tuple(fraction_values) != layout.mixture.species:
        raise ValueError(f"Step {step!r} mole fractions must follow mixture order.")
    for species, value in fraction_values.items():
        _number(value, f"gas_mole_fractions.values.{species}")

    properties = _mapping(values["properties"], f"steps.{step}.properties")
    _require_exact_keys(properties, ("scalar", "per_species", "pairwise"), "properties")
    for name, record in _mapping(properties["scalar"], "properties.scalar").items():
        _validate_value_record(record, f"scalar.{name}")
    for name, record in _mapping(properties["per_species"], "properties.per_species").items():
        item = _mapping(record, f"per_species.{name}")
        _require_exact_keys(item, ("unit", "values"), f"per_species.{name}")
        if not isinstance(item["unit"], str) or not item["unit"].strip():
            raise ValueError(f"per_species.{name} requires a non-empty unit.")
        per_species_values = _mapping(
            item["values"], f"per_species.{name}.values"
        )
        if tuple(per_species_values) != layout.mixture.species:
            raise ValueError(f"Per-species property {name!r} must follow mixture order.")
        for species, value in per_species_values.items():
            _number(value, f"per_species.{name}.values.{species}")

    pairwise = _mapping(properties["pairwise"], "properties.pairwise")
    _require_exact_keys(
        pairwise, ("binary_diffusivity",), "properties.pairwise"
    )
    binary = pairwise["binary_diffusivity"]
    binary = _mapping(binary, "properties.pairwise.binary_diffusivity")
    _require_exact_keys(
        binary, ("unit", "pairs"), "properties.pairwise.binary_diffusivity"
    )
    if not isinstance(binary["unit"], str) or not binary["unit"].strip():
        raise ValueError("Binary diffusivity requires a non-empty unit.")
    if not isinstance(binary["pairs"], list):
        raise TypeError("Binary-diffusivity pairs must be a list.")
    for index, pair in enumerate(binary["pairs"]):
        pair_values = _mapping(pair, f"binary_diffusivity.pairs[{index}]")
        _require_exact_keys(
            pair_values, ("species", "value"), f"binary_diffusivity.pairs[{index}]"
        )
        _number(pair_values["value"], f"binary_diffusivity.pairs[{index}].value")
    expected_pairs = [
        [species_i, species_j]
        for i, species_i in enumerate(layout.mixture.species)
        for species_j in layout.mixture.species[i + 1 :]
    ]
    if [pair.get("species") for pair in binary.get("pairs", [])] != expected_pairs:
        raise ValueError("Binary-diffusivity pairs must follow declared mixture order.")

    _validate_context_records(values["resolved_context"], expected_step=step)
    spatial = _mapping(values["spatial_values"], f"steps.{step}.spatial_values")
    _require_exact_keys(
        spatial,
        ("artifact", "layout_signature", "fields"),
        f"steps.{step}.spatial_values",
    )
    artifact_suffix = spatial_pattern.removeprefix("<step>/")
    expected_artifact = f"{step}/{artifact_suffix}"
    if spatial.get("artifact") != expected_artifact:
        raise ValueError(
            f"Step {step!r} spatial artifact must be {expected_artifact!r}."
        )
    if spatial.get("layout_signature") != layout.signature:
        raise ValueError(f"Step {step!r} spatial layout signature is incompatible.")
    expected_fields = [
        {
            "name": field.name,
            "unit": field.unit,
            "shape": list(field.shape),
            "storage": field.storage,
        }
        for field in layout.pde_fields
    ]
    if spatial.get("fields") != expected_fields:
        raise ValueError(f"Step {step!r} spatial field metadata differs from StateLayout.")


def _validate_models(models: object, layout: StateLayout) -> None:
    values = _mapping(models, "models")
    _require_exact_keys(values, ("equilibrium_systems", "kinetics"), "models")
    equilibrium = values["equilibrium_systems"]
    if not isinstance(equilibrium, list) or not equilibrium:
        raise ValueError("models.equilibrium_systems must be a non-empty list.")
    for index, selection in enumerate(equilibrium):
        item = _mapping(selection, f"models.equilibrium_systems[{index}]")
        _require_exact_keys(
            item,
            ("name", "implementation", "adsorbates", "datasets", "parameters"),
            f"models.equilibrium_systems[{index}]",
        )
        for name in ("name", "implementation"):
            if not isinstance(item[name], str) or not item[name].strip():
                raise ValueError(
                    f"models.equilibrium_systems[{index}].{name} must be text."
                )
        if not isinstance(item["adsorbates"], list):
            raise TypeError("Runtime equilibrium adsorbates must be a list.")
        _mapping(item["datasets"], f"equilibrium_systems[{index}].datasets")
        _mapping(item["parameters"], f"equilibrium_systems[{index}].parameters")
    covered = tuple(
        species for selection in equilibrium for species in selection.get("adsorbates", [])
    )
    if covered != layout.adsorbates:
        raise ValueError("Runtime equilibrium systems must cover layout adsorbates in order.")
    kinetics = _mapping(values["kinetics"], "models.kinetics")
    if tuple(kinetics) != layout.adsorbates:
        raise ValueError("Runtime kinetics must follow layout adsorbate order.")
    for species, record in kinetics.items():
        item = _mapping(record, f"models.kinetics.{species}")
        _require_exact_keys(
            item, ("name", "implementation", "parameters"), f"models.kinetics.{species}"
        )
        for name in ("name", "implementation"):
            if not isinstance(item[name], str) or not item[name].strip():
                raise ValueError(f"models.kinetics.{species}.{name} must be text.")
        _mapping(item["parameters"], f"models.kinetics.{species}.parameters")


def _validate_catalog_species(species: str, values: object) -> None:
    item = _mapping(values, f"species_catalog.{species}")
    required = (
        "molecular_weight",
        "lennard_jones_sigma",
        "lennard_jones_epsilon_over_k",
        "fuller_diffusion_volume",
        "thermal_conductivity_reference",
        "heat_capacity_correlation",
    )
    _require_exact_keys(item, required, f"species_catalog.{species}")
    for key in required[:4]:
        _validate_value_record(item[key], f"species_catalog.{species}.{key}")
    conductivity = _mapping(
        item["thermal_conductivity_reference"],
        f"species_catalog.{species}.thermal_conductivity_reference",
    )
    _require_exact_keys(
        conductivity,
        ("value", "unit", "temperature_K"),
        f"species_catalog.{species}.thermal_conductivity_reference",
    )
    _number(conductivity["value"], f"species_catalog.{species}.conductivity.value")
    _number(
        conductivity["temperature_K"],
        f"species_catalog.{species}.conductivity.temperature_K",
    )
    if not isinstance(conductivity["unit"], str) or not conductivity["unit"].strip():
        raise ValueError(f"Species {species} conductivity requires a unit.")

    correlation = _mapping(
        item["heat_capacity_correlation"],
        f"species_catalog.{species}.heat_capacity_correlation",
    )
    _require_exact_keys(
        correlation,
        ("name", "temperature_unit", "output_unit", "coefficients"),
        f"species_catalog.{species}.heat_capacity_correlation",
    )
    for name in ("name", "temperature_unit", "output_unit"):
        if not isinstance(correlation[name], str) or not correlation[name].strip():
            raise ValueError(f"Species {species} heat-capacity {name} must be text.")
    coefficients = _mapping(
        correlation["coefficients"],
        f"species_catalog.{species}.heat_capacity_correlation.coefficients",
    )
    expected_coefficients = (
        "a_J_per_molK",
        "b_J_per_molK",
        "c_J_per_molK",
        "d_J_per_molK",
        "e_J_per_molK",
    )
    _require_exact_keys(
        coefficients,
        expected_coefficients,
        f"species_catalog.{species}.heat_capacity_correlation.coefficients",
    )
    for name, value in coefficients.items():
        _number(value, f"species_catalog.{species}.coefficients.{name}")


def _validate_context_records(records: object, *, expected_step: str | None) -> None:
    if not isinstance(records, list) or not records:
        scope = "static" if expected_step is None else expected_step
        raise ValueError(f"Runtime {scope} context must be a non-empty list.")
    seen = set()
    for record in records:
        item = _mapping(record, "context record")
        _require_exact_keys(
            item, ("category", "parameter_id", "value", "unit"), "context record"
        )
        parameter_id = str(item["parameter_id"]).strip()
        if not parameter_id or parameter_id in seen:
            raise ValueError(f"Invalid or duplicate runtime context ID {parameter_id!r}.")
        seen.add(parameter_id)
        if not str(item["unit"]).strip():
            raise ValueError(f"Runtime context {parameter_id!r} needs a unit.")
        for name in ("category", "parameter_id", "unit"):
            if not isinstance(item[name], str) or not item[name].strip():
                raise ValueError(f"Runtime context field {name!r} must be text.")
        if isinstance(item["value"], str):
            if not item["value"].strip():
                raise ValueError(f"Runtime context {parameter_id!r} has empty text.")
        else:
            _number(item["value"], f"runtime context {parameter_id}.value")


def _value(value: object, unit: str) -> dict[str, object]:
    return {"value": value, "unit": unit}


def _validate_value_record(record: object, location: str) -> None:
    values = _mapping(record, location)
    _require_exact_keys(values, ("value", "unit"), location)
    if not isinstance(values["unit"], str) or not values["unit"].strip():
        raise ValueError(f"{location} requires a non-empty unit.")
    _number(values["value"], f"{location}.value")


def _number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{location} must be a real number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{location} must be finite.")
    return numeric


def _mapping(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping.")
    return dict(value)


def _require_exact_keys(values: Mapping[str, object], expected, location: str) -> None:
    expected_set = set(expected)
    actual = set(values)
    if actual != expected_set:
        raise ValueError(
            f"{location} keys differ; missing={sorted(expected_set - actual)}, "
            f"unknown={sorted(actual - expected_set)}."
        )


def _validate_text_mapping(value: object, location: str) -> None:
    values = _mapping(value, location)
    if not values or any(not isinstance(item, str) or not item.strip() for item in values.values()):
        raise ValueError(f"{location} values must be non-empty text.")


def _reject_nonfinite(value: object, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{location}[{index}]")
    elif isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{location} contains non-finite value {value!r}.")


__all__ = [
    "RUNTIME_REPORT_FILENAME",
    "RUNTIME_REPORT_SCHEMA_VERSION",
    "merge_runtime_reports",
    "runtime_report_from_model",
    "validate_runtime_report",
]
