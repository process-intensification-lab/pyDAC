"""Flatten initialized column models into simulation-context schema rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pydac.utils.solver import pressure_drop_metadata_from_model

from pydac.utils.postprocessing.context_schema import (
    ARCHITECTURE_STATIC_BY_ID,
    COMMON_STATIC_BY_ID,
    COMMON_STATIC_PARAMETERS,
    REGISTERED_ARCHITECTURE_MARKER,
    SIMULATION_CONTEXT_COLUMNS,
    STEP_ORDER,
    context_definition_row,
    nonempty_context_text,
    step_definitions_by_id,
)


_BASE_MODEL_ATTRIBUTES = {
    "grid.nz": "Nz",
    "grid.nr": "Nr",
    "bed.length": "L_b",
    "bed.radius": "R_b",
    "bed.volume": "bed_volume",
    "bed.aspect.ratio": "tau_ar",
    "particle.density": "rho_p",
    "particle.porosity": "eps_p",
    "particle.radius": "R_p",
}
_DESIGN_PARAMETER_IDS = {
    "eps_b": "bed.design.eps_b",
}
_STRUCTURE_PARAMETER_IDS = {
    "eps_flow": "bed.resolved.eps_flow",
    "eps_total": "bed.resolved.eps_total",
    "rho_sorbent": "bed.resolved.rho_sorbent",
    "acc_structure": "bed.resolved.acc_structure",
}
_TRANSPORT_PARAMETER_IDS = {
    "lambda_ax": "bed.resolved.lambda_ax",
    "lambda_rad": "bed.resolved.lambda_rad",
    "h_wall": "bed.resolved.h_wall",
    "wall_bed_htc": "bed.resolved.wall_bed_htc",
}
_MODEL_PARAMETER_IDS = {
    "wall_htc": "bed.models.wall_htc",
    "overall_wall_htc": "bed.models.overall_wall_htc",
    "pressure_drop": "bed.models.pressure_drop",
    "dispersion": "bed.models.dispersion",
}
_CLOSURE_PARAMETER_IDS = {
    "temperature": "bed.closure.temperature",
    "pressure": "bed.closure.pressure",
    "superficial_inlet_velocity": "bed.closure.superficial_inlet_velocity",
    "gas_composition_source": "bed.closure.gas_composition_source",
    "dispersion_composition_source": "bed.closure.dispersion_composition_source",
}


def _registered_architecture_rows(provider, context):
    """Return a registered architecture's rows, in the order its entry declares.

    ``bed.design.<key>`` and ``bed.material.<key>`` come from the canonical
    design/material mappings, ``bed.provenance.<key>`` from the resolved
    provenance. Every declared row needs its value, and every design/material
    key needs a declared row, so nothing of the case identity is left out.
    """
    resolved = context.get("resolved") or {}
    sources = {
        "design": context.get("design") or {},
        "material": context.get("material") or {},
        "provenance": resolved.get("provenance") or {},
    }
    written = {category: set() for category in sources}
    rows = []
    for definition in provider.context_parameters:
        _, category, key = definition.parameter_id.split(".")
        if key not in sources[category]:
            raise ValueError(
                f"Bed architecture '{provider.name}' declares "
                f"'{definition.parameter_id}', but its resolved context has no "
                f"{category} value '{key}'."
            )
        rows.append(context_definition_row(definition, sources[category][key]))
        written[category].add(key)
    for category in ("design", "material"):
        undeclared = sorted(set(sources[category]) - written[category])
        if undeclared:
            raise ValueError(
                f"Bed architecture '{provider.name}' has {category} keys "
                f"{undeclared} without a declared bed.{category}.* context row."
            )
    return rows


def simulation_context_dataframe_from_model(model):
    """Capture the actual resolved context from one initialized step model."""
    context = getattr(model, "bed_architecture_context", None)
    if not isinstance(context, dict):
        raise AttributeError(
            "Model has no resolved 'bed_architecture_context'."
        )
    architecture = context.get("architecture")
    # A registered architecture brings its own row definitions, pinned on the
    # model when it was built (never a fresh registry lookup).
    provider = getattr(model, "bed_architecture_provider", None)
    if provider is None and architecture not in ARCHITECTURE_STATIC_BY_ID:
        raise ValueError(
            f"Unsupported model bed architecture '{architecture}'."
        )
    if provider is not None and provider.name != architecture:
        raise ValueError(
            f"Model bed architecture '{architecture}' does not match its "
            f"registered provider '{provider.name}'."
        )
    step = nonempty_context_text(getattr(model, "step", None), "model.step").lower()
    if step not in STEP_ORDER:
        raise ValueError(f"Unsupported model step '{step}'.")
    species_names = tuple(str(name).strip().lower() for name in model.gas_species)
    step_by_id = step_definitions_by_id(species_names)

    rows = []
    for definition in COMMON_STATIC_PARAMETERS:
        if definition.parameter_id == "bed.architecture":
            value = architecture
        elif definition.parameter_id in _BASE_MODEL_ATTRIBUTES:
            attribute = _BASE_MODEL_ATTRIBUTES[definition.parameter_id]
            if not hasattr(model, attribute):
                raise AttributeError(
                    f"Model has no '{attribute}' required for "
                    f"'{definition.parameter_id}'."
                )
            value = getattr(model, attribute)
        else:
            continue
        rows.append(context_definition_row(definition, value))

    design = context.get("design")
    resolved = context.get("resolved")
    if not isinstance(design, dict) or not isinstance(resolved, dict):
        raise ValueError(
            "bed_architecture_context must contain design and resolved mappings."
        )
    if provider is None:
        architecture_definitions_by_id = ARCHITECTURE_STATIC_BY_ID[
            architecture
        ]
        for key, parameter_id in _DESIGN_PARAMETER_IDS.items():
            if key in design:
                rows.append(
                    context_definition_row(
                        architecture_definitions_by_id[parameter_id],
                        design[key],
                    )
                )
    else:
        rows.append(context_definition_row(REGISTERED_ARCHITECTURE_MARKER, provider.name))
        rows.extend(_registered_architecture_rows(provider, context))
    structure = resolved.get("structure")
    transport = resolved.get("transport")
    models = resolved.get("models")
    closure = resolved.get("step_property_context")
    for name, mapping in (
        ("structure", structure),
        ("transport", transport),
        ("models", models),
        ("step_property_context", closure),
    ):
        if not isinstance(mapping, dict):
            raise ValueError(
                f"bed_architecture_context resolved.{name} must be a mapping."
            )

    for key, parameter_id in _STRUCTURE_PARAMETER_IDS.items():
        rows.append(
            context_definition_row(
                COMMON_STATIC_BY_ID[parameter_id],
                structure[key],
            )
        )

    for key, parameter_id in _MODEL_PARAMETER_IDS.items():
        rows.append(
            context_definition_row(
                step_by_id[parameter_id],
                models[key],
                step=step,
            )
        )
    for key, parameter_id in _TRANSPORT_PARAMETER_IDS.items():
        rows.append(
            context_definition_row(
                step_by_id[parameter_id],
                transport[key],
                step=step,
            )
        )
    rows.append(
        context_definition_row(
            step_by_id["bed.resolved.effective_wall_bed_htc"],
            resolved["effective_wall_bed_htc"],
            step=step,
        )
    )

    pressure_metadata = pressure_drop_metadata_from_model(model)
    # A radial law has no single coefficient.  The context table is scalar, so
    # it carries the declared area mean; the per-ring law the bed actually used
    # stays in the saved pressure-drop metadata and in the resolved flow block.
    pressure_summary = pressure_metadata.get("area_mean", pressure_metadata)
    pressure_values = {
        "bed.pressure.linear_coefficient": pressure_summary[
            "linear_coefficient"
        ],
        "bed.pressure.inertial_coefficient_per_density": pressure_summary[
            "inertial_coefficient_per_density"
        ],
        "bed.pressure.reynolds_particle": model.Re,
        "bed.pressure.gas_viscosity": model.mu_g_mix,
    }
    for parameter_id, value in pressure_values.items():
        rows.append(
            context_definition_row(
                step_by_id[parameter_id],
                value,
                step=step,
            )
        )

    for key, parameter_id in _CLOSURE_PARAMETER_IDS.items():
        rows.append(
            context_definition_row(
                step_by_id[parameter_id],
                closure[key],
                step=step,
            )
        )

    dispersion = getattr(model, "resolved_dispersion", None)
    if not isinstance(dispersion, dict):
        raise AttributeError(
            "Model has no resolved dispersion mapping."
        )
    for direction, source_key in (("axial", "Disp_ax"), ("radial", "Disp_rad")):
        values = np.asarray(dispersion[source_key], dtype=float).reshape(-1)
        if values.size != len(species_names):
            raise ValueError(
                f"Resolved {direction} dispersion must contain "
                f"{len(species_names)} species values."
            )
        for species_name, value in zip(species_names, values, strict=True):
            parameter_id = f"bed.dispersion.{direction}.{species_name}"
            rows.append(
                context_definition_row(
                    step_by_id[parameter_id],
                    value,
                    step=step,
                )
            )

    return pd.DataFrame(rows, columns=SIMULATION_CONTEXT_COLUMNS)
