"""Schema definitions and validation for architecture-aware simulation context."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import pandas as pd


SIMULATION_CONTEXT_SCHEMA_VERSION = 2
SIMULATION_CONTEXT_COLUMNS = (
    "Category",
    "Parameter ID",
    "Step",
    "Value",
    "Unit",
)
STATIC_STEP = ""
STEP_ORDER = (
    "startup",
    "ads",
    "blow",
    "heat",
    "w_purge",
    "s_purge",
    "w_des",
    "s_des",
    "press",
    "cool",
)


@dataclass(frozen=True, slots=True)
class ContextParameter:
    """Definition of one stable simulation-context parameter."""

    category: str
    parameter_id: str
    unit: str
    value_type: str = "number"


COMMON_STATIC_PARAMETERS = (
    ContextParameter("grid", "grid.nz", "-"),
    ContextParameter("grid", "grid.nr", "-"),
    ContextParameter("geometry", "bed.length", "m"),
    ContextParameter("geometry", "bed.radius", "m"),
    ContextParameter("geometry", "bed.volume", "m3"),
    ContextParameter("geometry", "bed.aspect.ratio", "-"),
    ContextParameter("bed", "bed.architecture", "-", "text"),
    ContextParameter("particle", "particle.density", "kg/m3"),
    ContextParameter("particle", "particle.porosity", "-"),
    ContextParameter("particle", "particle.radius", "m"),
    ContextParameter("structure", "bed.resolved.eps_flow", "-"),
    ContextParameter("structure", "bed.resolved.eps_total", "-"),
    ContextParameter("structure", "bed.resolved.rho_sorbent", "kg/m3"),
    ContextParameter("structure", "bed.resolved.acc_structure", "J/m3/K"),
)

ARCHITECTURE_STATIC_PARAMETERS = {
    "packed_bed": (
        ContextParameter("design", "bed.design.eps_b", "-"),
    ),
}

BASE_STEP_PARAMETERS = (
    ContextParameter("models", "bed.models.heat_transfer", "-", "text"),
    ContextParameter("models", "bed.models.pressure_drop", "-", "text"),
    ContextParameter("models", "bed.models.dispersion", "-", "text"),
    ContextParameter("transport", "bed.resolved.lambda_ax", "W/m/K"),
    ContextParameter("transport", "bed.resolved.lambda_rad", "W/m/K"),
    ContextParameter("transport", "bed.resolved.h_wall", "W/m2/K"),
    ContextParameter("transport", "bed.resolved.wall_bed_htc", "W/m2/K"),
    ContextParameter(
        "transport",
        "bed.resolved.effective_wall_bed_htc",
        "W/m2/K",
    ),
    ContextParameter(
        "pressure",
        "bed.pressure.linear_coefficient",
        "Pa*s/m2",
    ),
    ContextParameter(
        "pressure",
        "bed.pressure.inertial_coefficient_per_density",
        "1/m",
    ),
    ContextParameter("pressure", "bed.pressure.reynolds_particle", "-"),
    ContextParameter("pressure", "bed.pressure.gas_viscosity", "Pa*s"),
    ContextParameter("closure", "bed.closure.temperature", "K"),
    ContextParameter("closure", "bed.closure.pressure", "Pa"),
    ContextParameter(
        "closure",
        "bed.closure.superficial_inlet_velocity",
        "m/s",
    ),
    ContextParameter(
        "closure",
        "bed.closure.gas_composition_source",
        "-",
        "text",
    ),
    ContextParameter(
        "closure",
        "bed.closure.dispersion_composition_source",
        "-",
        "text",
    ),
)

COMMON_STATIC_BY_ID = {
    definition.parameter_id: definition
    for definition in COMMON_STATIC_PARAMETERS
}
ARCHITECTURE_STATIC_BY_ID = {
    architecture: {
        definition.parameter_id: definition
        for definition in definitions
    }
    for architecture, definitions in ARCHITECTURE_STATIC_PARAMETERS.items()
}
REQUIRED_STATIC_PARAMETER_IDS = tuple(COMMON_STATIC_BY_ID)
REQUIRED_STEP_PARAMETER_IDS = tuple(
    definition.parameter_id for definition in BASE_STEP_PARAMETERS
)


def step_parameters(species):
    """Return the common step schema plus declared-order dispersion values."""
    names = tuple(str(name).strip().lower() for name in species)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Simulation-context species must be unique non-empty names.")
    return (
        *BASE_STEP_PARAMETERS,
        *(
            ContextParameter(
                "dispersion", f"bed.dispersion.axial.{name}", "m2/s"
            )
            for name in names
        ),
        *(
            ContextParameter(
                "dispersion", f"bed.dispersion.radial.{name}", "m2/s"
            )
            for name in names
        ),
    )


def step_definitions_by_id(species):
    return {
        definition.parameter_id: definition
        for definition in step_parameters(species)
    }


def simulation_context_species(dataframe):
    """Infer declared species order from paired axial/radial dispersion rows."""
    table = dataframe
    steps = tuple(dict.fromkeys(table.loc[table["Step"].ne(STATIC_STEP), "Step"]))
    if not steps:
        raise ValueError("SimulationContext has no per-step blocks.")

    expected = None
    for step in steps:
        ids = table.loc[table["Step"].eq(step), "Parameter ID"].tolist()
        axial = tuple(
            parameter_id.removeprefix("bed.dispersion.axial.")
            for parameter_id in ids
            if parameter_id.startswith("bed.dispersion.axial.")
        )
        radial = tuple(
            parameter_id.removeprefix("bed.dispersion.radial.")
            for parameter_id in ids
            if parameter_id.startswith("bed.dispersion.radial.")
        )
        if not axial or axial != radial or len(set(axial)) != len(axial):
            raise ValueError(
                f"Simulation-context step {step!r} needs matching, unique axial "
                "and radial dispersion rows in declared species order."
            )
        if expected is None:
            expected = axial
        elif axial != expected:
            raise ValueError(
                "Simulation-context species order differs between step blocks."
            )
    return expected


def finite_context_number(value, parameter_id):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"'{parameter_id}' must be a real number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"'{parameter_id}' must be finite.")
    return numeric


def nonempty_context_text(value, parameter_id):
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"'{parameter_id}' must be non-empty text.")
    return value.strip()


def context_definition_row(definition, value, *, step=STATIC_STEP):
    return {
        "Category": definition.category,
        "Parameter ID": definition.parameter_id,
        "Step": step,
        "Value": value,
        "Unit": definition.unit,
    }


def validate_simulation_context_dataframe(dataframe):
    """Return one normalized DataFrame conforming to the current schema."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if tuple(dataframe.columns) != SIMULATION_CONTEXT_COLUMNS:
        raise ValueError(
            "SimulationContext uses the current architecture-aware schema "
            f"with columns {SIMULATION_CONTEXT_COLUMNS}; found "
            f"{tuple(dataframe.columns)}. Legacy contexts are unsupported; "
            "regenerate the result."
        )
    if dataframe.empty:
        raise ValueError("SimulationContext DataFrame must not be empty.")

    table = dataframe.copy(deep=True).reset_index(drop=True)
    table["Category"] = table["Category"].astype(str).str.strip().str.lower()
    table["Parameter ID"] = (
        table["Parameter ID"].astype(str).str.strip().str.lower()
    )
    table["Step"] = (
        table["Step"].fillna(STATIC_STEP).astype(str).str.strip().str.lower()
    )
    table["Unit"] = table["Unit"].astype(str).str.strip()

    duplicate_mask = table.duplicated(subset=["Step", "Parameter ID"])
    if duplicate_mask.any():
        duplicates = sorted(
            set(
                zip(
                    table.loc[duplicate_mask, "Step"],
                    table.loc[duplicate_mask, "Parameter ID"],
                )
            )
        )
        raise ValueError(f"Duplicate simulation-context parameters: {duplicates}.")

    architecture_rows = table.loc[
        table["Parameter ID"].eq("bed.architecture")
        & table["Step"].eq(STATIC_STEP),
        "Value",
    ]
    if len(architecture_rows) != 1:
        raise ValueError(
            "SimulationContext must contain one static 'bed.architecture'."
        )
    architecture = nonempty_context_text(
        architecture_rows.iloc[0],
        "bed.architecture",
    ).lower()
    if architecture not in ARCHITECTURE_STATIC_BY_ID:
        raise ValueError(
            f"Unsupported saved bed architecture '{architecture}'."
        )
    table.loc[
        table["Parameter ID"].eq("bed.architecture"),
        "Value",
    ] = architecture

    static_definitions = {
        **COMMON_STATIC_BY_ID,
        **ARCHITECTURE_STATIC_BY_ID[architecture],
    }
    static_rows = table.loc[table["Step"].eq(STATIC_STEP)]
    actual_static_ids = set(static_rows["Parameter ID"])
    expected_static_ids = set(static_definitions)
    if actual_static_ids != expected_static_ids:
        raise ValueError(
            "Static SimulationContext parameters do not match the current "
            f"'{architecture}' schema; "
            f"missing={sorted(expected_static_ids - actual_static_ids)}, "
            f"unknown={sorted(actual_static_ids - expected_static_ids)}."
        )

    steps = tuple(dict.fromkeys(table.loc[table["Step"].ne(STATIC_STEP), "Step"]))
    if not steps:
        raise ValueError(
            "SimulationContext must contain at least one per-step context block."
        )
    species_names = simulation_context_species(table)
    step_by_id = step_definitions_by_id(species_names)
    expected_step_ids = set(step_by_id)
    for step in steps:
        if step not in STEP_ORDER:
            raise ValueError(f"Unsupported simulation-context step '{step}'.")
        actual_step_ids = set(
            table.loc[table["Step"].eq(step), "Parameter ID"]
        )
        if actual_step_ids != expected_step_ids:
            raise ValueError(
                f"SimulationContext parameters for step '{step}' do not match "
                "the current schema; "
                f"missing={sorted(expected_step_ids - actual_step_ids)}, "
                f"unknown={sorted(actual_step_ids - expected_step_ids)}."
            )

    for row_index, row in table.iterrows():
        parameter_id = row["Parameter ID"]
        definition = (
            static_definitions[parameter_id]
            if row["Step"] == STATIC_STEP
            else step_by_id[parameter_id]
        )
        if (
            row["Category"] != definition.category
            or row["Unit"] != definition.unit
        ):
            raise ValueError(
                f"Parameter '{parameter_id}' must use category "
                f"'{definition.category}' and unit '{definition.unit}'."
            )
        value = row["Value"]
        if definition.value_type == "text":
            normalized = nonempty_context_text(value, parameter_id)
        else:
            normalized = finite_context_number(value, parameter_id)
        table.at[row_index, "Value"] = normalized

    numeric = {
        parameter_id: float(
            static_rows.loc[
                static_rows["Parameter ID"].eq(parameter_id),
                "Value",
            ].iloc[0]
        )
        for parameter_id, definition in static_definitions.items()
        if definition.value_type == "number"
    }
    for parameter_id in ("grid.nz", "grid.nr"):
        value = numeric[parameter_id]
        if value < 1.0 or not value.is_integer():
            raise ValueError(f"'{parameter_id}' must be a positive integer.")
    for parameter_id in (
        "bed.length",
        "bed.radius",
        "bed.volume",
        "bed.aspect.ratio",
        "particle.density",
        "particle.radius",
        "bed.resolved.rho_sorbent",
    ):
        if numeric[parameter_id] <= 0.0:
            raise ValueError(f"'{parameter_id}' must be greater than zero.")
    for parameter_id in (
        "particle.porosity",
        "bed.resolved.eps_flow",
        "bed.resolved.eps_total",
    ):
        if not 0.0 <= numeric[parameter_id] < 1.0:
            raise ValueError(f"'{parameter_id}' must be in [0, 1).")
    if numeric["bed.resolved.eps_flow"] <= 0.0:
        raise ValueError("'bed.resolved.eps_flow' must be greater than zero.")
    if (
        numeric["bed.resolved.eps_total"]
        < numeric["bed.resolved.eps_flow"]
    ):
        raise ValueError(
            "'bed.resolved.eps_total' must not be smaller than "
            "'bed.resolved.eps_flow'."
        )
    if numeric["bed.resolved.acc_structure"] < 0.0:
        raise ValueError(
            "'bed.resolved.acc_structure' must be non-negative."
        )
    for parameter_id in (
        "bed.design.eps_b",
    ):
        if parameter_id in numeric and not 0.0 < numeric[parameter_id] < 1.0:
            raise ValueError(f"'{parameter_id}' must be in (0, 1).")
    positive_step_parameters = {
        "bed.resolved.lambda_ax",
        "bed.resolved.lambda_rad",
        "bed.resolved.h_wall",
        "bed.resolved.wall_bed_htc",
        "bed.resolved.effective_wall_bed_htc",
        "bed.pressure.linear_coefficient",
        "bed.pressure.inertial_coefficient_per_density",
        "bed.pressure.gas_viscosity",
        "bed.closure.temperature",
        "bed.closure.pressure",
        *(
            f"bed.dispersion.{direction}.{species}"
            for direction in ("axial", "radial")
            for species in species_names
        ),
    }
    nonnegative_step_parameters = {
        "bed.pressure.reynolds_particle",
        "bed.closure.superficial_inlet_velocity",
    }
    for step in steps:
        step_rows = table.loc[
            table["Step"].eq(step)
        ].set_index("Parameter ID")["Value"]
        for parameter_id in positive_step_parameters:
            if float(step_rows[parameter_id]) <= 0.0:
                raise ValueError(
                    f"'{parameter_id}' for step '{step}' must be "
                    "greater than zero."
                )
        for parameter_id in nonnegative_step_parameters:
            if float(step_rows[parameter_id]) < 0.0:
                raise ValueError(
                    f"'{parameter_id}' for step '{step}' must be "
                    "non-negative."
                )

    step_rank = {step: index for index, step in enumerate(STEP_ORDER)}
    static_rank = {
        parameter_id: index
        for index, parameter_id in enumerate(static_definitions)
    }
    per_step_rank = {
        parameter_id: index
        for index, parameter_id in enumerate(step_by_id)
    }
    table["_step_rank"] = table["Step"].map(
        lambda step: -1 if step == STATIC_STEP else step_rank[step]
    )
    table["_parameter_rank"] = [
        (
            static_rank[parameter_id]
            if step == STATIC_STEP
            else per_step_rank[parameter_id]
        )
        for step, parameter_id in zip(table["Step"], table["Parameter ID"])
    ]
    table = (
        table.sort_values(["_step_rank", "_parameter_rank"])
        .drop(columns=["_step_rank", "_parameter_rank"])
        .reset_index(drop=True)
    )
    table.attrs["schema_version"] = SIMULATION_CONTEXT_SCHEMA_VERSION
    return table
