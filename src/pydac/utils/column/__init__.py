"""Stable facade for the column-model plumbing kept out of the physical core.

`src/pydac/core/column_model.py` owns the physics. What lives here is everything that is
*not* physics: input validation and normalization, and the structured runtime
property record plus its terminal summary.
"""

from pydac.utils.column.runtime_properties import (
    RUNTIME_PROPERTIES_SCHEMA_VERSION,
    build_runtime_properties,
    print_runtime_property_summary,
)
from pydac.utils.column.validation import (
    REQUIRED_FIXED_ADSORBATE_FIELDS,
    SOLID_HEAT_CAPACITY_MODELS,
    require_water_component,
    resolve_fixed_adsorbates,
    resolve_heats_of_adsorption,
    resolve_initial_solid_loadings,
    resolve_solid_heat_capacity,
    validate_architecture_support,
    validate_flat_vector,
    validate_warm_start_fields,
)

__all__ = [
    "RUNTIME_PROPERTIES_SCHEMA_VERSION",
    "REQUIRED_FIXED_ADSORBATE_FIELDS",
    "SOLID_HEAT_CAPACITY_MODELS",
    "build_runtime_properties",
    "print_runtime_property_summary",
    "require_water_component",
    "resolve_fixed_adsorbates",
    "resolve_heats_of_adsorption",
    "resolve_initial_solid_loadings",
    "resolve_solid_heat_capacity",
    "validate_architecture_support",
    "validate_flat_vector",
    "validate_warm_start_fields",
]
