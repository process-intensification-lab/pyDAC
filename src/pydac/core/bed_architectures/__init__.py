"""Public detached API for bed-architecture configuration and contracts.

The registry in this module describes the names and spatial scope exposed by
the public run APIs. It does not construct or import the operational column
solver; the packed-bed implementation remains a detached pure-physics owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from pydac.core.bed_architectures.base import (
    ARCHITECTURE_NAMES,
    CHANNELING_FLOW_MODELS,
    CHANNELING_INLET_DISTRIBUTIONS,
    CHANNELING_POROSITY_MODELS,
    DEFAULT_CHANNELING_INLET_DISTRIBUTION,
    DEFAULT_CHANNELING_POROSITY_MODEL,
    DEFAULT_DRAG_LAYER_WIDTH_DP,
    DEFAULT_FLOW_MODEL,
    DEFAULT_WALL_HTC_MODEL,
    DRAG_LAYER_WIDTH_DP_BOUNDS,
    FLOW_MODELS,
    WALL_HTC_MODELS,
    BedArchitecture,
    ColumnGeometry,
    DispersionModel,
    FlowProperties,
    GasProperties,
    HeatTransferModel,
    ChannelingOptions,
    PackedBedDesign,
    ParticleProperties,
    PressureDropLaw,
    ResolvedArchitecture,
    ResolvedStructure,
    ResolvedTransport,
    StructureContext,
    TransportContext,
    build_bed_config,
    resolve_bed_config,
    resolve_channeling_options,
    validate_architecture_name,
    validate_channeling_porosity_model,
    validate_flow_model,
    validate_spatial_mode,
    validate_stagnant_conductivity_model,
    validate_wall_htc_model,
)
from pydac.core.bed_architectures.channeling import (
    CHANNELING_GENERATOR_REVISION,
    ChannelingStructure,
    DeKlerkRadialPorosity,
    GieseRadialPorosity,
    TabulatedRadialPorosity,
    channeling_inlet_flux_weights,
    drag_layer_faces,
    ergun_drag_coefficients,
    project_layer_field,
    resolve_channeling_structure,
    resolve_radial_porosity_profile,
    validate_channeling_support,
)
from pydac.core.bed_architectures.packed_bed import (
    ErgunPressureDrop,
    RadialErgunPressureDrop,
    axial_effective_conductivity,
    dixon_overall_coefficient,
    packed_bed_ring_inlet_flux_weights,
    packed_bed_ring_structure_factors,
    radial_effective_conductivity,
    resolve_packed_bed_dispersion,
    resolve_packed_bed_pressure_drop,
    resolve_packed_bed_radial_pressure_drop,
    resolve_packed_bed_structure,
    resolve_packed_bed_thermal_transport,
    solve_quadratic_drag_velocity,
    specchia_stagnant_conductivity,
    specchia_wall_heat_transfer_coefficient,
    WALL_HTC_LAWS,
    wall_heat_transfer_coefficient,
    zehner_schlunder_stagnant_conductivity,
)


@dataclass(frozen=True)
class ArchitectureSpec:
    """Stable registry metadata, independent of implementation objects."""

    name: str
    design_type: str
    spatial_modes: tuple[str, ...]

    def to_context_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "design_type": self.design_type,
            "spatial_modes": list(self.spatial_modes),
        }


ARCHITECTURE_REGISTRY: Mapping[str, ArchitectureSpec] = MappingProxyType(
    {
        "packed_bed": ArchitectureSpec(
            name="packed_bed",
            design_type="PackedBedDesign",
            spatial_modes=("1d", "2d"),
        ),
    }
)


def registered_architectures() -> tuple[str, ...]:
    """Return registered architecture names in deterministic order."""
    return tuple(sorted(ARCHITECTURE_REGISTRY))


def get_architecture_spec(name: str) -> ArchitectureSpec:
    """Return detached metadata for a registered architecture."""
    name = validate_architecture_name(name)
    return ARCHITECTURE_REGISTRY[name]


__all__ = [
    "ARCHITECTURE_NAMES",
    "ARCHITECTURE_REGISTRY",
    "CHANNELING_FLOW_MODELS",
    "CHANNELING_GENERATOR_REVISION",
    "CHANNELING_INLET_DISTRIBUTIONS",
    "CHANNELING_POROSITY_MODELS",
    "DEFAULT_CHANNELING_INLET_DISTRIBUTION",
    "DEFAULT_CHANNELING_POROSITY_MODEL",
    "DEFAULT_DRAG_LAYER_WIDTH_DP",
    "DEFAULT_FLOW_MODEL",
    "DRAG_LAYER_WIDTH_DP_BOUNDS",
    "FLOW_MODELS",
    "DEFAULT_WALL_HTC_MODEL",
    "WALL_HTC_MODELS",
    "ArchitectureSpec",
    "BedArchitecture",
    "ChannelingOptions",
    "ChannelingStructure",
    "ColumnGeometry",
    "DeKlerkRadialPorosity",
    "GieseRadialPorosity",
    "DispersionModel",
    "ErgunPressureDrop",
    "RadialErgunPressureDrop",
    "TabulatedRadialPorosity",
    "FlowProperties",
    "GasProperties",
    "HeatTransferModel",
    "PackedBedDesign",
    "ParticleProperties",
    "PressureDropLaw",
    "ResolvedArchitecture",
    "ResolvedStructure",
    "ResolvedTransport",
    "StructureContext",
    "TransportContext",
    "build_bed_config",
    "channeling_inlet_flux_weights",
    "drag_layer_faces",
    "ergun_drag_coefficients",
    "project_layer_field",
    "resolve_bed_config",
    "resolve_channeling_options",
    "resolve_channeling_structure",
    "resolve_radial_porosity_profile",
    "axial_effective_conductivity",
    "dixon_overall_coefficient",
    "get_architecture_spec",
    "registered_architectures",
    "packed_bed_ring_inlet_flux_weights",
    "packed_bed_ring_structure_factors",
    "radial_effective_conductivity",
    "resolve_packed_bed_dispersion",
    "resolve_packed_bed_pressure_drop",
    "resolve_packed_bed_radial_pressure_drop",
    "resolve_packed_bed_structure",
    "resolve_packed_bed_thermal_transport",
    "solve_quadratic_drag_velocity",
    "validate_architecture_name",
    "validate_channeling_porosity_model",
    "validate_channeling_support",
    "validate_flow_model",
    "validate_spatial_mode",
    "specchia_stagnant_conductivity",
    "specchia_wall_heat_transfer_coefficient",
    "validate_stagnant_conductivity_model",
    "validate_wall_htc_model",
    "WALL_HTC_LAWS",
    "wall_heat_transfer_coefficient",
    "zehner_schlunder_stagnant_conductivity",
]
