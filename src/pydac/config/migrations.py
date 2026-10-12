"""Deterministic upgrades for saved experiment documents.

Schema v1 described the current DAC models with CO2/H2O-specific fields.
Schema v2 states the same physical choices through named mixture and registry
declarations.  Schema v3 drops the per-step linear-solver choice and the GMRES
state gate: every step is solved with the direct sparse factorization.  Schema
v4 adds the per-step `outlet` group (product collection and condenser
switches) and `geometry.specified_by` (which pair gives the column size); an
older case gets the values that reproduce what it computed.  Schema v5 renames
the step valve conductances `flow.*_valve_conductance_m_s` to
`flow.*_valve_conductance`: the value is in 1/m, not m/s, and is kept as is.  Schema
v6 renames `wall.wall_htc_override_W_m2K` to `wall.overall_htc_override_W_m2K`: it
replaces the 1D overall (Dixon) coefficient, not the bed-side one.  Schema v7
replaces `sorbent.heat_of_adsorption_co2_J_mol` / `_h2o_J_mol` with the
species-keyed `sorbent.heats_of_adsorption_J_mol` and makes the solid
`heat_capacity` and `fixed_adsorbates` declarations public; an older case gets
the values it computed with.  Schema v8 adds the startup composition field
`initial_solid_loadings_mol_per_kg`; an older case gets null, the equilibrium
with the initial gas it computed with.  The upgrades are intentionally small and contain
no physics.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from pydac.config.errors import ConfigurationError
from pydac.inputs.defaults.steps import INITIAL_SOLID_LOADINGS_FIELD


CURRENT_SCHEMA_VERSION = 8
LEGACY_SCHEMA_VERSION = 1
#: The last schema with `steps.<step>.solver.linear_solver` and
#: `simulation.solver.krylov_min_states`.
SOLVER_CHOICE_SCHEMA_VERSION = 2
#: The last schema without `steps.<step>.outlet`.
PRE_OUTLET_SCHEMA_VERSION = 3
#: The last schema with the valve conductances named `*_valve_conductance_m_s`.
PRE_VALVE_RENAME_SCHEMA_VERSION = 4
#: The last schema with the 1D wall override named `wall_htc_override_W_m2K`.
PRE_WALL_RENAME_SCHEMA_VERSION = 5
#: The last schema with one heat-of-adsorption field per species and no public
#: solid `heat_capacity` / `fixed_adsorbates`.
PRE_SORBENT_MAPPING_SCHEMA_VERSION = 6
#: The last schema without the startup `composition.initial_solid_loadings_mol_per_kg`.
PRE_INITIAL_LOADING_SCHEMA_VERSION = 7
SUPPORTED_SCHEMA_VERSIONS = (
    LEGACY_SCHEMA_VERSION,
    SOLVER_CHOICE_SCHEMA_VERSION,
    PRE_OUTLET_SCHEMA_VERSION,
    PRE_VALVE_RENAME_SCHEMA_VERSION,
    PRE_WALL_RENAME_SCHEMA_VERSION,
    PRE_SORBENT_MAPPING_SCHEMA_VERSION,
    PRE_INITIAL_LOADING_SCHEMA_VERSION,
    CURRENT_SCHEMA_VERSION,
)

#: Schema-v4 `flow` field names and their schema-v5 names; the values are unchanged.
V4_VALVE_FIELD_RENAMES = {
    "blowdown_valve_conductance_m_s": "blowdown_valve_conductance",
    "pressurization_valve_conductance_m_s": "pressurization_valve_conductance",
}

#: Schema-v5 `wall` field names and their schema-v6 names; the values are unchanged.
V5_WALL_FIELD_RENAMES = {
    "wall_htc_override_W_m2K": "overall_htc_override_W_m2K",
}

#: Schema-v6 per-species heat fields and the species each one becomes in v7.
V6_HEAT_FIELDS = {
    "heat_of_adsorption_co2_J_mol": "CO2",
    "heat_of_adsorption_h2o_J_mol": "H2O",
}
#: What a schema-v6 case computed with: the reference heats (for a field the
#: case did not state), the Lewatit Low2023 solid cp and no fixed loadings.
V6_HEATS_OF_ADSORPTION_J_MOL = {"CO2": -81000.0, "H2O": -46000.0}
V6_SORBENT_DECLARATIONS = {
    "heat_capacity": {
        "model": "Low2023",
        "parameters": {
            "a_J_K_per_kg": -3.23e7,
            "b_J_per_kgK2": 2.27,
            "c_J_per_kgK": -994.0,
        },
    },
    "fixed_adsorbates": {},
}
V7_SORBENT_FIELDS = ("heats_of_adsorption_J_mol", *V6_SORBENT_DECLARATIONS)

#: Schema-v2 requests that already meant the direct solve: `auto` let pyDAC
#: choose, and pyDAC now always chooses it. `gmres` is refused on upgrade.
V2_DIRECT_LINEAR_SOLVERS = ("direct", "auto")
#: What schema v2 had and v3 dropped, with the v2 reference values.
V2_STEP_SOLVER_GROUP = {"linear_solver": "direct"}
V2_SOLVER_FIELDS = {"krylov_min_states": 2500}

#: Before schema v4 the post-processing sent the outlet of these steps, and only
#: these, through the condenser at 278.15 K to the vacuum separator.
V3_COLLECTING_STEPS = frozenset({"s_purge", "w_des", "s_des"})
V3_CONDENSER_TEMPERATURE_K = 278.15


def v3_outlet_group(step: str) -> dict[str, Any]:
    """Return the schema-v4 `outlet` group that reproduces a pre-v4 step."""
    collected = step in V3_COLLECTING_STEPS
    return {
        "collect_product": collected,
        "condenser": collected,
        "condenser_temperature_K": V3_CONDENSER_TEMPERATURE_K,
    }

CURRENT_DAC_MIXTURE = {
    "species": ["CO2", "H2O", "O2", "N2"],
    "closure_species": "N2",
    "adsorbates": ["CO2", "H2O"],
}

LEGACY_ADSORPTION_DEFAULTS = {
    "isotherm_co2": "Toth",
    "isotherm_h2o": "GAB",
    "toth_data": "Cleaned-Toth",
    "kinetic_co2": "LDFRateCO2",
    "kinetic_h2o": "LDFRateH2O",
    "k_ldf_co2": 0.003,
    "k_ldf_h2o": 8.3e-4,
    "co_adsorption": True,
}

def upgrade_experiment_document(
    document: Mapping[str, Any], *, label: str = "experiment"
) -> dict[str, Any]:
    """Return a schema-v8 copy, upgrading a schema-v1 to v7 document if needed."""
    if not isinstance(document, Mapping):
        raise ConfigurationError(f"{label} must be a JSON object.")
    result = deepcopy(dict(document))
    version = result.get("schema_version")
    legacy_adsorption = _legacy_adsorption_section(result)

    if version is None and legacy_adsorption is None:
        return result
    if isinstance(version, bool) or (version is not None and not isinstance(version, int)):
        raise ConfigurationError(f"`schema_version` in {label} must be an integer.")
    if version not in (None, *SUPPORTED_SCHEMA_VERSIONS):
        raise ConfigurationError(
            f"Unsupported `schema_version` {version!r} in {label}; "
            f"supported versions are {', '.join(map(str, SUPPORTED_SCHEMA_VERSIONS))}."
        )
    if version == CURRENT_SCHEMA_VERSION:
        return result

    simulation = result.setdefault("simulation", {})
    if not isinstance(simulation, dict):
        raise ConfigurationError(f"`simulation` in {label} must be a JSON object.")
    if version not in (
        PRE_VALVE_RENAME_SCHEMA_VERSION,
        PRE_WALL_RENAME_SCHEMA_VERSION,
        PRE_SORBENT_MAPPING_SCHEMA_VERSION,
        PRE_INITIAL_LOADING_SCHEMA_VERSION,
    ):
        if version == SOLVER_CHOICE_SCHEMA_VERSION:
            _drop_v2_solver_choice(simulation, label)
        elif version != PRE_OUTLET_SCHEMA_VERSION:
            simulation.setdefault("mixture", deepcopy(CURRENT_DAC_MIXTURE))
            if legacy_adsorption is not None:
                simulation["adsorption"] = _upgrade_legacy_adsorption(legacy_adsorption)
        _add_v3_outlet_groups(simulation, label)
        _add_v3_geometry_input(simulation, label)
    if version not in (
        PRE_WALL_RENAME_SCHEMA_VERSION,
        PRE_SORBENT_MAPPING_SCHEMA_VERSION,
        PRE_INITIAL_LOADING_SCHEMA_VERSION,
    ):
        _rename_v4_valve_fields(simulation, label)
    if version not in (PRE_SORBENT_MAPPING_SCHEMA_VERSION, PRE_INITIAL_LOADING_SCHEMA_VERSION):
        _rename_v5_wall_fields(simulation, label)
    if version != PRE_INITIAL_LOADING_SCHEMA_VERSION:
        _upgrade_v6_sorbent_fields(simulation, label)
    _add_v7_initial_loadings(simulation, label)
    if version is not None:
        result["schema_version"] = CURRENT_SCHEMA_VERSION
    return result


def _step_mapping(simulation: Any) -> Any:
    """Return the step mapping of a document section, or None if absent."""
    if not isinstance(simulation, Mapping):
        return None
    cycle = simulation.get("cycle")
    if not isinstance(cycle, Mapping):
        return None
    steps = cycle.get("steps")
    return steps if isinstance(steps, Mapping) else None


def _drop_v2_solver_choice(simulation: dict[str, Any], label: str) -> None:
    """Remove the schema-v2 linear-solver fields, refusing a GMRES request.

    Dropping `direct` or `auto` changes nothing a run computes. A step that
    asked for GMRES would silently get another solver, so it is refused.
    """
    steps = _step_mapping(simulation)
    for name, step in (steps or {}).items():
        if not isinstance(step, dict) or "solver" not in step:
            continue
        path = f"simulation.cycle.steps.{name}.solver"
        group = step.pop("solver")
        if not isinstance(group, Mapping):
            raise ConfigurationError(f"`{path}` in {label} must be a JSON object.")
        unknown = sorted(set(group) - set(V2_STEP_SOLVER_GROUP))
        if unknown:
            raise ConfigurationError(f"Unknown schema-v2 field `{path}.{unknown[0]}` in {label}.")
        request = group.get("linear_solver", "direct")
        if str(request).strip().lower() not in V2_DIRECT_LINEAR_SOLVERS:
            raise ConfigurationError(
                f"`{path}.linear_solver` is {request!r} in {label}, but pyDAC solves "
                "every step with the direct sparse solver since case schema "
                f"{SOLVER_CHOICE_SCHEMA_VERSION + 1}; GMRES was removed. Set it to 'direct' to "
                "run this case with the direct solver."
            )
    solver = simulation.get("solver")
    if isinstance(solver, dict):
        for field in V2_SOLVER_FIELDS:
            solver.pop(field, None)


def _add_v3_outlet_groups(simulation: dict[str, Any], label: str) -> None:
    """Give every step of a pre-v4 document the outlet group it computed with."""
    steps = _step_mapping(simulation)
    for name, step in (steps or {}).items():
        if not isinstance(step, dict):
            continue
        if "outlet" in step:
            raise ConfigurationError(
                f"`simulation.cycle.steps.{name}.outlet` in {label} exists only "
                f"since case schema {PRE_OUTLET_SCHEMA_VERSION + 1}; set `schema_version` "
                f"to {PRE_OUTLET_SCHEMA_VERSION + 1}."
            )
        step["outlet"] = v3_outlet_group(name)


def _rename_v4_valve_fields(simulation: dict[str, Any], label: str) -> None:
    """Give the valve conductances of a pre-v5 document their schema-v5 names."""
    steps = _step_mapping(simulation)
    for name, step in (steps or {}).items():
        flow = step.get("flow") if isinstance(step, dict) else None
        if not isinstance(flow, dict):
            continue
        for old, new in V4_VALVE_FIELD_RENAMES.items():
            if new in flow:
                raise ConfigurationError(
                    f"`simulation.cycle.steps.{name}.flow.{new}` in {label} exists only "
                    f"since case schema {PRE_VALVE_RENAME_SCHEMA_VERSION + 1}; set "
                    f"`schema_version` to {PRE_VALVE_RENAME_SCHEMA_VERSION + 1}."
                )
            if old in flow:
                flow[new] = flow.pop(old)


def _rename_v5_wall_fields(simulation: dict[str, Any], label: str) -> None:
    """Give the 1D wall override of a pre-v6 document its schema-v6 name."""
    wall = simulation.get("wall")
    if not isinstance(wall, dict):
        return
    for old, new in V5_WALL_FIELD_RENAMES.items():
        if new in wall:
            raise ConfigurationError(
                f"`simulation.wall.{new}` in {label} exists only since case schema "
                f"{PRE_WALL_RENAME_SCHEMA_VERSION + 1}; set `schema_version` to "
                f"{PRE_WALL_RENAME_SCHEMA_VERSION + 1}."
            )
        if old in wall:
            wall[new] = wall.pop(old)


def _upgrade_v6_sorbent_fields(simulation: dict[str, Any], label: str) -> None:
    """Give the sorbent of a pre-v7 document its schema-v7 declarations.

    The per-species heats become one species-keyed mapping (a species the
    document did not state keeps its reference value, as the profile gave it
    before); the solid cp and fixed loadings it computed with are written out.
    """
    sorbent = simulation.get("sorbent")
    if not isinstance(sorbent, dict):
        return
    for new in V7_SORBENT_FIELDS:
        if new in sorbent:
            raise ConfigurationError(
                f"`simulation.sorbent.{new}` in {label} exists only since case schema "
                f"{PRE_SORBENT_MAPPING_SCHEMA_VERSION + 1}; set `schema_version` to "
                f"{PRE_SORBENT_MAPPING_SCHEMA_VERSION + 1}."
            )
    stated = {
        species: sorbent.pop(old)
        for old, species in V6_HEAT_FIELDS.items()
        if old in sorbent
    }
    if not stated:
        return
    sorbent["heats_of_adsorption_J_mol"] = {**V6_HEATS_OF_ADSORPTION_J_MOL, **stated}
    for name, declaration in V6_SORBENT_DECLARATIONS.items():
        sorbent[name] = deepcopy(declaration)


def refuse_initial_loadings_before_v8(document: Any, label: str) -> None:
    """Refuse the schema-8 initial solid loadings in a document of schema 1 to 7.

    Called before an old document's shape is checked, so that the error names
    the schema version rather than an unknown field.
    """
    if not isinstance(document, Mapping):
        return
    version = document.get("schema_version")
    if isinstance(version, bool) or version not in SUPPORTED_SCHEMA_VERSIONS:
        return
    if version == CURRENT_SCHEMA_VERSION:
        return
    for name, step in (_step_mapping(document.get("simulation")) or {}).items():
        composition = step.get("composition") if isinstance(step, Mapping) else None
        if isinstance(composition, Mapping) and INITIAL_SOLID_LOADINGS_FIELD in composition:
            raise ConfigurationError(
                f"`simulation.cycle.steps.{name}.composition.{INITIAL_SOLID_LOADINGS_FIELD}` "
                f"in {label} exists only since case schema "
                f"{PRE_INITIAL_LOADING_SCHEMA_VERSION + 1}; set `schema_version` to "
                f"{PRE_INITIAL_LOADING_SCHEMA_VERSION + 1}."
            )


def _add_v7_initial_loadings(simulation: dict[str, Any], label: str) -> None:
    """Give every stated composition of a pre-v8 document a null initial loading.

    Null is what such a case computed with: the startup solid in equilibrium
    with the initial gas.
    """
    steps = _step_mapping(simulation)
    for name, step in (steps or {}).items():
        composition = step.get("composition") if isinstance(step, dict) else None
        if not isinstance(composition, dict):
            continue
        if INITIAL_SOLID_LOADINGS_FIELD in composition:
            raise ConfigurationError(
                f"`simulation.cycle.steps.{name}.composition.{INITIAL_SOLID_LOADINGS_FIELD}` "
                f"in {label} exists only since case schema "
                f"{PRE_INITIAL_LOADING_SCHEMA_VERSION + 1}; set `schema_version` to "
                f"{PRE_INITIAL_LOADING_SCHEMA_VERSION + 1}."
            )
        composition[INITIAL_SOLID_LOADINGS_FIELD] = None


#: Before schema v4 the column size was always given by the volume and the aspect ratio.
V3_GEOMETRY_SPECIFIED_BY = "volume_aspect_ratio"


def _add_v3_geometry_input(simulation: dict[str, Any], label: str) -> None:
    """Name the input pair of a pre-v4 geometry: the volume and the aspect ratio."""
    geometry = simulation.get("geometry")
    if not isinstance(geometry, dict):
        return
    if "specified_by" in geometry:
        raise ConfigurationError(
            f"`simulation.geometry.specified_by` in {label} exists only since case schema "
            f"{PRE_OUTLET_SCHEMA_VERSION + 1}; set `schema_version` to "
            f"{PRE_OUTLET_SCHEMA_VERSION + 1}."
        )
    geometry["specified_by"] = V3_GEOMETRY_SPECIFIED_BY


def pre_initial_loading_v7_vocabulary(
    current_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the schema-v7 complete-case shape for structural validation."""
    vocabulary = deepcopy(dict(current_profile))
    vocabulary["schema_version"] = PRE_INITIAL_LOADING_SCHEMA_VERSION
    for step in (_step_mapping(vocabulary.get("simulation")) or {}).values():
        composition = step.get("composition") if isinstance(step, dict) else None
        if isinstance(composition, dict):
            composition.pop(INITIAL_SOLID_LOADINGS_FIELD, None)
    return vocabulary


def pre_sorbent_mapping_v6_vocabulary(
    current_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the schema-v6 complete-case shape for structural validation."""
    vocabulary = pre_initial_loading_v7_vocabulary(current_profile)
    vocabulary["schema_version"] = PRE_SORBENT_MAPPING_SCHEMA_VERSION
    sorbent = (vocabulary.get("simulation") or {}).get("sorbent")
    if isinstance(sorbent, dict):
        for new in V7_SORBENT_FIELDS:
            sorbent.pop(new, None)
        for old, species in V6_HEAT_FIELDS.items():
            sorbent[old] = V6_HEATS_OF_ADSORPTION_J_MOL[species]
    return vocabulary


def pre_wall_rename_v5_vocabulary(current_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema-v5 complete-case shape for structural validation."""
    vocabulary = pre_sorbent_mapping_v6_vocabulary(current_profile)
    vocabulary["schema_version"] = PRE_WALL_RENAME_SCHEMA_VERSION
    wall = (vocabulary.get("simulation") or {}).get("wall")
    if isinstance(wall, dict):
        for old, new in V5_WALL_FIELD_RENAMES.items():
            if new in wall:
                wall[old] = wall.pop(new)
    return vocabulary


def pre_valve_rename_v4_vocabulary(current_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema-v4 complete-case shape for structural validation."""
    vocabulary = pre_wall_rename_v5_vocabulary(current_profile)
    vocabulary["schema_version"] = PRE_VALVE_RENAME_SCHEMA_VERSION
    for step in (_step_mapping(vocabulary.get("simulation")) or {}).values():
        flow = step.get("flow") if isinstance(step, dict) else None
        if isinstance(flow, dict):
            for old, new in V4_VALVE_FIELD_RENAMES.items():
                if new in flow:
                    flow[old] = flow.pop(new)
    return vocabulary


def pre_outlet_v3_vocabulary(current_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema-v3 complete-case shape for structural validation."""
    vocabulary = pre_valve_rename_v4_vocabulary(current_profile)
    vocabulary["schema_version"] = PRE_OUTLET_SCHEMA_VERSION
    for step in (_step_mapping(vocabulary.get("simulation")) or {}).values():
        if isinstance(step, dict):
            step.pop("outlet", None)
    geometry = (vocabulary.get("simulation") or {}).get("geometry")
    if isinstance(geometry, dict):
        geometry.pop("specified_by", None)
    return vocabulary


def legacy_v1_vocabulary(current_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the old complete-case shape for structural v1 validation."""
    vocabulary = pre_outlet_v3_vocabulary(current_profile)
    vocabulary["schema_version"] = LEGACY_SCHEMA_VERSION
    simulation = vocabulary["simulation"]
    simulation.pop("mixture", None)
    simulation["adsorption"] = deepcopy(LEGACY_ADSORPTION_DEFAULTS)
    return vocabulary


def solver_choice_v2_vocabulary(current_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema-v2 complete-case shape for structural validation."""
    vocabulary = pre_outlet_v3_vocabulary(current_profile)
    vocabulary["schema_version"] = SOLVER_CHOICE_SCHEMA_VERSION
    simulation = vocabulary["simulation"]
    for step in (_step_mapping(simulation) or {}).values():
        if isinstance(step, dict):
            step["solver"] = deepcopy(V2_STEP_SOLVER_GROUP)
    if isinstance(simulation.get("solver"), dict):
        simulation["solver"].update(V2_SOLVER_FIELDS)
    return vocabulary


def _legacy_adsorption_section(document: Mapping[str, Any]):
    simulation = document.get("simulation")
    if not isinstance(simulation, Mapping):
        return None
    adsorption = simulation.get("adsorption")
    if not isinstance(adsorption, Mapping):
        return None
    if set(adsorption) & set(LEGACY_ADSORPTION_DEFAULTS):
        return adsorption
    return None


def _upgrade_legacy_adsorption(values: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(values) - set(LEGACY_ADSORPTION_DEFAULTS))
    if unknown:
        raise ConfigurationError(
            f"Unknown schema-v1 adsorption field `simulation.adsorption.{unknown[0]}`."
        )
    legacy = {**LEGACY_ADSORPTION_DEFAULTS, **dict(values)}
    equilibrium = _legacy_equilibrium_systems(legacy)
    kinetics = {
        species: {
            "model": _legacy_kinetic_name(species, legacy[f"kinetic_{species.lower()}"]),
            "parameters": {
                "k_LDF_s_inv": legacy[f"k_ldf_{species.lower()}"]
            },
        }
        for species in ("CO2", "H2O")
    }
    return {"equilibrium_systems": equilibrium, "kinetics": kinetics}


def _legacy_equilibrium_systems(values: Mapping[str, Any]) -> dict[str, Any]:
    co2 = values["isotherm_co2"]
    h2o = values["isotherm_h2o"]
    dataset = values["toth_data"]
    if values["co_adsorption"]:
        if (co2, h2o) != ("Toth", "GAB"):
            raise ConfigurationError(
                "Schema-v1 co-adsorption requires isotherm_co2='Toth' and "
                "isotherm_h2o='GAB'."
            )
        return {
            "TothGABWADST": {
                "datasets": {"Toth": dataset},
                "parameters": {},
            }
        }

    systems = {}
    for model in (co2, h2o):
        if model in systems:
            raise ConfigurationError(
                f"Schema-v1 independent isotherm {model!r} is selected twice."
            )
        systems[model] = {
            "datasets": {"Toth": dataset} if model == "Toth" else {},
            "parameters": {},
        }
    return systems


def _legacy_kinetic_name(species: str, name: Any) -> str:
    accepted = {"LDFRate", f"LDFRate{species}"}
    if name not in accepted:
        raise ConfigurationError(
            f"Schema-v1 kinetic model {name!r} is not valid for {species}; "
            f"expected one of {sorted(accepted)}."
        )
    return "LDF"


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "LEGACY_SCHEMA_VERSION",
    "PRE_OUTLET_SCHEMA_VERSION",
    "PRE_SORBENT_MAPPING_SCHEMA_VERSION",
    "PRE_VALVE_RENAME_SCHEMA_VERSION",
    "PRE_WALL_RENAME_SCHEMA_VERSION",
    "SOLVER_CHOICE_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "V3_COLLECTING_STEPS",
    "V4_VALVE_FIELD_RENAMES",
    "V5_WALL_FIELD_RENAMES",
    "V6_HEAT_FIELDS",
    "V6_HEATS_OF_ADSORPTION_J_MOL",
    "V6_SORBENT_DECLARATIONS",
    "V7_SORBENT_FIELDS",
    "legacy_v1_vocabulary",
    "pre_outlet_v3_vocabulary",
    "pre_sorbent_mapping_v6_vocabulary",
    "pre_valve_rename_v4_vocabulary",
    "pre_wall_rename_v5_vocabulary",
    "solver_choice_v2_vocabulary",
    "upgrade_experiment_document",
    "v3_outlet_group",
]
