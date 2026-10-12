"""Domain-oriented default mappings for everyday pyDAC input editing."""

from pydac.inputs.defaults.adsorption import adsorption_defaults
from pydac.inputs.defaults.column import bed_defaults, column_defaults, geometry_defaults, wall_defaults
from pydac.inputs.defaults.execution import execution_defaults
from pydac.inputs.defaults.mixture import mixture_defaults
from pydac.inputs.defaults.particle import sorbent_defaults
from pydac.inputs.defaults.profile import default_profile, default_rules
from pydac.inputs.defaults.reference import reference_defaults
from pydac.inputs.defaults.solver import grid_defaults, solver_defaults
from pydac.inputs.defaults.steps import cycle_defaults

__all__ = [
    "adsorption_defaults",
    "bed_defaults",
    "column_defaults",
    "cycle_defaults",
    "default_profile",
    "default_rules",
    "execution_defaults",
    "geometry_defaults",
    "grid_defaults",
    "mixture_defaults",
    "reference_defaults",
    "solver_defaults",
    "sorbent_defaults",
    "wall_defaults",
]
