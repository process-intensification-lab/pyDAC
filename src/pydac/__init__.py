"""pyDAC: simulation of adsorption-based direct air capture processes.

The supported Python API (the same layer the ``pydac`` command uses)::

    import pydac

    case = pydac.load_case("my_case.json", settings={"execution.max_cycles": 30})
    result = pydac.run_css(case)            # RunResult: folder, css_reached, cycle, ...
    pydac.report(result.folder, fmt="png")
    kpis = pydac.CycleKPIs(pydac.CycleDataLoader(result.folder).load())

Names are imported on first use, so ``import pydac`` stays light: numpy,
matplotlib and pymoo load only when a name that needs them is used.
"""

from __future__ import annotations

import importlib

from pydac.utils.version import __version__

_EXPORTS = {
    "load_case": "pydac.api",
    "export_case": "pydac.api",
    "export_settings": "pydac.api",
    "run_steps": "pydac.api",
    "run_css": "pydac.api",
    "run_nsga2": "pydac.api",
    "resume": "pydac.api",
    "report": "pydac.api",
    "RunResult": "pydac.api",
    "ConfigurationError": "pydac.config.errors",
    "RunRefused": "pydac.config.errors",
    "ExperimentConfig": "pydac.config.schema",
    "CycleDataLoader": "pydac.post_processing.metrics.cycle_data_loader",
    "CycleKPIs": "pydac.post_processing.kpi.cycle_kpis",
    "register_equilibrium": "pydac.extensions",
    "register_kinetics": "pydac.extensions",
    "get_started": "pydac.starter_kit",
    "EquilibriumSystem": "pydac.core.isotherm_models",
    "KineticLaw": "pydac.core.kinetic_models",
    "EquilibriumRegistryEntry": "pydac.utils.model_registry",
    "KineticRegistryEntry": "pydac.utils.model_registry",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'pydac' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
