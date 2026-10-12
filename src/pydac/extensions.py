"""Use an equilibrium or kinetic model of your own without editing pyDAC.

A registered model is selected by its name in a case, like a built-in one:
``adsorption.equilibrium_systems.<name>`` or ``adsorption.kinetics.<species>.model``.
It lives in the Python session that registered it: register it again before a
``resume`` or ``report`` in a new session. NSGA-II workers started with the
``spawn`` method (the default on macOS and Windows) are new sessions and do not
see it. The case names the model, not its code: keep the code with the results.
"""

# ── MAP ────────────────────────────────────────────────────────────────────
# Pipeline : before stage 3 — the registries that model_validation and
#            ColumnModelFactory read.
# Owns     : run-time registration and its refusals.
# Not here : the built-in models (core/isotherm_models.py, core/kinetic_models.py).
# Guide    : docs/guide/08_extending.md
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from pydac.config.errors import ConfigurationError
from pydac.core import isotherm_models, kinetic_models
from pydac.utils.model_registry import EquilibriumRegistryEntry, KineticRegistryEntry


def register_equilibrium(name: str, entry: EquilibriumRegistryEntry) -> None:
    """Make the equilibrium system ``entry`` selectable as ``name`` in this session.

    Registering a name again replaces it; a built-in name is refused.
    """
    _register(name, entry, EquilibriumRegistryEntry, isotherm_models.EQUILIBRIUM_REGISTRY,
              isotherm_models._REGISTERED_EQUILIBRIA, "equilibrium system")


def register_kinetics(name: str, entry: KineticRegistryEntry) -> None:
    """Make the kinetic law ``entry`` selectable as ``name`` in this session.

    Registering a name again replaces it; a built-in name is refused.
    """
    _register(name, entry, KineticRegistryEntry, kinetic_models.KINETIC_REGISTRY,
              kinetic_models._REGISTERED_KINETICS, "kinetic law")


def _register(name, entry, entry_type, built_in, registered, label) -> None:
    if not isinstance(name, str) or not name or name != name.strip():
        raise ConfigurationError(f"A {label} name must be a nonempty string, got {name!r}.")
    if name in built_in:
        raise ConfigurationError(
            f"{name!r} is a built-in {label}; register yours under another name."
        )
    if not isinstance(entry, entry_type):
        raise ConfigurationError(
            f"A {label} is registered as a pydac.{entry_type.__name__}, got {type(entry).__name__}."
        )
    if not callable(entry.build):
        raise ConfigurationError(f"The `build` of {label} {name!r} must be callable.")
    registered[name] = entry


__all__ = ["register_equilibrium", "register_kinetics"]
