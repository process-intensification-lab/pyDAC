"""Shared access and normalization helpers for post-processing contracts."""

from __future__ import annotations


STEP_KEY_TO_CYCLE_STEP = {
    "startup": "startup",
    "ads": "adsorption",
    "blow": "blowdown",
    "heat": "preheating",
    "w_purge": "wall purge",
    "s_purge": "steam purge",
    "w_des": "desorption",
    "s_des": "desorption",
    "press": "pressurization",
    "cool": "cooldown",
}
_UNSET = object()
_MISSING = object()


def normalize_metric_species(
    species,
    *,
    allow_empty=False,
    empty_message="species must not be empty.",
):
    """Normalize a metric-record species while preserving ``system``."""
    token = str(species).strip()
    if not token:
        if allow_empty:
            return ""
        raise ValueError(empty_message)
    return "system" if token.lower() == "system" else token.upper()


def normalize_species(species):
    token = str(species).strip().upper()
    if not token or token == "SYSTEM":
        raise ValueError(f"Species must be a non-empty physical name; received {species!r}.")
    return token


def normalize_step_key(step_key, *, context=None):
    """Normalize and validate one canonical cycle-step key."""
    key = str(step_key).strip().lower()
    if key not in STEP_KEY_TO_CYCLE_STEP:
        location = "" if context is None else f" {context}"
        raise ValueError(
            f"Unknown step key '{step_key}'{location}. "
            f"Valid keys: {tuple(STEP_KEY_TO_CYCLE_STEP)}."
        )
    return key


def normalize_mass_basis(basis):
    token = str(basis).strip().lower()
    aliases = {
        "integrated": "integrated",
        "int": "integrated",
        "state": "state",
        "state-based": "state",
        "preferred": "preferred",
        "best": "preferred",
    }
    if token not in aliases:
        raise ValueError("basis must be 'integrated', 'state', or 'preferred'.")
    return aliases[token]


def _normalize_record_species(species):
    token = normalize_metric_species(species)
    if token == "system":
        return "system"
    return normalize_species(token)


def _cycle_metrics_class():
    from pydac.post_processing.metrics.cycle_metrics import CycleMetrics

    return CycleMetrics


class CycleMetricsAccessor:
    """Resolve cycle steps and read typed metrics without domain calculations."""

    def __init__(self, cycle_metrics):
        if not isinstance(cycle_metrics, _cycle_metrics_class()):
            raise TypeError("cycle_metrics must be a CycleMetrics instance.")
        self._metrics = cycle_metrics

    @property
    def cycle_label(self):
        return self._metrics.cycle_label

    @property
    def step_order(self):
        return self._metrics.step_order

    def resolve_step_key(self, step):
        token = str(step).strip().lower()
        if token in STEP_KEY_TO_CYCLE_STEP:
            if token not in self.step_order:
                raise KeyError(f"Step '{token}' is not present in {self.cycle_label}.")
            return token

        matches = [
            step_key
            for step_key in self.step_order
            if STEP_KEY_TO_CYCLE_STEP[step_key] == token
        ]
        if not matches:
            raise KeyError(f"Cycle step '{step}' is not present in {self.cycle_label}.")
        if len(matches) > 1:
            raise ValueError(
                f"Cycle step '{step}' is ambiguous; matching keys: {matches}."
            )
        return matches[0]

    def value(
        self,
        *,
        step,
        species,
        metric_id,
        basis=_UNSET,
        default=_MISSING,
    ):
        selection = {
            "step_key": self.resolve_step_key(step),
            "species": _normalize_record_species(species),
            "metric_id": metric_id,
        }
        if basis is not _UNSET:
            selection["basis"] = basis
        if default is not _MISSING:
            return self._metrics.value(default=default, **selection)
        return self._metrics.value(**selection)
