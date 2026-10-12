"""Filesystem adapter that assembles cycle metrics and simulation context."""

from __future__ import annotations

import json
from pathlib import Path
import re

from pydac.post_processing.context.store import SimulationContextStore
from pydac.utils.postprocessing.normalization import normalize_step_key

from pydac.post_processing.metrics.cycle_data import CycleData
from pydac.post_processing.metrics.cycle_metrics import STEP_KEY_TO_CYCLE_STEP, CycleMetrics
from pydac.post_processing.metrics.step_metrics_store import StepMetricsStore


_CYCLE_FOLDER_PATTERN = re.compile(r"^cycle_(\d+)$")
_SPATIAL_MODES = ("1D_mode", "2D_mode")
# Fallback ordering for a result folder that carries no `cycle_metadata.json`, i.e.
# a step-mode run.  It must be *execution* order, because anything that reads "the
# last product step" or "the end of regeneration" from it would otherwise read the
# bed state before the steam pulse rather than after it.
#
# `w_des` before `s_purge` before `s_des` is the one linear order that serves both
# legal desorption pairings: `w_des` + `s_purge` runs the pulse last, while the
# historical `s_purge` + `s_des` runs the purge first.  (`w_des` + `s_des` together
# is rejected by `_step_order_from_folders`.)
_DEFAULT_STEP_KEY_ORDER = (
    "startup",
    "ads",
    "blow",
    "heat",
    "w_purge",
    "w_des",
    "s_purge",
    "s_des",
    "cool",
    "press",
)


def _normalize_spatial_mode(spatial_mode):
    if spatial_mode is None:
        return None
    token = str(spatial_mode).strip().lower()
    aliases = {
        "1d": "1D_mode",
        "1d_mode": "1D_mode",
        "2d": "2D_mode",
        "2d_mode": "2D_mode",
    }
    if token not in aliases:
        raise ValueError("spatial_mode must be '1D_mode', '2D_mode', or None.")
    return aliases[token]


class CycleDataLoader:
    """Load one selected cycle without accessing solver-result files."""

    def __init__(self, base_folder, *, cycle_index=None, spatial_mode=None):
        self.base_folder = Path(base_folder)
        self.cycle_index = cycle_index
        self.spatial_mode = _normalize_spatial_mode(spatial_mode)

        if not self.base_folder.is_dir():
            raise FileNotFoundError(
                f"Cycle data base folder does not exist: '{self.base_folder}'."
            )

        self.cycle_folder, self.metrics_folder = self._resolve_selected_folders()

    @staticmethod
    def _cycle_index_from_path(path):
        match = _CYCLE_FOLDER_PATTERN.fullmatch(path.name)
        return None if match is None else int(match.group(1))

    def _resolve_spatial_folder(self, folder):
        available = [mode for mode in _SPATIAL_MODES if (folder / mode).is_dir()]
        if self.spatial_mode is not None:
            selected = folder / self.spatial_mode
            if available and not selected.is_dir():
                raise FileNotFoundError(
                    f"Requested spatial mode '{self.spatial_mode}' not found in "
                    f"'{folder}'. Available: {available}."
                )
            return selected if selected.is_dir() else folder

        if len(available) > 1:
            raise ValueError(
                f"Multiple spatial-mode folders found in '{folder}': {available}. "
                "Select one with spatial_mode."
            )
        return folder / available[0] if available else folder

    @staticmethod
    def _available_cycle_folders(folder):
        cycles = []
        for path in folder.iterdir():
            if not path.is_dir():
                continue
            match = _CYCLE_FOLDER_PATTERN.fullmatch(path.name)
            if match is not None:
                cycles.append((int(match.group(1)), path))
        return sorted(cycles, key=lambda item: item[0])

    def _select_cycle_folder(self, search_root):
        cycle_folders = self._available_cycle_folders(search_root)
        if not cycle_folders:
            if self.cycle_index is not None:
                raise ValueError(
                    f"No cycle_n folders found in '{search_root}', but "
                    f"cycle_index={self.cycle_index} was requested."
                )
            return None

        if self.cycle_index is None:
            return cycle_folders[-1][1]

        requested = int(self.cycle_index)
        if requested < 0:
            try:
                return cycle_folders[requested][1]
            except IndexError as exc:
                available = [index for index, _ in cycle_folders]
                raise ValueError(
                    f"Relative cycle index {requested} is out of range; "
                    f"available cycles: {available}."
                ) from exc

        matches = [path for index, path in cycle_folders if index == requested]
        if not matches:
            available = [index for index, _ in cycle_folders]
            raise ValueError(
                f"Requested cycle_{requested} not found in '{search_root}'. "
                f"Available cycles: {available}."
            )
        return matches[0]

    def _resolve_selected_folders(self):
        direct_cycle_index = self._cycle_index_from_path(self.base_folder)
        if direct_cycle_index is not None:
            if (
                self.cycle_index is not None
                and int(self.cycle_index) != direct_cycle_index
            ):
                raise ValueError(
                    f"Base folder is cycle_{direct_cycle_index}, but "
                    f"cycle_index={self.cycle_index} was requested."
                )
            metrics_folder = self._resolve_spatial_folder(self.base_folder)
            return self.base_folder, metrics_folder

        search_root = self._resolve_spatial_folder(self.base_folder)
        selected_cycle = self._select_cycle_folder(search_root)
        if selected_cycle is None:
            return None, search_root

        metrics_folder = self._resolve_spatial_folder(selected_cycle)
        return selected_cycle, metrics_folder

    @property
    def cycle_label(self):
        return self.cycle_folder.name if self.cycle_folder is not None else "selected_case"

    def _metadata_path(self):
        candidates = []
        if self.cycle_folder is not None:
            candidates.append(self.cycle_folder / "cycle_metadata.json")
        candidates.append(self.metrics_folder / "cycle_metadata.json")
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _step_order_from_metadata(self):
        metadata_path = self._metadata_path()
        if metadata_path is None:
            return None

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        raw_steps = metadata.get("executed_steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(
                f"'{metadata_path}' must contain a non-empty executed_steps list."
            )

        step_order = tuple(
            normalize_step_key(step, context="in cycle metadata")
            for step in raw_steps
        )
        if len(set(step_order)) != len(step_order):
            raise ValueError(
                f"'{metadata_path}' contains duplicate executed step keys: {step_order}."
            )
        return step_order

    def _step_order_from_folders(self):
        available = {
            step_key
            for step_key in STEP_KEY_TO_CYCLE_STEP
            if (self.metrics_folder / step_key).is_dir()
        }
        if "w_des" in available and "s_des" in available:
            raise ValueError(
                f"Both desorption variants are present in '{self.metrics_folder}'."
            )
        return tuple(key for key in _DEFAULT_STEP_KEY_ORDER if key in available)

    def _resolve_step_order(self):
        step_order = self._step_order_from_metadata()
        if step_order is None:
            step_order = self._step_order_from_folders()
        if not step_order:
            raise ValueError(
                f"No supported step folders found in '{self.metrics_folder}'."
            )
        return step_order

    def _load_metrics(self):
        step_order = self._resolve_step_order()
        store = StepMetricsStore(self.metrics_folder)
        step_metrics = {}

        for step_key in step_order:
            step_folder = self.metrics_folder / step_key
            if not step_folder.is_dir():
                raise FileNotFoundError(
                    f"Step '{step_key}' listed for {self.cycle_label} has no folder at "
                    f"'{step_folder}'."
                )
            step_metrics[step_key] = store.load(step_key)

        return CycleMetrics.from_step_metrics(
            cycle_label=self.cycle_label,
            step_metrics_by_key=step_metrics,
            step_order=step_order,
        )

    def _load_context(self):
        store = SimulationContextStore(self.metrics_folder)
        if store.exists():
            return store.load()
        raise FileNotFoundError(
            f"No current simulation_context.pkl found for {self.cycle_label} "
            f"at '{store.path}'. Legacy pyDAC result folders are unsupported; "
            "regenerate the result."
        )

    def load(self):
        """Return validated cycle metrics plus required architecture context."""
        from pydac.utils.provenance import file_artifact, metrics_sources
        sources = metrics_sources(self.metrics_folder, self._resolve_step_order())
        context_path = SimulationContextStore(self.metrics_folder).path
        context_artifact = file_artifact(context_path)
        metrics = self._load_metrics()
        context = self._load_context()
        if (sources != metrics_sources(self.metrics_folder, metrics.step_order)
                or context_artifact != file_artifact(context_path)):
            raise ValueError("Cycle inputs changed while loading their provenance.")
        return CycleData(
            metrics=metrics,
            context=context,
            provenance=[*sources, {"artifact": context_artifact}],
        )
