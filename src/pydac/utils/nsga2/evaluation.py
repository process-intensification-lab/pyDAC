"""Filesystem, formatting, and KPI adapters for one NSGA-II evaluation."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from pydac.utils.campaign import atomic_write_json

from pydac.utils.nsga2 import logging as nsga2_shared


CANDIDATE_RECORD_FILENAME = "candidate_record.json"
EVALUATION_SUMMARY_FILENAME = "evaluation_summary.json"
_COMPACT_CANDIDATE_FILES = {
    CANDIDATE_RECORD_FILENAME,
    EVALUATION_SUMMARY_FILENAME,
}
_GENERATION_FOLDER_PATTERN = re.compile(r"gen_\d+")


def make_eval_folder(base_folder: str) -> tuple[str, int, str]:
    gen_id = nsga2_shared.CURRENT_GEN.value + 1
    eval_id = uuid.uuid4().hex[:8]
    pid = os.getpid()
    eval_folder = os.path.join(
        base_folder,
        f"gen_{gen_id:02d}",
        f"pid_{pid}_eval_{eval_id}",
    )
    os.makedirs(eval_folder, exist_ok=True)
    return eval_folder, gen_id, eval_id


def format_candidate_summary(
    operating_settings: Mapping[str, float | None],
    *,
    is_steam_assisted: bool,
    flow_inputs: Mapping[str, float],
) -> str:
    parts = [
        f"t_ads={operating_settings['adsorption_time_s']:.2f} s",
        f"t_heat={operating_settings['heating_time_s']:.2f} s",
        f"t_des={operating_settings['desorption_time_s']:.2f} s",
        f"P_vac={operating_settings['low_pressure_Pa']:.2f} Pa",
        f"v_feed={flow_inputs['v_feed']:.4e} m/s",
        f"vol_flowrate={operating_settings['volumetric_flow_m3_s']:.4e} m3/s",
    ]
    cooling_target = operating_settings.get("cooling_target_temperature_K")
    if cooling_target is not None:
        parts.append(f"T_cool_target={cooling_target:.2f} K")
    if is_steam_assisted:
        parts.append(f"v_steam={flow_inputs['v_steam']:.4e} m/s")
        parts.append(f"rho_steam={flow_inputs['rho_steam_model']:.4e} kg/m3")
        parts.append(
            "m_dot_steam="
            f"{operating_settings['steam_mass_flow_kg_s']:.4e} kg/s"
        )
    return ", ".join(parts)


def extract_candidate_kpis(cycle_kpis) -> dict[str, Any]:
    """Map the pure cycle KPI surface to the NSGA-II evaluation contract."""
    purity_metrics = cycle_kpis.purity()
    preheating_temperatures = cycle_kpis.preheating_temperatures()
    return {
        "eq_work": float(cycle_kpis.equivalent_work()),
        "productivity": float(cycle_kpis.productivity()),
        "purity": float(purity_metrics["purity_wet"]),
        "oxygen_ppm": float(purity_metrics["oxygen_ppm"]),
        "T_bed_min_preheat": float(
            preheating_temperatures["bed_minimum_K"]
        ),
        # NaN for a mixture without H2O: no steam, no condensation limit.
        "T_steam_condensation": float(
            preheating_temperatures["steam_condensation_K"]
            if preheating_temperatures["steam_condensation_K"] is not None
            else "nan"
        ),
        "energy_breakdown": dict(cycle_kpis.specific_energy_breakdown()),
        "cyclic_working_capacity": float(
            cycle_kpis.cyclic_working_capacity()
        ),
    }


def write_candidate_record(eval_folder: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically persist one compact, trajectory-free candidate record."""
    path = Path(eval_folder) / CANDIDATE_RECORD_FILENAME
    atomic_write_json(path, dict(payload), sort_keys=True)
    return path


def prune_candidate_artifacts(
    eval_folder: str | Path,
    *,
    keep_trajectories: bool,
) -> list[str]:
    """Remove ordinary-candidate artifacts after KPI/failure extraction.

    The two compact JSON files survive until the generation callback has
    consolidated them.  Explicit reruns may retain the complete folder.
    """
    folder = Path(eval_folder)
    if keep_trajectories or not folder.exists():
        return []

    removed: list[str] = []
    for path in folder.iterdir():
        if path.name in _COMPACT_CANDIDATE_FILES:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path.name)
    return sorted(removed)


def commit_generation_candidate_records(
    base_folder: str | Path,
    generation: int,
    *,
    expected_count: int | None = None,
) -> tuple[Path, int]:
    """Atomically consolidate one completed generation into JSONL.

    Per-candidate JSON files live only in the transient ``gen_*`` folder.  A
    durable generation record appears after the optimizer callback has written
    its population/front checkpoints.  A count mismatch fails loudly instead
    of committing an incomplete record set.
    """
    base = Path(base_folder)
    generation_folder = base / f"gen_{int(generation):02d}"
    source_paths = sorted(
        generation_folder.glob(f"pid_*_eval_*/{CANDIDATE_RECORD_FILENAME}")
    )
    records = []
    for path in source_paths:
        with path.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))

    if expected_count is not None and len(records) != int(expected_count):
        raise RuntimeError(
            f"Generation {generation} produced {len(records)} compact candidate "
            f"records for {int(expected_count)} new evaluations."
        )

    destination = (
        base
        / "results"
        / "candidate_records"
        / f"generation_{int(generation):04d}.jsonl"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(destination)
    return destination, len(records)


def campaign_evaluation_summary(base_folder: str | Path) -> dict[str, Any]:
    """Candidates evaluated and failed in the recorded generations, with the distinct errors.

    Reads ``results/candidate_records/generation_*.jsonl``; ``errors`` maps each
    distinct failure message to its count, most frequent first.
    """
    folder = Path(base_folder) / "results" / "candidate_records"
    evaluated = 0
    errors: Counter[str] = Counter()
    for path in sorted(folder.glob("generation_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            evaluated += 1
            failure = json.loads(line).get("failure")
            if failure:
                errors[str(failure.get("error"))] += 1
    return {
        "evaluated": evaluated,
        "failed": sum(errors.values()),
        "errors": dict(errors.most_common()),
    }


def format_evaluation_summary(summary: Mapping[str, Any], *, limit: int = 5) -> str:
    """``Candidates evaluated N, failed F.`` plus the most frequent distinct errors."""
    lines = [f"Candidates evaluated {summary['evaluated']}, failed {summary['failed']}."]
    errors = list(summary["errors"].items())
    lines += [f"  {count} x {error}" for error, count in errors[:limit]]
    if len(errors) > limit:
        lines.append(f"  ... and {len(errors) - limit} other distinct errors")
    return "\n".join(lines)


def discard_transient_generation_folders(base_folder: str | Path) -> list[str]:
    """Delete uncommitted per-candidate folders before a continuation."""
    base = Path(base_folder)
    if not base.exists():
        return []
    removed = []
    for path in sorted(base.iterdir()):
        if (
            path.is_dir()
            and not path.is_symlink()
            and _GENERATION_FOLDER_PATTERN.fullmatch(path.name)
        ):
            shutil.rmtree(path)
            removed.append(path.name)
    return removed


__all__ = [
    "CANDIDATE_RECORD_FILENAME",
    "EVALUATION_SUMMARY_FILENAME",
    "campaign_evaluation_summary",
    "commit_generation_candidate_records",
    "discard_transient_generation_folders",
    "format_evaluation_summary",
    "extract_candidate_kpis",
    "format_candidate_summary",
    "make_eval_folder",
    "prune_candidate_artifacts",
    "write_candidate_record",
]
