"""Post-solver audit for oxygen present in a hot bed region.

The audit is report-only.  It never changes a solver event, termination mode,
or state vector.  It evaluates the saved dimensional ``sol.pkl`` trajectories
and checks the pointwise condition::

    T_bed(z, r, t) > temperature_limit_C
    and
    ppm_O2(z, r, t) > oxygen_floor_ppm

The oxygen floor is intentionally separate from the physical threshold of zero
to avoid treating floating-point residue as oxygen.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_TEMPERATURE_LIMIT_C = 90.0
DEFAULT_OXYGEN_LIMIT_PPM = 0.0
DEFAULT_OXYGEN_NUMERIC_FLOOR_PPM = 1.0e-6
SIMPLE_CHECK_COLUMNS = ("condition_met", "step", "point", "time_s")
SIMPLE_CHECK_FILENAME = "oxygen_hot_bed_check.csv"


def _load_solution(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        solution = pickle.load(stream)
    if not isinstance(solution, dict):
        raise ValueError(f"Expected a solution dictionary in {path}")
    return solution


def _saved_coordinates(extra: dict[str, Any], key: str, size: int) -> list[float] | None:
    values = extra.get(key)
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != size or not np.all(np.isfinite(array)):
        return None
    return [float(value) for value in array]


def _time_vector(solution: dict[str, Any], nt: int) -> np.ndarray:
    values = np.asarray(solution.get("t", np.arange(nt)), dtype=float).reshape(-1)
    if values.size != nt:
        raise ValueError(
            f"Saved time vector has {values.size} samples, but the fields have {nt}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Saved time vector contains non-finite values.")
    return values


def _location(
    mask: np.ndarray,
    field: np.ndarray,
    *,
    maximize: bool,
    times: np.ndarray,
    z_coordinates: list[float] | None,
    r_coordinates: list[float] | None,
) -> dict[str, Any] | None:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return None
    values = field[tuple(indices.T)]
    selected = int(np.argmax(values) if maximize else np.argmin(values))
    z_index, r_index, time_index = (int(value) for value in indices[selected])
    result: dict[str, Any] = {
        "z_index": z_index,
        "r_index": r_index,
        "time_index": time_index,
        "time_s": float(times[time_index]),
        "value": float(values[selected]),
    }
    if z_coordinates is not None:
        result["z_coordinate"] = z_coordinates[z_index]
    if r_coordinates is not None:
        result["r_coordinate"] = r_coordinates[r_index]
    return result


def _first_location(
    mask: np.ndarray,
    field: np.ndarray,
    *,
    times: np.ndarray,
    z_coordinates: list[float] | None,
    r_coordinates: list[float] | None,
) -> dict[str, Any] | None:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return None
    selected = int(np.argmin(indices[:, 2]))
    z_index, r_index, time_index = (int(value) for value in indices[selected])
    result: dict[str, Any] = {
        "z_index": z_index,
        "r_index": r_index,
        "time_index": time_index,
        "time_s": float(times[time_index]),
        "value": float(field[z_index, r_index, time_index]),
    }
    if z_coordinates is not None:
        result["z_coordinate"] = z_coordinates[z_index]
    if r_coordinates is not None:
        result["r_coordinate"] = r_coordinates[r_index]
    return result


def _integral_of_indicator(indicator: np.ndarray, times: np.ndarray) -> float:
    if indicator.size < 2:
        return 0.0
    if np.any(np.diff(times) < 0.0):
        raise ValueError("Saved time vector must be non-decreasing for exposure duration.")
    integrand = np.asarray(indicator, dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(integrand, times))
    return float(np.trapz(integrand, times))


def _episode_indices(active_samples: np.ndarray) -> Iterable[np.ndarray]:
    active = np.flatnonzero(active_samples)
    if active.size == 0:
        return ()
    split_points = np.flatnonzero(np.diff(active) > 1) + 1
    return np.split(active, split_points)


def _episode_report(
    risk_mask: np.ndarray,
    temperature_C: np.ndarray,
    oxygen_ppm: np.ndarray,
    times: np.ndarray,
    sample_indices: np.ndarray,
    z_coordinates: list[float] | None,
    r_coordinates: list[float] | None,
) -> dict[str, Any]:
    start = int(sample_indices[0])
    end = int(sample_indices[-1])
    window = slice(start, end + 1)
    window_mask = risk_mask[..., window]
    episode_indicator = np.zeros(times.size, dtype=bool)
    episode_indicator[sample_indices] = True
    return {
        "start_time_s": float(times[start]),
        "end_time_s": float(times[end]),
        "duration_s": _integral_of_indicator(episode_indicator, times),
        "sampled_span_s": max(float(times[end] - times[start]), 0.0),
        "sample_count": int(sample_indices.size),
        "max_temperature_C": float(np.max(temperature_C[..., window][window_mask])),
        "max_oxygen_ppm": float(np.max(oxygen_ppm[..., window][window_mask])),
        "first_risk_location": _first_location(
            risk_mask[..., window],
            temperature_C[..., window],
            times=times[window],
            z_coordinates=z_coordinates,
            r_coordinates=r_coordinates,
        ),
        "hottest_risk_location": _location(
            risk_mask[..., window],
            temperature_C[..., window],
            maximize=True,
            times=times[window],
            z_coordinates=z_coordinates,
            r_coordinates=r_coordinates,
        ),
        "highest_oxygen_risk_location": _location(
            risk_mask[..., window],
            oxygen_ppm[..., window],
            maximize=True,
            times=times[window],
            z_coordinates=z_coordinates,
            r_coordinates=r_coordinates,
        ),
    }


def audit_step_solution(
    solution_path: str | Path,
    *,
    step: str | None = None,
    temperature_limit_C: float = DEFAULT_TEMPERATURE_LIMIT_C,
    oxygen_limit_ppm: float = DEFAULT_OXYGEN_LIMIT_PPM,
    oxygen_floor_ppm: float = DEFAULT_OXYGEN_NUMERIC_FLOOR_PPM,
) -> dict[str, Any]:
    """Audit one saved step trajectory and return a detailed in-memory report."""
    path = Path(solution_path)
    solution = _load_solution(path)
    fields = np.asarray(solution.get("y_2d"), dtype=float)
    if fields.ndim != 4:
        raise ValueError(
            f"Expected saved y_2d with shape (fields, Nz, Nr, Nt), got {fields.shape} in {path}"
        )

    extra = solution.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    layout_document = extra.get("state_layout")
    if isinstance(layout_document, dict):
        fields_2d = [
            field
            for field in layout_document.get("fields", [])
            if field.get("storage") == "y_2d"
        ]
        field_index = {field["name"]: index for index, field in enumerate(fields_2d)}
        try:
            c_tot = fields[field_index["c_tot"]]
            temperature_K = fields[field_index["T_bed"]]
        except KeyError as exc:
            raise ValueError(
                f"Saved layout in {path} lacks required field {exc.args[0]!r}."
            ) from exc

        if "c_O2" in field_index:
            c_o2 = fields[field_index["c_O2"]]
        elif layout_document.get("mixture", {}).get("closure") == "O2":
            independent = layout_document["mixture"].get("independent_species", [])
            c_o2 = c_tot - sum(fields[field_index[f"c_{name}"]] for name in independent)
        else:
            raise ValueError(
                f"Saved layout {layout_document.get('signature')} contains no O2 gas state."
            )
    else:
        # Compatibility with saved DAC solutions created before layout metadata.
        if fields.shape[0] < 7:
            raise ValueError(
                f"Legacy y_2d requires at least seven fields, got {fields.shape} in {path}."
            )
        c_o2 = fields[2]
        c_tot = fields[5]
        temperature_K = fields[6]

    times = _time_vector(solution, fields.shape[-1])
    ppm_o2 = np.divide(
        1.0e6 * c_o2,
        c_tot,
        out=np.zeros_like(c_o2),
        where=c_tot > 0.0,
    )
    temperature_C = temperature_K - 273.15
    risk_mask = (temperature_C > float(temperature_limit_C)) & (
        ppm_o2 > max(float(oxygen_limit_ppm), float(oxygen_floor_ppm))
    )
    hot_mask = temperature_C > float(temperature_limit_C)
    oxygen_mask = ppm_o2 > max(float(oxygen_limit_ppm), float(oxygen_floor_ppm))
    z_coordinates = _saved_coordinates(extra, "z_c", fields.shape[1])
    r_coordinates = _saved_coordinates(extra, "r_c", fields.shape[2])

    risk_by_sample = np.any(risk_mask, axis=(0, 1))
    oxygen_hot_fraction = np.mean(risk_mask, axis=(0, 1))
    episodes = [
        _episode_report(
            risk_mask,
            temperature_C,
            ppm_o2,
            times,
            sample_indices,
            z_coordinates,
            r_coordinates,
        )
        for sample_indices in _episode_indices(risk_by_sample)
    ]

    max_temperature_when_oxygen = _location(
        oxygen_mask,
        temperature_C,
        maximize=True,
        times=times,
        z_coordinates=z_coordinates,
        r_coordinates=r_coordinates,
    )
    max_oxygen_when_hot = _location(
        hot_mask,
        ppm_o2,
        maximize=True,
        times=times,
        z_coordinates=z_coordinates,
        r_coordinates=r_coordinates,
    )
    first_risk = _first_location(
        risk_mask,
        temperature_C,
        times=times,
        z_coordinates=z_coordinates,
        r_coordinates=r_coordinates,
    )
    hottest_risk = _location(
        risk_mask,
        temperature_C,
        maximize=True,
        times=times,
        z_coordinates=z_coordinates,
        r_coordinates=r_coordinates,
    )
    highest_oxygen_risk = _location(
        risk_mask,
        ppm_o2,
        maximize=True,
        times=times,
        z_coordinates=z_coordinates,
        r_coordinates=r_coordinates,
    )

    report = {
        "step": step or path.parent.name,
        "solution_path": str(path),
        "thresholds": {
            "temperature_limit_C": float(temperature_limit_C),
            "oxygen_limit_ppm": float(oxygen_limit_ppm),
            "oxygen_numeric_floor_ppm": float(oxygen_floor_ppm),
            "effective_oxygen_threshold_ppm": max(
                float(oxygen_limit_ppm), float(oxygen_floor_ppm)
            ),
        },
        "sampling": {
            "Nz": int(fields.shape[1]),
            "Nr": int(fields.shape[2]),
            "Nt": int(fields.shape[3]),
            "time_start_s": float(times[0]),
            "time_end_s": float(times[-1]),
            "sampled_duration_s": max(float(times[-1] - times[0]), 0.0),
        },
        "risk_detected": bool(risk_mask.any()),
        "risk_sample_count": int(risk_mask.sum()),
        "risk_max_domain_fraction": float(np.max(oxygen_hot_fraction)),
        "risk_exposure_any_location_duration_s": _integral_of_indicator(
            risk_by_sample, times
        ),
        "risk_exposure_cell_fraction_time_s": _integral_of_indicator(
            oxygen_hot_fraction, times
        ),
        "overall_max_temperature_C": float(np.max(temperature_C)),
        "overall_max_oxygen_ppm": float(np.max(ppm_o2)),
        "max_temperature_when_oxygen_present": max_temperature_when_oxygen,
        "max_oxygen_when_temperature_hot": max_oxygen_when_hot,
        "first_risk_location": first_risk,
        "hottest_risk_location": hottest_risk,
        "highest_oxygen_risk_location": highest_oxygen_risk,
        "episodes": episodes,
    }
    return report


def _executed_steps(cycle_folder: Path) -> list[str]:
    metadata_path = cycle_folder / "cycle_metadata.json"
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        steps = metadata.get("executed_steps", [])
        if isinstance(steps, list) and all(isinstance(step, str) for step in steps):
            return steps
    return sorted(
        path.parent.name
        for path in cycle_folder.glob("*/sol.pkl")
    )


def audit_cycle_folder(
    cycle_folder: str | Path,
    *,
    steps: list[str] | None = None,
    temperature_limit_C: float = DEFAULT_TEMPERATURE_LIMIT_C,
    oxygen_limit_ppm: float = DEFAULT_OXYGEN_LIMIT_PPM,
    oxygen_floor_ppm: float = DEFAULT_OXYGEN_NUMERIC_FLOOR_PPM,
) -> dict[str, Any]:
    """Audit all saved step solutions in one cycle folder."""
    folder = Path(cycle_folder)
    step_names = list(steps) if steps is not None else _executed_steps(folder)
    step_reports = []
    errors = []
    for step in step_names:
        solution_path = folder / step / "sol.pkl"
        if not solution_path.is_file():
            errors.append({"step": step, "error": f"missing {solution_path}"})
            continue
        try:
            step_reports.append(
                audit_step_solution(
                    solution_path,
                    step=step,
                    temperature_limit_C=temperature_limit_C,
                    oxygen_limit_ppm=oxygen_limit_ppm,
                    oxygen_floor_ppm=oxygen_floor_ppm,
                )
            )
        except (OSError, ValueError, pickle.UnpicklingError) as exc:
            errors.append({"step": step, "error": str(exc)})

    risk_steps = [report["step"] for report in step_reports if report["risk_detected"]]
    return {
        "schema_version": 1,
        "report_type": "post_solver_oxygen_hot_bed_audit",
        "cycle_folder": str(folder),
        "thresholds": {
            "temperature_limit_C": float(temperature_limit_C),
            "oxygen_limit_ppm": float(oxygen_limit_ppm),
            "oxygen_numeric_floor_ppm": float(oxygen_floor_ppm),
            "effective_oxygen_threshold_ppm": max(
                float(oxygen_limit_ppm), float(oxygen_floor_ppm)
            ),
        },
        "risk_detected": bool(risk_steps),
        "risk_steps": risk_steps,
        "steps_audited": [report["step"] for report in step_reports],
        "errors": errors,
        "step_reports": step_reports,
    }


def simple_check_row(report: dict[str, Any]) -> dict[str, Any]:
    """Return the first detected event in the four fields used by the CSV.

    The detailed per-step report is kept in memory only to find the first
    pointwise event, respecting the executed step order.  The public result
    intentionally contains only the requested status, step, point, and time.
    """
    empty_row = {column: "" for column in SIMPLE_CHECK_COLUMNS}
    empty_row["condition_met"] = bool(report.get("risk_detected", False))
    if not empty_row["condition_met"]:
        return empty_row

    selected = None
    for step_report in report.get("step_reports", []):
        if not step_report.get("risk_detected"):
            continue
        location = step_report.get("first_risk_location") or {}
        selected = (step_report, location)
        break

    if selected is None:
        # Keep the status truthful even if a malformed detailed report is
        # supplied by a caller.
        return empty_row

    step_report, location = selected
    z_index = location.get("z_index")
    r_index = location.get("r_index")
    if z_index is None or r_index is None:
        point = "unknown"
    else:
        point = f"z_index={int(z_index)},r_index={int(r_index)}"
        z_coordinate = location.get("z_coordinate")
        r_coordinate = location.get("r_coordinate")
        if z_coordinate is not None and r_coordinate is not None:
            point += f" (z={float(z_coordinate):.12g},r={float(r_coordinate):.12g})"

    return {
        "condition_met": True,
        "step": step_report.get("step", ""),
        "point": point,
        "time_s": "" if location.get("time_s") is None else float(location["time_s"]),
    }


def write_simple_check_csv(
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write the compact post-processing result as a one-row CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = simple_check_row(report)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SIMPLE_CHECK_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return path


def simple_check_message(report: dict[str, Any]) -> str:
    """Return the single terminal line for the compact post-processing check."""
    row = simple_check_row(report)
    if not row["condition_met"]:
        return "Oxygen/hot-bed check: condition not detected."
    return (
        "Oxygen/hot-bed check: condition detected — "
        f"step={row['step']}, point={row['point']}, time={float(row['time_s']):.3f} s."
    )


def run_oxygen_hot_bed_check(
    results_base_folder: str | Path,
    cycle_idx: int,
) -> Path | None:
    """Run the compact report-only check and write its process KPI CSV.

    This wrapper deliberately owns the diagnostic failure boundary: an audit
    problem is reported, but never turns an already completed simulation into
    a failed run.
    """
    results_folder = Path(results_base_folder)
    try:
        safety_report = audit_cycle_folder(results_folder / f"cycle_{cycle_idx}")
        safety_csv = write_simple_check_csv(
            safety_report,
            results_folder / "kpis" / "csv" / "process" / SIMPLE_CHECK_FILENAME,
        )
        print(simple_check_message(safety_report))
        print(f"Oxygen/hot-bed check CSV: {safety_csv}")
        return safety_csv
    except Exception as exc:
        print(f"WARNING: oxygen/hot-bed check unavailable: {exc}")
        return None
