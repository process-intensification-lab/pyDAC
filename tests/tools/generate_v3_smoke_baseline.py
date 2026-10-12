"""Generate the deterministic pyDAC startup smoke baseline."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.constants import gas_constant as R_GAS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

from pydac.core.column_model_factory import ColumnModelFactory
from pydac.config import experiment_to_document, load_experiment
from pydac.config.cycle import CycleConfig
from pydac.inputs.case_io import case_fingerprint
from pydac.core.solver import Solver


SMOKE_SETTINGS = {
    "grid.nz": 4,
    "grid.nr": 1,
    "grid.nt": 5,
    "steps.startup.composition.co2_feed_mole_fraction": 0.0004,
    "steps.startup.composition.relative_humidity": 0.5,
    "steps.startup.composition.co2_initial_mole_fraction": 0.0,
    "steps.startup.composition.initial_relative_humidity": 0.0,
    "steps.startup.flow.volumetric_flow_m3_s": 0.001,
    "steps.startup.thermal.feed_temperature_K": 298.15,
    "steps.startup.thermal.initial_temperature_K": 298.15,
    "steps.startup.thermal.jacket_temperature_K": 298.15,
    "steps.startup.pressure.initial_pressure_Pa": 101325.0,
    "steps.startup.pressure.outlet_pressure_Pa": 101325.0,
    "steps.startup.pressure.feed_pressure_Pa": 101325.0,
    "steps.startup.horizon.nominal_time_s": 2.0,
    "steps.startup.horizon.event_max_time_s": 2.0,
    "steps.startup.termination.mode": "time",
    "solver.rtol": 1e-6,
    "solver.atol": 1e-8,
    "execution.use_sparsity": False,
    "execution.compute_numerical_sparsity": False,
    "execution.cycle_steps": ["STARTUP"],
    "execution.termination_mode": "time",
    "execution.plot_profiles": False,
    "execution.print_to_terminal": False,
    "execution.print_cycle_build": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float(value) -> float:
    return float(np.asarray(value).item())


def _field_summary(values: np.ndarray, unit: str) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    return {
        "unit": unit,
        "shape": list(array.shape),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def generate_observations(config, *, output_folder: Path) -> dict[str, object]:
    cycle = CycleConfig.from_experiment(config)
    factory = ColumnModelFactory(cycle)
    model = factory.build_column_model(
        step="startup",
        solver_options=cycle.solver_for_step(
            use_sparsity=False,
            compute_numerical_sparsity=False,
        ),
    )
    solver = Solver(model, base_folder=str(output_folder))
    solver.solve_cvode()
    solver.save_results()

    solution = solver.sol
    fields = np.asarray(solution["y_2d"], dtype=float)
    wall = np.asarray(solution["y_1d"][0], dtype=float)
    c_co2, c_h2o, c_o2, q_co2, q_h2o, c_total, temperature = fields
    c_n2 = c_total - c_co2 - c_h2o - c_o2
    concentrations = np.stack((c_co2, c_h2o, c_o2, c_n2), axis=0)
    mole_fractions = concentrations / c_total[np.newaxis, ...]
    pressure = c_total * R_GAS * temperature

    final_profiles = {
        "z_center": np.asarray(solution["extra"]["z_c"], dtype=float).tolist(),
        "c_co2_mol_m3": c_co2[:, 0, -1].tolist(),
        "c_h2o_mol_m3": c_h2o[:, 0, -1].tolist(),
        "c_o2_mol_m3": c_o2[:, 0, -1].tolist(),
        "c_n2_mol_m3": c_n2[:, 0, -1].tolist(),
        "q_co2_mol_kg": q_co2[:, 0, -1].tolist(),
        "q_h2o_mol_kg": q_h2o[:, 0, -1].tolist(),
        "c_total_mol_m3": c_total[:, 0, -1].tolist(),
        "temperature_bed_K": temperature[:, 0, -1].tolist(),
        "temperature_wall_K": wall[:, -1].tolist(),
        "pressure_Pa": pressure[:, 0, -1].tolist(),
    }

    return {
        "solver": {
            "implementation": solution["solver"],
            "success": bool(solution["cvode"]["success"]),
            "status": int(solution["cvode"]["status"]),
            "message": str(solution["cvode"]["message"]),
            "nfev_reference_only": int(solution["cvode"]["nfev"]),
            "njev_reference_only": int(solution["cvode"]["njev"]),
            "termination_reason": solution["termination"]["reason"],
        },
        "stored_time_s": np.asarray(solution["t"], dtype=float).tolist(),
        "runtime_properties": {
            "feed_temperature_K": _float(model.T_feed),
            "feed_pressure_Pa": _float(model.p_feed),
            "mixture_density_kg_m3": _float(model.rho_g_mix),
            "mixture_viscosity_Pa_s": _float(model.mu_g_mix),
            "mixture_heat_capacity_J_molK": _float(model.cp_g_mix),
            "mixture_thermal_conductivity_W_mK": _float(model.lambda_g_mix),
            "molecular_diffusivity_m2_s": np.asarray(model.D_m, dtype=float).tolist(),
            "reynolds_number": _float(model.Re),
            "prandtl_number": _float(model.Pr),
            "schmidt_number": np.asarray(model.Sc, dtype=float).tolist(),
            "sherwood_number": np.asarray(model.Sh, dtype=float).tolist(),
            "internal_wall_htc_W_m2K": _float(model.h_int),
        },
        "outlet_histories": {
            "co2_mole_fraction": mole_fractions[0, -1, 0, :].tolist(),
            "h2o_mole_fraction": mole_fractions[1, -1, 0, :].tolist(),
            "o2_mole_fraction": mole_fractions[2, -1, 0, :].tolist(),
            "n2_mole_fraction": mole_fractions[3, -1, 0, :].tolist(),
            "temperature_K": temperature[-1, 0, :].tolist(),
            "pressure_Pa": pressure[-1, 0, :].tolist(),
        },
        "final_profiles": final_profiles,
        "final_field_summaries": {
            "c_co2": _field_summary(c_co2[..., -1], "mol m^-3"),
            "c_h2o": _field_summary(c_h2o[..., -1], "mol m^-3"),
            "c_o2": _field_summary(c_o2[..., -1], "mol m^-3"),
            "c_n2": _field_summary(c_n2[..., -1], "mol m^-3"),
            "q_co2": _field_summary(q_co2[..., -1], "mol kg^-1"),
            "q_h2o": _field_summary(q_h2o[..., -1], "mol kg^-1"),
            "c_total": _field_summary(c_total[..., -1], "mol m^-3"),
            "temperature_bed": _field_summary(temperature[..., -1], "K"),
            "temperature_wall": _field_summary(wall[..., -1], "K"),
            "pressure": _field_summary(pressure[..., -1], "Pa"),
        },
    }


def main() -> int:
    """Regenerate the smoke case and its baseline.

    Careful: the committed `tests/cases/v3_smoke_startup.json` is a **schema-v1**
    document, kept deliberately so the v1 upgrade path stays exercised, while
    `experiment_to_document` here emits schema v2. Running this replaces it and
    silently retires that coverage. To refresh only the baseline against the
    committed v1 case, recompute `case.fingerprint` from `load_experiment(case)`
    instead of rewriting the case.
    """
    cases_root = PROJECT_ROOT / "tests" / "cases"
    baselines_root = PROJECT_ROOT / "tests" / "baselines"
    cases_root.mkdir(parents=True, exist_ok=True)
    baselines_root.mkdir(parents=True, exist_ok=True)

    config = load_experiment(settings=SMOKE_SETTINGS)
    case_path = cases_root / "v3_smoke_startup.json"
    case_path.write_text(
        json.dumps(experiment_to_document(config), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with tempfile.TemporaryDirectory(prefix="pydac_phase0_smoke_") as temporary:
        observations = generate_observations(config, output_folder=Path(temporary))

    baseline = {
        "schema_version": 1,
        "description": "Deterministic current-v3 1D startup smoke baseline.",
        "case": {
            "path": str(case_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(case_path),
            "fingerprint": case_fingerprint(config),
        },
        "comparison_policy": "v3_regression",
        "observations": observations,
    }
    baseline_path = baselines_root / "v3_smoke_startup.json"
    baseline_path.write_text(json.dumps(baseline, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(case_path)
    print(baseline_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
