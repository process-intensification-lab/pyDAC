"""Compact artifact contract for ordinary NSGA-II candidates."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pydac.utils.nsga2.evaluation import (
    CANDIDATE_RECORD_FILENAME,
    EVALUATION_SUMMARY_FILENAME,
    commit_generation_candidate_records,
    discard_transient_generation_folders,
    write_candidate_record,
)


def _run_config(nsga2, tmp_path):
    from pydac.utils.solver.composition_domain import CompositionDomainGuard

    return SimpleNamespace(
        is_steam_assisted=True,
        tau_ar=0.5,
        bed_volume=1e-3,
        base_folder=str(tmp_path),
        experiment=object(),
        require_css=True,
        constraints=nsga2.NSGA2Constraints(),
        var_names=("t_ads", "t_heat", "t_des", "P_vac", "v_feed", "v_steam"),
        composition_domain_guard=CompositionDomainGuard(),
    )


def _patch_candidate_inputs(monkeypatch, nsga2):
    monkeypatch.setattr(
        nsga2,
        "decision_vector_to_model_inputs",
        lambda *args, **kwargs: (
            {"t_ads": 900.0},
            {"v_feed": 0.01, "v_steam": 0.001},
        ),
    )
    monkeypatch.setattr(
        nsga2,
        "_format_candidate_summary",
        lambda *args, **kwargs: "candidate",
    )
    monkeypatch.setattr(nsga2, "operating_step_settings", lambda settings: {})
    from pydac.config import load_experiment as real_load_experiment
    provenance_case = real_load_experiment("tests/cases/v3_smoke_startup.json")
    monkeypatch.setattr(nsga2, "load_experiment", lambda *args, **kwargs: provenance_case)


def test_ordinary_candidate_is_bounded_recorded_and_pruned(monkeypatch, tmp_path):
    from pydac.workflows import orchestrator_NSGA2 as nsga2

    run_config = _run_config(nsga2, tmp_path)
    eval_folder = tmp_path / "candidate"
    _patch_candidate_inputs(monkeypatch, nsga2)

    def write_resolved_config(_experiment, path):
        Path(path).write_text("{}", encoding="utf-8")

    def run_css_simulation(*, base_folder, retain_cycles, **kwargs):
        assert retain_cycles == 2
        assert kwargs["composition_domain_guard"] == run_config.composition_domain_guard
        assert kwargs["persistence_policy"] == "moo_compact"
        folder = Path(base_folder)
        for cycle in (1, 2):
            step = folder / f"cycle_{cycle}" / "heat"
            step.mkdir(parents=True)
            (step / "sol.pkl").write_bytes(b"trajectory")
        (folder / "plots").mkdir()
        (folder / "plots" / "profile.pdf").write_bytes(b"plot")
        (folder / "verbose.log").write_text("verbose", encoding="utf-8")
        return {
            "css_reached": True,
            "cycles_completed": 2,
            "css_cycle_idx": 2,
            "results_base_folder": str(folder),
            "last_step_records": [
                {
                    "step": "heat",
                    "solver": {
                        "step": "heat",
                        "success": True,
                        "status": 0,
                        "message": "success",
                        "nfev": 12,
                        "effective_solver": "sparse",
                        "final_time_s": 60.0,
                    },
                }
            ],
        }

    class FakeLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return object()

    monkeypatch.setattr(nsga2, "write_resolved_config", write_resolved_config)
    monkeypatch.setattr(nsga2, "run_css_simulation", run_css_simulation)
    monkeypatch.setattr(nsga2, "CycleDataLoader", FakeLoader)
    monkeypatch.setattr(nsga2, "CycleKPIs", lambda data: object())
    monkeypatch.setattr(
        nsga2,
        "_extract_candidate_kpis",
        lambda cycle_kpis: {
            "eq_work": 4.2,
            "productivity": 1.3,
            "purity": 0.97,
            "oxygen_ppm": 12.0,
            "T_bed_min_preheat": 390.0,
            "T_steam_condensation": 380.0,
            "energy_breakdown": {"heat_Q": 3.0},
            "cyclic_working_capacity": 0.8,
        },
    )

    result = nsga2.evaluate_pydac_candidate(
        [900.0, 60.0, 300.0, 3000.0, 0.01, 0.001],
        run_config=run_config,
        eval_folder=str(eval_folder),
    )

    assert result["failed"] is False
    assert {path.name for path in eval_folder.iterdir()} == {
        CANDIDATE_RECORD_FILENAME,
        EVALUATION_SUMMARY_FILENAME,
    }
    assert not list(eval_folder.rglob("sol.pkl"))

    record = json.loads(
        (eval_folder / CANDIDATE_RECORD_FILENAME).read_text(encoding="utf-8")
    )
    assert record["schema_version"] == 2
    assert record["decision_vector"]["t_ads"] == 900.0
    assert record["objectives"] == {
        "eq_work_MJ_per_kgCO2": 4.2,
        "neg_productivity_kgCO2_per_m3_day": -1.3,
    }
    assert record["constraints"]["valid_evaluation_and_css"] == -1.0
    assert record["css"] == {
        "cycles_completed": 2,
        "reached": True,
        "history": [],
    }
    assert record["solver_outcomes"][0]["effective_solver"] == "sparse"
    assert record["persistence"] == {
        "policy": "moo_compact",
        "full_trajectories": False,
        "terminal_state_schema_version": 1,
    }
    assert record["failure"] is None


def test_explicit_rerun_keeps_candidate_trajectories(monkeypatch, tmp_path):
    from pydac.workflows import orchestrator_NSGA2 as nsga2

    run_config = _run_config(nsga2, tmp_path)
    eval_folder = tmp_path / "rerun"
    _patch_candidate_inputs(monkeypatch, nsga2)
    monkeypatch.setattr(nsga2, "write_resolved_config", lambda *args: None)

    def run_css_simulation(*, base_folder, retain_cycles, **kwargs):
        assert retain_cycles == 2
        assert kwargs["composition_domain_guard"] == run_config.composition_domain_guard
        assert kwargs["persistence_policy"] == "run_full"
        step = Path(base_folder) / "cycle_2" / "heat"
        step.mkdir(parents=True)
        (step / "sol.pkl").write_bytes(b"trajectory")
        return {
            "css_reached": True,
            "cycles_completed": 2,
            "css_cycle_idx": 2,
            "results_base_folder": str(base_folder),
            "last_step_records": [],
        }

    class FakeLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load(self):
            return object()

    monkeypatch.setattr(nsga2, "run_css_simulation", run_css_simulation)
    monkeypatch.setattr(nsga2, "CycleDataLoader", FakeLoader)
    monkeypatch.setattr(nsga2, "CycleKPIs", lambda data: object())
    monkeypatch.setattr(
        nsga2,
        "_extract_candidate_kpis",
        lambda cycle_kpis: {
            "eq_work": 4.2,
            "productivity": 1.3,
            "purity": 0.97,
            "oxygen_ppm": 12.0,
            "T_bed_min_preheat": 390.0,
            "T_steam_condensation": 380.0,
            "energy_breakdown": {},
            "cyclic_working_capacity": 0.8,
        },
    )

    nsga2.evaluate_pydac_candidate(
        [900.0, 60.0, 300.0, 3000.0, 0.01, 0.001],
        run_config=run_config,
        eval_folder=str(eval_folder),
        keep_trajectories=True,
    )

    assert (eval_folder / "cycle_2" / "heat" / "sol.pkl").is_file()


def test_generation_records_are_atomically_consolidated(tmp_path):
    for eval_id in ("a", "b"):
        write_candidate_record(
            tmp_path / "gen_01" / f"pid_1_eval_{eval_id}",
            {
                "evaluation_id": eval_id,
                "decision_vector": {"t_ads": 900.0},
                "solver_outcomes": [],
            },
        )

    path, count = commit_generation_candidate_records(
        tmp_path,
        1,
        expected_count=2,
    )

    assert count == 2
    assert path == tmp_path / "results/candidate_records/generation_0001.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["evaluation_id"] for record in records] == ["a", "b"]
    assert not path.with_suffix(".jsonl.tmp").exists()
    assert "sol.pkl" not in path.read_text(encoding="utf-8")


def test_resume_discards_only_transient_generation_folders(tmp_path):
    (tmp_path / "gen_01" / "pid_1_eval_partial").mkdir(parents=True)
    (tmp_path / "gen_120").mkdir()
    (tmp_path / "generation_notes").mkdir()
    (tmp_path / "results").mkdir()

    removed = discard_transient_generation_folders(tmp_path)

    assert removed == ["gen_01", "gen_120"]
    assert (tmp_path / "generation_notes").is_dir()
    assert (tmp_path / "results").is_dir()
