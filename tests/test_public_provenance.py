"""Execution provenance portability, attribution and artifact compatibility at public boundaries."""

from copy import deepcopy
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.inputs.case_io import case_fingerprint
from pydac.core.TVSA_cycle import TVSA_Cycle
from pydac.utils import provenance as p
from pydac.utils.postprocessing.terminal_state import load_terminal_state

CASE = "tests/cases/v3_smoke_startup.json"


def test_git_missing_failed_dirty_and_unpublished_never_exports_revision(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    def missing(*args, **kwargs):
        raise FileNotFoundError("private path must not escape")
    monkeypatch.setattr(p.subprocess, "run", missing)
    assert p.git_observation(tmp_path)["status"] == "git_unavailable"
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("git private args", 2)
    monkeypatch.setattr(p.subprocess, "run", timeout)
    assert p.git_observation(tmp_path)["status"] == "inspection_failed"
    def dirty(command, **kwargs):
        assert command[2] == str(tmp_path)
        assert kwargs["timeout"] == 2
        return subprocess.CompletedProcess(command, 0, b" M private-file\n", b"")
    monkeypatch.setattr(p.subprocess, "run", dirty)
    observed = p.git_observation(tmp_path)
    assert observed["dirty"] is True
    assert observed["revision"] is None
    assert observed["revision_status"] == "unverified_or_non_public"
    assert "private" not in json.dumps(observed)


def test_input_relocation_is_explicit_and_original_fingerprint_is_preserved(tmp_path):
    config = load_experiment(CASE, settings={"execution.output_root": str(tmp_path)})
    original = case_fingerprint(config)
    record = p.new_record("run", config=config)
    assert record["input"]["source_case_metadata"]["case_fingerprint"] == original
    assert record["input"]["resolved_config"]["execution"]["output_root"] == "."
    assert str(tmp_path) not in json.dumps(record)
    assert case_fingerprint(config) == original
    tampered = deepcopy(record)
    tampered["input"]["resolved_config"]["simulation"]["grid"]["nz"] = 99
    with pytest.raises((ValueError, KeyError)):
        p.validate_record(tampered)


def test_saved_records_stay_readable_when_a_later_version_adds_a_case_field(tmp_path, monkeypatch):
    """A recorded case is checked as stored, not against today's vocabulary."""
    from pydac.config.errors import ConfigurationError
    from pydac.inputs import case_io
    config = load_experiment(CASE, settings={"execution.output_root": str(tmp_path)})
    record = p.new_record("run", config=config)
    stored = record["input"]["resolved_config"]
    assert case_io.stored_case_metadata(stored) == case_io.case_metadata(stored)
    path = p.save_record(tmp_path, record)
    sources = [p.source_record(record)]

    current = case_io._case_vocabulary
    def later_version(source, document):
        vocabulary = current(source, document)
        vocabulary["execution"]["new_future_flag"] = False
        return vocabulary
    monkeypatch.setattr(case_io, "_case_vocabulary", later_version)

    assert p.read_record(path) == record
    assert p.new_record("step_metrics", sources=sources)["sources"] == sources
    # Running the old case again is another matter: it needs a schema upgrade.
    with pytest.raises(ConfigurationError, match="new_future_flag"):
        case_io.load_case(stored)


def test_schema_parent_integrity_and_atomic_failure_preserve_previous_pointer(tmp_path, monkeypatch):
    record = p.new_record("solve")
    p.save_record(tmp_path, record, pointer="current.json")
    parent = p.source_record(record)
    parent["record"]["execution"]["changed"] = True
    with pytest.raises(ValueError, match="hash mismatch"):
        p.new_record("step_metrics", sources=[parent])
    invalid = deepcopy(record)
    invalid["schema_version"] = 2
    with pytest.raises(ValueError, match="schema version"):
        p.validate_record(invalid)
    original_replace = Path.replace
    def fail_pointer(path, target):
        if Path(target).name == "current.json":
            raise OSError("interrupted replacement")
        return original_replace(path, target)
    monkeypatch.setattr(Path, "replace", fail_pointer)
    with pytest.raises(OSError):
        p.save_record(tmp_path, p.new_record("solve"), pointer="current.json")
    assert p.read_record(tmp_path / "current.json") == record
    assert not list(tmp_path.glob(".provenance-*"))


@pytest.mark.parametrize("nr", [1, 3])
def test_reload_preserves_solve_identity_and_compact_endpoint(tmp_path, monkeypatch, nr):
    config = load_experiment(CASE, settings={"grid.nr": nr})
    runner = TVSA_Cycle(CycleConfig.from_experiment(config), base_folder=str(tmp_path),
                       base_folder_is_spatial_root=True, print_to_terminal=False)
    before, before_time = runner._run_step("startup", solve=True, use_sparsity=False)
    sol_path = tmp_path / "startup/sol.pkl"
    original_bytes = sol_path.read_bytes()
    sol = pickle.loads(original_bytes)
    solve_record = sol["solve_provenance"]
    assert solve_record["input"]["source_case_metadata"]["case_fingerprint"] == case_fingerprint(config)
    first_metrics = p.read_record(tmp_path / "startup/metrics_provenance.json")
    monkeypatch.setattr(p, "__version__", "3.0.0.dev99")
    after, after_time = runner._run_step("startup", solve=False, use_sparsity=False)
    assert before_time == after_time
    for field in before:
        np.testing.assert_array_equal(before[field], after[field])
    assert sol_path.read_bytes() == original_bytes
    metrics = p.read_record(tmp_path / "startup/metrics_provenance.json")
    assert metrics["record_id"] != first_metrics["record_id"]
    assert metrics["software"]["version"] == "3.0.0.dev99"
    compute = metrics["sources"][0]["record"]
    assert compute["sources"][0]["record"] == solve_record
    from pydac.utils.postprocessing.terminal_state import terminal_state_from_solution, save_terminal_state
    terminal = terminal_state_from_solution(sol, step="startup", solver_outcome={"success": True})
    terminal_path = save_terminal_state(terminal, tmp_path / "terminal.pkl")
    assert load_terminal_state(terminal_path)["solve_provenance"] == solve_record
    # Existing supported trajectories without provenance remain explicitly unknown.
    sol.pop("solve_provenance")
    sol_path.write_bytes(pickle.dumps(sol))
    runner._run_step("startup", solve=False, use_sparsity=False)
    legacy_metrics = p.read_record(tmp_path / "startup/metrics_provenance.json")
    assert legacy_metrics["sources"][0]["record"]["sources"][0]["status"] == "unknown"


def test_source_archive_without_git_outside_repository(tmp_path):
    root = Path(__file__).resolve().parents[1]
    archive = tmp_path / "archive"
    shutil.copytree(root / "src", archive / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(root / "pyproject.toml", archive / "pyproject.toml")
    shutil.copy(root / CASE, archive / "case.json")
    command = '''
import sys
from pathlib import Path
archive = Path(sys.argv[1])
sys.path.insert(0, str(archive / "src"))
import pydac
assert Path(pydac.__file__).resolve().is_relative_to(archive.resolve()), pydac.__file__
from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.core.solver import Solver
from pydac.utils import provenance
assert provenance.checkout_root() == (archive).resolve()
for nr in (1, 3):
    config = load_experiment(archive / "case.json", settings={"grid.nr": nr})
    cycle = CycleConfig.from_experiment(config)
    model = ColumnModelFactory(cycle).build_column_model(
        step="startup", solver_options=cycle.solver_for_step(use_sparsity=False))
    solver = Solver(model, base_folder=str(archive / f"result_{nr}"))
    solver.solve_cvode()
    solver.save_results()
    assert solver.sol["cvode"]["success"]
    record = solver.sol["solve_provenance"]
    assert record["software"]["git"]["status"] == "no_checkout"
    assert record["input"]["status"] == "recorded"
'''
    import os
    result = subprocess.run([sys.executable, "-c", command, str(archive)], cwd=tmp_path,
                            env={**os.environ, "PATH": "", "PYTHONPATH": ""},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_checkout_is_located_only_from_a_source_tree(tmp_path):
    package = tmp_path / "src/pydac"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    assert p.checkout_root(package) is None
    (tmp_path / "pyproject.toml").write_text("")
    assert p.checkout_root(package) == tmp_path
    installed = tmp_path / "site-packages/pydac"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("")
    (tmp_path / "site-packages/pyproject.toml").write_text("")
    assert p.checkout_root(installed) is None


def test_metrics_sources_reject_changed_files_and_survive_relocation(tmp_path):
    from pydac.post_processing.metrics.step_metrics import StepMetrics
    from pydac.post_processing.metrics.step_metrics_store import StepMetricsStore
    store = StepMetricsStore(tmp_path / "old")
    store.save(StepMetrics(step="startup"), source_provenance=p.new_record("solve"))
    original = p.metrics_sources(tmp_path / "old", ["startup"])
    shutil.move(tmp_path / "old", tmp_path / "new")
    assert p.metrics_sources(tmp_path / "new", ["startup"]) == original
    with (tmp_path / "new/startup/kpis.pkl").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="no longer matches"):
        p.metrics_sources(tmp_path / "new", ["startup"])


def test_analysis_uses_current_software_and_keeps_original_campaign_identity(tmp_path, monkeypatch):
    from pydac.post_processing.optimization.nsga2_post_processing import NSGA2_pp
    original = p.new_record("optimization")
    (tmp_path / "optimization_summary.json").write_text(json.dumps({"provenance": original}))
    np.savez(tmp_path / "last_pareto_front.npz", X=np.ones((2, 3)), F=np.ones((2, 2)))
    np.savez(tmp_path / "last_population.npz", X=np.ones((3, 3)), F=np.ones((3, 2)))
    monkeypatch.setattr(p, "__version__", "3.0.0.dev99")
    analysis = NSGA2_pp(tmp_path)
    record = analysis.analysis_provenance
    assert record["software"]["version"] == "3.0.0.dev99"
    assert record["sources"][-1]["record"] == original
    assert record["sources"][0]["artifact"] == p.file_artifact(tmp_path / "last_pareto_front.npz")


def test_initial_event_keeps_the_preintegration_identity(tmp_path, monkeypatch):
    from pydac.core.column_model_factory import ColumnModelFactory
    from pydac.core.solver import Solver
    cycle = CycleConfig.from_experiment(load_experiment(CASE))
    model = ColumnModelFactory(cycle).build_column_model(step="startup")
    solver = Solver(model, base_folder=str(tmp_path))
    solver.event_specs = [object()]
    def initial_event(state):
        solver.sol = {"termination": {"status": "initial_event"}}
        return True
    monkeypatch.setattr(solver, "_handle_initial_event", initial_event)
    solver.solve_cvode()
    record = solver.sol["solve_provenance"]
    assert record["execution"]["initial_state"]["shape"][0] == model.state_layout.state_size
    assert p.read_record(tmp_path / "startup/provenance" / f"{record['record_id']}.json") == record


def test_public_build_identity_is_separate_from_optional_runtime_git(monkeypatch):
    revision = "a" * 40
    public_source = {"revision": revision,
                     "url": "https://github.com/example/pydac/commit/" + revision}
    monkeypatch.setattr(p, "PUBLIC_SOURCE", public_source)
    monkeypatch.setattr(p, "git_observation", lambda: {
        "status": "git_unavailable", "dirty": None,
        "revision": None, "revision_status": "unavailable"})
    record = p.new_record("run")
    assert record["software"]["public_source"] == public_source
    assert record["software"]["git"]["revision"] is None
    record["software"]["public_source"]["url"] = "https://private.invalid/commit/" + revision
    with pytest.raises(ValueError, match="qualified public build"):
        p.validate_record(record)


def test_archive_inside_unrelated_git_tree_does_not_inherit_its_identity(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("This boundary test needs Git to construct an unrelated checkout.")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    archive = tmp_path / "archive"
    (archive / "src/pydac").mkdir(parents=True)
    (archive / "src/pydac/__init__.py").write_text("")
    (archive / "pyproject.toml").write_text("")
    assert p.checkout_root(archive / "src/pydac") == archive
    assert p.git_observation(archive)["status"] == "no_checkout"


def test_records_name_the_arithmetic_switches(monkeypatch):
    monkeypatch.delitem(sys.modules, "pydac.core.column_model", raising=False)
    monkeypatch.delitem(sys.modules, "pydac.core.pymrm_caching", raising=False)
    monkeypatch.setenv("PYDAC_RHS_CSR_OPERATORS", "off")
    monkeypatch.delenv("PYDAC_CACHE_SLICE_WRITE", raising=False)
    assert p.software_identity()["solver_environment"] == {
        "PYDAC_RHS_CSR_OPERATORS": False, "PYDAC_CACHE_SLICE_WRITE": True}
    # Once the model is loaded, the value it bound at import is what a solve used.
    import pydac.core.column_model as column_model
    monkeypatch.setattr(column_model, "_RHS_CSR_OPERATORS", True)
    assert p.software_identity()["solver_environment"]["PYDAC_RHS_CSR_OPERATORS"] is True


def test_sdist_leaves_out_the_development_documents():
    import tomllib
    root = Path(__file__).resolve().parents[1]
    sdist = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"]["sdist"]
    # docs/ is shipped, its development roadmaps and internal reports are not (F5D-01).
    assert "docs" in sdist["only-include"]
    assert {"docs/development", "docs/reports"} <= set(sdist["exclude"])
