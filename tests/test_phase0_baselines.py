from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from pydac.config import load_experiment
from pydac.inputs.case_io import case_fingerprint
from tests.tools.generate_v3_smoke_baseline import generate_observations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINES = PROJECT_ROOT / "tests" / "baselines"
CASES = PROJECT_ROOT / "tests" / "cases"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_nested_allclose(actual, expected, *, rtol, atol, path="value"):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_nested_allclose(
                actual[key],
                expected[key],
                rtol=rtol,
                atol=atol,
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, list):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=path)
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=path)
        return
    assert actual == expected, path


def _dynamic_tolerance(name: str, policy: dict) -> tuple[float, float]:
    absolute = policy["absolute_tolerance_by_quantity"]
    token = name.lower()
    if "time" in token:
        return 0.0, absolute["time_s"]
    if "mole_fraction" in token:
        return policy["relative_tolerance"], absolute["mole_fraction"]
    if "temperature" in token:
        return policy["relative_tolerance"], absolute["temperature_K"]
    if "pressure" in token:
        return policy["relative_tolerance"], absolute["pressure_Pa"]
    if token.startswith("q_") or "loading" in token:
        return policy["relative_tolerance"], absolute["loading_mol_kg"]
    if token.startswith("c_") or "concentration" in token:
        return policy["relative_tolerance"], absolute["concentration_mol_m3"]
    return policy["relative_tolerance"], 1e-12


# Derived case values (e.g. the steam volumetric flow) differ in the last digits across
# numerical environments, so this fingerprint baseline holds only in the qualification lock.
@pytest.mark.qualification
def test_v3_smoke_matches_phase0_baseline(tmp_path):
    baseline = _read_json(BASELINES / "v3_smoke_startup.json")
    policies = _read_json(BASELINES / "comparison_rules.json")["policies"]
    policy = policies[baseline["comparison_policy"]]
    case_path = PROJECT_ROOT / baseline["case"]["path"]

    assert _sha256(case_path) == baseline["case"]["sha256"]
    config = load_experiment(case_path)
    assert case_fingerprint(config) == baseline["case"]["fingerprint"]

    actual = generate_observations(config, output_folder=tmp_path)
    expected = baseline["observations"]
    assert actual["solver"]["success"] is True
    assert actual["solver"]["termination_reason"] == "final_time"
    assert actual["solver"]["implementation"] == expected["solver"]["implementation"]

    property_tolerance = policy["runtime_properties"]
    _assert_nested_allclose(
        actual["runtime_properties"],
        expected["runtime_properties"],
        rtol=property_tolerance["relative_tolerance"],
        atol=property_tolerance["absolute_tolerance"],
        path="runtime_properties",
    )

    dynamic_policy = policy["dynamic_outputs"]
    rtol, atol = _dynamic_tolerance("stored_time_s", dynamic_policy)
    np.testing.assert_allclose(actual["stored_time_s"], expected["stored_time_s"], rtol=rtol, atol=atol)

    for group in ("outlet_histories", "final_profiles"):
        assert set(actual[group]) == set(expected[group])
        for name, expected_values in expected[group].items():
            rtol, atol = _dynamic_tolerance(name, dynamic_policy)
            np.testing.assert_allclose(
                actual[group][name],
                expected_values,
                rtol=rtol,
                atol=atol,
                err_msg=f"{group}.{name}",
            )

    assert set(actual["final_field_summaries"]) == set(expected["final_field_summaries"])
    for field, expected_summary in expected["final_field_summaries"].items():
        actual_summary = actual["final_field_summaries"][field]
        assert actual_summary["shape"] == expected_summary["shape"]
        assert actual_summary["unit"] == expected_summary["unit"]
        rtol, atol = _dynamic_tolerance(field, dynamic_policy)
        for statistic in ("minimum", "maximum", "mean"):
            np.testing.assert_allclose(
                actual_summary[statistic],
                expected_summary[statistic],
                rtol=rtol,
                atol=atol,
                err_msg=f"final_field_summaries.{field}.{statistic}",
            )


def test_hardcoding_inventory_covers_current_audit_matches():
    inventory = _read_json(BASELINES / "hardcoding_inventory.json")
    entries = inventory["entries"]
    valid_phases = {1, 2, 3, 4, 6, 7, 8, 9, 10, "5A", "5B"}
    covered = set()
    for entry in entries:
        assert entry["owner"]
        assert entry["disposition"]
        assert entry["target_phases"]
        assert set(entry["target_phases"]) <= valid_phases
        for relative in entry["paths"]:
            assert (PROJECT_ROOT / relative).is_file(), relative
            covered.add(relative)

    audit = re.compile(
        r"\b(?:CO2|H2O|O2|N2|Ar|SPECIES)\b|y_2d|y_1d|y_scalars|"
        r"(?:\d+\s*\*\s*(?:self\.)?N)|Nc_2D|Nc_1D|Nc_gas|Nc_solid|n_integrated|Block indices"
    )
    matched = set()
    # The package lives under src/pydac; runners stay at the root as contributor launchers.
    for root_name in ("src", "runners"):
        for path in (PROJECT_ROOT / root_name).rglob("*.py"):
            if audit.search(path.read_text(encoding="utf-8")):
                matched.add(path.relative_to(PROJECT_ROOT).as_posix())

    assert matched <= covered, f"Unclassified hardcoding files: {sorted(matched - covered)}"


def test_comparison_rules_define_public_regression_boundary():
    policies = _read_json(BASELINES / "comparison_rules.json")["policies"]
    assert set(policies) == {"v3_regression"}
    assert policies["v3_regression"]["dynamic_outputs"]["relative_tolerance"] > 0.0
