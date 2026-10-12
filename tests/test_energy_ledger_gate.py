"""The condenser-split ledger gate.

`energy.condenser` records the whole duty and `energy.condenser.ambient` carries
the part cooling water takes for free, so only the remainder is charged at the
chiller COP.

The fallback for a missing `energy.condenser.ambient` deliberately reproduces the *old*
charge.  That is right for reading an archived result, because the alternative would
make an un-backfilled file pay more than it actually did.  Inside a campaign it is a
silent trap: a run that fails to record the metric yields an old-ledger front and says
nothing.  So the gate is strict by default and the archive path is the one that has to
be asked for.

Two properties, and the second is the load-bearing one:

* reading a legacy cycle as if it were current raises instead of falling back;
* inside NSGA-II that raise **kills the campaign**, rather than being demoted to one
  failed candidate.  The split is absent or present for the whole campaign, never for
  one candidate, so demotion would spend the entire budget producing an old-ledger
  front indistinguishable from a physics failure.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import pydac.utils.postprocessing.context_schema as context_schema
from pydac.post_processing.context.simulation_context import SimulationContext
from pydac.post_processing.kpi.cycle_kpis import (
    CONDENSER_ABOVE_AMBIENT_METRIC,
    CycleKPIs,
    LedgerMetricMissing,
)
from pydac.post_processing.metrics.cycle_data import CycleData
from pydac.post_processing.metrics.cycle_metrics import (
    CYCLE_METRIC_COLUMNS,
    CycleMetrics,
)
from pydac.utils.postprocessing.normalization import STEP_KEY_TO_CYCLE_STEP


CYCLE_LABEL = "cycle_1"
STEP_KEYS = ("ads", "s_des")
SPECIES = ("CO2", "N2")

# Placeholder context values, by declared type.  Nothing here is read by the ledger
# gate; the context only exists because `CycleData` requires a real one.
_TEXT_CONTEXT_VALUES = {
    "bed.architecture": "packed_bed",
    "bed.models.heat_transfer": "dixon",
    "bed.models.pressure_drop": "ergun",
    "bed.models.dispersion": "wakao",
    "bed.closure.gas_composition_source": "feed",
    "bed.closure.dispersion_composition_source": "feed",
}

# Numeric context parameters the schema constrains to [0, 1); everything else takes the
# generic placeholder below.
_FRACTION_CONTEXT_IDS = frozenset(
    {
        "particle.porosity",
        "bed.resolved.eps_flow",
        "bed.resolved.eps_total",
        "bed.design.eps_b",
    }
)


def _simulation_context() -> SimulationContext:
    """Build the smallest context the current packed-bed schema accepts.

    Driven off the schema definitions rather than a hand-written row list, so a new
    required parameter does not silently turn this module into a skip.
    """
    static_definitions = {
        **context_schema.COMMON_STATIC_BY_ID,
        **context_schema.ARCHITECTURE_STATIC_BY_ID["packed_bed"],
    }
    step_definitions = context_schema.step_definitions_by_id(SPECIES)

    rows = []
    for definition in static_definitions.values():
        rows.append(_context_row(definition, context_schema.STATIC_STEP))
    for step_key in STEP_KEYS:
        for definition in step_definitions.values():
            rows.append(_context_row(definition, step_key))
    return SimulationContext(
        pd.DataFrame(rows, columns=list(context_schema.SIMULATION_CONTEXT_COLUMNS))
    )


def _context_row(definition, step: str) -> dict[str, object]:
    if definition.value_type == "text":
        value = _TEXT_CONTEXT_VALUES[definition.parameter_id]
    elif definition.parameter_id in _FRACTION_CONTEXT_IDS:
        value = 0.4
    else:
        value = 1.0
    return {
        "Category": definition.category,
        "Parameter ID": definition.parameter_id,
        "Step": step,
        "Value": value,
        "Unit": definition.unit,
    }


def _metric_row(
    step_key: str,
    metric_id: str,
    value: float,
    *,
    species: str = "system",
    unit: str = "MJ",
    basis: str | None = None,
) -> dict[str, object]:
    return {
        "Cycle": CYCLE_LABEL,
        "Step Index": STEP_KEYS.index(step_key),
        "Step Key": step_key,
        "Step": STEP_KEY_TO_CYCLE_STEP[step_key],
        "Species": species,
        "Metric ID": metric_id,
        "Value": value,
        "Unit": unit,
        "Basis": basis,
    }


def _cycle_kpis(*, record_split: bool, allow_legacy_ledger: bool = False) -> CycleKPIs:
    """One two-step cycle: adsorption with no condenser, desorption with one.

    `record_split` decides whether the desorption step also carries
    `energy.condenser.ambient`, i.e. whether it is a current or a pre-re-pricing result.
    """
    rows = [
        # Adsorption never runs the condenser.  It exists here to prove the gate does
        # not fire on a step that has no duty to split in the first place.
        _metric_row("ads", "energy.compression", 1.0),
        _metric_row("s_des", "energy.condenser", -20.0),
        # The aggregate breakdown normalises by cycle product, so the desorption step
        # has to discharge something for it to reach the condenser column at all.
        _metric_row(
            "s_des",
            "mass.after.vacuum",
            0.5,
            species="CO2",
            unit="kg",
        ),
    ]
    if record_split:
        rows.append(_metric_row("s_des", CONDENSER_ABOVE_AMBIENT_METRIC, -15.0))

    metrics = CycleMetrics(pd.DataFrame(rows, columns=list(CYCLE_METRIC_COLUMNS)))
    return CycleKPIs(
        CycleData(metrics, _simulation_context()),
        allow_legacy_ledger=allow_legacy_ledger,
    )


def test_a_legacy_cycle_raises_instead_of_falling_back_to_the_old_charge():
    kpis = _cycle_kpis(record_split=False)
    with pytest.raises(LedgerMetricMissing, match="2026-09-08 condenser split"):
        kpis.condenser_duty_above_ambient("s_des")


def test_the_gate_names_the_way_out_rather_than_just_failing():
    kpis = _cycle_kpis(record_split=False)
    with pytest.raises(LedgerMetricMissing, match="allow_legacy_ledger=True"):
        kpis.condenser_duty_above_ambient("s_des")


def test_a_step_with_no_condenser_duty_is_not_gated():
    """`None` here means "nothing to split", not "recorded on the old ledger"."""
    kpis = _cycle_kpis(record_split=False)
    assert kpis.condenser_duty_above_ambient("ads") is None


def test_a_recorded_split_passes_the_gate_and_is_returned_verbatim():
    kpis = _cycle_kpis(record_split=True)
    assert kpis.condenser_duty_above_ambient("s_des") == -15.0


def test_reading_an_archive_is_opt_in_and_reproduces_the_old_charge():
    legacy = _cycle_kpis(record_split=False, allow_legacy_ledger=True)
    assert legacy.condenser_duty_above_ambient("s_des") is None

    # The unsplit branch is the pre-2026-09-08 one: the whole duty at the
    # refrigeration COP.  Asserted against the KPI object's own conversion factors so
    # this stays a statement about which branch ran, not about their numeric values.
    _, refrigeration_cop = legacy.thermodynamic_conversion_factors(None)
    assert legacy.convert_energy_to_equivalent_work(
        "energy.condenser", -20.0, absolute=True
    ) == pytest.approx(20.0 / refrigeration_cop)


def test_the_split_charges_only_the_remainder_and_costs_less_than_the_old_ledger():
    kpis = _cycle_kpis(record_split=True)
    legacy = _cycle_kpis(record_split=False, allow_legacy_ledger=True)

    charged = kpis.convert_energy_to_equivalent_work(
        "energy.condenser", -20.0, absolute=True, above_ambient=-15.0
    )
    assert charged == pytest.approx(5.0 / kpis.chiller_cop(None))
    assert charged < legacy.convert_energy_to_equivalent_work(
        "energy.condenser", -20.0, absolute=True
    )


def test_the_aggregate_equivalent_work_path_is_gated_too():
    """The gate has to hold where the optimiser actually reads, not only at the leaf."""
    kpis = _cycle_kpis(record_split=False)
    with pytest.raises(LedgerMetricMissing):
        kpis.energy_breakdown(equivalent=True, absolute=True, specific=False)


def test_the_condenser_cooling_split_is_gated_too():
    kpis = _cycle_kpis(record_split=False)
    with pytest.raises(LedgerMetricMissing):
        kpis.condenser_cooling_split()


# ── the campaign-level half: the gate must not be demoted to a failed candidate ──


def _nsga2_candidate_harness(monkeypatch, nsga2, tmp_path, *, cycle_kpis_factory):
    """Drive one NSGA-II evaluation with everything but the KPI read stubbed out."""
    from pydac.utils.solver.composition_domain import CompositionDomainGuard

    run_config = SimpleNamespace(
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

    monkeypatch.setattr(
        nsga2,
        "decision_vector_to_model_inputs",
        lambda *args, **kwargs: (
            {"t_ads": 900.0},
            {"v_feed": 0.01, "v_steam": 0.001},
        ),
    )
    monkeypatch.setattr(
        nsga2, "_format_candidate_summary", lambda *args, **kwargs: "candidate"
    )
    monkeypatch.setattr(nsga2, "operating_step_settings", lambda settings: {})
    from pydac.config import load_experiment as real_load_experiment
    provenance_case = real_load_experiment("tests/cases/v3_smoke_startup.json")
    monkeypatch.setattr(nsga2, "load_experiment", lambda *args, **kwargs: provenance_case)
    monkeypatch.setattr(
        nsga2,
        "write_resolved_config",
        lambda _experiment, path: Path(path).write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        nsga2,
        "run_css_simulation",
        lambda *, base_folder, **kwargs: {
            "css_reached": True,
            "cycles_completed": 2,
            "css_cycle_idx": 2,
            "results_base_folder": str(base_folder),
            "last_step_records": [],
        },
    )
    monkeypatch.setattr(
        nsga2, "CycleDataLoader", lambda *a, **k: SimpleNamespace(load=lambda: object())
    )
    monkeypatch.setattr(nsga2, "CycleKPIs", cycle_kpis_factory)
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
    return run_config


def test_the_ledger_gate_kills_the_campaign_instead_of_failing_one_candidate(
    monkeypatch, tmp_path
):
    """The ledger gate must abort the campaign.

    An ordinary evaluator failure is caught and returned as a rejected candidate --
    that is right for a CVODE failure, which is a property of the point being
    evaluated.  The ledger gate is a property of the *code*, identical for every
    candidate, so swallowing it would burn a 100 x 100 budget producing an old-ledger
    front that looks exactly like a hard search space.  It has to escape the worker.
    """
    pytest.importorskip("pymoo")  # NSGA-II requires pymoo.
    from pydac.workflows import orchestrator_NSGA2 as nsga2

    def raise_ledger_gate(_cycle_data):
        raise LedgerMetricMissing("no energy.condenser.ambient in this cycle")

    run_config = _nsga2_candidate_harness(
        monkeypatch, nsga2, tmp_path, cycle_kpis_factory=raise_ledger_gate
    )

    with pytest.raises(LedgerMetricMissing):
        nsga2.evaluate_pydac_candidate(
            [900.0, 60.0, 300.0, 3000.0, 0.01, 0.001],
            run_config=run_config,
            eval_folder=str(tmp_path / "candidate"),
        )


def test_an_ordinary_failure_is_still_demoted_to_a_rejected_candidate(
    monkeypatch, tmp_path
):
    """The guard above must not have turned every failure into a campaign abort."""
    pytest.importorskip("pymoo")  # NSGA-II requires pymoo.
    from pydac.workflows import orchestrator_NSGA2 as nsga2

    def raise_ordinary(_cycle_data):
        raise RuntimeError("an ordinary evaluation failure")

    run_config = _nsga2_candidate_harness(
        monkeypatch, nsga2, tmp_path, cycle_kpis_factory=raise_ordinary
    )

    result = nsga2.evaluate_pydac_candidate(
        [900.0, 60.0, 300.0, 3000.0, 0.01, 0.001],
        run_config=run_config,
        eval_folder=str(tmp_path / "candidate"),
    )

    assert result["failed"] is True
    assert "an ordinary evaluation failure" in result["error"]


def test_the_gate_error_survives_the_trip_out_of_a_worker_process():
    """`mp.Pool` re-raises by pickling; an unpicklable gate would abort as something else."""
    original = LedgerMetricMissing("no energy.condenser.ambient in this cycle")
    restored = pickle.loads(pickle.dumps(original))

    assert isinstance(restored, LedgerMetricMissing)
    assert str(restored) == str(original)
