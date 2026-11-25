"""Mixture diffusivities (Blanc's law) when the reference gas is a single species.

Blanc's law D_i = (1 - y_i) / sum_j (y_j / D_ij) is 0/0 for a species alone in the gas.
Only there the model uses its limit with the other species in equal shares,
1 / mean_j(1 / D_ij), which is D_ij exactly for a binary; every value the law can
evaluate is kept as it is. Before this, a pure feed (e.g. N2 only) gave NaN
dispersion and crashed in the property summary.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pytest

from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.core.solver import Solver
from tests.neutral_packed_bed import neutral_profile


def _startup(profile=None, settings=None):
    config = load_experiment(default_profile=profile, settings={"grid.nz": 6, "grid.nr": 1, **(settings or {})})
    cycle = CycleConfig.from_experiment(config)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # no 0/0 warning either
        return ColumnModelFactory(cycle).build_column_model(
            step="startup", solver_options=cycle.solver_for_step(use_sparsity=False))


def _blanc(model, y):
    """The law itself, evaluated independently of the model code."""
    D = model.D_binary
    return np.array([(1 - y[i]) / sum(y[j] / D[i, j] for j in range(len(y)) if j != i)
                     for i in range(len(y))])


def _pure_n2_binary():
    profile = neutral_profile()
    profile["simulation"]["cycle"]["steps"]["startup"]["composition"]["co2_feed_mole_fraction"] = 0.0
    return profile


def test_a_pure_species_in_a_binary_takes_the_binary_diffusivity():
    model = _startup(_pure_n2_binary())
    assert np.asarray(model.y_feed).ravel().tolist() == [0.0, 1.0]
    assert model.D_m[1] == model.D_binary[0, 1]
    assert model.D_m[0] == model.D_binary[0, 1]          # CO2 traces in N2: the law, unchanged
    assert np.all(np.isfinite(model.Sc)) and np.all(np.isfinite(model.Sh))


def test_a_pure_species_in_a_multicomponent_gas_takes_the_equal_share_limit():
    model = _startup(settings={
        "steps.startup.composition.co2_feed_mole_fraction": 0.0,
        "steps.startup.composition.relative_humidity": 0.0,
        "steps.startup.composition.o2_feed_mole_fraction": 0.0,
        "steps.startup.composition.n2_feed_mole_fraction": 1.0,
    })
    i = model.gas_index["N2"]
    others = [j for j in range(model.Nc_gas) if j != i]
    expected = 1.0 / np.mean([1.0 / model.D_binary[i, j] for j in others])
    assert model.D_m[i] == pytest.approx(expected, rel=1e-15)
    # The equal-share limit is what the law gives along that direction, for any share.
    for eps in (1e-3, 1e-6):
        y = np.full(model.Nc_gas, eps)
        y[i] = 1.0 - (model.Nc_gas - 1) * eps
        assert _blanc(model, y)[i] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("co2", [1e-9, 0.1])
def test_every_composition_the_law_can_evaluate_keeps_its_value(co2):
    profile = neutral_profile()
    profile["simulation"]["cycle"]["steps"]["startup"]["composition"]["co2_feed_mole_fraction"] = co2
    model = _startup(profile)
    y = np.asarray(model.y_feed, dtype=float).ravel()
    inv = np.where(np.isinf(model.D_binary), 0.0, 1.0 / model.D_binary)
    np.testing.assert_array_equal(model.D_m, (1.0 - y) / (inv @ y))   # bit for bit


def test_a_pure_n2_feed_runs_to_a_finite_solution(tmp_path):
    profile = _pure_n2_binary()
    steps = profile["simulation"]["cycle"]["steps"]["startup"]
    steps["composition"]["co2_initial_mole_fraction"] = 0.1     # a bed with CO2 to purge
    steps["horizon"].update(nominal_time_s=60.0, event_max_time_s=60.0)
    model = _startup(profile)
    solver = Solver(model, base_folder=str(tmp_path))
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_cvode()
    assert solver.sol["cvode"]["success"]
    assert np.all(np.isfinite(solver.sol["y"]))


def test_non_finite_diffusivities_are_refused_with_the_step_and_the_gas(monkeypatch):
    from pydac.core.column_model import ColumnModelMoL

    monkeypatch.setattr(ColumnModelMoL, "_Fuller_binary_diff", lambda self, MW, v, T, p: float("nan"))
    with pytest.raises(ValueError, match=r"Step 'startup': the mixture diffusivities .* not finite and positive"):
        _startup(neutral_profile())
