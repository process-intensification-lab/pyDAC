from __future__ import annotations

import contextlib
import io

import numpy as np

import pydac.core.column_model as column_module
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.core.isotherm_models import TothIsotherm
import pydac.core.pymrm_caching as caching_module


CASE = "tests/cases/v3_smoke_startup.json"
RHS_OPERATOR_NAMES = (
    "Div_ax_ctot",
    "Div_ax_ci",
    "Div_rad_ci",
    "Div_ax_T",
    "Div_ax_Tw",
    "Disp_flux_ax",
    "Disp_flux_rad",
    "Cond_flux_ax",
    "Cond_flux_ax_Tw",
    "Grad_ax_p",
    "Div_Diff_rad_T",
)


def build_model(*, nz=4, nr=1):
    config = load_experiment(
        CASE,
        settings={"grid.nz": nz, "grid.nr": nr},
    )
    cycle = CycleConfig.from_experiment(config)
    with contextlib.redirect_stdout(io.StringIO()):
        return ColumnModelFactory(cycle).build_column_model(
            step="startup",
            initial_state=None,
            solver_options=cycle.solver_for_step(
                use_sparsity=False,
                compute_numerical_sparsity=False,
            ),
        )


def complete_initial_state(model):
    state = np.zeros(model.state_layout.state_size)
    state[: model.state_layout.pde_size] = model.get_initial_state()
    return state


def legacy_co_isotherm_wadst(model, p, T, q_co=None):
    """Reference wet-Toth path, including the discarded wet loading."""
    q_co = 0.0 if q_co is None else q_co

    q_eq_dry, _, _ = model.dry_isotherm(p=p, T=T)
    q_eq_dry = np.clip(q_eq_dry, 1e-32, model.qm)

    _, b_wet, th_wet = model._toth_terms(
        p=p,
        T=T,
        qm=model.qm_wet,
        b0=model.b0_wet,
        DH0=model.DH0_wet,
        T0=model.T0_wet,
        th0=model.th0_wet,
        alpha=model.alpha_wet,
        b_fact=model.Young_b_fact_wet,
    )

    p_safe = np.maximum(p, 1e-32)
    log_num = (
        np.log(model.qm_wet)
        + np.log(np.maximum(b_wet, 1e-32))
        + np.log(p_safe)
    )
    log_denom = (1.0 / th_wet) * np.log1p((b_wet * p_safe) ** th_wet)
    q_eq_wet = np.exp(log_num - log_denom)
    q_eq_wet = np.clip(q_eq_wet, 1e-32, None)

    weight_wet = np.exp(-model.A_fact / np.maximum(q_co, 1e-3))
    q_eq = (1.0 - weight_wet) * q_eq_dry + weight_wet * q_eq_wet
    return q_eq, b_wet, th_wet


def test_contiguous_cache_write_detection_and_off_switch(monkeypatch):
    contiguous = np.array([3, 4, 5, 6], dtype=np.intp)
    noncontiguous = np.array([3, 4, 6], dtype=np.intp)

    monkeypatch.setattr(caching_module, "_CACHE_SLICE_WRITE", True)
    assert caching_module._contiguous_slice(contiguous) == slice(3, 7)
    assert caching_module._contiguous_slice(noncontiguous) is None
    assert caching_module._contiguous_slice(np.array([], dtype=np.intp)) is None

    monkeypatch.setattr(caching_module, "_CACHE_SLICE_WRITE", False)
    assert caching_module._contiguous_slice(contiguous) is None


def test_wet_toth_dead_work_removal_is_bit_identical():
    model = TothIsotherm(data="Cleaned-Toth")
    pressure = np.logspace(-30, 4, 5000)
    temperature = np.linspace(283.0, 410.0, pressure.size)
    water_loading = np.logspace(-6, 1, pressure.size)

    before = legacy_co_isotherm_wadst(
        model,
        pressure,
        temperature,
        water_loading,
    )
    after = model.co_isotherm_wadst(
        pressure,
        temperature,
        water_loading,
    )

    for before_value, after_value in zip(before, after):
        np.testing.assert_array_equal(after_value, before_value)


def test_combined_tier0_rhs_is_bit_identical(monkeypatch):
    optimized_wadst = TothIsotherm.co_isotherm_wadst

    monkeypatch.setattr(column_module, "_RHS_CSR_OPERATORS", False)
    monkeypatch.setattr(caching_module, "_CACHE_SLICE_WRITE", False)
    baseline = build_model(nz=10, nr=10)

    monkeypatch.setattr(column_module, "_RHS_CSR_OPERATORS", True)
    monkeypatch.setattr(caching_module, "_CACHE_SLICE_WRITE", True)
    optimized = build_model(nz=10, nr=10)

    assert all(getattr(baseline, name).format == "csc" for name in RHS_OPERATOR_NAMES)
    assert all(getattr(optimized, name).format == "csr" for name in RHS_OPERATOR_NAMES)
    assert baseline.conv_ax_species_cache._int_write is None
    assert baseline.conv_ax_cache._int_write is None
    assert isinstance(optimized.conv_ax_species_cache._int_write, slice)
    assert isinstance(optimized.conv_ax_cache._int_write, slice)

    initial = complete_initial_state(baseline)
    np.testing.assert_array_equal(initial, complete_initial_state(optimized))
    structured = initial.copy()
    rng = np.random.default_rng(20260821)
    for field in baseline.state_layout.fields:
        if field.storage == "y_scalars":
            continue
        values = structured[field.state_slice]
        values *= rng.uniform(0.85, 1.15, size=values.shape)

    for state in (initial, structured):
        monkeypatch.setattr(TothIsotherm, "co_isotherm_wadst", legacy_co_isotherm_wadst)
        before = baseline.pde_system_kpis(0.0, state).copy()
        monkeypatch.setattr(TothIsotherm, "co_isotherm_wadst", optimized_wadst)
        after = optimized.pde_system_kpis(0.0, state).copy()
        np.testing.assert_array_equal(after, before)


def test_integrated_mass_path_passes_only_outlet_concentrations():
    model = build_model(nz=4, nr=2)
    state = complete_initial_state(model)
    observed_shapes = []
    compute_mass_fluxes = model._compute_outlet_mass_fluxes

    def record_shape(concentrations, velocity):
        observed_shapes.append(np.asarray(concentrations).shape)
        return compute_mass_fluxes(concentrations, velocity)

    model._compute_outlet_mass_fluxes = record_shape
    model.pde_system_kpis(0.0, state)

    assert observed_shapes == [(model.Nr, model.Nc_solid)]
