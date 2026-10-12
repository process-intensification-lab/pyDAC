"""Regression checks for shared thermal, outlet and logging behavior."""
import contextlib
from copy import deepcopy
import io
import pickle
import sys

import numpy as np
import pytest

from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.utils.logger import log_file
from tests.test_profile_inventory_outlet import _prepare
from pydac.post_processing.profiles.cyclic_plotting import CyclicSteadyStateProfiles
from pydac.utils.postprocessing.outlet import outlet_mole_fractions


@pytest.mark.parametrize('correlation', ['dixon', 'specchia_1980'])
@pytest.mark.parametrize('override', [None, 20.])
def test_bed_wall_exchange_is_equal_and_opposite(correlation, override):
    cycle = CycleConfig.from_experiment(load_experiment('tests/cases/v3_smoke_startup.json', settings={
        'grid.nz': 6, 'grid.nr': 4, 'bed.design.wall_htc_model': correlation,
        'wall.bed_wall_htc_override_W_m2K': override}))
    model = ColumnModelFactory(cycle).build_column_model(step='startup', solver_options=cycle.solver_for_step(use_sparsity=False))
    assert model.lambda_rad / (model.R_b * model.inv_Bi_h) == pytest.approx(float(model.h_int), rel=1e-12)
    assert float(model.h_int) == pytest.approx(override if override is not None else model.resolved_transport.h_wall)
    bed = (330 + 12 * model.z_c[:, None] + 7 * model.r_c[None, :]**2) / model.T_ref
    wall = (302 + 3 * model.z_c) / model.T_ref
    bc = ({'a': 1, 'b': 0, 'd': 0}, {'a': model.inv_Bi_h, 'b': 1, 'd': wall.reshape(model.shape_Tw)})
    grad, offset = model.grad_rad_T_cache.update_bc(bc)
    rate = model.Div_Diff_rad_T.dot(grad.dot(bed.ravel()) + offset).reshape(bed.shape)
    bed_power = np.sum(model.V_cells * rate) * model.T_ref / model.tau_hyd
    face = (wall - model.inv_Bi_h * (model.wall_face_c_N * bed[:, -1] + model.wall_face_c_M * bed[:, -2])) / (1 + model.inv_Bi_h * model.wall_face_c_F)
    wall_power = np.sum(np.diff(model.z_f) * model.L_b * 2 * np.pi * model.R_b * model.h_w * (face-wall) * model.T_ref)
    assert wall_power > 0
    assert bed_power == pytest.approx(-wall_power, rel=1e-11, abs=1e-10)


@pytest.mark.parametrize('nr, field, needs', [
    (1, 'wall.bed_wall_htc_override_W_m2K', '2D grid'),
    (4, 'wall.overall_htc_override_W_m2K', '1D grid'),
])
def test_a_wall_override_for_the_other_dimension_is_refused(nr, field, needs):
    from pydac.config.errors import ConfigurationError
    with pytest.raises(ConfigurationError, match=f"read only on a {needs}"):
        load_experiment('tests/cases/v3_smoke_startup.json', settings={'grid.nr': nr, field: 20.})


def cyclic_reader():
    cyclic = CyclicSteadyStateProfiles.__new__(CyclicSteadyStateProfiles)
    cyclic.layout_signature = None
    cyclic.gas_species = cyclic.adsorbates = ()
    cyclic.field_specs = dict(cyclic.FIELD_SPECS)
    cyclic.default_plot_fields = tuple(cyclic.DEFAULT_PLOT_FIELDS)
    cyclic.is_1d = None
    cyclic._mixed_cup_zero_den_warned = False
    return cyclic


@pytest.mark.parametrize('nr', [1, 5])
@pytest.mark.parametrize('step', ['startup', 'blow', 'heat', 'cool', 'press'])
def test_cyclic_native_stream_semantics_and_no_mutation(tmp_path, nr, step):
    model, solver, _ = _prepare(tmp_path, nr)
    sol = solver.sol
    before = pickle.dumps(sol)
    result = cyclic_reader()._extract_outlet_profiles(sol, step)
    assert pickle.dumps(sol) == before
    if step in ['cool', 'press']:
        np.testing.assert_array_equal(result['u_out'], 0.)
        assert np.all(np.isfinite(result['y_CO2']))
    else:
        for species, fractions in outlet_mole_fractions(sol).items():
            np.testing.assert_array_equal(result['y_'+species], fractions)
            np.testing.assert_array_equal(result['ppm_'+species], fractions*1e6)
        np.testing.assert_allclose(result['u_out'], np.average(sol['outlet_face']['velocity_m_s'], weights=model.A_rings, axis=0))
        for values in sol['outlet_face']['species_molar_flow_mol_s'].values():
            values[:, 0] = 0.
            values[:, 1] = -abs(values[:, 1])
        undefined = cyclic_reader()._extract_outlet_profiles(sol, step)
        assert np.all(np.isnan(undefined['y_CO2']))
        np.testing.assert_array_equal(undefined['T_bed'], result['T_bed'])
        np.testing.assert_array_equal(undefined['pressure'], result['pressure'])


def test_logger_nesting_capture_stderr_and_exception_restoration(tmp_path):
    captured = io.StringIO()
    outer, inner = tmp_path/'outer.log', tmp_path/'inner.log'
    with contextlib.redirect_stdout(captured):
        with log_file(str(outer), also_stdout=True):
            print('outer')
            with pytest.raises(RuntimeError):
                with log_file(str(inner), also_stdout=True):
                    print('inner')
                    print('error', file=sys.stderr)
                    raise RuntimeError('test')
            print('restored')
        assert sys.stdout is captured
    assert outer.read_text() == captured.getvalue() == 'outer\ninner\nerror\nrestored\n'
    assert inner.read_text() == 'inner\nerror\n'
    with contextlib.redirect_stdout(captured):
        with log_file(str(inner), append=True, also_stdout=False): print('private')
    assert inner.read_text().endswith('error\nprivate\n')
    assert 'private' not in captured.getvalue()
