"""Schema-11 output schedule, one-way outlet and product accounting contracts."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
from pydac import load_case
from pydac.config import load_experiment, ConfigurationError
from pydac.config.cycle import CycleConfig
from pydac.config.migrations import pre_output_schedule_v10_vocabulary, upgrade_experiment_document
from pydac.config.serialization import experiment_to_document
from pydac.core.ads_outlet import AdsOutletValve, check_ads_outlet
from pydac.core.column_model import ColumnModelMoL
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.core.solver import Solver
from pydac.post_processing.profiles.step_post_processing import StepPostProcessor
from pydac.utils.output_schedule import OutputSchedule, SCHEDULE_DEFAULTS, validate_schedule_spec
from tests.neutral_packed_bed import neutral_profile


def test_j24_schedule_roundtrip_and_switch_neighborhood():
    a=OutputSchedule.dense_start(900,transition_times_s=(1.,40.))
    b=OutputSchedule(**json.loads(json.dumps(a.to_dict())))
    assert np.array_equal(a.resolve(900),b.resolve(900))
    t=b.resolve(900)
    assert {0.,1.,40.,900.}.issubset(t)
    assert ((t>39.99)&(t<40)).sum()>1
    assert ((t>40)&(t<40.01)).sum()>1


@pytest.mark.parametrize('times',[(0,0,10),(1,10),(0,9),(0,float('nan'),10),(0,11,10)])
def test_schedule_invalid(times):
    with pytest.raises(ValueError):OutputSchedule(times).resolve(10)


@pytest.mark.parametrize('changes',[{'dense_start_s':1}, {'dense_start_points':4},
 {'first_output_s':1e-7}, {'dense_start_s':1,'dense_start_points':True},
 {'dense_start_s':1,'dense_start_points':2.1}, {'dense_start_s':1,'dense_start_points':1},
 {'dense_start_s':1,'dense_start_points':3,'first_output_s':2},
 {'dense_start_s':11,'dense_start_points':3}, {'max_output_spacing_s':0},
 {'max_output_spacing_s':float('inf')}, {'max_output_spacing_s':float('nan')}])
def test_schedule_spec_refuses(changes):
    with pytest.raises(ValueError):validate_schedule_spec(dict(SCHEDULE_DEFAULTS,**changes),horizon_s=10,nt=10)


def model(profile=None):
    profile=neutral_profile() if profile is None else profile
    profile['simulation']['grid'].update(nz=6,nr=1,nt=101)
    with redirect_stdout(StringIO()):
        cfg=load_experiment(default_profile=profile)
        cycle=CycleConfig.from_experiment(cfg)
        pb=ColumnModelFactory(cycle).build_column_model(step='startup',solver_options=cycle.solver_for_step())
    return pb


def test_schema_schedule_union_and_null_exact_path(tmp_path):
    pb=model();pb.t_end=10.;pb.t_end_max=10.;pb.output_schedule_spec=dict(SCHEDULE_DEFAULTS)
    plain=Solver(pb,base_folder=str(tmp_path))
    assert np.array_equal(plain.t_eval,np.linspace(0,pb.t_end/pb.tau_hyd,pb.Nt))
    pb.output_schedule_spec=dict(SCHEDULE_DEFAULTS,dense_start_s=1.,dense_start_points=101,max_output_spacing_s=.7)
    dense=Solver(pb,base_folder=str(tmp_path));t=dense.t_eval*pb.tau_hyd
    assert np.all(np.isin(plain.t_eval,dense.t_eval))
    assert np.diff(t).max()<=.7+1e-10
    assert np.all(np.isin(np.geomspace(1e-6,1,101)/pb.tau_hyd,dense.t_eval))
    assert t[0]==0 and t[-1]==pytest.approx(pb.t_end)
    state=dense._initial_state();dense.sol={'y':np.column_stack([state,state]),'t':np.array([0.,1.])}
    dense.prepare_results();meta=dense.sol['extra']['output_schedule']
    assert meta['version']==1 and meta['requested_count']==len(dense.t_eval)
    assert meta['spec']==pb.output_schedule_spec
    assert pickle.loads(pickle.dumps(dense.sol))['extra']['output_schedule']==meta


@pytest.mark.parametrize('version',range(1,11))
@pytest.mark.parametrize('field',['schedule','valve'])
def test_earlier_schema_refuses_new_names_even_null(version,field):
    doc={'schema_version':version,'simulation':{'grid':{},'cycle':{'steps':{'ads':{'flow':{}}}}}}
    if field=='schedule':doc['simulation']['grid']['output_schedule']=None
    else:doc['simulation']['cycle']['steps']['ads']['flow']['ads_outlet_overpressure_Pa']=None
    with pytest.raises(ConfigurationError,match='is new in pyDAC 3.2.0; set `schema_version` to 11'):load_experiment(doc)


def test_schema10_preserves_blower_and_pump():
    cfg=load_case('tests/cases/v3_smoke_startup.json',settings={
      'grid.nr':1,'steps.press.flow.press_inlet_model':'feed_blower',
      'steps.press.flow.feed_blower_flow_m3_s':1e-4,
      'steps.blow.flow.vacuum_outlet_model':'vacuum_pump','steps.blow.flow.vacuum_pump_speed_m3_s':1e-4})
    doc=experiment_to_document(cfg);old=pre_output_schedule_v10_vocabulary(doc)
    assert upgrade_experiment_document(old)==doc
    assert load_case(old)==cfg


@pytest.mark.parametrize('step,nr',[('blow',1),('heat',1),('press',1),('cool',1),('ads',2),('startup',2)])
def test_valve_scope(step,nr):
    with pytest.raises(ValueError):check_ads_outlet(step=step,is_1d=nr==1,overpressure_Pa=100.,feed_flow_m3_s=.01)


@pytest.mark.parametrize('value',[0,-1,True,float('nan'),float('inf')])
def test_valve_bad_pressure(value):
    with pytest.raises(ValueError):check_ads_outlet(step='ads',is_1d=True,overpressure_Pa=value,feed_flow_m3_s=.01)


def test_valve_units_calibration_replay_and_saved_reconstruction(tmp_path):
    profile=neutral_profile();profile['simulation']['cycle']['steps']['startup']['flow']['ads_outlet_overpressure_Pa']=100.
    pb=model(profile);law=pb.ads_outlet_valve
    assert law.conductance_m_s_Pa==pb.vol_flowrate/pb.inlet_area/100.
    np.testing.assert_allclose(law.outlet_velocity(np.array([.99,1,1.001])*pb.p_out/pb.p_ref)*pb.u_ref,
                               [0,0,law.conductance_m_s_Pa*pb.p_out*.001],atol=1e-14)
    solver=Solver(pb,base_folder=str(tmp_path));state=solver._initial_state()
    # Isobaric overpressure and underpressure, with the actual RHS for comparison.
    states=[]
    for factor in (.99,1.001):
        y=state.copy();field=pb.state_layout.field('c_tot');y[field.state_slice]*=factor
        pb.pde_system_kpis(0,y)
        states.append(y)
        assert pb.u_ax_single[-1,0,0]>=0
    solver.sol={'y':np.column_stack(states),'t':np.array([0.,1.])};solver.prepare_results()
    metadata=solver.sol['extra']['ads_outlet_valve'];assert metadata['law']=='one_way_linear'
    restored=pickle.loads(pickle.dumps(solver.sol));pb.ads_outlet_valve=None
    solver.sol=restored
    with redirect_stdout(StringIO()):pp=StepPostProcessor(pb,solver)
    np.testing.assert_allclose(pp.u_ax[-1],restored['outlet_face']['velocity_m_s'],atol=1e-8,rtol=0)


@pytest.mark.parametrize('step',['blow','heat','w_purge','w_des'])
@pytest.mark.parametrize('collected',[False,True])
def test_signed_bed_product_inerts_and_vent(step,collected):
    pp=StepPostProcessor.__new__(StepPostProcessor);pp.step=step
    pp.pb=SimpleNamespace(mass_flow_in_i=np.zeros(2),MW_CO2=.04401)
    pp.gas_species=('CO2','N2');pp.time=np.array([0.,1.]);pp.molecular_weight_by_species={'CO2':.04401,'N2':.0280134}
    pp.molar_flow_out_by_species={'CO2':np.array([1.,1.]),'N2':np.array([2.,2.])};pp.n_CO2_out=np.ones(2)
    pp._get_flowrates=lambda:None;pp._outlet_settings=lambda:SimpleNamespace(collect_product=collected)
    pp.get_energies=lambda:(1.,2.,3.)
    pp.get_mass_balance_summary=lambda:{'CO2':{'delta_state':-.1,'delta_integrated':-.11},'N2':{'delta_state':.2,'delta_integrated':.21}}
    pp._require_species=lambda *args:None
    records,_,_=pp._outlet_metric_records({'CO2':{'change':-.11},'N2':{'change':.21}})
    values={(r.definition.metric_id,r.species):r.value for r in records}
    mid='mass.after.vacuum' if collected else 'mass.vented'
    assert values[mid,'CO2']==.1 and values[mid,'N2']==-.2
    for sp in pp.gas_species:
        assert ('mass.outlet.saved.simpson',sp) in values
        assert ('mass.outlet.accumulator',sp) in values


def test_wet_collected_separator_path_unchanged():
    pp=StepPostProcessor.__new__(StepPostProcessor);pp.step='w_des';pp.gas_species=('CO2','H2O','N2')
    pp._require_species=lambda *args:None
    pp._outlet_settings=lambda:SimpleNamespace(collect_product=True,condenser=False,condenser_temperature_K=278.15)
    pp.run_vacuum_separator=lambda **kw:(.3,.4);pp.get_energies=lambda:(0.,2.,0.)
    pp._closed_inlet_accounting=lambda:(_ for _ in ()).throw(AssertionError('wet product must retain separator'))
    rec,e,_=pp._outlet_metric_records({'N2':{'change':-.5}})
    assert [r.value for r in rec]==[.3,.4,.5] and e==2.


def power_model():
    return SimpleNamespace(_area_rings=np.array([1.]),_area_rings_sum=1.,_compression_prefactor=5.,
      _energy_poly_exponent=1/3.5,_energy_gamma_ratio=3.5,p_amb=1e5,p_feed=1e5,
      feed_blower=None,vacuum_manifold=None,_vacuum_eta_pressure_scale=19.55/101325.,
      _vacuum_eta_prefactor=.8*19.55/101325.,_jacket_power_constant=0.,h_ext=0.,_jacket_quad_weights=np.array([1.]))


@pytest.mark.parametrize('u',[-2.,0.,2.])
@pytest.mark.parametrize('pin',[9e4,1e5,1.1e5])
@pytest.mark.parametrize('pout',[9e4,1e5,1.1e5])
def test_clamps_sign_and_exact_nonnegative_legacy_integrands(u,pin,pout):
    m=power_model();a,b,_=ColumnModelMoL._compute_instantaneous_energies(m,[pin],[pout],[u],[u],[300.])
    assert a>=0 and b>=0
    rc=(pin/pout)**m._energy_poly_exponent-1
    rv=(m.p_amb/pout)**m._energy_poly_exponent-1
    if u>=0 and rc>=0:
        old=m._compression_prefactor*pin*u;old*=rc
        assert a==old
    if u>=0 and rv>=0:
        eta=m._vacuum_eta_prefactor*pout/(1.+m._vacuum_eta_pressure_scale*pout)
        old=(m._energy_gamma_ratio/eta)*pout*u;old*=rv
        assert b==old


@pytest.mark.parametrize('overpressure,order', [(None, 3), (100., 2)])
def test_valve_step_runs_bdf_order_two(tmp_path, monkeypatch, overpressure, order):
    """BDF above order 2 is not A-stable; only a step with the one-way outlet drops to 2."""
    import pydac.core.solver as solver_module
    profile = neutral_profile()
    profile['simulation']['cycle']['steps']['startup']['flow']['ads_outlet_overpressure_Pa'] = overpressure
    pb = model(profile)
    captured = {}

    class Stop(Exception):
        pass

    def fake_cvode(*args, **kwargs):
        captured.update(kwargs)
        raise Stop

    monkeypatch.setattr(solver_module, 'CVODE', fake_cvode)
    with redirect_stdout(StringIO()), pytest.raises(Stop):
        Solver(pb, base_folder=str(tmp_path)).solve_cvode()
    assert captured['max_order'] == order


def test_valve_steep_outlet_gradient_does_not_break_drag_inversion(tmp_path):
    """A last cell at 1/10 of its neighbour extrapolates a negative face density; the
    valve replaces that face, so the drag inversion must use the last cell's density."""
    profile = neutral_profile()
    profile['simulation']['cycle']['steps']['startup']['flow']['ads_outlet_overpressure_Pa'] = 100.
    pb = model(profile)
    solver = Solver(pb, base_folder=str(tmp_path))
    y = solver._initial_state()
    for name in ('c_tot', 'c_CO2'):
        y[pb.state_layout.field(name).state_slice][-1] *= .1
    assert solver._state_is_physical(y)
    out = np.zeros_like(y)
    solver.pde_callback(0., y, out)
    assert np.isfinite(out).all() and solver.rhs_rejections == 0
    assert pb.u_ax_single[-1, 0, 0] == 0.0            # bed below p_out: valve closed
    solver.sol = {'y': np.column_stack([y, y]), 't': np.array([0., 1.])}
    solver.prepare_results()
    restored = pickle.loads(pickle.dumps(solver.sol))
    pb.ads_outlet_valve = None                         # reconstruct from saved metadata only
    solver.sol = restored
    with redirect_stdout(StringIO()):
        pp = StepPostProcessor(pb, solver)
    np.testing.assert_array_equal(pp.u_ax[-1], 0.0)
    np.testing.assert_allclose(pp.u_ax[-1], restored['outlet_face']['velocity_m_s'], atol=1e-8, rtol=0)


# The schedule must fit every horizon a run can give the step, with the
# execution overrides (global, cycle, step) taken as the runner takes them.
_EVENT_STARTUP = {'execution.termination_mode': 'event',   # the case file says 'time'
                  'steps.startup.termination.mode': 'event',
                  'steps.startup.horizon.nominal_time_s': 1.0,
                  'steps.startup.horizon.event_max_time_s': 10.0,
                  'grid.output_schedule.dense_start_s': 2.0,
                  'grid.output_schedule.dense_start_points': 3}


def test_schedule_fits_own_event_horizon():
    cfg = load_case('tests/cases/v3_smoke_startup.json', settings=_EVENT_STARTUP)
    assert cfg.simulation.grid.output_schedule['dense_start_s'] == 2.0


@pytest.mark.parametrize('override', [
    {'execution.termination_mode': 'time'},
    {'execution.step_options': {'startup': {'termination_mode': 'time'}}},
    {'execution.cycle_options': {1: {'termination_mode': 'TIME'}}},
])
def test_schedule_checked_against_execution_termination_override(override):
    with pytest.raises(ConfigurationError, match=r'startup.*time mode.*horizon 1 s'):
        load_case('tests/cases/v3_smoke_startup.json', settings={**_EVENT_STARTUP, **override})
