"""Feed blower T4/T6/T8 and analytic supply-work checks (schema 10)."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from types import SimpleNamespace
import pickle
import warnings
import numpy as np
import pytest
from scipy.constants import gas_constant as R
from pydac.config import load_experiment, ConfigurationError
from pydac.config.cycle import CycleConfig
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.core.feed_blower import FeedBlower, check_feed_blower, build_feed_blower, target_inside_band_warning
from pydac.core.solver import Solver
from pydac.inputs.case_io import validate_case_document, case_fingerprint
from pydac import load_case
from pydac.config.migrations import V9_FEED_BLOWER_FIELDS, upgrade_experiment_document, pre_feed_blower_v9_vocabulary
from pydac.post_processing.profiles.step_post_processing import StepPostProcessor
from tests.neutral_packed_bed import neutral_profile

BLOWER = {'steps.press.flow.press_inlet_model': 'feed_blower', 'steps.press.flow.feed_blower_flow_m3_s': 1e-4}
CASE='tests/cases/v3_smoke_startup.json'


def make_model(nz=6, *, q=1e-4, band=None, initial_p=5000., feed_p=101325., valve=False, conductance=0., t_end=20., rtol=1e-8):
    profile=neutral_profile();sim=profile['simulation'];sim['grid'].update(nz=nz,nr=1,nt=1001)
    sim['geometry'].update(volume_m3=.001,aspect_ratio=2.)
    sim['adsorption']['kinetics']={s:{'model':'LDF','parameters':{'k_LDF_s_inv':1e-12}} for s in ['CO2','N2']}
    steps=sim['cycle']['steps'];steps['startup']['pressure']['initial_pressure_Pa']=initial_p
    steps['startup']['composition']['co2_initial_mole_fraction']=.1
    for st in steps.values(): st['thermal']['feed_temperature_K']=298.15
    steps['startup']['thermal']['initial_temperature_K']=298.15
    press=steps['press'];press['pressure']['feed_pressure_Pa']=feed_p
    press['flow'].update(press_inlet_model='valve' if valve else 'feed_blower',
                         feed_blower_flow_m3_s=None if valve else q,feed_blower_band_fraction=None if valve else band,
                         pressurization_valve_conductance=None if valve else conductance)
    press['termination']['mode']='time';press['horizon'].update(nominal_time_s=t_end,event_max_time_s=t_end)
    with redirect_stdout(StringIO()):
        cycle=CycleConfig.from_experiment(load_experiment(default_profile=profile))
        factory=ColumnModelFactory(cycle);opts=cycle.solver_for_step(use_sparsity=True)
        start=factory.build_column_model(step='startup',solver_options=opts)
        warm={'Nz':start.Nz,'Nr':start.Nr,'ci':start.ci.copy(),'qi':start.qi.copy(),
              'c_tot':start.c_tot[...,0].copy(),'T':start.T[...,0].copy(),'T_wall':start.Tw[:,0,0].copy(),
              'layout_signature':start.state_layout.signature}
        model=factory.build_column_model(step='press',initial_state=warm,solver_options=opts)
    model.rtol=rtol;model.atol=rtol*.01
    return model


def initial_state(model):
    state=np.zeros(model.state_layout.state_size)
    state[:model.state_layout.pde_size]=model.get_initial_state()
    return state


def controlled_solve(model, folder, *, thermostat=True):
    """Controlled inert test only: impose isothermal bath on the energy rows.

    No production RHS changes; species and momentum equations remain active.
    """
    original=model.pde_system_kpis
    if thermostat:
        rows=[model.state_layout.field(n).state_slice for n in ('T_bed','T_wall')]
        def rhs(t,y,out=None):
            result=original(t,y,out=out)
            for row in rows: result[row]=0.
            return result
        model.pde_system_kpis=rhs
    with redirect_stdout(StringIO()):
        solver=Solver(model,base_folder=str(folder));solver.solve_cvode();solver.prepare_results()
    assert solver.sol['cvode']['success']
    return solver


def test_law_scalar_vector_edges_monotone_c1_and_trial_pressures():
    law=FeedBlower(2.,1.,.001)
    with np.errstate(all='raise'):
        p=np.array([-np.finfo(float).max,-1.,0.,.9,.999,.9995,1.,1.1,np.finfo(float).max])
        np.testing.assert_allclose(law.inlet_velocity(p),[2,2,2,2,2,1,0,0,0],atol=1e-12)
        for pi,ui in zip(p,law.inlet_velocity(p)): assert law.inlet_velocity(float(pi))==ui
        u=law.inlet_velocity(np.linspace(-.1,1.1,20001));assert np.all(np.diff(u)<=0) and np.all((u>=0)&(u<=2))
        h=law.delta*1e-6
        for edge in (1.,.999):
            left=(law.inlet_velocity(edge)-law.inlet_velocity(edge-h))/h
            right=(law.inlet_velocity(edge+h)-law.inlet_velocity(edge))/h
            assert abs(left-right)<1e-4*law.u_cap/law.delta


@pytest.mark.parametrize('key,value',[
 ('press_inlet_model','bad'),('press_inlet_model',True),
 ('feed_blower_flow_m3_s',None),('feed_blower_flow_m3_s',0),('feed_blower_flow_m3_s',-1),
 ('feed_blower_flow_m3_s',True),('feed_blower_flow_m3_s',float('nan')),('feed_blower_flow_m3_s',float('inf')),
 ('feed_blower_band_fraction',0),('feed_blower_band_fraction',.1001),('feed_blower_band_fraction',-.1),
 ('feed_blower_band_fraction',True),('feed_blower_band_fraction',float('nan')),('feed_blower_band_fraction',float('inf'))])
def test_invalid_settings(key,value):
    with pytest.raises((ValueError,ConfigurationError)):
        load_experiment(CASE,settings={**BLOWER,f'steps.press.flow.{key}':value})


@pytest.mark.parametrize('changes',[
 {'model':'valve'}, {'model':'bad'}, {'step':'ads'}, {'is_1d':False},
 {'flow_m3_s':None},{'flow_m3_s':True},{'flow_m3_s':0.},{'flow_m3_s':-1.},{'flow_m3_s':float('inf')},
 {'flow_m3_s':float('nan')},{'band_fraction':True},{'band_fraction':0.},{'band_fraction':.11},
 {'band_fraction':float('nan')},{'band_fraction':float('inf')},
 {'setpoint_Pa':0.},{'setpoint_Pa':True},{'setpoint_Pa':float('nan')},{'setpoint_Pa':float('inf')}])
def test_direct_guard(changes):
    args=dict(path='flow',step='press',is_1d=True,model='feed_blower',flow_m3_s=1e-4,band_fraction=None,setpoint_Pa=1e5)
    with pytest.raises(ValueError): check_feed_blower(**{**args,**changes})


@pytest.mark.parametrize('st',['startup','ads','blow','heat','w_purge','s_purge','s_des','w_des','cool'])
def test_only_press(st):
    with pytest.raises((ValueError,ConfigurationError)):
        load_experiment(CASE,settings={f'steps.{st}.flow.press_inlet_model':'feed_blower',f'steps.{st}.flow.feed_blower_flow_m3_s':1e-4})


@pytest.mark.parametrize('key',['feed_blower_flow_m3_s','feed_blower_band_fraction'])
def test_supply_fields_refused_on_valve(key):
    with pytest.raises((ValueError,ConfigurationError)): load_experiment(CASE,settings={f'steps.press.flow.{key}':.001})


def test_2d_config_and_build_refused():
    with pytest.raises((ValueError,ConfigurationError)): load_experiment(CASE,settings={**BLOWER,'grid.nr':2})
    m=SimpleNamespace(step='press',is_1d=False,press_inlet_model='feed_blower',feed_blower_flow_m3_s=1e-4,feed_blower_band_fraction=None,p_feed=1e5)
    with pytest.raises(ValueError,match='1D'):build_feed_blower(m)


@pytest.mark.parametrize('version',range(1,10))
@pytest.mark.parametrize('field',list(V9_FEED_BLOWER_FIELDS))
@pytest.mark.parametrize('loader',[upgrade_experiment_document,validate_case_document,load_experiment])
def test_every_new_field_even_null_refused_in_every_old_schema(version,field,loader):
    document={'schema_version':version,'simulation':{'cycle':{'steps':{'press':{'flow':{field:None}}}}}}
    with pytest.raises((ValueError,ConfigurationError),match='is new in pyDAC 3.2.0; set `schema_version` to 11'):loader(document)


def test_v9_retains_vacuum_standby_and_roundtrip(tmp_path):
    import pydac
    c=load_experiment(CASE,settings={'steps.heat.flow.vacuum_outlet_model':'vacuum_pump',
           'steps.heat.flow.vacuum_pump_speed_m3_s':.04,'steps.heat.flow.vacuum_pump_idle_mode':'fixed_speed',
           'steps.heat.flow.vacuum_pump_idle_power_W':3000.})
    doc=validate_case_document(c);old=pre_feed_blower_v9_vocabulary(doc)
    assert validate_case_document(old)==doc
    assert load_case(old,exact=True)==c
    flow={'press_inlet_model':'feed_blower','feed_blower_flow_m3_s':1e-4,'feed_blower_band_fraction':.002}
    preset={'schema_version':10,'simulation':{'grid':{'nr':1},'cycle':{'steps':{'press':{'flow':flow}}}}}
    c=load_experiment(preset,settings={f'steps.press.flow.{k}':None for k in flow if k != 'press_inlet_model'})
    p=tmp_path/'case.json';pydac.export_case(c,p);reloaded=load_case(p,exact=True)
    assert c==reloaded
    f=next(s.config.flow for s in reloaded.simulation.cycle.steps if s.name=='press')
    assert (f.press_inlet_model,f.feed_blower_flow_m3_s,f.feed_blower_band_fraction)==('feed_blower',1e-4,.002)


def test_default_all_steps_and_missing_v9():
    c=load_experiment({'schema_version':9,'simulation':{'cycle':{'steps':{'press':{'flow':{}}}}}})
    assert all(s.config.flow.press_inlet_model=='valve' for s in c.simulation.cycle.steps)


@pytest.mark.parametrize('alpha',[0.,1.,None])
def test_ignored_conductance(alpha,capsys):
    m=make_model(conductance=alpha)
    assert m.ps_fact==0. and m.feed_blower is not None and m.inlet_flux_source=='feed_blower'
    m.alpha_ps=np.inf if alpha is None else alpha
    build_feed_blower(m);out=capsys.readouterr().out
    assert ('is not read' in out)==(alpha is not None)


@pytest.mark.parametrize('target,expected',[(99899.,False),(99900.,True),(100000.,True),(101000.,True)])
def test_warning_band_edges_and_above(target,expected):
    term=SimpleNamespace(metric='min_internal_pressure',direction='above',threshold_source=None,threshold_value=target,threshold_offset=0)
    assert bool(target_inside_band_warning(path='term',termination=term,setpoint_Pa=1e5,band_fraction=.001,outlet_pressure_Pa=5000.))==expected


@pytest.mark.parametrize('source,value,offset,expected', [('p_feed',None,-101.,False),('p_feed',None,-100.,True),('p_out',None,95000.,True),('other',None,0.,False)])
def test_warning_sources_offsets(source,value,offset,expected):
    term=SimpleNamespace(metric='min_internal_pressure',direction='above',threshold_source=source,threshold_value=value,threshold_offset=offset)
    assert bool(target_inside_band_warning(path='term',termination=term,setpoint_Pa=1e5,band_fraction=None,outlet_pressure_Pa=5000.))==expected


@pytest.mark.parametrize('settings,expected',[
 ({'steps.press.termination.threshold_offset':0.},True),
 ({'steps.press.termination.threshold_offset':0.,'steps.press.termination.mode':'time'},False),
 ({'steps.press.termination.threshold_offset':0.,'execution.cycle_steps':['ads']},False),
 ({'steps.press.termination.threshold_offset':-1013.25},False),
])
def test_warning_effective_execution(settings,expected):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always');load_experiment(CASE,settings={**BLOWER,'execution.cycle_steps':['press'],'execution.termination_mode':None,**settings})
    assert any('feed blower' in str(x.message) for x in w)==expected


@pytest.mark.parametrize('feed,flow',[(101325.,1e-4),(200000.,1e-4),(50000.,1e-4),(200000.,0.)])
def test_full_supply_power_formula(feed,flow):
    m=make_model(feed_p=feed)
    cp,vac,jacket=m._compute_instantaneous_energies(np.array([5000.]),np.array([6000.]),np.array([flow/m.inlet_area]),np.array([0.]),np.full(m.Nz,298.15))
    gamma=m.cp_g_mix/(m.cp_g_mix-R)
    expected=gamma/(.7*(gamma-1))*feed*flow*(max(1.,feed/101325.)**((gamma-1)/gamma)-1.)
    assert cp==pytest.approx(expected,rel=1e-12,abs=1e-15)


def test_saved_and_reloaded_multitime_profiles_rhs_and_cutoff(tmp_path):
    m=make_model();state=initial_state(m);history=[];expected=[]
    for p in (5000.,m.p_feed*.9995,m.p_feed*(1.-1e-9),m.p_feed,m.p_feed*1.01):
        y=state.copy();c=m.state_layout.field('c_tot');T=m.state_layout.field('T_bed')
        y[c.state_slice]=p/m.p_ref/y[T.state_slice]
        # Supply density guard is exercised separately with a first cell four times less dense.
        m.pde_system_kpis(0.,y);history.append(y);expected.append(m.u_ax_single[0,0,0]*m.u_ref)
    s=Solver(m,base_folder=str(tmp_path));s.sol={'y':np.column_stack(history),'t':np.arange(len(history),dtype=float)}
    with redirect_stdout(StringIO()):s.save_results();pp=StepPostProcessor(m,s)
    expected=np.array(expected);clipped=expected.copy();clipped[abs(clipped)<1e-8]=0.
    np.testing.assert_allclose(pp.u_ax[0,0],clipped,rtol=1e-10,atol=1e-14)
    assert 0.<expected[2]<1e-8 and pp.u_ax[0,0,2]==0.
    # Public profiles explicitly permit the existing display cutoff.
    np.testing.assert_allclose(pp.u_ax[0,0],expected,rtol=1e-10,atol=1e-8)
    s.sol=None
    with redirect_stdout(StringIO()):reloaded=StepPostProcessor(m,s)
    np.testing.assert_array_equal(reloaded.u_ax,pp.u_ax)
    m.feed_blower=None;del m.feed_blower
    with redirect_stdout(StringIO()):StepPostProcessor(m,s)  # old models have no new attribute


def test_density_guard_avoids_negative_extrapolated_face():
    m=make_model();y=initial_state(m);row=m.state_layout.field('c_tot').state_slice
    c=y[row].reshape(m.shape_single);c[0]=.25*c[1];m.pde_system_kpis(0.,y)
    assert np.all(np.isfinite(m.u_ax_single))
    # Independent adverse interpolation: [1,4,4,...] extrapolates a negative face.
    rho=np.full(m.Nz,4.);rho[0]=1.
    p=np.full(m.Nz,.5);law=m.feed_blower;m.feed_blower=None
    with pytest.raises(ValueError,match='nonnegative'):
        m._velocity_from_pressure_law(p,rho)
    m.feed_blower=law
    assert np.isfinite(m._velocity_from_pressure_law(p,rho)).all()


def test_nsga_candidate_capacity_remains_absolute_and_persists(tmp_path):
    import pydac
    from pydac.utils.nsga2.decisions import decision_vector_to_model_inputs
    op,_=decision_vector_to_model_inputs([.1,1200,1200,5000,.1],is_steam_assisted=False,tau_ar=2.,bed_volume=.001)
    from pydac.config.operating import operating_step_settings
    c=load_experiment(CASE,settings={**BLOWER,**operating_step_settings(op)})
    p=tmp_path/'candidate.json';pydac.export_case(c,p);c=load_case(p,exact=True)
    f=next(s.config.flow for s in c.simulation.cycle.steps if s.name=='press')
    assert f.feed_blower_flow_m3_s==1e-4
    assert next(s.config.flow for s in c.simulation.cycle.steps if s.name=='ads').volumetric_flow_m3_s!=f.feed_blower_flow_m3_s


@pytest.mark.parametrize('pressure',[0.,-1.,True,float('inf'),float('nan')])
def test_invalid_setpoint_config(pressure):
    with pytest.raises((ValueError,ConfigurationError)):
        load_experiment(CASE,settings={**BLOWER,'steps.press.pressure.feed_pressure_Pa':pressure})


@pytest.mark.parametrize('band',[None,1e-3,.1])
def test_valid_band_endpoints_and_default(band):
    m=make_model(band=band)
    assert m.feed_blower.delta/m.feed_blower.p_set==pytest.approx(1e-3 if band is None else band)


def test_warning_effective_cycle_override():
    settings={**BLOWER,'execution.cycle_steps':['press'],'execution.termination_mode':'time',
              'execution.cycle_options':{'2':{'step_options':{'press':{'termination_mode':'event'}}}},
              'steps.press.termination.threshold_offset':0.}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always');load_experiment(CASE,settings=settings)
    assert any('feed blower' in str(x.message) for x in w)


@pytest.mark.parametrize('band', [0.0, 1e-20, 9.9e-7, 0.1000001])
def test_unresolvable_or_too_wide_band_is_refused(band):
    args = dict(path='flow', step='press', is_1d=True, model='feed_blower', flow_m3_s=1e-4,
                band_fraction=band, setpoint_Pa=1e5)
    with pytest.raises(ValueError, match=r'\[1e-6, 0.1\]'):
        check_feed_blower(**args)


def test_smallest_band_opens_far_below_and_closes_at_setpoint():
    law = FeedBlower(u_cap=2.0, p_set=1.0, delta=1e-6)
    np.testing.assert_array_equal(law.inlet_velocity(np.array([0.05, 1.0 - 1e-6, 1.0, 1.5])), [2.0, 2.0, 0.0, 0.0])
