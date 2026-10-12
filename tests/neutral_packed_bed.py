"""Synthetic binary adsorption case for conservation tests; no experimental fit."""
from pydac.inputs.defaults import default_profile


def neutral_profile():
    profile = default_profile()
    sim = profile['simulation']
    sim['grid'].update(nz=6, nr=5)
    sim['bed'] = {'architecture': 'packed_bed', 'design': {'eps_b': 0.388}, 'material': None}
    sim['mixture'] = {'species': ['CO2', 'N2'], 'closure_species': 'N2', 'adsorbates': ['CO2', 'N2']}
    component = dict(qinf_ref_mol_per_kg=2., qinf_temperature_coefficient=0.,
                     affinity_prefactor_inverse_pressure_unit=0.1, affinity_heat_J_per_mol=1000.,
                     heterogeneity_ref=1., heterogeneity_temperature_coefficient=0.)
    sim['adsorption'] = {
        'equilibrium_systems': {'CompetitiveSips': {
            'datasets': {'CompetitiveSips': 'synthetic-test'},
            'parameters': {'component_parameters': {'CO2': dict(component), 'N2': dict(component)},
                           'reference_temperature_K': 300., 'pressure_unit_Pa': 100000.}}},
        'kinetics': {s: {'model': 'LDF', 'parameters': {'k_LDF_s_inv': 0.003}}
                     for s in ['CO2', 'N2']}}
    sim['sorbent']['heats_of_adsorption_J_mol'] = {'CO2': -20000., 'N2': -10000.}
    for step in sim['cycle']['steps'].values():
        step['composition'].update(co2_feed_mole_fraction=0.1, relative_humidity=0.,
                                   o2_feed_mole_fraction=0., n2_feed_mole_fraction=0.9)
    sim['cycle']['steps']['startup']['horizon'].update(nominal_time_s=0.2, event_max_time_s=0.2)
    return profile
