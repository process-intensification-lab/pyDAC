import numpy as np
import pytest

from pydac.core.bed_architectures import PackedBedDesign, packed_bed_ring_inlet_flux_weights


def test_default_design_keeps_uniform_flux_and_canonical_mapping():
    design = PackedBedDesign(eps_b=0.388)
    assert design.to_config_dict() == {"eps_b": 0.388}
    np.testing.assert_array_equal(
        packed_bed_ring_inlet_flux_weights(design, np.linspace(0, 1, 26)), np.ones(25)
    )


def test_profile_weights_conserve_total_flow_and_follow_the_profile():
    design = PackedBedDesign(
        eps_b=0.388, inlet_flux_profile=[[0, 0.8], [0.85, 0.8], [0.95, 2.0], [1, 2.0]]
    )
    faces = np.linspace(0, 1, 26)
    weights = packed_bed_ring_inlet_flux_weights(design, faces)
    areas = np.diff(faces**2)
    assert np.sum(weights * areas) / np.sum(areas) == pytest.approx(1.0)
    assert weights[0] < 1.0 < weights[-1]
    assert design.to_config_dict()["inlet_flux_profile"][0] == [0.0, 0.8]
    assert PackedBedDesign(**design.to_config_dict()) == design


@pytest.mark.parametrize(
    "profile",
    [[[0, 1]], [[0.1, 1], [1, 1]], [[0, 1], [0.5, 1], [0.5, 1], [1, 1]], [[0, 0], [1, 0]], [[0, -1], [1, 1]]],
)
def test_invalid_profiles_are_rejected(profile):
    with pytest.raises(ValueError):
        PackedBedDesign(eps_b=0.388, inlet_flux_profile=profile)


def test_radial_porosity_factors_conserve_holdup_and_sorbent():
    from pydac.core.bed_architectures import ParticleProperties, packed_bed_ring_structure_factors

    particle = ParticleProperties(density=1200.0, porosity=0.5, radius=1e-3, thermal_conductivity=0.2)
    faces = np.linspace(0, 1, 26)
    areas = np.diff(faces**2)
    uniform = packed_bed_ring_structure_factors(PackedBedDesign(eps_b=0.388), particle, faces)
    np.testing.assert_array_equal(uniform[0], np.ones(25))
    np.testing.assert_array_equal(uniform[1], np.ones(25))
    x = np.linspace(0, 1, 401)
    eps = 0.37 + 0.63 * np.exp(-40 * (1 - x))
    eps += 0.388 - np.sum(np.interp(0.5 * (faces[1:] + faces[:-1]), x, eps) * areas) / np.sum(areas)
    design = PackedBedDesign(eps_b=0.388, radial_porosity_profile=np.column_stack([x, np.minimum(eps, 1)]).tolist())
    gas, solid = packed_bed_ring_structure_factors(design, particle, faces)
    assert np.sum(gas * areas) / np.sum(areas) == pytest.approx(1.0)
    assert np.sum(solid * areas) / np.sum(areas) == pytest.approx(1.0)
    assert solid[-1] < 1.0 < gas[-1]
    assert PackedBedDesign(**design.to_config_dict()) == design
    with pytest.raises(ValueError):
        packed_bed_ring_structure_factors(
            PackedBedDesign(eps_b=0.30, radial_porosity_profile=design.radial_porosity_profile), particle, faces
        )


def test_nonuniform_bed_rhs_conserves_each_species_and_sensible_inventory():
    """A nonuniform state exercises the RHS, beyond normalising profile weights."""
    import contextlib
    import io

    from pydac.core.column_model_factory import ColumnModelFactory
    from pydac.config import load_experiment
    from pydac.config.cycle import CycleConfig
    from tests.neutral_packed_bed import neutral_profile

    profile = neutral_profile()
    simulation = profile["simulation"]
    simulation["grid"].update(nz=6, nr=5)
    simulation["bed"]["design"].update(
        # eps = 0.34 + 0.072 r/R has cylindrical area mean 0.388.
        radial_porosity_profile=[[0, 0.34], [1, 0.412]],
        inlet_flux_profile=[[0, 0.8], [0.8, 0.8], [1, 2]],
    )
    cycle = CycleConfig.from_experiment(load_experiment(default_profile=profile))
    with contextlib.redirect_stdout(io.StringIO()):
        model = ColumnModelFactory(cycle).build_column_model(
            step="startup", solver_options=cycle.solver_for_step(use_sparsity=False))
    z = np.asarray(model.z_c)[:, None]
    r = np.asarray(model.r_c)[None, :]
    temperature = model.T_ref * (1.1 + 0.1 * (1 - z) * (1 - r))
    concentration = model.c_tot_ref * (1 + 0.001 * (1 - z)) * model.T_ref / temperature
    state_values = {f.name: np.zeros(f.shape) for f in model.state_layout.fields}
    state_values.update(
        c_tot=concentration / model.c_tot_ref,
        c_CO2=concentration * (0.03 + 0.06 * (1 - z) * (1 - r)) / model.c_ref_by_species["CO2"],
        q_CO2=np.broadcast_to(0.3 + 0.2 * z, (model.Nz, model.Nr)) / model.q_ref_by_species["CO2"],
        q_N2=np.broadcast_to(0.2 + 0.1 * r, (model.Nz, model.Nr)) / model.q_ref_by_species["N2"],
        T_bed=temperature / model.T_ref,
        T_wall=np.full(model.Nz, 1.02),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        rhs = model.pde_system_kpis(0, model.state_layout.pack(state_values)).copy()

    def rate(name):
        field = model.state_layout.field(name)
        return rhs[field.state_slice].reshape(field.shape) * model.state_scale_by_field[name] / model.tau_hyd

    volume = np.diff(model.z_f)[:, None] * model.L_b * np.asarray(model.A_rings)[None, :]
    gas = model.eps_total * model._gas_2d
    sorbent = model.rho_sorbent * model._solid_2d
    for species in ("CO2", "N2"):
        dc = rate("c_CO2") if species == "CO2" else rate("c_tot") - rate("c_CO2")
        storage_rate = np.sum(volume * (gas * dc + sorbent * rate("q_" + species)))
        flow_rate = rate("m_" + species).item() / model.MW[model.gas_index[species]]
        np.testing.assert_allclose(storage_rate, flow_rate, rtol=1e-11, atol=1e-13)

    # acc_g/acc_ads include reference scales; independent construction in
    # dimensional concentration/loading rates avoids the RHS matrix products.
    gas_sensible = sum(
        model.acc_g[i] / model.c_ref[i]
        * (rate("c_CO2") if species == "CO2" else rate("c_tot") - rate("c_CO2"))
        for i, species in enumerate(model.gas_species)
    ) * model._gas_2d
    ads_sensible = sum(
        model.acc_ads[i] / model.q_ref_by_species[species] * rate("q_" + species)
        for i, species in enumerate(model.adsorbates)
    ) * model._solid_2d
    expected = -(temperature / model.T_ref - model.T_enthalpy_ref_nd) * (gas_sensible + ads_sensible) * model.tau_hyd
    np.testing.assert_allclose(model._source_extra_buf.reshape(model.Nz, model.Nr), expected, rtol=1e-11, atol=1e-10)


