"""Saved outlet observations must match the PDE balances they are read against."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pydac.core.column_model_factory import ColumnModelFactory
from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.core.solver import Solver
from pydac.core.outlet_reporting import outlet_face_history
from pydac.post_processing.profiles.step_post_processing import StepPostProcessor
from pydac.post_processing.profiles.cyclic_plotting import CyclicSteadyStateProfiles
from pydac.utils.postprocessing.outlet import outlet_mole_fractions
from pydac.utils.postprocessing.saved_solution import saved_solution_view
from tests.neutral_packed_bed import neutral_profile


def _prepare(tmp_path, nr, profiled=False):
    """A radially non-uniform state at uniform porosity, replayed through the solver.

    Temperature, composition and velocity all vary with r. That is enough to
    separate a molar-flow mixing cup from an area or mass-flux average: no radial
    porosity profile is involved, and none is needed to expose the difference.
    """
    profile = neutral_profile()
    profile["simulation"]["grid"].update(nz=6, nr=nr)
    if profiled:
        profile["simulation"]["bed"]["design"].update(
            radial_porosity_profile=[[0, 0.34], [1, 0.412]],
            inlet_flux_profile=[[0, 0.8], [0.8, 0.8], [1, 2]])
    cycle = CycleConfig.from_experiment(load_experiment(default_profile=profile))
    model = ColumnModelFactory(cycle).build_column_model(
        step="startup", solver_options=cycle.solver_for_step(use_sparsity=False))
    z = model.z_c[:, None]
    r = model.r_c[None, :]
    temperature = model.T_ref * (1.1 + 0.1 * (1 - z) * (1 - r))
    concentration = model.c_tot_ref * (1 + 0.001 * (1 - z)) * model.T_ref / temperature
    model.c_tot[...] = concentration[..., None]
    model.ci[..., 0] = concentration * (0.03 + 0.06 * (1 - z) * (1 - r))
    model.qi[..., model.adsorbate_index["CO2"]] = 0.3 + 0.2 * z + 0.5 * r
    model.qi[..., model.adsorbate_index["N2"]] = 0.2 + 0.1 * r
    model.T[...] = temperature[..., None]
    solver = Solver(model, base_folder=str(tmp_path))
    state = solver._initial_state()
    history = np.column_stack([state, state])
    solver.sol = {"y": history.copy(), "t": np.array([0., 1.])}
    solver.prepare_results()
    return model, solver, history


@pytest.fixture(params=[False, True], ids=["uniform", "profiled"])
def prepared(request, tmp_path):
    return _prepare(tmp_path, nr=5, profiled=request.param)


@pytest.fixture
def prepared_1d(tmp_path):
    return _prepare(tmp_path, nr=1)


def test_persisted_face_flows_equal_rhs_boundary_balance(prepared):
    model, solver, history = prepared
    original = history.copy()
    for k in range(history.shape[1]):
        rhs = model.pde_system_kpis(0, history[:, k]).copy()
        for species in model.adsorbates:
            field = model.state_layout.field("m_" + species)
            mass_rate = rhs[field.state_slice].item() * model.state_scale_by_field[field.name] / model.tau_hyd
            index = model.gas_index[species]
            expected_out = (model.mass_flow_in_i[index] - mass_rate) / model.MW[index]
            saved_out = solver.sol["outlet_face"]["species_molar_flow_mol_s"][species][:, k].sum()
            assert saved_out == pytest.approx(expected_out, rel=1e-12, abs=1e-15)
        np.testing.assert_allclose(solver.sol["outlet_face"]["velocity_m_s"][:, k], model.u_ax_single[-1, :, 0] * model.u_ref)
    np.testing.assert_array_equal(history, original)
    before = deepcopy(solver.sol["outlet_face"])
    solver.prepare_results()
    np.testing.assert_array_equal(solver.sol["outlet_face"]["velocity_m_s"], before["velocity_m_s"])


def test_persisted_enthalpy_flow_equals_rhs_convective_energy_flux(prepared):
    # The energy balance transports `c_tot_dim * h_g` through the same convection
    # operator the species use, so the reported face flow has to equal that
    # operator's outlet row. There is no integrated energy state at the outlet to
    # check against, unlike the species balance above.
    model, solver, history = prepared
    for k in range(history.shape[1]):
        state = history[:, k]
        # Run the balance itself, so the operator is built on the velocity field
        # the RHS actually used for this state.
        model.pde_system_kpis(0, state)
        c_tot = state[model.state_field_by_name["c_tot"].state_slice]
        T_bed = state[model.state_field_by_name["T_bed"].state_slice]
        c_tot_dim = c_tot * model.c_tot_ref
        _, _, y_full_matrix = model._reconstruct_gas_state(state, c_tot_dim, 1e-32)

        h_g = (y_full_matrix @ model.cp_g) * (T_bed - model.T_enthalpy_ref_nd)
        conv, conv_bc = model.conv_ax_cache.update(model.u_ax_single)
        shape = (model.Nz + 1, model.Nr)
        # The outlet face is built from the interior cells alone; the species
        # path assumes this and never checks it.
        assert np.all(np.asarray(conv_bc).reshape(shape)[-1] == 0.0)

        flux = conv.dot(c_tot_dim * h_g).reshape(shape)[-1]
        expected = flux * model.u_ref * model.T_ref * model.A_rings
        np.testing.assert_allclose(
            solver.sol["outlet_face"]["sensible_enthalpy_flow_W"][:, k],
            expected,
            rtol=1e-12,
        )


def test_mixing_cup_is_the_molar_flow_ratio_not_an_area_or_mass_average(prepared):
    model, solver, _ = prepared
    processor = StepPostProcessor(model, solver)
    cups = processor.get_mixed_cups()
    flows = solver.sol["outlet_face"]["species_molar_flow_mol_s"]
    total = sum(np.sum(values, axis=0) for values in flows.values())
    for species in model.gas_species:
        expected = np.sum(flows[species], axis=0) / total
        np.testing.assert_allclose(cups[f"y_{species}"], expected, rtol=1e-12)
    np.testing.assert_allclose(sum(cups[f"y_{s}"] for s in model.gas_species), 1.0, rtol=1e-12)

    # An area average of the same face is a different number, which is the whole
    # point: the state varies radially at uniform porosity.
    y_face = {s: np.sum(flows[s], axis=0) for s in model.gas_species}
    area_average = np.average(
        solver.sol["outlet_face"]["velocity_m_s"], axis=0, weights=model.A_rings
    )
    assert not np.allclose(cups["y_CO2"], area_average)
    assert y_face["CO2"].shape == cups["y_CO2"].shape


def test_mixing_temperature_uses_an_enthalpy_balance(prepared):
    model, solver, _ = prepared
    processor = StepPostProcessor(model, solver)
    cups = processor.get_mixed_cups()
    flows = solver.sol["outlet_face"]["species_molar_flow_mol_s"]
    enthalpy = np.sum(solver.sol["outlet_face"]["sensible_enthalpy_flow_W"], axis=0)
    capacity = sum(
        np.sum(flows[species], axis=0) * model.cp_g[model.gas_index[species]]
        for species in model.gas_species
    )
    expected = model.T_enthalpy_ref_K + enthalpy / capacity
    np.testing.assert_allclose(cups["T_out"], expected, rtol=1e-12)

    # Weighting T by the molar flows is a different number, so the enthalpy
    # balance is not a relabelled molar average.
    molar_weighted = np.average(
        np.asarray(processor.T_bed[-1, :, :]),
        axis=0,
        weights=np.sum([flows[s][:, 0] for s in model.gas_species], axis=0),
    )
    assert not np.allclose(cups["T_out"], molar_weighted)


def test_one_ring_reduces_to_the_face_composition(prepared_1d):
    # With a single ring the areas and the velocity cancel, so the mixing cup is
    # the face composition itself. Anything left over would be a weighting bug.
    model, solver, _ = prepared_1d
    processor = StepPostProcessor(model, solver)
    cups = processor.get_mixed_cups()
    flows = solver.sol["outlet_face"]["species_molar_flow_mol_s"]
    total = sum(np.sum(values, axis=0) for values in flows.values())
    for species in model.gas_species:
        np.testing.assert_allclose(
            cups[f"y_{species}"], np.sum(flows[species], axis=0) / total, rtol=1e-12
        )
    assert np.all(np.isfinite(cups["T_out"]))




def test_zero_and_closed_outlet_have_no_mixing_cup(prepared):
    model, _, history = prepared
    model.step = "cool"
    outlet = outlet_face_history(model, history)
    assert np.all(outlet["velocity_m_s"] == 0)
    assert np.all(outlet["sensible_enthalpy_flow_W"] == 0)
    assert np.all(np.isnan(outlet_mole_fractions({"outlet_face": outlet})["CO2"]))


def test_condenser_refuses_a_closed_outlet_and_a_net_inward_stream(prepared):
    model, solver, _ = prepared
    processor = StepPostProcessor(model, solver)

    processor.step = "cool"
    with pytest.raises(ValueError, match="closed outlet"):
        processor._condenser_stream_mask()

    processor.step = model.step
    for values in solver.sol["outlet_face"]["species_molar_flow_mol_s"].values():
        values *= -1.0
    with pytest.raises(ValueError, match="net inward"):
        processor._condenser_stream_mask()


def test_condenser_reads_a_tiny_inward_start_as_no_stream_but_refuses_reflux(prepared):
    # A step that starts from a bed already at its outlet pressure has a ~zero
    # face flow at t = 0 whose sign is noise; that is not reflux.
    model, solver, _ = prepared
    processor = StepPostProcessor(model, solver)
    flows = solver.sol["outlet_face"]["species_molar_flow_mol_s"]
    peak = np.max(sum(np.sum(values, axis=0) for values in flows.values()))
    species = next(iter(flows))
    for values in flows.values():
        values[:, 0] = 0.0
    flows[species][:, 0] = -1e-4 * peak / flows[species].shape[0]
    mask = processor._condenser_stream_mask()
    assert mask[0] == False  # noqa: E712 - comparing the mask entry, not truthiness
    assert mask[1] == True   # noqa: E712

    flows[species][:, 0] = -0.1 * peak / flows[species].shape[0]
    with pytest.raises(ValueError, match="net inward stream in 1 of"):
        processor._condenser_stream_mask()


@pytest.mark.parametrize("values, start, expected", [
    ([-0.01, 1.0], 0.0, [False, True]),
    ([-0.01001, 1.0], 0.0, None),
    ([-0.0001, 1.0], 1.0, None),
    ([1.0, -1e-12], 0.0, None),
    ([1.0, -1e-12, 1.0], 0.0, None),
    (np.r_[1.0, np.full(3999, -0.009)], 0.0, None),
    ([-1e-12, 0.0], 0.0, None),
    ([0.0, 0.0], 0.0, [False, False]),
])
def test_condenser_tolerates_only_initial_stagnant_flow(values, start, expected):
    processor = StepPostProcessor.__new__(StepPostProcessor)
    processor.step = "s_des"
    processor.time = start + np.arange(len(values), dtype=float)
    processor._outlet_net_flows = lambda: ({"CO2": np.asarray(values)}, None, None)
    if expected is None:
        with pytest.raises(ValueError, match="net inward stream"):
            processor._condenser_stream_mask()
    else:
        np.testing.assert_array_equal(processor._condenser_stream_mask(), expected)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("index", [0, 1])
def test_condenser_rejects_nonfinite_flow_before_masking(invalid, index):
    processor = StepPostProcessor.__new__(StepPostProcessor)
    processor.step = "s_des"
    processor.time = np.arange(3, dtype=float)
    values = np.array([1.0, 1.0, -0.1])
    values[index] = invalid
    processor._outlet_net_flows = lambda: ({"CO2": values}, None, None)
    with pytest.raises(ValueError, match="not finite"):
        processor._condenser_stream_mask()


def test_condenser_rejects_nonfinite_aggregate_of_finite_species():
    processor = StepPostProcessor.__new__(StepPostProcessor)
    processor.step = "s_des"
    processor.time = np.array([0.0, 1.0])
    flows = {species: np.full(2, np.finfo(float).max) for species in ("CO2", "H2O")}
    processor._outlet_net_flows = lambda: (flows, None, None)
    with pytest.raises(ValueError, match="Total outlet molar flow is not finite"):
        processor._condenser_stream_mask()


def test_condenser_keeps_a_streamless_sample_on_the_time_axis(prepared):
    model, solver, _ = prepared
    processor = StepPostProcessor(model, solver)
    for values in solver.sol["outlet_face"]["species_molar_flow_mol_s"].values():
        values[:, 0] = 0.0
    mask = processor._condenser_stream_mask()
    assert mask[0] == False  # noqa: E712 - comparing the mask entry, not truthiness
    assert mask[1] == True   # noqa: E712
    assert mask.size == solver.sol["outlet_face"]["velocity_m_s"].shape[1]


def test_cyclic_profiles_use_saved_face_composition_and_velocity(prepared):
    model, solver, _ = prepared
    cyclic = CyclicSteadyStateProfiles.__new__(CyclicSteadyStateProfiles)
    cyclic.layout_signature = None
    cyclic.gas_species = ()
    cyclic.adsorbates = ()
    cyclic.field_specs = dict(cyclic.FIELD_SPECS)
    cyclic.default_plot_fields = tuple(cyclic.DEFAULT_PLOT_FIELDS)
    cyclic.is_1d = None
    cyclic._mixed_cup_zero_den_warned = False
    outlet = cyclic._extract_outlet_profiles(solver.sol, "startup")
    fractions = outlet_mole_fractions(solver.sol)
    for species in model.gas_species:
        np.testing.assert_array_equal(outlet["y_" + species], fractions[species])
    expected_velocity = np.average(solver.sol["outlet_face"]["velocity_m_s"], axis=0, weights=model.A_rings)
    np.testing.assert_allclose(outlet["u_out"], expected_velocity)
    solver.sol.pop("outlet_face")
    with pytest.raises(ValueError, match="Vector-drag"):
        cyclic._extract_outlet_profiles(solver.sol, "startup")
    # An archived scalar-law result lacks the additive formulation marker.
    solver.sol["extra"]["pressure_drop"].pop("formulation")
    legacy = cyclic._extract_outlet_profiles(solver.sol, "startup")
    assert np.all(np.isfinite(legacy["y_CO2"]))




def test_initial_final_and_history_inventories_use_local_storage(prepared):
    model, solver, _ = prepared
    processor = StepPostProcessor(model, solver)
    for species in model.gas_species:
        c = (model.c_tot[..., 0] - model.ci.sum(axis=-1)
             if species == model.closure_species else model.ci[..., model.independent_index[species]])
        mw = model.MW[model.gas_index[species]]
        gas = np.sum(c * model.V_cells * model.eps_total * model._gas_2d) * mw
        solid = np.sum(model.qi[..., model.adsorbate_index[species]] * model.V_cells
                       * model.rho_sorbent * model._solid_2d) * mw
        assert model.calculate_initial_mass(species, "gas") == pytest.approx(gas)
        assert model.calculate_initial_mass(species, "solid") == pytest.approx(solid)
        assert processor._calculate_species_mass(species)["total"] == pytest.approx(gas + solid)
        np.testing.assert_allclose(processor._calculate_species_mass(species, time_index=None)["total"], gas + solid)
        assert processor.compute_final_gas_masses()[species] == pytest.approx(gas)
    weights = model.V_cells * model.rho_sorbent * model._solid_2d
    expected = np.sum(model.qi[..., model.adsorbate_index["CO2"]] * weights) / weights.sum() / model.q_eq_CO2_ref
    assert processor.compute_bed_loading_fraction_co2() == pytest.approx(expected)

    # Old artifacts recover spatial weights from the configured profile.
    solver.sol["extra"].pop("ring_gas_holdup_factor")
    solver.sol["extra"].pop("ring_sorbent_factor")
    legacy = StepPostProcessor(model, solver)
    np.testing.assert_array_equal(legacy.gas_holdup, processor.gas_holdup)
    np.testing.assert_array_equal(legacy.sorbent_density, processor.sorbent_density)


def test_cooldown_relaxation_preserves_spatial_total_gas_inventory(prepared):
    model, _, _ = prepared
    weights = model.V_cells * model.eps_total * model._gas_2d
    before = np.sum(weights * model.c_tot[..., 0])
    fractions = model.ci / model.c_tot
    model._relax_closed_bed_hydraulics()
    assert np.sum(weights * model.c_tot[..., 0]) == pytest.approx(before, rel=1e-14)
    np.testing.assert_allclose(model.ci / model.c_tot, fractions, rtol=1e-14)
    pressure_over_R = model.c_tot * model.T
    np.testing.assert_allclose(pressure_over_R, pressure_over_R.flat[0], rtol=1e-14)


class _DacResidualProbe(StepPostProcessor):
    """Only what the O2/N2 residual check reads, on a 2 x 1 gas volume."""

    _is_current_dac_layout = True

    def __init__(self, step):
        self.step = step
        self.V_cells = np.array([[2e-4], [4e-4]])
        self.gas_holdup = np.array([[0.5]])
        self.molecular_weight_by_species = {"O2": 0.032, "N2": 0.028}
        self.c_by_species = {"O2": np.full((2, 1, 3), 1e-12)}
        self.y_by_species = {"O2": np.full((2, 1, 3), 1e-12)}


def test_dac_residual_check_warns_beyond_the_floor_and_never_zeroes_fields(capsys):
    probe = _DacResidualProbe("s_des")
    budget = 1e-12 * 0.032 * 3e-4
    fields = {name: values.copy() for name, values in probe.c_by_species.items()}

    # The floor alone (initial at the model floor, final at the post-processing floor).
    assert probe._check_current_dac_residual_inventory("O2", 1e-37, budget) == budget
    assert capsys.readouterr().out == ""

    rise = 10 * budget
    assert probe._check_current_dac_residual_inventory("O2", 0.0, rise) == rise
    assert "O2 mass in step s_des" in capsys.readouterr().out
    np.testing.assert_array_equal(probe.c_by_species["O2"], fields["O2"])

    # Air-admitting steps may gain O2.
    assert _DacResidualProbe("ads")._check_current_dac_residual_inventory("O2", 0.0, rise) == rise
    assert capsys.readouterr().out == ""


def test_dac_residual_check_ignores_a_negative_closure_species_start(capsys):
    # N2 is the closure species: its step-start inventory c_tot - sum(ci) is
    # unfloored and solver undershoot can leave it slightly negative.
    probe = _DacResidualProbe("s_des")
    budget = 1e-12 * 0.028 * 3e-4

    assert probe._check_current_dac_residual_inventory("N2", -0.5 * budget, budget) == budget
    assert capsys.readouterr().out == ""

    # A real rise beyond the floor still warns, with the true change in the text.
    assert probe._check_current_dac_residual_inventory("N2", -0.5 * budget, 3 * budget) == 3 * budget
    assert f"change {3.5 * budget:+.3e} kg" in capsys.readouterr().out


def test_dac_residual_check_tolerates_the_floor_summed_in_another_order(capsys):
    # A species at the floor in every cell sums to the budget itself, but in a
    # different order, so it can exceed it on the last bits.
    probe = _DacResidualProbe("s_des")
    budget = 1e-12 * 0.032 * 3e-4
    at_floor = budget * (1 + 4 * np.finfo(float).eps)

    assert probe._check_current_dac_residual_inventory("O2", 0.0, at_floor) == at_floor
    assert capsys.readouterr().out == ""
