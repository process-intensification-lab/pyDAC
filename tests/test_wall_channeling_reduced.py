"""Packed-bed channeling: contract, closure and conservation.

The reduced mode resolves a radial voidage profile in storage and in the Ergun
drag, with uniform or closure-distributed prescribed inlet flow.  These
tests pin the three things that make it a model rather than a knob:

- storage and drag are two DIFFERENT fields of one profile, and nothing
  substitutes one for the other;
- the drag closure is anchored to physical layers, so it does not follow the
  numerical grid into the correlation's wall singularity;
- the inlet allocation and the in-bed law are one closure, not two.
"""

import contextlib
import io

import numpy as np
import pytest

from pydac.core.bed_architectures import (
    ErgunPressureDrop,
    PackedBedDesign,
    ParticleProperties,
    RadialErgunPressureDrop,
    channeling_inlet_flux_weights,
    drag_layer_faces,
    ergun_drag_coefficients,
    project_layer_field,
    resolve_bed_config,
    resolve_channeling_structure,
    solve_quadratic_drag_velocity,
    validate_channeling_support,
)
from pydac.core.bed_architectures.channeling import area_mean
from pydac.utils.solver.event_config import build_event_specs_from_model
from pydac.utils.solver.events import (
    avg_bed_loading_fraction_CO2_nd,
    make_cvode_events_callback,
)


# Synthetic transport inputs; conservation assertions do not depend on a fitted case.
RADIUS = 0.04
PARTICLE_DIAMETER = 0.002
MEAN_POROSITY = 0.388
VISCOSITY = 1.8e-5
DENSITY = 1.2
SUPERFICIAL_VELOCITY = 0.05

PARTICLE = ParticleProperties(
    density=1130.0,
    porosity=0.6,
    radius=PARTICLE_DIAMETER / 2,
    thermal_conductivity=0.35,
)


def reduced_design(**overrides):
    fields = {"eps_b": MEAN_POROSITY, "flow_model": "wall-channeling-reduced"}
    fields.update(overrides)
    return PackedBedDesign(**fields)


def structure(n_rings, design=None, radius=RADIUS):
    design = design or reduced_design()
    faces = np.linspace(0.0, radius, n_rings + 1)
    return resolve_channeling_structure(
        design,
        PARTICLE,
        radius=radius,
        particle_diameter=PARTICLE_DIAMETER,
        r_faces=faces,
    )


# ---------------------------------------------------------------------------
# 1. Public input contract
# ---------------------------------------------------------------------------

def test_omitted_selector_keeps_the_historical_canonical_mapping():
    """An ordinary case must not gain fields, or its fingerprint would move."""
    design = PackedBedDesign(eps_b=MEAN_POROSITY)
    assert design.to_config_dict() == {"eps_b": MEAN_POROSITY}
    assert design.resolved_flow_model == "uniform"
    assert design.compatibility_status == "native"


def test_reduced_mode_serializes_its_selector_and_resolved_closure():
    design = reduced_design()
    config = design.to_config_dict()
    assert config == {
        "eps_b": MEAN_POROSITY,
        "flow_model": "wall-channeling-reduced",
        "channeling": {
            "porosity_model": "de_klerk",
            "drag_layer_width_dp": 1.0,
            "inlet_distribution": "uniform",
        },
    }
    # The closure length is declared physics, so it round-trips into the case.
    assert PackedBedDesign(**config) == design
    assert resolve_bed_config(
        {"architecture": "packed_bed", "design": config, "material": None}
    )["design"] == config


def test_selector_spelling_is_normalized_but_near_misses_are_errors():
    assert (
        PackedBedDesign(
            eps_b=MEAN_POROSITY, flow_model="  Wall-Channeling-Reduced "
        ).resolved_flow_model
        == "wall-channeling-reduced"
    )
    for spelling in ("wall_channeling_reduced", "reduced", "channeling", ""):
        with pytest.raises(ValueError):
            PackedBedDesign(eps_b=MEAN_POROSITY, flow_model=spelling)


def test_full_mode_is_rejected_until_it_is_qualified():
    with pytest.raises(ValueError, match="not implemented yet"):
        PackedBedDesign(eps_b=MEAN_POROSITY, flow_model="wall-channeling")


def test_explicit_null_selector_is_an_error_not_a_default():
    with pytest.raises(ValueError, match="must not be null"):
        resolve_bed_config(
            {
                "architecture": "packed_bed",
                "design": {"eps_b": MEAN_POROSITY, "flow_model": None},
                "material": None,
            }
        )


def test_legacy_prescribed_profile_keeps_its_own_compatibility_status():
    """Omitted selector + radial profile is a replay status, not a fourth mode."""
    design = PackedBedDesign(
        eps_b=MEAN_POROSITY, radial_porosity_profile=[[0, 0.34], [1, 0.412]]
    )
    assert design.resolved_flow_model == "uniform"
    assert design.compatibility_status == "legacy-prescribed-profile"
    assert "flow_model" not in design.to_config_dict()


def test_explicit_uniform_rejects_the_radial_profile_and_channeling():
    with pytest.raises(ValueError, match="explicit `flow_model='uniform'`"):
        PackedBedDesign(
            eps_b=MEAN_POROSITY,
            flow_model="uniform",
            radial_porosity_profile=[[0, 0.34], [1, 0.412]],
        )
    with pytest.raises(ValueError, match="`channeling` belongs to"):
        PackedBedDesign(
            eps_b=MEAN_POROSITY,
            flow_model="uniform",
            channeling={"porosity_model": "de_klerk"},
        )


def test_reduced_mode_rejects_a_competing_inlet_profile():
    with pytest.raises(ValueError, match="derives its own inlet allocation"):
        reduced_design(inlet_flux_profile=[[0, 0.8], [0.8, 0.8], [1, 2]])


def test_porosity_model_and_profile_must_agree():
    with pytest.raises(ValueError, match="generates the radial"):
        reduced_design(radial_porosity_profile=[[0, 0.34], [1, 0.412]])
    with pytest.raises(ValueError, match="requires"):
        reduced_design(channeling={"porosity_model": "tabulated"})


def test_unknown_channeling_options_are_errors():
    with pytest.raises(ValueError, match="Unknown `channeling` fields"):
        reduced_design(channeling={"brinkman_viscosity_ratio": 2.0})
    with pytest.raises(ValueError, match="Unknown channeling porosity model"):
        reduced_design(channeling={"porosity_model": "vortmeyer_schuster"})


@pytest.mark.parametrize("width", [0.4, 2.5, 3.0, 0.0, -1.0])
def test_closure_length_outside_the_qualified_band_is_rejected(width):
    """3 d_p smears the whole channelling region; it is not a free parameter."""
    with pytest.raises(ValueError):
        reduced_design(channeling={"drag_layer_width_dp": width})


# ---------------------------------------------------------------------------
# 2. Resolved radial structure
# ---------------------------------------------------------------------------

def test_storage_and_drag_both_preserve_the_measured_mean_exactly():
    for n_rings in (25, 40, 50, 100):
        resolved = structure(n_rings)
        faces = resolved.r_faces
        assert area_mean(resolved.eps_storage, faces) == pytest.approx(
            MEAN_POROSITY, abs=1e-12
        )
        assert area_mean(resolved.eps_drag, faces) == pytest.approx(
            MEAN_POROSITY, abs=1e-12
        )
        assert area_mean(resolved.gas_holdup_factor, faces) == pytest.approx(
            1.0, abs=1e-12
        )
        assert area_mean(resolved.sorbent_factor, faces) == pytest.approx(
            1.0, abs=1e-12
        )


def test_de_klerk_core_is_solved_for_the_mean_not_taken_literally():
    """The core porosity must be lower than the area mean of a wall-enriched bed."""
    resolved = structure(40)
    assert resolved.core_porosity < MEAN_POROSITY
    literal = PackedBedDesign(
        eps_b=resolved.core_porosity, flow_model="wall-channeling-reduced"
    )
    # The same correlation evaluated with eps_b taken literally overshoots.
    assert structure(40, design=literal).mean_porosity < MEAN_POROSITY


def test_drag_layers_are_physical_and_do_not_follow_the_grid():
    """The wall layer is the same interval, hence the same eps, on every grid."""
    wall_layer = {float(structure(n).layer_porosity[-1]) for n in (25, 40, 50, 100)}
    assert len(wall_layer) == 1
    assert wall_layer.pop() > MEAN_POROSITY
    assert structure(40).layer_porosity.size == 20


def test_drag_field_is_grid_independent_on_layer_conforming_grids():
    design = reduced_design()
    layer_faces = drag_layer_faces(RADIUS, PARTICLE_DIAMETER)
    reference = None
    for per_layer in (1, 2, 4):
        faces = np.concatenate(
            [
                np.linspace(layer_faces[i], layer_faces[i + 1], per_layer + 1)[:-1]
                for i in range(layer_faces.size - 1)
            ]
            + [layer_faces[-1:]]
        )
        resolved = resolve_channeling_structure(
            design,
            PARTICLE,
            radius=RADIUS,
            particle_diameter=PARTICLE_DIAMETER,
            r_faces=faces,
        )
        blocks = resolved.eps_drag.reshape(-1, per_layer)
        # Every cell of one layer carries that layer's mean. The residual is
        # pure floating point: a `linspace` face can land an ulp off the layer
        # boundary and pick up a vanishing overlap with its neighbour.
        assert np.max(np.abs(blocks - blocks[:, :1])) < 1e-15
        if reference is None:
            reference = blocks[:, 0]
        else:
            np.testing.assert_allclose(blocks[:, 0], reference, atol=1e-15)


def test_the_drag_field_never_reaches_the_ergun_singularity():
    """Storage follows the profile to eps -> 1; drag must not, or A and B vanish."""
    wall_storage = []
    for n_rings in (25, 40, 100, 320):
        resolved = structure(n_rings)
        assert np.all(resolved.eps_drag < 0.48)
        assert np.all(resolved.eps_drag > 0.36)
        wall_storage.append(float(resolved.eps_storage[-1]))
    # Storage keeps its OWN grid sensitivity, rising towards the correlation's
    # eps(R) = 1 as the wall ring narrows.  W does not remove that, and it is an
    # open qualification item (probe temperatures, wall heat flux, energy
    # balance), not a defect of the drag closure.
    assert wall_storage == sorted(wall_storage)
    assert wall_storage[0] < 0.46 < 0.9 < wall_storage[-1]
    # The drag closure, by contrast, is inert: same physical wall layer, same
    # voidage, on every one of those grids.
    assert len({round(float(structure(n).layer_porosity[-1]), 12) for n in (25, 320)}) == 1


def test_area_overlap_projection_conserves_the_mean_on_any_grid():
    layer_faces = np.array([0.0, 0.01, 0.025, 0.0401])
    layer_values = np.array([0.37, 0.40, 0.47])
    faces = np.linspace(0.0, 0.0401, 18)
    projected = project_layer_field(layer_values, layer_faces, faces)
    assert area_mean(projected, faces) == pytest.approx(
        area_mean(layer_values, layer_faces), abs=1e-15
    )
    assert projected.min() >= layer_values.min() - 1e-15
    assert projected.max() <= layer_values.max() + 1e-15


def test_tabulated_profile_must_already_match_the_declared_mean():
    # eps = 0.34 + 0.072 r/R has exact cylindrical mean 0.388.
    good = reduced_design(
        channeling={"porosity_model": "tabulated"},
        radial_porosity_profile=[[0, 0.34], [1, 0.412]],
    )
    resolved = structure(20, design=good)
    assert area_mean(resolved.eps_storage, resolved.r_faces) == pytest.approx(
        MEAN_POROSITY, abs=1e-12
    )
    bad = PackedBedDesign(
        eps_b=0.30,
        flow_model="wall-channeling-reduced",
        channeling={"porosity_model": "tabulated"},
        radial_porosity_profile=[[0, 0.34], [1, 0.412]],
    )
    with pytest.raises(ValueError, match="rather than having it shifted"):
        structure(20, design=bad)


# ---------------------------------------------------------------------------
# 3. Radial Ergun law
# ---------------------------------------------------------------------------

LAW_KWARGS = dict(
    viscosity=VISCOSITY,
    particle_diameter=PARTICLE_DIAMETER,
    bed_length=0.2,
    reference_velocity=SUPERFICIAL_VELOCITY,
    reference_pressure=2.3e5,
)


def test_uniform_drag_field_recovers_the_native_scalar_law_exactly():
    scalar = ErgunPressureDrop(eps_flow=MEAN_POROSITY, **LAW_KWARGS)
    radial = RadialErgunPressureDrop(eps_drag=np.full(40, MEAN_POROSITY), **LAW_KWARGS)
    gradient = np.linspace(-3.0, 3.0, 21).reshape(21, 1, 1) * np.ones((1, 40, 1))
    density = np.full((21, 40, 1), DENSITY)
    np.testing.assert_array_equal(
        radial.velocity_from_gradient(gradient, density),
        scalar.velocity_from_gradient(gradient, density),
    )


def test_coefficients_carry_an_explicit_radial_axis():
    radial = RadialErgunPressureDrop(eps_drag=structure(40).eps_drag, **LAW_KWARGS)
    assert np.asarray(radial.linear_coefficient).shape == (1, 40, 1)
    assert np.asarray(radial.inertial_coefficient_per_density).shape == (1, 40, 1)
    coefficients = np.asarray(radial.linear_coefficient).reshape(-1)
    # The wall layer is less resistant, but only by a factor, never by orders:
    # that ratio is the whole point of the layer closure and is grid-independent.
    assert coefficients[-1] / coefficients[0] == pytest.approx(0.352, abs=5e-3)


def test_zero_and_signed_flow_behave():
    radial = RadialErgunPressureDrop(eps_drag=structure(40).eps_drag, **LAW_KWARGS)
    density = np.full((3, 40, 1), DENSITY)
    assert np.all(radial.velocity_from_gradient(np.zeros((3, 40, 1)), density) == 0.0)
    forward = radial.velocity_from_gradient(np.full((3, 40, 1), 1.5), density)
    backward = radial.velocity_from_gradient(np.full((3, 40, 1), -1.5), density)
    np.testing.assert_array_equal(forward, -backward)


@pytest.mark.parametrize("eps", [np.full(4, 1.0), np.full(4, 0.0), np.array([0.4])])
def test_a_degenerate_drag_field_is_rejected(eps):
    with pytest.raises(ValueError):
        RadialErgunPressureDrop(eps_drag=eps, **LAW_KWARGS)


# ---------------------------------------------------------------------------
# 4. Inlet allocation from the same closure
# ---------------------------------------------------------------------------

def test_inlet_weights_preserve_the_feed_and_follow_the_closure():
    resolved = structure(40)
    weights = channeling_inlet_flux_weights(
        resolved,
        viscosity=VISCOSITY,
        density=DENSITY,
        particle_diameter=PARTICLE_DIAMETER,
        mean_velocity=SUPERFICIAL_VELOCITY,
    )
    assert area_mean(weights, resolved.r_faces) == pytest.approx(1.0, abs=1e-14)
    assert 0 < weights[0] < 1 < weights[-1]


def test_a_zero_feed_has_no_allocation_to_solve():
    with pytest.raises(ValueError, match="skip the normalization"):
        channeling_inlet_flux_weights(
            structure(20),
            viscosity=VISCOSITY,
            density=DENSITY,
            particle_diameter=PARTICLE_DIAMETER,
            mean_velocity=0.0,
        )


def test_reverse_feed_mirrors_the_same_allocation():
    resolved = structure(20)
    kwargs = dict(
        viscosity=VISCOSITY,
        density=DENSITY,
        particle_diameter=PARTICLE_DIAMETER,
    )
    forward = channeling_inlet_flux_weights(
        resolved, mean_velocity=SUPERFICIAL_VELOCITY, **kwargs
    )
    backward = channeling_inlet_flux_weights(
        resolved, mean_velocity=-SUPERFICIAL_VELOCITY, **kwargs
    )
    np.testing.assert_allclose(forward, backward, rtol=1e-12)


@pytest.mark.parametrize(
    "mean_velocity",
    [
        SUPERFICIAL_VELOCITY,
        -SUPERFICIAL_VELOCITY,
        1.0e-6,
        1.0e-8,
        1.0e-10,
        -1.0e-10,
    ],
)
def test_a_small_nonzero_feed_is_allocated_and_preserved(mean_velocity):
    """The root bracket has to start at zero.

    The gradient a feed needs shrinks with the feed, so a fixed positive lower
    bound left both endpoints with the same residual sign and the allocation
    raised instead of solving.  Exact zero is a different case and is refused on
    purpose, above.

    """
    resolved = structure(40)
    weights = channeling_inlet_flux_weights(
        resolved,
        viscosity=VISCOSITY,
        density=DENSITY,
        particle_diameter=PARTICLE_DIAMETER,
        mean_velocity=mean_velocity,
    )
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert area_mean(weights, resolved.r_faces) == pytest.approx(1.0, abs=1e-14)
    # The integrated feed is the declared one, sign included.
    assert area_mean(weights * mean_velocity, resolved.r_faces) == pytest.approx(
        mean_velocity, rel=1e-14
    )


@pytest.mark.parametrize("mean_velocity", [1e-12, -1e-12, 1e-20, -1e-20, 1e-100, -1e-100, 1e-300])
def test_tiny_signed_feed_has_a_finite_darcy_allocation(mean_velocity):
    resolved = structure(40)
    kwargs = dict(viscosity=VISCOSITY, density=DENSITY,
                  particle_diameter=PARTICLE_DIAMETER)
    weights = channeling_inlet_flux_weights(resolved, mean_velocity=mean_velocity, **kwargs)
    a, _ = ergun_drag_coefficients(resolved.eps_drag, **kwargs)
    expected = (1/a) / area_mean(1/a, resolved.r_faces)
    np.testing.assert_allclose(weights, expected, rtol=1e-10, atol=0)
    assert area_mean(weights * mean_velocity, resolved.r_faces) / mean_velocity == pytest.approx(1, rel=1e-14)


@pytest.mark.parametrize("name,value", [("mean_velocity", np.nan), ("mean_velocity", np.inf),
    ("viscosity", -1), ("density", np.nan), ("particle_diameter", 0)])
def test_invalid_allocation_input_is_explicitly_rejected(name, value):
    kwargs = dict(viscosity=VISCOSITY, density=DENSITY,
                  particle_diameter=PARTICLE_DIAMETER, mean_velocity=SUPERFICIAL_VELOCITY)
    kwargs[name] = value
    with pytest.raises(ValueError, match="finite"):
        channeling_inlet_flux_weights(structure(40), **kwargs)


def test_a_vanishing_feed_approaches_the_darcy_allocation():
    """As inertia vanishes the weights converge to normalized inverse resistance."""
    resolved = structure(40)
    viscous, _ = ergun_drag_coefficients(
        resolved.eps_drag,
        viscosity=VISCOSITY,
        density=DENSITY,
        particle_diameter=PARTICLE_DIAMETER,
    )
    darcy = (1.0 / viscous) / area_mean(1.0 / viscous, resolved.r_faces)
    kwargs = dict(
        viscosity=VISCOSITY,
        density=DENSITY,
        particle_diameter=PARTICLE_DIAMETER,
    )
    vanishing = channeling_inlet_flux_weights(
        resolved, mean_velocity=1.0e-10, **kwargs
    )
    nominal = channeling_inlet_flux_weights(
        resolved, mean_velocity=SUPERFICIAL_VELOCITY, **kwargs
    )
    np.testing.assert_allclose(vanishing, darcy, rtol=1.0e-4)
    # At the real feed inertia is not negligible, so the check above is not vacuous.
    assert np.max(np.abs(nominal / darcy - 1.0)) > 0.1


# ---------------------------------------------------------------------------
# 5. Support matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(architecture="other_bed", is_1d=False, Nr=40, Nz=20),
        dict(architecture="packed_bed", is_1d=True, Nr=1, Nz=20),
        dict(architecture="packed_bed", is_1d=False, Nr=1, Nz=20),
        dict(architecture="packed_bed", is_1d=False, Nr=40, Nz=2),
    ],
)
def test_unsupported_combinations_are_rejected_not_silently_averaged(kwargs):
    with pytest.raises(ValueError):
        validate_channeling_support("wall-channeling-reduced", **kwargs)


# ---------------------------------------------------------------------------
# 6. Stable small-driving-force inversion
# ---------------------------------------------------------------------------

def stable_drag_velocity(driving_force, linear, inertial):
    """``u = sgn(G) 2|G| / (a + sqrt(a^2 + 4b|G|))``, the cancellation-free form."""
    force = np.asarray(driving_force, dtype=float)
    magnitude = np.abs(force)
    return (
        np.sign(force)
        * 2.0
        * magnitude
        / (linear + np.sqrt(linear**2 + 4.0 * inertial * magnitude))
    )


@pytest.mark.parametrize("driving_force", [3.0e2, 1.0, 1.0e-3])
def test_production_inversion_is_accurate_over_the_reachable_range(driving_force):
    """At any gradient this bed actually sees, the two forms agree to 1e-10."""
    linear, inertial = 1747.0, 1.24e4
    produced = float(solve_quadratic_drag_velocity(driving_force, linear, inertial))
    reference = float(stable_drag_velocity(driving_force, linear, inertial))
    assert produced == pytest.approx(reference, rel=1e-10)


@pytest.mark.parametrize("force", [0.0, 1e-6, -1e-6, 1e-12, -1e-12, 1e-100, -1e-100, 1e-300])
def test_inversion_resolves_zero_and_vanishing_signed_gradients(force):
    a, b = 1747.0, 1.24e4
    u = float(solve_quadratic_drag_velocity(force, a, b))
    if force == 0:
        assert u == 0
    else:
        assert (a*u + b*u*abs(u)) / force == pytest.approx(1, rel=1e-14)
        assert np.sign(u) == np.sign(force)


# ---------------------------------------------------------------------------
# 7. Integration with the column model
# ---------------------------------------------------------------------------

def build_model(*, step="startup", nz=20, nr=40, initial_state=None, **design):
    from pydac.core.column_model_factory import ColumnModelFactory
    from pydac.config import load_experiment
    from pydac.config.cycle import CycleConfig
    from tests.neutral_packed_bed import neutral_profile

    profile = neutral_profile()
    simulation = profile["simulation"]
    simulation["grid"].update(nz=nz, nr=nr)
    # Generated channeling must not inherit competing prescribed profiles.
    simulation["bed"]["design"].pop("radial_porosity_profile", None)
    simulation["bed"]["design"].pop("inlet_flux_profile", None)
    simulation["bed"]["design"].update(**design)
    cycle = CycleConfig.from_experiment(load_experiment(default_profile=profile))
    with contextlib.redirect_stdout(io.StringIO()) as log:
        model = ColumnModelFactory(cycle).build_column_model(
            step=step,
            initial_state=initial_state,
            solver_options=cycle.solver_for_step(use_sparsity=False),
        )
    return model, log.getvalue()


def warm_start_from(model):
    """A valid previous-step state, taken from an initialized startup model."""
    return {
        "ci": np.asarray(model.ci).reshape(model.Nz, model.Nr, model.Nc_ci).copy(),
        "qi": np.asarray(model.qi).reshape(model.Nz, model.Nr, model.Nc_solid).copy(),
        "c_tot": np.asarray(model.c_tot).reshape(model.Nz, model.Nr).copy(),
        "T": np.asarray(model.T).reshape(model.Nz, model.Nr).copy(),
        "T_wall": np.asarray(model.Tw).reshape(model.Nz).copy(),
        "Nz": model.Nz,
        "Nr": model.Nr,
        "layout_signature": model.state_layout.signature,
    }


@pytest.fixture(scope="module")
def reduced_model():
    return build_model(flow_model="wall-channeling-reduced")


@pytest.fixture(scope="module")
def distributed_model():
    return build_model(
        flow_model="wall-channeling-reduced",
        channeling={"inlet_distribution": "distributed"},
    )


def test_the_default_inlet_is_uniform_and_conserves_the_feed(reduced_model):
    """The interior closure, not the boundary, develops the channeling."""
    model, log = reduced_model
    assert model.bed_design.channeling.inlet_distribution == "uniform"
    assert model.inlet_flux_source == "uniform"
    np.testing.assert_array_equal(model.inlet_flux_weights, np.ones(model.Nr))
    assert "Channeling inlet allocation" not in log
    # Storage and drag stay local: only the boundary allocation differs.
    assert model.channeling_structure is not None
    assert model.pressure_drop_law.name == "ergun_radial"


def test_unknown_inlet_distribution_is_an_error():
    with pytest.raises(ValueError, match="inlet_distribution"):
        reduced_design(channeling={"inlet_distribution": "profiled"})


def test_the_model_binds_the_radial_law_and_the_local_structure(reduced_model):
    model, _ = reduced_model
    assert model.flow_model == "wall-channeling-reduced"
    assert model.flow_model_compatibility == "native"
    assert model.pressure_drop_law.name == "ergun_radial"
    assert isinstance(model.pressure_drop_law, RadialErgunPressureDrop)
    # Storage factors come from the resolved profile, not from the legacy path.
    np.testing.assert_array_equal(
        model.ring_gas_holdup_factor, model.channeling_structure.gas_holdup_factor
    )
    np.testing.assert_array_equal(
        model.ring_sorbent_factor, model.channeling_structure.sorbent_factor
    )
    # The scalar means every retained closure still uses are untouched.
    assert model.eps_flow == pytest.approx(MEAN_POROSITY)


def test_the_inlet_allocation_and_the_in_bed_law_are_one_closure(distributed_model):
    """Not merely consistent: the same function of the same drag field."""
    from scipy.optimize import brentq

    model, log = distributed_model
    assert model.inlet_flux_source == "reduced_closure"
    areas = np.asarray(model.A_rings)
    density = float(np.asarray(model.rho_g_mix).item())

    def mean_velocity(gradient_nd):
        field = np.full((model.Nz + 1, model.Nr, 1), gradient_nd)
        velocity = np.asarray(
            model.pressure_drop_law.velocity_from_gradient(
                field, np.full_like(field, density)
            )
        )[0, :, 0]
        return float(np.sum(velocity * areas) / np.sum(areas)) * model.u_ref

    gradient = brentq(
        lambda g: mean_velocity(g) - model.u_inlet, -1.0, -1e-9, xtol=1e-16
    )
    field = np.full((model.Nz + 1, model.Nr, 1), gradient)
    velocity = np.asarray(
        model.pressure_drop_law.velocity_from_gradient(
            field, np.full_like(field, density)
        )
    )[0, :, 0]
    shape = velocity / (np.sum(velocity * areas) / np.sum(areas))
    np.testing.assert_allclose(shape, model.inlet_flux_weights, atol=1e-13)

    assert -gradient * model.p_ref / model.L_b > 0
    assert "Channeling inlet allocation from the reduced closure" in log


def test_the_resolved_context_declares_the_mode_and_its_transport_policy(
    reduced_model,
):
    model, _ = reduced_model
    flow = model.bed_architecture_context["resolved"]["flow"]
    assert flow["model"] == "wall-channeling-reduced"
    assert flow["compatibility_status"] == "native"
    assert flow["inlet_allocation_source"] == "uniform"
    assert flow["local_structure"] == "storage_and_drag"
    # Dispersion, conductivity and wall HTC stay on the bed mean; the context
    # says so rather than letting a reader assume local wall transport.
    assert flow["bulk_transport_policy"] == "mean_eps_step_bulk"
    channeling = flow["channeling"]
    assert channeling["porosity_model"] == "de_klerk"
    assert channeling["drag_layer_width_dp"] == 1.0
    assert channeling["drag_layer_count"] == len(drag_layer_faces(model.R_b, model.d_p)) - 1
    assert channeling["storage_area_mean"] == pytest.approx(MEAN_POROSITY, abs=1e-12)
    assert channeling["drag_area_mean"] == pytest.approx(MEAN_POROSITY, abs=1e-12)
    assert channeling["generator_revision"]


def test_the_persisted_pressure_law_is_per_ring_and_reconstructs_per_ring(
    reduced_model,
):
    from pydac.utils.postprocessing.cyclic_profiles import (
        pressure_drop_parameters_from_metadata,
        solve_pressure_drop_velocity,
    )
    from pydac.utils.solver import pressure_drop_metadata_from_model

    model, _ = reduced_model
    metadata = pressure_drop_metadata_from_model(model)
    assert metadata["model"] == "ergun_radial"
    assert metadata["schema_version"] == 2
    assert metadata["radial_axis"]["size"] == model.Nr
    assert len(metadata["linear_coefficient"]) == model.Nr

    parameters = pressure_drop_parameters_from_metadata(
        {"pressure_drop": metadata}, "startup"
    )
    assert np.asarray(parameters["linear_coefficient"]).shape == (model.Nr,)
    gradient = np.full((model.Nz + 1, model.Nr, 2), -300.0)
    density = np.full((model.Nz + 1, model.Nr, 2), DENSITY)
    velocity = solve_pressure_drop_velocity(gradient, density, **parameters)
    # A scalar reconstruction would give a radially flat field; this one channels.
    assert np.abs(velocity[3, -1, 0]) > 1.5 * np.abs(velocity[3, 0, 0])
    assert np.all(np.sign(velocity) == -1.0)

    with pytest.raises(ValueError, match="do not match the declared radial axis"):
        broken = dict(metadata)
        broken["linear_coefficient"] = metadata["linear_coefficient"][:-1]
        pressure_drop_parameters_from_metadata({"pressure_drop": broken}, "startup")


def test_the_native_default_still_writes_the_scalar_schema():
    model, _ = build_model(nz=6, nr=5)
    from pydac.utils.solver import pressure_drop_metadata_from_model

    metadata = pressure_drop_metadata_from_model(model)
    assert metadata["model"] == "ergun"
    assert "schema_version" not in metadata
    assert isinstance(metadata["linear_coefficient"], float)
    assert model.flow_model == "uniform"
    assert model.channeling_structure is None
    np.testing.assert_array_equal(model.ring_gas_holdup_factor, np.ones(5))


def test_cool_keeps_its_warm_start_instead_of_an_unqualified_transformation():
    """All flow models preserve the preceding state at the cooldown boundary."""
    from pydac.core.column_model_factory import ColumnModelFactory
    from pydac.config import load_experiment
    from pydac.config.cycle import CycleConfig

    def cool_model(**design):
        # The framework default cycle, whose `cool` step carries a complete
        # composition for all cycle steps.
        settings = {"grid.nz": 6, "grid.nr": 5}
        settings.update({f"bed.design.{k}": v for k, v in design.items()})
        cycle = CycleConfig.from_experiment(load_experiment(settings=settings))
        factory = ColumnModelFactory(cycle)
        options = cycle.solver_for_step(use_sparsity=False)
        with contextlib.redirect_stdout(io.StringIO()):
            seed = factory.build_column_model(step="startup", solver_options=options)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            model = factory.build_column_model(
                step="cool",
                initial_state=warm_start_from(seed),
                solver_options=options,
            )
        for field in ("ci", "qi", "c_tot", "T", "Tw"):
            np.testing.assert_array_equal(np.asarray(getattr(model, field)).ravel(), np.asarray(getattr(seed, field)).ravel())
        return model, log.getvalue()

    native, _ = cool_model()
    assert native.relax_cool_hydraulics is False
    assert native.flatten_cool_axial_profiles is False

    reduced, log = cool_model(flow_model="wall-channeling-reduced")
    assert reduced.flow_model == "wall-channeling-reduced"
    assert reduced.relax_cool_hydraulics is False
    assert reduced.flatten_cool_axial_profiles is False
    # Both ends are closed, so there is no prescribed feed to allocate.
    assert reduced.u_inlet == 0.0
    assert reduced.inlet_flux_source == "zero_feed"


STEP_INLET_SOURCES = [
    ("startup", "reduced_closure"),
    ("ads", "reduced_closure"),
    ("s_purge", "reduced_closure"),
    ("s_des", "reduced_closure"),
    ("press", "upstream_pressure"),
    ("blow", "zero_feed"),
    ("heat", "zero_feed"),
    ("cool", "zero_feed"),
    ("w_purge", "zero_feed"),
    ("w_des", "zero_feed"),
]


@pytest.mark.parametrize(
    ("distribution", "step", "expected_source"),
    [("distributed", step, source) for step, source in STEP_INLET_SOURCES]
    + [
        (
            "uniform",
            step,
            "uniform" if source == "reduced_closure" else source,
        )
        for step, source in STEP_INLET_SOURCES
    ],
)
def test_step_families_select_closed_pressure_or_prescribed_inlet(
    distribution, step, expected_source
):
    """Closed inlets bypass allocation; pressure and prescribed feed stay distinct."""
    from pydac.core.column_model_factory import ColumnModelFactory
    from pydac.config import load_experiment
    from pydac.config.cycle import CycleConfig

    settings = {
        "grid.nz": 6,
        "grid.nr": 5,
        "bed.design.flow_model": "wall-channeling-reduced",
        "bed.design.channeling.inlet_distribution": distribution,
    }
    cycle = CycleConfig.from_experiment(load_experiment(settings=settings))
    factory = ColumnModelFactory(cycle)
    options = cycle.solver_for_step(use_sparsity=False)
    with contextlib.redirect_stdout(io.StringIO()):
        seed = factory.build_column_model(step="startup", solver_options=options)
        model = factory.build_column_model(
            step=step,
            initial_state=warm_start_from(seed),
            solver_options=options,
        )

    assert model.inlet_flux_source == expected_source
    if expected_source == "zero_feed":
        assert float(model.u_inlet) == 0.0
        np.testing.assert_array_equal(model.inlet_flux_weights, np.ones(model.Nr))
    elif expected_source == "reduced_closure":
        # Far above the limit, and allocated rather than left uniform.
        assert abs(float(model.u_inlet)) > 1.0e-3
        assert not np.allclose(model.inlet_flux_weights, 1.0)
    elif expected_source == "uniform":
        assert abs(float(model.u_inlet)) > 1.0e-3
        np.testing.assert_array_equal(model.inlet_flux_weights, np.ones(model.Nr))


def test_reduced_mode_rhs_conserves_each_species_and_the_sensible_inventory():
    """The local factors must enter storage and source exactly once."""
    model, _ = build_model(nz=6, nr=5, flow_model="wall-channeling-reduced")
    z = np.asarray(model.z_c)[:, None]
    r = np.asarray(model.r_c)[None, :]
    temperature = model.T_ref * (1.1 + 0.1 * (1 - z) * (1 - r))
    concentration = (
        model.c_tot_ref * (1 + 0.001 * (1 - z)) * model.T_ref / temperature
    )
    state_values = {f.name: np.zeros(f.shape) for f in model.state_layout.fields}
    state_values.update(
        c_tot=concentration / model.c_tot_ref,
        c_CO2=concentration
        * (0.03 + 0.06 * (1 - z) * (1 - r))
        / model.c_ref_by_species["CO2"],
        q_CO2=np.broadcast_to(0.3 + 0.2 * z, (model.Nz, model.Nr))
        / model.q_ref_by_species["CO2"],
        q_N2=np.broadcast_to(0.2 + 0.1 * r, (model.Nz, model.Nr))
        / model.q_ref_by_species["N2"],
        T_bed=temperature / model.T_ref,
        T_wall=np.full(model.Nz, 1.02),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        rhs = model.pde_system_kpis(0, model.state_layout.pack(state_values)).copy()

    def rate(name):
        field = model.state_layout.field(name)
        return (
            rhs[field.state_slice].reshape(field.shape)
            * model.state_scale_by_field[name]
            / model.tau_hyd
        )

    volume = (
        np.diff(model.z_f)[:, None] * model.L_b * np.asarray(model.A_rings)[None, :]
    )
    gas = model.eps_total * model._gas_2d
    sorbent = model.rho_sorbent * model._solid_2d
    for species in ("CO2", "N2"):
        dc = rate("c_CO2") if species == "CO2" else rate("c_tot") - rate("c_CO2")
        storage_rate = np.sum(volume * (gas * dc + sorbent * rate("q_" + species)))
        flow_rate = rate("m_" + species).item() / model.MW[model.gas_index[species]]
        np.testing.assert_allclose(storage_rate, flow_rate, rtol=1e-11, atol=1e-13)

    gas_sensible = (
        sum(
            model.acc_g[i]
            / model.c_ref[i]
            * (rate("c_CO2") if species == "CO2" else rate("c_tot") - rate("c_CO2"))
            for i, species in enumerate(model.gas_species)
        )
        * model._gas_2d
    )
    ads_sensible = (
        sum(
            model.acc_ads[i]
            / model.q_ref_by_species[species]
            * rate("q_" + species)
            for i, species in enumerate(model.adsorbates)
        )
        * model._solid_2d
    )
    expected = (
        -(temperature / model.T_ref - model.T_enthalpy_ref_nd)
        * (gas_sensible + ads_sensible)
        * model.tau_hyd
    )
    np.testing.assert_allclose(
        model._source_extra_buf.reshape(model.Nz, model.Nr),
        expected,
        rtol=1e-11,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# 8. Consumers of the local structure: loading events and the RHS diagnostic
# ---------------------------------------------------------------------------

def radially_loaded_state(model):
    """The model's own initial state, with a deliberately radial CO2 loading."""
    state = np.zeros(model.state_layout.state_size)
    pde = np.asarray(model.get_initial_state())
    state[: pde.size] = pde
    field = model.state_layout.field("q_CO2")
    state[field.state_slice] = np.broadcast_to(
        0.3 + 0.5 * np.asarray(model.r_c)[None, :], (model.Nz, model.Nr)
    ).ravel()
    return state


def loading_fraction_by_weight(model, state, weights):
    """Independently summed mean loading fraction under the given cell weights."""
    field = model.state_layout.field("q_CO2")
    loading = state[field.state_slice].reshape(field.shape)
    return float(np.sum(loading * weights) / np.sum(weights))


def test_the_loading_event_weights_local_sorbent_mass():
    """A capacity fraction is per unit sorbent mass, not per unit bed volume.

    The wall rings are the least packed, so they hold less sorbent than their
    volume suggests; weighting by volume overstates how full the bed is and
    would terminate the step early.
    """
    model, _ = build_model(nz=6, nr=40, flow_model="wall-channeling-reduced")
    state = radially_loaded_state(model)
    sorbent_weights = np.asarray(model.V_cells) * model._solid_2d

    metric = avg_bed_loading_fraction_CO2_nd(
        state, layout=model.state_layout, sorbent_weights=sorbent_weights
    )
    assert metric == pytest.approx(
        loading_fraction_by_weight(model, state, sorbent_weights), rel=1e-15
    )
    assert metric < loading_fraction_by_weight(model, state, np.asarray(model.V_cells))
    # Volume weighting overstates loading in the less packed wall rings.
    geometric = loading_fraction_by_weight(model, state, np.asarray(model.V_cells))
    assert geometric == pytest.approx(0.63328125, abs=1e-12)


def test_the_loading_event_threshold_follows_the_sorbent_weighted_fraction():
    """The wiring, not just the helper: the built event must carry the weights."""
    model, _ = build_model(nz=6, nr=40, flow_model="wall-channeling-reduced")
    state = radially_loaded_state(model)
    model.termination_mode = "event"
    model.termination_metric = "avg_bed_loading_fraction_CO2"
    model.termination_direction = "above"
    model.termination_threshold_source = None
    # A threshold between the two answers separates them.
    weighted = loading_fraction_by_weight(model, state, model.V_cells * model._solid_2d)
    geometric = loading_fraction_by_weight(model, state, model.V_cells)
    model.termination_threshold_value = (weighted + geometric) / 2
    model.termination_threshold_offset = 0.0

    specs = build_event_specs_from_model(
        model=model,
        step="ads",
        A_rings=model.A_rings,
        V_cells=model.V_cells,
        p_ref=model.p_ref,
        T_ref=model.T_ref,
    )
    assert specs[0]["metric_fn"](state) == pytest.approx(
        weighted, abs=1e-12
    )
    residuals = np.zeros(len(specs))
    make_cvode_events_callback(specs)(0.0, state, residuals)
    # The two weightings straddle the threshold: on sorbent mass the step keeps
    # running, on cell volume it would have terminated here.
    geometric_residual = loading_fraction_by_weight(
        model, state, np.asarray(model.V_cells)
    ) - float(model.termination_threshold_value)
    assert residuals[0] < 0.0 < geometric_residual


def test_the_uniform_bed_keeps_its_loading_answer_unchanged():
    """The uniform limit must be untouched: the factors are exactly one there."""
    model, _ = build_model(nz=6, nr=40)
    state = radially_loaded_state(model)
    np.testing.assert_array_equal(model._solid_2d, np.ones((model.Nz, model.Nr)))
    metric = avg_bed_loading_fraction_CO2_nd(
        state,
        layout=model.state_layout,
        sorbent_weights=np.asarray(model.V_cells) * model._solid_2d,
    )
    assert metric == loading_fraction_by_weight(
        model, state, np.asarray(model.V_cells)
    )
    assert metric == pytest.approx(0.63328125, abs=1e-12)


@pytest.mark.parametrize("flow_model", ["wall-channeling-reduced", "uniform"])
def test_the_species_diagnostic_decomposes_the_production_rhs(flow_model):
    """The attribution must carry the RHS's own local factors, exactly once.

    The reduced decomposition previously divided by the mean `eps_total` only,
    omitting the local gas-storage and sorbent factors, and disagreed with its
    own derivative by 16.8 mol/(m3 s) while uniform was already exact.  Until
    the sum of the reported terms IS the returned derivative, the diagnostic
    cannot be used to attribute a cooling or composition failure.
    """
    model, _ = build_model(nz=6, nr=40, flow_model=flow_model)
    state = radially_loaded_state(model)
    records = []
    model.set_species_balance_diagnostic_callback(records.append)
    with contextlib.redirect_stdout(io.StringIO()):
        rhs = np.asarray(model.pde_system_kpis(0, state)).copy()
    diagnostic = records[-1]

    assert diagnostic["decomposition_max_abs_error_mol_m3_s"] < 1.0e-12
    if flow_model == "wall-channeling-reduced":
        # Otherwise the check above would be satisfied by a bed with no structure.
        # The gas factor moves least because the particle pore gas damps it.
        assert np.max(np.abs(model._gas_2d - 1.0)) > 0.01
        assert np.max(np.abs(model._solid_2d - 1.0)) > 0.01

    def production_rate(name, index):
        field = model.state_layout.field(name)
        return (
            rhs[field.state_slice].reshape(field.shape)[index]
            * model.state_scale_by_field[name]
            / model.tau_hyd
        )

    for key in ("minimum_closure_cell", "most_negative_closure_derivative_cell"):
        cell = diagnostic[key]
        index = tuple(cell["index"])
        independent_rates = 0.0
        for species, terms in cell["independent_derivatives_mol_m3_s"].items():
            components = sum(
                value for name, value in terms.items() if name != "total"
            )
            assert components == pytest.approx(terms["total"], rel=1e-9, abs=1e-12)
            produced = production_rate(f"c_{species}", index)
            assert terms["total"] == pytest.approx(produced, rel=1e-11, abs=1e-14)
            independent_rates += produced
        closure = cell["closure_derivative_mol_m3_s"]
        components = sum(value for name, value in closure.items() if name != "total")
        assert components == pytest.approx(closure["total"], rel=1e-9, abs=1e-12)
        # The closure species is the total minus the independent ones, and that
        # total is the RHS's own, not a quantity the diagnostic reconstructed.
        assert closure["total"] == pytest.approx(
            production_rate("c_tot", index) - independent_rates, rel=1e-11, abs=1e-14
        )


def test_the_reduced_case_is_selectable_through_dotted_settings():
    """The API/CLI path must reach the same physics as the JSON block."""
    from pydac.config import load_experiment
    from tests.neutral_packed_bed import neutral_profile

    profile = neutral_profile()
    design = profile["simulation"]["bed"]["design"]
    design.pop("radial_porosity_profile", None)
    design.pop("inlet_flux_profile", None)
    experiment = load_experiment(
        default_profile=profile,
        settings={"bed.design.flow_model": "wall-channeling-reduced"},
    )
    resolved = experiment.simulation.bed["design"]
    assert resolved["flow_model"] == "wall-channeling-reduced"
    assert resolved["channeling"] == {
        "porosity_model": "de_klerk",
        "drag_layer_width_dp": 1.0,
        "inlet_distribution": "uniform",
    }

    # The channeling fields are settable one by one, as the CLI documents.
    experiment = load_experiment(
        default_profile=profile,
        settings={
            "bed.design.flow_model": "wall-channeling-reduced",
            "bed.design.channeling.drag_layer_width_dp": 1.5,
            "bed.design.channeling.inlet_distribution": "distributed",
        },
    )
    assert experiment.simulation.bed["design"]["channeling"] == {
        "porosity_model": "de_klerk",
        "drag_layer_width_dp": 1.5,
        "inlet_distribution": "distributed",
    }




# ---------------------------------------------------------------------------
# Giese (1998) radial voidage — the published alternative to de Klerk
# ---------------------------------------------------------------------------

def test_giese_is_a_selectable_porosity_model_that_still_refuses_a_table():
    design = reduced_design(channeling={"porosity_model": "giese"})
    assert design.channeling.porosity_model == "giese"
    assert design.to_config_dict()["channeling"]["porosity_model"] == "giese"
    with pytest.raises(ValueError, match="generates the radial"):
        reduced_design(
            channeling={"porosity_model": "giese"},
            radial_porosity_profile=[[0, 0.34], [1, 0.412]],
        )


def test_giese_preserves_the_measured_mean_exactly_at_every_grid():
    """Same inventory contract as de Klerk: the core value is normalised, not assumed."""
    from pydac.core.bed_architectures import GieseRadialPorosity

    profile = GieseRadialPorosity(
        radius=RADIUS, particle_diameter=PARTICLE_DIAMETER, mean_porosity=MEAN_POROSITY
    )
    for n_rings in (20, 25, 40, 80, 160):
        faces = np.linspace(0.0, RADIUS, n_rings + 1)
        assert area_mean(profile.ring_means(faces), faces) == pytest.approx(
            MEAN_POROSITY, abs=1e-12
        )


def test_giese_ring_means_are_closed_form_and_match_quadrature():
    """The antiderivative is analytic; pin it against a high-order Gauss rule."""
    from pydac.core.bed_architectures import GieseRadialPorosity

    profile = GieseRadialPorosity(
        radius=RADIUS, particle_diameter=PARTICLE_DIAMETER, mean_porosity=MEAN_POROSITY
    )
    faces = np.linspace(0.0, RADIUS, 41)
    nodes, weights = np.polynomial.legendre.leggauss(200)
    for i in range(faces.size - 1):
        low, high = faces[i], faces[i + 1]
        mid, half = 0.5 * (high + low), 0.5 * (high - low)
        r = mid + half * nodes
        quadrature = (
            half * float(np.sum(weights * r * profile(r))) * 2.0 / (high**2 - low**2)
        )
        assert profile.ring_means(faces)[i] == pytest.approx(quadrature, abs=1e-12)


def test_giese_has_no_wall_singularity_unlike_de_klerk():
    """This is the physical difference that motivates offering it at all.

    de Klerk reaches eps = 1 exactly at the wall, where both Ergun coefficients
    vanish — the singularity the declared drag-layer width W regularises. Giese
    tops out at 2.36 * eps_core, short of 1, so the resistance stays finite.
    """
    from pydac.core.bed_architectures import DeKlerkRadialPorosity, GieseRadialPorosity

    kwargs = dict(
        radius=RADIUS, particle_diameter=PARTICLE_DIAMETER, mean_porosity=MEAN_POROSITY
    )
    de_klerk = DeKlerkRadialPorosity(**kwargs)
    giese = GieseRadialPorosity(**kwargs)
    assert de_klerk.wall_value == pytest.approx(1.0, abs=1e-9)
    assert giese.wall_value < 0.95
    assert giese.wall_value == pytest.approx(2.36 * giese.core_porosity, rel=1e-12)
    # and the weaker wall excess is the whole point: less voidage at the wall
    assert giese.wall_value < de_klerk.wall_value
    assert giese.core_porosity > de_klerk.core_porosity


def test_giese_resolves_through_the_public_structure_entry_point():
    design = reduced_design(channeling={"porosity_model": "giese"})
    resolved = structure(40, design=design)
    faces = resolved.r_faces
    assert resolved.porosity_model == "giese"
    assert area_mean(resolved.eps_storage, faces) == pytest.approx(MEAN_POROSITY, abs=1e-12)
    assert area_mean(resolved.eps_drag, faces) == pytest.approx(MEAN_POROSITY, abs=1e-12)
    # It is a genuinely different profile, not a rescaling of de Klerk. Note the
    # sign trap: Giese's POINTWISE wall value is lower (0.892 vs 1.0, no
    # singularity) while its outermost RING MEAN is higher, because de Klerk
    # oscillates and dips below the core value just inside the wall whereas
    # Giese decays monotonically. Comparing the two pointwise and by ring mean
    # gives opposite answers, so neither alone predicts the flow split.
    de_klerk = structure(40)
    assert resolved.eps_storage[-1] != pytest.approx(de_klerk.eps_storage[-1], rel=1e-3)
    assert resolved.core_porosity > de_klerk.core_porosity
