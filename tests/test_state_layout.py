from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from pydac.core.column_model_factory import ColumnModelFactory
from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.core.state_layout import (
    CURRENT_DAC_SCALARS,
    Mixture,
    ScalarState,
    StateLayout,
    current_dac_state_layout,
    simulation_scalar_states,
)


BREAKTHROUGH_SCALARS = (
    ScalarState("m_CO2", "kg", "mass"),
    ScalarState("m_N2", "kg", "mass"),
    ScalarState("E_fan", "J", "energy"),
    ScalarState("E_blower", "J", "energy"),
    ScalarState("E_compression", "J", "energy"),
    ScalarState("E_vacuum", "J", "energy"),
    ScalarState("E_jacket", "J", "energy"),
)


def breakthrough_layout(*, nz=2, nr=2):
    return StateLayout.build(
        mixture=Mixture(("CO2", "N2"), closure="N2"),
        adsorbates=("CO2", "N2"),
        nz=nz,
        nr=nr,
        scalar_states=BREAKTHROUGH_SCALARS,
    )


def values_following(layout, fields):
    return {
        field.name: np.arange(field.start, field.stop, dtype=float).reshape(field.shape)
        for field in fields
    }


def test_current_dac_layout_exactly_matches_legacy_order():
    layout = current_dac_state_layout(nz=2, nr=3)
    n_cells = 6

    assert layout.mixture.species == ("CO2", "H2O", "O2", "N2")
    assert layout.independent_gas_species == ("CO2", "H2O", "O2")
    assert layout.closure_species == "N2"
    assert layout.adsorbates == ("CO2", "H2O")
    assert tuple(field.name for field in layout.fields) == (
        "c_CO2", "c_H2O", "c_O2", "q_CO2", "q_H2O",
        "c_tot", "T_bed", "T_wall",
        "m_CO2", "m_H2O", "E_compression", "E_vacuum", "E_jacket",
    )
    assert layout.slice_for("c_CO2") == slice(0, n_cells)
    assert layout.slice_for("q_CO2") == slice(3 * n_cells, 4 * n_cells)
    assert layout.slice_for("c_tot") == slice(5 * n_cells, 6 * n_cells)
    assert layout.slice_for("T_bed") == slice(6 * n_cells, 7 * n_cells)
    assert layout.slice_for("T_wall") == slice(7 * n_cells, 7 * n_cells + 2)
    assert layout.pde_size == 7 * n_cells + 2
    assert layout.state_size == layout.pde_size + len(CURRENT_DAC_SCALARS)
    assert (layout.n_2d_fields, layout.n_1d_fields, layout.n_scalar_states) == (7, 1, 5)
    assert tuple(state.category for state in layout.scalar_states) == (
        "mass", "mass", "energy", "energy", "energy",
    )


def test_current_layout_packs_legacy_model_initial_state_without_reordering():
    config = load_experiment("tests/cases/v3_smoke_startup.json")
    cycle = CycleConfig.from_experiment(config)
    model = ColumnModelFactory(cycle).build_column_model(
        step="startup",
        solver_options=cycle.solver_for_step(
            use_sparsity=False,
            compute_numerical_sparsity=False,
        ),
    )
    layout = current_dac_state_layout(model.Nz, model.Nr)
    named_fields = {
        "c_CO2": model.ci[..., 0] / model.c_CO2_ref,
        "c_H2O": model.ci[..., 1] / model.c_H2O_ref,
        "c_O2": model.ci[..., 2] / model.c_O2_ref,
        "q_CO2": model.qi[..., 0] / model.q_eq_CO2_ref,
        "q_H2O": model.qi[..., 1] / model.q_eq_H2O_ref,
        "c_tot": model.c_tot[..., 0] / model.c_tot_ref,
        "T_bed": model.T[..., 0] / model.T_ref,
        "T_wall": model.Tw.ravel() / model.T_ref,
    }

    np.testing.assert_array_equal(layout.pack_pde(named_fields), model.get_initial_state())


def test_breakthrough_layout_comes_from_the_same_generic_builder():
    layout = breakthrough_layout()

    assert layout.independent_gas_species == ("CO2",)
    assert layout.closure_species == "N2"
    assert tuple(field.name for field in layout.fields_in("y_2d")) == (
        "c_CO2", "q_CO2", "q_N2", "c_tot", "T_bed",
    )
    assert tuple(field.name for field in layout.fields_in("y_1d")) == ("T_wall",)
    assert tuple(field.name for field in layout.fields_in("y_scalars")) == tuple(
        state.name for state in BREAKTHROUGH_SCALARS
    )
    assert (layout.n_2d_fields, layout.n_1d_fields, layout.n_scalar_states) == (5, 1, 7)
    assert layout.pde_size == 5 * (2 * 2) + 2
    assert layout.state_size == layout.pde_size + 7


@pytest.mark.parametrize("layout", [current_dac_state_layout(2, 3), breakthrough_layout()])
def test_pack_unpack_round_trip_uses_only_field_names(layout):
    all_values = values_following(layout, layout.fields)
    packed = layout.pack(all_values)
    np.testing.assert_array_equal(packed, np.arange(layout.state_size, dtype=float))
    for name, expected in all_values.items():
        np.testing.assert_array_equal(layout.unpack(packed)[name], expected)

    pde_values = values_following(layout, layout.pde_fields)
    packed_pde = layout.pack_pde(pde_values)
    np.testing.assert_array_equal(packed_pde, np.arange(layout.pde_size, dtype=float))
    for name, expected in pde_values.items():
        np.testing.assert_array_equal(layout.unpack_pde(packed_pde)[name], expected)


def test_json_report_signature_and_text_are_deterministic_and_readable():
    first = current_dac_state_layout(3, 1)
    second = current_dac_state_layout(3, 1)
    document = json.loads(json.dumps(first.to_dict(), allow_nan=False))

    assert document["mixture"]["closure"] == "N2"
    assert document["fields"][0]["state_slice"] == [0, 3]
    assert first.signature == second.signature == document["signature"]
    assert first.signature != current_dac_state_layout(4, 1).signature
    assert len(first.signature) == 64
    assert "Mixture: CO2, H2O, O2, N2 (closure: N2)" in str(first)
    assert "c_CO2" in str(first) and "T_wall" in str(first)


def test_layout_owns_scale_constraint_and_tolerance_policies():
    layout = current_dac_state_layout(2, 1)

    assert layout.field("c_CO2").scale_source == "gas_reference_concentration"
    assert layout.field("c_CO2").constraint == "none"
    assert layout.field("c_CO2").tolerance_group == "trace_gas"
    assert layout.field("q_CO2").constraint == "nonnegative"
    assert layout.field("c_tot").scale_source == "total_reference_concentration"
    assert layout.field("T_wall").scale_source == "reference_temperature"
    assert layout.field("m_CO2").tolerance_group == "mass_scalar"
    assert layout.field("E_jacket").tolerance_group == "energy_scalar"

    document = layout.to_dict()
    assert document["schema_version"] == 2
    assert document["fields"][0]["constraint"] == "none"


def test_default_scalar_states_follow_declared_adsorbates():
    assert tuple(state.name for state in simulation_scalar_states(("CO2", "N2"))) == (
        "m_CO2",
        "m_N2",
        "E_compression",
        "E_vacuum",
        "E_jacket",
    )


def test_closure_is_omitted_from_gas_fields_but_may_have_a_loading():
    layout = breakthrough_layout()

    with pytest.raises(ValueError, match="closure species"):
        layout.gas_field("N2")
    assert layout.loading_field("N2").name == "q_N2"


@pytest.mark.parametrize(
    ("species", "closure", "match"),
    [
        (("CO2", "CO2"), "CO2", "duplicates"),
        (("CO2",), "CO2", "at least one independent"),
        (("CO2", "N2"), "H2O", "not in"),
    ],
)
def test_mixture_rejects_invalid_species_or_closure(species, closure, match):
    with pytest.raises(ValueError, match=match):
        Mixture(species, closure=closure)


def test_layout_rejects_empty_duplicate_or_mismatched_adsorbates():
    mixture = Mixture(("CO2", "N2"), closure="N2")

    with pytest.raises(ValueError, match="must not be empty"):
        StateLayout.build(mixture=mixture, adsorbates=(), nz=2, nr=1)
    with pytest.raises(ValueError, match="duplicates"):
        StateLayout.build(
            mixture=mixture, adsorbates=("CO2", "CO2"), nz=2, nr=1
        )
    with pytest.raises(ValueError, match="not present"):
        StateLayout.build(
            mixture=mixture, adsorbates=("CO2", "H2O"), nz=2, nr=1
        )


def test_layout_rejects_invalid_grid_and_name_collisions():
    mixture = Mixture(("CO2", "N2"), closure="N2")

    with pytest.raises(ValueError, match="at least 1"):
        StateLayout.build(mixture=mixture, adsorbates=("CO2",), nz=0, nr=1)
    with pytest.raises(ValueError, match="collide"):
        StateLayout.build(
            mixture=mixture,
            adsorbates=("CO2",),
            nz=2,
            nr=1,
            scalar_states=(ScalarState("c_CO2", "kg"),),
        )


def test_layout_records_are_immutable():
    layout = current_dac_state_layout(2, 1)

    with pytest.raises(FrozenInstanceError):
        layout.nz = 3
    with pytest.raises(FrozenInstanceError):
        layout.mixture.closure = "O2"
    with pytest.raises(FrozenInstanceError):
        layout.fields[0].start = 5


def test_pack_unpack_errors_explain_names_shapes_and_sizes():
    layout = breakthrough_layout()
    values = values_following(layout, layout.fields)
    values.pop("c_CO2")
    with pytest.raises(ValueError, match="missing=.*c_CO2"):
        layout.pack(values)

    pde_values = values_following(layout, layout.pde_fields)
    pde_values["c_CO2"] = np.zeros((1, 1))
    with pytest.raises(ValueError, match="shape.*expected"):
        layout.pack_pde(pde_values)

    with pytest.raises(ValueError, match="must have shape"):
        layout.unpack(np.zeros(layout.state_size - 1))
