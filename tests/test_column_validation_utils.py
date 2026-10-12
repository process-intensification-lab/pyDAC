"""Direct coverage for the plumbing lifted out of the supervised column model.

`utils/column/validation.py` and the steam primitives in `utils/flow_transforms.py`
now own error paths that used to live inline in `src/pydac/core/column_model.py`, where they
were only ever reached through a full model build. These tests pin the messages
and the numbers so a later edit to the helpers cannot silently change what the
model rejects, or what a steam step is fed.
"""

from __future__ import annotations

import contextlib
import io
import re
import warnings

import numpy as np
import pytest
from scipy.constants import gas_constant as R_gas

from pydac.core.isotherm_models import GABIsotherm, water_saturation_pressure
from pydac.core.thermodynamics_separators import ThermoModels as BinarySeparator
from pydac.core.ternary_thermodynamics_separators import ThermoModels as TernarySeparator
from pydac.utils.column import (
    require_water_component,
    resolve_fixed_adsorbates,
    resolve_heats_of_adsorption,
    resolve_initial_solid_loadings,
    validate_architecture_support,
    validate_flat_vector,
    validate_warm_start_fields,
)
from pydac.utils.flow_transforms import (
    MW_H2O,
    cap_pressure_at_saturation,
    check_saturation_pressure_agrees,
    ideal_vapor_density,
    steam_density_from_pressure,
    water_saturation_pressure_iapws,
)
from pydac.utils.solver.flags import env_flag


ADSORBATES = ("CO2", "H2O")
WARM_START_GRID = dict(step="s_ads", layout_signature="sig", Nz=6, Nr=1, Nc_ci=3, Nc_solid=2)


def literal(message):
    """Match an error message as text, not as a regular expression."""
    return re.escape(message)


def warm_start_fields(*, Nz=6, Nr=1, Nc_ci=3, Nc_solid=2, signature="sig"):
    """Build one previous-step state consistent with `WARM_START_GRID`."""
    return {
        "Nz": Nz,
        "Nr": Nr,
        "layout_signature": signature,
        "ci": np.zeros((Nz, Nr, Nc_ci)),
        "qi": np.zeros((Nz, Nr, Nc_solid)),
        "c_tot": np.zeros((Nz, Nr)),
        "T": np.zeros((Nz, Nr)),
        "T_wall": np.zeros(Nz),
    }


def gab_isotherm():
    """Return the configured H2O component without its construction banner."""
    with contextlib.redirect_stdout(io.StringIO()):
        return GABIsotherm()


# --- packed-vector interfaces ---------------------------------------------


def test_flat_vector_validation_returns_the_original_array():
    vector = np.arange(4.0)
    assert validate_flat_vector(vector, size=4, label="Packed state") is vector


def test_flat_vector_validation_reports_label_expected_and_actual_shapes():
    with pytest.raises(
        ValueError,
        match=literal("Packed state must have shape (4,); got (2, 2)."),
    ):
        validate_flat_vector(np.zeros((2, 2)), size=4, label="Packed state")


# --- architecture support --------------------------------------------------


def test_supported_architecture_and_grid_pairs_pass():
    validate_architecture_support("packed_bed", is_1d=False, Nr=8)
    validate_architecture_support("packed_bed", is_1d=True, Nr=1)


@pytest.mark.parametrize("nr", [1, 8])
def test_unknown_architecture_is_rejected_on_any_grid(nr):
    with pytest.raises(ValueError, match="Unknown architecture"):
        validate_architecture_support("unsupported", is_1d=(nr == 1), Nr=nr)


# --- sorbent inventory -----------------------------------------------------


def test_heats_of_adsorption_follow_the_declared_adsorbate_order():
    resolved = resolve_heats_of_adsorption({"H2O": -50e3, "CO2": -75e3}, ADSORBATES)
    assert tuple(resolved) == ADSORBATES
    assert resolved == {"CO2": -75e3, "H2O": -50e3}


@pytest.mark.parametrize(
    "configured, error, message",
    [
        ([], TypeError, "simulation.sorbent.heats_of_adsorption_J_mol must be a species-keyed mapping."),
        ({"CO2": -75e3}, ValueError, "Sorbent heats of adsorption are missing declared adsorbates ['H2O']."),
        ({"CO2": -75e3, "H2O": 0.0}, ValueError, "Heats of adsorption must be finite and negative for ['H2O']."),
        ({"CO2": np.nan, "H2O": -50e3}, ValueError, "Heats of adsorption must be finite and negative for ['CO2']."),
    ],
)
def test_invalid_heats_of_adsorption_are_rejected(configured, error, message):
    with pytest.raises(error, match=literal(message)):
        resolve_heats_of_adsorption(configured, ADSORBATES)


def test_fixed_adsorbates_are_normalized_to_upper_case_species():
    resolved = resolve_fixed_adsorbates(
        {" n2 ": {"loading_mol_per_kg": 0.0, "heat_capacity_J_molK": 29.1}}
    )
    assert resolved == {"N2": {"loading_mol_per_kg": 0.0, "heat_capacity_J_molK": 29.1}}
    assert resolve_fixed_adsorbates({}) == {}


@pytest.mark.parametrize(
    "configured, error, message",
    [
        (None, TypeError, "simulation.sorbent.fixed_adsorbates must be a mapping."),
        ({"N2": 3.0}, TypeError, "Each fixed adsorbate must have a species name and a property mapping."),
        (
            {"N2": {"loading_mol_per_kg": 0.0, "typo": 1}},
            ValueError,
            "Invalid fixed-adsorbate properties for N2 "
            "(missing=['heat_capacity_J_molK'], unknown=['typo']).",
        ),
        (
            {"N2": {"loading_mol_per_kg": -1.0, "heat_capacity_J_molK": 29.1}},
            ValueError,
            "Fixed loading for N2 must be finite and non-negative.",
        ),
        (
            {"N2": {"loading_mol_per_kg": 0.0, "heat_capacity_J_molK": 0.0}},
            ValueError,
            "Fixed-adsorbate heat capacity for N2 must be finite and positive.",
        ),
    ],
)
def test_invalid_fixed_adsorbates_are_rejected(configured, error, message):
    with pytest.raises(error, match=literal(message)):
        resolve_fixed_adsorbates(configured)


def test_absent_initial_loadings_stay_none():
    # `None` marks the value the model supplies by equilibrating the initial state.
    assert resolve_initial_solid_loadings(None, ADSORBATES) is None


def test_initial_loadings_are_normalized_and_ordered():
    assert resolve_initial_solid_loadings({"co2": 0.1, "h2o": 2.0}, ADSORBATES) == {
        "CO2": 0.1,
        "H2O": 2.0,
    }


@pytest.mark.parametrize(
    "configured, error, message",
    [
        (3.0, TypeError, "initial_solid_loadings_mol_per_kg must be an adsorbate-keyed mapping."),
        (
            {"CO2": 0.1, "Ar": 0.0},
            ValueError,
            "Initial solid loadings must cover the declared adsorbates exactly "
            "(missing=['H2O'], unknown=['AR']).",
        ),
        (
            {"CO2": -0.1, "H2O": 1.0},
            ValueError,
            "Initial solid loadings must be finite and non-negative for ['CO2'].",
        ),
    ],
)
def test_invalid_initial_loadings_are_rejected(configured, error, message):
    with pytest.raises(error, match=literal(message)):
        resolve_initial_solid_loadings(configured, ADSORBATES)


# --- H2O equilibrium component --------------------------------------------


def test_configured_water_component_is_returned():
    water = gab_isotherm()
    system = type("System", (), {"component_models": {"H2O": water}})()
    assert require_water_component(system) is water


def test_gab_reports_extrapolation_once_and_stays_finite_up_to_its_pole():
    water = gab_isotherm()
    T = 298.15
    p_sat = water.EOS(T)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted = water.isotherm(np.array([0.5, 0.9]) * p_sat, T)[0]
    with pytest.warns(RuntimeWarning, match="above the fitted data"):
        humid = water.isotherm(np.array([0.95, 1.0, 5.0]) * p_sat, T)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        water.isotherm(np.array([1.0]) * p_sat, T)  # reported once per object
    q, rh = humid[0], humid[1]
    # Above RH 0.9 the loading keeps rising; far past saturation it stops 1 % short of 1/k.
    assert fitted[1] < q[0] < q[1] < q[2]
    assert np.all(np.isfinite(q))
    k = np.exp(((water.F + water.G * T) - (water.a_lin + water.b_lin * T)) / (R_gas * T))
    assert rh[2] == pytest.approx(0.99 / k)


@pytest.mark.parametrize("component_models", [None, {}, {"H2O": object()}])
def test_without_a_water_eos_the_iapws_vapor_pressure_is_used(component_models):
    # A user H2O model need not give water's vapor pressure: it is the one GAB gives.
    attributes = {} if component_models is None else {"component_models": component_models}
    system = type("System", (), attributes)()
    T = np.array([278.15, 298.15, 373.15, 423.15])
    assert np.array_equal(require_water_component(system).EOS(T), gab_isotherm().EOS(T))


# --- warm start ------------------------------------------------------------


def test_matching_warm_start_returns_the_previous_grid():
    assert validate_warm_start_fields(warm_start_fields(), **WARM_START_GRID) == (6, 1)


def test_warm_start_grid_falls_back_to_the_saved_field_shape():
    fields = warm_start_fields()
    del fields["Nz"], fields["Nr"]
    assert validate_warm_start_fields(fields, **WARM_START_GRID) == (6, 1)


def test_warm_start_accepts_state_without_a_recorded_layout_signature():
    # Results saved before layout signatures existed stay loadable.
    assert validate_warm_start_fields(
        warm_start_fields(signature=None), **WARM_START_GRID
    ) == (6, 1)


def test_missing_warm_start_names_the_step_being_built():
    with pytest.raises(
        ValueError,
        match=literal("Received None while building step 's_ads'."),
    ):
        validate_warm_start_fields(None, **WARM_START_GRID)


def test_warm_start_from_a_different_state_layout_is_rejected():
    with pytest.raises(
        ValueError,
        match=literal("Previous signature: other; current signature: sig."),
    ):
        validate_warm_start_fields(warm_start_fields(signature="other"), **WARM_START_GRID)


@pytest.mark.parametrize(
    "fields, message",
    [
        (warm_start_fields(Nc_ci=4), "saved `ci` field shape is inconsistent"),
        (warm_start_fields(Nc_solid=5), "saved `qi` field shape is inconsistent"),
        (warm_start_fields(Nz=8), "the previous step was saved on a different grid"),
    ],
)
def test_inconsistent_warm_start_shapes_are_rejected(fields, message):
    with pytest.raises(ValueError, match=literal(message)):
        validate_warm_start_fields(fields, **WARM_START_GRID)


@pytest.mark.parametrize("field, shape", [("T", (6, 2)), ("c_tot", (6, 2))])
def test_inconsistent_scalar_field_shapes_are_rejected(field, shape):
    fields = warm_start_fields()
    fields[field] = np.zeros(shape)
    with pytest.raises(ValueError, match=literal("scalar field shapes are inconsistent")):
        validate_warm_start_fields(fields, **WARM_START_GRID)


def test_inconsistent_wall_field_shape_is_rejected():
    fields = warm_start_fields()
    fields["T_wall"] = np.zeros(7)
    with pytest.raises(ValueError, match=literal("saved `T_wall` field shape is inconsistent")):
        validate_warm_start_fields(fields, **WARM_START_GRID)


# --- steam inlet primitives ------------------------------------------------


def test_pressure_below_saturation_passes_through_untouched():
    assert cap_pressure_at_saturation(2.0e4, 1.0e5, temperature_K=373.15) == 2.0e4


def test_pressure_above_saturation_is_capped_silently_without_a_context():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cap_pressure_at_saturation(3.0e5, 1.0e5, temperature_K=373.15) == 1.0e5


def test_capping_with_a_context_names_the_step_and_both_pressures():
    p_sat = water_saturation_pressure_iapws(373.15)
    expected = (
        f"s_des feed pressure ({3.0e5:.2f} Pa) exceeds H2O saturation pressure at "
        f"{373.15:.2f} K ({p_sat:.2f} Pa). Using saturation pressure for steam "
        "density estimate."
    )
    with pytest.warns(RuntimeWarning, match=literal(expected)):
        capped = cap_pressure_at_saturation(
            3.0e5, p_sat, temperature_K=373.15, context="s_des"
        )
    assert capped == p_sat


def test_steam_density_matches_the_ideal_gas_law():
    assert ideal_vapor_density(1.0e5, 373.15, MW_H2O) == 1.0e5 * MW_H2O / (R_gas * 373.15)


def test_steam_density_from_pressure_caps_at_the_owned_saturation_pressure():
    temperature = 373.15
    p_sat = water_saturation_pressure(temperature)
    rho, p_used, reported_p_sat = steam_density_from_pressure(3.0e5, T_feed=temperature)
    assert (p_used, reported_p_sat) == (p_sat, p_sat)
    assert rho == p_sat * MW_H2O / (R_gas * temperature)


# --- one owner for the water saturation pressure ---------------------------


def reference_saturation_pressure(T):
    """IAPWS-1995 transcribed from the pre-refactor implementations.

    Kept spelled out on purpose: the point of this test is that no rewrite of
    `isotherm_models.water_saturation_pressure` changes its result, and an
    algebraically equal reformulation can still move the last bits.
    """
    T_crit, P_crit = 647.096, 220.64 * 1e5
    b1, b2, b3, b4, b5, b6 = (
        -7.85951783, 1.8440825, -11.7866497, 22.6807411, -15.9618719, 1.80122502,
    )
    v = 1.0 - T / T_crit
    ln_p_sat_p_crit = T_crit / T * (
        b1 * v + b2 * v**1.5 + b3 * v**3 + b4 * v**3.5 + b5 * v**4 + b6 * v**7.5
    )
    return P_crit * np.exp(ln_p_sat_p_crit)


@pytest.mark.parametrize("T", [273.15, 298.15, 333.15, 353.15, 373.15, 393.15, 450.0])
def test_every_water_vapor_pressure_caller_resolves_to_one_relation(T):
    # The simulation's own EOS is the reference; the callers that need a vapor
    # pressure before an equilibrium system exists must not answer differently.
    expected = reference_saturation_pressure(T)
    assert gab_isotherm().EOS(T) == expected
    assert water_saturation_pressure(T) == expected
    assert water_saturation_pressure_iapws(T) == expected
    assert BinarySeparator.p_sat_H2O(T) == expected
    assert TernarySeparator.p_sat_H2O(T) == expected


def test_water_vapor_pressure_is_bit_stable_across_the_operating_range():
    # A dense sweep, not a handful of round numbers: an algebraically equal
    # reformulation of the relation moves only some temperatures by one ulp, and
    # hand-picked operating points miss most of them.
    water = gab_isotherm()
    for T in np.linspace(273.16, 500.0, 200):
        T = float(T)
        expected = reference_saturation_pressure(T)
        assert water.EOS(T) == expected, T
        assert water_saturation_pressure(T) == expected, T
        assert water_saturation_pressure_iapws(T) == expected, T
        assert BinarySeparator.p_sat_H2O(T) == expected, T
        assert TernarySeparator.p_sat_H2O(T) == expected, T


def test_water_saturation_pressure_maps_over_arrays():
    temperatures = np.array([300.0, 373.15])
    np.testing.assert_array_equal(
        water_saturation_pressure(temperatures),
        [reference_saturation_pressure(T) for T in temperatures],
    )


def test_configured_eos_agreeing_with_the_owner_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_saturation_pressure_agrees(
            gab_isotherm().EOS(373.15), 373.15, context="s_des"
        )


def test_a_divergent_configured_eos_is_reported():
    # The Buck relation also living in `GABIsotherm`: close, but not the same.
    temperature = 393.15
    buck = 611.21 * np.exp(
        (18.678 - (temperature - 273.15) / 234.5)
        * ((temperature - 273.15) / (temperature - 16.01))
    )
    with pytest.warns(RuntimeWarning, match="describes a different stream than this step"):
        assert not check_saturation_pressure_agrees(buck, temperature, context="s_des")


# --- environment switches --------------------------------------------------


@pytest.mark.parametrize("raw", ["0", "off", "false", "FALSE", " no ", "No"])
def test_env_flag_recognizes_every_disabling_token(monkeypatch, raw):
    monkeypatch.setenv("PYDAC_TEST_FLAG", raw)
    assert env_flag("PYDAC_TEST_FLAG") is False


@pytest.mark.parametrize("raw", ["", "1", "yes", "on", "anything"])
def test_env_flag_leaves_every_other_value_enabled(monkeypatch, raw):
    monkeypatch.setenv("PYDAC_TEST_FLAG", raw)
    assert env_flag("PYDAC_TEST_FLAG") is True


@pytest.mark.parametrize("default", [True, False])
def test_env_flag_falls_back_to_the_default_when_unset(monkeypatch, default):
    monkeypatch.delenv("PYDAC_TEST_FLAG", raising=False)
    assert env_flag("PYDAC_TEST_FLAG", default=default) is default
