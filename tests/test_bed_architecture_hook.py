"""A bed architecture registered at run time, proved on a clone of the packed bed.

The clone's entry calls the packed-bed closures, so a run through the hook must
equal the built-in packed-bed run bit for bit: this proves that the column binds
a registered architecture's results exactly as it binds the packed bed's.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import pydac
from pydac.config import load_experiment
from pydac.config.cycle import CycleConfig
from pydac.core.bed_architectures import (
    ClosureInputs,
    PackedBedDesign,
    ResolvedTransport,
    get_architecture_spec,
    registered_architectures,
    registered_bed_architecture,
    resolve_bed_config,
    resolve_packed_bed_dispersion,
    resolve_packed_bed_pressure_drop,
    resolve_packed_bed_structure,
    resolve_packed_bed_thermal_transport,
)
from pydac.core.bed_architectures import registry
from pydac.core.column_model_factory import ColumnModelFactory
from pydac.core.solver import Solver
from pydac.post_processing.context.simulation_context import SimulationContext
from pydac.utils.column.validation import validate_architecture_support
from pydac.utils.postprocessing.context_schema import ContextParameter

CASE = str(Path(__file__).resolve().parent / "cases" / "v3_smoke_startup.json")
CLONE = "test_clone_bed"
CLONE_BED = {"architecture": CLONE, "design": {"eps_b": 0.38}, "material": None}
CLONE_ROWS = (
    ContextParameter("design", "bed.design.eps_b", "-"),
    ContextParameter("provenance", "bed.provenance.source", "-", "text"),
    ContextParameter("provenance", "bed.provenance.year", "-", "text"),
)
# A numeric-looking provenance text must stay text through a saved context.
CLONE_PROVENANCE = {"source": "packed-bed clone (test)", "year": "1996"}


@pytest.fixture(autouse=True)
def session_registry(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTERED_BED_ARCHITECTURES", {})


def _build_design(design, material):
    if material is not None or set(design) != {"eps_b"}:
        raise ValueError("the clone takes exactly design.eps_b and material=None")
    return PackedBedDesign(eps_b=design["eps_b"])


def _clone_entry(calls=None, **overrides):
    """Entry whose callables are the packed-bed closures; ``calls`` records inputs."""
    calls = calls if calls is not None else []

    def pressure(design, inputs):
        calls.append(("pressure", inputs))
        return resolve_packed_bed_pressure_drop(
            eps_flow=inputs.eps_flow, viscosity=inputs.viscosity,
            particle_diameter=inputs.particle_diameter, bed_length=inputs.bed_length,
            reference_velocity=inputs.reference_velocity,
            reference_pressure=inputs.reference_pressure)

    def dispersion(design, inputs):
        calls.append(("dispersion", inputs))
        return resolve_packed_bed_dispersion(
            eps_flow=inputs.eps_flow, molecular_diffusivity=inputs.molecular_diffusivity,
            reynolds=inputs.reynolds, schmidt=inputs.schmidt,
            superficial_inlet_velocity=inputs.superficial_inlet_velocity,
            particle_diameter=inputs.particle_diameter,
            reference_velocity=inputs.reference_velocity, bed_length=inputs.bed_length,
            bed_radius=inputs.bed_radius)

    def thermal(design, inputs):
        calls.append(("thermal", inputs))
        return resolve_packed_bed_thermal_transport(
            stagnant_conductivity_model=design.stagnant_conductivity_model,
            wall_htc_model=design.wall_htc_model, eps_flow=inputs.eps_flow,
            gas_conductivity=inputs.gas_conductivity,
            particle_conductivity=inputs.particle_conductivity, reynolds=inputs.reynolds,
            prandtl=inputs.prandtl, particle_diameter=inputs.particle_diameter,
            bed_diameter=inputs.bed_diameter, reference_velocity=inputs.reference_velocity,
            bed_length=inputs.bed_length, is_1d=inputs.is_1d)

    fields = dict(
        spatial_modes=("1d", "2d"),
        build_design=_build_design,
        to_config=lambda design: ({"eps_b": design.eps_b}, None),
        resolve_structure=resolve_packed_bed_structure,
        resolve_pressure_drop=pressure,
        resolve_dispersion=dispersion,
        resolve_thermal_transport=thermal,
        wall_htc_model="dixon",
        provenance=CLONE_PROVENANCE,
        context_parameters=CLONE_ROWS,
    )
    fields.update(overrides)
    return pydac.BedArchitectureEntry(**fields)


def _model(nr, bed=None, aspect_ratio=None):
    settings = {"grid.nz": 4, "grid.nr": nr}
    if aspect_ratio is not None:
        settings["geometry.aspect_ratio"] = aspect_ratio
    if bed is not None:
        settings["bed"] = bed
    cycle = CycleConfig.from_experiment(load_experiment(CASE, settings=settings))
    return ColumnModelFactory(cycle).build_column_model(
        step="startup", initial_state=None,
        solver_options=cycle.solver_for_step(use_sparsity=False, compute_numerical_sparsity=False))


def _same_bits(left, right, label):
    left, right = np.asarray(left), np.asarray(right)
    assert left.dtype == right.dtype and left.shape == right.shape, label
    assert np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes(), label


_BOUND = (
    "eps_flow", "eps_total", "rho_sorbent", "acc_structure", "Disp_ax", "Disp_rad",
    "inv_Pe_m_ax", "inv_Pe_m_rad", "lambda_ax", "lambda_rad", "inv_Pe_h_ax", "inv_Pe_h_rad",
    "h_w", "inv_Bi_h", "h_int", "inlet_flux_weights",
)


# ClosureInputs field -> the model attribute it must carry (C4 `_closure_inputs`).
_INPUT_SOURCES = {
    "eps_flow": "eps_flow", "viscosity": "mu_g_mix", "molecular_diffusivity": "D_m",
    "reynolds": "Re", "schmidt": "Sc", "prandtl": "Pr", "gas_conductivity": "lambda_g_mix",
    "particle_conductivity": "lambda_p", "superficial_inlet_velocity": "u_inlet",
    "particle_diameter": "d_p", "bed_length": "L_b", "bed_radius": "R_b", "bed_diameter": "d_b",
    "reference_velocity": "u_ref", "reference_pressure": "p_ref", "is_1d": "is_1d",
}


# aspect_ratio 1.3 makes L_b, R_b and d_b three different numbers, so a swapped
# geometry input cannot pass unseen (the case's L/D = 0.5 has L_b == R_b).
@pytest.mark.parametrize("nr, aspect_ratio", [(1, None), (3, None), (1, 1.3), (3, 1.3)])
def test_a_packed_bed_clone_through_the_hook_runs_bit_identical(nr, aspect_ratio, tmp_path):
    calls = []
    pydac.register_bed_architecture(CLONE, _clone_entry(calls))
    built_in = _model(nr, aspect_ratio=aspect_ratio)
    clone = _model(nr, CLONE_BED, aspect_ratio=aspect_ratio)
    assert built_in.bed_architecture_provider is None
    assert clone.bed_architecture_provider is registered_bed_architecture(CLONE)
    if aspect_ratio is not None:
        assert len({clone.L_b, clone.R_b, clone.d_b}) == 3

    # The closures see the model's own values, with D_m and Sc one per gas species.
    for _, inputs in calls:
        for field_name, attribute in _INPUT_SOURCES.items():
            expected = getattr(clone, attribute)
            if field_name == "viscosity":
                expected = float(np.asarray(expected).item())
            _same_bits(getattr(inputs, field_name), expected, field_name)
        assert inputs.gas_species == tuple(clone.gas_species)
    inputs = calls[0][1]
    n_gas = len(clone.gas_species)
    assert inputs.molecular_diffusivity.shape == inputs.schmidt.shape == (n_gas,)
    assert not inputs.molecular_diffusivity.flags.writeable
    assert type(clone.pressure_drop_law).__name__ == "QuadraticPressureDropLaw"

    for name in _BOUND:
        _same_bits(getattr(built_in, name), getattr(clone, name), name)
    if nr == 1:
        _same_bits(built_in.U_w, clone.U_w, "U_w")
    for name in ("linear_coefficient", "inertial_coefficient_per_density"):
        _same_bits(getattr(built_in.pressure_drop_law, name),
                   getattr(clone.pressure_drop_law, name), name)
    assert built_in.resolved_transport == clone.resolved_transport

    solutions = []
    for model, folder in ((built_in, "built_in"), (clone, "clone")):
        solver = Solver(model, base_folder=str(tmp_path / folder))
        solver.solve_cvode()
        solver.prepare_results()
        solutions.append(solver.sol)
    for key in ("t", "y_2d", "y_1d", "y_scalars"):
        _same_bits(solutions[0][key], solutions[1][key], key)
    assert solutions[0]["termination"] == solutions[1]["termination"]

    # Same context numbers; only the architecture name, its registered-architecture
    # marker, the clone's provenance and the overall-coefficient label differ (the
    # clone's closure gives the 1D coefficient itself, under its own law's name).
    frames = [SimulationContext.from_model(m).to_dataframe() for m in (built_in, clone)]
    marker = frames[1].loc[frames[1]["Parameter ID"].eq("bed.registered_architecture")]
    assert marker["Value"].tolist() == [CLONE] and marker["Step"].tolist() == [""]
    differing = {"bed.architecture", "bed.registered_architecture",
                 "bed.provenance.source", "bed.provenance.year", "bed.models.overall_wall_htc"}
    kept = [f.loc[~f["Parameter ID"].isin(differing)].reset_index(drop=True) for f in frames]
    pd.testing.assert_frame_equal(kept[0], kept[1], check_exact=True)
    context = clone.bed_architecture_context
    assert context["resolved"]["models"]["wall_htc"] == "dixon"
    assert context["resolved"]["provenance"] == CLONE_PROVENANCE
    assert "flow" not in context["resolved"]


def test_registration_names_and_lookups():
    with pytest.raises(pydac.ConfigurationError, match="Unknown architecture"):
        load_experiment(CASE, settings={"bed": CLONE_BED})
    for name in ("packed_bed", "Test_Clone", "", "clone-bed", " clone"):
        with pytest.raises(pydac.ConfigurationError):
            pydac.register_bed_architecture(name, _clone_entry())
    with pytest.raises(pydac.ConfigurationError, match="BedArchitectureEntry"):
        pydac.register_bed_architecture(CLONE, {"spatial_modes": ("1d",)})
    for bad in (
        dict(spatial_modes=()), dict(spatial_modes=("3d",)), dict(spatial_modes="1d"),
        dict(resolve_structure=None), dict(wall_htc_model=" "),
        dict(provenance={"source": ""}),
        dict(context_parameters=(ContextParameter("design", "bed.geometry.eps_b", "-"),)),
        dict(context_parameters=(ContextParameter("material", "bed.design.eps_b", "-"),)),
        dict(context_parameters=(ContextParameter("design", "bed.design.eps_b", "-", "text"),)),
        dict(context_parameters=(ContextParameter("design", "bed.design.eps_b", " "),)),
        dict(context_parameters=CLONE_ROWS + CLONE_ROWS[:1]),
        dict(context_parameters=(ContextParameter("provenance", "bed.provenance.doi", "-", "text"),)),
    ):
        with pytest.raises(pydac.ConfigurationError):
            pydac.register_bed_architecture(CLONE, _clone_entry(**bad))

    pydac.register_bed_architecture(CLONE, _clone_entry())
    assert registered_architectures() == ("packed_bed", CLONE)
    spec = get_architecture_spec(CLONE)
    assert (spec.design_type, spec.spatial_modes) == ("registered", ("1d", "2d"))
    assert resolve_bed_config(CLONE_BED) == CLONE_BED
    with pytest.raises(ValueError, match="clone takes exactly"):
        resolve_bed_config({**CLONE_BED, "design": {"eps_b": 0.38, "porosity": 0.9}})


def test_spatial_modes_are_enforced_at_load_and_model_build():
    pydac.register_bed_architecture(CLONE, _clone_entry(spatial_modes=("1d",)))
    load_experiment(CASE, settings={"grid.nr": 1, "bed": CLONE_BED})
    with pytest.raises(pydac.ConfigurationError, match="supports only 1d"):
        load_experiment(CASE, settings={"grid.nr": 3, "bed": CLONE_BED})
    with pytest.raises(ValueError, match="supports only 1d"):
        validate_architecture_support(CLONE, is_1d=False, Nr=3)
    validate_architecture_support("packed_bed", is_1d=False, Nr=3)


@pytest.mark.parametrize("nr, field", [
    (1, "overall_htc_override_W_m2K"), (3, "bed_wall_htc_override_W_m2K"),
    (1, "bed_wall_htc_override_W_m2K"),
])
def test_wall_htc_overrides_are_refused_for_a_registered_bed(nr, field):
    pydac.register_bed_architecture(CLONE, _clone_entry())
    with pytest.raises(pydac.ConfigurationError, match="applies only to the packed bed"):
        load_experiment(CASE, settings={"grid.nr": nr, "bed": CLONE_BED, f"wall.{field}": 150.0})


@pytest.mark.parametrize("nr, extra, message", [
    (1, "wall.overall_htc_override_W_m2K=150", "applies only to the packed bed"),
    (3, "wall.overall_htc_override_W_m2K=150", "applies only to the packed bed"),
    (1, "wall.bed_wall_htc_override_W_m2K=150", "applies only to the packed bed"),
    (3, "wall.bed_wall_htc_override_W_m2K=150", "applies only to the packed bed"),
    (3, None, "supports only 1d"),
])
def test_pydac_validate_refuses_what_a_registered_bed_does_not_support(nr, extra, message,
                                                                        capsys):
    from pydac.cli import main

    modes = ("1d",) if extra is None else ("1d", "2d")
    pydac.register_bed_architecture(CLONE, _clone_entry(spatial_modes=modes))
    command = ["validate", "--config", CASE, "--set", f"grid.nr={nr}",
               "--set", f"bed={json.dumps(CLONE_BED)}"]
    if extra is not None:
        assert main(command) == 0
        capsys.readouterr()
        command += ["--set", extra]
    assert main(command) == 2
    assert message in capsys.readouterr().err


def test_an_exact_case_load_refuses_an_override_for_a_registered_bed(tmp_path):
    pydac.register_bed_architecture(CLONE, _clone_entry())
    path = tmp_path / "case.json"
    pydac.export_case(pydac.load_case(CASE, settings={"grid.nr": 1, "bed": CLONE_BED}), path)
    pydac.load_case(path, exact=True)
    document = json.loads(path.read_text())
    document["simulation"]["wall"]["overall_htc_override_W_m2K"] = 150.0
    path.write_text(json.dumps(document))
    with pytest.raises(pydac.ConfigurationError, match="applies only to the packed bed"):
        pydac.load_case(path, exact=True)


def test_a_clone_run_writes_the_built_in_kpis_bit_for_bit(tmp_path, monkeypatch):
    pydac.register_bed_architecture(CLONE, _clone_entry())
    kpis = []
    for name, bed in (("built_in", None), ("clone", CLONE_BED)):
        folder = tmp_path / name
        folder.mkdir()
        monkeypatch.chdir(folder)
        settings = {"grid.nz": 4, "grid.nr": 1}
        if bed is not None:
            settings["bed"] = bed
        result = pydac.run_steps(pydac.load_case(CASE, settings=settings))
        kpis.append(pd.read_pickle(result.results_folder / "startup" / "kpis.pkl"))
    assert not kpis[0].empty
    pd.testing.assert_frame_equal(kpis[0], kpis[1], check_exact=True)
    for column in kpis[0].columns:
        if kpis[0][column].dtype.kind == "f":
            _same_bits(kpis[0][column].to_numpy(), kpis[1][column].to_numpy(), column)


@pytest.mark.parametrize("nr, attribute", [
    (1, "U_w_override"), (3, "h_bed_wall_override"),
    (1, "h_bed_wall_override"), (3, "U_w_override"),
])
def test_the_core_thermal_dispatch_refuses_an_override_before_the_closure_runs(nr, attribute):
    """Backstop in the column: an override that reached the model is refused there."""
    calls = []
    pydac.register_bed_architecture(CLONE, _clone_entry(calls))
    model = _model(nr, CLONE_BED)
    thermal_calls = sum(kind == "thermal" for kind, _ in calls)
    setattr(model, attribute, 150.0)
    with pytest.raises(ValueError, match="overrides apply only to the packed bed"):
        model._resolve_bed_thermal_transport()
    assert sum(kind == "thermal" for kind, _ in calls) == thermal_calls


def _inputs(is_1d=True, n_gas=4):
    return ClosureInputs(
        eps_flow=0.38, viscosity=1.8e-5, molecular_diffusivity=np.full(n_gas, 2e-5),
        reynolds=np.asarray(2.2), schmidt=np.full(n_gas, 0.8), prandtl=0.71,
        gas_conductivity=0.026, particle_conductivity=0.43, superficial_inlet_velocity=0.1,
        particle_diameter=5e-4, bed_length=0.07, bed_radius=0.07, bed_diameter=0.14,
        reference_velocity=0.1, reference_pressure=101325.0, is_1d=is_1d,
        gas_species=("CO2", "H2O", "O2", "N2")[:n_gas])


def _provider(**overrides):
    pydac.register_bed_architecture(CLONE, _clone_entry(**overrides))
    provider = registered_bed_architecture(CLONE)
    return provider, provider.build_design({"eps_b": 0.38}, None)


def test_the_wrapper_checks_every_closure_result():
    provider, design = _provider()
    inputs = _inputs()
    resolved = provider.resolve_dispersion(design, inputs)
    assert type(resolved) is dict

    _, design = _provider(resolve_dispersion=lambda d, x: MappingProxyType(
        resolve_packed_bed_dispersion(
            eps_flow=0.38, molecular_diffusivity=x.molecular_diffusivity, reynolds=x.reynolds,
            schmidt=x.schmidt, superficial_inlet_velocity=0.1, particle_diameter=5e-4,
            reference_velocity=0.1, bed_length=0.07, bed_radius=0.07)))
    assert type(registered_bed_architecture(CLONE).resolve_dispersion(design, inputs)) is dict

    class Linear:
        """A law whose velocity ignores its inertial coefficient: not quadratic."""
        name = "linear_only"
        linear_coefficient = 2.0
        inertial_coefficient_per_density = 3.0

        def velocity_from_gradient(self, gradient, density):
            return -np.asarray(gradient, dtype=float) / self.linear_coefficient

    def thermal_2d_mismatch(design, x):
        return ResolvedTransport(lambda_ax=1.0, lambda_rad=1.0, h_wall=100.0, wall_bed_htc=50.0,
                                 pressure_drop_model="ergun", dispersion_model="x")

    bad_results = [
        ("resolve_structure", dict(resolve_structure=lambda d, p: {"eps_flow": 0.38}),
         "ResolvedStructure"),
        ("resolve_pressure_drop", dict(resolve_pressure_drop=lambda d, x: Linear()),
         "must solve"),
        ("resolve_dispersion", dict(resolve_dispersion=lambda d, x: {"model": "x"}),
         "missing keys"),
        ("resolve_dispersion", dict(resolve_dispersion=lambda d, x: {
            "model": "x", **{k: np.ones((1, 1, 4)) for k in
                             ("Disp_ax", "Disp_rad", "inv_Pe_m_ax", "inv_Pe_m_rad")}}),
         "shape"),
        ("resolve_thermal_transport", dict(resolve_thermal_transport=lambda d, x: ResolvedTransport(
            lambda_ax=1.0, lambda_rad=None, h_wall=100.0, wall_bed_htc=50.0,
            pressure_drop_model="ergun", dispersion_model="x")), "lambda_rad"),
    ]
    for method, override, message in bad_results:
        provider, design = _provider(**override)
        argument = None if method == "resolve_structure" else inputs
        if method == "resolve_structure":
            from pydac.core.bed_architectures import ParticleProperties
            argument = ParticleProperties(density=744.0, porosity=0.338, radius=2.5e-4,
                                          thermal_conductivity=0.43)
        with pytest.raises((TypeError, ValueError), match=message) as error:
            getattr(provider, method)(design, argument)
        assert CLONE in str(error.value) and method in str(error.value)
    provider, design = _provider(resolve_thermal_transport=thermal_2d_mismatch)
    provider.resolve_thermal_transport(design, _inputs(is_1d=True))
    with pytest.raises(ValueError, match="in 2D `wall_bed_htc` must equal `h_wall`"):
        provider.resolve_thermal_transport(design, _inputs(is_1d=False))


@pytest.mark.parametrize("to_config, message", [
    (lambda d: ({"eps_b": d.eps_b + 0.01}, None), "round-trip"),
    (lambda d: ({"eps_b": "0.38"}, None), "must be a number"),
    (lambda d: ({"eps_b": float("nan")}, None), "finite"),
    (lambda d: ({"eps_b": d.eps_b, "extra": 1.0}, None), "declared"),
    (lambda d: ({"Eps_b": d.eps_b}, None), "lowercase token"),
    (lambda d: {"eps_b": d.eps_b}, "tuple"),
])
def test_the_canonical_mapping_is_checked_at_load(to_config, message):
    build = (lambda design, material: PackedBedDesign(eps_b=float(design.get("eps_b", 0.38))))
    pydac.register_bed_architecture(CLONE, _clone_entry(
        to_config=to_config, build_design=build))
    with pytest.raises(pydac.ConfigurationError, match=message):
        load_experiment(CASE, settings={"bed": CLONE_BED})


def test_a_model_keeps_the_entry_it_was_built_with(tmp_path):
    provenance = dict(CLONE_PROVENANCE)
    pydac.register_bed_architecture(CLONE, _clone_entry(provenance=provenance))
    provenance["source"] = "changed after registration"
    first = _model(1, CLONE_BED)
    assert first.bed_architecture_provider.provenance["source"] == CLONE_PROVENANCE["source"]

    pydac.register_bed_architecture(CLONE, _clone_entry(
        provenance={"source": "second entry", "year": "2026"}, wall_htc_model="second"))
    second = _model(1, CLONE_BED)
    assert first.bed_architecture_provider is not second.bed_architecture_provider
    old = SimulationContext.from_model(first)
    new = SimulationContext.from_model(second)
    assert old.text("bed.provenance.source") == CLONE_PROVENANCE["source"]
    assert old.text("bed.models.wall_htc", step="startup") == "dixon"
    assert new.text("bed.provenance.year") == "2026"
    assert new.text("bed.models.wall_htc", step="startup") == "second"


def test_a_saved_context_reads_back_without_the_registration(tmp_path):
    pydac.register_bed_architecture(CLONE, _clone_entry())
    frame = SimulationContext.from_model(_model(1, CLONE_BED)).to_dataframe()
    path = tmp_path / "context.pkl"
    frame.to_pickle(path)
    script = (
        "import sys, pandas as pd\n"
        "from pydac.core.bed_architectures import registered_architectures\n"
        "from pydac.post_processing.context.simulation_context import SimulationContext\n"
        "assert registered_architectures() == ('packed_bed',)\n"
        "context = SimulationContext(pd.read_pickle(sys.argv[1]))\n"
        "year = context.text('bed.provenance.year')\n"
        "assert isinstance(year, str) and year == '1996', year\n"
        "assert context.value('bed.design.eps_b') == 0.38\n"
        "print(context.architecture)\n"
    )
    result = subprocess.run([sys.executable, "-c", script, str(path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == CLONE

    # The marker must repeat the saved name: a misspelled name is refused.
    typo = frame.copy()
    typo.loc[typo["Parameter ID"].eq("bed.architecture"), "Value"] = "test_clone_bd"
    with pytest.raises(ValueError, match="does not match"):
        SimulationContext(typo)
    with pytest.raises(ValueError, match="no 'bed.registered_architecture' row"):
        SimulationContext(frame.loc[frame["Parameter ID"].ne("bed.registered_architecture")])

    duplicated = pd.concat([frame, frame.loc[frame["Parameter ID"].eq("bed.provenance.year")]])
    with pytest.raises(ValueError, match="Duplicate"):
        SimulationContext(duplicated)
    retyped = frame.copy()
    retyped.loc[retyped["Parameter ID"].eq("bed.design.eps_b"), "Value"] = "0.38"
    with pytest.raises(TypeError, match="real number"):
        SimulationContext(retyped)


class _Law:
    def __init__(self, name="law", linear=2.0, inertial=3.0):
        self.name, self.linear_coefficient = name, linear
        self.inertial_coefficient_per_density = inertial

    def velocity_from_gradient(self, gradient, density):
        from pydac.core.bed_architectures import solve_quadratic_drag_velocity
        return solve_quadratic_drag_velocity(
            -np.asarray(gradient, dtype=float), self.linear_coefficient,
            self.inertial_coefficient_per_density * np.asarray(density, dtype=float))


def _dispersion_result(**changes):
    result = {"model": "x", **{key: np.full(4, 1e-4) for key in
                               ("Disp_ax", "Disp_rad", "inv_Pe_m_ax", "inv_Pe_m_rad")}}
    result.update(changes)
    return result


def _structure(**changes):
    from pydac.core.bed_architectures import ResolvedStructure
    fields = dict(eps_flow=0.38, eps_total=0.6, rho_sorbent=460.0, acc_structure=0.0)
    fields.update(changes)
    return ResolvedStructure(**fields)


@pytest.mark.parametrize("method, override, message", [
    ("resolve_structure", dict(resolve_structure=lambda d, p: _structure(rho_sorbent=0.0)),
     "rho_sorbent"),
    ("resolve_structure", dict(resolve_structure=lambda d, p: _structure(eps_total=0.3)),
     "eps_flow"),
    ("resolve_pressure_drop", dict(resolve_pressure_drop=lambda d, x: _Law()), None),
    ("resolve_pressure_drop", dict(resolve_pressure_drop=lambda d, x: _Law(name=" ")), "name"),
    ("resolve_pressure_drop", dict(resolve_pressure_drop=lambda d, x: _Law(linear=0.0)),
     "finite and positive"),
    ("resolve_pressure_drop", dict(resolve_pressure_drop=lambda d, x: _Law(
        linear=np.full(3, 2.0))), "scalar"),
    ("resolve_dispersion", dict(resolve_dispersion=lambda d, x: _dispersion_result()), None),
    ("resolve_dispersion", dict(resolve_dispersion=lambda d, x: _dispersion_result(model="")),
     "model"),
    ("resolve_dispersion", dict(resolve_dispersion=lambda d, x: _dispersion_result(
        architecture="monolith", provenance="reused packed-bed dispersion")), "unexpected keys"),
    ("resolve_dispersion", dict(resolve_dispersion=lambda d, x: _dispersion_result(
        Disp_ax=np.array([1e-4, -1e-4, 1e-4, 1e-4]))), "finite and positive"),
    ("resolve_dispersion", dict(resolve_dispersion=lambda d, x: _dispersion_result(
        inv_Pe_m_rad=np.array([1e-4, np.nan, 1e-4, 1e-4]))), "finite and positive"),
    ("resolve_thermal_transport", dict(resolve_thermal_transport=lambda d, x: {"lambda_ax": 1.0}),
     "ResolvedTransport"),
    ("resolve_structure", dict(resolve_structure=lambda d, p: 1 / 0), "division by zero"),
])
def test_each_wrapper_check(method, override, message):
    from pydac.core.bed_architectures import ParticleProperties
    provider, design = _provider(**override)
    argument = (ParticleProperties(density=744.0, porosity=0.338, radius=2.5e-4,
                                   thermal_conductivity=0.43)
                if method == "resolve_structure" else _inputs())
    if message is None:  # a well-formed result passes
        getattr(provider, method)(design, argument)
        return
    with pytest.raises((TypeError, ValueError), match=message) as error:
        getattr(provider, method)(design, argument)
    assert CLONE in str(error.value) and method in str(error.value)


def test_the_column_binds_pydacs_own_quadratic_law_with_float_coefficients():
    from pydac.core.bed_architectures import solve_quadratic_drag_velocity

    for bad in (np.array(2.0), True, np.bool_(True), "2.0"):
        provider, design = _provider(resolve_pressure_drop=lambda d, x, bad=bad: _Law(linear=bad))
        with pytest.raises(TypeError, match="real scalar"):
            provider.resolve_pressure_drop(design, _inputs())

    class OnlyAtAirDensity(_Law):
        """Right at rho = 1.2 only: the law of another density everywhere."""

        def velocity_from_gradient(self, gradient, density):
            return super().velocity_from_gradient(gradient, 1.2)

    provider, design = _provider(resolve_pressure_drop=lambda d, x: OnlyAtAirDensity())
    with pytest.raises(ValueError, match="at every density"):
        provider.resolve_pressure_drop(design, _inputs())

    returned = _Law(name=" my_law ", linear=np.float64(2.0), inertial=3)
    provider, design = _provider(resolve_pressure_drop=lambda d, x: returned)
    law = provider.resolve_pressure_drop(design, _inputs())
    assert law is not returned and law.name == "my_law"
    assert type(law.linear_coefficient) is float and type(law.inertial_coefficient_per_density) is float
    returned.linear_coefficient = 99.0  # the provider's object no longer matters
    assert law.linear_coefficient == 2.0
    gradients = np.array([-2.0, 0.0, 3.0])
    expected = solve_quadratic_drag_velocity(-gradients, 2.0, 3.0 * np.asarray(1.7))
    _same_bits(law.velocity_from_gradient(gradients, 1.7), expected, "velocity")


def test_results_are_read_once_and_rebuilt_as_pydac_values():
    from pydac.core.bed_architectures import ParticleProperties, ResolvedStructure

    class Lying(ResolvedStructure):
        """Once armed: positive on the first read of rho_sorbent, negative afterwards."""

        def __getattribute__(self, name):
            state = object.__getattribute__(self, "__dict__")
            if name == "rho_sorbent" and "reads" in state:
                state["reads"] += 1
                return 460.0 if state["reads"] == 1 else -99.0
            return object.__getattribute__(self, name)

    def lying(design, particle):
        result = Lying(0.38, 0.6, 1.0, 0.0)
        object.__setattr__(result, "reads", 0)  # arm it once built
        return result

    particle = ParticleProperties(density=744.0, porosity=0.338, radius=2.5e-4,
                                  thermal_conductivity=0.43)
    provider, design = _provider(resolve_structure=lying)
    structure = provider.resolve_structure(design, particle)
    assert type(structure) is ResolvedStructure and structure.rho_sorbent == 460.0

    source = _dispersion_result()
    provider, design = _provider(resolve_dispersion=lambda d, x: source)
    resolved = provider.resolve_dispersion(design, _inputs())
    source["Disp_ax"][:] = 7.0  # the provider keeps and changes its own array
    assert np.all(resolved["Disp_ax"] == 1e-4)
    assert not np.shares_memory(resolved["Disp_ax"], source["Disp_ax"])


def test_any_failure_on_a_result_names_the_architecture_and_keeps_the_cause():
    pydac.register_bed_architecture(CLONE, _clone_entry(
        to_config=lambda d: ({"eps_b": 10**10000}, None)))
    provider = registered_bed_architecture(CLONE)
    with pytest.raises(ValueError, match="`to_config` result") as error:
        provider.build_design({"eps_b": 0.38}, None)
    assert CLONE in str(error.value) and isinstance(error.value.__cause__, OverflowError)

    class Broken:
        name = "broken"
        linear_coefficient = 2.0
        inertial_coefficient_per_density = 3.0

        def velocity_from_gradient(self, gradient, density):
            raise ZeroDivisionError("inside the law")

    provider, design = _provider(resolve_pressure_drop=lambda d, x: Broken())
    with pytest.raises(ValueError, match="`resolve_pressure_drop` result") as error:
        provider.resolve_pressure_drop(design, _inputs())
    assert isinstance(error.value.__cause__, ZeroDivisionError)


def test_material_rows_are_declared_one_to_one():
    from pydac.core.bed_architectures import RegisteredDesign

    rows = CLONE_ROWS + (ContextParameter("material", "bed.material.k_solid", "W/m/K"),)
    build = (lambda design, material: (PackedBedDesign(eps_b=design["eps_b"]), dict(material)))
    entry = dict(build_design=build, context_parameters=rows,
                 to_config=lambda obj: ({"eps_b": obj[0].eps_b}, obj[1]),
                 resolve_structure=lambda d, p: resolve_packed_bed_structure(d[0], p))
    pydac.register_bed_architecture(CLONE, _clone_entry(**entry))
    provider = registered_bed_architecture(CLONE)
    design = provider.build_design({"eps_b": 0.38}, {"k_solid": 2.5})
    assert isinstance(design, RegisteredDesign) and dict(design.material) == {"k_solid": 2.5}
    with pytest.raises(ValueError, match="material keys"):
        provider.build_design({"eps_b": 0.38}, {"k_solid": 2.5, "k_fluid": 0.1})
    with pytest.raises(ValueError, match="material keys"):
        provider.build_design({"eps_b": 0.38}, {})


def test_the_snapshot_cannot_be_changed_or_emptied():
    provider, _ = _provider()
    for name in ("wall_htc_model", "provenance", "_callbacks"):
        with pytest.raises(AttributeError):
            delattr(provider, name)
        with pytest.raises(AttributeError):
            setattr(provider, name, None)
    with pytest.raises(TypeError):
        provider.provenance["source"] = "changed"


def test_closure_inputs_hold_detached_copies_also_after_pickling():
    import pickle

    class Aliasing(np.ndarray):
        def copy(self, *args, **kwargs):  # would hand back the same memory
            return self

    source = np.full(4, 2e-5).view(Aliasing)
    inputs = ClosureInputs(**{**vars(_inputs()), "molecular_diffusivity": source})
    stored = inputs.molecular_diffusivity
    assert type(stored) is np.ndarray and not np.shares_memory(stored, source)
    assert not stored.flags.writeable
    again = pickle.loads(pickle.dumps(inputs))
    for name, value in vars(inputs).items():
        _same_bits(getattr(again, name), value, name)
    assert not again.molecular_diffusivity.flags.writeable
    assert not again.schmidt.flags.writeable
