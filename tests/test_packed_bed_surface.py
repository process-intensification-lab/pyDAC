"""Public architecture boundary, including explicit rejection of unknown inputs."""

import pytest

from pydac.core import bed_architectures
from pydac.core.bed_architectures import build_bed_config, registered_architectures, resolve_bed_config
from pydac.config import ConfigurationError, experiment_to_document, load_experiment
from pydac.inputs.case_io import case_fingerprint


def test_registry_and_exports_are_packed_bed_only():
    assert registered_architectures() == ("packed_bed",)
    assert bed_architectures.get_architecture_spec("packed_bed").spatial_modes == ("1d", "2d")


@pytest.mark.parametrize("nr", [1, 3])
def test_default_preset_and_explicit_bed_have_identical_fingerprints(nr):
    default = load_experiment(settings={"grid.nr": nr})
    alias = load_experiment(settings={"grid.nr": nr, "bed": "packed_bed_default"})
    explicit = load_experiment(settings={"grid.nr": nr, "bed": resolve_bed_config()})
    assert case_fingerprint(default) == case_fingerprint(alias) == case_fingerprint(explicit)
    document = experiment_to_document(explicit)
    assert experiment_to_document(load_experiment(document)) == document


@pytest.mark.parametrize("nr", [1, 3])
@pytest.mark.parametrize("bed", [
    "other_bed_default",
    {"architecture": "other_bed", "design": {"cell_size": 0.005, "porosity": 0.9},
     "material": {"thermal_conductivity": 20.0, "density": 2000.0, "heat_capacity": 900.0}},
])
def test_unknown_architecture_is_rejected_before_model_construction(nr, bed):
    error = ValueError if isinstance(bed, str) else ConfigurationError
    with pytest.raises(error, match="Unknown (bed preset|architecture)"):
        load_experiment(settings={"grid.nr": nr, "bed": bed})


def test_detached_api_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_bed_config(architecture="other_bed", design={})


@pytest.mark.parametrize("extra", [{"porosity": 0.9}, {"cell_size": 0.005}])
def test_packed_bed_rejects_foreign_design_fields(extra):
    with pytest.raises(ValueError):
        build_bed_config(architecture="packed_bed", design={"eps_b": 0.38, **extra})


def test_packed_bed_rejects_structural_material():
    with pytest.raises(ValueError, match="material=None"):
        build_bed_config(architecture="packed_bed", design={"eps_b": 0.38},
                         material={"density": 2710.0})


@pytest.mark.parametrize("nr", [1, 3])
def test_persisted_context_roundtrip_and_unknown_architecture_rejection(nr):
    from pydac.core.column_model_factory import ColumnModelFactory
    from pydac.config.cycle import CycleConfig
    from pydac.post_processing.context.simulation_context import SimulationContext

    cycle = CycleConfig.from_experiment(load_experiment(settings={"grid.nz": 4, "grid.nr": nr}))
    model = ColumnModelFactory(cycle).build_column_model(
        step="startup", solver_options=cycle.solver_for_step(
            use_sparsity=False, compute_numerical_sparsity=False))
    context = SimulationContext.from_model(model)
    frame = context.to_dataframe()
    assert SimulationContext(frame).architecture == "packed_bed"
    frame.loc[frame["Parameter ID"].eq("bed.architecture"), "Value"] = "other_bed"
    with pytest.raises(ValueError, match="[Uu]nsupported.*architecture"):
        SimulationContext(frame)


def test_radial_conductivity_is_continuous_at_zero_flow():
    from pydac.core.bed_architectures.packed_bed import radial_effective_conductivity

    def lambda_rad(reynolds):
        return radial_effective_conductivity(
            eps_flow=0.38, lambda_0=0.2, gas_conductivity=0.026, reynolds=reynolds,
            prandtl=0.71, particle_diameter=5e-4, inlet_diameter=0.1,
        )[0]

    # lambda_0 already holds the fluid conduction: no molecular term at Pe_h = 0.
    assert lambda_rad(0.0) == pytest.approx(0.2)
    assert lambda_rad(10.0) == pytest.approx(0.2 + 10.0 * 0.71 / (8.65 * (1.0 + 19.4 / 200.0**2)) * 0.026)
    assert lambda_rad(1e-12) == pytest.approx(lambda_rad(0.0))
    assert lambda_rad(1.0) > lambda_rad(0.0)
