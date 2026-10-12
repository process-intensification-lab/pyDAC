"""
pyDAC - Core column method-of-lines model.

This module implements `ColumnModelMoL`, the main 1D/2D adsorber model used in
pyDAC, including transport operators, source terms, boundary conditions,
reference scaling, and KPI state bookkeeping for TVSA steps.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

import math
from collections.abc import Mapping
import numpy as np
from pydac.core.pymrm_caching import ConvfluxUpwindCache, GradCache
from scipy.constants import gas_constant as R_gas
from scipy.sparse import csr_array, issparse
from pydac.core.bed_architectures import (
    CHANNELING_FLOW_MODELS,
    PackedBedDesign,
    ParticleProperties,
    ResolvedArchitecture,
    channeling_inlet_flux_weights,
    packed_bed_ring_inlet_flux_weights,
    packed_bed_ring_structure_factors,
    resolve_channeling_structure,
    resolve_packed_bed_dispersion,
    resolve_packed_bed_pressure_drop,
    resolve_packed_bed_radial_pressure_drop,
    resolve_packed_bed_structure,
    resolve_packed_bed_thermal_transport,
    validate_channeling_support,
)
from pydac.core.column_model_input import ColumnModelInput
from pydac.core.species_catalog import get_species_properties
from pydac.core.state_layout import Mixture, StateLayout, simulation_scalar_states
from pymrm import (
    construct_convflux_bc,
    construct_div,
    construct_coefficient_matrix,
    construct_grad,
    interp_cntr_to_stagg_tvd,
    interp_cntr_to_stagg,
    interp_stagg_to_cntr,
    vanleer,
    muscl,
)

from pydac.utils.column import (
    build_runtime_properties,
    print_runtime_property_summary,
    require_water_component,
    resolve_fixed_adsorbates,
    resolve_heats_of_adsorption,
    resolve_initial_solid_loadings,
    resolve_solid_heat_capacity,
    validate_architecture_support,
    validate_flat_vector,
    validate_warm_start_fields,
)
from pydac.utils.flow_transforms import (
    cap_pressure_at_saturation,
    check_saturation_pressure_agrees,
    ideal_vapor_density,
)
from pydac.utils.vector_drag import ISOTROPIC_VECTOR, normal_velocity, transverse_gradients
from pydac.utils.solver.flags import env_flag
from pydac.utils.solver.species_balance import build_species_balance_diagnostic


# Fixed operators are built by pymrm as CSC, while the RHS only applies them
# as matrix-vector products. CSR performs that row-wise operation more cheaply.
# Keep the original representation available for exact A/B checks.
_RHS_CSR_OPERATORS = env_flag("PYDAC_RHS_CSR_OPERATORS")

# Thermodynamic datum for sensible enthalpy.  This is distinct from
# ``self.T_ref``, which is the user-declared temperature scale T0 used to
# nondimensionalize the temperature state.
ENTHALPY_REFERENCE_TEMPERATURE_K = 298.15


class ColumnModelMoL:
    """
    Simulates adsorption-diffusion-convection systems in a 2D cylindrical geometry 
    with axial and radial components.
    """
    # 1. Construction, configuration, geometry, and structure resolution

    def __init__(self, column_input: ColumnModelInput):
        """
        Initialize one column-model step from its typed construction request.

        Parameters
        ----------
        column_input : ColumnModelInput
            Complete typed configuration, runtime thermo/kinetic components,
            and optional warm-start state for one concrete cycle step.

        Initializes default physical properties and sets up the concentration grid.
        Also initializes boundary conditions, initial conditions, velocity fields, 
        physical parameters and constant Jacobian matrices.
        """        
        self.N2_O2_ratio = 78.084 / 20.94       # N2/O2 molar ratio in dry air (National Oceanic and Atmospheric Administration (NOAA))

        if not isinstance(column_input, ColumnModelInput):
            raise TypeError("`column_input` must be a ColumnModelInput instance.")

        # Optional, read-only hook used by targeted numerical diagnostics.  It
        # stays disabled in ordinary simulations and must never alter the RHS.
        self._species_balance_diagnostic_callback = None

        self.column_config = column_input.config
        self.bed_config = self.column_config.bed
        self.bed_architecture = self.bed_config["architecture"]

        # Bind the resolved configuration to the model's own namespace
        self._bind_inputs()
        self._set_options(column_input)
        
        # Set simulations parameters
        self.set_simulation_parameters()
        if not self.startup:
            self.store_prev_fields(column_input.initial_state)
        validate_architecture_support(
            self.bed_architecture, is_1d=self.is_1d, Nr=self.Nr
        )

        # Resolve the architecture structure
        self._resolve_bed_structure()
        self.init_radial_structure_factors()

        # Set reference conditions
        self.set_reference_conditions()
        
        # Set initial conditions
        self.set_ic()

        # Set the feed conditions
        self.set_feed()

        # Initialize gas mixture properties
        self.init_gas_mix()

        # Initialize solid heat capacity
        self.init_solid_thermo()

        # Calculate the system parameters
        self.calculate_system_parameters()

        # Set the dispersion coefficients
        self.set_disp()
        
        # Set conduction coefficients
        self.set_cond()

        # Set the boundary conditions
        self.set_bc()

        # Initialize Jacobian matrices for the system
        self.init_Jac()
        
        # Initialize the inlet fluxes and velocity fields
        self.init_inlet_fluxes()

        # One structured, JSON-safe source for runtime property reporting.
        self._build_runtime_properties()
        self._print_runtime_property_summary()

    def _bind_inputs(self):
        """Bind the resolved public configuration to the model's own namespace.

        Every value below comes from one resolved `ColumnModelConfig` — the
        same record a JSON case, the terminal, and a runner all produce — so no
        simulation input is stated in Python any more. The physics keeps
        reading short names (`self.T_feed`, `self.p_feed`, ...); this is the
        only place where those names are given their values.
        """
        config = self.column_config
        step = config.step_settings
        composition, flow = step.composition, step.flow
        thermal, pressure = step.thermal, step.pressure
        horizon, termination = step.horizon, step.termination

        # Discretization and solver tolerances
        self.Nz = int(config.grid.nz)                    # axial grid points
        self.Nr = int(config.grid.nr)                    # radial grid points; `Nr = 1` activates 1D mode
        self.Nr_plot_2d = int(config.grid.nr_plot_2d)    # pseudo-radial resolution to render 2D plots from 1D runs
        self.Nt = int(config.grid.nt)                    # output time points
        self.rtol = float(config.solver_parameters.rtol)
        self.atol = float(config.solver_parameters.atol)
        self.atol_scalar_mass = float(config.solver_parameters.atol_scalar_mass)
        self.atol_scalar_energy = float(config.solver_parameters.atol_scalar_energy)

        # Column geometry and wall material
        self.L_b = config.geometry.length_m              # column length [m]
        self.R_b = config.geometry.radius_m              # column internal radius [m]
        self.tau_ar = config.geometry.aspect_ratio       # aspect ratio [-]
        self.V_bed = config.geometry.volume_m3           # selected bed volume [m3]
        self.t_b = config.wall.thickness_m               # wall thickness [m]
        self.cp_w = config.wall.specific_heat_J_kgK      # wall specific heat capacity [J/kg/K]
        self.rho_w = config.wall.density_kg_m3           # wall density [kg/m3]
        self.lambda_w = config.wall.conductivity_W_mK    # wall thermal conductivity [W/m/K]
        self.U_w_override = config.wall.overall_htc_override_W_m2K
        self.h_bed_wall_override = config.wall.bed_wall_htc_override_W_m2K

        # Sorbent particle properties
        self.lambda_p = config.sorbent.conductivity_W_mK          # [W/m/K]
        self.rho_p = config.sorbent.density_kg_m3                 # [kg/m3]
        self.R_p = config.sorbent.particle_radius_m               # [m]
        self.eps_p = config.sorbent.porosity                      # [-]
        self.solid_heat_capacity = config.sorbent.heat_capacity

        # Sorbent inventory: validated and normalized by `utils.column.validation`,
        # which owns the declared-adsorbate coverage and sign conventions.
        self.heat_of_adsorption_J_mol = resolve_heats_of_adsorption(
            config.sorbent.heats_of_adsorption_J_mol, config.mixture.adsorbates
        )
        self.fixed_adsorbates = resolve_fixed_adsorbates(config.sorbent.fixed_adsorbates)
        self.initial_solid_loadings_mol_per_kg = resolve_initial_solid_loadings(
            getattr(composition, "initial_solid_loadings_mol_per_kg", None),
            config.mixture.adsorbates,
        )

        # Reference state used for nondimensionalization
        self.T_ref = config.reference.temperature_K
        self.T_enthalpy_ref_K = ENTHALPY_REFERENCE_TEMPERATURE_K
        self.T_enthalpy_ref_nd = self.T_enthalpy_ref_K / self.T_ref
        self.p_ref = config.reference.pressure_Pa
        self.rh_ref = config.reference.relative_humidity
        self.u_ref = config.reference.velocity_m_s

        # Step composition: `None` marks a value the warm start or the model
        # itself supplies (initial state, O2/N2 closure of an air feed).
        self.rh_feed = composition.relative_humidity
        self.rh_initial = composition.initial_relative_humidity

        # Step flow: an unstated valve conductance is an ideal valve.
        blowdown = flow.blowdown_valve_conductance
        pressurization = flow.pressurization_valve_conductance
        self.vol_flowrate = flow.volumetric_flow_m3_s
        self.m_dot_steam = flow.steam_mass_flow_kg_s
        self.alpha_bd = np.inf if blowdown is None else blowdown
        self.alpha_ps = np.inf if pressurization is None else pressurization

        # Step temperatures and external wall heat transfer
        self.T_feed = thermal.feed_temperature_K
        self.T_initial = thermal.initial_temperature_K
        self.T_jacket = thermal.jacket_temperature_K
        self.T_amb = thermal.ambient_temperature_K
        self.htc_ext = thermal.external_htc_W_m2K

        # Step pressures
        self.p_initial = pressure.initial_pressure_Pa
        self.p_out = pressure.outlet_pressure_Pa
        self.p_feed = pressure.feed_pressure_Pa
        self.p_amb = pressure.ambient_pressure_Pa

        # Step horizon: `t_end` is the nominal duration used by time-based
        # termination, `t_end_max` the cap the solver applies in event mode.
        self.t_end = horizon.nominal_time_s
        self.t_end_max = horizon.event_max_time_s

        # Step termination control
        self.termination_mode = termination.mode
        self.termination_metric = termination.metric
        self.termination_direction = termination.direction
        self.termination_threshold_value = termination.threshold_value
        self.termination_threshold_source = termination.threshold_source
        self.termination_threshold_offset = termination.threshold_offset
        self.termination_label = termination.label
        self.cool_stop_on_low_pressure = termination.stop_on_low_pressure
        self.cool_pressure_stop_pa = (
            0.0
            if termination.low_pressure_stop_Pa is None
            else termination.low_pressure_stop_Pa
        )
        self.stop_on_pressure_plateau = termination.stop_on_pressure_plateau
        self.pressure_plateau_rate_tolerance_nd = (
            termination.pressure_plateau_rate_tolerance_nd
        )
        self.pressure_plateau_arm_below_Pa = termination.pressure_plateau_arm_below_Pa

        # Cooldown inherits the preceding state without artificial redistribution.
        # Step-driven behavior
        self.startup = config.step == "startup"
        self.relax_cool_hydraulics = False
        self.flatten_cool_axial_profiles = False

    def _set_options(self, column_input: ColumnModelInput):
        """Bind runtime components, step options, and the optional warm start."""
        components = column_input.components
        config = column_input.config

        self.equilibrium_system = components.equilibrium_system
        self.kinetics = components.kinetics
        self.step = config.step
        self.use_sparsity = config.solver.use_sparsity
        self.compute_numerical_sparsity = config.solver.compute_numerical_sparsity
        self.cvode_sparsity_mode = config.solver.cvode_sparsity_mode

    def _water_component_model(self):
        """Return the selected H2O correlation used for RH and steam EOS calls."""
        return require_water_component(self.equilibrium_system)

    def set_simulation_parameters(self):
        """
        Set the simulation parameters including grids, geometry, and reference values.

        The spatial discretization is specified directly through `Nz` and
        `Nr`; it is no longer inferred from the bed aspect ratio.
        """
        self.Nz = int(self.Nz)
        self.Nr = int(self.Nr)
        if self.Nz < 1 or self.Nr < 1:
            raise ValueError("`Nz` and `Nr` must both be positive integers.")

        self.is_1d = (self.Nr == 1)
        if self.is_1d:
            print(
                "[INFO] 1D simulation mode enabled because `Nr = 1`. "
                "The solve runs on the axial grid only; post-processing will "
                "expand fields uniformly in radius for 2D visualizations."
            )

        mixture_config = self.column_config.mixture
        self.mixture = Mixture(
            mixture_config.species,
            closure=mixture_config.closure_species,
        )
        self.state_layout = StateLayout.build(
            mixture=self.mixture,
            adsorbates=mixture_config.adsorbates,
            nz=self.Nz,
            nr=self.Nr,
            scalar_states=simulation_scalar_states(mixture_config.adsorbates),
        )

        self.gas_species = self.mixture.species
        self.independent_species = self.mixture.independent_species
        self.closure_species = self.mixture.closure
        self.adsorbates = self.state_layout.adsorbates
        self.gas_index = {species: i for i, species in enumerate(self.gas_species)}
        self.independent_index = {
            species: i for i, species in enumerate(self.independent_species)
        }
        self.adsorbate_index = {
            species: i for i, species in enumerate(self.adsorbates)
        }
        self.independent_gas_indices = np.array(
            [self.gas_index[species] for species in self.independent_species],
            dtype=int,
        )
        self.adsorbate_gas_indices = np.array(
            [self.gas_index[species] for species in self.adsorbates],
            dtype=int,
        )
        self.state_field_by_name = {
            field.name: field for field in self.state_layout.fields
        }
        self.gas_state_fields = tuple(
            self.state_layout.gas_field(species)
            for species in self.independent_species
        )
        self.loading_state_fields = tuple(
            self.state_layout.loading_field(species)
            for species in self.adsorbates
        )
        self.scalar_state_fields = self.state_layout.fields_in("y_scalars")
        self.mass_scalar_fields = tuple(
            self.state_field_by_name[f"m_{species}"]
            for species in self.adsorbates
        )

        # Immutable catalog records and plain numerical arrays share the
        # declared mixture order everywhere below.
        self._species_properties = tuple(
            get_species_properties(species) for species in self.gas_species
        )
        self.species_properties = {
            record.name: record for record in self._species_properties
        }
        self.MW = np.array(
            [record.molecular_weight_kg_per_mol for record in self._species_properties]
        )
        self.lennard_jones_sigma_A = np.array(
            [record.lennard_jones_sigma_m / 1e-10 for record in self._species_properties]
        )
        self.lennard_jones_epsilon_over_k_K = np.array(
            [record.lennard_jones_epsilon_over_k_K for record in self._species_properties]
        )
        self.fuller_diffusion_volume_cm3_per_mol = np.array(
            [
                record.fuller_diffusion_volume_m3_per_mol * 1e6
                for record in self._species_properties
            ]
        )
        self.species_conductivity_W_mK = np.array(
            [
                record.thermal_conductivity_reference_W_per_mK
                for record in self._species_properties
            ]
        )

        self._bind_current_dac_property_aliases()

        # State dimensions now come directly from the generated layout.
        self.Nc_gas = len(self.gas_species)
        self.Nc_solid = len(self.adsorbates)
        self.Nc_ci = len(self.independent_species)

        if self.startup:
            print("\nGenerated state layout...")
            print(self.state_layout)

        self.shape_gas = (self.Nz, self.Nr, self.Nc_gas)
        self.shape_ci = (self.Nz, self.Nr, self.Nc_ci)
        self.shape_q = (self.Nz, self.Nr, self.Nc_solid)
        self.shape_T = (self.Nz, self.Nr, 1)
        self.shape_Tw = (self.Nz, 1, 1)   
        self.shape_single = (self.Nz, self.Nr, 1)
        self.shape_u_ax = (self.Nz + 1, self.Nr, self.Nc_ci)
        self.shape_u_ax_single = (self.Nz + 1, self.Nr, 1)
        self.shape_u_rad = (self.Nz, self.Nr + 1, self.Nc_ci)
        self.shape_u_rad_single = (self.Nz, self.Nr + 1, 1)
        
        # Setup axial and radial grids
        self.z_f = np.linspace(0, 1, self.shape_gas[0] + 1)
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])
        self.r_f = np.linspace(0, 1, self.shape_gas[1] + 1)
        self.r_c = 0.5 * (self.r_f[:-1] + self.r_f[1:])
        self.r_f_dim = np.linspace(0.0, self.R_b, self.Nr + 1)  

        # Geometric parameters
        self.d_p = 2 * self.R_p
        self.R_b_ext = self.R_b + self.t_b
        self.d_b = 2 * self.R_b
        self.L_R_ratio = self.L_b / self.R_b
        self.pressure_drop_formulation = "scalar" if self.is_1d else ISOTROPIC_VECTOR

        self.inlet_area = math.pi * self.R_b ** 2
        self.bed_volume = self.V_bed if self.V_bed is not None else math.pi * self.R_b**2 * self.L_b
        
        self.A_rings = np.pi * (self.r_f_dim[1:]**2 - self.r_f_dim[:-1]**2)          
        self.dz = np.diff(self.z_f) * self.L_b    
        self.V_cells = self.dz[:, None] * self.A_rings[None, :]
        
        # Parameters for flow conditions
        self._resolve_inlet_velocity()
        self.bd_fact = 1 / (self.alpha_bd * self.L_b)
        self.ps_fact = 1 / (self.alpha_ps * self.L_b)

        # Parameters for interpolation
        self.dz1 = self.z_f[2] - self.z_f[0]
        self.dz2 = self.z_f[1] - self.z_f[0]
        self.dz3 = self.z_f[2] - self.z_f[1]
        
        # Printing progress tracker        
        self.progress_marks = [
            0.01, 0.1, 0.5, 1, 2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        ]
        self.next_mark_idx = 0  
        self.parameters_computed = False

    def _bind_current_dac_property_aliases(self):
        """Expose only the DAC aliases still consumed by reporting code."""
        for species in ("CO2", "H2O", "O2", "N2"):
            if species not in self.gas_index:
                continue
            index = self.gas_index[species]
            setattr(self, f"MW_{species}", self.MW[index])

        for species, heat in self.heat_of_adsorption_J_mol.items():
            if species in ("CO2", "H2O"):
                setattr(self, f"DHiso_{species}", heat)

    def _resolve_bed_structure(self):
        """Resolve and expose the selected architecture's structure contract."""
        particle = ParticleProperties(
            density=self.rho_p,
            porosity=self.eps_p,
            radius=self.R_p,
            thermal_conductivity=self.lambda_p,
        )
        self.particle_properties = particle

        design = PackedBedDesign(**self.bed_config["design"])
        resolved = resolve_packed_bed_structure(design, particle)

        self.bed_design = design
        self.resolved_structure = resolved

        self.eps_flow = self.resolved_structure.eps_flow
        self.eps_total = self.resolved_structure.eps_total
        self.rho_sorbent = self.resolved_structure.rho_sorbent
        self.acc_structure = self.resolved_structure.acc_structure

        self._resolve_channeling_structure()

    def _resolve_channeling_structure(self):
        """Resolve the radial bed structure of a channeling flow model.

        The scalar `eps_flow`, `eps_total` and `rho_sorbent` above stay the bed
        means used for normalization and for every retained mean-voidage
        closure; the radial fields resolved here carry the local structure.
        """
        self.flow_model = "uniform"
        self.flow_model_compatibility = "native"
        self.channeling_structure = None

        if self.bed_architecture != "packed_bed":
            return

        self.flow_model = self.bed_design.resolved_flow_model
        self.flow_model_compatibility = self.bed_design.compatibility_status
        if self.flow_model not in CHANNELING_FLOW_MODELS:
            return

        validate_channeling_support(
            self.flow_model,
            architecture=self.bed_architecture,
            is_1d=self.is_1d,
            Nr=self.Nr,
            Nz=self.Nz,
        )
        self.channeling_structure = resolve_channeling_structure(
            self.bed_design,
            self.particle_properties,
            radius=self.R_b,
            particle_diameter=self.d_p,
            r_faces=self.r_f_dim,
        )

    def _resolve_inlet_velocity(self):
        """Resolve superficial inlet velocity from step flow inputs."""
        if self.step in ("s_purge", "s_des") and self.m_dot_steam is not None:
            self.m_dot_steam = float(self.m_dot_steam)
            print(f"Resolving inlet velocity from `m_dot_steam` = {self.m_dot_steam:.2e} kg/s for {self.step} step.")
            if self.m_dot_steam < 0:
                raise ValueError(f"`m_dot_steam` must be non-negative. Received {self.m_dot_steam}.")

            # Saturation pressure comes from the *configured* H2O component, not from
            # a hard-coded correlation, so swapping the water model moves this too.
            # The cap and the ideal-gas density are the ones `config.derivations`
            # uses when it turns `m_dot_steam` into a volumetric flow, so the two
            # paths cannot drift apart.
            water = self._water_component_model()
            p_sat = float(np.asarray(water.EOS(self.T_feed)).squeeze())
            check_saturation_pressure_agrees(p_sat, self.T_feed, context=self.step)
            p_for_rho = cap_pressure_at_saturation(
                self.p_feed, p_sat, temperature_K=self.T_feed, context=self.step
            )
            self.rho_steam_inlet = ideal_vapor_density(
                p_for_rho, self.T_feed, self.MW_H2O
            )
            print(f"Estimated steam density at inlet: {self.rho_steam_inlet:.2f} kg/m³ (using p={p_for_rho:.2f} Pa, T={self.T_feed:.2f} K)")
            if self.rho_steam_inlet <= 0:
                raise ValueError(
                    "Computed steam density is non-positive. Check `p_feed`, `T_feed`, and EOS settings."
                )

            self.vol_flowrate = self.m_dot_steam / self.rho_steam_inlet
            self.u_inlet = self.vol_flowrate / self.inlet_area if self.inlet_area > 0 else 0.0
            return

        if self.vol_flowrate is not None:
            self.vol_flowrate = float(self.vol_flowrate)
            self.u_inlet = self.vol_flowrate / self.inlet_area if self.inlet_area > 0 else 0.0
            return

        raise AttributeError(
            "Inlet flow is undefined. Set `m_dot_steam` [kg/s] for s_purge or s_des "
            "or `vol_flowrate` [m³/s] for other steps."
        )

    # 2. Reference state, feed, initial conditions, and inventories

    def _composition_from_named_inputs(
        self,
        supplied,
        *,
        relative_humidity,
        temperature_K,
        pressure_Pa,
        water_floor=0.0,
        pure_steam=False,
    ):
        """Resolve one declared mixture and reconstruct its closure species.

        Explicit mole fractions are read by species name. Water may instead be
        stated as relative humidity, and the current DAC air default may omit
        both O2 and N2. Exactly one final operation is special: the configured
        closure species is always `1 - sum(independent species)`.
        """
        fractions = {
            species: float(value)
            for species, value in supplied.items()
            if species in self.gas_index and value is not None
        }

        if "H2O" in self.gas_index:
            if pure_steam:
                fractions["H2O"] = 1.0
            elif relative_humidity is not None:
                p_sat = self._water_component_model().EOS(temperature_K)
                fractions["H2O"] = (
                    relative_humidity * p_sat / pressure_Pa + water_floor
                )

        # Preserve the documented dry-air N2/O2 ratio when those current DAC
        # species are both declared but their independent value is omitted.
        has_air_pair = "O2" in self.gas_index and "N2" in self.gas_index
        if has_air_pair:
            other_fraction = sum(
                fractions.get(species, 0.0)
                for species in self.independent_species
                if species not in ("O2", "N2")
            )
            air_fraction = 1.0 - other_fraction
            if "O2" in self.independent_index and "O2" not in fractions:
                fractions["O2"] = air_fraction / (1.0 + self.N2_O2_ratio)
            if "N2" in self.independent_index and "N2" not in fractions:
                fractions["N2"] = (
                    air_fraction * self.N2_O2_ratio / (1.0 + self.N2_O2_ratio)
                )

        missing = [
            species
            for species in self.independent_species
            if species not in fractions
        ]
        if missing:
            raise ValueError(
                "Missing mole fractions for independent gas species "
                f"{missing}; closure species is {self.closure_species!r}."
            )

        closure_fraction = 1.0
        for species in self.independent_species:
            closure_fraction -= fractions[species]
        if closure_fraction < -1e-12:
            raise ValueError(
                f"Independent mole fractions sum above one; computed "
                f"y_{self.closure_species}={closure_fraction:.6e}."
            )
        fractions[self.closure_species] = max(closure_fraction, 0.0)
        return {species: fractions[species] for species in self.gas_species}

    def _configured_mole_fractions(self, state):
        """Read reference/feed/initial fractions without embedding an order."""
        if state == "reference":
            source = self.column_config.reference
            supplied = {
                species: getattr(source, f"{species.lower()}_mole_fraction", None)
                for species in self.gas_species
            }
            return self._composition_from_named_inputs(
                supplied,
                relative_humidity=self.rh_ref,
                temperature_K=self.T_ref,
                pressure_Pa=self.p_ref,
            )

        composition = self.column_config.step_settings.composition
        suffix = "feed_mole_fraction" if state == "feed" else "initial_mole_fraction"
        supplied = {
            species: getattr(composition, f"{species.lower()}_{suffix}", None)
            for species in self.gas_species
        }
        if state == "feed":
            no_feed_flow = self.vol_flowrate == 0
            relative_humidity = 0.0 if no_feed_flow else self.rh_feed
            return self._composition_from_named_inputs(
                supplied,
                relative_humidity=relative_humidity,
                temperature_K=self.T_feed,
                pressure_Pa=self.p_feed,
                pure_steam=self.step in ("s_purge", "s_des"),
            )
        if state == "initial":
            return self._composition_from_named_inputs(
                supplied,
                relative_humidity=self.rh_initial,
                temperature_K=self.T_initial,
                pressure_Pa=self.p_initial,
                water_floor=1e-32,
            )
        raise ValueError(f"Unknown composition state {state!r}.")

    def _equilibrium_loadings_at_state(self, mole_fractions, temperature_K, pressure_Pa):
        """Evaluate equilibrium loadings, including ordered coupled dependencies."""
        partial_pressures = {
            species: mole_fractions[species] * pressure_Pa
            for species in self.gas_species
        }
        loadings = {species: 1e-32 for species in self.adsorbates}

        # One pass per adsorbate propagates one-way coupled dependencies. 
        for _ in range(len(self.adsorbates)):
            evaluated = self.equilibrium_system.equilibrium_loadings(
                partial_pressures_Pa=partial_pressures,
                temperature_K=temperature_K,
                current_loadings_mol_per_kg=loadings,
            )
            loadings.update(evaluated)
        return {species: loadings[species] for species in self.adsorbates}

    def set_reference_conditions(self):
        """
        Set the reference conditions for the simulation.

        This method sets the reference conditions, including total pressure,
        temperature, initial concentrations, equilibrium loadings, and key flow metrics.
        """

        # The reference state itself (T_ref, p_ref, rh_ref, y_CO2_ref, u_ref)
        # is a public input, bound with the rest of the configuration.
        self.y_ref_by_species = self._configured_mole_fractions("reference")
        self.y_ref = np.array(
            [[[self.y_ref_by_species[species] for species in self.gas_species]]]
        )

        self.c_tot_ref = self.p_ref / R_gas / self.T_ref
        self.c_ref = self.y_ref.reshape(-1) * self.c_tot_ref
        self.c_ref_by_species = {
            species: self.c_ref[self.gas_index[species]]
            for species in self.gas_species
        }
        nonpositive_reference_species = [
            species
            for species in self.gas_species
            if self.c_ref_by_species[species] <= 0.0
        ]
        if nonpositive_reference_species:
            raise ValueError(
                "Gas species require positive reference mole fractions because "
                "every concentration, including the closure species, is scaled by "
                f"its reference value: {nonpositive_reference_species}."
            )

        self.q_ref_by_species = self._equilibrium_loadings_at_state(
            self.y_ref_by_species,
            self.T_ref,
            self.p_ref,
        )
        self.q_ref = np.array(
            [self.q_ref_by_species[species] for species in self.adsorbates]
        )

        # Named species aliases remain for reporting compatibility.
        for species in self.gas_species:
            setattr(self, f"y_{species}_ref", self.y_ref_by_species[species])
            setattr(self, f"c_{species}_ref", self.c_ref_by_species[species])
        for species in self.adsorbates:
            setattr(self, f"q_eq_{species}_ref", self.q_ref_by_species[species])

        self.state_scale_by_field = {}
        for field in self.state_layout.fields:
            if field.scale_source == "gas_reference_concentration":
                scale = self.c_ref_by_species[field.species]
            elif field.scale_source == "adsorbate_reference_loading":
                scale = self.q_ref_by_species[field.species]
            elif field.scale_source == "total_reference_concentration":
                scale = self.c_tot_ref
            elif field.scale_source == "reference_temperature":
                scale = self.T_ref
            else:
                scale = 1.0
            self.state_scale_by_field[field.name] = float(scale)

        # Flow and transport properties
        self.beta_fact = self.p_ref / self.u_ref / self.L_b
        self.mass_flow_ref = self.u_ref * self.c_tot_ref * self.inlet_area * self.MW
        self.tau_hyd = self.L_b / self.u_ref


    def set_feed(self):
        """Set the feed concentrations and feed-state auxiliary values."""
        self.y_feed_by_species = self._configured_mole_fractions("feed")
        for species in self.gas_species:
            setattr(self, f"y_{species}_feed", self.y_feed_by_species[species])
        self.c_tot_feed = self.p_feed / R_gas / self.T_feed
        self.y_feed = np.array(
            [[[self.y_feed_by_species[species] for species in self.gas_species]]]
        )
        self.c_feed = self.c_tot_feed * self.y_feed
        self.T_cond = 197 * (self.p_feed / 1e5) ** 0.1367 - 97.97 + 273.15

        if self.step in ("startup", "ads") and "H2O" in self.gas_index:
            print(
                "Air-feed conditions: "
                f"T_feed={self.T_feed:.2f} K, RH={self.rh_feed:.4f}, "
                f"y_H2O={self.y_H2O_feed:.6e}."
            )

    def store_prev_fields(self, prev_fields):
        """
        Store previous-step fields for warm-start initialization.

        Parameters
        ----------
        prev_fields : dict
            Dictionary containing `ci`, `qi`, `T`, `Tw`, and `c_tot` arrays.
        """
        validate_warm_start_fields(
            prev_fields,
            step=self.step,
            layout_signature=self.state_layout.signature,
            Nz=self.Nz,
            Nr=self.Nr,
            Nc_ci=self.Nc_ci,
            Nc_solid=self.Nc_solid,
        )

        self.ci_prev = np.maximum(prev_fields['ci'], 1e-32)
        self.qi_prev = np.maximum(prev_fields['qi'], 1e-32)
        self.c_tot_prev = prev_fields['c_tot']
        self.T_prev = prev_fields['T']
        self.Tw_prev = prev_fields['T_wall']
        
    def set_ic(self):
        """
        Set the initial conditions for solved fields.

        This method sets the initial conditions for the fields, including
        molar concentrations, solid loadings, temperature and total concentration. 
        """
        if self.startup:
            self.set_ic_startup()
        else:
            self.set_ic_step()
                
    def set_ic_startup(self):
        """
        Set the initial conditions for the concentration field.

        The initial conditions are set for each species in the gas and solid phases and temperature field.
        The method calculates the initial concentrations based on the initial pressures and temperature.

        The concentration field (`c`) is initialized as a 3D array with dimensions corresponding to
        (axial grid points, radial grid points, species).
        """
        self.ci = np.empty(self.shape_ci, dtype=float)
        self.qi = np.empty(self.shape_q, dtype=float)
        self.T = np.empty(self.shape_T, dtype=float)
        self.Tw = np.empty(self.shape_Tw, dtype=float)
        
        self.ci[...] = np.full(self.shape_ci, 0.0, dtype=float)
        self.c_tot_initial = self.p_initial / R_gas / self.T_initial
        self.y_initial_by_species = self._configured_mole_fractions("initial")
        for species in self.gas_species:
            setattr(self, f"y_{species}_initial", self.y_initial_by_species[species])

        # Independent gas fields follow StateLayout order; the closure species
        # is reconstructed whenever the full composition is needed.
        for species, index in self.independent_index.items():
            self.ci[..., index] = (
                self.y_initial_by_species[species] * self.c_tot_initial + 1e-32
            )

        initial_loadings = self.initial_solid_loadings_mol_per_kg
        if initial_loadings is None:
            initial_loadings = self._equilibrium_loadings_at_state(
                self.y_initial_by_species,
                self.T_initial,
                self.p_initial,
            )
            for species, index in self.adsorbate_index.items():
                self.qi[..., index] = initial_loadings[species] + 1e-32
        else:
            for species, index in self.adsorbate_index.items():
                self.qi[..., index] = initial_loadings[species]

        # Temperature field
        self.T[..., 0] = self.T_initial 
        self.Tw[..., 0] = self.T_initial
                
        # Total concentration field
        self.c_tot = np.full(self.shape_single, self.c_tot_initial, dtype=float)
        
        self.y_initial = np.array(
            [[[self.y_initial_by_species[species] for species in self.gas_species]]]
        ) + 1e-32
        self._store_initial_mass_aliases(startup=True)

    def _relax_closed_bed_hydraulics(self):
        """
        Remove the residual pressure profile from a sealed-bed warm start.

        The previous step may end with a non-uniform pressure field and non-zero
        internal Ergun velocities. For the sealed cooldown step, we replace that
        state with a pressure-uniform gas profile that preserves the total gas
        inventory and the local gas composition while keeping solid loadings and
        temperatures unchanged.
        """
        c_tot_old = np.maximum(np.asarray(self.c_tot).copy(), 1e-32)
        ci_old = np.maximum(self.ci.copy(), 1e-32)
        T_old = np.maximum(np.asarray(self.T).copy(), 1e-32)

        if c_tot_old.ndim == 3:
            c_tot_old = c_tot_old[..., 0]
        if T_old.ndim == 3:
            T_old = T_old[..., 0]

        y_old = ci_old / c_tot_old[..., None]
        gas_holdup = self.eps_total * self._gas_2d * self.V_cells

        n_gas_old = np.sum(gas_holdup * c_tot_old)
        denom = np.sum(gas_holdup / T_old)
        if denom <= 0.0:
            raise ValueError(
                "Cooldown hydraulic relaxation failed because the equivalent-pressure denominator is non-positive."
            )

        p_old = c_tot_old * R_gas * T_old
        p_eq = n_gas_old * R_gas / denom
        c_tot_new = p_eq / (R_gas * T_old)

        c_tot_relaxed = np.maximum(c_tot_new, 1e-32)
        if np.asarray(self.c_tot).ndim == 3:
            self.c_tot[..., 0] = c_tot_relaxed
        else:
            self.c_tot[...] = c_tot_relaxed
        self.ci[...] = np.maximum(y_old * c_tot_relaxed[..., None], 1e-32)

        print(
            "Cooldown hydraulic relaxation enabled: "
            f"p_old min/max = {np.min(p_old):.2f}/{np.max(p_old):.2f} Pa -> p_eq = {p_eq:.2f} Pa"
        )

    def _flatten_cool_axial_profiles(self):
        """
        Remove axial structure from the cooldown warm start.

        Each field is replaced by its axial average at fixed radius/species so
        the cooldown starts from a z-uniform state while preserving the radial
        structure and the total inventory of each field.
        """
        def _flatten(field):
            field_arr = np.maximum(np.asarray(field).copy(), 1e-32)
            axial_mean = np.mean(field_arr, axis=0, keepdims=True)
            return np.broadcast_to(axial_mean, field_arr.shape).copy()

        self.ci[...] = _flatten(self.ci)
        self.qi[...] = _flatten(self.qi)
        self.c_tot[...] = _flatten(self.c_tot)
        self.T[...] = _flatten(self.T)
        self.Tw[...] = _flatten(self.Tw)

        c_tot_flat = np.asarray(self.c_tot)
        T_flat = np.asarray(self.T)
        if c_tot_flat.ndim == 3:
            c_tot_flat = c_tot_flat[..., 0]
        if T_flat.ndim == 3:
            T_flat = T_flat[..., 0]
        p_flat = c_tot_flat * R_gas * T_flat
        print(
            "Cooldown axial flattening enabled: "
            f"p min/max = {np.min(p_flat):.2f}/{np.max(p_flat):.2f} Pa"
        )

    def set_ic_step(self):
        """
        Set the initial conditions for the concentration field.

        The initial conditions are set for each species in the gas and solid phases and temperature field.
        The method calculates the initial concentrations based on the initial pressures and temperature.

        The concentration field (`c`) is initialized as a 3D array with dimensions corresponding to
        (axial grid points, radial grid points, species).
        """
        self.ci = self.ci_prev.copy()
        self.qi = self.qi_prev.copy()
        self.c_tot = self.c_tot_prev.copy()
        self.T = self.T_prev.copy()
        self.Tw = self.Tw_prev.copy()

        if self.relax_cool_hydraulics:
            self._relax_closed_bed_hydraulics()

        if self.flatten_cool_axial_profiles:
            self._flatten_cool_axial_profiles()

        self.c_tot_initial = np.mean(self.c_tot)
        
        c_tot = np.asarray(self.c_tot)
        if c_tot.ndim == 3:
            c_tot = c_tot[..., 0]
        independent_fractions = {
            species: np.mean(self.ci[..., index] / c_tot)
            for species, index in self.independent_index.items()
        }
        closure_fraction = 1.0
        for species in self.independent_species:
            closure_fraction -= independent_fractions[species]
        self.y_initial_by_species = {
            species: (
                closure_fraction
                if species == self.closure_species
                else independent_fractions[species]
            )
            for species in self.gas_species
        }
        self.y_initial = np.array(
            [[[self.y_initial_by_species[species] for species in self.gas_species]]]
        ) + 1e-32
        self._store_initial_mass_aliases(startup=False)

    def calculate_initial_mass(self, species, phase="both"):
        """
        Calculate the mass of a given species in the column.

        Parameters
        ----------
        species : str
            One declared mixture species (case-insensitive).
        phase : str, optional
            "gas", "solid", or "both" (default). A solid mass exists only
            when the species is a declared adsorbate.

        Returns
        -------
        float
            If phase is "gas" or "solid", returns that phase mass in kg.
        dict
            If phase is "both", returns {"gas": mgas, "solid": msolid, "total": mgas + msolid}.
        """
        sp = species.strip().upper()
        if sp not in self.gas_index:
            raise ValueError(
                f"Unknown species {species!r}. Declared mixture: {self.gas_species}."
            )
        MW = self.MW[self.gas_index[sp]]

        # Gas-phase calculation
        c_tot = np.asarray(self.c_tot)
        if c_tot.ndim == 3:
            c_tot = c_tot[..., 0]
        if sp == self.closure_species:
            c_species = c_tot - np.sum(self.ci, axis=-1)
        else:
            c_species = self.ci[..., self.independent_index[sp]]

        gas_holdup = self.eps_total * self._gas_2d
        m_gas = np.sum(c_species * self.V_cells * gas_holdup) * MW  # kg

        m_solid = 0.0
        if sp in self.adsorbate_index:
            m_solid = np.sum(
                self.qi[..., self.adsorbate_index[sp]] * self.V_cells * self._solid_2d
            ) * self.rho_sorbent * MW

        if phase.lower() == "gas":
            return m_gas
        elif phase.lower() == "solid":
            return m_solid
        elif phase.lower() == "both":
            return {"gas": m_gas, "solid": m_solid, "total": m_gas + m_solid}
        else:
            raise ValueError("phase must be 'gas', 'solid', or 'both'")

    def _store_initial_mass_aliases(self, *, startup):
        """Store generic inventories and the existing DAC reporting aliases."""
        self.initial_mass_by_species = {
            species: self.calculate_initial_mass(species)
            for species in self.gas_species
        }

        for species in ("CO2", "H2O"):
            if species not in self.initial_mass_by_species:
                continue
            masses = self.initial_mass_by_species[species]
            setattr(
                self,
                f"mass_{species}_initial_f",
                0.0 if startup else float(np.asarray(masses["gas"]).item()),
            )
            setattr(
                self,
                f"mass_{species}_initial_s",
                0.0 if startup else float(np.asarray(masses["solid"]).item()),
            )
        for species in ("O2", "N2"):
            if species in self.initial_mass_by_species:
                if startup:
                    mass = (
                        self.c_tot_initial
                        * self.y_initial_by_species[species]
                        * self.bed_volume
                        * self.MW[self.gas_index[species]]
                        * self.eps_total
                    )
                else:
                    mass = self.initial_mass_by_species[species]["gas"]
                setattr(self, f"mass_{species}_initial", float(np.asarray(mass).item()))

    # 3. Gas, particle, and thermodynamic properties

    def _gas_property_composition(self):
        """Return the legacy step-level mixture used for gas properties.

        Closed-inlet steps evaluate properties from the gas already in the
        bed. Open-inlet steps evaluate them from the feed. This selection is
        architecture-independent and is resolved afresh for every model.
        """
        if self.step in ("blow", "heat", "w_purge", "w_des", "cool"):
            self.gas_property_composition_source = "initial"
            return self.y_initial
        self.gas_property_composition_source = "feed"
        return self.y_feed

    def init_gas_mix(self):
        """
        Initialize the gas-mixture properties used by the model.

        The reference gas composition follows the legacy inlet-state rule:
        closed-inlet `blow`, `heat`, `w_purge`, `w_des`, and `cool` steps use the gas
        already present in the bed (`y_initial`, and `c_tot_initial` for
        density), while open-inlet steps use the feed state (`y_feed`,
        `c_tot_feed`). The individual property initializers apply this rule
        consistently.
        """
        print("\nInitializing gas mixture properties...")

        self.init_rho_mix()
        self.init_visc_mix()
        self.init_cp_mix()
        self.init_lambda_mix()

    def init_rho_mix(self):
        """
        Calculate and initialize the gas-mixture density. The ideal gas law is used.
        """
        # Calculate the density [kg/m³]
        y_temp = self._gas_property_composition()
        if self.gas_property_composition_source == "initial":
            c_temp = self.c_tot_initial * y_temp
        else:
            c_temp = self.c_tot_feed * y_temp

        molar_mass = self.MW.reshape((1,) * (c_temp.ndim - 1) + (-1,))
        
        self.rho_g_mix = np.sum(c_temp * molar_mass, axis=-1).item()

    def init_visc_mix(self):
        """
        Calculate and initialize the gas-mixture viscosity.

        This is done using the Wilke mixture rule based on Chapman-Enskog theory:
            Wilke, C. R. (1950). A Viscosity Equation for Gas Mixtures. The Journal of Chemical Physics, 18(4), 517-519. https://doi.org/10.1063/1.1747673
        """
        y_temp = self._gas_property_composition()

        # Chapman-Enskog pure-component viscosities [Pa·s]: M in g/mol, sigma in Angstrom
        self.mu_i = (
            2.6693e-6
            * np.sqrt(self.MW * 1e3 * self.T_feed)
            / (
                self.lennard_jones_sigma_A**2
                * self._collision_integral(
                    self.lennard_jones_epsilon_over_k_K,
                    self.T_feed,
                )
            )
        )

        # Wilke's method for mixture viscosity
        Mj = self.MW[None, :]
        Mi_col = self.MW[:, None]
        phi_ij = (1.0 / np.sqrt(8.0)) * (1.0 + Mi_col / Mj) ** (-0.5) * (
            1.0 + np.sqrt(self.mu_i[:, None] / self.mu_i[None, :]) * (Mj / Mi_col) ** 0.25
        ) ** 2

        y_vec = np.asarray(y_temp, dtype=float).reshape(-1)
        if y_vec.shape[0] != self.MW.shape[0]:
            raise ValueError("y_temp length must match number of species in MW.")

        denom = y_vec @ phi_ij.T
        self.mu_g_mix = np.sum(y_vec * self.mu_i / denom)

    def _collision_integral(self, eps_k, T):
        """Evaluate the Neufeld collision-integral correlation."""
        coefficients = {
            "A": 1.16145,
            "B": 0.14874,
            "C": 0.52487,
            "D": 0.77320,
            "E": 2.16178,
            "F": 2.43787,
        }

        Tstar = T / eps_k
        return (
            coefficients["A"] / (Tstar ** coefficients["B"])
            + coefficients["C"] / np.exp(coefficients["D"] * Tstar)
            + coefficients["E"] / np.exp(coefficients["F"] * Tstar)
        )

    def init_cp_mix(self):
        """
        Calculate and initialize the gas heat capacities.

        The per-species molar heat capacities are obtained from the Shomate
        correlation evaluated at the step property temperature `T_feed`, and are
        the single source of gas `cp` for the whole model: they feed the thermal
        closures through `cp_g_mix` (Prandtl number) and the energy balance
        itself through `acc_g_*`, `acc_ads_*` and the gas enthalpy. They are
        therefore constant during the integration of a step - as the energy
        balance derivation requires, since it assumes an enthalpy linear in T -
        and are refreshed from one step to the next, consistently with the solid
        heat capacity `cp_p`.
        """
        y_temp = self._gas_property_composition()

        self.cp_g = np.array(
            [self._shomate_cp(species=species, T=self.T_feed) for species in self.gas_species]
        )
        for species in ("CO2", "H2O", "O2", "N2"):
            if species in self.gas_index:
                setattr(self, f"cp_g_{species}", self.cp_g[self.gas_index[species]])

        # Calculate the average specific heat capacity [J/mol/K] using mole fractions
        self.cp_g_mix = np.dot(y_temp, self.cp_g).item()

    def _shomate_cp(self, species=None, T=298.15):
        """Calculate gas heat capacity with the Shomate correlation."""
        if species is None:
            raise ValueError("Species must be provided to calculate heat capacity.")
        return get_species_properties(species).heat_capacity_shomate.evaluate_J_per_molK(T)

    def init_lambda_mix(self):
        """
        Calculate and initialize the gas-mixture thermal conductivity.

        """
        y_temp = self._gas_property_composition()

        # Calculate the average thermal conductivity [W/m/K] using mole fractions
        self.lambda_g_mix = np.dot(
            y_temp,
            self.species_conductivity_W_mK,
        ).item()

    def init_solid_thermo(self):
        """
        Initialize the thermodynamic properties of the solid phase.

        The selected material profile supplies either a constant heat capacity
        or the established Low et al. (2023) temperature correlation.
        """
        self.cp_p = resolve_solid_heat_capacity(self.solid_heat_capacity, self.T_feed)

    def calc_Diff_coeffs(self):
        """
        Initialize particle diffusion by calculating Knudsen and molecular diffusion coefficients.
        Molecular diffusion coefficients are calculated using the Fuller method for binary mixtures:
            Fuller, E. N., Schettler, P. D., & Giddings, J. C. (1966). A new method for prediction of binary gas-phase diffusion coefficients. 
            Industrial and Engineering Chemistry, 58(5), 18-27. https://doi.org/10.1021/IE50677A007/ASSET/IE50677A007.FP.PNG_V03.
        The Wilke-like approximation is used to obtain the mixture diffusion coefficient of component i:
            Fairbanks, D. F., & Wilke, C. R. (1950). Diffusion Coefficients in Multicomponent Gas Mixtures.
            Industrial & Engineering Chemistry, 42(3), 471-475. https://doi.org/10.1021/IE50483A022
        """
        print("\nInitializing particle diffusion...")

        # Build the symmetric binary diffusion matrix from Fuller correlations.
        D_bin = np.full((self.Nc_gas, self.Nc_gas), np.inf)
        for i in range(self.Nc_gas):
            for j in range(i + 1, self.Nc_gas):
                D_ij = self._Fuller_binary_diff(
                    [self.MW[i], self.MW[j]],
                    [
                        self.fuller_diffusion_volume_cm3_per_mol[i],
                        self.fuller_diffusion_volume_cm3_per_mol[j],
                    ],
                    self.T_feed,
                    self.p_feed,
                ) / 1e4
                D_bin[i, j] = D_ij
                D_bin[j, i] = D_ij
        self.D_binary = D_bin

        # Mixture diffusion coefficients from the binary matrix and the step-dependent bulk composition.
        if self.step in ("blow", "heat", "w_purge", "s_purge", "s_des", "w_des", "cool"):
            # Pure-steam s_purge and s_des feed makes the
            # multicomponent Wilke diffusivity expression singular, so its
            # dispersion mixture is evaluated from the initial-bed gas.
            self.dispersion_composition_source = "initial"
            y_temp = self.y_initial
        else:
            self.dispersion_composition_source = "feed"
            y_temp = self.y_feed

        y_vec = np.asarray(y_temp, dtype=float).reshape(-1)
        if y_vec.shape[0] != self.Nc_gas:
            raise ValueError("Diffusion composition must contain one mole fraction per gas species.")

        inv_D_bin = np.zeros_like(D_bin)
        off_diag = ~np.eye(self.Nc_gas, dtype=bool)
        inv_D_bin[off_diag] = 1.0 / D_bin[off_diag]

        denom = inv_D_bin @ y_vec
        self.D_m = (1.0 - y_vec) / denom

    def _Fuller_binary_diff(self, MW, v, T, p):
        """Compute a binary molecular diffusion coefficient with Fuller."""
        # Molar masses [g/mol]
        MW_A = MW[0] * 1000
        MW_B = MW[1] * 1000
        MW_combined = 1 / MW_A + 1 / MW_B

        # Diffusion volumes [cm³/mol]
        v_A, v_B = v
        return 0.001 * T**1.75 * np.sqrt(MW_combined) / (
            p / 101325 * (v_A ** (1 / 3) + v_B ** (1 / 3)) ** 2
        )

    def calc_adim_numbs(self):
        """
        Calculate adimensional numbers (Schmidt, Reynolds, and Sherwood numbers).
        """
        print("\nInitializing dimensionless numbers...")
        y_temp = self._gas_property_composition()

        self.Sc = self.mu_g_mix / self.D_m / self.rho_g_mix + 1e-32
        self.Pr = (self.mu_g_mix * (self.cp_g_mix / np.sum(y_temp * self.MW, axis=-1)) / self.lambda_g_mix).item() + 1e-32
        self.Re = (self.rho_g_mix * self.u_inlet * self.d_p / self.mu_g_mix).item() + 1e-32
        self.Sh = 2 + 1.1 * self.Re**0.6 * self.Sc**(1 / 3)
                                    
    # 4. Transport closures and non-dimensional physical coefficients

    def calculate_system_parameters(self):
        """
        Compute all necessary system parameters required.

        After execution, the flag `parameters_computed` is set to `True`, indicating that 
        the system parameters have been successfully initialized.
        """

        # Calculate effective diffusion in the particle
        self.calc_Diff_coeffs()

        # Calculate adimensional numbers
        self.calc_adim_numbs()

        # Resolve every closure model
        self._resolve_bed_pressure_drop()
        self._resolve_bed_dispersion()
        self._resolve_bed_thermal_transport()
        self._finalize_resolved_architecture()
        self.calc_jacket_htc()

        # Calculate non dimensional PDE parameters
        self.calc_nondim_pde_params()

        # Conditional flag to indicate that parameters have been computed
        self.parameters_computed = True

    def _resolve_bed_pressure_drop(self):
        """Resolve and bind the selected architecture's pressure-drop law."""
        return self._resolve_packed_bed_pressure_drop()

    def _resolve_packed_bed_pressure_drop(self):
        """Resolve and bind the detached packed-bed pressure-drop law.

        A channeling flow model binds the radial Ergun law on the resolved DRAG
        voidage field.  That field is the exact area-overlap projection of the
        fixed physical drag layers, never the ring-mean storage voidage: the
        storage voidage reaches the correlation's wall value, where both Ergun
        coefficients vanish and the resistance has no continuum limit.

        Only the resistance becomes radially structured.  The pressure field,
        the local density and the step-level viscosity policy are unchanged,
        with locally isotropic vector drag and radial convection in 2D.
        """
        if self.channeling_structure is not None:
            self.pressure_drop_law = resolve_packed_bed_radial_pressure_drop(
                eps_drag=self.channeling_structure.eps_drag,
                viscosity=float(np.asarray(self.mu_g_mix).item()),
                particle_diameter=self.d_p,
                bed_length=self.L_b,
                reference_velocity=self.u_ref,
                reference_pressure=self.p_ref,
            )
            self._radial_drag_coefficients = self.pressure_drop_law.coefficients_on_radial_faces(
                self.r_f, self.r_c)
            return self.pressure_drop_law

        self.pressure_drop_law = resolve_packed_bed_pressure_drop(
            eps_flow=self.eps_flow,
            viscosity=float(np.asarray(self.mu_g_mix).item()),
            particle_diameter=self.d_p,
            bed_length=self.L_b,
            reference_velocity=self.u_ref,
            reference_pressure=self.p_ref,
        )
        return self.pressure_drop_law

    def _resolve_bed_dispersion(self):
        """Resolve material dispersion for the selected bed architecture."""
        return self._resolve_packed_bed_dispersion()

    def _resolve_packed_bed_dispersion(self):
        """Resolve the packed-bed material dispersion and Pe_m groups."""
        resolved = resolve_packed_bed_dispersion(
            eps_flow=self.eps_flow,
            molecular_diffusivity=self.D_m,
            reynolds=self.Re,
            schmidt=self.Sc,
            superficial_inlet_velocity=self.u_inlet,
            particle_diameter=self.d_p,
            reference_velocity=self.u_ref,
            bed_length=self.L_b,
            bed_radius=self.R_b,
        )
        self.resolved_dispersion = resolved
        self.dispersion_model = resolved["model"]
        self.Disp_ax = resolved["Disp_ax"]
        self.Disp_rad = resolved["Disp_rad"]
        self.inv_Pe_m_ax = resolved["inv_Pe_m_ax"]
        self.inv_Pe_m_rad = resolved["inv_Pe_m_rad"]
        return resolved

    def _resolve_bed_thermal_transport(self):
        """Resolve thermal transport for the selected bed architecture."""
        return self._resolve_packed_bed_thermal_transport()

    def _resolve_packed_bed_thermal_transport(self):
        """Resolve and expose the detached packed-bed thermal contract."""
        resolved = resolve_packed_bed_thermal_transport(
            stagnant_conductivity_model=self.bed_design.stagnant_conductivity_model,
            wall_htc_model=self.bed_design.wall_htc_model,
            eps_flow=self.eps_flow,
            gas_conductivity=self.lambda_g_mix,
            particle_conductivity=self.lambda_p,
            reynolds=self.Re,
            prandtl=self.Pr,
            particle_diameter=self.d_p,
            bed_diameter=self.d_b,
            reference_velocity=self.u_ref,
            bed_length=self.L_b,
            is_1d=self.is_1d,
        )
        self.resolved_transport = resolved

        diagnostics = resolved.diagnostics
        self.lambda_0 = diagnostics["lambda_0"]
        self.lambda_rad_conv = diagnostics["lambda_rad_conv"]
        self.lambda_ax = resolved.lambda_ax
        self.lambda_rad = resolved.lambda_rad
        self.inv_Pe_h_ax = diagnostics["inv_Pe_h_ax"]
        self.inv_Pe_h_rad = self.lambda_rad / self.u_ref / self.R_b

        self.Nu_w0 = diagnostics["Nu_w0"]
        self.Nu_m = diagnostics["Nu_m"]
        self.Nu_w_star = diagnostics["Nu_w_star"]
        self.Nu_w_convection = diagnostics["Nu_w_convection"]
        self.Nu_w = diagnostics["Nu_w"]
        self.h_w = resolved.h_wall

        self.inv_Bi_h = diagnostics["inv_Bi_h"]
        self.U_w = diagnostics["U_dixon"]
        U_w_override = self.U_w_override
        if self.is_1d and U_w_override is not None:
            self.U_w = np.asarray(float(U_w_override))
        # In 2D there is no lumped coefficient: the bed-side h_w is what couples
        # the bed to the wall, so selecting a different wall-HTC correlation means
        # replacing h_w itself. The resolved correlation value stays in
        # `resolved_transport` for the record.
        h_w_override = self.h_bed_wall_override
        if not self.is_1d and h_w_override is not None:
            self.h_w = np.asarray(float(h_w_override))
            # Match the bed Robin coefficient to the wall-side override.
            self.inv_Bi_h = self.lambda_rad / (self.R_b * self.h_w)
        self.h_int = self.U_w if self.is_1d else self.h_w

        print("\nInitializing packed-bed thermal transport...")
        print(
            "Stagnant effective conductivity "
            f"({self.bed_design.stagnant_conductivity_model}): "
            f"{self.lambda_0:.4f} W/m/K"
        )
        print(f"Axial effective conductivity: {self.lambda_ax:.2f} W/m/K")
        print(f"Radial effective conductivity: {self.lambda_rad:.4f} W/m/K")
        print(f"Nusselt wall number: {self.Nu_w:.2f}")
        print(
            "Internal wall heat transfer coefficient h_w "
            f"({self.bed_design.wall_htc_model}): "
            f"{np.asarray(self.h_w).item():.2f} W/m2/K"
        )
        if not self.is_1d and h_w_override is not None:
            print(
                "OVERRIDE: replacing the resolved bed-side wall coefficient in 2D mode "
                f"with h_w = {np.asarray(self.h_w).item():.2f} W/m2/K (user-supplied constant)"
            )
        if self.is_1d:
            if U_w_override is not None:
                print(
                    "OVERRIDE: replacing the Dixon lumped coefficient in 1D mode with "
                    f"U_w = {np.asarray(self.U_w).item():.2f} W/m2/K (user-supplied constant)"
                )
            else:
                print(
                    "Using Dixon lumped overall coefficient in 1D mode: "
                    f"U_w = {np.asarray(self.U_w).item():.2f} W/m2/K"
                )

        return resolved

    def _finalize_resolved_architecture(self):
        """Expose one JSON-safe, inspectable architecture context."""
        self.resolved_architecture = ResolvedArchitecture(
            architecture=self.bed_architecture,
            design=self.bed_design,
            structure=self.resolved_structure,
            transport=self.resolved_transport,
        )
        context = self.resolved_architecture.to_context_dict()
        heat_transfer_model = "dixon_1996"
        context["resolved"]["models"] = {
            "heat_transfer": heat_transfer_model,
            "pressure_drop": self.pressure_drop_law.name,
            "dispersion": self.dispersion_model,
        }
        if self.bed_architecture == "packed_bed":
            context["resolved"]["models"]["stagnant_conductivity"] = (
                self.bed_design.stagnant_conductivity_model
            )
            context["resolved"]["models"]["wall_htc"] = self.bed_design.wall_htc_model
            context["resolved"]["models"]["flow"] = self.flow_model
            context["resolved"]["flow"] = {
                "model": self.flow_model,
                "compatibility_status": self.flow_model_compatibility,
                # Filled in by `init_inlet_fluxes`, which resolves the inlet
                # policy later in this same initialization.
                "inlet_allocation_source": None,
                # Every empirical transport closure below stays on the declared
                # bed mean and the step-bulk Reynolds number, including in the
                # channeling modes.  That is an explicit effective-bed
                # approximation, not a claim that local wall transport is wrong.
                "bulk_transport_policy": "mean_eps_step_bulk",
                "local_structure": (
                    "storage_and_drag"
                    if self.channeling_structure is not None
                    else "none"
                ),
            }
            if self.channeling_structure is not None:
                context["resolved"]["flow"]["channeling"] = (
                    self.channeling_structure.to_context_dict()
                )
        context["resolved"]["effective_wall_bed_htc"] = float(
            np.asarray(self.h_int).item()
        )
        context["resolved"]["step_property_context"] = {
            "temperature": float(self.T_feed),
            "pressure": float(self.p_feed),
            "superficial_inlet_velocity": float(self.u_inlet),
            "gas_composition_source": self.gas_property_composition_source,
            "dispersion_composition_source": self.dispersion_composition_source,
        }
        self.bed_architecture_context = context
        return context

    def calc_jacket_htc(self):
        """
        Calculate the external heat transfer coefficient.
        """
        print("\nInitializing external wall heat transfer coefficients...")

        if self.step in ("heat", "w_purge", "w_des", "s_des"):
            medium = "hot-oil"
        else:
            medium = "cooling-water"
        self.h_ext = self.htc_ext
        print(f'External wall heat transfer coefficient ({medium}): {self.h_ext:.2f} W/m2/K')

    def calc_nondim_pde_params(self):
        """
        Calculate non-dimensional parameters for the system.
        """
        print("\nInitializing non-dimensional PDE parameters...")

        # Adsorption scales in declared adsorbate order.
        self.phi_adsorbate = (
            self.rho_sorbent * self.q_ref / self.c_tot_ref
        )
        self.psi_adsorbate = np.array(
            [
                self.rho_sorbent
                * self.q_ref_by_species[species]
                / self.c_ref_by_species[species]
                for species in self.adsorbates
            ]
        )

        # Gas concentration and energy scales in declared mixture order.
        self.c_tot_i_ratio = self.c_tot_ref / self.c_ref
        self.acc_g = self.eps_total * self.cp_g * self.c_ref
        closure = self.gas_index[self.closure_species]
        independent = self.independent_gas_indices
        closure_reference = self.c_ref[closure]
        self._acc_g_ind = self.acc_g[independent]
        self._acc_g_closure = self.acc_g[closure]
        self._c_ref_ind_over_closure = (
            self.c_ref[independent] / closure_reference
        )
        self._c_tot_ref_over_closure = self.c_tot_ref / closure_reference
        self.acc_sorbent = self.rho_sorbent * self.cp_p
        self.acc_fixed_adsorbates = self.rho_sorbent * sum(
            properties["loading_mol_per_kg"]
            * properties["heat_capacity_J_molK"]
            for properties in self.fixed_adsorbates.values()
        )
        self.acc_ads = np.array(
            [
                self.rho_sorbent
                * self.cp_g[self.gas_index[species]]
                * self.q_ref_by_species[species]
                for species in self.adsorbates
            ]
        )
        self.omega_p = self.eps_total * self.c_tot_ref * R_gas
        self.Omega_adsorbate = np.array(
            [
                -self.rho_sorbent
                * self.heat_of_adsorption_J_mol[species]
                * self.q_ref_by_species[species]
                / self.T_ref
                for species in self.adsorbates
            ]
        )

        self.omega_wall = (4 / self.d_b) * self.tau_hyd * self.h_int

        # Calculate parameters: wall temperature PDE
        self.T_jacket_nd = self.T_jacket / self.T_ref
        C_w = self.rho_w * self.cp_w
        wall_section = self.R_b_ext**2 - self.R_b**2
        self.alpha_in = self.tau_hyd * (2 * self.R_b * float(np.asarray(self.h_int).item())) / wall_section / C_w
        self.alpha_out = self.tau_hyd * (2 * self.R_b_ext * self.h_ext) / wall_section / C_w
        self.k_wall = (self.lambda_w / self.L_b**2 / C_w) * self.tau_hyd

        # 2D mode: second-order face-reconstruction stencil for the bed temperature
        # at r = R_b, identical to the one pymrm applies to the radial Robin BC.
        if not self.is_1d:
            x_F, x_N, x_M = self.r_f[-1], self.r_c[-1], self.r_c[-2]
            self.wall_face_c_M = (x_F - x_N) / ((x_M - x_N) * (x_M - x_F))
            self.wall_face_c_N = (x_F - x_M) / ((x_N - x_M) * (x_N - x_F))
            self.wall_face_c_F = (2 * x_F - x_M - x_N) / ((x_F - x_M) * (x_F - x_N))

    # 5. Discretization operators, boundary conditions, and Jacobian

    def set_disp(self):
        """
        Set the material-dispersion coefficients for axial and radial directions.

        This method initializes the axial and radial dispersion and conduction coefficients 
        for all the species in the system. The coefficients are stored in 3D arrays 
        with dimensions corresponding to (axial grid points, radial grid points, species).
        ``inv_Pe_m_ax`` and ``inv_Pe_m_rad`` are the inverse material Péclet
        groups resolved by the selected bed architecture.
        """
        if not self.parameters_computed:
            raise RuntimeError("Error: calculate_pde_parameters() must be called before set_disp()!")

        print("\nSetting dispersion and conduction coefficients...")
        ratios = np.asarray(self.c_tot_i_ratio, dtype=float)
        self.D_ax = (ratios * np.asarray(self.inv_Pe_m_ax, dtype=float)).reshape(1, 1, -1)
        self.D_rad = (self.L_R_ratio * ratios * np.asarray(self.inv_Pe_m_rad, dtype=float)).reshape(1, 1, -1)

    def set_cond(self):
        """
        Set the conduction coefficients for axial and radial directions.

        This method initializes the axial and radial conduction coefficients for the temperature
        equation in the system. The coefficients are stored in 3D arrays with dimensions 
        corresponding to (axial grid points, radial grid points, 1).
        """
        if not self.parameters_computed:
            raise RuntimeError("Error: calculate_pde_parameters() must be called before set_cond()!")

        print("\nSetting conduction coefficients...")
        self.k_cond_ax = np.asarray(self.inv_Pe_h_ax, dtype=float).reshape(1, 1, -1)
        self.k_cond_rad = self.L_R_ratio * np.asarray(self.inv_Pe_h_rad, dtype=float).reshape(1, 1, -1)
        
    def set_bc(self):
        """
        Initialize boundary-condition dictionaries for species, energy, wall,
        and pressure equations.

        Each boundary condition uses the generalized linear form
        `a * (du/dn) + b * u  = d`, represented as a dictionary with keys:
        `a`, `b`, and `d`. For each equation, a tuple is stored as
        `(inlet_boundary, outlet_boundary)` in axial direction, or
        `(centerline, wall)` in radial direction.

        Notes on implementation:
        - Species/energy inlet in axial direction is intentionally left in
          flux form (`a=0`, `b=0`, `d=0`) because inlet transport is imposed
          later through a Danckwerts-type flux condition using the velocity
          field and `flux_in_*` terms.
        - In 2D mode, the radial energy wall condition uses `inv_Bi_h` and
          `T_jacket_nd` to enforce convective exchange with the jacket.
        - In 1D mode, radial bed-temperature BCs are disabled because wall
          coupling is moved into an explicit source term in the bed energy
          equation.
        - Wall-temperature axial BCs (`bc_ax_Tw`) are zero-gradient at both
          ends.
        - Pressure BCs are step-dependent:
          `blow`, `heat`, `w_purge`, `w_des` -> closed inlet / outlet pressure relation;
          `press` -> inlet valve to the feed pressure, closed outlet;
          otherwise -> open inlet/outlet (pressure-driven).
        """
        self.bc_ax = ({'a': 0, 'b': 0, 'd': 0 }, {'a': 1, 'b': 0, 'd': 0})
        self.bc_rad = ({'a': 1, 'b': 0, 'd': 0 }, {'a': 1, 'b': 0, 'd': 0})
        
        self.bc_ax_T = ({'a': 0, 'b': 0, 'd': 0}, {'a': 1, 'b': 0, 'd': 0})
        if self.is_1d:
            self.bc_rad_T = ({'a': 0, 'b': 0, 'd': 0}, {'a': 0, 'b': 0, 'd': 0})
        else:
            self.bc_rad_T = ({'a': 1, 'b': 0, 'd': 0}, {'a': self.inv_Bi_h, 'b': 1, 'd': self.T_jacket_nd})
        self.bc_ax_Tw = ({'a': 1, 'b': 0, 'd': 0}, {'a': 1, 'b': 0, 'd': 0})

        if self.step == "cool":
            # Sealed bed: dp/dz = 0 at both inlet and outlet (no flow)
            self.bc_ax_p = ({'a': 1, 'b': 0, 'd': 0}, {'a': 1, 'b': 0, 'd': 0})

        elif self.step in ("blow", "heat", "w_purge", "w_des"):
            # Vacuum-connected steps: closed inlet, open outlet with pressure relation
            self.bc_ax_p = ({'a': 1, 'b': 0, 'd': 0}, {'a': self.bd_fact, 'b': 1, 'd': self.p_out/self.p_ref})

        elif self.step == "press":
            # Pressurization step: inlet valve to the feed pressure, closed outlet
            self.bc_ax_p = ({'a': self.ps_fact, 'b': 1, 'd': self.p_feed/self.p_ref}, {'a': 1, 'b': 0, 'd': 0})
            
        else:
            # Open inlet/outlet for general steps
            self.bc_ax_p = ({'a': 0, 'b': 0, 'd': 0}, {'a': 0, 'b': 1, 'd': self.p_out/self.p_ref})
            
        self.bc_rad_p = ({'a': 1, 'b': 0, 'd': 0}, {'a': 1, 'b': 0, 'd': 0})

    def init_Jac(self):
        """
        Initialize all fixed spatial operators used in the semi-discrete PDE system.

        The method assembles, once per simulation setup, the sparse gradient,
        divergence, and diffusive-flux operators for:
        - Gas species transport in axial/radial directions (`shape_ci`),
        - Bed temperature conduction in axial/radial directions (`shape_single`),
        - Total-pressure gradients/divergences (`shape_single`).

        Boundary-condition contributions produced by `construct_grad` are also
        converted to dense vectors (`*_bc_dense`) so they can be added directly
        in RHS evaluations without rebuilding operators every call.

        Stored operator groups:
        - Gas-species concentration transport operators and BC vectors.
        - Bed-temperature conduction operators and BC vectors.
        - Total-pressure gradient/divergence operators and BC vectors.
        - Convective upwind cache objects for repeated RHS assembly.
        """

        # Gas species transport PDE
        self.Grad_ax_c, grad_bc_ax_c = construct_grad(
            self.shape_ci, self.z_f, self.z_c, self.bc_ax, axis=0
        )
        self.Grad_rad_c, grad_bc_rad_c = construct_grad(
            self.shape_ci, self.r_f, self.r_c, self.bc_rad, axis=1
        )
        self.grad_ax_c_bc_dense = grad_bc_ax_c.toarray().ravel()
        self.grad_rad_c_bc_dense = grad_bc_rad_c.toarray().ravel()

        independent = self.independent_gas_indices
        closure_index = self.gas_index[self.closure_species]
        self.Diff_mat_ax = construct_coefficient_matrix(
            self.D_ax[..., independent], shape=self.shape_ci, axis=0
        )
        self.Diff_mat_rad = construct_coefficient_matrix(
            self.D_rad[..., independent], shape=self.shape_ci, axis=1
        )
        self.Div_ax_ci = construct_div(self.shape_ci, self.z_f, nu=0, axis=0)
        self.Div_rad_ci = construct_div(self.shape_ci, self.r_f, nu=1, axis=1)

        self.Disp_flux_ax = - self.Diff_mat_ax @ self.Grad_ax_c
        self.disp_flux_ax_bc = - self.Diff_mat_ax @ grad_bc_ax_c
        self.disp_flux_ax_bc_dense = self.disp_flux_ax_bc.toarray().ravel()
        
        self.Disp_flux_rad = - self.Diff_mat_rad @ self.Grad_rad_c
        self.disp_flux_rad_bc = - self.Diff_mat_rad @ grad_bc_rad_c
        self.disp_flux_rad_bc_dense = self.disp_flux_rad_bc.toarray().ravel()

        # Mixture-averaged correction velocity.  The physical dispersive flux is
        #
        #   J_i = -c_tot D_i grad(y_i)
        #         + c_tot y_i sum_j[D_j grad(y_j)],
        #
        # where the sum contains every gas species.  The closure gradient is
        # -sum(grad(y_independent)), so the sum can be evaluated entirely from
        # the generated independent-species layout.  The concentration-reference
        # ratios convert each flux back to that species' nondimensional state.
        ratios = np.asarray(self.c_tot_i_ratio, dtype=float)
        inv_pe_ax = np.broadcast_to(
            np.asarray(self.inv_Pe_m_ax, dtype=float).reshape(-1),
            (self.Nc_gas,),
        )
        inv_pe_rad = np.broadcast_to(
            np.asarray(self.inv_Pe_m_rad, dtype=float).reshape(-1),
            (self.Nc_gas,),
        )
        self._dispersion_ratio_ind = ratios[independent].reshape(1, 1, -1)
        self._dispersion_inv_pe_ax_ind = inv_pe_ax[independent].reshape(1, 1, -1)
        self._dispersion_inv_pe_rad_ind = (
            self.L_R_ratio * inv_pe_rad[independent]
        ).reshape(1, 1, -1)
        self._dispersion_inv_pe_ax_delta = (
            inv_pe_ax[independent] - inv_pe_ax[closure_index]
        ).reshape(1, 1, -1)
        self._dispersion_inv_pe_rad_delta = (
            self.L_R_ratio
            * (inv_pe_rad[independent] - inv_pe_rad[closure_index])
        ).reshape(1, 1, -1)

        # Temperature PDE
        Grad_ax_T, grad_ax_T_bc = construct_grad(self.shape_single, self.z_f, self.z_c, self.bc_ax_T, axis=0)    
        Grad_rad_T, grad_rad_T_bc = construct_grad(self.shape_single, self.r_f, self.r_c, self.bc_rad_T, axis=1)    
        Diff_mat_ax_T = construct_coefficient_matrix(self.k_cond_ax, shape=self.shape_single, axis=0)
        self.Diff_mat_rad_T = construct_coefficient_matrix(self.k_cond_rad, shape=self.shape_single, axis=1)
        self.Div_ax_T = construct_div(self.shape_single, self.z_f, nu=0, axis=0)
        self.Div_rad_T = construct_div(self.shape_single, self.r_f, nu=1, axis=1)
        
        self.Cond_flux_ax = - Diff_mat_ax_T @ Grad_ax_T
        self.cond_flux_ax_bc  = - Diff_mat_ax_T @ grad_ax_T_bc
        self.cond_flux_ax_bc_dense = self.cond_flux_ax_bc.toarray().ravel()
        
        self.Cond_flux_rad = - self.Diff_mat_rad_T @ Grad_rad_T
        self.cond_flux_rad_bc  = - self.Diff_mat_rad_T @ grad_rad_T_bc
        self.cond_flux_rad_bc_dense = self.cond_flux_rad_bc.toarray().ravel()
        
        # Wall temperature PDE
        Grad_ax_Tw, grad_ax_Tw_bc = construct_grad(self.shape_Tw, self.z_f, self.z_c, self.bc_ax_Tw, axis=0)  
        Diff_mat_ax_Tw = construct_coefficient_matrix(self.k_wall, shape=self.shape_Tw, axis=0)
        self.Div_ax_Tw = construct_div(self.shape_Tw, self.z_f, nu=0, axis=0)
        self.Cond_flux_ax_Tw = - Diff_mat_ax_Tw @ Grad_ax_Tw
        self.cond_flux_ax_Tw_bc  = - Diff_mat_ax_Tw @ grad_ax_Tw_bc
        self.cond_flux_ax_Tw_bc_dense = self.cond_flux_ax_Tw_bc.toarray().ravel()

        # Total concentration PDE
        self.Grad_ax_ctot, grad_ax_ctot_bc = construct_grad(self.shape_single, self.z_f, self.z_c, self.bc_ax, axis=0)
        self.grad_ax_ctot_bc_dense = grad_ax_ctot_bc.toarray().ravel()
        self.Grad_rad_ctot, grad_rad_ctot_bc = construct_grad(self.shape_single, self.r_f, self.r_c, self.bc_rad, axis=1)
        self.grad_rad_ctot_bc_dense = grad_rad_ctot_bc.toarray().ravel()
        self.Div_ax_ctot = construct_div(self.shape_single, self.z_f, nu=0, axis=0)
        self.Div_rad_ctot = construct_div(self.shape_single, self.r_f, nu=1, axis=1)

        # Total pressure PDE
        self.Grad_ax_p, self.grad_ax_p_bc = construct_grad(self.shape_single, self.z_f, self.z_c, self.bc_ax_p, axis=0)
        self.grad_ax_p_bc_dense = self.grad_ax_p_bc.toarray().ravel()
        self.Grad_rad_p, self.grad_rad_p_bc = construct_grad(self.shape_single, self.r_f, self.r_c, self.bc_rad_p, axis=1)
        self.grad_rad_p_bc_dense = self.grad_rad_p_bc.toarray().ravel()
        self.Div_ax_p = construct_div(self.shape_single, self.z_f, nu=0, axis=0)
        self.Div_rad_p = construct_div(self.shape_single, self.r_f, nu=1, axis=1)

        # Convection caches (reused at each RHS evaluation)
        self.conv_ax_species_cache = ConvfluxUpwindCache(
            self.shape_ci, self.z_f, self.z_c, bc_inlet=None, bc_outlet=self.bc_ax[1], axis=0
        )
        
        self.conv_ax_cache = ConvfluxUpwindCache(
            self.shape_single, self.z_f, self.z_c, bc_inlet=None, bc_outlet=self.bc_ax[1], axis=0
        )

        # Radial convection caches.  Both radial boundary faces are closed
        # (symmetry at the centerline, impermeable wall), so the velocity there
        # is zero and no boundary stencil is required: the internal upwind
        # operator carries the whole flux.
        self.conv_rad_species_cache = ConvfluxUpwindCache(
            self.shape_ci, self.r_f, self.r_c, bc_inlet=None, bc_outlet=None, axis=1
        )

        self.conv_rad_cache = ConvfluxUpwindCache(
            self.shape_single, self.r_f, self.r_c, bc_inlet=None, bc_outlet=None, axis=1
        )
        
        # Temperature gradient caches for BCs (reused at each RHS evaluation)
        self.grad_rad_T_cache = GradCache(self.shape_single, self.r_f, self.r_c, axis=1)

        # Fused radial conduction: Div_rad_T @ Diff_mat_rad_T
        self.Div_Diff_rad_T = self.Div_rad_T @ self.Diff_mat_rad_T

        # These fixed operators are only used as ``operator.dot(vector)`` in
        # the RHS. Convert them once after assembly instead of paying CSC's
        # column-oriented matvec cost on every solver evaluation.
        if _RHS_CSR_OPERATORS:
            for operator_name in (
                "Div_ax_ctot",
                "Div_ax_ci",
                "Div_rad_ci",
                "Div_ax_T",
                "Div_ax_Tw",
                "Disp_flux_ax",
                "Disp_flux_rad",
                "Grad_ax_c",
                "Grad_rad_c",
                "Cond_flux_ax",
                "Cond_flux_ax_Tw",
                "Grad_ax_p",
                "Div_Diff_rad_T",
            ):
                operator = getattr(self, operator_name)
                if issparse(operator) and operator.format != "csr":
                    setattr(self, operator_name, csr_array(operator))

        # Quadrature weights for the jacket energy KPI. These MUST be the wall
        # PDE's own finite-volume weights, because `alpha_out` applies the jacket
        # loss per finite-volume cell: any other quadrature makes the reported
        # `E_jacket` differ from the energy the model actually removes.
        #
        # A Simpson rule over the cell centres leaves a stable gap between the two.
        # The cause is not the span correction: both weight sets sum to the exact
        # lateral area 2 pi R_b_ext L_b to machine precision. It is that Simpson's
        # alternating 4/3, 2/3 pattern differs from the uniform finite-volume
        # weight CELL BY CELL, by up to 66 % of the mean weight at Nz = 50..200.
        # The two integrals therefore agree on a constant wall temperature and
        # diverge as soon as T_wall varies along z, which is every real run.
        self._jacket_quad_weights = (
            np.diff(self.z_f) * self.L_b * 2 * np.pi * self.R_b_ext
        )
        self._jacket_quad_weight_sum = float(np.sum(self._jacket_quad_weights))

        # Reused geometry/energy factors for KPI helper calls.
        self._area_rings = self.A_rings.copy()
        self._area_rings_sum = float(np.sum(self._area_rings))
        self.outlet_face_weights = self._axial_outlet_face_weights()
        self._outlet_face_inner_index = -2 if self.Nz > 1 else -1
        gamma = self.cp_g_mix / (self.cp_g_mix - R_gas)
        self._energy_gamma_ratio = gamma / (gamma - 1.0)
        self._energy_poly_exponent = (gamma - 1.0) / gamma
        self._compression_prefactor = self._energy_gamma_ratio / 0.7
        self._vacuum_eta_pressure_scale = 19.55 / 1e5
        self._vacuum_eta_prefactor = 0.8 * self._vacuum_eta_pressure_scale
        self._jacket_power_constant = self.h_ext * self.T_jacket * self._jacket_quad_weight_sum

        # Reusable work buffers for RHS assembly.
        self._ci_ordered_buf = np.empty(self.shape_ci, dtype=float)
        self._yi_ordered_buf = np.empty(self.shape_ci, dtype=float)
        self._c_full_nd_buf = np.empty(self.shape_gas, dtype=float)
        self._c_full_dim_buf = np.empty(self.shape_gas, dtype=float)
        self._y_full_buf = np.empty(self.shape_gas, dtype=float)
        self._q_ordered_buf = np.empty(self.shape_q, dtype=float)
        self._dq_ordered_buf = np.empty(self.shape_q, dtype=float)
        n_cells = self.Nz * self.Nr
        self._dc_closure_dtau_buf = np.empty(n_cells, dtype=float)
        self._gas_extra_buf = np.empty(n_cells, dtype=float)
        self._ads_extra_buf = np.empty(n_cells, dtype=float)
        self._source_extra_buf = np.empty(n_cells, dtype=float)

        # Pre-allocate the complete layout-owned RHS output vector.
        self._rhs_buf = np.empty(self.state_layout.state_size)
    
    # 6. Inlet fluxes, pressure drop, and velocity fields

    def init_inlet_fluxes(self):
        """
        Initialize the inlet fluxes and velocity fields.
        Behavior depends on whether the system is mass-flow-driven (self.press = False)
        or pressure-driven (self.press = True).
        """
        print("\nInitializing inlet fluxes ...")

        if self.step != "press":
            print("Mass-flow-driven BC: setting inlet fluxes from feed values.")
            
            # Total concentration flux at inlet
            self.flux_in_ctot = self.c_tot_feed / self.c_tot_ref * self.u_inlet / self.u_ref
            self.flux_in_ctot = np.broadcast_to(np.array(self.flux_in_ctot).reshape((1, 1, -1)), (1, *self.shape_single[1:]))
            
            # Total enthalpy flux at inlet
            theta_feed = (
                self.T_feed - self.T_enthalpy_ref_K
            ) / self.T_ref
            h_g_feed = np.sum(self.c_feed * self.cp_g) * theta_feed
            flux_in_T = h_g_feed * (self.u_inlet / self.u_ref)
            self.flux_in_T = np.broadcast_to(np.array(flux_in_T).reshape((1, 1, -1)), (1, *self.shape_single[1:]))

            # Species flux at inlet
            self.flux_in_ci_full = self.c_feed / self.c_ref * (self.u_inlet / self.u_ref)
            independent_fluxes = np.take(
                self.flux_in_ci_full,
                self.independent_gas_indices,
                axis=-1,
            )
            self.flux_in_ci = np.broadcast_to(
                independent_fluxes.reshape((1, 1, -1)),
                (1, *self.shape_ci[1:]),
            )

            # Radially non-uniform inlet flux (streamtube channelling); the
            # weights have area mean 1, so the total molar flow is unchanged.
            # A channeling bed with a uniform inlet keeps unit weights: its
            # interior drag and radial convection redistribute the feed.
            self.inlet_flux_weights = np.ones(self.Nr)
            self.inlet_flux_source = "uniform"
            if self.channeling_structure is not None:
                if self.bed_design.channeling.inlet_distribution == "distributed":
                    self.inlet_flux_weights = self._channeling_inlet_flux_weights()
                elif float(self.u_inlet) == 0.0:
                    self.inlet_flux_source = "zero_feed"
            elif self.bed_architecture == "packed_bed" and not self.is_1d:
                self.inlet_flux_weights = packed_bed_ring_inlet_flux_weights(
                    self.bed_design, self.r_f
                )
                if self.bed_design.inlet_flux_profile is not None:
                    self.inlet_flux_source = "prescribed_profile"
            ring_weights = self.inlet_flux_weights.reshape((1, self.Nr, 1))
            self.flux_in_ctot = self.flux_in_ctot * ring_weights
            self.flux_in_T = self.flux_in_T * ring_weights
            self.flux_in_ci = self.flux_in_ci * ring_weights

            self.mole_flow_in_i = self.inlet_area * (self.c_feed * self.u_inlet).flatten()
            self.mass_flow_in_i = self.mole_flow_in_i * self.MW

        else:
            print("Pressure-driven BC: setting inlet fluxes to zero.")

            # `press` keeps its upstream-pressure condition: the inlet velocity
            # is recovered from the bed pressure field through the bound law,
            # which already carries the radial resistance.  No generated w(r) is
            # imposed or multiplied on top of it.
            self.inlet_flux_weights = np.ones(self.Nr)
            self.inlet_flux_source = "upstream_pressure"

            self.flux_in_ctot = np.zeros((1, *self.shape_single[1:]))

            self.flux_in_T = np.zeros((1, *self.shape_single[1:]))

            self.flux_in_ci = np.zeros((1, *self.shape_ci[1:]))
            
            self.mole_flow_in_i = np.zeros_like(self.c_feed).flatten()
            self.mass_flow_in_i = np.zeros_like(self.mole_flow_in_i)

        self._record_inlet_allocation_source()

        # Always initialize axial velocity to zero
        u_ax = np.zeros(self.shape_u_ax_single, dtype=float)
        self.u_ax = np.broadcast_to(u_ax, self.shape_u_ax)
        self.u_ax_single = u_ax

        # Same for the radial velocity, so the first RHS call finds it bound.
        u_rad = np.zeros(self.shape_u_rad_single, dtype=float)
        self.u_rad = np.broadcast_to(u_rad, self.shape_u_rad)
        self.u_rad_single = u_rad

    def _record_inlet_allocation_source(self):
        """Publish the resolved inlet policy in the architecture context.

        The context is assembled while the step properties are resolved, which
        is before the inlet policy is known; this fills the placeholder rather
        than letting a reader see a default that was never applied.
        """
        context = getattr(self, "bed_architecture_context", None)
        if not isinstance(context, dict):
            return
        flow = context.get("resolved", {}).get("flow")
        if isinstance(flow, dict):
            flow["inlet_allocation_source"] = self.inlet_flux_source

    def _channeling_inlet_flux_weights(self):
        """Generate the feed allocation from the resolved reduced closure.

        The weights come from the same local Ergun resistance the interior
        uses, under one common axial gradient chosen so the integrated flow
        matches the declared total feed.  They are evaluated once, at the
        declared uniform upstream state, and then frozen for the whole step.

        A zero prescribed feed has no allocation to solve: it keeps unit
        weights and zero fluxes rather than running a normalization on nothing.
        """
        u_inlet = float(self.u_inlet)
        if u_inlet == 0.0:
            print(
                "Channeling inlet allocation skipped: the prescribed total feed "
                "is zero, so every inlet flux is zero."
            )
            self.inlet_flux_source = "zero_feed"
            return np.ones(self.Nr)

        weights = channeling_inlet_flux_weights(
            self.channeling_structure,
            viscosity=float(np.asarray(self.mu_g_mix).item()),
            density=float(np.asarray(self.rho_g_mix).item()),
            particle_diameter=self.d_p,
            mean_velocity=u_inlet,
        )
        self.inlet_flux_source = "reduced_closure"
        print(
            "Channeling inlet allocation from the reduced closure: "
            f"centre {weights[0]:.4f}, wall ring {weights[-1]:.4f} "
            f"(area mean {np.sum(weights * self.A_rings) / np.sum(self.A_rings):.12f})."
        )
        return weights

    def init_radial_structure_factors(self):
        """Per-cell gas-holdup and sorbent factors (area mean 1) from eps(r).

        The scalar ``eps_total`` and ``rho_sorbent`` stay the bed means used by
        normalization; the same factors weight PDE storage and inventories.
        """
        gas = np.ones(self.Nr)
        solid = np.ones(self.Nr)
        if self.channeling_structure is not None:
            # Exact cylindrical cell means of the resolved profile.  The legacy
            # prescribed-profile path keeps its historical midpoint sampling and
            # mean shift below, so old cases stay reproducible.
            gas = self.channeling_structure.gas_holdup_factor
            solid = self.channeling_structure.sorbent_factor
        elif self.bed_architecture == "packed_bed" and not self.is_1d:
            gas, solid = packed_bed_ring_structure_factors(
                self.bed_design, self.particle_properties, self.r_f
            )
        self.ring_gas_holdup_factor = gas
        self.ring_sorbent_factor = solid
        self._gas_cell = np.tile(gas, self.Nz)
        self._solid_cell = np.tile(solid, self.Nz)
        self._gas_2d = self._gas_cell.reshape(self.Nz, self.Nr)
        self._solid_2d = self._solid_cell.reshape(self.Nz, self.Nr)

    def update_inlet_fluxes(self):
        """
        Update inlet fluxes for the pressure-driven pressurization step.

        The inlet flow rate is determined by the current inlet velocity
        recovered from the bed pressure field, while the thermodynamic state of
        the entering gas is fixed by the upstream supply (`self.p_feed`,
        `self.T_feed`, and `self.y_feed`).
        """
        # Axial velocity at inlet (z = 0), pressure-driven (non-dimensional)
        u_inlet = np.maximum(self.u_ax_single[0], 0.0)[np.newaxis, ...]  # shape (1, Nr, 1)
        
        # Supply concentration based on ideal gas law and inlet composition
        p_supply_dim = float(self.p_feed)
        c_tot_feed_dim = np.full((1, self.Nr, 1), p_supply_dim / (R_gas * self.T_feed), dtype=float)
        c_feed = c_tot_feed_dim * self.y_feed
        
        # Compute fluxes based on updated inlet velocity
        self.flux_in_ctot = u_inlet * (c_tot_feed_dim / self.c_tot_ref)
        self.flux_in_ctot = np.array(self.flux_in_ctot).reshape((1, -1, 1))

        h_g_feed = np.sum(
            c_feed * self.cp_g.reshape(1, 1, -1),
            axis=-1,
            keepdims=True,
        ) * (
            (self.T_feed - self.T_enthalpy_ref_K) / self.T_ref
        )
        flux_in_T = h_g_feed * u_inlet
        self.flux_in_T = np.array(flux_in_T).reshape((1, -1, 1))
        
        self.flux_in_ci_full = u_inlet * c_feed / self.c_ref
        self.flux_in_ci = np.take(
            self.flux_in_ci_full,
            self.independent_gas_indices,
            axis=-1,
        )

        # Molar and mass flow rates 
        u_inlet_dim = u_inlet * self.u_ref        
        mole_flux_in_i = (u_inlet_dim * c_feed)
        mole_flow_in_i = np.sum(mole_flux_in_i * self.A_rings[:, None], axis=1)  
        mass_flow_in_i = mole_flow_in_i * self.MW 
        self.mole_flow_in_i = mole_flow_in_i.flatten()
        self.mass_flow_in_i = mass_flow_in_i.flatten()

    def _build_runtime_properties(self):
        """Collect every value shown by the terminal property summary."""
        self.runtime_properties = build_runtime_properties(self)

    def _print_runtime_property_summary(self):
        """Print only values already present in `runtime_properties`."""
        print_runtime_property_summary(self.runtime_properties, self.gas_species)

    def _velocity_from_pressure_law(self, p_tot, rho):
        """Compute dimensionless superficial velocity from the bound law."""
        rho_stagg_ax = interp_cntr_to_stagg(rho.reshape(self.shape_single), self.z_f, self.z_c, axis=0)

        # pressure gradients from your operators
        grad_ax_p = self.Grad_ax_p.dot(p_tot) + self.grad_ax_p_bc_dense

        return self.pressure_drop_law.velocity_from_gradient(
            grad_ax_p.reshape(self.shape_u_ax_single),
            rho_stagg_ax,
        )

    def _vector_velocity_from_pressure_law(self, p_tot, rho):
        """Invert isotropic drag on both face sets using collocated gradients.

        All gradients use p_ref/L_b; the radial normal gradient therefore
        receives L_b/R_b here. The RHS separately scales radial divergence.
        """
        grad_ax = (self.Grad_ax_p.dot(p_tot) + self.grad_ax_p_bc_dense).reshape(self.shape_u_ax_single)
        grad_rad = self.L_R_ratio * (self.Grad_rad_p.dot(p_tot) + self.grad_rad_p_bc_dense).reshape(self.shape_u_rad_single)
        fixed = tuple(float(bc['a']) == 0.0 and float(bc['b']) != 0.0 for bc in self.bc_ax_p)
        transverse_ax, transverse_rad = transverse_gradients(
            grad_ax, grad_rad, self.z_f, self.z_c, self.r_f, self.r_c,
            axial_dirichlet=fixed)
        density = rho.reshape(self.shape_single)
        coefficients = dict(linear_coefficient=self.pressure_drop_law.linear_coefficient,
                            inertial_coefficient_per_density=self.pressure_drop_law.inertial_coefficient_per_density)
        u_ax = normal_velocity(grad_ax, transverse_ax,
                              interp_cntr_to_stagg(density, self.z_f, self.z_c, axis=0), **coefficients)
        radial_coefficients = coefficients
        if self.channeling_structure is not None:
            linear, inertial = self._radial_drag_coefficients
            radial_coefficients = dict(linear_coefficient=linear,
                                       inertial_coefficient_per_density=inertial)
        u_rad = normal_velocity(grad_rad, transverse_rad,
                               interp_cntr_to_stagg(density, self.r_f, self.r_c, axis=1), **radial_coefficients)
        return u_ax, u_rad

    def update_velocity_fields(self, p_tot, rho):
        """
        Compute the axial velocity field based on the total pressure.

        Parameters
        ----------
        p_tot : numpy.ndarray
            Total pressure field.
        rho : numpy.ndarray
            Density field.

        Returns
        -------
        tuple of scipy.sparse.csc_array
            - u_ax: Axial velocity field.
        """
        # Selected pressure-drop law
        if self.is_1d:
            u_ax = self._velocity_from_pressure_law(p_tot, rho)
            u_rad = np.zeros(self.shape_u_rad_single, dtype=float)
        else:
            u_ax, u_rad = self._vector_velocity_from_pressure_law(p_tot, rho)

        # Enforce only the faces that are physically closed; 
        if self.step in ("blow", "heat", "w_purge", "w_des", "cool"):
            u_ax[0, :] = 0.0
        elif self.step != "press":
            u_ax[0, :] = (self.dz1 * u_ax[1, :] - self.dz2 * u_ax[2, :]) / self.dz3

        if self.step in ("press", "cool"):
            u_ax[-1, :] = 0.0
            
        # Symmetry and impermeable wall: no radial boundary flux.
        u_rad[:, 0] = 0.0
        u_rad[:, -1] = 0.0

        # Set the velocity field (read only field)
        self.u_ax = np.broadcast_to(u_ax, self.shape_u_ax)
        self.u_ax_single = u_ax
        self.u_rad = np.broadcast_to(u_rad, self.shape_u_rad)
        self.u_rad_single = u_rad
        
        # Update the velocity field in the inlet fluxes
        if self.step == "press":
            self.update_inlet_fluxes() 

    # 7. Solver state, right-hand side, and dynamic KPI helpers

    def get_initial_state(self):
        """Pack the dimensional initial fields in generated layout order."""
        values = {
            f"c_{species}": (
                self.ci[..., self.independent_index[species]]
                / self.c_ref_by_species[species]
            )
            for species in self.independent_species
        }
        values.update(
            {
                f"q_{species}": (
                    self.qi[..., self.adsorbate_index[species]]
                    / self.q_ref_by_species[species]
                )
                for species in self.adsorbates
            }
        )
        values.update(
            {
                "c_tot": np.asarray(self.c_tot).reshape(self.Nz, self.Nr)
                / self.c_tot_ref,
                "T_bed": np.asarray(self.T).reshape(self.Nz, self.Nr) / self.T_ref,
                "T_wall": np.asarray(self.Tw).reshape(self.Nz) / self.T_ref,
            }
        )
        return self.state_layout.pack_pde(values)

    def _report_progress(self, t):
        """Print each progress mark crossed by the current solver time."""
        nondim_horizon = getattr(
            self,
            "nondim_horizon",
            self.t_end / self.tau_hyd,
        )
        percent = 100.0 * t / nondim_horizon
        while (
            self.next_mark_idx < len(self.progress_marks)
            and percent >= self.progress_marks[self.next_mark_idx]
        ):
            mark = self.progress_marks[self.next_mark_idx]
            print(
                f"Simulation progress: {mark:.2f}% "
                f"({t * self.tau_hyd:.2f} s)"
            )
            self.next_mark_idx += 1

    def set_species_balance_diagnostic_callback(self, callback):
        """Attach an optional callback for gas-closure/RHS diagnostics.

        The callback receives one JSON-safe record per evaluated RHS state.
        Diagnostics are entirely disabled when ``callback`` is ``None`` and
        do not clip, project, or otherwise modify the solver state.
        """
        if callback is not None and not callable(callback):
            raise TypeError("Species-balance diagnostic callback must be callable or None.")
        self._species_balance_diagnostic_callback = callback

    def _reconstruct_gas_state(self, packed_state, c_tot_dim, eps):
        """Fill reusable independent/full gas fields, including closure.

        Clamped independent concentrations feed nonlinear properties and
        kinetics. Raw values remain in the mole-fraction field used by the
        central-difference dispersion operator, preserving its established
        behavior near species depletion.
        """
        Nz, Nr = self.Nz, self.Nr
        c_tot_dim_view = c_tot_dim.reshape(Nz, Nr)

        ci_ordered = self._ci_ordered_buf
        yi_ordered = self._yi_ordered_buf
        c_full_nd = self._c_full_nd_buf
        c_full_dim = self._c_full_dim_buf
        y_full = self._y_full_buf

        for independent, (species, field) in enumerate(
            zip(self.independent_species, self.gas_state_fields)
        ):
            gas = self.gas_index[species]
            raw = packed_state[field.state_slice]
            clamped = np.maximum(raw, eps)
            reference = self.c_ref_by_species[species]
            ci_ordered[..., independent] = clamped.reshape((Nz, Nr))
            yi_ordered[..., independent] = (
                raw * reference / c_tot_dim
            ).reshape((Nz, Nr))
            c_full_nd[..., gas] = clamped.reshape((Nz, Nr))
            c_full_dim[..., gas] = (raw * reference).reshape((Nz, Nr))
            # Floor only. An upper bound at 1 - eps (== 1.0 in float64) cut only
            # round-off above 1 in single-species gas, where the closure is ~0,
            # and made p_i = y_i * p_tot switch between depending on c_tot and
            # on c_i: a kink that stalled the tau = 8 cooldown in Newton.
            y_full[..., gas] = np.maximum(
                clamped * reference / c_tot_dim,
                eps,
            ).reshape((Nz, Nr))

        closure = self.gas_index[self.closure_species]
        y_full[..., closure] = 1.0
        c_full_dim[..., closure] = c_tot_dim_view
        for species in self.independent_species:
            gas = self.gas_index[species]
            y_full[..., closure] -= y_full[..., gas]
            c_full_dim[..., closure] -= c_full_dim[..., gas]
        c_full_nd[..., closure] = (
            c_tot_dim_view
            * y_full[..., closure]
            / self.c_ref_by_species[self.closure_species]
        )

        return (
            ci_ordered.ravel(),
            yi_ordered.ravel(),
            y_full.reshape(-1, self.Nc_gas),
        )

    def _corrected_species_dispersion_flux(self, yi_ordered_flat, *, axis):
        """Return mixture-averaged dispersive fluxes for independent species.

        Each species uses its own material diffusivity.  A correction velocity
        then makes the sum of the corresponding full-mixture molar fluxes zero.
        The returned values are in each independent species' nondimensional
        concentration units and still need the staggered total-concentration
        factor applied by the caller.
        """
        yi_ordered = np.asarray(yi_ordered_flat).reshape(self.shape_ci)
        if axis == 0:
            gradients = (
                self.Grad_ax_c.dot(yi_ordered_flat)
                + self.grad_ax_c_bc_dense
            ).reshape(self.shape_u_ax)
            yi_face = interp_cntr_to_stagg(
                yi_ordered, self.z_f, self.z_c, axis=0
            )
            inv_pe_ind = self._dispersion_inv_pe_ax_ind
            inv_pe_delta = self._dispersion_inv_pe_ax_delta
        elif axis == 1:
            gradients = (
                self.Grad_rad_c.dot(yi_ordered_flat)
                + self.grad_rad_c_bc_dense
            ).reshape(self.shape_u_rad)
            yi_face = interp_cntr_to_stagg(
                yi_ordered, self.r_f, self.r_c, axis=1
            )
            inv_pe_ind = self._dispersion_inv_pe_rad_ind
            inv_pe_delta = self._dispersion_inv_pe_rad_delta
        else:
            raise ValueError("Species dispersion axis must be 0 (axial) or 1 (radial).")

        correction_velocity = np.sum(
            inv_pe_delta * gradients,
            axis=-1,
            keepdims=True,
        )
        return (
            self._dispersion_ratio_ind
            * (-inv_pe_ind * gradients + yi_face * correction_velocity)
        ).ravel()

    def _evaluate_adsorption_rates(
        self,
        packed_state,
        y_full,
        p_tot_dim,
        T_bed_dim,
        eps,
    ):
        """Evaluate named equilibrium and kinetic models into RHS buffers."""
        Nz, Nr = self.Nz, self.Nr
        partial_pressures = {
            species: y_full[..., self.gas_index[species]].reshape(-1)
            * p_tot_dim
            for species in self.gas_species
        }

        current_loadings = {}
        q_ordered = self._q_ordered_buf
        for species, field in zip(self.adsorbates, self.loading_state_fields):
            adsorbate = self.adsorbate_index[species]
            loading_nd = np.maximum(packed_state[field.state_slice], eps)
            q_ordered[..., adsorbate] = loading_nd.reshape((Nz, Nr))
            current_loadings[species] = (
                loading_nd * self.q_ref_by_species[species]
            )

        equilibrium_loadings = self.equilibrium_system.equilibrium_loadings(
            partial_pressures_Pa=partial_pressures,
            temperature_K=T_bed_dim,
            current_loadings_mol_per_kg=current_loadings,
        )

        dq_ordered = self._dq_ordered_buf
        for species in self.adsorbates:
            adsorbate = self.adsorbate_index[species]
            physical_rates = self.kinetics[species].loading_rates_mol_per_kg_s(
                partial_pressures_Pa=partial_pressures,
                temperature_K=T_bed_dim,
                current_loadings_mol_per_kg=current_loadings,
                equilibrium_loadings_mol_per_kg=equilibrium_loadings,
            )
            dq_ordered[..., adsorbate] = (
                np.asarray(physical_rates[species]).reshape((Nz, Nr))
                * self.tau_hyd
                / self.q_ref_by_species[species]
            )

        return q_ordered, dq_ordered

    def _apply_adsorption_to_independent_gas_rates(
        self,
        dc_independent_dt,
        dq_ordered,
    ):
        """Apply adsorbate sinks to independently integrated gas balances."""
        for species in self.adsorbates:
            if species == self.closure_species:
                continue
            independent = self.independent_index[species]
            adsorbate = self.adsorbate_index[species]
            dc_independent_dt[..., independent] -= (
                self.psi_adsorbate[adsorbate]
                * np.reshape(dq_ordered[..., adsorbate], self._solid_2d.shape)
                * self._solid_2d
            )
        dc_independent_dt /= self.eps_total * self._gas_2d[..., None]
        return dc_independent_dt

    def _sensible_energy_accumulation_source(
        self,
        dc_tot_dt,
        dc_independent_dt,
        dq_matrix,
        theta_bed,
    ):
        """Return the sensible-inventory source relative to 298.15 K.

        ``theta_bed = (T - T_enthalpy_ref) / T0``.  The negative sign appears
        because the concentration/loading part of the enthalpy accumulation
        is moved to the right-hand side of the temperature equation.
        """
        dc_ind_dtau = dc_independent_dt.reshape(-1, self.Nc_ci)
        dc_closure_dtau = self._dc_closure_dtau_buf
        source_extra = self._source_extra_buf

        np.matmul(
            dc_ind_dtau,
            self._c_ref_ind_over_closure,
            out=source_extra,
        )
        np.multiply(
            self._c_tot_ref_over_closure,
            dc_tot_dt,
            out=dc_closure_dtau,
        )
        np.subtract(dc_closure_dtau, source_extra, out=dc_closure_dtau)

        gas_extra = self._gas_extra_buf
        np.matmul(dc_ind_dtau, self._acc_g_ind, out=gas_extra)
        np.multiply(self._acc_g_closure, dc_closure_dtau, out=source_extra)
        np.add(gas_extra, source_extra, out=gas_extra)
        np.multiply(gas_extra, self._gas_cell, out=gas_extra)

        ads_extra = self._ads_extra_buf
        np.matmul(dq_matrix, self.acc_ads, out=ads_extra)
        np.multiply(ads_extra, self._solid_cell, out=ads_extra)

        np.add(gas_extra, ads_extra, out=source_extra)
        np.multiply(source_extra, theta_bed, out=source_extra)
        np.negative(source_extra, out=source_extra)
        return source_extra

    def _assemble_rhs(
        self,
        out,
        *,
        dc_independent_dt,
        dq_ordered,
        dc_tot_dt,
        dT_dt,
        dTw_dt,
        dm_i_dt,
        dE_compression_dt,
        dE_vacuum_dt,
        dE_jacket_dt,
    ):
        """Pack PDE and integrated-state derivatives in generated layout order."""
        if out is None:
            out = self._rhs_buf
        else:
            out = validate_flat_vector(
                out,
                size=self.state_layout.state_size,
                label="RHS output buffer",
            )

        for species, field in zip(
            self.independent_species,
            self.gas_state_fields,
        ):
            out[field.state_slice] = dc_independent_dt[
                ..., self.independent_index[species]
            ].ravel()
        for species, field in zip(self.adsorbates, self.loading_state_fields):
            out[field.state_slice] = dq_ordered[
                ..., self.adsorbate_index[species]
            ].ravel()
        out[self.state_field_by_name["c_tot"].state_slice] = dc_tot_dt
        out[self.state_field_by_name["T_bed"].state_slice] = dT_dt
        out[self.state_field_by_name["T_wall"].state_slice] = dTw_dt

        for adsorbate, field in enumerate(self.mass_scalar_fields):
            out[field.state_slice] = dm_i_dt[adsorbate]
        out[self.state_field_by_name["E_compression"].state_slice] = (
            dE_compression_dt.item()
        )
        out[self.state_field_by_name["E_vacuum"].state_slice] = (
            dE_vacuum_dt.item()
        )
        out[self.state_field_by_name["E_jacket"].state_slice] = (
            dE_jacket_dt.item()
        )
        return out

    def pde_system_kpis(self, t, Y, out=None):
        """
        Evaluate the semi-discrete model RHS including integrated KPI states.

        Thermal coupling depends on the spatial mode:
        - 2D mode: bed/wall heat exchange is enforced through the radial
          boundary condition of the bed-temperature equation.
        - 1D mode: radial bed conduction is disabled and the bed temperature
          equation receives an explicit source term
          `omega_wall * (T_wall - T_bed)`; the wall balance keeps the same
          form and uses the resolved architecture coefficient `h_int = U_w`
          (or the explicit experimental override).

        Parameters
        ----------
        t : float
            Current nondimensional time.
        Y : np.ndarray
            Current nondimensional state vector in ``self.state_layout`` order.

        Returns
        -------
        np.ndarray
            Time derivative `dY/dt` in nondimensional form.
        """
        self._report_progress(t)

        # ---------------------------------------------------------
        # 1) Layout-owned state decomposition and mixture closure
        # ---------------------------------------------------------
        packed_state = validate_flat_vector(
            Y,
            size=self.state_layout.state_size,
            label="Packed state",
        )
        eps = 1e-32

        c_tot = packed_state[self.state_field_by_name["c_tot"].state_slice]
        T_bed = packed_state[self.state_field_by_name["T_bed"].state_slice]
        T_wall = packed_state[self.state_field_by_name["T_wall"].state_slice]
        c_tot_dim = c_tot * self.c_tot_ref
        ci_ordered_flat, yi_ordered_flat, y_full_matrix = (
            self._reconstruct_gas_state(packed_state, c_tot_dim, eps)
        )
        c_full_nd = self._c_full_nd_buf
        c_full_dim = self._c_full_dim_buf
        y_full = self._y_full_buf

        # Total concentration at staggered points for dispersion and velocity.
        p_tot = c_tot * T_bed
        c_tot_view = c_tot.reshape(self.shape_single)
        c_tot_stagg_ax = interp_cntr_to_stagg(
            c_tot_view,
            self.z_f,
            self.z_c,
            axis=0,
        )
        c_tot_stagg_rad = interp_cntr_to_stagg(
            c_tot_view,
            self.r_f,
            self.r_c,
            axis=1,
        )

        # Update velocity fields
        Mw = y_full_matrix @ self.MW
        rho = c_tot_dim * Mw
        self.update_velocity_fields(p_tot, rho)

        # ---------------------------------------------------------
        # 2) Coupled equilibrium and physical adsorption rates
        # ---------------------------------------------------------
        p_tot_dim = p_tot * self.p_ref
        p_tot_dim_view = p_tot_dim.reshape(self.shape_single)
        T_bed_dim = T_bed * self.T_ref
        q_ordered, dq_ordered = self._evaluate_adsorption_rates(
            packed_state,
            y_full,
            p_tot_dim,
            T_bed_dim,
            eps,
        )
        
        # ---------------------------------------------------------
        # 3) Total concentration balance
        # ---------------------------------------------------------
        Conv_ax_single, conv_ax_single_bc = self.conv_ax_cache.update(self.u_ax_single)
        bc = conv_ax_single_bc.copy()
        bc[:self.flux_in_ctot.size] = self.flux_in_ctot.ravel()
        conv_flux_ax_ctot = Conv_ax_single.dot(c_tot) + bc

        # Radial bulk flow.  Unlike dispersion (whose species fluxes sum to
        # zero by construction), convection transports total moles, so it
        # belongs in the c_tot balance exactly as the axial term does.
        Conv_rad_single, conv_rad_single_bc = self.conv_rad_cache.update(self.u_rad_single)
        conv_flux_rad_ctot = Conv_rad_single.dot(c_tot) + conv_rad_single_bc

        # `Div_rad` acts on the radial grid scaled by `R_b` while the time scale
        # is `tau_hyd = L_b/u_ref`, so the radial divergence needs `L_b/R_b` to
        # be commensurate with the axial one.  This is a SECOND factor, distinct
        # from the one inside `_radial_velocity_from_pressure_law`, which only
        # makes the pressure law see the true dimensional `dp/dr`.  The radial
        # diffusive terms already carry the same factor (`D_rad`, `k_cond_rad`,
        # `_dispersion_inv_pe_rad_*`).
        conv_ctot = -(
            self.Div_ax_ctot.dot(conv_flux_ax_ctot)
            + self.L_R_ratio * self.Div_rad_ctot.dot(conv_flux_rad_ctot)
        )
        dq_matrix = dq_ordered.reshape(-1, self.Nc_solid)
        source = -(dq_matrix @ self.phi_adsorbate) * self._solid_cell
        
        # Discretized total gas phase equations
        dc_tot_dt = (conv_ctot + source) / (self.eps_total * self._gas_cell)
        
        # ---------------------------------------------------------
        # 4) Independent gas-species balances
        # ---------------------------------------------------------
        # Compute the convective and dispersion terms:
        Conv_ax_ci, conv_ax_bc_ci = self.conv_ax_species_cache.update(self.u_ax_single)
        bc = conv_ax_bc_ci.copy()
        bc[:self.flux_in_ci.size] = self.flux_in_ci.ravel()
        conv_flux_ax_ci = Conv_ax_ci.dot(ci_ordered_flat) + bc
        disp_flux_ax = self._corrected_species_dispersion_flux(
            yi_ordered_flat, axis=0
        )
        disp_flux_ax_i = (c_tot_stagg_ax * disp_flux_ax.reshape(self.shape_u_ax)).ravel()
        disp_flux_rad = self._corrected_species_dispersion_flux(
            yi_ordered_flat, axis=1
        )
        disp_flux_rad_i = (c_tot_stagg_rad * disp_flux_rad.reshape(self.shape_u_rad)).ravel()
        Conv_rad_ci, conv_rad_bc_ci = self.conv_rad_species_cache.update(self.u_rad_single)
        conv_flux_rad_ci = Conv_rad_ci.dot(ci_ordered_flat) + conv_rad_bc_ci
        # The radial dispersive flux already carries `L_R_ratio` through
        # `_dispersion_inv_pe_rad_*`; the convective one needs it applied here.
        conv_disp = -(
            self.Div_ax_ci.dot(conv_flux_ax_ci + disp_flux_ax_i)
            + self.Div_rad_ci.dot(
                self.L_R_ratio * conv_flux_rad_ci + disp_flux_rad_i
            )
        )
        
        dc_independent_dt = conv_disp.reshape(self.shape_ci)
        self._apply_adsorption_to_independent_gas_rates(
            dc_independent_dt,
            dq_ordered,
        )

        if self._species_balance_diagnostic_callback is not None:
            diagnostic = build_species_balance_diagnostic(
                self,
                t=t,
                packed_state=packed_state,
                c_tot_dim=c_tot_dim,
                conv_ctot=conv_ctot,
                source_ctot=source,
                conv_flux_ax_ci=conv_flux_ax_ci,
                conv_flux_rad_ci=conv_flux_rad_ci,
                disp_flux_ax_i=disp_flux_ax_i,
                disp_flux_rad_i=disp_flux_rad_i,
                dq_ordered=dq_ordered,
                dc_tot_dt=dc_tot_dt,
                dc_independent_dt=dc_independent_dt,
            )
            self._species_balance_diagnostic_callback(diagnostic)

        # ---------------------------------------------------------
        # 5) Energy balance
        # ---------------------------------------------------------
        # Ideal-gas sensible enthalpy relative to 298.15 K.  T_bed remains the
        # absolute nondimensional temperature T/T0 everywhere else.
        theta_bed = T_bed - self.T_enthalpy_ref_nd
        h_g = (y_full_matrix @ self.cp_g) * theta_bed
        
        # Accumulation terms
        acc_gas = c_full_nd.reshape(-1, self.Nc_gas) @ self.acc_g
        acc_ads = q_ordered.reshape(-1, self.Nc_solid) @ self.acc_ads
        rho_cp_eff = (
            acc_gas * self._gas_cell
            + (self.acc_sorbent + self.acc_fixed_adsorbates + acc_ads)
            * self._solid_cell
            + self.acc_structure
        )
        
        # Convection terms div(v * c_tot * h_g)
        bc = conv_ax_single_bc.copy()
        bc[:self.flux_in_T.size] = self.flux_in_T.ravel()
        conv_flux_ax_T = Conv_ax_single.dot(c_tot_dim * h_g) + bc
        conv_flux_rad_T = Conv_rad_single.dot(c_tot_dim * h_g) + conv_rad_single_bc
        conv_T = -(
            self.Div_ax_T.dot(conv_flux_ax_T)
            + self.L_R_ratio * self.Div_rad_T.dot(conv_flux_rad_T)
        )
        
        # Conduction terms
        cond_ax_T = -self.Div_ax_T.dot(self.Cond_flux_ax.dot(T_bed) + self.cond_flux_ax_bc_dense)

        if self.is_1d:
            cond_rad_T = np.zeros_like(T_bed)
            wall_exch = self.omega_wall * (T_wall - T_bed)
            T_bed_at_wall = T_bed.reshape(self.shape_single[:-1])[:, -1]
        else:
            bc_rad_T = ({'a': 1, 'b': 0, 'd': 0}, {'a': self.inv_Bi_h, 'b': 1, 'd': T_wall.reshape(self.shape_Tw)})
            Grad_rad_T, grad_rad_T_bc = self.grad_rad_T_cache.update_bc(bc_rad_T)
            cond_rad_T = self.Div_Diff_rad_T.dot(Grad_rad_T.dot(T_bed) + grad_rad_T_bc)
            wall_exch = 0.0

            T_bed_2d = T_bed.reshape(self.shape_single[:-1])
            T_bed_at_wall = (T_wall - self.inv_Bi_h * (self.wall_face_c_N * T_bed_2d[:, -1] + self.wall_face_c_M * T_bed_2d[:, -2])
            ) / (1.0 + self.inv_Bi_h * self.wall_face_c_F)
        cond_T = cond_ax_T + cond_rad_T
        
        # Sink term due to adsorption heat 
        source_q = (dq_matrix @ self.Omega_adsorbate) * self._solid_cell
        
        # Pressure-accumulation correction from p = c_tot * R_g * T.  This
        # depends on absolute temperature, not on the enthalpy datum.
        omega_p_cell = self.omega_p * self._gas_cell
        source_p = omega_p_cell * T_bed * dc_tot_dt

        source_extra = self._sensible_energy_accumulation_source(
            dc_tot_dt,
            dc_independent_dt,
            dq_matrix,
            theta_bed,
        )

        # Total source term
        source = source_q + source_p + source_extra + wall_exch
        
        # Discretized temperature equations
        dT_dt = (conv_T + cond_T + source) / (rho_cp_eff - omega_p_cell * c_tot)
        
        # ---------------------------------------------------------
        # 6) Wall energy balance
        # ---------------------------------------------------------
        cond_Tw = -self.Div_ax_Tw.dot(self.Cond_flux_ax_Tw.dot(T_wall) + self.cond_flux_ax_Tw_bc_dense)
        dTw_dt = self.alpha_in * (T_bed_at_wall - T_wall) - self.alpha_out * (T_wall - self.T_jacket_nd) + cond_Tw
        
        # ---------------------------------------------------------
        # 7) Integrated mass and energy states
        # ---------------------------------------------------------        
        # Compute instantaneous net flow rate [kg/s].
        weight_last, weight_inner = self.outlet_face_weights
        outlet_face_concentrations = (
            weight_last * c_full_dim[-1]
            + weight_inner * c_full_dim[self._outlet_face_inner_index]
        )
        outlet_concentrations = np.take(
            outlet_face_concentrations, self.adsorbate_gas_indices, axis=-1
        )
        mass_flow_out_i = self._compute_outlet_mass_fluxes(
            outlet_concentrations,
            self.u_ax_single[-1] * self.u_ref,
        )
        dm_i_dt = (
            self.mass_flow_in_i[self.adsorbate_gas_indices] - mass_flow_out_i
        ) * self.tau_hyd
   
        # Compute instantaneous energy powers [W] 
        power_compression, power_vacuum, power_jacket = self._compute_instantaneous_energies(
            p_tot_dim_view[0],                                          # inlet pressure [Pa]
            p_tot_dim_view[-1],                                         # outlet pressure [Pa]
            self.u_ax_single[0] * self.u_ref,                           # inlet velocity [m/s]
            self.u_ax_single[-1] * self.u_ref,                          # outlet velocity [m/s]
            T_wall * self.T_ref                                         # wall temperature [K]
            )
                
        # Scale by tau_hyd because of nondimensionalized time
        dE_compression_dt = (power_compression * self.tau_hyd).flatten()
        dE_vacuum_dt   = (power_vacuum * self.tau_hyd).flatten()
        dE_jacket_dt     = (power_jacket * self.tau_hyd).flatten()

        return self._assemble_rhs(
            out,
            dc_independent_dt=dc_independent_dt,
            dq_ordered=dq_ordered,
            dc_tot_dt=dc_tot_dt,
            dT_dt=dT_dt,
            dTw_dt=dTw_dt,
            dm_i_dt=dm_i_dt,
            dE_compression_dt=dE_compression_dt,
            dE_vacuum_dt=dE_vacuum_dt,
            dE_jacket_dt=dE_jacket_dt,
        )
    
    def _axial_outlet_face_weights(self):
        """Return the two cell weights that build the outlet face value.

        `bc_ax[1]` imposes a zero axial gradient at ``z = L``, so the
        concentration the discretization actually transports through the last
        face is an extrapolation of the last two cells, not the last cell
        centre. Any flux reported against that boundary has to use the same
        combination, or the integrated balance drifts by one cell of the outlet
        gradient per unit throughput. The weights are read back from the pymrm
        operator instead of being re-derived here, so a change of boundary
        condition or a non-uniform axial grid follows automatically.

        Returns
        -------
        tuple of float
            ``(w_last, w_inner)`` such that the face value is
            ``w_last * c[-1] + w_inner * c[-2]``. A single-cell bed has no
            second cell to lean on and falls back to the cell value.
        """
        if self.Nz < 2:
            return 1.0, 0.0
        probe, _ = construct_convflux_bc(
            (self.Nz, 1, 1), self.z_f, self.z_c,
            bc=(None, self.bc_ax[1]), v=1.0, axis=0,
        )
        outlet_row = np.asarray(probe.toarray()[self.Nz]).ravel()
        return float(outlet_row[-1]), float(outlet_row[-2])

    def _compute_outlet_mass_fluxes(self, ci, u_ax):
        """
        Compute outlet mass flow rates for the declared adsorbates.

        Parameters
        ----------
        ci : np.ndarray
            Outlet **face** concentrations in adsorbate order [mol/m³],
            shaped ``(Nr, n_adsorbates)``. Build them with
            `outlet_face_weights`, not from the last cell centre.
        u_ax : np.ndarray
            Outlet interstitial axial velocity [m/s] (radial profile).

        Returns
        -------
        np.ndarray
            Outlet mass flow rates in adsorbate order [kg/s].
        """
        molar_flow = np.dot(self._area_rings, np.asarray(ci) * np.asarray(u_ax))
        return molar_flow * self.MW[self.adsorbate_gas_indices]
        
    def _compute_instantaneous_energies(self, p_inlet, p_outlet, u_inlet, u_outlet, T_wall):
        """
        Compute instantaneous power terms for auxiliaries and jacket.

        Parameters
        ----------
        p_inlet : np.ndarray
            Inlet pressure profile [Pa].
        p_outlet : np.ndarray
            Outlet pressure profile [Pa].
        u_inlet : np.ndarray
            Inlet velocity profile [m/s].
        u_outlet : np.ndarray
            Outlet velocity profile [m/s].
        T_bed_wall : np.ndarray
            Bed temperature profile at the wall (r = R_b) along the axial direction [K].
        T_wall : np.ndarray
            Wall temperature profile along the axial direction [K].

        Returns
        -------
        tuple
            `(power_compression, power_vacuum, dE_jacket_dt)` [W].
        """

        # Total volumetric flow [m³/s]
        Q_in_avg = np.dot(self._area_rings, np.ravel(u_inlet))        # [m³/s]
        Q_out_avg = np.dot(self._area_rings, np.ravel(u_outlet))      # [m³/s]

        # Pressure ratio and pressure drop
        p_inlet_avg = np.dot(self._area_rings, np.ravel(p_inlet)) / self._area_rings_sum
        p_outlet_avg = np.dot(self._area_rings, np.ravel(p_outlet)) / self._area_rings_sum
        pressure_ratio_compr = p_inlet_avg / p_outlet_avg               # Compression ratio
        pressure_ratio_vac = self.p_amb / p_outlet_avg                  # Discharge at atmospheric pressure

        # Compression Model (Polytropic Compressible)
        power_compression = self._compression_prefactor * p_inlet_avg * Q_in_avg
        power_compression *= pressure_ratio_compr ** self._energy_poly_exponent - 1.0
        
        # Vacuum Pump Model (Fully Compressible, Polytropic) 
        pressure_scale = self._vacuum_eta_pressure_scale * p_outlet_avg
        eta_vacuum_pump = self._vacuum_eta_prefactor * p_outlet_avg / (1.0 + pressure_scale)                 # Ward et al.
        power_vacuum = (self._energy_gamma_ratio / eta_vacuum_pump) * p_outlet_avg * Q_out_avg 
        power_vacuum *= pressure_ratio_vac ** self._energy_poly_exponent - 1.0

        # --- Jacket Energy ---
        dE_jacket_dt = self._jacket_power_constant - self.h_ext * np.dot(
            self._jacket_quad_weights,
            np.ravel(T_wall),
        )

        return power_compression, power_vacuum, dE_jacket_dt
