"""Conventional packed-bed structure resolution.

The resolver is deliberately independent of ``ColumnModelMoL``.  It maps the
packed-bed design and common particle properties to the semantic structure
contract consumed by the common column equations.

The source follows the runtime resolution narrative:

1. structure and inventory;
2. pressure drop;
3. mass dispersion;
4. thermal transport primitives and their typed resolver.
"""

from __future__ import annotations

import builtins
import math
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
from pymrm import interp_cntr_to_stagg

from pydac.core.bed_architectures.base import (
    DEFAULT_WALL_HTC_MODEL,
    DEFAULT_STAGNANT_CONDUCTIVITY_MODEL,
    PackedBedDesign,
    ParticleProperties,
    ResolvedStructure,
    ResolvedTransport,
    validate_stagnant_conductivity_model,
    validate_wall_htc_model,
)


# ---------------------------------------------------------------------------
# 1. Structure and inventory
# ---------------------------------------------------------------------------

def resolve_packed_bed_structure(
    design: PackedBedDesign,
    particle: ParticleProperties,
) -> ResolvedStructure:
    """Resolve conventional packed-bed storage and sorbent properties."""
    if not builtins.isinstance(design, PackedBedDesign):
        raise TypeError("`design` must be a PackedBedDesign instance.")
    if not builtins.isinstance(particle, ParticleProperties):
        raise TypeError("`particle` must be a ParticleProperties instance.")

    return ResolvedStructure(
        eps_flow=design.eps_b,
        eps_total=design.eps_b + (1.0 - design.eps_b) * particle.porosity,
        rho_sorbent=(1.0 - design.eps_b) * particle.density,
        acc_structure=0.0,
    )


def _ring_area_average(profile, r_faces, samples_per_ring=64):
    xs, ys = (np.array(v, dtype=float) for v in zip(*profile))
    out = np.empty(r_faces.size - 1)
    for i in range(out.size):
        r = np.linspace(r_faces[i], r_faces[i + 1], samples_per_ring + 1)
        rc = 0.5 * (r[1:] + r[:-1])
        area = rc * np.diff(r)
        out[i] = np.sum(np.interp(rc, xs, ys) * area) / np.sum(area)
    return out


def _area_mean(values, r_faces):
    ring_area = r_faces[1:] ** 2 - r_faces[:-1] ** 2
    return np.sum(values * ring_area) / np.sum(ring_area)


def packed_bed_ring_inlet_flux_weights(
    design: PackedBedDesign,
    r_faces_over_R: Any,
) -> np.ndarray:
    """Ring-averaged relative inlet flux, normalised to an area mean of 1."""
    r_faces = np.asarray(r_faces_over_R, dtype=float)
    if design.inlet_flux_profile is None:
        return np.ones(r_faces.size - 1)
    ring_flux = _ring_area_average(design.inlet_flux_profile, r_faces)
    return ring_flux / _area_mean(ring_flux, r_faces)


def packed_bed_ring_structure_factors(
    design: PackedBedDesign,
    particle: ParticleProperties,
    r_faces_over_R: Any,
    *,
    mean_tolerance: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-ring (gas holdup, sorbent inventory) factors relative to the bed mean.

    Both have area mean 1, so total gas holdup and sorbent mass equal the
    uniform-voidage bed.
    """
    r_faces = np.asarray(r_faces_over_R, dtype=float)
    n_rings = r_faces.size - 1
    if design.radial_porosity_profile is None:
        return np.ones(n_rings), np.ones(n_rings)
    eps_ring = _ring_area_average(design.radial_porosity_profile, r_faces)
    shift = design.eps_b - _area_mean(eps_ring, r_faces)
    if abs(shift) > mean_tolerance:
        raise ValueError(
            "`radial_porosity_profile` area mean differs from `eps_b` by "
            f"{-shift:+.4f} on this grid."
        )
    eps_ring = eps_ring + shift
    eps_total_mean = design.eps_b + (1.0 - design.eps_b) * particle.porosity
    gas = (eps_ring + (1.0 - eps_ring) * particle.porosity) / eps_total_mean
    solid = (1.0 - eps_ring) / (1.0 - design.eps_b)
    if np.any(solid <= 0.0):
        raise ValueError("`radial_porosity_profile` leaves a ring without sorbent.")
    return gas, solid


# ---------------------------------------------------------------------------
# 2. Pressure drop
# ---------------------------------------------------------------------------

def solve_quadratic_drag_velocity(
    driving_force: Any,
    linear_coefficient: Any,
    inertial_coefficient: Any,
) -> Any:
    """Solve ``a*u + b*u*abs(u) = G`` without cancellation near zero.

    Nonnegative finite drag coefficients are required, with at least one
    positive coefficient per entry. Both signs of G are physical. Zero drag
    is not regularized into an arbitrary velocity, and invalid inputs or an
    unrepresentable result raise instead of propagating NaN/Inf.

    The rationalized root is G / (a/2 + hypot(a/2, sqrt(b)*sqrt(abs(G)))).
    Scaling the denominator avoids squaring coefficients or overflowing its
    sum. The exactly linear and quadratic limits need no switching tolerance.
    """
    force, linear, inertial = np.broadcast_arrays(
        np.asarray(driving_force, dtype=float),
        np.asarray(linear_coefficient, dtype=float),
        np.asarray(inertial_coefficient, dtype=float),
    )
    if not (np.all(np.isfinite(force)) and np.all(np.isfinite(linear))
            and np.all(np.isfinite(inertial))):
        raise ValueError("Drag inversion requires finite force and coefficients.")
    if np.any(linear < 0.0) or np.any(inertial < 0.0):
        raise ValueError("Drag coefficients must be nonnegative; signed flow belongs in the force.")
    if np.any((linear == 0.0) & (inertial == 0.0)):
        raise ValueError("Drag inversion is undefined when both coefficients are zero.")

    magnitude = np.abs(force)
    velocity = np.zeros_like(force)
    active = magnitude > 0.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            cross = np.sqrt(inertial[active]) * np.sqrt(magnitude[active])
            scale = np.maximum(linear[active], cross)
            half_linear = 0.5 * (linear[active] / scale)
            denominator = half_linear + np.hypot(half_linear, cross / scale)
            # If G/scale alone overflows, divide by the O(1) denominator
            # first. The final velocity can still be representable.
            with np.errstate(over="ignore"):
                scaled_force = magnitude[active] / scale
            result = scaled_force / denominator
            overflow = ~np.isfinite(scaled_force)
            result[overflow] = (magnitude[active][overflow] / denominator[overflow]) / scale[overflow]
            velocity[active] = result
    except FloatingPointError as exc:
        raise ValueError("Drag inversion result is outside the floating-point range.") from exc
    if np.any(~np.isfinite(velocity)) or np.any(active & (velocity == 0.0)):
        raise ValueError("Drag inversion result is outside the floating-point range.")
    return np.sign(force) * velocity


@dataclass(frozen=True)
class ErgunPressureDrop:
    """Conventional packed-bed Ergun pressure-drop law.

    The dimensional constitutive law is the Ergun relation:

        S. Ergun, ``Fluid flow through packed columns``, Chemical Engineering
        Progress, 48(2) (1952), 89-94.

    ``velocity_from_gradient`` accepts the dimensionless pressure gradient
    ``d(p/p_ref)/d(z/L_b)`` and returns the dimensionless superficial velocity
    ``u/u_ref``.  Local density is dimensional and evaluated at the axial
    staggered faces by the caller.
    """

    eps_flow: float
    viscosity: float
    particle_diameter: float
    bed_length: float
    reference_velocity: float
    reference_pressure: float
    name: ClassVar[str] = "ergun"

    def __post_init__(self) -> None:
        if not 0.0 < self.eps_flow < 1.0:
            raise ValueError("`eps_flow` must lie strictly between zero and one.")
        for name in (
            "viscosity",
            "particle_diameter",
            "bed_length",
            "reference_velocity",
            "reference_pressure",
        ):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"`{name}` must be finite and positive.")

    @property
    def linear_coefficient(self) -> float:
        """Return the dimensionless viscous Ergun coefficient ``A``."""
        return (
            self.bed_length
            * self.reference_velocity
            / self.reference_pressure
            * 150.0
            * self.viscosity
            / self.particle_diameter**2
            * (1.0 - self.eps_flow) ** 2
            / self.eps_flow**3
        )

    @property
    def inertial_coefficient_per_density(self) -> float:
        """Return the inertial coefficient per dimensional density."""
        return (
            self.bed_length
            * self.reference_velocity**2
            / self.reference_pressure
            * 1.75
            / self.particle_diameter
            * (1.0 - self.eps_flow)
            / self.eps_flow**3
        )

    def velocity_from_gradient(
        self,
        pressure_gradient: Any,
        local_density: Any,
    ) -> Any:
        """Return dimensionless superficial velocity from pressure gradient."""
        driving_force = -np.asarray(pressure_gradient, dtype=float)
        inertial_coefficient = (
            self.inertial_coefficient_per_density
            * np.asarray(local_density, dtype=float)
        )
        return solve_quadratic_drag_velocity(
            driving_force,
            self.linear_coefficient,
            inertial_coefficient,
        )


@dataclass(frozen=True, eq=False)
class RadialErgunPressureDrop:
    """Ergun law evaluated on a resolved radial DRAG voidage field.

    Same constitutive relation and same signed inversion as
    :class:`ErgunPressureDrop`; only the voidage entering the two coefficients
    becomes a radial field.  ``eps_drag`` is the regularised drag field owned by
    ``bed_architectures.channeling`` - the exact area-overlap projection of the
    fixed physical drag layers - and never the ring-mean storage voidage, whose
    wall value has no continuum limit.

    The same local isotropic resistance acts in both directions in the 2D
    vector closure. Radial faces use interpolated regularised drag voidage;
    interpolation precedes the nonlinear Ergun coefficient evaluation.

    The coefficients carry an explicit radial axis, shape ``(1, Nr, 1)``, so
    they broadcast against the staggered axial faces ``(Nz+1, Nr, 1)`` and
    cannot be transposed onto the axial axis by accident.
    """

    eps_drag: Any
    viscosity: float
    particle_diameter: float
    bed_length: float
    reference_velocity: float
    reference_pressure: float
    name: ClassVar[str] = "ergun_radial"

    def __post_init__(self) -> None:
        eps = np.asarray(self.eps_drag, dtype=float)
        if eps.ndim != 1 or eps.size < 2:
            raise ValueError("`eps_drag` must be a 1-D radial field of >= 2 rings.")
        if not np.all(np.isfinite(eps)) or not np.all((eps > 0.0) & (eps < 1.0)):
            raise ValueError(
                "`eps_drag` must lie strictly between zero and one in every ring."
            )
        object.__setattr__(self, "eps_drag", eps)
        for name in (
            "viscosity",
            "particle_diameter",
            "bed_length",
            "reference_velocity",
            "reference_pressure",
        ):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"`{name}` must be finite and positive.")

    @property
    def radial_size(self) -> int:
        """Return the number of radial rings the law is resolved on."""
        return int(np.asarray(self.eps_drag).size)

    @property
    def linear_coefficient(self) -> np.ndarray:
        """Return the dimensionless viscous Ergun coefficients, shape (1,Nr,1)."""
        eps = np.asarray(self.eps_drag, dtype=float)
        values = (
            self.bed_length
            * self.reference_velocity
            / self.reference_pressure
            * 150.0
            * self.viscosity
            / self.particle_diameter**2
            * (1.0 - eps) ** 2
            / eps**3
        )
        return values.reshape((1, -1, 1))

    @property
    def inertial_coefficient_per_density(self) -> np.ndarray:
        """Return the inertial coefficients per density, shape (1,Nr,1)."""
        eps = np.asarray(self.eps_drag, dtype=float)
        values = (
            self.bed_length
            * self.reference_velocity**2
            / self.reference_pressure
            * 1.75
            / self.particle_diameter
            * (1.0 - eps)
            / eps**3
        )
        return values.reshape((1, -1, 1))

    def coefficients_on_radial_faces(self, r_faces, r_centres):
        """Evaluate Ergun on Nr+1 faces using the native scalar interpolation.

        The regularised drag voidage is bounded away from 0 and 1. Boundary
        faces are impermeable; use adjacent-cell voidage there rather than
        unconstrained extrapolation outside the physical interval.
        """
        eps_faces = interp_cntr_to_stagg(
            self.eps_drag.reshape(1, -1, 1), r_faces, r_centres, axis=1)
        eps_faces[:, 0, :] = self.eps_drag[0]
        eps_faces[:, -1, :] = self.eps_drag[-1]
        face_law = replace(self, eps_drag=eps_faces.ravel())
        return face_law.linear_coefficient, face_law.inertial_coefficient_per_density

    def velocity_from_gradient(
        self,
        pressure_gradient: Any,
        local_density: Any,
    ) -> Any:
        """Return dimensionless superficial velocity from pressure gradient."""
        driving_force = -np.asarray(pressure_gradient, dtype=float)
        inertial_coefficient = (
            self.inertial_coefficient_per_density
            * np.asarray(local_density, dtype=float)
        )
        return solve_quadratic_drag_velocity(
            driving_force,
            self.linear_coefficient,
            inertial_coefficient,
        )


def resolve_packed_bed_radial_pressure_drop(
    *,
    eps_drag: Any,
    viscosity: float,
    particle_diameter: float,
    bed_length: float,
    reference_velocity: float,
    reference_pressure: float,
) -> RadialErgunPressureDrop:
    """Resolve the radial packed-bed Ergun law on a resolved drag field."""
    return RadialErgunPressureDrop(
        eps_drag=eps_drag,
        viscosity=viscosity,
        particle_diameter=particle_diameter,
        bed_length=bed_length,
        reference_velocity=reference_velocity,
        reference_pressure=reference_pressure,
    )


def resolve_packed_bed_pressure_drop(
    *,
    eps_flow: float,
    viscosity: float,
    particle_diameter: float,
    bed_length: float,
    reference_velocity: float,
    reference_pressure: float,
) -> ErgunPressureDrop:
    """Resolve the detached conventional packed-bed pressure law."""
    return ErgunPressureDrop(
        eps_flow=eps_flow,
        viscosity=viscosity,
        particle_diameter=particle_diameter,
        bed_length=bed_length,
        reference_velocity=reference_velocity,
        reference_pressure=reference_pressure,
    )


# ---------------------------------------------------------------------------
# 3. Material dispersion
# ---------------------------------------------------------------------------

def resolve_packed_bed_dispersion(
    *,
    eps_flow: float,
    molecular_diffusivity: Any,
    reynolds: float,
    schmidt: Any,
    superficial_inlet_velocity: float,
    particle_diameter: float,
    reference_velocity: float,
    bed_length: float,
    bed_radius: float,
) -> dict[str, Any]:
    """Resolve the current packed-bed axial/radial material dispersion model.

    ``Pe_m`` denotes the material Péclet number. It is evaluated as
    ``Re * Sc`` in the axial closure, with the Schmidt number formed from the
    molecular diffusivity ``D_m``. The returned ``inv_Pe_m_*`` values are the
    inverse material Péclet groups used by the PDE after the effective
    dispersion coefficients have been resolved.

    The axial definition follows:

        E. Tsotsas & E. U. Schlunder, ``On axial dispersion in packed beds
        with fluid flow: Über die axiale dispersion in durchströmten
        festbetten``, Chemical Engineering and Processing: Process
        Intensification, 24(1) (1988), 15-31.
        doi:10.1016/0255-2701(88)87002-8.

    The radial definition follows:

        J. M. P. Q. Delgado, ``Longitudinal and transverse dispersion in
        porous media``, Chemical Engineering Research and Design, 85(9)
        (2007), 1245-1252. doi:10.1205/CHERD07017.
    """
    molecular_diffusivity = np.asarray(molecular_diffusivity, dtype=float)
    schmidt = np.asarray(schmidt, dtype=float)

    peclet_material = reynolds * schmidt
    effective_molecular_diffusivity = (
        1.0 - np.sqrt(1.0 - eps_flow)
    ) * molecular_diffusivity
    dispersion_axial = (
        effective_molecular_diffusivity
        + peclet_material * molecular_diffusivity / 2.0
    )
    inverse_peclet_material_axial = (
        dispersion_axial / reference_velocity / bed_length
    )

    dispersion_radial = (
        effective_molecular_diffusivity
        + superficial_inlet_velocity * particle_diameter / 12.0
    )
    inverse_peclet_material_radial = (
        dispersion_radial / reference_velocity / bed_radius
    )

    return {
        "model": "current_pydac",
        "Disp_ax": dispersion_axial,
        "Disp_rad": dispersion_radial,
        "inv_Pe_m_ax": inverse_peclet_material_axial,
        "inv_Pe_m_rad": inverse_peclet_material_radial,
    }


# ---------------------------------------------------------------------------
# 4. Thermal transport
# ---------------------------------------------------------------------------

def zehner_schlunder_stagnant_conductivity(
    eps_flow: float,
    gas_conductivity: float,
    particle_conductivity: float,
) -> float:
    """Return stagnant conductivity from the modified Zehner-Schlunder law.

    The original Zehner-Schlunder correlation is:

        P. Zehner & E. U. Schlunder, ``Wärmeleitfähigkeit von
        Schüttungen bei mäßigen Temperaturen``, Chemie Ingenieur Technik,
        42(14) (1970), 933-941. doi:10.1002/CITE.330421408.

    This implementation uses the modified correlation:

        C. T. Hsu, P. Cheng & K. W. Wong, ``Modified Zehner-Schlunder
        models for stagnant thermal conductivity of porous media``,
        International Journal of Heat and Mass Transfer, 37(17) (1994),
        2751-2759. doi:10.1016/0017-9310(94)90392-1.
    """
    m = 0.9676
    gamma = gas_conductivity / particle_conductivity
    b = ((1.0 - eps_flow) / eps_flow) ** m

    first_term = 1.0 - (1.0 - eps_flow) ** 0.5
    second_term = (1.0 - eps_flow**0.5) / gamma
    third_term = (1.0 - eps_flow) ** 0.5 + eps_flow**0.5 - 1.0
    fourth_term = (
        b * (1.0 - gamma)
        / (1.0 - b * gamma) ** 2 * math.log(1.0 / (gamma * b))
    )
    fifth_term = -(b - 1.0) / (1.0 - gamma * b)

    return gas_conductivity * (
        first_term + second_term + third_term * (fourth_term + fifth_term)
    )


def specchia_stagnant_conductivity(
    eps_flow: float,
    gas_conductivity: float,
    particle_conductivity: float,
) -> float:
    """Return stagnant conductivity from the Specchia-Baldi-Sicardi law.

    This is the static term of the effective bed conductivity used by the
    Zanco et al. breakthrough reference, whose Eq. (14) reads

        lambda_0 = lambda_g [eps + (1 - eps) / (0.22 eps^2 + (2/3) lambda_g/lambda_p)]

    The correlation is:

        V. Specchia, G. Baldi & S. Sicardi, ``Heat transfer in packed bed
        reactors with one phase flow``, Chemical Engineering Communications,
        4(1) (1980), 361-380. doi:10.1080/00986448008935916.

    It predicts a markedly higher stagnant conductivity than the modified
    Zehner-Schlunder law at the same pellet conductivity (about +32 % at
    eps = 0.388 and lambda_p = 0.17 W/(m K)), which is the documented
    correlation spread rather than a difference in the bed.
    """
    conductivity_ratio = gas_conductivity / particle_conductivity
    resistance = 0.22 * eps_flow**2 + (2.0 / 3.0) * conductivity_ratio
    return gas_conductivity * (eps_flow + (1.0 - eps_flow) / resistance)


STAGNANT_CONDUCTIVITY_LAWS = {
    "zehner_schlunder": zehner_schlunder_stagnant_conductivity,
    "specchia_1980": specchia_stagnant_conductivity,
}


def axial_effective_conductivity(
    lambda_0: float,
    reynolds: float,
    prandtl: float,
    gas_conductivity: float,
    reference_velocity: float,
    bed_length: float,
) -> tuple[float, float]:
    """Return axial effective conductivity and its dimensionless coefficient.

    The axial effective conductivity uses the Tsotsas-Schlunder correlation:

        E. Tsotsas & E. U. Schlunder, ``On axial dispersion in packed beds
        with fluid flow: Über die axiale dispersion in durchströmten
        festbetten``, Chemical Engineering and Processing: Process
        Intensification, 24(1) (1988), 15-31.
        doi:10.1016/0255-2701(88)87002-8.
    """
    lambda_ax = lambda_0 + (reynolds * prandtl / 2.0) * gas_conductivity
    inv_pe_h_ax = lambda_ax / reference_velocity / bed_length
    return lambda_ax, inv_pe_h_ax


def radial_effective_conductivity(
    eps_flow: float,
    lambda_0: float,
    gas_conductivity: float,
    reynolds: float,
    prandtl: float,
    particle_diameter: float,
    inlet_diameter: float,
) -> tuple[float, dict[str, float]]:
    """Return the radial effective conductivity and its diagnostics.

    The convective radial contribution uses the constant ``Pe_inf`` closure
    from:

        A. G. Dixon, ``Fixed bed catalytic reactor modelling-the radial
        heat transfer problem``, Canadian Journal of Chemical Engineering,
        90(3) (2012), 507-527. doi:10.1002/CJCE.21630.

    It is Dixon's Eq. (6) with (alpha beta) = 1/Pe_inf, Pe_inf from his Eq. (13)
    (Specchia et al. 1980). The molecular term 2 eps/(3 Pe_h) of his radial
    Peclet interpolation, Eq. (7), is not added: the stagnant conductivity
    ``lambda_0`` already accounts for conduction through the fluid, so
    ``lambda_rad = lambda_0`` at no flow.
    """
    n_geometry = inlet_diameter / particle_diameter
    pe_inf = 8.65 * (1.0 + 19.4 / n_geometry**2)
    pe_h = reynolds * prandtl
    lambda_rad_conv = pe_h / pe_inf * gas_conductivity
    lambda_rad = lambda_0 + lambda_rad_conv

    return lambda_rad, {"lambda_rad_conv": lambda_rad_conv}


def wall_heat_transfer_coefficient(
    lambda_0: float,
    gas_conductivity: float,
    reynolds: float,
    prandtl: float,
    particle_diameter: float,
    inlet_diameter: float,
) -> tuple[float, dict[str, float]]:
    """Return the packed-bed wall HTC and its Nusselt diagnostics.

    The wall Nusselt number and its convective contribution follow Yagi-Kunii:

        S. Yagi & D. Kunii, ``Studies on heat transfer near wall surface
        in packed beds``, AIChE Journal, 6(1) (1960), 97-104.
        doi:10.1002/aic.690060119.

    The stagnant contribution to the wall Nusselt number follows
    Martin-Nilles:

        H. Martin & M. Nilles, ``Radiale Wärmeleitung in durchströmten
        Schüttungsrohren``, Chemie Ingenieur Technik, 65(12) (1993),
        1468-1477. doi:10.1002/CITE.330651206.

    The gas-film contribution follows Tobis-Ziolkowski:

        J. Tobis & D. Ziolkowski, ``Modelling of heat transfer at the
        wall of a packed-bed apparatus``, Chemical Engineering Science,
        43(11) (1988), 3031-3036. doi:10.1016/0009-2509(88)80056-3.

    The complete radial wall-transfer closure is documented by Dixon:

        A. G. Dixon, ``Fixed bed catalytic reactor modelling-the radial
        heat transfer problem``, Canadian Journal of Chemical Engineering,
        90(3) (2012), 507-527. doi:10.1002/CJCE.21630.
    """
    n_geometry = inlet_diameter / particle_diameter
    nu_w0 = (1.3 + 5.0 / n_geometry) * lambda_0 / gas_conductivity
    nu_m = 0.054 * prandtl * reynolds
    nu_w_star = 0.3 * prandtl ** (1.0 / 3.0) * reynolds**0.75
    if reynolds > 1.0e-16:
        nu_w_convection = 1.0 / (1.0 / nu_w_star + 1.0 / nu_m)
    else:
        nu_w_convection = 0.0
    nu_w = nu_w0 + nu_w_convection
    h_wall = nu_w * gas_conductivity / particle_diameter

    diagnostics = {
        "Nu_w0": nu_w0,
        "Nu_m": nu_m,
        "Nu_w_star": nu_w_star,
        "Nu_w_convection": nu_w_convection,
        "Nu_w": nu_w,
    }
    return h_wall, diagnostics


def specchia_wall_heat_transfer_coefficient(
    eps_flow: float,
    gas_conductivity: float,
    particle_conductivity: float,
    reynolds: float,
    prandtl: float,
    particle_diameter: float,
    inlet_diameter: float,
) -> tuple[float, dict[str, float]]:
    """Return the Specchia-Baldi-Sicardi packed-bed wall HTC.

    The closure is the sum of the static and convective terms in Zanco et al.,
    Chem. Eng. J. 430 (2022) 131000, Eqs. (9)-(12), based on Specchia et al.
    (1980), doi:10.1080/00986448008935916.  The static denominator uses
    ``A + 2/3 * (k_g/k_p)`` exactly as printed in Zanco Eq. (10); this is the
    transcription used here despite the visually compressed PDF layout.

    The ``Nu_m`` and ``Nu_w_star`` diagnostics retain the legacy keys expected
    by the column model; for this closure both are the direct Specchia
    convective contribution rather than Dixon's intermediate correlations.
    """
    geometry_ratio = inlet_diameter / particle_diameter
    static_denominator = 0.0024 * geometry_ratio**1.58 + (2.0 / 3.0) * (
        gas_conductivity / particle_conductivity
    )
    nu_w0 = 2.0 * eps_flow + (1.0 - eps_flow) / static_denominator
    if reynolds <= 1200.0:
        convective_prefactor = 0.0835
        reynolds_exponent = 0.91
    else:
        convective_prefactor = 1.23
        reynolds_exponent = 0.53
    nu_w_convection = (
        convective_prefactor
        * reynolds**reynolds_exponent
        * (prandtl / 0.7018) ** (1.0 / 3.0)
    )
    nu_w = nu_w0 + nu_w_convection
    h_wall = nu_w * gas_conductivity / particle_diameter
    diagnostics = {
        "Nu_w0": nu_w0,
        "Nu_m": nu_w_convection,
        "Nu_w_star": nu_w_convection,
        "Nu_w_convection": nu_w_convection,
        "Nu_w": nu_w,
        "h_w_static": nu_w0 * gas_conductivity / particle_diameter,
        "h_w_convection": nu_w_convection * gas_conductivity / particle_diameter,
    }
    return h_wall, diagnostics


WALL_HTC_LAWS = {
    "dixon": wall_heat_transfer_coefficient,
    "specchia_1980": specchia_wall_heat_transfer_coefficient,
}


def dixon_overall_coefficient(
    h_wall: float,
    lambda_rad: float,
    bed_diameter: float,
) -> tuple[float, float, float]:
    """Return the Dixon lumped coefficient, Biot number, and inverse Biot.

    This is the ``U_eq``/overall wall-to-bed closure used when the radial bed
    profile is lumped, with ``Bi_w = h_w R_b / lambda_rad``:

        A. G. Dixon, ``An improved equation for the overall heat
        transfer coefficient in packed beds``, Chemical Engineering and
        Processing: Process Intensification, 35(5) (1996), 323-331.
        doi:10.1016/0255-2701(96)80012-2.
    """
    bed_radius = bed_diameter / 2.0
    bi_wall = h_wall * bed_radius / lambda_rad
    inv_bi_wall = 1.0 / bi_wall
    overall_htc = (
        1.0 / h_wall
        + bed_diameter
        / (6.0 * lambda_rad)
        * (bi_wall + 3.0)
        / (bi_wall + 4.0)
    ) ** (-1)
    return overall_htc, bi_wall, inv_bi_wall


def resolve_packed_bed_thermal_transport(
    *,
    eps_flow: float,
    gas_conductivity: float,
    particle_conductivity: float,
    reynolds: float,
    prandtl: float,
    particle_diameter: float,
    bed_diameter: float,
    reference_velocity: float,
    bed_length: float,
    is_1d: bool,
    stagnant_conductivity_model: str = DEFAULT_STAGNANT_CONDUCTIVITY_MODEL,
    wall_htc_model: str = DEFAULT_WALL_HTC_MODEL,
) -> ResolvedTransport:
    """Resolve the legacy packed-bed thermal closures into a typed result.

    The returned fields preserve the legacy separation between effective
    radial conductivity, wall HTC, and the lumped ``U_eq`` coefficient.

    ``stagnant_conductivity_model`` selects the no-flow bed conductivity law.
    It feeds the axial and radial effective conductivities and the wall Nusselt
    number, so it is the single closure choice behind all three.

    ``wall_htc_model`` selects the bed-to-wall heat-transfer correlation while
    leaving the stagnant-conductivity and effective-radial-conductivity choices
    independent.
    """
    stagnant_conductivity_model = validate_stagnant_conductivity_model(
        stagnant_conductivity_model
    )
    wall_htc_model = validate_wall_htc_model(wall_htc_model)
    lambda_0 = STAGNANT_CONDUCTIVITY_LAWS[stagnant_conductivity_model](
        eps_flow,
        gas_conductivity,
        particle_conductivity,
    )
    lambda_ax, inv_pe_h_ax = axial_effective_conductivity(
        lambda_0,
        reynolds,
        prandtl,
        gas_conductivity,
        reference_velocity,
        bed_length,
    )
    lambda_rad, radial_diagnostics = radial_effective_conductivity(
        eps_flow,
        lambda_0,
        gas_conductivity,
        reynolds,
        prandtl,
        particle_diameter,
        bed_diameter,
    )
    if wall_htc_model == DEFAULT_WALL_HTC_MODEL:
        h_wall, wall_diagnostics = WALL_HTC_LAWS[wall_htc_model](
            lambda_0,
            gas_conductivity,
            reynolds,
            prandtl,
            particle_diameter,
            bed_diameter,
        )
    else:
        h_wall, wall_diagnostics = WALL_HTC_LAWS[wall_htc_model](
            eps_flow,
            gas_conductivity,
            particle_conductivity,
            reynolds,
            prandtl,
            particle_diameter,
            bed_diameter,
        )
    overall_htc, bi_wall, inv_bi_wall = dixon_overall_coefficient(
        h_wall,
        lambda_rad,
        bed_diameter,
    )
    diagnostics = {
        "lambda_0": lambda_0,
        "inv_Pe_h_ax": inv_pe_h_ax,
        "U_dixon": overall_htc,
        "Bi_w": bi_wall,
        "inv_Bi_h": inv_bi_wall,
        **radial_diagnostics,
        **wall_diagnostics,
    }
    wall_bed_htc = overall_htc
    if not is_1d:
        wall_bed_htc = h_wall

    return ResolvedTransport(
        lambda_ax=lambda_ax,
        lambda_rad=lambda_rad,
        h_wall=h_wall,
        wall_bed_htc=wall_bed_htc,
        pressure_drop_model="ergun",
        dispersion_model="current_pydac",
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Public exports, kept in the same human-first order as the source above.
# ---------------------------------------------------------------------------

__all__ = [
    "resolve_packed_bed_structure",
    "solve_quadratic_drag_velocity",
    "ErgunPressureDrop",
    "resolve_packed_bed_pressure_drop",
    "resolve_packed_bed_dispersion",
    "zehner_schlunder_stagnant_conductivity",
    "specchia_stagnant_conductivity",
    "STAGNANT_CONDUCTIVITY_LAWS",
    "axial_effective_conductivity",
    "radial_effective_conductivity",
    "wall_heat_transfer_coefficient",
    "specchia_wall_heat_transfer_coefficient",
    "WALL_HTC_LAWS",
    "dixon_overall_coefficient",
    "resolve_packed_bed_thermal_transport",
]
