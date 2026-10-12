"""Radial voidage structure and wall-channeling closure for packed beds.

This module owns the *reduced* wall-channeling closure: a radial voidage
profile, its exact cylindrical ring means for storage, the fixed physical drag
layers on which the Ergun resistance is evaluated, and the inlet allocation
derived from that same closure.  It is deliberately independent of
``ColumnModelMoL``, of any solver and of any case: geometry, the declared mean
voidage and the gas state are arguments.

Two porosity fields, one profile, never interchanged
----------------------------------------------------
STORAGE uses the exact cylindrical mean of ``eps(r)`` over each PDE ring.  That
is the honest inventory of a resolved profile and it keeps the total gas
holdup and sorbent mass equal to the uniform-voidage bed.

DRAG uses a *different* field.  Evaluating Ergun on the raw ring mean does not
converge: de Klerk's profile reaches ``eps = 1`` exactly at the wall, where
both Ergun coefficients vanish, so the wall ring's resistance collapses as the
grid is refined (measured: ``a_wall/a_core`` 0.164 at Nr=40, 0.0010 at Nr=320,
and the outer-5 mm flow share drifts 33.2% -> 44.9%).  There is no pointwise
continuum limit to converge to.  The drag field is therefore built on
wall-anchored physical layers of declared thickness ``W`` (default one particle
diameter), whose cylindrical means are projected onto the PDE rings by exact
area overlap.  ``W`` is a modelling length that regularises a genuine
singularity, not a mesh parameter, and it belongs to the case identity.

What this closure does not do
-----------------------------
It changes the axial resistance with radius, so ``u_z(z, r, t)`` develops wall
channelling.  It introduces no ``du_z/dr`` term, no radial convection and no
automatic lateral redistribution of mass.  Empirical dispersion, conductivity
and wall-HTC closures keep their declared mean-voidage, step-bulk form.

Sources
-------
- A. de Klerk, "Voidage variation in packed beds at small column to particle
  diameter ratio", AIChE Journal 49(8) (2003) 2022-2029. Eq. 3a/3b with
  ``z = (R - r)/d_p``; ``eps_b`` is defined as the voidage *in the absence of
  wall effects*, which is why it is solved for rather than set to the measured
  mean.  Declared limitations: the correlation was validated at
  ``D/d_p = 2.02-20.3``, it carries ~11% mean relative error on the radial
  voidage, and its two published branches do not join at ``z = 0.637`` (they
  are continuous only at ``eps_b = 0.3609``).  That discontinuity belongs to
  the published model and is reported, not smoothed.
- S. Ergun, "Fluid flow through packed columns", Chem. Eng. Prog. 48(2) (1952)
  89-94, with the conventional constants ``alpha = 150`` and ``beta = 1.75``.
- W. Giese, "Stroemung in porioesen Medien unter Beruecksichtigung effektiver
  Viskositaeten", dissertation (1998), as transcribed by M. Gaeini, A. L. Rohde,
  H. A. Zondag et al., Int. J. Heat Mass Transfer 113 (2017) 1116-1129, Eq. 2:
  ``eps(r) = eps_c [1 + 1.36 exp(-5 (R - r)/d_p)]``.  A smooth single-branch
  alternative to de Klerk with no oscillation and, at this geometry, no
  singularity: its wall value is ``2.36 eps_c``, short of 1, so the Ergun
  coefficients do not vanish at the wall.  The drag-layer treatment is applied
  unchanged all the same, so that switching profile changes the profile alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq

from pydac.core.bed_architectures.packed_bed import solve_quadratic_drag_velocity

from pydac.core.bed_architectures.base import (
    ChannelingOptions,
    PackedBedDesign,
    ParticleProperties,
)


# Bump when the generated structure changes numerically.  It is reported in the
# resolved runtime metadata so a saved result names the generator that produced
# its radial fields.
CHANNELING_GENERATOR_REVISION = "2026-09-18.2"

_DE_KLERK_BRANCH_X = 0.637
_MEAN_POROSITY_TOLERANCE = 1.0e-8

# The de Klerk profile is analytic on each of its two branches, so a composite
# Gauss-Legendre rule that splits at the branch radius integrates it to machine
# precision.  An adaptive quadrature reaches the same answer but reports
# roundoff on the oscillatory outer branch; a fixed rule is also deterministic,
# which matters because the core voidage is solved through these integrals.
_QUADRATURE_ORDER = 20
_QUADRATURE_STEPS_PER_DP = 4
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(_QUADRATURE_ORDER)


# ---------------------------------------------------------------------------
# 1. Radial voidage profiles
# ---------------------------------------------------------------------------

def area_mean(values: Any, r_faces: Any) -> float:
    """Exact cylindrical area mean of a ring-wise constant field."""
    values = np.asarray(values, dtype=float)
    r_faces = np.asarray(r_faces, dtype=float)
    weights = r_faces[1:] ** 2 - r_faces[:-1] ** 2
    return float(np.sum(values * weights) / np.sum(weights))


class DeKlerkRadialPorosity:
    """de Klerk (2003) near-wall voidage, normalised to the measured mean.

    The correlation's own core parameter ``eps_b`` is the voidage away from the
    wall, so it is solved such that the *exact* cylindrical mean of the profile
    equals the declared bed mean.  Taking the measured mean literally as the
    core value instead would put the profile mean 1.5 percentage points high
    and break the inventory contract.
    """

    name = "de_klerk"

    def __init__(
        self,
        *,
        radius: float,
        particle_diameter: float,
        mean_porosity: float,
    ) -> None:
        self.radius = float(radius)
        self.particle_diameter = float(particle_diameter)
        self.mean_porosity = float(mean_porosity)
        self.core_porosity = float(
            brentq(
                lambda core: self._mean(core) - self.mean_porosity,
                0.20,
                0.60,
                xtol=1e-15,
                rtol=1e-15,
            )
        )
        self.wall_value = float(self(np.array([self.radius]))[0])

    def _raw(self, r: Any, core: float) -> np.ndarray:
        x = (self.radius - np.asarray(r, dtype=float)) / self.particle_diameter
        near = 2.14 * x**2 - 2.53 * x + 1.0
        far = (
            core
            + 0.29 * np.exp(-0.6 * x) * np.cos(2.3 * np.pi * (x - 0.16))
            + 0.15 * np.exp(-0.9 * x)
        )
        return np.clip(np.where(x <= _DE_KLERK_BRANCH_X, near, far), 0.2, 1.0)

    def _panel_edges(self, r1: float, r2: float) -> np.ndarray:
        """Quadrature panels, split at the branch radius and refined per d_p."""
        r_break = self.radius - _DE_KLERK_BRANCH_X * self.particle_diameter
        edges = [r1, r2]
        if r1 < r_break < r2:
            edges.append(r_break)
        step = self.particle_diameter / _QUADRATURE_STEPS_PER_DP
        refined = [np.array([r1, r2])]
        bounds = np.unique(edges)
        for low, high in zip(bounds[:-1], bounds[1:]):
            count = max(1, int(np.ceil((high - low) / step)))
            refined.append(np.linspace(low, high, count + 1))
        return np.unique(np.concatenate(refined))

    def _ring_mean(self, r1: float, r2: float, core: float) -> float:
        total = 0.0
        edges = self._panel_edges(r1, r2)
        for low, high in zip(edges[:-1], edges[1:]):
            mid, half = 0.5 * (high + low), 0.5 * (high - low)
            r = mid + half * _GAUSS_NODES
            total += half * float(
                np.sum(_GAUSS_WEIGHTS * r * self._raw(r, core))
            )
        return total * 2.0 / (r2**2 - r1**2)

    def _mean(self, core: float) -> float:
        return self._ring_mean(0.0, self.radius, core)

    def __call__(self, r: Any) -> np.ndarray:
        return self._raw(r, self.core_porosity)

    def ring_means(self, r_faces: Any) -> np.ndarray:
        """Exact cylindrical mean of the profile over each ring."""
        r_faces = np.asarray(r_faces, dtype=float)
        return np.array(
            [
                self._ring_mean(r_faces[i], r_faces[i + 1], self.core_porosity)
                for i in range(r_faces.size - 1)
            ]
        )


class GieseRadialPorosity:
    """Giese (1998) exponential near-wall voidage, normalised to the measured mean.

    ``eps(r) = eps_c [1 + 1.36 exp(-5 (R - r) / d_p)]``, the form Gaeini et al.
    (2017) use as their Eq. 2.  As for de Klerk, ``eps_c`` is the voidage away
    from the wall and is solved so the *exact* cylindrical mean of the profile
    equals the declared bed mean, which keeps the inventory contract.

    The ring means are closed-form: with ``a = 5 / d_p``,
    ``int r exp(-a(R-r)) dr = exp(-a(R-r)) (r/a - 1/a^2)``.  No quadrature is
    used, so the ring means are exact at any grid, and a test pins them against
    the same Gauss-Legendre rule the de Klerk profile uses.
    """

    name = "giese"

    _AMPLITUDE = 1.36
    _DECAY_PER_DP = 5.0

    def __init__(
        self,
        *,
        radius: float,
        particle_diameter: float,
        mean_porosity: float,
    ) -> None:
        self.radius = float(radius)
        self.particle_diameter = float(particle_diameter)
        self.mean_porosity = float(mean_porosity)
        self._decay = self._DECAY_PER_DP / self.particle_diameter
        # The profile is linear in the core value, so the normalisation is a
        # division rather than a root find; brentq is kept out on purpose.
        self.core_porosity = self.mean_porosity / self._shape_mean(0.0, self.radius)
        self.wall_value = float(self(np.array([self.radius]))[0])
        if not 0.0 < self.wall_value <= 1.0:
            raise ValueError(
                "The Giese profile gives a wall voidage of "
                f"{self.wall_value:.4f} at this geometry, which is outside "
                "(0, 1]. The correlation is not usable here."
            )

    def _shape(self, r: Any) -> np.ndarray:
        """Profile divided by the core value: 1 + 1.36 exp(-5 (R - r)/d_p)."""
        r = np.asarray(r, dtype=float)
        return 1.0 + self._AMPLITUDE * np.exp(-self._decay * (self.radius - r))

    def _shape_moment(self, r: float) -> float:
        """Antiderivative of ``r * shape(r)``, in closed form."""
        a = self._decay
        return 0.5 * r**2 + self._AMPLITUDE * np.exp(-a * (self.radius - r)) * (
            r / a - 1.0 / a**2
        )

    def _shape_mean(self, r1: float, r2: float) -> float:
        """Exact cylindrical mean of ``shape`` over the ring ``[r1, r2]``."""
        return (
            2.0
            * (self._shape_moment(r2) - self._shape_moment(r1))
            / (r2**2 - r1**2)
        )

    def __call__(self, r: Any) -> np.ndarray:
        return self.core_porosity * self._shape(r)

    def ring_means(self, r_faces: Any) -> np.ndarray:
        """Exact cylindrical mean of the profile over each ring."""
        r_faces = np.asarray(r_faces, dtype=float)
        return np.array(
            [
                self.core_porosity * self._shape_mean(r_faces[i], r_faces[i + 1])
                for i in range(r_faces.size - 1)
            ]
        )


class TabulatedRadialPorosity:
    """User-supplied ``(r/R, eps)`` profile, integrated exactly per ring.

    The supplied profile is piecewise linear between its points, so the
    cylindrical ring integrals are closed-form.  The profile's own exact mean
    must already match the declared ``eps_b``: a mismatch is rejected rather
    than absorbed by shifting the physics the user supplied.
    """

    name = "tabulated"

    def __init__(
        self,
        *,
        radius: float,
        mean_porosity: float,
        profile: Any,
    ) -> None:
        points = np.asarray(profile, dtype=float)
        self.radius = float(radius)
        self.mean_porosity = float(mean_porosity)
        self.r_nodes = points[:, 0] * self.radius
        self.eps_nodes = points[:, 1]
        self.core_porosity = float(self.eps_nodes[0])
        self.wall_value = float(self.eps_nodes[-1])
        mean = self._ring_mean(0.0, self.radius)
        if abs(mean - self.mean_porosity) > _MEAN_POROSITY_TOLERANCE:
            raise ValueError(
                "`radial_porosity_profile` exact cylindrical mean "
                f"{mean:.12f} differs from `eps_b` {self.mean_porosity:.12f} by "
                f"{mean - self.mean_porosity:+.3e}; supply a profile whose mean "
                "matches, rather than having it shifted."
            )

    def __call__(self, r: Any) -> np.ndarray:
        return np.interp(
            np.asarray(r, dtype=float), self.r_nodes, self.eps_nodes
        )

    def _ring_mean(self, r1: float, r2: float) -> float:
        """Exact ``2/(r2^2-r1^2) * int r eps(r) dr`` for a piecewise-linear eps."""
        edges = np.unique(
            np.concatenate(
                ([r1, r2], self.r_nodes[(self.r_nodes > r1) & (self.r_nodes < r2)])
            )
        )
        total = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            eps_a = float(self(np.array([a]))[0])
            eps_b = float(self(np.array([b]))[0])
            # eps is linear on [a, b]: eps(r) = eps_a + slope * (r - a).
            slope = 0.0 if b == a else (eps_b - eps_a) / (b - a)
            intercept = eps_a - slope * a
            total += (
                intercept * (b**2 - a**2) / 2.0
                + slope * (b**3 - a**3) / 3.0
            )
        return total * 2.0 / (r2**2 - r1**2)

    def ring_means(self, r_faces: Any) -> np.ndarray:
        r_faces = np.asarray(r_faces, dtype=float)
        return np.array(
            [
                self._ring_mean(r_faces[i], r_faces[i + 1])
                for i in range(r_faces.size - 1)
            ]
        )


# ---------------------------------------------------------------------------
# 2. Fixed physical drag layers and their exact projection
# ---------------------------------------------------------------------------

def drag_layer_faces(radius: float, width: float) -> np.ndarray:
    """Wall-anchored physical layer boundaries of declared thickness ``width``.

    Boundaries are laid from the wall inward.  The remaining central interval
    stays its own layer when it is at least half a layer wide, and is otherwise
    merged into the innermost full layer, so no degenerate sliver is created.
    The partition depends only on geometry and ``width``: the same wall layer is
    the same physical interval on every numerical grid.
    """
    radius = float(radius)
    width = float(width)
    if not radius > 0.0 or not width > 0.0:
        raise ValueError("`radius` and `width` must both be positive.")
    if width > radius:
        raise ValueError(
            f"Drag layer width {width} m exceeds the bed radius {radius} m."
        )
    count = int(np.floor(radius / width))
    if radius - count * width < 0.5 * width:
        count -= 1
    faces = [radius - k * width for k in range(count, -1, -1)]
    return np.concatenate(([0.0], np.asarray(faces, dtype=float)))


def project_layer_field(
    layer_values: Any,
    layer_faces: Any,
    r_faces: Any,
) -> np.ndarray:
    """Exact cylindrical area-overlap projection of a piecewise-constant field.

    A PDE ring straddling a layer boundary receives the area-weighted blend of
    the layers it overlaps, so the projected field's area mean equals the layer
    field's to machine precision on *any* grid.  Mean conservation alone does
    not make the hydraulics grid-invariant, because Ergun evaluated on a
    blended voidage is not the blend of the Ergun responses; the residual grid
    sensitivity is measured (0.10 percentage points in outer-5 mm flow share
    over Nr = 25..640), not assumed away.

    The blend is also what keeps the drag away from the singularity on a
    nonconforming grid: an area-weighted mean of values far from one stays far
    from one.
    """
    layer_values = np.asarray(layer_values, dtype=float)
    layer_faces = np.asarray(layer_faces, dtype=float)
    r_faces = np.asarray(r_faces, dtype=float)
    out = np.empty(r_faces.size - 1)
    for i in range(out.size):
        low, high = r_faces[i], r_faces[i + 1]
        left = np.maximum(layer_faces[:-1], low)
        right = np.minimum(layer_faces[1:], high)
        overlap = np.maximum(right**2 - left**2, 0.0)
        out[i] = float(np.sum(layer_values * overlap) / np.sum(overlap))
    return out


# ---------------------------------------------------------------------------
# 3. Resolved channeling structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class ChannelingStructure:
    """Resolved radial bed structure for one grid.

    ``eps_storage`` and ``eps_drag`` are deliberately separate fields: the
    first is the inventory of the resolved profile, the second is the
    regularised field the momentum closure may be evaluated on.  Nothing in
    this package substitutes one for the other.
    """

    porosity_model: str
    drag_layer_width_dp: float
    drag_layer_width: float
    core_porosity: float
    wall_porosity: float
    mean_porosity: float
    r_faces: np.ndarray
    eps_storage: np.ndarray
    eps_drag: np.ndarray
    layer_faces: np.ndarray
    layer_porosity: np.ndarray
    gas_holdup_factor: np.ndarray
    sorbent_factor: np.ndarray
    generator_revision: str = CHANNELING_GENERATOR_REVISION

    def to_context_dict(self) -> dict[str, Any]:
        """Bounded, JSON-safe summary; full arrays belong to run artifacts."""
        return {
            "porosity_model": self.porosity_model,
            "drag_layer_width_dp": self.drag_layer_width_dp,
            "drag_layer_width_m": self.drag_layer_width,
            "drag_layer_count": int(self.layer_porosity.size),
            "core_porosity": self.core_porosity,
            "wall_porosity": self.wall_porosity,
            "mean_porosity": self.mean_porosity,
            "storage_area_mean": area_mean(self.eps_storage, self.r_faces),
            "drag_area_mean": area_mean(self.eps_drag, self.r_faces),
            "storage_min": float(np.min(self.eps_storage)),
            "storage_max": float(np.max(self.eps_storage)),
            "drag_min": float(np.min(self.eps_drag)),
            "drag_max": float(np.max(self.eps_drag)),
            "wall_layer_porosity": float(self.layer_porosity[-1]),
            "generator_revision": self.generator_revision,
        }


def resolve_radial_porosity_profile(
    design: PackedBedDesign,
    *,
    radius: float,
    particle_diameter: float,
):
    """Return the radial voidage profile declared by one packed-bed design."""
    options = design.channeling
    if not isinstance(options, ChannelingOptions):
        raise ValueError(
            "A channeling flow model requires resolved `channeling` options."
        )
    if options.porosity_model == "de_klerk":
        return DeKlerkRadialPorosity(
            radius=radius,
            particle_diameter=particle_diameter,
            mean_porosity=design.eps_b,
        )
    if options.porosity_model == "giese":
        return GieseRadialPorosity(
            radius=radius,
            particle_diameter=particle_diameter,
            mean_porosity=design.eps_b,
        )
    return TabulatedRadialPorosity(
        radius=radius,
        mean_porosity=design.eps_b,
        profile=design.radial_porosity_profile,
    )


def resolve_channeling_structure(
    design: PackedBedDesign,
    particle: ParticleProperties,
    *,
    radius: float,
    particle_diameter: float,
    r_faces: Any,
) -> ChannelingStructure:
    """Resolve the storage and drag voidage fields on one radial grid.

    ``r_faces`` are dimensional radial cell faces in metres, from 0 to
    ``radius``.
    """
    r_faces = np.asarray(r_faces, dtype=float)
    if r_faces.ndim != 1 or r_faces.size < 3:
        raise ValueError("`r_faces` must be a 1-D grid with at least two rings.")
    if not np.isclose(r_faces[0], 0.0) or not np.isclose(r_faces[-1], radius):
        raise ValueError("`r_faces` must span the full bed radius from 0 to R.")

    options = design.channeling
    profile = resolve_radial_porosity_profile(
        design, radius=radius, particle_diameter=particle_diameter
    )

    eps_storage = profile.ring_means(r_faces)
    storage_mean = area_mean(eps_storage, r_faces)
    if abs(storage_mean - design.eps_b) > _MEAN_POROSITY_TOLERANCE:
        raise ValueError(
            "Resolved storage voidage has exact area mean "
            f"{storage_mean:.12f} against `eps_b` {design.eps_b:.12f}; the "
            "inventory contract requires them to agree."
        )

    width = options.drag_layer_width_dp * particle_diameter
    layer_faces = drag_layer_faces(radius, width)
    layer_porosity = profile.ring_means(layer_faces)
    eps_drag = project_layer_field(layer_porosity, layer_faces, r_faces)
    if not np.all((eps_drag > 0.0) & (eps_drag < 1.0)):
        raise ValueError(
            "Projected drag voidage left the open interval (0, 1); the drag "
            "closure would be singular on this grid."
        )

    eps_total_mean = design.eps_b + (1.0 - design.eps_b) * particle.porosity
    gas = (
        eps_storage + (1.0 - eps_storage) * particle.porosity
    ) / eps_total_mean
    solid = (1.0 - eps_storage) / (1.0 - design.eps_b)
    if np.any(solid <= 0.0):
        raise ValueError("The resolved voidage leaves a ring without sorbent.")

    return ChannelingStructure(
        porosity_model=options.porosity_model,
        drag_layer_width_dp=options.drag_layer_width_dp,
        drag_layer_width=float(width),
        core_porosity=float(profile.core_porosity),
        wall_porosity=float(profile.wall_value),
        mean_porosity=float(design.eps_b),
        r_faces=r_faces,
        eps_storage=eps_storage,
        eps_drag=eps_drag,
        layer_faces=layer_faces,
        layer_porosity=layer_porosity,
        gas_holdup_factor=gas,
        sorbent_factor=solid,
    )


def validate_channeling_support(
    flow_model: str,
    *,
    architecture: str,
    is_1d: bool,
    Nr: int,
    Nz: int,
) -> None:
    """Reject grids and architectures the channeling closure does not cover.

    The closure resolves a radial structure, so it is packed-bed 2-D only.  An
    unsupported combination is an error: silently averaging a radial model onto
    one ring would report a channeling prediction that was never computed.
    """
    if architecture != "packed_bed":
        raise ValueError(
            f"`flow_model={flow_model!r}` is a packed-bed closure; the "
            f"{architecture!r} architecture has no radial voidage structure."
        )
    if is_1d or int(Nr) < 2:
        raise ValueError(
            f"`flow_model={flow_model!r}` resolves a radial profile and needs "
            f"Nr >= 2; received Nr={Nr}. No uniform fallback was applied."
        )
    if int(Nz) < 3:
        raise ValueError(
            f"`flow_model={flow_model!r}` needs Nz >= 3 for the axial "
            f"pressure-gradient stencil; received Nz={Nz}."
        )


# ---------------------------------------------------------------------------
# 4. Inlet allocation from the same closure
# ---------------------------------------------------------------------------

def ergun_drag_coefficients(
    eps: Any,
    *,
    viscosity: float,
    density: float,
    particle_diameter: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dimensional Ergun coefficients ``a`` (viscous) and ``b`` (inertial)."""
    eps = np.asarray(eps, dtype=float)
    viscous = (
        150.0 * viscosity * (1.0 - eps) ** 2 / (eps**3 * particle_diameter**2)
    )
    inertial = 1.75 * density * (1.0 - eps) / (eps**3 * particle_diameter)
    return viscous, inertial


def channeling_inlet_flux_weights(
    structure: ChannelingStructure,
    *,
    viscosity: float,
    density: float,
    particle_diameter: float,
    mean_velocity: float,
) -> np.ndarray:
    """Ring-wise relative inlet flux, from the reduced closure itself.

    The feed is allocated by the same local Ergun resistance the interior uses,
    under one common axial gradient chosen so the integrated flow matches the
    declared total.  The weights therefore have an exact area mean of one and
    the integrated molar, species and enthalpy feed is unchanged; only its
    radial distribution follows the resolved bed structure.

    The allocation is deliberately *not* imported from an offline Brinkman
    solve: the inlet and the interior must be one closure, or the profile and
    the resistance it enters would be two unrelated models.
    """
    mean_velocity = float(mean_velocity)
    if not np.isfinite(mean_velocity):
        raise ValueError("Inlet allocation requires a finite mean velocity.")
    if mean_velocity == 0.0:
        raise ValueError(
            "A zero total feed has no hydraulic allocation to solve; skip the "
            "normalization instead of calling this."
        )
    for name, value in (("viscosity", viscosity), ("density", density),
                        ("particle_diameter", particle_diameter)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Inlet allocation requires finite positive {name}.")

    viscous, inertial = ergun_drag_coefficients(
        structure.eps_drag,
        viscosity=viscosity,
        density=density,
        particle_diameter=particle_diameter,
    )
    if not (np.all(np.isfinite(viscous)) and np.all(viscous > 0.0)
            and np.all(np.isfinite(inertial)) and np.all(inertial > 0.0)):
        raise ValueError("Inlet allocation requires finite positive drag coefficients.")

    # Solve for relative speed w=u/abs(mean_velocity), rather than a dimensional
    # gradient that vanishes with the feed. Dividing Ergun by a_max*abs(U) gives
    # A*w + B*w**2 = x, with area_mean(w)=1. The same positive weights serve
    # either feed direction; its sign remains in the imposed total flux.
    a_scale = float(np.max(viscous))
    linear = viscous / a_scale
    with np.errstate(over="ignore", under="ignore"):
        quadratic = (inertial / a_scale) * abs(mean_velocity)
        upper = float(np.max(linear + quadratic))
    if not np.isfinite(upper) or not np.all(np.isfinite(quadratic)):
        raise ValueError("Inlet allocation coefficients exceed the floating-point range.")

    def weights_at(gradient):
        return solve_quadratic_drag_velocity(gradient, linear, quadratic)

    def residual(gradient):
        return area_mean(weights_at(gradient), structure.r_faces) - 1.0

    # At x=0 every speed is zero; at max(A+B) every speed is at least one.
    # A slightly expanded upper endpoint protects the sign change from roundoff.
    upper *= 1.0 + 1e-12
    if not np.isfinite(upper):
        raise ValueError("Inlet allocation bracket exceeds the floating-point range.")
    gradient = brentq(residual, 0.0, upper, xtol=np.nextafter(0.0, 1.0),
                      rtol=1e-14, maxiter=256)
    weights = weights_at(gradient)
    mean = area_mean(weights, structure.r_faces)
    if (not np.all(np.isfinite(weights)) or np.any(weights <= 0.0)
            or not np.isfinite(mean) or abs(mean - 1.0) > 1e-10):
        raise ValueError("Inlet allocation failed its finite, positive, unit-feed check.")
    return weights / mean


__all__ = [
    "CHANNELING_GENERATOR_REVISION",
    "ChannelingStructure",
    "DeKlerkRadialPorosity",
    "GieseRadialPorosity",
    "TabulatedRadialPorosity",
    "area_mean",
    "channeling_inlet_flux_weights",
    "drag_layer_faces",
    "ergun_drag_coefficients",
    "project_layer_field",
    "resolve_channeling_structure",
    "resolve_radial_porosity_profile",
    "validate_channeling_support",
]
