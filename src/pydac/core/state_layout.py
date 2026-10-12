"""Generate pyDAC state-vector slices from physical names.

An engineer declares a mixture, its closure species, the adsorbates, and the
grid. :class:`StateLayout` then generates every field and offset; nobody assigns
numerical state indices by hand.

Example::

    mixture = Mixture(("CO2", "H2O", "O2", "N2"), closure="N2")
    layout = StateLayout.build(
        mixture=mixture,
        adsorbates=("CO2", "H2O"),
        nz=100,
        nr=1,
        scalar_states=CURRENT_DAC_SCALARS,
    )
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from numbers import Integral

import numpy as np


LAYOUT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class Mixture:
    """Ordered gas species and the one species reconstructed by closure."""

    species: tuple[str, ...]
    closure: str

    def __post_init__(self) -> None:
        species = _names(self.species, "species")
        closure = _name(self.closure, "closure")
        if closure not in species:
            raise ValueError(f"Closure species {closure!r} is not in {species}.")
        if len(species) < 2:
            raise ValueError(
                "A mixture needs a closure species and at least one independent species."
            )
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "closure", closure)

    @property
    def independent_species(self) -> tuple[str, ...]:
        return tuple(species for species in self.species if species != self.closure)


@dataclass(frozen=True, slots=True)
class ScalarState:
    """Named integrated scalar, such as accumulated mass or energy."""

    name: str
    unit: str
    category: str = "other"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "name"))
        object.__setattr__(self, "unit", _name(self.unit, "unit"))
        object.__setattr__(self, "category", _name(self.category, "category"))


@dataclass(frozen=True, slots=True)
class Field:
    """One named, contiguous field in the flat numerical state."""

    name: str
    kind: str
    shape: tuple[int, ...]
    start: int
    stop: int
    unit: str
    species: str | None = None

    @property
    def state_slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def storage(self) -> str:
        if self.kind == "wall temperature":
            return "y_1d"
        if self.kind.endswith(" scalar"):
            return "y_scalars"
        return "y_2d"

    @property
    def scale_source(self) -> str:
        """Describe where the model obtains this field's dimensional scale."""
        if self.kind == "gas concentration":
            return "gas_reference_concentration"
        if self.kind == "solid loading":
            return "adsorbate_reference_loading"
        if self.kind == "total concentration":
            return "total_reference_concentration"
        if self.kind in ("bed temperature", "wall temperature"):
            return "reference_temperature"
        return "identity"

    @property
    def constraint(self) -> str:
        """Return the CVODE constraint applied to this numerical field."""
        if self.kind in (
            "solid loading",
            "total concentration",
            "bed temperature",
            "wall temperature",
        ):
            return "nonnegative"
        return "none"

    @property
    def tolerance_group(self) -> str:
        """Return the solver absolute-tolerance policy for this field."""
        if self.kind == "gas concentration":
            return "trace_gas"
        if self.kind == "mass scalar":
            return "mass_scalar"
        if self.kind == "energy scalar":
            return "energy_scalar"
        return "default"


@dataclass(frozen=True, slots=True)
class StateLayout:
    """Generated packing contract for one mixture and spatial grid."""

    mixture: Mixture
    adsorbates: tuple[str, ...]
    nz: int
    nr: int
    scalar_states: tuple[ScalarState, ...] = ()
    fields: tuple[Field, ...] = field(init=False, repr=False)

    @classmethod
    def build(
        cls,
        *,
        mixture: Mixture,
        adsorbates: Iterable[str],
        nz: int,
        nr: int,
        scalar_states: Iterable[ScalarState] = (),
    ) -> "StateLayout":
        """Build from physical names; all indices are generated."""
        return cls(
            mixture=mixture,
            adsorbates=tuple(adsorbates),
            nz=nz,
            nr=nr,
            scalar_states=tuple(scalar_states),
        )

    @classmethod
    def from_dict(cls, document: Mapping[str, object]) -> "StateLayout":
        """Rebuild and verify the layout persisted beside a saved solution.

        Deserialization never trusts saved offsets.  It rebuilds them from the
        physical names and grid, then requires the complete generated document
        and signature to match the saved report exactly.
        """
        if not isinstance(document, Mapping):
            raise TypeError("Saved state layout must be a mapping.")

        try:
            mixture_document = document["mixture"]
            grid_document = document["grid"]
            scalar_documents = document["scalar_states"]
        except KeyError as exc:
            raise ValueError(
                f"Saved state layout is missing required field {exc.args[0]!r}."
            ) from exc
        if not isinstance(mixture_document, Mapping):
            raise TypeError("Saved state layout `mixture` must be a mapping.")
        if not isinstance(grid_document, Mapping):
            raise TypeError("Saved state layout `grid` must be a mapping.")
        if not isinstance(scalar_documents, list):
            raise TypeError("Saved state layout `scalar_states` must be a list.")

        try:
            mixture = Mixture(
                tuple(mixture_document["species"]),
                closure=mixture_document["closure"],
            )
            scalars = tuple(
                ScalarState(
                    name=scalar["name"],
                    unit=scalar["unit"],
                    category=scalar["category"],
                )
                for scalar in scalar_documents
            )
            layout = cls.build(
                mixture=mixture,
                adsorbates=tuple(document["adsorbates"]),
                nz=grid_document["nz"],
                nr=grid_document["nr"],
                scalar_states=scalars,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid saved state layout: {exc}") from exc

        rebuilt = layout.to_dict()
        saved = dict(document)
        if saved != rebuilt:
            saved_signature = saved.get("signature")
            if saved_signature != rebuilt["signature"]:
                raise ValueError(
                    "Saved StateLayout signature does not match the layout "
                    f"rebuilt from its physical declaration ({saved_signature!r} != "
                    f"{rebuilt['signature']!r})."
                )
            raise ValueError(
                "Saved StateLayout metadata differs from the canonical report "
                "rebuilt from its physical declaration."
            )
        return layout

    def __post_init__(self) -> None:
        if not isinstance(self.mixture, Mixture):
            raise TypeError("`mixture` must be a Mixture instance.")
        adsorbates = _names(self.adsorbates, "adsorbates")
        missing = [name for name in adsorbates if name not in self.mixture.species]
        if missing:
            raise ValueError(f"Adsorbates not present in the mixture: {missing}.")

        scalar_states = tuple(self.scalar_states)
        if any(not isinstance(state, ScalarState) for state in scalar_states):
            raise TypeError("Every scalar state must be a ScalarState instance.")
        _names(
            (state.name for state in scalar_states),
            "scalar state names",
            allow_empty=True,
        )

        object.__setattr__(self, "adsorbates", adsorbates)
        object.__setattr__(self, "nz", _positive_integer(self.nz, "nz"))
        object.__setattr__(self, "nr", _positive_integer(self.nr, "nr"))
        object.__setattr__(self, "scalar_states", scalar_states)
        object.__setattr__(self, "fields", self._make_fields())

    def _make_fields(self) -> tuple[Field, ...]:
        """Generate the complete order in one explicit, auditable block."""
        cell = (self.nz, self.nr)
        descriptions = [
            *(
                (f"c_{species}", "gas concentration", cell, "mol m^-3", species)
                for species in self.mixture.independent_species
            ),
            *(
                (f"q_{species}", "solid loading", cell, "mol kg^-1", species)
                for species in self.adsorbates
            ),
            ("c_tot", "total concentration", cell, "mol m^-3", None),
            ("T_bed", "bed temperature", cell, "K", None),
            ("T_wall", "wall temperature", (self.nz,), "K", None),
            *(
                (state.name, f"{state.category} scalar", (), state.unit, None)
                for state in self.scalar_states
            ),
        ]

        fields: list[Field] = []
        offset = 0
        for name, kind, shape, unit, species in descriptions:
            size = int(np.prod(shape)) if shape else 1
            fields.append(Field(name, kind, shape, offset, offset + size, unit, species))
            offset += size

        duplicates = _duplicates(field.name for field in fields)
        if duplicates:
            raise ValueError(f"Generated field names collide: {duplicates}.")
        return tuple(fields)

    @property
    def closure_species(self) -> str:
        return self.mixture.closure

    @property
    def independent_gas_species(self) -> tuple[str, ...]:
        return self.mixture.independent_species

    @property
    def pde_fields(self) -> tuple[Field, ...]:
        return tuple(field for field in self.fields if field.storage != "y_scalars")

    @property
    def pde_size(self) -> int:
        return self.pde_fields[-1].stop

    @property
    def state_size(self) -> int:
        return self.fields[-1].stop

    @property
    def n_2d_fields(self) -> int:
        return len(self.fields_in("y_2d"))

    @property
    def n_1d_fields(self) -> int:
        return len(self.fields_in("y_1d"))

    @property
    def n_scalar_states(self) -> int:
        return len(self.scalar_states)

    def fields_in(self, storage: str) -> tuple[Field, ...]:
        return tuple(field for field in self.fields if field.storage == storage)

    def field(self, name: str) -> Field:
        for state_field in self.fields:
            if state_field.name == name:
                return state_field
        raise KeyError(f"Unknown state field {name!r}.")

    def slice_for(self, name: str) -> slice:
        return self.field(name).state_slice

    def gas_field(self, species: str) -> Field:
        if species == self.closure_species:
            raise ValueError(f"{species} is the closure species; it has no gas field.")
        return self._species_field("gas concentration", species)

    def loading_field(self, species: str) -> Field:
        return self._species_field("solid loading", species)

    def _species_field(self, kind: str, species: str) -> Field:
        for state_field in self.fields:
            if state_field.kind == kind and state_field.species == species:
                return state_field
        raise ValueError(f"No {kind} field exists for {species!r}.")

    def pack(self, values: Mapping[str, object]) -> np.ndarray:
        """Pack all fields by name."""
        return self._pack(values, self.fields)

    def pack_pde(self, values: Mapping[str, object]) -> np.ndarray:
        """Pack spatial fields, excluding integrated scalars."""
        return self._pack(values, self.pde_fields)

    @staticmethod
    def _pack(values: Mapping[str, object], fields: tuple[Field, ...]) -> np.ndarray:
        expected = {field.name for field in fields}
        received = set(values)
        if received != expected:
            raise ValueError(
                f"State fields differ (missing={sorted(expected - received)}, "
                f"extra={sorted(received - expected)})."
            )
        arrays = []
        for state_field in fields:
            values_array = np.asarray(values[state_field.name])
            if values_array.shape != state_field.shape:
                raise ValueError(
                    f"Field {state_field.name!r} has shape {values_array.shape}; "
                    f"expected {state_field.shape}."
                )
            arrays.append(values_array.reshape(-1))
        return np.concatenate(arrays)

    def unpack(self, vector, *, copy: bool = False) -> dict[str, np.ndarray]:
        """Unpack all fields into name-keyed arrays."""
        return self._unpack(vector, self.fields, copy)

    def unpack_pde(self, vector, *, copy: bool = False) -> dict[str, np.ndarray]:
        """Unpack a PDE-only vector into name-keyed arrays."""
        return self._unpack(vector, self.pde_fields, copy)

    @staticmethod
    def _unpack(vector, fields: tuple[Field, ...], copy: bool) -> dict[str, np.ndarray]:
        packed = np.asarray(vector)
        expected_shape = (fields[-1].stop,)
        if packed.shape != expected_shape:
            raise ValueError(f"Packed state must have shape {expected_shape}; got {packed.shape}.")
        return {
            field.name: (
                packed[field.state_slice].reshape(field.shape).copy()
                if copy
                else packed[field.state_slice].reshape(field.shape)
            )
            for field in fields
        }

    def __str__(self) -> str:
        lines = [
            f"Mixture: {', '.join(self.mixture.species)} "
            f"(closure: {self.closure_species})",
            f"{'field':<18} {'kind':<24} {'shape':<10} {'slice':<14} unit",
        ]
        for field in self.fields:
            shape = " x ".join(map(str, field.shape)) or "scalar"
            lines.append(
                f"{field.name:<18} {field.kind:<24} {shape:<10} "
                f"{field.start}:{field.stop:<12} {field.unit}"
            )
        return "\n".join(lines)

    def _document(self) -> dict[str, object]:
        return {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "mixture": {
                "species": list(self.mixture.species),
                "closure": self.closure_species,
                "independent_species": list(self.independent_gas_species),
            },
            "adsorbates": list(self.adsorbates),
            "grid": {"nz": self.nz, "nr": self.nr},
            "scalar_states": [
                {"name": state.name, "unit": state.unit, "category": state.category}
                for state in self.scalar_states
            ],
            "fields": [
                {
                    "name": field.name,
                    "kind": field.kind,
                    "species": field.species,
                    "shape": list(field.shape),
                    "state_slice": [field.start, field.stop],
                    "unit": field.unit,
                    "storage": field.storage,
                    "scale_source": field.scale_source,
                    "constraint": field.constraint,
                    "tolerance_group": field.tolerance_group,
                }
                for field in self.fields
            ],
            "pde_size": self.pde_size,
            "state_size": self.state_size,
        }

    @property
    def signature(self) -> str:
        """Stable identity for future layout-dependent sparsity caches."""
        encoded = json.dumps(
            self._document(), sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return a human-readable, JSON-compatible layout report."""
        return {**self._document(), "signature": self.signature}


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{label}` must be a non-empty string.")
    return value.strip()


def _names(
    values: Iterable[str], label: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"`{label}` must be a sequence, not a string.")
    names = tuple(_name(value, label) for value in values)
    if not names and not allow_empty:
        raise ValueError(f"`{label}` must not be empty.")
    duplicates = _duplicates(names)
    if duplicates:
        raise ValueError(f"`{label}` contains duplicates: {duplicates}.")
    return names


def _duplicates(names: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return tuple(duplicates)


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{label}` must be an integer.")
    if value < 1:
        raise ValueError(f"`{label}` must be at least 1.")
    return int(value)


# Explicit compatibility preset; the generic generator above has no DAC branch.
CURRENT_DAC_MIXTURE = Mixture(("CO2", "H2O", "O2", "N2"), closure="N2")
CURRENT_DAC_ADSORBATES = ("CO2", "H2O")


def simulation_scalar_states(adsorbates: Iterable[str]) -> tuple[ScalarState, ...]:
    """Return mass integrals for the adsorbates plus the three cycle energies."""
    species = _names(adsorbates, "adsorbates")
    return (
        *(ScalarState(f"m_{name}", "kg", "mass") for name in species),
        ScalarState("E_compression", "J", "energy"),
        ScalarState("E_vacuum", "J", "energy"),
        ScalarState("E_jacket", "J", "energy"),
    )


CURRENT_DAC_SCALARS = simulation_scalar_states(CURRENT_DAC_ADSORBATES)


def current_dac_state_layout(nz: int, nr: int) -> StateLayout:
    """Generate the exact field order used by the current pyDAC v3 model."""
    return StateLayout.build(
        mixture=CURRENT_DAC_MIXTURE,
        adsorbates=CURRENT_DAC_ADSORBATES,
        nz=nz,
        nr=nr,
        scalar_states=CURRENT_DAC_SCALARS,
    )


__all__ = [
    "CURRENT_DAC_ADSORBATES",
    "CURRENT_DAC_MIXTURE",
    "CURRENT_DAC_SCALARS",
    "Field",
    "Mixture",
    "ScalarState",
    "StateLayout",
    "current_dac_state_layout",
    "simulation_scalar_states",
]
