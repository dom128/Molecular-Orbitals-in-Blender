# SPDX-License-Identifier: GPL-3.0-only
"""Explicit AO polynomials and normalized-primitive weights; no hidden conventions."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np

BOHR_ANGSTROM = 0.529177210544
SCHEMA = "molecular-orbitals-osl/1"


def real_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name}: complex wavefunctions are unsupported")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: all values must be finite")
    return result


@dataclass
class AtomicOrbital:
    """sum_p weights[p] exp(-alpha[p]*r²) * sum_t polynomial[t]*x^a*y^b*z^c.

    weights already include primitive normalization, but preserve the source's
    contraction normalization. polynomial uses *unnormalized* Cartesian monomials.
    """

    center: int
    exponents: np.ndarray
    weights: np.ndarray
    polynomial: dict[tuple[int, int, int], float]
    label: str = ""

    def __post_init__(self):
        self.exponents = real_array(self.exponents, "exponents")
        self.weights = real_array(self.weights, "weights")
        if (self.exponents.ndim != 1 or not len(self.exponents)
                or self.weights.shape != self.exponents.shape
                or np.any(self.exponents <= 0)):
            raise ValueError("AO needs positive exponents and one weight per primitive")
        if not isinstance(self.center, int) or self.center < 0:
            raise ValueError("AO center must be a nonnegative integer")
        if not self.polynomial:
            raise ValueError("AO polynomial is empty")
        for powers, coefficient in self.polynomial.items():
            if len(powers) != 3 or any(not isinstance(p, int) or p < 0 for p in powers):
                raise ValueError("Polynomial powers must be three nonnegative integers")
            real_array(coefficient, "polynomial coefficient")


@dataclass
class OrbitalSet:
    coefficients: np.ndarray  # (AO, MO), in exactly the AO list's order
    occupations: np.ndarray  # spin resolved, in [0, 1]
    energies: np.ndarray | None = None  # Hartree

    def __post_init__(self):
        self.coefficients = real_array(self.coefficients, "MO coefficients")
        self.occupations = real_array(self.occupations, "occupations")
        if (self.coefficients.ndim != 2 or self.coefficients.shape[1] == 0
                or self.occupations.shape != (self.coefficients.shape[1],)
                or np.any(self.occupations < 0) or np.any(self.occupations > 1 + 1e-8)):
            raise ValueError("MO dimensions or spin-resolved occupations are invalid")
        if self.energies is not None:
            self.energies = real_array(self.energies, "orbital energies")
            if self.energies.shape != self.occupations.shape:
                raise ValueError("One energy is required per MO")


@dataclass
class Wavefunction:
    atomic_numbers: np.ndarray
    coordinates: np.ndarray
    aos: list[AtomicOrbital]
    orbitals: dict[str, OrbitalSet]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        numbers = real_array(self.atomic_numbers, "atomic numbers")
        if numbers.ndim != 1 or np.any(numbers != np.floor(numbers)) or np.any(numbers < 1):
            raise ValueError("Real nuclei with positive integer atomic numbers are required")
        self.atomic_numbers = numbers.astype(int)
        self.coordinates = real_array(self.coordinates, "coordinates")
        if self.coordinates.shape != (len(numbers), 3) or not len(numbers):
            raise ValueError("Coordinates must have shape (natom, 3)")
        if not self.aos or any(ao.center >= len(numbers) for ao in self.aos):
            raise ValueError("AO list is empty or has an invalid center")
        if not self.orbitals or set(self.orbitals) - {"alpha", "beta"}:
            raise ValueError("Only real alpha/beta orbitals are supported")
        for block in self.orbitals.values():
            if block.coefficients.shape[0] != len(self.aos):
                raise ValueError("MO coefficient rows do not match the AO list")

    def select(self, orbital="HOMO", spin="alpha"):
        """Return a zero-based MO index. Integers supplied by API are zero-based.

        Frontier selection uses energies when present, so unsorted files work.
        Fractional occupations: HOMO is highest-energy nonzero occupation;
        LUMO is lowest-energy zero occupation. No zero occupation means no LUMO.
        """
        if spin not in self.orbitals:
            raise ValueError(f"No {spin} spin block")
        block = self.orbitals[spin]
        if isinstance(orbital, str):
            name = orbital.upper()
            if name not in {"HOMO", "LUMO"}:
                raise ValueError("Orbital must be HOMO, LUMO, or a zero-based integer")
            candidates = np.flatnonzero(block.occupations > 1e-8 if name == "HOMO"
                                        else block.occupations <= 1e-8)
            if not len(candidates):
                raise ValueError(f"No {name} in the {spin} block")
            if block.energies is None:
                raise ValueError("Frontier selection requires orbital energies")
            choose = np.argmax if name == "HOMO" else np.argmin
            return int(candidates[choose(block.energies[candidates])])
        if not isinstance(orbital, (int, np.integer)) or not 0 <= orbital < block.coefficients.shape[1]:
            raise ValueError("MO index is out of range (API indices are zero-based)")
        return int(orbital)

    def to_json(self, path):
        data = {"schema": SCHEMA, "units": "bohr", "atomic_numbers": self.atomic_numbers.tolist(),
                "coordinates": self.coordinates.tolist(), "metadata": self.metadata,
                "aos": [{"center": ao.center, "exponents": ao.exponents.tolist(),
                         "weights": ao.weights.tolist(), "label": ao.label,
                         "polynomial": [[*p, c] for p, c in ao.polynomial.items()]}
                        for ao in self.aos],
                "orbitals": {spin: {"coefficients": b.coefficients.tolist(),
                                    "occupations": b.occupations.tolist(),
                                    "energies": None if b.energies is None else b.energies.tolist()}
                             for spin, b in self.orbitals.items()}}
        Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf8"))
        if data.get("schema") != SCHEMA or data.get("units") != "bohr":
            raise ValueError("Unsupported wavefunction JSON schema or units")
        aos = [AtomicOrbital(a["center"], a["exponents"], a["weights"],
                             {tuple(t[:3]): t[3] for t in a["polynomial"]}, a["label"])
               for a in data["aos"]]
        return cls(data["atomic_numbers"], data["coordinates"], aos,
                   {s: OrbitalSet(**b) for s, b in data["orbitals"].items()}, data["metadata"])
