# SPDX-License-Identifier: GPL-3.0-only
"""Delegate file dialects to IOData; preserve its AO ordering and contractions."""

from pathlib import Path
import warnings
import numpy as np

from iodata import load_one
from iodata.overlap import compute_overlap

from .model import AtomicOrbital, OrbitalSet, Wavefunction
from .basis import angular_polynomial, primitive_normalization


def from_iodata(data, source="", check_orthonormality=True):
    if data.obasis is None or data.mo is None or data.atcoords is None or data.atnums is None:
        raise ValueError("File must contain nuclei, Gaussian basis, and MO coefficients")
    if data.obasis.primitive_normalization != "L2":
        raise ValueError("Only L2-normalized Gaussian primitives are supported")
    if data.mo.kind not in {"restricted", "unrestricted"}:
        raise ValueError("Complex/spinor/generalized wavefunctions are unsupported")
    if data.mo.coeffs is None or np.iscomplexobj(data.mo.coeffs):
        raise ValueError("Real MO coefficients are required")
    if data.mo.occs is None:
        raise ValueError("Orbital occupations are required")
    if getattr(data, "cellvecs", None) is not None:
        raise ValueError("Periodic wavefunctions are unsupported")
    aos = []
    for shell_index, shell in enumerate(data.obasis.shells):
        for column, (l, kind) in enumerate(zip(shell.angmoms, shell.kinds)):
            l = int(l)
            labels = data.obasis.conventions.get((l, kind))
            expected = (l + 1) * (l + 2) // 2 if kind == "c" else 2*l+1
            if labels is None or len(labels) != expected or len(set(labels)) != expected:
                raise ValueError(f"Missing or invalid AO ordering for shell ({l}, {kind})")
            for label in labels:
                poly = angular_polynomial(l, kind, label)
                norm = primitive_normalization(shell.exponents, powers=next(iter(poly))) if kind == "c" else primitive_normalization(shell.exponents, angular_momentum=l)
                aos.append(AtomicOrbital(int(shell.icenter), shell.exponents.copy(),
                                         shell.coeffs[:, column] * norm, poly,
                                         f"shell{shell_index}:{column}:{l}{kind}:{label}"))
    blocks = {}
    for spin, suffix in (("alpha", "a"), ("beta", "b")):
        coefficients = getattr(data.mo, "coeffs" + suffix)
        if coefficients is not None and coefficients.shape[1]:
            blocks[spin] = OrbitalSet(coefficients.copy(), getattr(data.mo, "occs" + suffix).copy(),
                                      getattr(data.mo, "energies" + suffix))
    wf = Wavefunction(data.atnums, data.atcoords, aos, blocks,
                      {"source": str(source), "importer": "IOData", "mo_kind": data.mo.kind,
                       "basis_name": getattr(data, "obasis_name", None),
                       "primitive_normalization": "L2", "angular_momentum_limit": 3})
    if check_orthonormality:
        overlap = compute_overlap(data.obasis, data.atcoords)
        errors = {}
        for spin, block in blocks.items():
            gram = block.coefficients.T @ overlap @ block.coefficients
            error = float(np.max(np.abs(gram - np.eye(len(gram)))))
            if error > 5e-4:
                raise ValueError(f"{spin} MOs fail C.T S C = I: max error {error:.3g}; check file conventions")
            errors[spin] = error
        wf.metadata["orthonormality_max_abs_error"] = errors
    return wf


def load_wavefunction(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return Wavefunction.from_json(path)
    name = path.name.lower()
    if name.endswith((".fchk", ".fch")):
        fmt = "fchk"
    elif name.endswith((".molden", ".molden.input")):
        fmt = "molden"
    else:
        raise ValueError("Supported inputs: .fchk, .fch, .molden, .molden.input, prototype .json")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = load_one(str(path), fmt=fmt)
    wf = from_iodata(data, path.name)
    wf.metadata["import_warnings"] = [str(item.message) for item in caught]
    for item in caught:
        warnings.warn(str(item.message), item.category, stacklevel=2)
    return wf
