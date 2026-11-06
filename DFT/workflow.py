# SPDX-License-Identifier: GPL-3.0-only
"""Read native PySCF job literals or the supported GAMESS subset without chemistry dependencies.

Geometry: explicit Cartesian atoms in a C1 $DATA group, in ANGS or BOHR.
This is an input translator, not a GAMESS wavefunction/output-file parser.
"""

import ast
from dataclasses import dataclass
import math
from pathlib import Path
import re

BOHR_ANGSTROM = 0.529177210544
DEFAULT_INPUT = Path(__file__).with_name("example_C22H14.py")
DEFAULT_PREFIX = DEFAULT_INPUT.parent / "results" / DEFAULT_INPUT.stem
SYMBOLS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
           "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
           "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
           "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
# Cordero et al., Dalton Trans. 2008, 2832: selected covalent radii, angstrom.
COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02}
GAMESS_B3LYP_XC = ".20*HF + .08*SLATER + .72*B88, .81*LYP + .19*VWN5"


@dataclass
class InputJob:
    title: str
    symbols: list[str]
    atomic_numbers: list[int]
    coordinates: list[tuple[float, float, float]]  # Always angstrom, original frame.
    groups: dict[str, dict[str, str]]
    charge: int
    multiplicity: int
    native: dict | None = None

    @property
    def electrons(self):
        return sum(self.atomic_numbers) - self.charge

    def validate_spin(self):
        spin = self.multiplicity - 1
        if self.multiplicity < 1 or self.electrons < spin or (self.electrons - spin) % 2:
            raise ValueError(f"Inconsistent charge/multiplicity: {self.electrons} electrons, MULT={self.multiplicity}")


def _number(text):
    value = float(text.replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ValueError("Coordinates and nuclear charges must be finite")
    return value


def read_input(path):
    if Path(path).suffix.lower() == ".py":
        return read_pyscf_input(path)
    source = Path(path).read_text(encoding="utf-8-sig")
    source = "\n".join(line.split("!", 1)[0] for line in source.splitlines())
    pattern = r"\$(\w+)\b(.*?)\$END\b"
    if re.sub(pattern, "", source, flags=re.I | re.S).strip():
        raise ValueError("Incomplete or unsupported GAMESS group syntax")
    blocks = {}
    for match in re.finditer(pattern, source, flags=re.I | re.S):
        name = match.group(1).upper()
        if name in blocks:
            raise ValueError(f"Duplicate ${name} group")
        blocks[name] = match.group(2)
    if "DATA" not in blocks:
        raise ValueError("Expected a GAMESS $DATA ... $END group")
    groups = {}
    for name, block in blocks.items():
        if name == "DATA":
            continue
        pairs = list(re.finditer(r"([A-Z][A-Z0-9_]*)\s*=\s*([^\s,]+)", block, flags=re.I))
        remainder = re.sub(r"([A-Z][A-Z0-9_]*)\s*=\s*([^\s,]+)", "", block, flags=re.I)
        if remainder.strip(" \t\r\n,"):
            raise ValueError(f"Unsupported syntax in ${name}: {remainder.strip()}")
        groups[name] = {}
        for pair in pairs:
            key, value = pair.group(1).upper(), pair.group(2).upper()
            if key in groups[name]:
                raise ValueError(f"Duplicate ${name} {key}")
            groups[name][key] = value
    control = groups.get("CONTRL", {})
    units = control.get("UNITS", "ANGS")
    if units not in {"ANGS", "BOHR"}:
        raise ValueError("Supported coordinate units: UNITS=ANGS or BOHR")
    if control.get("COORD", "UNIQUE") not in {"UNIQUE", "CART"}:
        raise ValueError("Only explicit Cartesian coordinates are supported")
    lines = [line.strip() for line in blocks["DATA"].strip().splitlines() if line.strip()]
    if len(lines) < 3 or lines[1].upper() != "C1":
        raise ValueError("Expected title, C1 symmetry, then explicit atoms in $DATA")
    symbols, numbers, coordinates = [], [], []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected 'element Z x y z', got: {line}")
        label, nuclear_charge, *xyz = fields
        z = _number(nuclear_charge)
        if not z.is_integer() or not 1 <= z < len(SYMBOLS):
            raise ValueError("Only real nuclei with integral atomic numbers are supported")
        z = int(z)
        element = re.fullmatch(r"([A-Za-z]+)\d*", label)
        if element is None or element.group(1).lower() != SYMBOLS[z].lower():
            raise ValueError(f"Element label and nuclear charge disagree: {line}")
        position = tuple(_number(c) * (BOHR_ANGSTROM if units == "BOHR" else 1.0) for c in xyz)
        if position in coordinates:
            raise ValueError("Duplicate nuclear position")
        symbols.append(SYMBOLS[z])
        numbers.append(z)
        coordinates.append(position)
    return InputJob(lines[0], symbols, numbers, coordinates, groups,
                    int(control.get("ICHARG", "0")), int(control.get("MULT", "1")))


def read_pyscf_input(path):
    """Read the supplied job's literal PySCF settings without executing Python.

    The job exposes MOLECULE, METHOD, XC, TITLE and MAX_CYCLES. General Python
    expressions/custom builders are deliberately not interpreted by Blender.
    """
    fields = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8-sig")).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"MOLECULE", "METHOD", "XC", "TITLE", "MAX_CYCLES"}:
                if target.id in fields:
                    raise ValueError(f"Duplicate native job field {target.id}")
                try:
                    fields[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{target.id} must be literal data for geometry-only loading") from exc
    molecule = fields.get("MOLECULE")
    if not isinstance(molecule, dict) or not {"atom", "basis"} <= molecule.keys():
        raise ValueError("Native job must define a literal MOLECULE with atom and basis")
    allowed = {"atom", "basis", "charge", "spin", "unit", "cart", "symmetry"}
    if set(molecule) - allowed:
        raise ValueError(f"Unsupported native molecule settings: {sorted(set(molecule)-allowed)}")
    if molecule.get("symmetry", False) is not False:
        raise ValueError("Native geometry loading requires symmetry=False")
    if not isinstance(molecule["basis"], str):
        raise ValueError("Native job must use a named PySCF basis")
    charge, spin = molecule.get("charge", 0), molecule.get("spin", 0)
    if type(charge) is not int or type(spin) is not int or spin < 0:
        raise ValueError("Native charge and nonnegative spin must be integers")
    unit = str(molecule.get("unit", "Angstrom")).lower()
    if unit not in {"angstrom", "angs", "bohr", "au"}:
        raise ValueError("Native coordinate unit must be Angstrom or Bohr")
    if not isinstance(molecule["atom"], str):
        raise ValueError("Native atom geometry must be a multiline string")
    symbols, numbers, positions = [], [], []
    for line in molecule["atom"].replace(";", "\n").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4 or parts[0] not in SYMBOLS[1:]:
            raise ValueError(f"Expected native 'element x y z', got: {line}")
        symbol, *xyz = parts
        position = tuple(_number(c) * (BOHR_ANGSTROM if unit in {"bohr", "au"} else 1) for c in xyz)
        if position in positions:
            raise ValueError("Duplicate nuclear position")
        symbols.append(symbol)
        numbers.append(SYMBOLS.index(symbol))
        positions.append(position)
    if not symbols:
        raise ValueError("Native atom geometry is empty")
    cart = molecule.get("cart", False)
    if type(cart) is not bool:
        raise ValueError("Native cart must be True or False")
    native = dict(basis=molecule["basis"], method=fields.get("METHOD", "rhf"),
                  xc=fields.get("XC"), cartesian=cart, max_cycles=fields.get("MAX_CYCLES", 50))
    result = InputJob(fields.get("TITLE", Path(path).stem), symbols, numbers, positions,
                      {}, charge, spin+1, native)
    result.validate_spin()
    return result


def infer_bonds(job, factor=1.2):
    """Distance-based guides; aromatic/bond orders are intentionally not assigned."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Bond factor must be positive and finite")
    unknown = sorted(set(job.atomic_numbers) - COVALENT_RADII.keys())
    if unknown:
        raise ValueError(f"No bond radii for {[SYMBOLS[z] for z in unknown]}; use --no-bonds")
    bonds = []
    for i, first in enumerate(job.coordinates):
        for j in range(i + 1, len(job.coordinates)):
            distance = math.dist(first, job.coordinates[j])
            cutoff = factor * (COVALENT_RADII[job.atomic_numbers[i]] + COVALENT_RADII[job.atomic_numbers[j]])
            if 0.4 < distance <= cutoff:
                bonds.append((i, j))
    return bonds


def calculation_settings(job, *, basis=None, xc=None, method=None):
    """Translate only settings whose meaning is implemented; reject other physics."""
    if job.native is not None:
        job.validate_spin()
        settings = job.native.copy()
        for key, value in (("basis", basis), ("xc", xc), ("method", method)):
            if value is not None:
                settings[key] = value
        methods = {"rhf", "rohf", "uhf", "rks", "roks", "uks"}
        if settings["method"] not in methods:
            raise ValueError(f"Native method must be one of {sorted(methods)}")
        settings["requested_method"] = settings["method"]
        # Match PySCF's scf.RHF factory: nonzero spin selects ROHF.
        if settings["method"] == "rhf" and job.multiplicity != 1:
            settings["method"] = "rohf"
        if settings["method"] in {"rhf", "rks"} and job.multiplicity != 1:
            raise ValueError("Restricted closed-shell methods require a singlet")
        if settings["method"].endswith("hf") and settings["xc"] is not None:
            raise ValueError("Hartree-Fock methods do not use XC; select a DFT method explicitly")
        if settings["method"].endswith("ks") and not settings["xc"]:
            raise ValueError("A DFT method requires an explicit XC functional")
        if type(settings["max_cycles"]) is not int or settings["max_cycles"] < 1:
            raise ValueError("MAX_CYCLES must be a positive integer")
        settings.update(charge=job.charge, multiplicity=job.multiplicity,
                        spin=job.multiplicity-1, electrons=job.electrons,
                        coordinate_units="angstrom", input_format="pyscf")
        return settings
    unknown_groups = set(job.groups) - {"CONTRL", "BASIS"}
    if unknown_groups:
        raise ValueError(f"Unsupported calculation groups: {sorted(unknown_groups)}")
    control = job.groups.get("CONTRL", {})
    known = {"SCFTYP", "RUNTYP", "ICHARG", "MULT", "DFTTYP", "UNITS", "COORD", "MAXIT", "ISPHER"}
    if set(control) - known:
        raise ValueError(f"Unsupported $CONTRL settings: {sorted(set(control) - known)}")
    if control.get("RUNTYP", "ENERGY") != "ENERGY":
        raise ValueError("This runner implements single-point RUNTYP=ENERGY only")
    job.validate_spin()
    if basis is None:
        basis_group = job.groups.get("BASIS", {})
        if set(basis_group) - {"GBASIS", "NGAUSS"}:
            raise ValueError("Unsupported $BASIS options; select a complete PySCF basis explicitly with --basis")
        key = (basis_group.get("GBASIS", ""), basis_group.get("NGAUSS", ""))
        names = {("N21", "3"): "3-21g", ("N31", "6"): "6-31g", ("STO", "3"): "sto-3g"}
        if key not in names:
            raise ValueError(f"Unsupported GAMESS basis {key}; supply --basis using a PySCF basis name")
        basis = names[key]
    if xc is None:
        functional = control.get("DFTTYP", "NONE")
        names = {"B3LYP": GAMESS_B3LYP_XC, "BLYP": "BLYP", "PBE": "PBE", "PBE0": "PBE0"}
        if functional not in names:
            raise ValueError(f"Unsupported DFTTYP={functional}; supply --xc explicitly")
        xc = names[functional]
    if method is None:
        name = control.get("SCFTYP", "RHF")
        if name not in {"RHF", "ROHF", "UHF"}:
            raise ValueError(f"Unsupported SCFTYP={name}; supply --method explicitly")
        method = {"RHF": "rks", "ROHF": "roks", "UHF": "uks"}[name]
    if method not in {"rks", "roks", "uks"}:
        raise ValueError("Method must be rks, roks, or uks")
    if method == "rks" and job.multiplicity != 1:
        raise ValueError("RKS requires a singlet; choose --method roks or uks for open shells")
    ispher = int(control.get("ISPHER", "-1"))
    if ispher not in {-1, 1}:
        raise ValueError("Supported ISPHER values: -1 (Cartesian) or 1 (spherical)")
    max_cycles = int(control.get("MAXIT", "100"))
    if max_cycles < 1:
        raise ValueError("MAXIT must be positive")
    return dict(basis=basis, xc=xc, method=method, charge=job.charge,
                multiplicity=job.multiplicity, spin=job.multiplicity - 1,
                electrons=job.electrons, cartesian=ispher == -1, max_cycles=max_cycles,
                coordinate_units="angstrom")
