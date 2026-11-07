# SPDX-License-Identifier: GPL-3.0-only
"""Specialize a calculated real molecular orbital into a standalone Cycles OSL shader.

python DFT/gen_shader.py DFT/results/example_C22H14.molden --orbital HOMO --spin alpha
Requires NumPy and qc-iodata for Molden/FCHK. No PySCF or Blender is required.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The optional local cache contains Windows wheels in this checkout. Never
# prepend it to a WSL/Linux environment with its own installed dependencies.
if sys.platform == "win32" and (ROOT / ".local-deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local-deps"))
from DFT.workflow import DEFAULT_PREFIX
from molecular_orbitals import load_wavefunction
from molecular_orbitals.model import BOHR_ANGSTROM
from molecular_orbitals.osl import generate_osl


def generate_volume_osl(wavefunction, orbital="HOMO", spin="alpha"):
    """Reuse the exact LCAO kernel with conditional Cycles volume components.

    Scattering is opt-in and adds extinction to the existing absorption. Raw
    Amplitude/Probability remain independent of display controls when connected.
    With no active volume and no numerical connections, no Gaussians are evaluated.
    This wrapper keeps the volume customization local to the DFT entry point.
    """
    kernel, separator, _ = generate_osl(wavefunction, orbital, spin).partition("\nshader molecular_orbital(")
    if not separator:
        raise RuntimeError("The LCAO generator did not produce the expected shader interface")
    return kernel + f"""
shader molecular_orbital(
    float BohrInBlender = {BOHR_ANGSTROM:.17g},
    float Density = 1.0,
    float Emission = 10.0,
    float Absorption = 0.0,
    float Scattering = 0.0,
    float Anisotropy = 0.0,
    float Threshold = 0.000,
    color Positive = color(1.00, 0.55, 0.39),
    color Negative = color(0.42, 0.59, 1.00),
    output float Amplitude = 0.0,
    output float Probability = 0.0,
    output closure color PA = 0,
    output closure color PD = 0)
{{
    int volume_active = Density > 0.0 &&
                        (Emission > 0.0 || Absorption > 0.0 || Scattering > 0.0);
    // Numerical connections must still receive psi/psi^2 with the volume off.
    if (volume_active || isconnected(Amplitude) || isconnected(Probability)) {{
        point r_bohr = transform("object", P) / max(BohrInBlender, 1.0e-12);
        Amplitude = evaluate_mo(r_bohr);
        Probability = Amplitude * Amplitude;
        float magnitude = abs(Amplitude);
        // Thresholding requires psi first; it only skips closure construction.
        if (volume_active && magnitude > 0.0 && magnitude >= max(Threshold, 0.0)) {{
            color phase_color = Amplitude >= 0.0 ? Positive : Negative;
            closure color volume = 0;
            if (Emission > 0.0)
                volume += Emission * phase_color * emission();
            if (Absorption > 0.0)
                volume += Absorption * (color(1.0) - phase_color) * absorption();
            if (Scattering > 0.0)
                volume += Scattering * phase_color *
                          henyey_greenstein(clamp(Anisotropy, -0.999, 0.999));
            PA = Density * magnitude * volume;
            PD = Density * Probability * volume;
        }}
    }}
}}
"""


def create_shader(input_path, output=None, orbital="HOMO", spin="alpha", wavefunction_json=None):
    input_path = Path(input_path)
    report_path = input_path.with_suffix(".results.json")
    calculation = None
    if input_path.suffix.lower() == ".molden" and report_path.is_file():
        calculation = json.loads(report_path.read_text(encoding="utf8"))
        if calculation.get("converged") is not True:
            raise ValueError("The latest calculation did not converge; refusing to use a stale Molden file")
        actual_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if calculation.get("molden_sha256") != actual_hash:
            raise ValueError("Molden file does not match the converged calculation report")
    selected = orbital.upper() if isinstance(orbital, str) else orbital
    if isinstance(selected, str) and selected.isdigit():
        if int(selected) < 1:
            raise ValueError("Orbital numbers are one-based and must be positive")
        selected = int(selected) - 1
    wf = load_wavefunction(input_path)
    index = wf.select(selected, spin)
    source = generate_volume_osl(wf, index, spin)
    label = str(orbital).lower()
    output = Path(output) if output else input_path.with_name(f"{input_path.stem}_{spin}_{label}.osl")
    portable = Path(wavefunction_json) if wavefunction_json else input_path.with_suffix(".wavefunction.json")
    manifest = output.with_suffix(".selection.json")
    if len({p.resolve() for p in (input_path, output, portable, manifest)}) != 4:
        raise ValueError("Input and output files must have distinct paths")
    for path in (output, portable, manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    if calculation:
        wf.metadata["dft_calculation"] = calculation
    wf.to_json(portable)
    output.write_text(source, encoding="utf8")
    selection = dict(source=str(input_path.resolve()), atoms=len(wf.atomic_numbers),
                     atomic_orbitals=len(wf.aos), spin=spin, orbital_one_based=index + 1,
                     atomic_numbers=wf.atomic_numbers.tolist(),
                     coordinates_angstrom=(wf.coordinates * BOHR_ANGSTROM).tolist(),
                     occupation=float(wf.orbitals[spin].occupations[index]),
                     energy_hartree=None if wf.orbitals[spin].energies is None else float(wf.orbitals[spin].energies[index]),
                     shader=output.name, wavefunction_json=portable.name)
    manifest.write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return output, portable, selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_PREFIX.with_suffix(".molden"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--orbital", default="HOMO", help="HOMO, LUMO, or a one-based orbital number")
    parser.add_argument("--spin", choices=("alpha", "beta"), default="alpha")
    parser.add_argument("--wavefunction-json", type=Path)
    args = parser.parse_args(argv)
    output, portable, selection = create_shader(args.input, args.output, args.orbital, args.spin, args.wavefunction_json)
    print(json.dumps(selection, indent=2))
    print(f"Shader: {output}\nPortable wavefunction: {portable}")


if __name__ == "__main__":
    main()
