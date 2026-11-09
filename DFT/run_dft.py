# SPDX-License-Identifier: GPL-3.0-only
"""Run a native PySCF job or supported GAMESS input with wavefunction exports.

python DFT/run_dft.py DFT/example_C22H14.py --dry-run
python DFT/run_dft.py DFT/example_C22H14.py --threads 8 --memory-mb 4000
PySCF runs on Linux, macOS, or WSL. Blender does not need PySCF installed.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import pyscf
from pyscf import dft, gto, lib, scf
from pyscf.tools import molden

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DFT.workflow import calculation_settings, read_input


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path, help="Default: INPUT_DIRECTORY/results/INPUT_STEM")
    parser.add_argument("--basis", help="Explicit PySCF basis name; overrides the input basis")
    parser.add_argument("--xc", help="Explicit PySCF exchange-correlation expression")
    parser.add_argument("--method", choices=("rhf", "rohf", "uhf", "rks", "roks", "uks"))
    parser.add_argument("--charge", type=int)
    parser.add_argument("--multiplicity", type=int)
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--memory-mb", type=int, default=4000)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--conv-tol", type=float, default=1e-9)
    parser.add_argument("--grid-level", type=int, choices=range(10), help="DFT only; default 3")
    parser.add_argument("--density-fit", action="store_true", help="Opt into a density-fitting approximation")
    parser.add_argument("--dry-run", action="store_true", help="Print effective settings without running a calculation or writing files")
    return parser.parse_args(argv)


def resolve(args):
    job = read_input(args.input)
    if args.charge is not None:
        job.charge = args.charge
    if args.multiplicity is not None:
        job.multiplicity = args.multiplicity
    settings = calculation_settings(job, basis=args.basis, xc=args.xc, method=args.method)
    if args.max_cycles is not None:
        settings["max_cycles"] = args.max_cycles
    if args.threads < 1 or args.memory_mb < 1 or settings["max_cycles"] < 1:
        raise ValueError("Threads, memory and maximum SCF cycles must be positive")
    if not math.isfinite(args.conv_tol) or args.conv_tol <= 0:
        raise ValueError("Convergence tolerance must be positive and finite")
    if settings["method"].endswith("hf") and args.grid_level is not None:
        raise ValueError("A numerical quadrature grid applies only to DFT methods")
    grid_level = (3 if args.grid_level is None else args.grid_level) if settings["method"].endswith("ks") else None
    settings.update(threads=args.threads, memory_mb=args.memory_mb, conv_tol=args.conv_tol,
                    grid_level=grid_level, density_fit=args.density_fit)
    return job, settings


def write_report(path, report):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def solve(job, settings, prefix, input_path):
    prefix = Path(prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {key: Path(str(prefix) + suffix) for key, suffix in (
        ("molden", ".molden"), ("checkpoint", ".chk"), ("log", ".log"), ("report", ".results.json"))}
    report = dict(engine="PySCF", engine_version=pyscf.__version__, converged=False,
                  input_file=str(Path(input_path).resolve()),
                  input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                  title=job.title, settings=settings,
                  symbols=job.symbols, coordinates_angstrom=job.coordinates,
                  files={name: path.name for name, path in paths.items()})
    # Publish failure status first so a stale Molden file cannot appear current.
    write_report(paths["report"], report)
    lib.num_threads(settings["threads"])
    try:
        mol = gto.M(atom=list(zip(job.symbols, job.coordinates)), unit="Angstrom",
                    basis=settings["basis"], charge=settings["charge"], spin=settings["spin"],
                    cart=settings["cartesian"], symmetry=False, max_memory=settings["memory_mb"],
                    verbose=4, output=str(paths["log"]), parse_arg=False)
        calculation = {"rhf": scf.RHF, "rohf": scf.ROHF, "uhf": scf.UHF,
                       "rks": dft.RKS, "roks": dft.ROKS, "uks": dft.UKS}[settings["method"]](mol)
        if settings["density_fit"]:
            calculation = calculation.density_fit()
        if settings["method"].endswith("ks"):
            calculation.xc = settings["xc"]
            calculation.grids.level = settings["grid_level"]
        calculation.conv_tol = settings["conv_tol"]
        calculation.max_cycle = settings["max_cycles"]
        calculation.chkfile = str(paths["checkpoint"])
        energy = float(calculation.kernel())
        report.update(energy_hartree=energy if math.isfinite(energy) else None,
                      atomic_orbitals=mol.nao_nr(), alpha_electrons=mol.nelec[0], beta_electrons=mol.nelec[1])
        if not calculation.converged or not math.isfinite(energy):
            raise RuntimeError(f"SCF did not converge; no new Molden file exported. Inspect {paths['log']}")
        temporary = paths["molden"].with_name(paths["molden"].name + ".tmp")
        # Never silently drop high-l functions; the shader importer rejects
        # unsupported angular momenta explicitly instead.
        molden.from_scf(calculation, str(temporary), ignore_h=False)
        temporary.replace(paths["molden"])
        report["molden_sha256"] = hashlib.sha256(paths["molden"].read_bytes()).hexdigest()
        report["converged"] = True
        write_report(paths["report"], report)
        return report
    except Exception as exc:
        report["error"] = str(exc)
        write_report(paths["report"], report)
        raise


def main(argv=None):
    args = parse_args(argv)
    job, settings = resolve(args)
    print(json.dumps(dict(atoms=len(job.symbols), **settings), indent=2))
    if args.dry_run:
        return
    prefix = args.output_prefix or args.input.parent / "results" / args.input.stem
    print(f"Running PySCF; progress log: {prefix}.log", flush=True)
    report = solve(job, settings, prefix, args.input)
    print(f"Converged energy: {report['energy_hartree']:.12f} Hartree\nOrbitals: {prefix}.molden")


if __name__ == "__main__":
    main()
