# Title: C22H14 | Single Point | RHF/3-21g
# SPDX-License-Identifier: GPL-3.0-only
"""Native PySCF pentacene job with checkpoint/Molden exports.

python DFT/example_C22H14.py --dry-run
python DFT/example_C22H14.py --threads 8
The molecule and method below are read without running PySCF by Blender.
"""
from pathlib import Path
import sys

from pyscf import gto

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DFT.run_dft import main as run

TITLE = "Pentacene C22H14 | PySCF HF/3-21g"
MOLECULE = {
    "atom": """
   C     -4.865300   -1.412300   -0.000100
   C     -6.018700   -0.698400   -0.000200
   C     -6.018800    0.698200   -0.000300
   C     -4.865400    1.412300   -0.000400
   C     -3.625300    0.734500   -0.000300
   C     -2.425800    1.435200    0.000100
   C     -1.210100    0.735500    0.000600
   C      0.000100    1.434600    0.000600
   C      1.210100    0.735200    0.000700
   C      2.425800    1.434900    0.000100
   C      3.625200    0.734300   -0.000300
   C      4.865200    1.412300   -0.000400
   C      6.018700    0.698500   -0.000300
   C      6.018800   -0.698100   -0.000100
   C      4.865500   -1.412200    0.000000
   C      3.625300   -0.734500   -0.000100
   C      2.425900   -1.435200   -0.000500
   C      1.210100   -0.735500    0.000800
   C     -0.000000   -1.434700   -0.000100
   C     -1.210100   -0.735300    0.000700
   C     -2.425800   -1.435000   -0.000200
   C     -3.625200   -0.734300   -0.000200
   H     -4.894300   -2.491900    0.004600
   H     -6.962100   -1.224000   -0.000500
   H     -6.962300    1.223700   -0.000800
   H     -4.894700    2.491900   -0.000900
   H     -2.429300    2.515200   -0.000300
   H      0.000100    2.514600    0.000200
   H      2.429300    2.514900   -0.000300
   H      4.894300    2.491900   -0.000900
   H      6.962100    1.224100   -0.000700
   H      6.962300   -1.223600   -0.000300
   H      4.894800   -2.491800   -0.000200
   H      2.429400   -2.515200   -0.000300
   H     -0.000100   -2.514700   -0.001800
   H     -2.429400   -2.515000   -0.001900
""",
    "basis": '3-21g',
    "charge": -1,
    "spin": 1,
    "unit": "Angstrom",
    "cart": False,
    "symmetry": False,
}
METHOD = "rhf"
XC = None
MAX_CYCLES = 50


def build_molecule(**overrides):
    """Construct this native PySCF molecule; importing the job does no work."""
    return gto.M(**(MOLECULE | overrides))


def main():
    run([str(Path(__file__).resolve()), *sys.argv[1:]])


if __name__ == "__main__":
    main()
