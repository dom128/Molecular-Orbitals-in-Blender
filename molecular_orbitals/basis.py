# SPDX-License-Identifier: GPL-3.0-only
"""L2 primitive normalization and real solid harmonics through f.

Pure labels follow IOData: c0,c1,s1,... with the Condon–Shortley phase removed.
Polynomials here act on raw monomials, not separately normalized Cartesian AOs.
"""

from math import pi, sqrt, prod
import numpy as np


def odd_double_factorial(n):
    return prod(range(n, 0, -2))  # includes (-1)!! = 1


def primitive_normalization(exponents, powers=None, angular_momentum=None):
    a = np.asarray(exponents, dtype=float)
    if not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("Gaussian exponents must be finite and positive")
    if powers is not None:
        l = sum(powers)
        denominator = prod(odd_double_factorial(2 * p - 1) for p in powers)
    elif angular_momentum is not None:
        l = angular_momentum
        denominator = odd_double_factorial(2 * l - 1)
    else:
        raise ValueError("Specify Cartesian powers or pure angular momentum")
    return (2 * a / pi) ** 0.75 * np.sqrt((4 * a) ** l / denominator)


def angular_polynomial(l, kind, label):
    if l not in range(4):
        raise ValueError(f"Angular momentum l={l} is unsupported; implemented: s, p, d, f")
    sign = -1 if label.startswith("-") else 1
    label = label.lstrip("-")
    if kind == "c":
        if label == "1" and l == 0:
            poly = {(0, 0, 0): 1.0}
        elif len(label) == l and set(label) <= set("xyz"):
            poly = {(label.count("x"), label.count("y"), label.count("z")): 1.0}
        else:
            raise ValueError(f"Unknown Cartesian convention {l}, {label}")
    elif kind == "p":
        pure = {
            0: {"c0": {(0, 0, 0): 1}},
            1: {"c0": {(0, 0, 1): 1}, "c1": {(1, 0, 0): 1}, "s1": {(0, 1, 0): 1}},
            2: {
                "c0": {(0, 0, 2): 1, (2, 0, 0): -0.5, (0, 2, 0): -0.5},
                "c1": {(1, 0, 1): sqrt(3)}, "s1": {(0, 1, 1): sqrt(3)},
                "c2": {(2, 0, 0): sqrt(3)/2, (0, 2, 0): -sqrt(3)/2},
                "s2": {(1, 1, 0): sqrt(3)}},
            3: {
                "c0": {(0, 0, 3): 1, (2, 0, 1): -1.5, (0, 2, 1): -1.5},
                "c1": {(1, 0, 2): sqrt(6), (3, 0, 0): -sqrt(6)/4, (1, 2, 0): -sqrt(6)/4},
                "s1": {(0, 1, 2): sqrt(6), (2, 1, 0): -sqrt(6)/4, (0, 3, 0): -sqrt(6)/4},
                "c2": {(2, 0, 1): sqrt(15)/2, (0, 2, 1): -sqrt(15)/2},
                "s2": {(1, 1, 1): sqrt(15)},
                "c3": {(3, 0, 0): sqrt(10)/4, (1, 2, 0): -3*sqrt(10)/4},
                "s3": {(2, 1, 0): 3*sqrt(10)/4, (0, 3, 0): -sqrt(10)/4}}
        }
        if label not in pure[l]:
            raise ValueError(f"Unknown real spherical convention {l}, {label}")
        poly = pure[l][label]
    else:
        raise ValueError(f"Unsupported basis kind {kind!r}")
    return {p: sign * float(c) for p, c in poly.items()}
