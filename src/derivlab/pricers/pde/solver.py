from __future__ import annotations

import numpy as np

def solve_tridiagonal(lower: np.ndarray, main: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:

    n = len(main)
    c_prime = np.empty(n - 1)
    d_prime = np.empty(n)

    # Forward elimination: transform the matrix into upper-triangular form,
    # storing modified upper diagonal and rhs coefficients.
    c_prime[0] = upper[0] / main[0]
    d_prime[0] = rhs[0] / main[0]
    for i in range(1, n - 1):
        denom = main[i] - lower[i - 1] * c_prime[i - 1]
        c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom
    d_prime[-1] = (rhs[-1] - lower[-1] * d_prime[-2]) / (main[-1] - lower[-1] * c_prime[-1])

    # Backward substitution: recover the solution from the last node backward.
    out = np.empty(n)
    out[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        out[i] = d_prime[i] - c_prime[i] * out[i + 1]
    return out