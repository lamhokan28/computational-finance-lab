from __future__ import annotations

import numpy as np
from scipy.stats import norm

from derivlab.core.enums import StatisticalDistribution

def cdf(x: float | np.ndarray, dist: StatisticalDistribution) -> float | np.ndarray:
    if dist == StatisticalDistribution.NORMAL:
        return norm.cdf(x)
    raise ValueError(f"Unsupported distribution: {dist}")


def pdf(x: float | np.ndarray, dist: StatisticalDistribution) -> float | np.ndarray:
    if dist == StatisticalDistribution.NORMAL:
        return norm.pdf(x)
    raise ValueError(f"Unsupported distribution: {dist}")


def ppf(u: float | np.ndarray, dist: StatisticalDistribution) -> float | np.ndarray:
    if dist == StatisticalDistribution.NORMAL:
        return norm.ppf(u)
    raise ValueError(f"Unsupported distribution: {dist}")
