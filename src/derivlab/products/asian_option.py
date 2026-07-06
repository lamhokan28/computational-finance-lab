from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from derivlab.core.validation import validate_positive
from derivlab.core.enums import AverageType, OptionKind, parse_average_type, parse_option_kind


@dataclass(frozen=True)
class AsianOption:
    kind: OptionKind | str
    strike: float
    maturity: float
    average: AverageType | str = AverageType.ARITHMETIC

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", parse_option_kind(self.kind))
        object.__setattr__(self, "average", parse_average_type(self.average))
        validate_positive("strike", self.strike)
        validate_positive("maturity", self.maturity)

    def payoff(self, average: ArrayLike) -> np.ndarray:
        average_values = np.asarray(average, dtype=float)
        return np.maximum(int(self.kind) * (average_values - self.strike), 0.0)

    def average_from_paths(self, paths: np.ndarray) -> np.ndarray:
        sampled_paths = paths[:, 1:]
        if self.average is AverageType.ARITHMETIC:
            return np.mean(sampled_paths, axis=1)
        return np.exp(np.mean(np.log(sampled_paths), axis=1))

    def payoff_from_paths(self, paths: np.ndarray) -> np.ndarray:
        return self.payoff(self.average_from_paths(paths))
