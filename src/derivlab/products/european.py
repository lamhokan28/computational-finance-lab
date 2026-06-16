from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from derivlab.core.validation import validate_positive
from derivlab.core.enums import OptionKind, parse_option_kind


@dataclass(frozen=True)
class EuropeanOption:
    kind: OptionKind | str
    strike: float
    maturity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", parse_option_kind(self.kind))
        validate_positive("strike", self.strike)
        validate_positive("maturity", self.maturity)

    def payoff(self, spot: ArrayLike) -> np.ndarray:
        spot_values = np.asarray(spot, dtype=float)
        return np.maximum(int(self.kind) * (spot_values - self.strike), 0.0)
