from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from derivlab.core.validation import validate_positive
from derivlab.core.enums import (
                                    BarrierDirection,
                                    BarrierKnock,
                                    OptionKind,
                                    parse_barrier_direction,
                                    parse_barrier_knock,
                                    parse_option_kind,
                                )


@dataclass(frozen=True)
class BarrierOption:
    kind: OptionKind | str
    strike: float
    maturity: float
    barrier: float
    direction: BarrierDirection | str
    knock: BarrierKnock | str = BarrierKnock.OUT
    flag: str = "XYZ"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", parse_option_kind(self.kind))
        object.__setattr__(self, "direction", parse_barrier_direction(self.direction))
        object.__setattr__(self, "knock", parse_barrier_knock(self.knock))
        validate_positive("strike", self.strike)
        validate_positive("maturity", self.maturity)
        validate_positive("barrier", self.barrier)
        
        x = "U" if self.direction == BarrierDirection.UP else "D"
        y = "I" if self.knock == BarrierKnock.IN else "O"
        z = "C" if self.kind == OptionKind.CALL else "P"
        object.__setattr__(self, "flag", x + y + z)

    def payoff(self, spot: ArrayLike) -> np.ndarray:
        spot_values = np.asarray(spot, dtype=float)
        return np.maximum(int(self.kind) * (spot_values - self.strike), 0.0)

    def is_breached_by_spot(self, spot: float) -> bool:
        if self.direction is BarrierDirection.UP:
            return spot >= self.barrier
        return spot <= self.barrier

    def is_alive(self, paths: np.ndarray) -> np.ndarray:
        if self.direction is BarrierDirection.UP:
            touched = np.max(paths, axis=1) >= self.barrier
        else:
            touched = np.min(paths, axis=1) <= self.barrier

        if self.knock is BarrierKnock.OUT:
            return ~touched
        return touched

    def payoff_from_paths(self, paths: np.ndarray) -> np.ndarray:
        return self.payoff(paths[:, -1]) * self.is_alive(paths)
