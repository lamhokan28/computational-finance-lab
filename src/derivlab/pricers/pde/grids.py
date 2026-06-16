from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridConfig:

    # Lower and upper spacial space (i.e., spot) truncation.
    s_min: float = 0.0 # Stock price floors at zero
    s_max: float = 300.0 # Arbitrary ceiling for stock price

    # Number of grid intervals, not grid points. The solver therefore builds
    # n_space + 1 spot nodes including both boundaries.
    n_space: int = 200
    n_time: int = 200

    # Theta determines the time-stepping scheme:
    # - theta = 0.0: explicit Euler.
    # - theta = 1.0: implicit Euler.
    # - (Default) theta = 0.5: Crank-Nicolson.
    theta: float = 0.5

    def __post_init__(self) -> None:
        # Input validation
        if self.s_min < 0:
            raise ValueError("s_min must be non-negative")
        if self.s_max <= self.s_min:
            raise ValueError("s_max must be greater than s_min")
        if self.n_space < 3:
            raise ValueError("n_space must be at least 3")
        if self.n_time < 1:
            raise ValueError("n_time must be positive")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must be in [0, 1]")
