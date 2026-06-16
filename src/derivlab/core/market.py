from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class MarketParams:
    
    spot: float
    rate: float
    dividend: float = 0.0
    volatility: float = 0.2

    # Validate inputs
    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")

    # def discount_factor(self, maturity: float) -> float:
    #     if maturity < 0:
    #         raise ValueError("maturity must be non-negative")
    #     return exp(-self.rate * maturity)

    # def dividend_factor(self, maturity: float) -> float:
    #     if maturity < 0:
    #         raise ValueError("maturity must be non-negative")
    #     return exp(-self.dividend * maturity)

