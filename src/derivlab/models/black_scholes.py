from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from derivlab.core.market import MarketParams
from derivlab.core.random import make_rng


@dataclass(frozen=True)
class BlackScholesModel:
    """Risk-neutral Black-Scholes model with continuous dividend yield."""

    market: MarketParams

    @property
    def spot(self) -> float:
        return self.market.spot

    @property
    def rate(self) -> float:
        return self.market.rate

    @property
    def dividend(self) -> float:
        return self.market.dividend

    @property
    def volatility(self) -> float:
        return self.market.volatility

    def simulate_paths(
                            self,
                            maturity: float,
                            n_paths: int,
                            n_steps: int,
                            seed: int | None = None,
                            antithetic: bool = False,
                        ) -> np.ndarray:
        
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        
        rng = make_rng(seed)
        
        if antithetic:
            half = (n_paths + 1) // 2
            shocks = rng.standard_normal((half, n_steps))
            shocks = np.vstack([shocks, -shocks])[:n_paths]
        else:
            shocks = rng.standard_normal((n_paths, n_steps))

        return self.paths_from_shocks(maturity, shocks)

    def paths_from_shocks(self, maturity: float, shocks: np.ndarray) -> np.ndarray:
        """Build Black-Scholes paths from supplied standard-normal shocks."""

        shocks = np.asarray(shocks, dtype=float)
        if shocks.ndim != 2:
            raise ValueError("shocks must be a two-dimensional array")

        n_paths, n_steps = shocks.shape
        if n_paths <= 0:
            raise ValueError("shocks must contain at least one path")
        if n_steps <= 0:
            raise ValueError("shocks must contain at least one time step")

        dt = maturity / n_steps
        drift = (self.rate - self.dividend - 0.5 * self.volatility**2) * dt
        diffusion = self.volatility * np.sqrt(dt) * shocks
        log_returns = drift + diffusion
        log_paths = np.cumsum(log_returns, axis=1)
        log_paths = np.column_stack([np.zeros(n_paths), log_paths])
        
        return self.spot * np.exp(log_paths)

    def log_spot_characteristic_function(self, maturity: float, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        mean = np.log(self.spot) + (self.rate - self.dividend - 0.5 * self.volatility**2) * maturity
        variance = self.volatility**2 * maturity
        return np.exp(1j * u * mean - 0.5 * variance * u**2)

