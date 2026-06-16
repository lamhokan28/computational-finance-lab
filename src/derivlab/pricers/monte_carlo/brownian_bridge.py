from __future__ import annotations

from math import exp, sqrt
from time import perf_counter

import numpy as np

from derivlab.core.enums import BarrierDirection, BarrierKnock
from derivlab.core.results import PricingResult
from derivlab.models.black_scholes import BlackScholesModel
from derivlab.pricers.monte_carlo.engine import SimulationConfig
from derivlab.products.barrier import BarrierOption


class BarrierCrossingCorrectionMonteCarloPricer:
    """Monte Carlo pricer using Brownian-bridge barrier crossing probabilities."""

    method = "barrier_crossing_corrected_monte_carlo"

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()

    def price(
                    self,
                    model: BlackScholesModel,
                    option: BarrierOption,
                    config: SimulationConfig | None = None,
                ) -> PricingResult:
        
        cfg = config or self.config
        start = perf_counter()

        # generate ordinary monte carlo paths
        paths = model.simulate_paths(
                                        option.maturity,
                                        cfg.n_paths,
                                        cfg.n_steps,
                                        seed=cfg.seed,
                                        antithetic=cfg.antithetic,
                                    )
        
        # calculate the probability of survival between each grid point
        survival = self.survival_probabilities(paths, option, model.volatility, option.maturity)
        activity = survival if option.knock is BarrierKnock.OUT else 1.0 - survival
        
        # survival probability corrected sample payoff
        samples = exp(-model.rate * option.maturity) * option.payoff(paths[:, -1]) * activity
        price = float(np.mean(samples))
        stderr = float(np.std(samples, ddof=1) / sqrt(len(samples)))
        
        return PricingResult(
                                price=price,
                                stderr=stderr,
                                runtime=perf_counter() - start,
                                method=self.method,
                                metadata={
                                            "n_paths": cfg.n_paths,
                                            "n_steps": cfg.n_steps,
                                            "seed": cfg.seed,
                                            "antithetic": cfg.antithetic,
                                            "bridge_correction": True,
                                        },
                            )

    @staticmethod
    def survival_probabilities(paths: np.ndarray, option: BarrierOption, volatility: float, maturity: float) -> np.ndarray:
        paths = np.asarray(paths, dtype=float)
        if paths.ndim != 2 or paths.shape[1] < 2:
            raise ValueError("paths must be a two-dimensional array with at least one time interval")

        if volatility == 0:
            return option.is_alive(paths).astype(float)

        dt = maturity / (paths.shape[1] - 1)
        log_paths = np.log(paths)
        log_barrier = np.log(option.barrier)
        left = log_paths[:, :-1]
        right = log_paths[:, 1:]

        if option.direction is BarrierDirection.UP:
            # For up barrier, we are only interested in the prob of crossing when both end points are under the barrier.
            # This is because other scenarios yields prob of 1.
            # Therefore, "touched" is a prelim parser for whether prob = 1.0 or crossing_prob
            touched = (left >= log_barrier) | (right >= log_barrier)
            distance_product = (log_barrier - left) * (log_barrier - right)
        else:
            # vice versa
            touched = (left <= log_barrier) | (right <= log_barrier)
            distance_product = (left - log_barrier) * (right - log_barrier)

        crossing_prob = np.exp(-2.0 * distance_product / (volatility**2 * dt))
        crossing_prob = np.clip(np.where(touched, 1.0, crossing_prob), 0.0, 1.0)
        interval_survival = 1.0 - crossing_prob
        return np.prod(interval_survival, axis=1)
