from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from time import perf_counter # for runtime

import numpy as np

from derivlab.core.results import PricingResult
from derivlab.core.enums import StatisticalDistribution
from derivlab.models.black_scholes import BlackScholesModel
from derivlab.products.asian_option import AsianOption
from derivlab.products.barrier_option import BarrierOption
from derivlab.products.european_option import EuropeanOption


@dataclass(frozen=True)
class SimulationConfig:

    n_paths: int = 100_000 # Number of trajectories to be simulated
    n_steps: int = 252 # Number of time steps per path (i..e, discretisation grid points) -- default is trading days
    seed: int | None = 42 # Seed for random number generation
    distribution: StatisticalDistribution = StatisticalDistribution.NORMAL
    antithetic: bool = False


class MonteCarloPricer:

    method = "plain_monte_carlo"

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()

    def price(
                self,
                model: BlackScholesModel,
                option: EuropeanOption | AsianOption | BarrierOption,
                config: SimulationConfig | None = None,
            ) -> PricingResult:
        
        # Override of the pricer default config is allowed for one experiment.
        cfg = config or self.config
        start = perf_counter()

        # Simulate paths according to the model of choice, refers to dedicated class under "models"
        paths = model.simulate_paths(
                                        option.maturity,
                                        cfg.n_paths,
                                        cfg.n_steps,
                                        seed=cfg.seed,
                                        antithetic=cfg.antithetic,
                                    )

        # Risk-neutral pricing discounts expected payoff at the money-market numeraire: V_0 = exp(-rT) * E_Q[payoff].
        discount_factor = exp(-model.rate * option.maturity)
        payoffs = self._payoffs(paths, option)

        # Each sample is one discounted simulated payoff. The price estimate is the sample mean; accuracy is measured by the standard error.
        samples = discount_factor * payoffs
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
                                            },
                            )

    @staticmethod
    def _payoffs(paths: np.ndarray, option: EuropeanOption | AsianOption | BarrierOption) -> np.ndarray:
        
        # produce payoffs from generated paths
        if isinstance(option, EuropeanOption):
            return option.payoff(paths[:, -1])
        if isinstance(option, AsianOption):
            return option.payoff_from_paths(paths)
        if isinstance(option, BarrierOption):
            return option.payoff_from_paths(paths)
        raise TypeError("unsupported option type")
