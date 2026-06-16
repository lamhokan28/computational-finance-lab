from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from time import perf_counter

import numpy as np

from derivlab.core.enums import OptionKind
from derivlab.core.results import PricingResult
from derivlab.models.black_scholes import BlackScholesModel
from derivlab.pricers.monte_carlo.engine import SimulationConfig
from derivlab.products.american import AmericanOption


@dataclass(frozen=True)
class LSMCConfig(SimulationConfig):

    # inherents master simulation config
    # polynomial_degree of the regression function
    polynomial_degree: int = 2


class LSMCPricer:
    """Longstaff-Schwartz Monte Carlo for American puts."""

    method = "least_squares_monte_carlo"

    def __init__(self, config: LSMCConfig | None = None) -> None:
        self.config = config or LSMCConfig(n_paths=50_000, n_steps=50, seed=42, polynomial_degree=2)

    def price(self, model: BlackScholesModel, option: AmericanOption, config: LSMCConfig | None = None) -> PricingResult:
        if option.kind is not OptionKind.PUT:
            raise ValueError("LSMC is currently only implemented for American puts")
        cfg = config or self.config
        start = perf_counter()

        # Simulate all paths first.
        paths = model.simulate_paths(option.maturity, cfg.n_paths, cfg.n_steps, cfg.seed, cfg.antithetic)
        dt = option.maturity / cfg.n_steps
        discount_factor = exp(-model.rate * dt)

        cashflows = option.payoff(paths[:, -1]) # No continuation value at maturity

        # Move backward through possible exercise dates. Step 0 is today, so the
        # loop stops at 1; the final discount below brings values to time 0.
        for step in range(cfg.n_steps - 1, 0, -1):
            spot = paths[:, step]
            exercise = option.payoff(spot)

            # Only in-the-money paths are relevant for the exercise decision;
            # out-of-the-money puts have zero immediate exercise value.
            itm = exercise > 0

            # Bring all future cashflows back by one time step before comparing
            # them with exercise value at the current date.
            cashflows *= discount_factor

            # Regression needs more observations than basis functions. If too
            # few paths are in the money, keep continuation values unchanged.
            if np.count_nonzero(itm) <= cfg.polynomial_degree + 1:
                continue

            # Regress discounted future cashflows on current spot for in-the-
            # money paths. The fitted values approximate continuation value.
            coefficients = np.polyfit(spot[itm], cashflows[itm], deg=cfg.polynomial_degree)
            continuation = np.polyval(coefficients, spot[itm])

            # Exercise where immediate payoff exceeds estimated continuation.
            # For those paths, replace the future cashflow with termination.
            termination = exercise[itm] > continuation
            indices = np.flatnonzero(itm)
            cashflows[indices[termination]] = exercise[indices[termination]]

        # Discount one final step from the first exercise date back to today,
        # then average across paths.
        samples = cashflows * discount_factor
        return PricingResult(
                                price=float(np.mean(samples)),
                                stderr=float(np.std(samples, ddof=1) / sqrt(len(samples))),
                                runtime=perf_counter() - start,
                                method=self.method,
                                metadata={
                                            "n_paths": cfg.n_paths,
                                            "n_steps": cfg.n_steps,
                                            "seed": cfg.seed,
                                            "polynomial_degree": cfg.polynomial_degree,
                                        },
                            )
