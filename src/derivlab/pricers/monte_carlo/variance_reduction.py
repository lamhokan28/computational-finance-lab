from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from time import perf_counter

import numpy as np

from derivlab.core.distributions import ppf
from derivlab.core.enums import AverageType, ControlVariateType, StatisticalDistribution
from derivlab.core.random import make_rng
from derivlab.core.results import PricingResult
from derivlab.models.black_scholes import BlackScholesModel
from derivlab.pricers.analytic.black_scholes import geometric_asian_price
from derivlab.pricers.monte_carlo.engine import MonteCarloPricer, SimulationConfig
from derivlab.products.asian_option import AsianOption
from derivlab.products.barrier_option import BarrierOption
from derivlab.products.european_option import EuropeanOption

### Antithetic Variates ###
class AntitheticMonteCarloPricer(MonteCarloPricer):

    method = "monte_carlo_antithetic_variate"

    def price(
                self,
                model: BlackScholesModel,
                option: EuropeanOption | AsianOption | BarrierOption,
                config: SimulationConfig | None = None,
            ) -> PricingResult:

        cfg = config or self.config
        if cfg.n_paths < 2:
            raise ValueError("antithetic sampling requires at least two paths")

        start = perf_counter()
        
        n_pairs = cfg.n_paths // 2
        rng = make_rng(cfg.seed)
        shocks = rng.standard_normal((n_pairs, cfg.n_steps))
        paths = model.paths_from_shocks(option.maturity, shocks) # regular shocks
        antithetic_paths = model.paths_from_shocks(option.maturity, -shocks) # antithetic shocks

        # Risk-neutral pricing discounts expected payoff at the money-market numeraire: V_0 = exp(-rT) * E_Q[payoff].
        discount_factor = exp(-model.rate * option.maturity)
        
        # Each sample is one discounted simulated payoff. The price estimate is the sample mean; accuracy is measured by the standard error.
        samples = 0.5 * discount_factor * (self._payoffs(paths, option) + self._payoffs(antithetic_paths, option))
        price = float(np.mean(samples))
        stderr = float(np.std(samples, ddof=1) / sqrt(len(samples)))
        
        return PricingResult(
                                price=price,
                                stderr=stderr,
                                runtime=perf_counter() - start,
                                method=self.method,
                                metadata={
                                                "n_paths": 2 * n_pairs,
                                                "n_pairs": n_pairs,
                                                "n_steps": cfg.n_steps,
                                                "seed": cfg.seed,
                                                "antithetic": True,
                                            },
                            )


@dataclass(frozen=True)
class StratifiedSamplingConfig(SimulationConfig):

    n_strata: int = 100 # Number of strata for stratified sampling

    def __post_init__(self) -> None:
        if not isinstance(self.n_strata, int):
            raise TypeError("n_strata must be an integer")
        if self.n_strata <= 1:
            raise ValueError("n_strata must be greater than 1")
        if self.n_paths < self.n_strata:
            raise ValueError("n_paths must be at least n_strata")
        if self.n_paths % self.n_strata != 0:
            raise ValueError("n_paths must be divisible by n_strata")

### Stratified Sampling ###
class StratifiedMonteCarloPricer(MonteCarloPricer):
    """Monte Carlo pricer using randomized stratified normal shocks."""

    method = "stratified_monte_carlo"

    def __init__(self, config: StratifiedSamplingConfig | None = None) -> None:
        self.config = config or StratifiedSamplingConfig()

    def price(
                self,
                model: BlackScholesModel,
                option: EuropeanOption | AsianOption,
                config: StratifiedSamplingConfig | None = None,
            ) -> PricingResult:

        cfg = config or self.config
        if not isinstance(cfg, StratifiedSamplingConfig):
            raise TypeError("StratifiedMonteCarloPricer requires StratifiedSamplingConfig ")
        start = perf_counter()

        shocks, stratum_ids = self._stratified_sampler(cfg.n_paths, cfg.n_steps, cfg.seed, cfg.n_strata, cfg.distribution)
        paths = model.paths_from_shocks(option.maturity, shocks)

        # Risk-neutral pricing discounts expected payoff at the money-market numeraire: V_0 = exp(-rT) * E_Q[payoff].
        discount_factor = exp(-model.rate * option.maturity)
        
        # Each sample is one discounted simulated payoff. The price estimate is the sample mean; accuracy is measured by the standard error.
        samples = discount_factor * self._payoffs(paths, option)
        price = float(np.mean(samples))
        stderr = self._stratified_standard_error(samples, stratum_ids, cfg.n_strata, cfg.n_steps)

        return PricingResult(
                                price=price,
                                stderr=stderr,
                                runtime=perf_counter() - start,
                                method=self.method,
                                metadata={
                                                "n_paths": cfg.n_paths,
                                                "n_steps": cfg.n_steps,
                                                "seed": cfg.seed,
                                                "n_strata": cfg.n_strata,
                                                "samples_per_stratum": cfg.n_paths // cfg.n_strata,
                                            },
                            )
    
    @staticmethod
    def _stratified_sampler(
                            n_paths: int, n_steps: int, seed: int | None, 
                            n_strata: int, dist: StatisticalDistribution
                        ) -> tuple[np.ndarray, np.ndarray]:
        # Validate Inputs
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if n_strata <= 1:
            raise ValueError("n_strata must be greater than 1")
        if n_paths % n_strata != 0:
            raise ValueError("n_paths must be divisible by n_strata")

        rng = make_rng(seed)
        samples_per_stratum = n_paths // n_strata
        base_strata = np.repeat(np.arange(n_strata), samples_per_stratum)
        shocks = np.empty((n_paths, n_steps))
        first_step_strata = np.empty(n_paths, dtype=int)

        for step in range(n_steps):
            stratum_ids = rng.permutation(base_strata)
            uniforms = (stratum_ids + rng.random(n_paths)) / n_strata
            shocks[:, step] = ppf(uniforms, dist)
            if step == 0:
                first_step_strata = stratum_ids

        return shocks, first_step_strata

    @staticmethod
    def _stratified_standard_error(
                                        samples: np.ndarray, stratum_ids: np.ndarray, n_strata: int, n_steps: int
                                    ) -> float:
        if n_steps != 1:
            return float(np.std(samples, ddof=1) / sqrt(len(samples)))

        stratum_variance_sum = 0.0
        for stratum in range(n_strata):
            stratum_samples = samples[stratum_ids == stratum]
            p_i = 1.0 / n_strata
            n_i = len(stratum_samples)
            stratum_variance_sum += p_i**2 * float(np.var(stratum_samples, ddof=1)) / n_i

        return sqrt(stratum_variance_sum)

### Control Variates ###
@dataclass(frozen=True)
class ControlVariateConfig(SimulationConfig):
    # placeholder for control-variate specific settings
    pass


class ControlVariatePricer:

    method = "monte_carlo_control_variate"

    def __init__(
                    self,
                    control_type: ControlVariateType = ControlVariateType.DISCOUNTED_STOCK,
                    config: ControlVariateConfig | None = None,
                ) -> None:
        
        self.control_type = control_type
        self.config = config or ControlVariateConfig()


    def _control_variate_parser(
                                self,
                                model: BlackScholesModel,
                                option: EuropeanOption | AsianOption,
                                paths: np.ndarray,
                            ) -> tuple[np.ndarray, np.ndarray, float]:
        
        if self.control_type is ControlVariateType.GEOMETRIC_ASIAN:
            return self._geometric_asian_control(model, option, paths)
        if self.control_type is ControlVariateType.DISCOUNTED_STOCK:
            return self._discounted_stock_control(model, option, paths)
        raise ValueError(f"unsupported control variate type: {self.control_type}")


    def price(
                self,
                model: BlackScholesModel,
                option: EuropeanOption | AsianOption,
                config: ControlVariateConfig | None = None,
            ) -> PricingResult:
        
        cfg = config or self.config
        start = perf_counter()

        # Shared randomness matters: X and Y must be computed on the same paths
        # so that their sample covariance captures their co-movement.
        paths = model.simulate_paths(option.maturity, cfg.n_paths, cfg.n_steps, cfg.seed, cfg.antithetic)
        discounted_target, discounted_control, known_control = self._control_variate_parser(model, option, paths)

        covariance = np.cov(discounted_target, discounted_control, ddof=1)
        cov_xy = covariance[0, 1]
        var_y = covariance[1, 1]
        var_x = covariance[0, 0]
        if var_y <= 0:
            raise ValueError("control payoff has zero variance; cannot compute beta")

        # Calculate OLS regression coefficients analytically
        beta = cov_xy / var_y
        correlation = cov_xy / sqrt(var_x * var_y) if var_x > 0 else 0.0
        
        # Control variate estimator
        cv_estimator = discounted_target - beta * (discounted_control - known_control)
        price = float(np.mean(cv_estimator))
        stderr = float(np.std(cv_estimator, ddof=1) / sqrt(len(cv_estimator)))

        return PricingResult(
                                price=price,
                                stderr=stderr,
                                runtime=perf_counter() - start,
                                method=self.method,
                                metadata={
                                                "beta": float(beta),
                                                "correlation": float(correlation),
                                                "control_type": self.control_type.value,
                                                "n_paths": cfg.n_paths,
                                                "n_steps": cfg.n_steps,
                                                "seed": cfg.seed,
                                            },
                            )


    @staticmethod
    def _geometric_asian_control(
                                    model: BlackScholesModel,
                                    option: EuropeanOption | AsianOption,
                                    paths: np.ndarray,
                                ) -> tuple[np.ndarray, np.ndarray, float]:
        
        if not isinstance(option, AsianOption) or option.average is not AverageType.ARITHMETIC:
            raise ValueError("geometric Asian control requires an arithmetic Asian target option")

        # Target Y: discounted arithmetic Asian payoff.
        arithmetic_avg = option.average_from_paths(paths)
        target = option.payoff(arithmetic_avg)
        discounted_target = exp(-model.rate * option.maturity) * target

        # Control X: discounted geometric Asian payoff
        geometric_option = AsianOption(option.kind, option.strike, option.maturity, average=AverageType.GEOMETRIC)
        geometric_avg = geometric_option.average_from_paths(paths)
        control = geometric_option.payoff(geometric_avg)
        discounted_control = exp(-model.rate * option.maturity) * control
        
        # Known control E[X]: geometric Asian price with Black-Scholes closed-form formula
        known_control = geometric_asian_price(model, geometric_option)

        return discounted_target, discounted_control, known_control

    @staticmethod
    def _discounted_stock_control(
                                        model: BlackScholesModel,
                                        option: EuropeanOption | AsianOption,
                                        paths: np.ndarray,
                                    ) -> tuple[np.ndarray, np.ndarray, float]:
        
        if not isinstance(option, EuropeanOption):
            raise ValueError("discounted stock control variate requires a European target option")

        # Target Y: discounted European payoff.
        terminal_S = paths[:, -1]
        discounted_payoff = exp(-model.rate * option.maturity) * option.payoff(terminal_S)

        # Control Variate X: discounted terminal value of stock (exp(-rT) * E[S_T])
        discounted_control = exp(-model.rate * option.maturity) * terminal_S
        
        # Known control E[X]: S0 * exp(-qT)
        known_control = model.spot * exp(-model.dividend * option.maturity)

        return discounted_payoff, discounted_control, known_control
