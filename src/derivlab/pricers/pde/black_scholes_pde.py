from __future__ import annotations
from dataclasses import replace
from collections.abc import Callable
from time import perf_counter

import numpy as np

from derivlab.core.enums import BarrierDirection, BarrierKnock
from derivlab.core.results import PricingResult
from derivlab.models.black_scholes import BlackScholesModel
from derivlab.pricers.analytic.black_scholes import black_scholes_price
from derivlab.pricers.pde.boundary_conditions import lower_boundary, upper_boundary
from derivlab.pricers.pde.grids import GridConfig
from derivlab.pricers.pde.solver import solve_tridiagonal
from derivlab.products.american import AmericanOption
from derivlab.products.barrier import BarrierOption
from derivlab.products.european import EuropeanOption

Boundary = Callable[[float], float]
Projection = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _fd_scheme_name_name(theta: float, prefix: str) -> str:
    if theta == 0.0:
        scheme = "explicit_pde"
    elif theta == 1.0:
        scheme = "implicit_pde"
    elif theta == 0.5:
        scheme = "crank_nicolson_pde"
    else:
        raise ValueError(f"Unknown theta value: {theta}")
    return f"{prefix}_{scheme}"


def _solve_theta_grid(
                        model: BlackScholesModel,
                        maturity: float,
                        payoff: Callable[[np.ndarray], np.ndarray],
                        spot: float,
                        config: GridConfig,
                        s_min: float,
                        s_max: float,
                        left_boundary: Boundary,
                        right_boundary: Boundary,
                        projection: Projection | None = None,
                    ) -> tuple[float, dict[str, float | int]]:
    
    """
    Shared Black-Scholes PDE solver backbone with theta as time-stepping scheme parsing parameter. 
    Please refer to methodology document for details
    """

    s_grid = np.linspace(s_min, s_max, config.n_space + 1)
    ds = s_grid[1] - s_grid[0]
    dt = maturity / config.n_time

    values = payoff(s_grid)
    values[0] = left_boundary(0.0)
    values[-1] = right_boundary(0.0)

    interior = np.arange(1, config.n_space)
    s = s_grid[interior]

    # Black-Scholes spatial operator coefficients (final expression after derivation):
    a = 0.5 * model.volatility**2 * s**2 / ds**2 - 0.5 * (model.rate - model.dividend) * s / ds
    b = -(model.volatility**2 * s**2 / ds**2 + model.rate)
    c = 0.5 * model.volatility**2 * s**2 / ds**2 + 0.5 * (model.rate - model.dividend) * s / ds

    # static input for tridiagonal matrix
    lower_diag = -config.theta * dt * a[1:]
    main_diag = 1.0 - config.theta * dt * b
    upper_diag = -config.theta * dt * c[:-1]

    for step in range(config.n_time):
        tau_old = step * dt
        tau_new = (step + 1) * dt
        old = values.copy()

        rhs = old[interior] + (1.0 - config.theta) * dt * (
            a * old[interior - 1] + b * old[interior] + c * old[interior + 1]
        )

        # adapt to spacial boundaries
        left_old = left_boundary(tau_old)
        right_old = right_boundary(tau_old)
        left_new = left_boundary(tau_new)
        right_new = right_boundary(tau_new)

        # update values according to spacial boundaries
        values[0] = left_new
        values[-1] = right_new
        rhs[0] += (1.0 - config.theta) * dt * a[0] * left_old + config.theta * dt * a[0] * left_new
        rhs[-1] += (1.0 - config.theta) * dt * c[-1] * right_old + config.theta * dt * c[-1] * right_new

        # solve the tridiagonal matrix
        values[interior] = solve_tridiagonal(lower_diag, main_diag, upper_diag, rhs)

        # projection of values (mainly for American / Bermudan early exercise option)
        if projection is not None:
            values = projection(values, s_grid)

    # Interpolate values according to current spot
    price = np.interp(spot, s_grid, values)
    
    metadata = {
                    "s_min": s_min,
                    "s_max": s_max,
                    "n_space": config.n_space,
                    "n_time": config.n_time,
                    "theta": config.theta,
                }
    
    return float(price), metadata


class BlackScholesPDEPricer:
    """Finite-difference solver for vanilla European, American, single barrier options."""

    def __init__(self, config: GridConfig | None = None) -> None:
        self.config = config or GridConfig()
        self.method = _fd_scheme_name_name(self.config.theta, "black_scholes")

    def price(
                self,
                model: BlackScholesModel,
                option: EuropeanOption | AmericanOption | BarrierOption,
                config: GridConfig | None = None,
            ) -> PricingResult:
        
        cfg = config or self.config
        option_clone = option
        vanilla = EuropeanOption(option.kind, option.strike, option.maturity)
        
        # Case specific setting for Barrier Option
        if isinstance(option, BarrierOption):
            
            # PDE to parse knock out option only, knock in options are priced with vanilla - knock out
            if option.knock is BarrierKnock.IN:
                option_clone = replace(option, knock=BarrierKnock.OUT)
            
            # Fast track when knock out option is knocked out at inception
            if option.is_breached_by_spot(model.spot):
                price = 0.0 if option.knock == BarrierKnock.OUT else black_scholes_price(model, vanilla)
                return PricingResult(
                                        price=price,
                                        runtime=0.0,
                                        method=_fd_scheme_name_name(cfg.theta, "black_scholes_barrier"),
                                        metadata={"barrier_breached_at_inception": True},
                                    )
        
        # Determine projection function
        projection = None
        if isinstance(option_clone, AmericanOption):
            projection = lambda values, s_grid: np.maximum(values, option_clone.payoff(s_grid))

        start = perf_counter()
        s_min, s_max = self._domain(option_clone, cfg)
        price, metadata = _solve_theta_grid(
                                                model=model,
                                                maturity=option_clone.maturity,
                                                payoff=option_clone.payoff,
                                                spot=model.spot,
                                                config=cfg,
                                                s_min=s_min,
                                                s_max=s_max,
                                                left_boundary=lambda tau: lower_boundary(
                                                    option_clone, option.kind, cfg.s_min, option.strike, model.rate, model.dividend, tau
                                                ),
                                                right_boundary=lambda tau: upper_boundary(
                                                    option_clone, option.kind, cfg.s_max, option.strike, model.rate, model.dividend, tau
                                                ),
                                                projection=projection,
                                            )
        
        if isinstance(option, BarrierOption) and option.knock is BarrierKnock.IN:
            # knock in = vanilla - knock out
            price = black_scholes_price(model, vanilla) - price
        
        return PricingResult(
                                price=price,
                                runtime=perf_counter() - start,
                                method=_fd_scheme_name_name(cfg.theta, "black_scholes"),
                                metadata=metadata,
                            )


    @staticmethod
    def _domain(option: EuropeanOption | AmericanOption | BarrierOption, cfg: GridConfig) -> tuple[float, float]:
        
        if isinstance(option, BarrierOption):
            if option.direction is BarrierDirection.UP:
                if cfg.s_min >= option.barrier:
                    raise ValueError("s_min must be below the up barrier")
                return cfg.s_min, option.barrier
            if cfg.s_max <= option.barrier:
                raise ValueError("s_max must be above the down barrier")
            return option.barrier, cfg.s_max
        return cfg.s_min, cfg.s_max

