from __future__ import annotations

import numpy as np
from math import exp, log, sqrt
from time import perf_counter

from derivlab.core.distributions import cdf
from derivlab.core.enums import AverageType, BarrierKnock, StatisticalDistribution
from derivlab.core.results import PricingResult
from derivlab.models.black_scholes import BlackScholesModel
from derivlab.products.asian_option import AsianOption
from derivlab.products.barrier_option import BarrierOption
from derivlab.products.european_option import EuropeanOption


class BlackScholesAnalyticPricer:
    method = "analytic_black_scholes"

    def price(self, model: BlackScholesModel, option: EuropeanOption | AsianOption | BarrierOption, config=None) -> PricingResult:
        start = perf_counter()
        if isinstance(option, EuropeanOption):
            value = black_scholes_price(model, option)
        elif isinstance(option, AsianOption) and option.average is AverageType.GEOMETRIC:
            value = geometric_asian_price(model, option)
        elif isinstance(option, BarrierOption):
            value = single_barrier_price(model, option)
        else:
            raise TypeError("analytic Black-Scholes only supports European, geometric Asian, and selected barrier options")
        return PricingResult(price=float(value), method=self.method, runtime=perf_counter() - start)


def d1_d2(model: BlackScholesModel, option: EuropeanOption) -> tuple[float, float]:
    s = model.spot
    k = option.strike
    t = option.maturity
    sigma = model.volatility
    
    if sigma == 0:
        raise ValueError("Black-Scholes d1/d2 are undefined for zero volatility")
    
    d1 = (log(s / k) + (model.rate - model.dividend + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    
    return d1, d2


def black_scholes_price(model: BlackScholesModel, option: EuropeanOption) -> float:
    s = model.spot
    k = option.strike
    t = option.maturity
    df_r = exp(-model.rate * t)
    df_q = exp(-model.dividend * t)
    phi = int(option.kind)
    
    if model.volatility == 0:
        forward_intrinsic = s * exp((model.rate - model.dividend) * t) - k
        return df_r * max(phi * forward_intrinsic, 0.0)
    
    d1, d2 = d1_d2(model, option)
    
    return phi * (s * df_q * cdf(phi * d1, StatisticalDistribution.NORMAL) - k * df_r * cdf(phi * d2, StatisticalDistribution.NORMAL))


def geometric_asian_price(model: BlackScholesModel, option: AsianOption) -> float:
    s = model.spot
    k = option.strike
    t = option.maturity
    phi = int(option.kind)
    sigma_g = model.volatility / sqrt(3.0)
    b_g = 0.5 * (model.rate - model.dividend - 0.5 * model.volatility**2) + 0.5 * sigma_g**2
    df = exp(-model.rate * t)
    
    if sigma_g == 0:
        avg_forward = s * exp(b_g * t)
        return df * max(phi * (avg_forward - k), 0.0)
    
    d1 = (log(s / k) + (b_g + 0.5 * sigma_g**2) * t) / (sigma_g * sqrt(t))
    d2 = d1 - sigma_g * sqrt(t)
    
    return df * phi * (s * exp(b_g * t) * cdf(phi * d1, StatisticalDistribution.NORMAL) - k * cdf(phi * d2, StatisticalDistribution.NORMAL))


def single_barrier_price(model: BlackScholesModel, option: BarrierOption) -> float:
    
    r = model.rate
    s = model.spot
    q = model.dividend
    k = option.strike
    b = option.barrier
    t = option.maturity
    sigma = model.volatility
    cp = option.kind
    if option.flag == "XYZ":
        raise ValueError("Option flag is not properly defined.")
    
    # If the barrier is already touched, knock-outs are dead and knock-ins have already become the corresponding vanilla European option.
    if option.is_breached_by_spot(s):
        vanilla = black_scholes_price(model, EuropeanOption(option.kind, k, t))
        return vanilla if option.knock is BarrierKnock.IN else 0.0
    
    # appropriate power fund such that the process is a Gaussian continuous martingale
    eta = (q - r) / (sigma**2) + 0.5
    
    # Spot prices of numeraire
    mma_spot = exp(-r * t)
    stock_spot = s * exp(-q * t)
    two_eta_spot = s ** (2 * eta) * exp(-r * t)
    two_eta_minus_one_spot = s ** (2 * eta - 1) * exp(-q * t)
    
    # Forward prices
    # Drifts
    mma_drift = r - q - 0.5 * sigma**2
    stock_drift = r - q + 0.5 * sigma**2
    two_eta_drift = -r + q + 0.5 * sigma**2
    two_eta_minus_one_drift = -r + q - 0.5 * sigma**2
    
    normCDF = lambda omega, z, drift: cdf(omega * (log(s/z) + drift * t) / (sigma * sqrt(t)), StatisticalDistribution.NORMAL)
    between = lambda low, high, drift: normCDF(1, low, drift) - normCDF(1, high, drift)
    
    # forward price term with money market numeraire
    if option.flag == "UIC" or option.flag == "DOC":
        mma_fwd = (cp * -k) * normCDF(cp, np.maximum(b,k), mma_drift)
    elif option.flag == "UOP" or option.flag == "DIP":
        mma_fwd = (cp * -k) * normCDF(cp, np.minimum(b,k), mma_drift)
    elif option.flag == "UIP" or option.flag == "DOP":
        mma_fwd = (cp * -k) * (cp * (normCDF(1, k, mma_drift) - normCDF(1, b, mma_drift))) if k > b else 0
    else:
        # option.flag == "UOC" or option.flag == "DIC"
        mma_fwd = (cp * -k) * (cp * (normCDF(1, k, mma_drift) - normCDF(1, b, mma_drift))) if k < b else 0
    
    # forward price term with stock numeraire
    if option.flag == "UOP" or option.flag == "DIP":
        stock_fwd = cp * normCDF(cp, np.minimum(b,k), stock_drift) 
    elif option.flag == "UIC" or option.flag == "DOC":
        stock_fwd = cp * normCDF(cp, np.maximum(b,k), stock_drift)
    elif option.flag == "UIP" or option.flag == "DOP":
        stock_fwd = normCDF(1, k, stock_drift) - normCDF(1, b, stock_drift) if k > b else 0
    else:
        # option.flag == "UOC" or option.flag == "DIC"
        stock_fwd = normCDF(1, k, stock_drift) - normCDF(1, b, stock_drift) if k < b else 0

    # reflected strike
    bk = b**2 / k
    # forward price term with 2eta numeraire
    coeff = k * b ** (-2 * eta)
    two_eta_fwd = 0

    if option.flag in ["UIC", "UOC"]:
        if b > k:
            sign = -1 if option.flag == "UIC" else 1
            two_eta_fwd = sign * coeff * between(b, bk, two_eta_drift)
    elif option.flag in ["UIP", "UOP"]:
        sign = 1 if option.flag == "UIP" else -1
        two_eta_fwd = sign * coeff * (
                                        normCDF(1, b, two_eta_drift) if k > b
                                        else normCDF(1, bk, two_eta_drift)
                                    )
    elif option.flag in ["DIC", "DOC"]:
        sign = -1 if option.flag == "DIC" else 1
        two_eta_fwd = sign * coeff * (
                                        normCDF(-1, bk, two_eta_drift) if k > b
                                        else normCDF(-1, b, two_eta_drift)
                                    )
    elif option.flag in ["DIP", "DOP"]:
        if k > b:
            sign = 1 if option.flag == "DIP" else -1
            two_eta_fwd = sign * coeff * between(bk, b, two_eta_drift)

    # forward price term with 2eta-1 numeraire
    coeff = b ** (2 - 2 * eta)
    two_eta_minus_one_fwd = 0

    if option.flag in ["UIC", "UOC"]:
        if b > k:
            sign = 1 if option.flag == "UIC" else -1
            two_eta_minus_one_fwd = sign * coeff * between(b, bk, two_eta_minus_one_drift)
    elif option.flag in ["UIP", "UOP"]:
        sign = -1 if option.flag == "UIP" else 1
        two_eta_minus_one_fwd = sign * coeff * (
                                                    normCDF(1, b, two_eta_minus_one_drift) if k > b
                                                    else normCDF(1, bk, two_eta_minus_one_drift)
                                                )
    elif option.flag in ["DIC", "DOC"]:
        sign = 1 if option.flag == "DIC" else -1
        two_eta_minus_one_fwd = sign * coeff * (
                                                    normCDF(-1, bk, two_eta_minus_one_drift) if k > b
                                                    else normCDF(-1, b, two_eta_minus_one_drift)
                                                )
    elif option.flag in ["DIP", "DOP"]:
        if k > b:
            sign = -1 if option.flag == "DIP" else 1
            two_eta_minus_one_fwd = sign * coeff * between(bk, b, two_eta_minus_one_drift)
            
    # Spot Price of Option = spot price of numeraire * forward price under numeraire
    return mma_spot * mma_fwd \
            + stock_spot * stock_fwd \
            + two_eta_spot * two_eta_fwd \
            + two_eta_minus_one_spot * two_eta_minus_one_fwd

