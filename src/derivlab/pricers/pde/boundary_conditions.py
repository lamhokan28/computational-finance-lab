from __future__ import annotations

from math import exp

from derivlab.core.enums import OptionKind
from derivlab.products.american_option import AmericanOption
from derivlab.products.barrier_option import BarrierOption, BarrierDirection
from derivlab.products.european_option import EuropeanOption


def lower_boundary(option: EuropeanOption | AmericanOption | BarrierOption, kind: OptionKind, s_min: float, strike: float, rate: float, dividend: float, tau: float) -> float:

    if isinstance(option, BarrierOption) and option.direction is BarrierDirection.DOWN:
            # Lower bound is when spot = barrier --> knocked out (note that PDE only calculates knock out barrier price)
            return 0.0

    if kind is OptionKind.PUT:
        # Deep ITM, the put's payoff is approximately a forward contract. At zero spot, it's the discounted strike.
        return strike * exp(-rate * tau) - s_min * exp(-dividend * tau)
    # Deep OTM, the call is worthless, it is assumed that the s_min is selected low enough to reach OTM states
    return 0.0


def upper_boundary(option: EuropeanOption | AmericanOption | BarrierOption, kind: OptionKind, s_max: float, strike: float, rate: float, dividend: float, tau: float) -> float:

    if isinstance(option, BarrierOption) and option.direction is BarrierDirection.UP:
            # Upper bound is when spot = barrier --> knocked out (note that PDE only calculates knock out barrier price)
            return 0.0
    
    if kind is OptionKind.CALL:
        # Deep ITM, the call's payoff is approximately a forward contract
        return s_max * exp(-dividend * tau) - strike * exp(-rate * tau)
    # Deep OTM, the put is worthless, it is assumed that the s_max is selected high enough to reach OTM states
    return 0.0
