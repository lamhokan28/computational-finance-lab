from __future__ import annotations

from enum import Enum, IntEnum


# ___________Enumerations____________
class OptionKind(IntEnum):
    CALL = 1
    PUT = -1

    @property
    def label(self) -> str:
        return "call" if self is OptionKind.CALL else "put"


class AverageType(Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


class BarrierDirection(Enum):
    UP = "up"
    DOWN = "down"


class BarrierKnock(Enum):
    IN = "in"
    OUT = "out"


class ControlVariateType(Enum):
    """Available control variables for the control-variate method."""
    GEOMETRIC_ASIAN = "geometric_asian"
    DISCOUNTED_STOCK = "discounted_stock"


class StatisticalDistribution(Enum):
    NORMAL = "normal"

# __________Parsers___________
def _parse_enum(name: str, value: Enum | str, enum_type: type[Enum]) -> Enum:
    if isinstance(value, enum_type):
        return value
    normalized = str(value).lower()
    for member in enum_type:
        if member.value == normalized:
            return member
    raise _choice_error(name, enum_type)


def _choice_error(name: str, enum_type: type[Enum]) -> ValueError:
    allowed = ", ".join(str(member.value).lower() for member in enum_type)
    return ValueError(f"{name} must be one of: {allowed}")


def parse_option_kind(kind: OptionKind | str) -> OptionKind:
    if isinstance(kind, OptionKind):
        return kind
    normalized = str(kind).lower()
    if normalized == "call":
        return OptionKind.CALL
    if normalized == "put":
        return OptionKind.PUT
    raise _choice_error("kind", OptionKind)


def parse_average_type(average: AverageType | str) -> AverageType:
    return _parse_enum("average", average, AverageType)


def parse_barrier_direction(direction: BarrierDirection | str) -> BarrierDirection:
    return _parse_enum("direction", direction, BarrierDirection)


def parse_barrier_knock(knock: BarrierKnock | str) -> BarrierKnock:
    return _parse_enum("knock", knock, BarrierKnock)