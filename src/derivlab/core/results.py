from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PricingResult:

    price: float
    method: str
    stderr: float | None = None
    runtime: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

