# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Optional


@dataclass
class MutationParams:
    strength: float
    min_strength: float
    max_strength: float
    rate: float
    min_rate: float
    max_rate: float
    bounds: tuple[float, float]
    bias: Optional[float] = None
