# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Optional


@dataclass
class MutationParams:
    strength: float
    min_strength: float
    max_strength: float
    probability: float
    min_probability: float
    max_probability: float
    bounds: tuple[float, float]
    bias: Optional[float] = None
    tau: Optional[float] = None
