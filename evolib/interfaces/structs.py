# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Optional


@dataclass
class MutationParams:
    strength: float
    min_strength: float
    max_strength: float
    probabality: float
    min_probabality: float
    max_probabality: float
    bounds: tuple[float, float]
    bias: Optional[float] = None
