# SPDX-License-Identifier: MIT
import random
from typing import Optional

import numpy as np


def set_random_seed(seed: Optional[int]) -> None:
    """Set random seed for reproducibility; no-op if None."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
