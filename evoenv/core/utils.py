# SPDX-License-Identifier: MIT


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a scalar value to a closed interval."""
    return max(lower, min(upper, float(value)))


def clamp01(value: float) -> float:
    """Clamp a scalar value to [0, 1]."""
    return clamp(value, 0.0, 1.0)
