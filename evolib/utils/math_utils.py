# SPDX-License-Identifier: MIT

import numpy as np


def scaled_mutation_factor(tau: float) -> float:
    """
    Returns a scaling factor for mutation strength based on a normal-distributed noise
    and a given tau parameter.

    Args:
        tau (float): Learning rate for log-normal update. Typically 1/sqrt(n).

    Returns:
        float: A multiplicative factor sampled from exp(tau * N(0,1)).
    """
    if tau <= 0:
        return 1.0  # keine Skalierung
    return np.exp(tau * np.random.normal())


def clip_mutation_strength(
    value: float, min_strength: float, max_strength: float
) -> float:
    """
    Clips the mutation strength to lie within [min_strength, max_strength].

    Args:
        value (float): Current mutation strength value.
        min_strength (float): Minimum allowed strength.
        max_strength (float): Maximum allowed strength.

    Returns:
        float: Clipped mutation strength.
    """
    return float(np.clip(value, min_strength, max_strength))


def clip(value: float, min_value: float, max_value: float) -> float:
    """
    Clips the value to lie within [min_value, max_value].

    Args:
        value (float): Current mutation rate value.
        min_value (float): Minimum allowed value.
        max_value (float): Maximum allowed value.

    Returns:
        float: Clipped value.
    """
    return float(np.clip(value, min_value, max_value))
