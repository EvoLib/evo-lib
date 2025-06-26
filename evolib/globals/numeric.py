# SPDX-License-Identifier: MIT
"""Numeric constants."""

# Smallest positive float used to prevent log(0) or division by zero
MIN_FLOAT = 1e-12

# Bigest positive float
MAX_FLOAT = float("inf")

# Small threshold for comparing floating-point values (e.g., abs(a - b) < EPSILON)
EPSILON = 1e-8

# Default float data type for internal computations
DEFAULT_FLOAT_DTYPE = "float64"
