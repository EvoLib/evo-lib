# SPDX-License-Identifier: MIT
"""
This module defines enumerations used to represent common categorical values such as
origin types and status indicators.

Usage:
    from evolib.globals.enums import Origin

    if indiv.origin == Origin.PARENT:
        ...
"""

from enum import Enum


class Origin(Enum):
    PARENT = "parent"
    OFFSPRING = "offspring"


class Strategy(Enum):
    MU_PLUS_LAMBDA = "mu_plus_lambda"
    MU_COMMA_LAMBDA = "mu_comma_lambda"


class MutationStrategy(Enum):
    EXPONENTIAL_DECAY = "exponential_decay"
    ADAPTIVE_GLOBAL = "adaptive_global"
    ADAPTIVE_INDIVIDUAL = "adaptive_individual"
    ADAPTIVE_PER_PARAMETER = "adaptive_per_parameter"
    CONSTANT = "constant"


class CrossoverStrategy(Enum):
    NONE = "none"
    EXPONENTIAL_DECAY = "exponential_decay"
    ADAPTIVE_GLOBAL = "adaptive_global"
    CONSTANT = "constant"


class Representation(Enum):
    NEURAL = "neural"
    VECTOR = "vector"


class DiversityMethod(Enum):
    IQR = "iqr"
    RELATIVE_IQR = "relative_iqr"  # (IQR / median)
    STD = "std"
    VAR = "var"
    RANGE = "range"
    NORMALIZED_STD = "normalized_std"
