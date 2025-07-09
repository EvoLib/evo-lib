# SPDX-License-Identifier: MIT
"""Registry mapping ReplacementStrategy enums to replacement functions."""

from functools import partial
from typing import Callable

from evolib.config.schema import ReplacementConfig
from evolib.interfaces.enums import ReplacementStrategy
from evolib.operators.replacement import (
    replace_generational,
    replace_random,
    replace_steady_state,
    replace_truncation,
    replace_weighted_stochastic,
)


def build_replacement_registry(
    cfg: ReplacementConfig,
) -> dict[ReplacementStrategy, Callable]:
    return {
        ReplacementStrategy.TRUNCATION: replace_truncation,
        ReplacementStrategy.GENERATIONAL: replace_generational,
        ReplacementStrategy.RANDOM: replace_random,
        ReplacementStrategy.STEADY_STATE: partial(
            replace_steady_state, num_replace=cfg.num_replace or 1
        ),
        ReplacementStrategy.STOCHASTIC: partial(
            replace_weighted_stochastic, temperature=cfg.temperature or 1.0
        ),
    }
