# SPDX-License-Identifier: MIT
"""
Maps strategy identifiers to their corresponding evolution functions.

This registry enables the selection and execution of different evolutionary strategies
by referencing their symbolic identifiers defined in the `Strategy` enum.

Supported strategies:
- Strategy.MU_PLUS_LAMBDA: (μ + λ) strategy using combined parent and offspring pool.
- Strategy.MU_COMMA_LAMBDA: (μ, λ) strategy using offspring-only selection.
- Strategy.STEADY_STATE: Steady-state replacement with incremental updates.

Usage:
    evolve_fn = strategy_registry[Strategy.MU_PLUS_LAMBDA]
    evolve_fn(pop)

Note:
Each evolution function must accept a `Pop` object and update it in-place
across one generation.
"""

from evolib.interfaces.enums import EvolutionStrategy
from evolib.interfaces.types import EvolutionStrategyFunction
from evolib.operators.strategy import (
    evolve_flexible,
    evolve_mu_comma_lambda,
    evolve_mu_plus_lambda,
    evolve_steady_state,
)

strategy_registry: dict[EvolutionStrategy, EvolutionStrategyFunction] = {
    EvolutionStrategy.MU_PLUS_LAMBDA: evolve_mu_plus_lambda,
    EvolutionStrategy.MU_COMMA_LAMBDA: evolve_mu_comma_lambda,
    EvolutionStrategy.STEADY_STATE: evolve_steady_state,
    EvolutionStrategy.FLEXIBLE: evolve_flexible,
}
