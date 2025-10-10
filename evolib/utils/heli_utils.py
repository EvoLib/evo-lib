# SPDX-License-Identifier: MIT
"""
Utility functions for HELI (Hierarchical Evolution with Lineage Incubation).

These helpers handle the backup, modification, and restor of module-level evolution
parameters during HELI incubation phases.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def backup_module_state(module: Any) -> Dict[str, Any]:
    """
    Create a deep backup of all HELI-relevant attributes of a module.

    Returns

    dict     Contains deep copies of evo_params, structural_cfg, and
    activation_probability.
    """

    return {
        "evo_params": deepcopy(getattr(module, "evo_params", None)),
        "structural_cfg": deepcopy(getattr(module, "structural_cfg", None)),
        "activation_probability": getattr(module, "activation_probability", None),
    }


def apply_heli_overrides(module: Any, sigma_factor: float = 1.0) -> None:
    """
    Apply HELI-local parameter overrides for incubation phase.

    - Mutation strategy set to 'constant'
    - Mutation strength reduced by sigma_factor
    - Crossover disabled
    - Structural and activation mutations disabled
    """
    ep = getattr(module, "evo_params", None)
    if ep is not None:
        ep.mutation_strategy = "constant"
        if getattr(ep, "mutation_strength", None) is not None:
            ep.mutation_strength *= float(sigma_factor or 1.0)
        ep.crossover_strategy = None

    if hasattr(module, "structural_cfg"):
        module.structural_cfg = None

    if hasattr(module, "activation_probability"):
        module.activation_probability = 0.0


def restore_module_state(module: Any, backup: Dict[str, Any]) -> None:
    """
    Restore a module's parameters after HELI incubation.

    Restores evo_params (field-wise), structural_cfg, and activation_probability.
    """
    if not backup:
        return

    # evo_params
    if "evo_params" in backup and hasattr(module, "evo_params"):
        orig = backup["evo_params"]
        ep = module.evo_params
        if orig is not None and ep is not None:
            ep.mutation_strategy = getattr(
                orig, "mutation_strategy", ep.mutation_strategy
            )
            ep.mutation_strength = getattr(
                orig, "mutation_strength", ep.mutation_strength
            )
            ep.mutation_probability = getattr(
                orig, "mutation_probability", ep.mutation_probability
            )
            ep.crossover_strategy = getattr(
                orig, "crossover_strategy", ep.crossover_strategy
            )

    # structural_cfg
    if "structural_cfg" in backup:
        setattr(module, "structural_cfg", deepcopy(backup["structural_cfg"]))

    # activation_probability
    if "activation_probability" in backup and hasattr(module, "activation_probability"):
        orig_value = backup["activation_probability"]
        if orig_value is not None:
            module.activation_probability = orig_value
