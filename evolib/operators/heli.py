# SPDX-License-Identifier: MIT
"""
HELI (Hierarchical Evolution with Lineage Incubation) operator.

This module provides the logic for running short micro-evolutions ("incubations")
on structure-mutated individuals of a specific module (e.g., EvoNet).
The goal is to allow newly created topologies to stabilize before rejoining
the main population.

Usage
-----
Called from within a module mutation routine, e.g.:

    if self.heli_cfg:
        run_heli(pop, offspring, self.heli_cfg)

Design
------
- HELI is *module-local*: each module decides independently
  whether and how to run incubation.
- HELI does *not* alter global evolution strategy or scheduling.
- Fitness evaluations during incubation are isolated from the main loop.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from evolib.core.individual import Indiv
    from evolib.core.population import Pop

from evolib.utils.heli_utils import (
    apply_heli_overrides,
    backup_module_state,
    restore_module_state,
)


def run_heli(pop: "Pop", offspring: List["Indiv"]) -> None:
    """
    Run HELI incubation for structure-mutated offspring.

    Parameters
    ----------
    pop : Population
        The main population context, used for configuration and
        access to evolutionary operators.
    offspring : list[Indiv]
        Offspring individuals from the main generation.
        Structure-mutated individuals will be extracted and incubated.

    Notes
    -----
    - Structure-mutated individuals are *temporarily removed* from `offspring`
      to avoid double evaluation.
    - Only the best individual from each incubation subpopulation is returned.
    - Mutation strength can be damped by `reduce_sigma_factor`.
    """

    from evolib.core.population import Pop
    from evolib.operators.strategy import evolve_mu_plus_lambda

    if not offspring or not pop.heli_enabled:
        return

    # 1: Select structure-mutated offspring
    struct_mutants = [indiv for indiv in offspring if indiv.para.has_structural_change]
    if not struct_mutants:
        if pop.heli_verbosity >= 2:
            print(f"[HELI] Gen: {pop.generation_num} - No struct_mutants")
        return

    if pop.heli_verbosity >= 2:
        print(f"[HELI] Start: Number of structural mutants: {len(struct_mutants)}")

    # 2: Limit number of incubated seeds
    max_seeds = max(1, round(len(offspring) * pop.heli_max_fraction))
    seeds = struct_mutants[:max_seeds]

    if len(seeds) < 1:
        if pop.heli_verbosity >= 2:
            print(f"[HELI] Gen: {pop.generation_num} - No Seed")
        return

    # Remove selected seeds from the main offspring pool
    for seed in seeds:
        if seed in offspring:
            offspring.remove(seed)

    new_candidates = []

    if pop.heli_verbosity >= 1:
        print(f"[HELI] Running for {len(seeds)} candidates")

    # 3: Incubate each selected seed
    for seed_idx, seed in enumerate(seeds):
        if pop.heli_verbosity >= 1:
            print(f"[HELI] Candidate: {seed_idx+1}")

        # Create SubPopulation
        cfg = deepcopy(pop.config)

        # Deactivate HELI in SubPopulation Config
        if cfg.evolution is not None:
            cfg.evolution.heli = None

        subpop = Pop.from_config(
            cfg, fitness_function=pop.fitness_function, initialize=False
        )
        subpop.indivs = [seed.copy()]
        subpop.parent_pool_size = 1
        subpop.offspring_pool_size = pop.heli_offspring_per_seed
        subpop.max_generations = pop.heli_generations
        subpop.heli_enabled = False

        heli_backup = {}

        indiv = subpop.indivs[0]
        para_dict = vars(indiv.para)
        comp_dict = para_dict["components"]

        # Backup original mutation config
        for module_name, module in comp_dict.items():
            heli_backup[module_name] = backup_module_state(module)
            apply_heli_overrides(module, pop.heli_reduce_sigma_factor)

        # Run short local evolution
        for gen in range(pop.heli_generations):
            evolve_mu_plus_lambda(subpop)
            best = subpop.best()
            if pop.heli_verbosity >= 2:
                print(
                    f"[HELI] Candidate: {seed_idx+1}/{len(seeds)} "
                    f"Gen: {gen+1}/pop.generation_num Fit: {best.fitness:.3f}"
                )

        # Restore evo_params
        para_dict = vars(best.para)
        comp_dict = para_dict["components"]

        for module_name, module in comp_dict.items():
            restore_module_state(module, heli_backup.get(module_name, {}))

        # 4: Reintegration
        new_candidates.append(best)

    # 5: Reattach improved candidates to the main offspring
    offspring.extend(new_candidates)

    if pop.heli_verbosity >= 2:
        print("[HELI] End")
