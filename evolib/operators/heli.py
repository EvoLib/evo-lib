# SPDX-License-Identifier: MIT
"""
HELI (Hierarchical Evolution with Lineage Incubation)
----------------------------------------------------

Runs short micro-evolutions (“incubations”) on structure-mutated individuals.
Allow topologically changed individuals (e.g. via add/remove neuron) to
stabilize before rejoining the main population.

This operator is **module-local** and does not modify the global evolution loop.

Workflow:
    1. Identify structural mutants
    2. Spawn a subpopulation per seed
    3. Run μ+λ micro-evolution for a few generations
    4. Return best candidate to main offspring pool
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

    Drift rule:
        drift = max(0, Δfitness) / |mean_parent - best_parent|
    """

    from evolib.core.population import Pop
    from evolib.operators.strategy import evolve_mu_plus_lambda

    if not offspring or not pop.heli_enabled:
        return

    # Skip HELI until the main population has been evaluated at least once
    if pop.mean_fitness is None:
        if pop.heli_verbosity >= 1:
            print("[HELI] Skipped: main population not yet evaluated.")
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

    new_candidates: list[Indiv] = []

    if pop.heli_verbosity >= 1:
        print(f"[HELI] Running for {len(seeds)} seeds")

    # 3: Incubate each selected seed
    for seed_idx, seed in enumerate(seeds):
        if pop.heli_verbosity >= 1:
            print(f"[HELI] Seed: {seed_idx+1}")

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

        # Backup original evolutionary parameters and apply temporary HELI damping
        for module_name, module in comp_dict.items():
            heli_backup[module_name] = backup_module_state(module)
            apply_heli_overrides(module, pop.heli_reduce_sigma_factor)

        # Run short local evolution
        for gen in range(pop.heli_generations):
            evolve_mu_plus_lambda(subpop)
            best = subpop.best()

            # Early termination if fitness drift too large
            heli_cfg = getattr(pop.config.evolution, "heli", None)

            maximize = (
                getattr(pop.config.selection, "fitness_maximization", False)
                if pop.config.selection
                else False
            )

            # Reference values from main population
            mu = float(pop.mean_fitness or 0.0)
            main_best = pop.best()
            fit_main_best = float(main_best.fitness or mu if main_best else mu)
            fit_seed = float(best.fitness or 0.0)

            # Only penalize fitness *worsening* relative to main population
            # For maximization: worse if seed < mean
            # For minimization: worse if seed > mean
            delta_fitness = (mu - fit_seed) if maximize else (fit_seed - mu)
            denom = max(1e-12, abs(mu - fit_main_best))
            drift = max(0.0, delta_fitness) / denom

            if pop.heli_verbosity >= 2:
                print(
                    f"[HELI] Seed {seed_idx+1}/{len(seeds)} "
                    f"| Gen {gen+1}/{pop.heli_generations} "
                    f"| FitSeed={fit_seed:.3f} "
                    f"| FitMainBest={fit_main_best:.3f} "
                    f"| FitMainMean={mu:.3f} "
                    f"| Drift={drift:.3f}"
                )

            if heli_cfg and heli_cfg.drift_threshold is not None:
                max_drift = heli_cfg.drift_threshold
                if drift > max_drift:
                    if pop.heli_verbosity > 1:
                        print(
                            f"[HELI] Aborting incubation: "
                            f"drift={drift:.2f} > {max_drift:.2f} "
                            f"(FitSeed={fit_seed:.3f}, mean={mu:.3f})"
                        )
                    break  # abort subpopulation evolution early

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
        print(f"[HELI] Reattached {len(new_candidates)} improved candidates.")
        print("[HELI] End")
