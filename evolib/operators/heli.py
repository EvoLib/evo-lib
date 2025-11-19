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

from evolib.utils.fitness import sort_by_fitness
from evolib.utils.heli_utils import (
    apply_heli_overrides,
    backup_module_state,
    restore_module_state,
)


def evaluate_heli_drift(
    pop: "Pop", best: "Indiv", seed_idx: int, gen: int, n_seeds: int
) -> float | None:
    """
    Compute and evaluate HELI drift for one subpopulation generation.

    Drift measures how far a seed's fitness lies outside the main population's
    survival window (using the worst individual as reference).

    Positive drift: seed worse than population (hopeless)
    Negative drift: seed better than population (already viable)

    Returns
    -------
    drift : float | None
        Drift value if evaluated; None if HELI config not found.
    """

    heli_cfg = getattr(pop.config.evolution, "heli", None)
    if heli_cfg is None:
        if pop.heli_verbosity >= 2:
            print("[HELI] Skipped drift check: no HELI config found.")
        return None

    maximize = (
        getattr(pop.config.selection, "fitness_maximization", False)
        if pop.config.selection
        else False
    )

    # Reference values from main population
    fit_seed = float(best.fitness or 0.0)
    fit_main_worst = float(pop.worst_fitness or 0.0)
    fit_main_best = float(pop.best_fitness or 0.0)

    # Compute drift
    delta_fitness = (fit_main_worst - fit_seed) * (1 if maximize else -1)
    scale = abs(fit_main_worst) + max(
        1e-12, 0.001 * abs(fit_main_best - fit_main_worst)
    )
    drift = delta_fitness / scale

    # Verbose diagnostics
    if pop.heli_verbosity >= 2:
        print(
            f"[HELI] Seed {seed_idx+1}/{n_seeds} | Gen {gen+1}/{pop.heli_generations} "
            f"| FitSeed={fit_seed:.3f} | FitMainBest={fit_main_best:.3f} "
            f"| FitMainWorst={fit_main_worst:.3f} | Drift={drift:.3f}"
        )

    stop_above = getattr(heli_cfg, "drift_stop_above", None)
    stop_below = getattr(heli_cfg, "drift_stop_below", None)

    if stop_above is not None and drift >= stop_above:
        if pop.heli_verbosity > 1:
            print(
                f"[HELI] Aborting incubation: drift={drift:.2f} > {stop_above:.2f} "
                f"(FitSeed={fit_seed:.3f}, WorstMain={fit_main_worst:.3f})"
            )
        return float("inf")  # signal --> hopeless

    if stop_below is not None and drift <= stop_below:
        if pop.heli_verbosity > 1:
            print(
                f"[HELI] Early finish: drift={drift:.2f} < {stop_below:.2f} "
                f"(FitSeed={fit_seed:.3f}, WorstMain={fit_main_worst:.3f})"
            )
        return float("-inf")  # signal --> already viable

    return drift


def run_heli(pop: "Pop", offspring: List["Indiv"]) -> int:
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

    fitness_evaluations = 0

    heli_cfg = getattr(pop.config.evolution, "heli", None)
    if heli_cfg is None:
        if pop.heli_verbosity >= 1:
            print("[HELI] Warning: run_heli() called without HELI config. Skipping.")
        return fitness_evaluations

    if not offspring or not pop.heli_enabled:
        return fitness_evaluations

    # Skip HELI in generation 0 (no evaluation baseline yet)
    if pop.generation_num == 0:
        if pop.heli_verbosity >= 1:
            print("[HELI] Skipped: main population not yet evaluated.")
        return fitness_evaluations

    # 1: Select structure-mutated offspring
    struct_mutants = [indiv for indiv in offspring if indiv.para.has_structural_change]
    if not struct_mutants:
        if pop.heli_verbosity >= 2:
            print(f"[HELI] Gen: {pop.generation_num} - No struct_mutants")
        return fitness_evaluations

    if pop.heli_verbosity >= 2:
        print(f"[HELI] Start: Number of structural mutants: {len(struct_mutants)}")

    # 2: Limit number of incubated seeds
    max_seeds = max(1, round(len(offspring) * pop.heli_max_fraction))
    if len(struct_mutants) > max_seeds:
        pop.evaluate_indivs(struct_mutants)
        struct_mutants = sort_by_fitness(struct_mutants)
    seeds = struct_mutants[:max_seeds]

    if len(seeds) < 1:
        if pop.heli_verbosity >= 2:
            print(f"[HELI] Gen: {pop.generation_num} - No Seed")
        return fitness_evaluations

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
        subpop.lineage_logger = None

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

            # Drift evaluation
            drift = evaluate_heli_drift(pop, best, seed_idx, gen, len(seeds))
            if drift == float("inf") or drift == float("-inf"):
                break  # abort incubation early

        fitness_evaluations += gen * subpop.offspring_pool_size

        # Restore evo_params
        para_dict = vars(best.para)
        comp_dict = para_dict["components"]

        for module_name, module in comp_dict.items():
            restore_module_state(module, heli_backup.get(module_name, {}))

        # 4: Reintegration
        new_candidates.append(best)

    # Mark candidates as reintegrated and set is_structural_mutant (lost during copy)
    for candidate in new_candidates:
        candidate.heli_reintegrated = True
        candidate.is_structural_mutant = True

    # 5: Reattach improved candidates to the main offspring
    offspring.extend(new_candidates)

    if pop.heli_verbosity >= 2:
        print("[HELI] End.")

    return fitness_evaluations
