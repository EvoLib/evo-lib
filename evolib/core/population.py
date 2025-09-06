# SPDX-License-Identifier: MIT
"""
Population class coordinating evolutionary operations over generations.

This class acts as the central coordinator for a population-based evolutionary
algorithm. It manages individuals, loads the configuration from YAML, and integrates
mutation, selection, crossover, replacement, and stopping logic.

Supports:
- Initialization from config (parameter modules, mutation strategies, etc.)
- Configurable evolutionary strategies (mu+lambda, steady state, flexible, etc.)
- Early stopping (target fitness, patience, time limit)
- Callbacks: on_start, on_improvement, on_end
- Fitness logging and history tracking

Use Pop.run() for full evolution, or run_one_generation() for manual stepping.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from evolib.config.schema import FullConfig
from evolib.core.individual import Indiv
from evolib.initializers.registry import build_composite_initializer
from evolib.interfaces.enums import (
    DiversityMethod,
    EvolutionStrategy,
    Origin,
    ReplacementStrategy,
)
from evolib.interfaces.types import (
    FitnessFunction,
    OnEndHook,
    OnGenerationHook,
    OnImprovementHook,
    OnStartHook,
    ReplaceFunction,
    SelectionFunction,
)
from evolib.registry.replacement_registry import build_replacement_registry
from evolib.registry.selection_registry import build_selection_registry
from evolib.registry.strategy_registry import strategy_registry
from evolib.utils.config_loader import load_config
from evolib.utils.history_logger import HistoryLogger
from evolib.utils.random import set_random_seed


class Pop:
    """Represents a population for evolutionary optimization, including configuration,
    statistics, and operator integration."""

    def __init__(
        self,
        config_path: str,
        fitness_function: Optional[FitnessFunction] = None,
        initialize: bool = True,
    ):
        """
        Initialize the population from a YAML configuration file.

        Loads all parameter module configurations, strategy settings,
        and operator registries. Optionally initializes individuals immediately.

        Args:
            config_path (str): Path to the YAML configuration file.
            fitness_function (Callable[[Indiv], None], optional): Fitness function.
                If provided, it is stored and used during evolution. Can also be set
                later via set_fitness_function(...).
            initialize (bool): Whether to immediately initialize the population.
                Set to False if you want to control initialization manually
                (e.g., for testing).
        """
        self.fitness_function: Optional[FitnessFunction] = fitness_function

        cfg: FullConfig = load_config(config_path)

        self.config = cfg
        self.para_initializer = build_composite_initializer(cfg)
        self.indivs: List[Any] = []

        # Core parameters
        self.parent_pool_size = cfg.parent_pool_size
        self.offspring_pool_size = cfg.offspring_pool_size
        self.max_generations = cfg.max_generations
        self.max_indiv_age = cfg.max_indiv_age
        self.num_elites = cfg.num_elites

        random_seed = cfg.random_seed
        set_random_seed(random_seed)

        # Strategies (initially None – set externally later)
        self.mutation_strategy = None
        self.selection_strategy = None
        self.selection_fn: Optional[SelectionFunction] = None
        self.pairing_strategy = None
        self.crossover_strategy = None
        self.evolution_strategy = None
        self.replacement_strategy: Optional[ReplacementStrategy] = None
        self._replacement_fn: Optional[ReplaceFunction] = None

        # Evolution
        if cfg.evolution is not None:
            self.evolution_strategy = cfg.evolution.strategy
        else:
            self.evolution_strategy = None

        # Selection
        if cfg.selection is not None:
            self.selection_strategy = cfg.selection.strategy
            self._selection_registry = build_selection_registry(cfg.selection)
            self.selection_fn = self._selection_registry[self.selection_strategy]

        # Replacement
        if cfg.replacement is not None:
            self.replacement_strategy = cfg.replacement.strategy
            self._replacement_registry = build_replacement_registry(cfg.replacement)
            self._replacement_fn = self._replacement_registry[self.replacement_strategy]

        else:
            self.replacement_strategy = None
            self._replacement_registry = {}
            self._replacement_fn = None

        # Statistics
        self.history_logger = HistoryLogger(
            columns=[
                "generation",
                "best_fitness",
                "worst_fitness",
                "mean_fitness",
                "median_fitness",
                "std_fitness",
                "iqr_fitness",
                "diversity",
            ]
        )
        self.generation_num = 0
        self.best_fitness = 0.0
        self.worst_fitness = 0.0
        self.mean_fitness = 0.0
        self.median_fitness = 0.0
        self.std_fitness = 0.0
        self.iqr_fitness = 0.0
        self.diversity = 0.0
        self.diversity_ema = 0.0

        # Autoinitialize Population
        if initialize is True:
            self.initialize_population()

    @property
    def mu(self) -> int:
        return self.parent_pool_size

    @property
    def lambda_(self) -> int:
        return self.offspring_pool_size

    @property
    def sample_indiv(self) -> Indiv:
        """
        Returns a representative individual from the current population.

        Useful for inspecting parameter module dimensions, shapes, or bounds
        """

        if not self.indivs:
            raise RuntimeError("No individuals initialized.")
        return self.indivs[0]

    @property
    def history_df(self) -> pd.DataFrame:
        """
        Return the evolution history as a pandas DataFrame.

        This is a convenience wrapper around `history_logger.to_dataframe()` so users
        don't need to access internal logger details.
        """
        return self.history_logger.to_dataframe()

    @property
    def history_dicts(self) -> List[Dict[str, Any]]:
        """Return history as list of dicts (pandas-free)."""
        return self.history_logger.to_dicts()

    def initialize_population(
        self, initializer: Callable[["Pop"], Any] | None = None
    ) -> None:
        """
        Initializes the population using the provided para initializer function.

        Args:
            initializer (Callable[[Pop], ParaBase], optional):
                Function to generate Para instances for each individual.
        """
        self.clear_indivs()

        init_fn = initializer if initializer is not None else self.para_initializer

        for _ in range(self.mu):
            para = init_fn(self)
            self.add_indiv(Indiv(para=para))

    def set_fitness_function(self, func: FitnessFunction) -> None:
        """
        Sets the fitness function to be used for evaluating individuals.

        Args:
            func (Callable[[Indiv], None]): A function that modifies an individual
            in-place by assigning `indiv.fitness = ...`.

        Raises:
            TypeError: If the argument is not callable.
        """
        if not callable(func):
            raise TypeError("Fitness function must be callable.")
        self.fitness_function = func

    def set_functions(self, fitness_function: FitnessFunction) -> None:
        """
        [DEPRECATED] Use set_fitness_function() or constructor argument instead.

        Registers the fitness function used during evolution.

        Args:
            fitness_function (Callable): Function to assign fitness to an individual.
        """
        self.fitness_function = fitness_function

    def evaluate_fitness(self) -> None:
        """Evaluate the fitness function for all individuals in the population."""
        if self.fitness_function is None:
            raise ValueError("No fitness function has been set.")
        for indiv in self.indivs:
            result = self.fitness_function(indiv)
            if result is not None:
                indiv.fitness = float(result)
            indiv.is_evaluated = True

    def evaluate_indivs(self, indivs: list[Indiv]) -> None:
        """Evaluate fitness for a custom list of individuals."""
        if self.fitness_function is None:
            raise ValueError("No fitness function has been set.")
        for indiv in indivs:
            result = self.fitness_function(indiv)
            if result is not None:
                indiv.fitness = float(result)
            indiv.is_evaluated = True

    def get_elites(self) -> list[Indiv]:
        """Return a list of elite individuals and set their is_elite flag."""
        # Reset is_elite for all
        for indiv in self.indivs:
            indiv.is_elite = False

        self.sort_by_fitness()
        elites = self.indivs[: self.num_elites]
        for indiv in elites:
            indiv.is_elite = True
        return elites

    def print_status(self, verbosity: int = 1) -> None:
        """
        Prints information about the population based on the verbosity level.

        Args:
            verbosity (int, optional): Level of detail for the output.
                - 0: No output
                - 1: Basic information (generation, fitness, diversity)
                - 2: Additional parameters (e.g., mutation rate, population fitness)
                - 3: Detailed information (e.g., number of individuals, elites)
                - 10: Full details including a call to info_indivs()
            Default: 1

        Raises:
            TypeError: If verbosity is not an integer.
            AttributeError: If required population data is incomplete.
        """
        if not isinstance(verbosity, int):
            raise TypeError("verbosity must be an integer")

        if verbosity <= 0:
            return

        if not hasattr(self, "indivs") or not self.indivs:
            raise AttributeError(
                "Population contains no individuals (self.indivs is missing or empty)"
            )

        # Start output
        if verbosity >= 1:
            line = (
                f"Population: Gen: {self.generation_num:3d} "
                f"Fit: {self.best_fitness:.8f}"
            )
            print(line)

        if verbosity >= 2:
            print(f"Best Indiv age: {self.indivs[0].age}")
            print(f"Max Generation: {self.max_generations}")
            print(f"Number of Indivs: {len(self.indivs)}")
            print(f"Number of Elites: {self.num_elites}")
            print(f"Population fitness: {self.mean_fitness:.3f}")
            print(f"Worst Indiv: {self.worst_fitness:.3f}")

        if verbosity == 10:
            self.print_indivs()

    def print_history(
        self, last_n: int | None = None, columns: list[str] | None = None
    ) -> None:
        """
        Prints the evolution history in a simple tabular format using pandas.

        Args:
            last_n: If set, only print the last N generations.
            columns: If set, restrict output to specific columns.
        """
        df = self.history_df

        if columns is None:
            columns = [
                "generation",
                "best_fitness",
                "worst_fitness",
                "mean_fitness",
                "std_fitness",
                "iqr_fitness",
            ]

        df = df[columns]

        if last_n is not None:
            df = df.tail(last_n)

        print("\nEvolution History:")
        print(df.to_string(index=False))

    def print_indivs(self) -> None:
        """Print the status of all individuals in the population."""
        for indiv in self.indivs:
            indiv.print_status()

    def create_indiv(self) -> Indiv:
        """Create a new individual using default settings."""
        para = self.para_initializer(self)
        return Indiv(para=para)

    def add_indiv(self, new_indiv: Indiv | None = None) -> None:
        """
        Add a new individual to the population.

        Args:
            new_indiv (Indiv): The individual to be added.
        """

        if new_indiv is None:
            new_indiv = Indiv()

        self.indivs.append(new_indiv)

    def remove_indiv(self, indiv: Indiv) -> None:
        """
        Remove an individual from the population.

        Args:
            indiv (Indiv): The individual to be removed.
        """

        if not isinstance(indiv, Indiv):
            raise TypeError("Only an object of type 'Indiv' can be removed.")
        if indiv not in self.indivs:
            raise ValueError("Individual not found in the population.")

        self.indivs.remove(indiv)

    def get_fitness_array(self, include_none: bool = False) -> np.ndarray:
        """
        Return a NumPy array of all fitness values in the population.

        Returns:
            np.ndarray: Array of fitness values (ignores None).
        """

        values = [i.fitness for i in self.indivs]
        return np.array(
            values if include_none else [v for v in values if v is not None]
        )

    def sort_by_fitness(self, reverse: bool = False) -> None:
        """
        Sorts the individuals in the population by their fitness (ascending by default).

        Args:
            reverse (bool): If True, sort in descending order.
        """

        self.indivs.sort(key=lambda indivs: indivs.fitness, reverse=reverse)

    def best(self, sort: bool = False) -> Indiv:
        """
        Return the best individual (lowest fitness).

        Args:
            sort (bool): If True, sort the population before returning the best.
                         If False, return first individual as-is.
                         Default: False.
        """

        if not self.indivs:
            raise ValueError("Population is empty; cannot return best individual.")

        if sort:
            self.sort_by_fitness()

        return self.indivs[0]

    def remove_old_indivs(self) -> int:
        """
        Removes individuals whose age exceeds the maximum allowed age, excluding elite
        individuals.

        Returns:
            int: Number of individuals removed.
        """

        if self.max_indiv_age <= 0:
            return 0

        elite_cutoff = self.num_elites if self.num_elites > 0 else 0

        survivors = self.indivs[:elite_cutoff] + [
            indiv
            for indiv in self.indivs[elite_cutoff:]
            if indiv.age < self.max_indiv_age
        ]

        removed_count = len(self.indivs) - len(survivors)
        self.indivs = survivors

        return removed_count

    def age_indivs(self) -> None:
        """
        Increment the age of all individuals in the population by 1 and set their
        'origin' to indicate they are now considered parents in the evolutionary
        process.

        Raises:
            ValueError: If the population is empty.
        """

        if not self.indivs:
            raise ValueError("Population contains no individuals (indivs is empty)")

        for indiv in self.indivs:
            indiv.age += 1
            indiv.origin = Origin.PARENT

    def update_statistics(self) -> None:
        """
        Update all fitness-related statistics of the population.

        Raises:
            ValueError: If no individuals have a valid fitness value.
        """

        self.generation_num += 1

        fitnesses = self.get_fitness_array()

        if fitnesses.size == 0:
            raise ValueError("No valid fitness values to compute statistics.")

        self.best_fitness = min(fitnesses)
        self.worst_fitness = max(fitnesses)
        self.mean_fitness = np.mean(fitnesses)
        self.std_fitness = np.std(fitnesses)
        self.median_fitness = np.median(fitnesses)
        self.iqr_fitness = np.percentile(fitnesses, 75) - np.percentile(fitnesses, 25)
        self.diversity = self.fitness_diversity(method=DiversityMethod.IQR)

        if self.diversity_ema is None:
            self.diversity_ema = self.diversity
        else:
            alpha = 0.1
            self.diversity_ema = (1 - alpha) * self.diversity_ema + (
                alpha * self.diversity
            )

        # Logging
        row = {
            "generation": self.generation_num,
            "best_fitness": self.best_fitness,
            "worst_fitness": self.worst_fitness,
            "mean_fitness": self.mean_fitness,
            "median_fitness": self.median_fitness,
            "std_fitness": self.std_fitness,
            "iqr_fitness": self.iqr_fitness,
            "diversity": self.diversity,
        }

        if hasattr(self.best(), "para") and hasattr(self.best().para, "get_history"):
            row.update(self.best().para.get_history())

        if hasattr(self.best().para, "get_status"):
            row["status_str"] = self.best().para.get_status()

        self.history_logger.log(row)

    def fitness_diversity(self, method: DiversityMethod = DiversityMethod.IQR) -> float:
        """
        Computes population diversity based on fitness values.

        Args:
            method (str): One of ['iqr', 'std', 'var', 'range', 'normalized_std']

        Returns:
            float: Diversity score.
        """

        fitnesses = self.get_fitness_array()
        return compute_fitness_diversity(fitnesses.tolist(), method=method)

    def clear_indivs(self) -> None:
        """Remove all individuals from the population."""
        self.indivs.clear()

    def reset(self) -> None:
        """
        Reset the population to an empty state and reset all statistics.

        Keeps configuration and mutation/crossover strategy, but removes all individuals
        and clears the history logger.
        """
        self.indivs.clear()
        self.generation_num = 0

        # Reset statistics
        self.best_fitness = 0.0
        self.worst_fitness = 0.0
        self.mean_fitness = 0.0
        self.median_fitness = 0.0
        self.std_fitness = 0.0
        self.iqr_fitness = 0.0
        self.diversity = 0.0
        self.diversity_ema = 0.0

        self.history_logger.reset()

    def update_parameters(self) -> None:
        """
        Update all strategy-dependent parameters for the current generation.

        Calls both `update_mutation_parameters()` and `update_crossover_parameters()`.

        Raises:
            ValueError or AttributeError if the population or its individuals
            are invalid.
        """
        self.update_mutation_parameters()
        self.update_crossover_parameters()

    def update_mutation_parameters(self) -> None:
        """
        Triggers per-generation mutation parameter updates for all individuals in the
        population via their `para` objects.

        This ensures that all individuals – including parents – are updated consistently
        based on current generation number.

        Uses a polymorphic call to `para.update_mutation_parameters()`, preserving
        encapsulation.
        """
        for indiv in self.indivs:
            indiv.para.update_mutation_parameters(
                self.generation_num, self.max_generations, self.diversity_ema
            )

    def update_crossover_parameters(self) -> None:
        """
        Triggers per-generation update of crossover parameters for all individuals.

        Applies strategy-dependent crossover control (e.g. exponential decay or
        adaptive global), using generation number and population diversity.

        Raises:
            ValueError: If population is uninitialized or empty.
            AttributeError: If an individual lacks a valid 'para' object with method
                            'update_crossover_parameters'.
        """
        if not self.indivs:
            raise ValueError(
                "Population is empty – cannot update crossover parameters."
            )

        for indiv in self.indivs:
            if not hasattr(indiv, "para") or not hasattr(
                indiv.para, "update_crossover_parameters"
            ):
                raise AttributeError(
                    "Individual is missing a valid 'para' object "
                    "with 'update_crossover_parameters' method."
                )

            indiv.para.update_crossover_parameters(
                self.generation_num,
                self.max_generations,
                self.diversity_ema,
            )

    def run_one_generation(
        self, strategy: EvolutionStrategy | None = None, sort: bool = False
    ) -> None:
        """
        Executes a single evolutionary generation using the selected strategy.

        Args:
            strategy (EvolutionStrategy | None): Optional override for the evolution
            strategy.
            If None, uses the strategy defined during initialization.

        Raises:
            ValueError: If no strategy is defined or the strategy is unknown.
        """
        if strategy is None:
            strategy = self.evolution_strategy

        if strategy is None:
            raise ValueError("Evolution Strategy must be defined")

        fn = strategy_registry.get(strategy)
        if fn is None:
            raise ValueError(f"Unknown strategy: {strategy}")

        fn(self)

        if sort:
            self.sort_by_fitness()

    def run(
        self,
        *,
        strategy: EvolutionStrategy | None = None,
        max_generations: Optional[int] = None,
        target_fitness: Optional[float] = None,
        minimize: Optional[bool] = None,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
        time_limit_s: Optional[float] = None,
        verbosity: int = 1,
        on_start: OnStartHook = None,
        on_generation: OnGenerationHook = None,
        on_improvement: OnImprovementHook = None,
        on_end: OnEndHook = None,
    ) -> int:
        """
        Run the evolutionary process until a stopping criterion is met.

        Args:
            strategy: Optional override of the evolution strategy.
            max_generations: Maximum number of generations to run
                             (fallback: self.max_generations).
            target_fitness: Desired fitness threshold to stop evolution early.
            minimize: If True, lower fitness is better; else maximize. Defaults to True.
            patience: Stop if no improvement after this many generations.
            min_delta: Minimum improvement to reset patience counter.
            time_limit_s: Stop evolution after this many seconds (wall clock).
            verbosity: 0 = silent, 1 = status messages (default).
            on_start: Optional callback(pop)
            on_generation: Optional callback(pop)
            on_improvement: Optional callback(pop)
            on_end: Optional callback(pop)

        Returns:
            int: Number of generations completed.
        """

        if self.fitness_function is None:
            raise ValueError(
                "Population.run() requires a fitness_function. "
                "Set it via constructor or pop.set_fitness_function(...)."
            )

        # Strategy
        strategy = strategy or self.evolution_strategy
        if strategy is None:
            raise ValueError("Evolution Strategy must be defined")

        # Load stopping criteria from config if not provided
        if self.config.stopping:
            cfg = self.config.stopping
            if target_fitness is None:
                target_fitness = cfg.target_fitness
            if minimize is None:
                minimize = cfg.minimize
            if patience is None:
                patience = cfg.patience
            if min_delta == 0.0 and cfg.min_delta != 0.0:
                min_delta = cfg.min_delta
            if time_limit_s is None:
                time_limit_s = cfg.time_limit_s

        # Fallback to default: minimize = True
        if minimize is None:
            minimize = True

        # Determine maximum number of generations
        gen_cap = max_generations or self.max_generations
        if gen_cap <= 0:
            return 0

        if verbosity >= 1:
            print(
                f"start: strategy={strategy}, parents(mu)={self.mu}, "
                f"offspring(lambda)={self.lambda_}, max_gen={gen_cap}"
            )

        start_time = time.time()
        best_fitness = math.inf if minimize else -math.inf
        no_improve = 0

        # ON_START
        if on_start:
            on_start(self)

        for _ in range(gen_cap):
            self.run_one_generation(strategy=strategy)
            self.print_status(verbosity=1 if verbosity >= 1 else 0)

            # ON GENERATION
            if on_generation:
                on_generation(self)

            current_fitness = self.best().fitness

            assert current_fitness is not None, (
                "Strategy must evaluate " "fitness each generation"
            )

            if minimize:
                has_improved = (best_fitness - current_fitness) > min_delta
            else:
                has_improved = (current_fitness - best_fitness) > min_delta

            # ON_IMPROVEMENT
            if has_improved:
                best_fitness = current_fitness
                no_improve = 0
                if on_improvement:
                    on_improvement(self)
            else:
                no_improve += 1

            # Target fitness reached
            if target_fitness is not None:

                fitness_reached = False
                if minimize:
                    if current_fitness <= target_fitness:
                        fitness_reached = True
                else:
                    if current_fitness >= target_fitness:
                        fitness_reached = True

                if fitness_reached:
                    if verbosity >= 1:
                        print(
                            f"stop: target_fitness={target_fitness} reached "
                            f"at gen {self.generation_num} (best={best_fitness:.6g})"
                        )
                    break

            # Patience exhausted
            if patience is not None and no_improve >= patience:
                if verbosity >= 1:
                    print(
                        f"stop: no improvement for {patience} generations "
                        f"(best={best_fitness:.6g})"
                    )
                break

            # Time limit exceeded
            if time_limit_s is not None and (time.time() - start_time) > time_limit_s:
                if verbosity >= 1:
                    print(
                        f"stop: time limit of {time_limit_s}s exceeded "
                        f"at gen {self.generation_num} (best={best_fitness:.6g})"
                    )
                break

        # ON_END
        if on_end:
            on_end(self)

        return self.generation_num

    def select_parents(self, num_parents: int) -> list[Indiv]:
        """
        Selects parents using the configured selection strategy.

        Args:
            num_parents (int): Number of parents to select.

        Returns:
            list[Indiv]: Selected parents (deep copies).
        """

        if self.selection_fn is None:
            raise ValueError("Selection Strategy must be defined")

        return self.selection_fn(self, num_parents)

    def ensure_evaluated(self) -> None:
        """Evaluate individuals that have no valid fitness yet."""
        no_fitness = [ind for ind in self.indivs if not _is_valid_fitness(ind.fitness)]
        if no_fitness:
            if self.fitness_function is None:
                raise ValueError(
                    "No fitness_function set, but evaluation is required. "
                    "Provide it via Population(..., fitness_function=...) "
                    "or pop.set_fitness_function(...)."
                )
            self.evaluate_indivs(no_fitness)


##############################################################################


def _is_valid_fitness(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def compute_fitness_diversity(
    fitnesses: list[float],
    method: DiversityMethod = DiversityMethod.IQR,
    epsilon: float = 1e-8,
) -> float:
    """
    Computes a diversity metric for a list of fitness values.

    Args:
        fitnesses (list[float]): Fitness values of individuals.
        method (DiversityMethod): Diversity metric to use.
        epsilon (float): Small constant to prevent division by zero.

    Returns:
        float: Computed diversity score.
    """
    if not fitnesses:
        return 0.0

    values = np.array(fitnesses)
    median = np.median(values)

    if method == DiversityMethod.IQR:
        return float(np.percentile(fitnesses, 75) - np.percentile(fitnesses, 25))

    if method == DiversityMethod.RELATIVE_IQR:
        q75, q25 = np.percentile(values, [75, 25])
        median = np.median(values)
        return (q75 - q25) / (median + epsilon)

    if method == DiversityMethod.STD:
        return np.std(values)

    if method == DiversityMethod.VAR:
        return np.var(values)

    if method == DiversityMethod.RANGE:
        return (np.max(values) - np.min(values)) / (median + epsilon)

    if method == DiversityMethod.NORMALIZED_STD:
        return np.std(values) / (median + epsilon)

    raise ValueError(f"Unsupported diversity method: '{method}'")


##############################################################################
# EOF
