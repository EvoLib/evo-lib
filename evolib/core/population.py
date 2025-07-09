# SPDX-License-Identifier: MIT
"""
Coordinates application of evolutionary operations over generations.

While the core logic for operations like fitness, mutation, and crossover is externally
defined and dynamically loaded, this class coordinates their application over
generations.
"""

from typing import Any, Callable, List, Optional

import numpy as np

from evolib.core.individual import Indiv
from evolib.initializers.registry import get_initializer
from evolib.interfaces.enums import (
    CrossoverStrategy,
    DiversityMethod,
    EvolutionStrategy,
    MutationStrategy,
    Origin,
    ReplacementStrategy,
    SelectionStrategy,
)
from evolib.interfaces.types import FitnessFunction, ReplaceFunction, SelectionFunction
from evolib.registry.replacement_registry import build_replacement_registry
from evolib.registry.selection_registry import selection_registry
from evolib.registry.strategy_registry import strategy_registry
from evolib.utils.config_loader import get_enum, load_config
from evolib.utils.config_validator import validate_full_config
from evolib.utils.history_logger import HistoryLogger


class Pop:
    """Represents a population for evolutionary optimization, including configuration,
    statistics, and operator integration."""

    def __init__(self, config_path: str, initialize: bool = True):
        """
        Initialize a population from a YAML config file.

        Args:
        config_path (str): Path to the population configuration file.
        """

        cfg = load_config(config_path)
        validate_full_config(cfg)

        self.full_config = cfg
        self.representation_cfg = cfg["representation"]

        initializer_name = self.representation_cfg["initializer"]
        initializer_factory = get_initializer(initializer_name)
        self.para_initializer = initializer_factory(cfg)

        self.indivs: List[Any] = []

        # Core parameters
        self.parent_pool_size = cfg["parent_pool_size"]
        self.offspring_pool_size = cfg["offspring_pool_size"]
        self.max_generations = cfg["max_generations"]
        self.max_indiv_age = cfg["max_indiv_age"]
        self.num_elites = cfg["num_elites"]

        # Strategies (initially None – set externally later)
        self.mutation_strategy = None
        self.selection_strategy = None
        self.selection_fn: Optional[SelectionFunction] = None
        self.pairing_strategy = None
        self.crossover_strategy = None
        self.evolution_strategy = None
        self.replacement_strategy: Optional[ReplacementStrategy] = None
        self._replacement_fn: Optional[ReplaceFunction] = None

        # User-defined functions
        self.fitness_function: FitnessFunction | None = None

        # Evolution
        evolution_cfg = cfg.get("evolution", None)
        if evolution_cfg is not None:
            self.evolution_strategy = get_enum(
                EvolutionStrategy,
                evolution_cfg.get("strategy", "mu_plus_lambda"),
                "evolution strategy",
            )
        else:
            self.evolution_strategy = None

        # Selection
        selection_cfg = cfg.get("selection", None)
        if selection_cfg is not None:
            self.selection_strategy = get_enum(
                SelectionStrategy, selection_cfg["strategy"], "selection strategy"
            )
            self.selection_fn = selection_registry[self.selection_strategy]
        else:
            self.selection_strategy = None
            self.selection_fn = None

        # Replacement
        replacement_cfg = cfg.get("replacement", None)
        if replacement_cfg is not None:
            self.replacement_strategy = get_enum(
                ReplacementStrategy, replacement_cfg["strategy"], "replacement strategy"
            )

            self._replacement_registry = build_replacement_registry(cfg["replacement"])
            self._replacement_fn = self._replacement_registry[self.replacement_strategy]

        else:
            self.replacement_strategy = None
            self._replacement_registry = {}
            self._replacement_fn = None

        # Mutation
        self.mutation_strategy = get_enum(
            MutationStrategy, cfg["mutation"]["strategy"], "mutation strategy"
        )

        # Crossover
        crossover_cfg = cfg.get("crossover", None)
        if crossover_cfg is None:
            self.crossover_strategy = CrossoverStrategy.NONE
            self.crossover_probability = None

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

    def set_functions(self, fitness_function: FitnessFunction) -> None:
        """
        Registers core evolutionary functions used during evolution.

        Args:
            fitness_function (Callable): Function to assign fitness to an individual.
            mutation_function (Callable): Function to mutate parameters
            of an individual.
            tau_update_function (Callable): Optional function to update
            `tau` based on `para`.
        """
        self.fitness_function = fitness_function

    def evaluate_fitness(self) -> None:
        """Evaluate the fitness function for all individuals in the population."""
        if self.fitness_function is None:
            raise ValueError("No fitness function has been set.")
        for indiv in self.indivs:
            self.fitness_function(indiv)

    def evaluate_indivs(self, indivs: list[Indiv]) -> None:
        """Evaluate fitness for a custom list of individuals."""
        if self.fitness_function is None:
            raise ValueError("No fitness function has been set.")
        for indiv in indivs:
            self.fitness_function(indiv)

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
            include_none (bool): If True, include None values in the array
                                (as object dtype).
                                If False, exclude individuals with undefined fitness.
                                 Default is False.
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
            self.diversity_ema = (1 - alpha) * (
                self.diversity_ema + alpha * self.diversity
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


##############################################################################


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
