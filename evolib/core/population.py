# SPDX-License-Identifier: MIT
"""
Coordinates application of evolutionary operations over generations.

While the core logic for operations like fitness, mutation, and crossover is externally
defined and dynamically loaded, this class coordinates their application over
generations.
"""

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from evolib.core.individual import Indiv
from evolib.interfaces.enums import (
    CrossoverStrategy,
    DiversityMethod,
    MutationStrategy,
    Origin,
    Representation,
)
from evolib.interfaces.types import (
    FitnessFunction,
    MutationFunction,
    TauUpdateFunction,
)
from evolib.utils.config_loader import load_config
from evolib.utils.config_validator import validate_full_config
from evolib.utils.default_tau_update import default_update_tau
from evolib.utils.history_logger import HistoryLogger


class Pop:
    """Represents a population for evolutionary optimization, including configuration,
    statistics, and operator integration."""

    def __init__(
        self, config_path: str, mutation_function: MutationFunction | None = None
    ):
        """
        Initialize a population from a YAML config file.

        Args:
        config_path (str): Path to the population configuration file.
        mutation_function (MutationFunction, optional): Custom mutation function
        override.
        """

        cfg = load_config(config_path)
        validate_full_config(cfg)

        self.indivs: List[Any] = []

        # Core parameters
        self.parent_pool_size = cfg["parent_pool_size"]
        self.offspring_pool_size = cfg["offspring_pool_size"]
        self.max_generations = cfg["max_generations"]
        self.max_indiv_age = cfg["max_indiv_age"]
        self.num_elites = cfg["num_elites"]
        self.representation = Representation(cfg["representation"])

        # Strategies (initially None – set externally later)
        self.mutation_strategy = None
        self.selection_strategy = None
        self.pairing_strategy = None
        self.crossover_strategy = None

        # User-defined functions
        self.fitness_function: FitnessFunction | None = None
        self.mutation_function: MutationFunction | None = mutation_function
        self.tau_update_function: TauUpdateFunction = default_update_tau

        # Mutation
        self.mutation_strategy = MutationStrategy(cfg["mutation"]["strategy"])

        if self.mutation_strategy == MutationStrategy.CONSTANT:
            self.mutation_probability = cfg["mutation"]["probability"]
            self.mutation_strength = cfg["mutation"]["strength"]

        if self.mutation_strategy == MutationStrategy.EXPONENTIAL_DECAY:
            self.min_mutation_probability = cfg["mutation"]["min_probability"]
            self.max_mutation_probability = cfg["mutation"]["max_probability"]

            self.min_mutation_strength = cfg["mutation"]["min_strength"]
            self.max_mutation_strength = cfg["mutation"]["max_strength"]

        if self.mutation_strategy == MutationStrategy.ADAPTIVE_GLOBAL:
            self.mutation_probability = cfg["mutation"]["init_probability"]
            self.min_mutation_probability = cfg["mutation"]["min_probability"]
            self.max_mutation_probability = cfg["mutation"]["max_probability"]

            self.mutation_strength = cfg["mutation"]["init_strength"]
            self.min_mutation_strength = cfg["mutation"]["min_strength"]
            self.max_mutation_strength = cfg["mutation"]["max_strength"]

            self.min_diversity_threshold = cfg["mutation"]["min_diversity_threshold"]
            self.max_diversity_threshold = cfg["mutation"]["max_diversity_threshold"]

            self.mutation_inc_factor = cfg["mutation"]["increase_factor"]
            self.mutation_dec_factor = cfg["mutation"]["decrease_factor"]

        if self.mutation_strategy == MutationStrategy.ADAPTIVE_INDIVIDUAL:
            self.mutation_probability = None
            self.mutation_strength = None

            self.min_mutation_probability = cfg["mutation"]["min_probability"]
            self.max_mutation_probability = cfg["mutation"]["max_probability"]

            self.min_mutation_strength = cfg["mutation"]["min_strength"]
            self.max_mutation_strength = cfg["mutation"]["max_strength"]

        if self.mutation_strategy == MutationStrategy.ADAPTIVE_PER_PARAMETER:
            self.mutation_probability = None
            self.mutation_strength = None

            self.min_mutation_probability = None
            self.max_mutation_probability = None

            self.min_mutation_strength = cfg["mutation"]["min_strength"]
            self.max_mutation_strength = cfg["mutation"]["max_strength"]

        # Crossover
        crossover_cfg = cfg.get("crossover", None)
        if crossover_cfg is None:
            self.crossover_strategy = CrossoverStrategy.NONE
            self.crossover_probability = None
        else:
            self.crossover_strategy = CrossoverStrategy(cfg["crossover"]["strategy"])
            if self.crossover_strategy == CrossoverStrategy.CONSTANT:
                self.crossover_probability = cfg["crossover"]["probability"]

            if self.crossover_strategy == CrossoverStrategy.EXPONENTIAL_DECAY:
                self.min_crossover_probability = cfg["crossover"]["min_probability"]
                self.max_crossover_probability = cfg["crossover"]["max_probability"]

            if self.crossover_strategy == CrossoverStrategy.ADAPTIVE_GLOBAL:
                self.crossover_probability = cfg["crossover"]["init_probability"]
                self.min_crossover_probability = cfg["crossover"]["min_probability"]
                self.max_crossover_probability = cfg["crossover"]["max_probability"]

                self.min_diversity_threshold = cfg["crossover"][
                    "min_diversity_threshold"
                ]
                self.max_diversity_threshold = cfg["crossover"][
                    "max_diversity_threshold"
                ]

                self.crossover_inc_factor = cfg["crossover"]["increase_factor"]
                self.crossover_dec_factor = cfg["crossover"]["decrease_factor"]

        # Representation
        if self.representation == Representation.NEURAL:
            self.mutation_strength_bias = cfg["mutation"]["strength_bias"]
            self.mutation_rate_bias = cfg["mutation"]["rate_bias"]

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
                "mutation_probability",
                "mutation_strength",
                "mutation_probability_mean",
                "mutation_strength_mean",
                "crossover_probability",
                "crossover_probability_mean",
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

    @property
    def mu(self) -> int:
        return self.parent_pool_size

    @property
    def lambda_(self) -> int:
        return self.offspring_pool_size

    def initialize_random_population(
        self, initializer: Callable[["Pop"], Any] | None = None
    ) -> None:
        """
        Create a new random population using the given parameter initializer.

        Args:
            initializer (Callable[[Pop], Any], optional): Function that generates
                a valid `para` object for each individual.
                If None, individuals are created with `para=None`.
        """
        self.clear_indivs()
        for _ in range(self.mu):
            para = initializer(self) if initializer else None
            self.add_indiv(Indiv(para=para))

    def set_functions(
        self,
        fitness_function: FitnessFunction,
        mutation_function: MutationFunction,
        tau_update_function: TauUpdateFunction = default_update_tau,
    ) -> None:
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
        self.mutation_function = mutation_function
        self.tau_update_function = tau_update_function

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
            if self.mutation_strategy == MutationStrategy.ADAPTIVE_GLOBAL:
                print(f"min_diversity_threshold: {self.min_diversity_threshold}")
                print(f"max_diversity_threshold: {self.max_diversity_threshold}")

        if verbosity >= 3:
            if self.mutation_probability_effective is not None:
                print(
                    f"mutation_probability: {self.mutation_probability_effective:.3f}"
                )
            else:
                print("mutation_probability: n/a")

            if self.mutation_strength_effective is not None:
                print(f"mutation_strength: {self.mutation_strength_effective:.3f}")
            else:
                print("mutation_strength: n/a")

            if self.mutation_strength_bias_effective is not None:
                print(
                    f"mutation_strength_bias: "
                    f"{self.mutation_strength_bias_effective:.3f}"
                )
            else:
                print("mutation_strength_bias: n/a")

            if self.crossover_rate_effective is not None:
                print(f"crossover_probability: {self.crossover_rate_effective:.3f}")
            else:
                print("crossover_probability: n/a")

        if verbosity == 10:
            self.print_indivs()

    def print_indivs(self) -> None:
        """Print the status of all individuals in the population."""
        for indiv in self.indivs:
            indiv.print_status()

    def create_indiv(self) -> Indiv:
        """Create a new individual using default settings."""
        return Indiv()

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

        if self.mutation_strategy == MutationStrategy.ADAPTIVE_INDIVIDUAL:
            mutation_probabilities_mean = np.mean(
                np.array([indiv.mutation_probability for indiv in self.indivs])
            )
            mutation_strengths_mean = np.mean(
                np.array([indiv.mutation_strength for indiv in self.indivs])
            )
            if self.crossover_strategy != CrossoverStrategy.NONE:
                crossover_probabilities_mean = np.mean(
                    np.array([indiv.crossover_probability for indiv in self.indivs])
                )
            else:
                crossover_probabilities_mean = None
        else:
            mutation_probabilities_mean = None
            mutation_strengths_mean = None
            crossover_probabilities_mean = None

        # Logging
        self.history_logger.log(
            {
                "generation": self.generation_num,
                "best_fitness": self.best_fitness,
                "worst_fitness": self.worst_fitness,
                "mean_fitness": self.mean_fitness,
                "median_fitness": self.median_fitness,
                "std_fitness": self.std_fitness,
                "iqr_fitness": self.iqr_fitness,
                "mutation_probability": self.mutation_probability_effective,
                "mutation_strength": self.mutation_strength_effective,
                "mutation_probability_mean": mutation_probabilities_mean,
                "mutation_strength_mean": mutation_strengths_mean,
                "crossover_probability_mean": crossover_probabilities_mean,
                "crossover_probability": self.crossover_rate_effective,
                "diversity": self.diversity,
            }
        )

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

    @property
    def mutation_probability_effective(self) -> float | None:
        """Return the global mutation rate if defined for the current strategy."""
        if self.mutation_strategy in {
            MutationStrategy.CONSTANT,
            MutationStrategy.EXPONENTIAL_DECAY,
            MutationStrategy.ADAPTIVE_GLOBAL,
        }:
            return self.mutation_probability
        return None

    @property
    def mutation_strength_effective(self) -> float | None:
        """Return the global mutation strength if defined for the current strategy."""
        if self.mutation_strategy in {
            MutationStrategy.CONSTANT,
            MutationStrategy.EXPONENTIAL_DECAY,
            MutationStrategy.ADAPTIVE_GLOBAL,
        }:
            return self.mutation_strength
        return None

    @property
    def crossover_rate_effective(self) -> float | None:
        """Return the global crossover rate if defined for the current strategy."""
        if self.crossover_strategy in {
            CrossoverStrategy.CONSTANT,
            CrossoverStrategy.EXPONENTIAL_DECAY,
            CrossoverStrategy.ADAPTIVE_GLOBAL,
        }:
            return self.crossover_probability
        return None

    @property
    def mutation_strength_bias_effective(self) -> float | None:
        """Return the mutation strength bias if defined (e.g. for neural
        representation)."""
        return getattr(self, "mutation_strength_bias", None)


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


def _merge_config(
    config: Optional[Dict[str, Any]], default_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge user-provided configuration with default values."""

    base = deepcopy(default_config)
    if config:
        # Merge recursively (flat structure for simplicity)
        for k, v in config.items():
            if isinstance(v, dict) and k in base:
                base[k].update(v)
            else:
                base[k] = v
    return base


##############################################################################
# EOF
