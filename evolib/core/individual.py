# SPDX-License-Identifier: MIT
"""
individual.py - Definition and functionality of evolutionary individuals.

This module defines the `Indiv` class, representing a single individual
within a population used in evolutionary algorithms. Each individual
contains a parameter vector, fitness value, and potentially other
adaptive traits such as mutation rate or strength.

It supports initialization, parameter bounds, fitness assignment,
and cloning operations. The design enables use in both simple and
advanced strategies, including individual-level adaptation and
multi-objective optimization.

Typical use cases include:
- Representation of solution candidates in genetic and evolutionary strategies.
- Adaptive mutation schemes on a per-individual basis.
- Integration into population-level operations (selection, crossover, etc.).

Attributes:
    para (any): Parameter vector.
    fitness (float | None): Fitness value assigned after evaluation.
    mutation_probability (float | None): Optional per-individual mutation rate.
    mutation_strength (float | None): Optional per-individual mutation strength.

Classes:
    Indiv: Core data structure for evolutionary optimization.
"""
from copy import deepcopy
from typing import Any, Dict, Optional

from evolib.interfaces.enums import Origin


class Indiv:
    """
    Represents an individual in an evolutionary optimization algorithm.

    Attributes:
        para (Any): Parameters of the individual (e.g., list, array).
        fitness (float): Fitness value of the individual.
        age (int): Current age of the individual.
        max_age (Optional[int]): Maximum allowed age of the individual.
        origin (str): Origin of the individual ('parent' or 'child').
        parent_idx (Optional[int]): Index of the parent individual.
        mutation_strength (float): Mutation strength.
        mutation_strength_bias (float): Bias term for mutation strength.
        mutation_probability (float): Mutation rate.
    """

    __slots__ = (
        "para",
        "fitness",
        "age",
        "max_age",
        "origin",
        "parent_idx",
        "mutation_strength",
        "mutation_strengths",
        "mutation_probability",
        "mutation_strength_bias",
        "crossover_probability",
        "tau",
        "extra_metrics",
    )

    extra_metrics: dict[str, float]

    def __init__(self, para: Any = None):
        """
        Initializes an individual with the given parameters.

        Args:
            para (Any, optional): Parameter values of the individual. Default: None.
        """
        self.para = para
        self.fitness: float = float("inf")  # Optional[float] = None
        self.age = 0
        self.max_age = 0
        self.origin: Origin = Origin.PARENT
        self.parent_idx: Optional[int] = None

        self.mutation_strength: float | None = None
        self.mutation_strengths: list[float] | None = None
        self.mutation_strength_bias: float | None = None
        self.mutation_probability: float | None = None
        self.crossover_probability: float | None = None

        self.tau = 0.0

        self.extra_metrics = {}

    def __lt__(self, other: "Indiv") -> bool:
        return self.fitness < other.fitness

    def print_status(self) -> None:
        """Prints information about the individual."""
        mutation_probability = getattr(self, "mutation_probability", None)
        mutation_strength = getattr(self, "mutation_strength", None)
        mutation_strength_bias = getattr(self, "mutation_strength_bias", None)
        crossover_probability = getattr(self, "crossover_probability", None)

        print("Individual:")
        print(f"  Fitness: {self.fitness}")
        print(f"  Age: {self.age}")
        print(f"  Max Age: {self.max_age}")
        print(f"  Origin: {self.origin}")
        print(f"  Parent Index: {self.parent_idx}")
        if mutation_strength is not None:
            print(f"  Mutation Strength: {mutation_strength:.4f}")
        if mutation_strength_bias is not None:
            print(f"  Mutation Strength Bias: {mutation_strength_bias:.4f}")
        if mutation_probability is not None:
            print(f"  Mutation Rate: {mutation_probability:.4f}")
        if crossover_probability is not None:
            print(f"  Crossover Rate: {crossover_probability:.4f}")

    def to_dict(self) -> Dict:
        """Return a dictionary with selected attributes for logging or serialization."""
        return {
            "fitness": self.fitness,
            "age": self.age,
            "mutation_strength": self.mutation_strength,
            "mutation_strength_bias": self.mutation_strength_bias,
            "mutation_probability": self.mutation_probability,
        }

    def is_parent(self) -> bool:
        """Return True if the individual is a parent."""
        return self.origin == Origin.PARENT

    def is_child(self) -> bool:
        """Return True if the individual is an offspring."""
        return self.origin == Origin.OFFSPRING

    def copy(self) -> "Indiv":
        """
        Create a copy of the individual.

        This method can be overridden in subclasses to implement
        optimized or custom copy behavior.

        Returns:
            Indiv: A copy of this individual.
        """
        return deepcopy(self)
