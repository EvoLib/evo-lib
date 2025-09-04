# SPDX-License-Identifier: MIT
"""
individual.py - Definition and functionality of evolutionary individuals.

This module defines the `Indiv` class, representing a single individual
within a population used in evolutionary algorithms.

It supports initialization, parameter bounds, fitness assignment,
and cloning operations. The design enables use in both simple and
advanced strategies, including individual-level adaptation and
multi-objective optimization.

Typical use cases include:
- Representation of solution candidates in genetic and evolutionary strategies.
- Adaptive mutation schemes on a per-individual basis.
- Integration into population-level operations (selection, crossover, etc.).

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
            None means the individual has not yet been evaluated.
        is_evaluated (bool): Flag indicating whether the individual's fitness
            has been evaluated. False after initialization, True after the first
            call to a fitness evaluation.
        age (int): Current age of the individual.
        max_age (Optional[int]): Maximum allowed age of the individual.
        origin (str): Origin of the individual ('parent' or 'child').
        parent_idx (Optional[int]): Index of the parent individual.
    """

    __slots__ = (
        "para",
        "fitness",
        "is_evaluated",
        "age",
        "max_age",
        "origin",
        "parent_idx",
        "extra_metrics",
        "is_elite",
    )

    extra_metrics: dict[str, float]

    def __init__(self, para: Any = None):
        """
        Initializes an individual with the given parameters.

        Args:
            para (Any, optional): Parameter values of the individual. Default: None.
        """
        self.para = para
        self.fitness: float | None = None
        self.is_evaluated: bool = False
        self.age = 0
        self.max_age = 0
        self.origin: Origin = Origin.PARENT
        self.parent_idx: Optional[int] = None
        self.is_elite: bool = False

        self.extra_metrics = {}

    def __lt__(self, other: "Indiv") -> bool:
        if self.fitness is None or other.fitness is None:
            raise ValueError("Comparison attempted with unevaluated individuals")
        return self.fitness < other.fitness

    def mutate(self) -> None:
        """
        Apply mutation to this individual.

        Delegates the mutation process to the underlying parameter object `para`.
        This ensures that mutation behavior is defined polymorphically in the
        specific `ParaBase` subclass (e.g. `Vector`, `ParaNet`, ...).
        """

        if hasattr(self.para, "__iter__"):
            for p in self.para:
                p.mutate()
        else:
            self.para.mutate()

    def crossover(self) -> None:
        """
        Apply crossover to this individual.

        Delegates the crossover process to the underlying parameter object `para`.
        This ensures that crossover behavior is defined polymorphically in the
        specific `ParaBase` subclass (e.g. `Vector`, `ParaNet`, ...).
        """
        if hasattr(self.para, "__iter__"):
            for p in self.para:
                p.crossover()
            else:
                self.para.crossover()

    def get_status(self) -> str:
        """Get status string from all components."""
        if hasattr(self.para, "get_status"):
            return self.para.get_status()
        return "para has no status method"

    def print_status(self) -> None:
        """Prints information about the individual."""
        print("Individual:")
        print(f"  Fitness: {self.fitness}")
        print(f"  Age: {self.age}")
        print(f"  Max Age: {self.max_age}")
        print(f"  Origin: {self.origin}")
        print(f"  Parent Index: {self.parent_idx}")

        if hasattr(self.para, "__iter__"):
            for i, p in enumerate(self.para):
                print(f"  Component {i}:")
                if hasattr(p, "print_status"):
                    p.print_status()
                else:
                    print("    <no print_status>")
        elif hasattr(self.para, "print_status"):
            self.para.print_status()
        else:
            print("  <no parameter status>")

    def to_dict(self) -> Dict:
        """Return a dictionary with selected attributes for logging or serialization."""
        return {
            "fitness": self.fitness,
            "age": self.age,
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
