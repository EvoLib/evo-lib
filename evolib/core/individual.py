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
from uuid import uuid4

from evolib.interfaces.enums import Origin
from evolib.representation.dummy import ParaDummy


class Indiv:
    """Represents an individual in an evolutionary optimization algorithm."""

    #: unique identifier (UUID)
    id: str

    #: para (Any): Parameter values of the individual. Default: None.
    para: Any

    #: fitness (float): Fitness value of the individual.
    #: None means the individual has not yet been evaluated.
    fitness: float | None

    #: is_evaluated (bool): Flag indicating whether the individual's fitness
    #: has been evaluated. False after initialization, True after the first
    #: call to a fitness evaluation.
    is_evaluated: bool

    #: age (int): Current age of the individual. 0 means "no limit".
    age: int

    #: max_age (Optional[int]): Maximum allowed age of the individual.
    max_age: int

    #: origin (str): Origin of the individual (e.g. Origin.PARENT, Origin.OFFSPRING).
    origin: Origin

    #: is_elite (bool): Flag to explicitly mark elites for logging/analysis.
    is_elite: bool

    #: extra_metrics (dict[str, float]): Optional extra per-individual
    #: metrics for logging.
    extra_metrics: dict[str, float]

    #: parent_id (Optional[str]): Unique identifier of the parent individual.
    #: Used for lineage and survival tracking.
    parent_id: Optional[str]

    #: birth_gen (int): Generation in which this individual was created.
    #: Set by the reproduction operator. Defaults to 0 for the initial population.
    birth_gen: int

    #: exit_gen (Optional[int]): Generation in which this individual was removed
    #: from the population. None if still alive. Can be assigned during analysis.
    exit_gen: Optional[int]

    #: is_structural_mutant (bool): Indicates whether this individual resulted
    #: from a structural mutation (e.g., add/remove neuron, connection, etc.).
    #: Used for distinguishing structural from weight-only mutations.
    is_structural_mutant: bool

    #: heli_seed (bool): True if this individual was created as a HELI seed
    #: during micro-evolution incubation.
    heli_seed: bool

    #: heli_reintegrated (bool): True if this individual was reintegrated into
    #: the main population after a HELI subpopulation run.
    heli_reintegrated: bool

    __slots__ = (
        "id",
        "para",
        "fitness",
        "is_evaluated",
        "age",
        "max_age",
        "origin",
        "extra_metrics",
        "is_elite",
        # --- Lineage tracking extensions ---
        "birth_gen",
        "parent_id",
        "exit_gen",
        "heli_seed",
        "heli_reintegrated",
        "is_structural_mutant",
    )

    def __init__(self, para: Any = None):
        self.id: str = str(uuid4())
        self.para = para if para is not None else ParaDummy()
        self.fitness = None
        self.is_evaluated = False
        self.age = 0
        self.max_age = 0
        self.origin = Origin.PARENT
        self.is_elite = False
        self.extra_metrics = {}

        # --- Lineage tracking extensions ---
        self.birth_gen = 0
        self.exit_gen = None
        self.parent_id = None
        self.heli_seed = False
        self.heli_reintegrated = False
        self.is_structural_mutant = False

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

        if self.para is not None and hasattr(self.para, "mutate"):
            self.para.mutate()

        # Synchronize structural change flag
        struct_mutated = getattr(self.para, "has_structural_change", False)
        self.is_structural_mutant = bool(struct_mutated)

    def crossover(self) -> None:
        """
        Apply crossover to this individual.

        Delegates the crossover process to the underlying parameter object `para`.
        This ensures that crossover behavior is defined polymorphically in the
        specific `ParaBase` subclass (e.g. `Vector`, `ParaNet`, ...).
        """
        if self.para is not None and hasattr(self.para, "crossover"):
            self.para.crossover()

    def get_status(self) -> str:
        """Get a one-line status string of the parameter representation."""
        if self.para is None:
            return "<no parameter>"

        if hasattr(self.para, "get_status"):
            return self.para.get_status()
        return "para has no status method"

    def print_status(self) -> None:
        """Print a human-readable status summary of this individual and its
        components."""
        print("Individual:")
        print(f"  Fitness: {self.fitness}")
        print(f"  Age: {self.age}")
        print(f"  Max Age: {self.max_age}")
        print(f"  Origin: {self.origin}")

        print(f"   ID: {self.id}")
        print(f"   Parent-ID: {self.parent_id}")
        print(f"   birth_gen: {self.birth_gen}")
        print(f"   is_elite: {self.is_elite}")
        print(f"   is_structural_mutant: {self.is_structural_mutant}")
        print(f"   heli_seed: {self.heli_seed}")
        print(f"   heli_reintegrated: {self.heli_reintegrated}")

        if self.para is None:
            print("<no parameter>")
            return

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

    def copy(
        self,
        *,
        reset_id: bool = True,
        reset_fitness: bool = False,
        reset_age: bool = False,
        reset_evaluation: bool = False,
        reset_origin: bool = False,
    ) -> "Indiv":
        """
        Create a copy of the individual, with optional resets.

        Args:
            reset_id (bool): If True (default), assign a new unique ID to the copy.
            reset_fitness (bool): If True, set fitness to None in the copy.
            reset_age (bool): If True, set age to 0 in the copy.
            reset_evaluation (bool): If True, set is_evaluated = False in the copy.
            reset_origin (bool): If True, set origin = Origin.OFFSPRING

        Returns:
            Indiv: A (possibly reset) copy of this individual.
        """
        new_indiv: Indiv = deepcopy(self)

        if reset_id:
            new_indiv.id = str(uuid4())
        if reset_fitness:
            new_indiv.fitness = None
        if reset_age:
            new_indiv.age = 0
        if reset_evaluation:
            new_indiv.is_evaluated = False
        if reset_origin:
            from evolib.interfaces.enums import Origin

            new_indiv.origin = Origin.OFFSPRING

        # Lineage reset
        new_indiv.parent_id = self.id
        new_indiv.birth_gen = 0  # assigned later by population
        new_indiv.exit_gen = None
        new_indiv.is_structural_mutant = False
        new_indiv.heli_seed = False
        new_indiv.heli_reintegrated = False

        return new_indiv
