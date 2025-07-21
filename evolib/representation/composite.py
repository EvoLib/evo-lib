from typing import List

from evolib.representation.base import ParaBase


class ParaComposite(ParaBase):
    """
    Composite container for multiple evolvable parameter representations.

    This class allows a single individual (Indiv) to consist of multiple, logically
    distinct ParaBase components, such as:

        - A global ParaVector (e.g. hyperparameters)
        - One or more neural network components (e.g. ParaNet, TopoNet)
        - Specialized modules (e.g. rule systems, PID controllers)

    The composite supports standard ParaBase operations like mutate() and
    crossover_with(), delegating them to its components.

    Access to individual components is provided via indexing (e.g. para[0], para[1]).

    Example:
        indiv.para = ParaComposite([
            ParaVector(...),
            ParaNet(...),
            MyCustomController(...)
        ])

        indiv.para[1].mutate()
        output = indiv.para[2].forward(x)
    """

    def __init__(self, components: List[ParaBase]):
        self.components = components

    def __getitem__(self, index: int) -> ParaBase:
        return self.components[index]

    def __len__(self) -> int:
        return len(self.components)

    def mutate(self) -> None:
        for comp in self.components:
            comp.mutate()

    def crossover_with(self, partner: ParaBase) -> None:
        if not isinstance(partner, ParaComposite):
            raise TypeError("Crossover partner must also be ParaComposite")
        for self_comp, partner_comp in zip(self.components, partner.components):
            self_comp.crossover_with(partner_comp)

    def print_status(self) -> None:
        for i, comp in enumerate(self.components):
            print(f"Component {i}:")
            comp.print_status()

    def get_status(self) -> str:
        return " | ".join(
            f"[{i}] {comp.get_status()}" for i, comp in enumerate(self.components)
        )
