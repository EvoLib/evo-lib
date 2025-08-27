# SPDX-License-Identifier: MIT

from typing import Dict

from evolib.interfaces.types import ModuleConfig
from evolib.representation.base import ParaBase


class ParaComposite(ParaBase):
    """
    Composite container for multiple evolvable parameter representations.

    This class allows a single individual (Indiv) to consist of multiple, logically
    distinct ParaBase components, such as:

        - A global Vector (e.g. hyperparameters)
        - One or more neural network components (e.g. EvoNet, NetVector)
        - Specialized modules (e.g. rule systems, PID controllers)

    The composite supports standard ParaBase operations like mutate() and
    crossover_with(), delegating them to its components.

    Access to individual components is provided via name or index:
        para["controller"], para[0]
    """

    def __init__(self, components: Dict[str, ParaBase]):
        self.components = components

    def __getitem__(self, key: str | int) -> ParaBase:
        if isinstance(key, int):
            return list(self.components.values())[key]
        return self.components[key]

    def __len__(self) -> int:
        return len(self.components)

    def mutate(self) -> None:
        for comp in self.components.values():
            comp.mutate()

    def apply_config(self, cfg: ModuleConfig) -> None:
        """
        Apply sub-configs to each component of the ParaComposite.

        Each component receives the matching config from cfg.modules by name.
        """
        if not hasattr(cfg, "modules"):
            raise TypeError(
                "ParaComposite requires a FullConfig with 'modules', "
                f"but received: {type(cfg).__name__}"
            )

        for name, component in self.components.items():
            subcfg = cfg.modules.get(name)
            if subcfg is not None:
                component.apply_config(subcfg)
            else:
                print(f"[Warning] No config found for component '{name}' – skipping.")

    def crossover_with(self, partner: ParaBase) -> None:
        if not isinstance(partner, ParaComposite):
            raise TypeError("Crossover partner must also be ParaComposite")

        for (k1, comp1), (k2, comp2) in zip(
            self.components.items(), partner.components.items()
        ):
            if k1 != k2:
                raise ValueError("Component keys do not match between composites.")
            comp1.crossover_with(comp2)

    def print_status(self) -> None:
        for name, comp in self.components.items():
            print(f"Component '{name}':")
            comp.print_status()

    def get_status(self) -> dict:
        return {
            name: component.get_status() for name, component in self.components.items()
        }

    def get_history(self) -> dict[str, float]:
        """
        Aggregates history dicts from all components for logging purposes.

        Keys are prefixed with their component name.
        """
        history = {}
        for name, comp in self.components.items():
            comp_history = comp.get_history()
            for key, value in comp_history.items():
                history[f"{name}_{key}"] = value
        return history

    def update_mutation_parameters(
        self, generation: int, max_generations: int, diversity_ema: float | None = None
    ) -> None:
        for comp in self.components.values():
            comp.update_mutation_parameters(generation, max_generations, diversity_ema)

    def update_crossover_parameters(
        self, generation: int, max_generations: int, diversity_ema: float | None = None
    ) -> None:
        for comp in self.components.values():
            comp.update_crossover_parameters(generation, max_generations, diversity_ema)
