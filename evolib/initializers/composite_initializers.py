from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from evolib.core.population import Pop

from evolib.config.schema import FullConfig
from evolib.initializers.registry import get_initializer
from evolib.representation.base import ParaBase
from evolib.representation.composite import ParaComposite


def composite_initializer(cfg: FullConfig) -> Callable[["Pop"], ParaComposite]:
    """
    Factory function to create a composite ParaInitializer from FullConfig.

    Each component defined under cfg.modules will be initialized using its
    assigned initializer and assembled into a ParaComposite.

    Returns:
        Callable[[Pop], ParaComposite]: A function that takes a Pop and
        returns a ParaComposite.
    """

    def init_fn(pop: "Pop") -> ParaComposite:
        components: dict[str, ParaBase] = {}

        for name, component_cfg in cfg.modules.items():
            # Verwende denselben Initializer-Mechanismus wie bei klassischen Vektoren
            initializer_fn = get_initializer(component_cfg.initializer)

            # Übergib nur die ComponentConfig als Scheinausschnitt (für apply_config())
            para = initializer_fn(component_cfg)(pop)

            components[name] = para

        return ParaComposite(components)

    return init_fn
