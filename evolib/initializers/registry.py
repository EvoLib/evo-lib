from __future__ import annotations

from evolib.interfaces.types import ParaInitializer

# Zentrales Registry-Objekt
initializer_registry: dict[str, ParaInitializer] = {}


def register_initializer(name: str, fn: ParaInitializer) -> None:
    """
    Register a new Para initializer function under a string key.

    Args:
        name (str): Unique identifier for this initializer, e.g. 'vector' or 'tree'.
        fn (Callable): Factory function that returns a Para-initializer
        given (cfg, dim).
    """
    if name in initializer_registry:
        raise ValueError(f"Initializer '{name}' is already registered.")
    initializer_registry[name] = fn


def get_initializer(name: str) -> ParaInitializer:
    """
    Retrieve a registered initializer function by name.

    Args:
        name (str): The key under which the initializer was registered.

    Returns:
        Callable[[dict], Callable[[Pop], ParaBase]]: A two-stage initializer function.
            The first stage receives the configuration dictionary.
            The second stage returns a callable that creates a ParaBase instance
            when passed a Pop object.

    Raises:
        ValueError: If the name is not registered.
    """
    if name not in initializer_registry:
        raise ValueError(f"Initializer '{name}' is not registered.")
    return initializer_registry[name]
