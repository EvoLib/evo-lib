# SPDX-License-Identifier: MIT
"""
Persistence utilities for saving and resuming evolutionary runs.

This module provides standardized support for:
- Checkpointing full populations (Pop)
- Saving and restoring best individuals (Indiv)
- Loading checkpoints for resuming interrupted runs

All files are stored in the 'checkpoints/' directory by default.
"""


import pickle
from pathlib import Path
from typing import Any, Optional, cast

from evolib.core.individual import Indiv
from evolib.core.population import Pop
from evolib.interfaces.types import FitnessFunction

# Internal checkpoint directory
_CHECKDIR = Path("checkpoints")
_CHECKDIR.mkdir(exist_ok=True)

CHECKPOINT_DIR = _CHECKDIR


def _checkpoint_path(run_name: str) -> Path:
    return _CHECKDIR / f"{run_name}.pkl"


def _best_indiv_path(run_name: str) -> Path:
    return _CHECKDIR / f"{run_name}_best.pkl"


def save_checkpoint(pop: Pop, *, run_name: str = "default") -> None:
    """
    Save the full population to a checkpoint file.

    The file will be stored under 'checkpoints/{run_name}.pkl'.

    Args:
        pop (Pop): Population instance to be saved.
        run_name (str): Optional name to distinguish checkpoint runs.
    """
    path = _checkpoint_path(run_name)

    initializer_backup = pop.para_initializer
    pop.para_initializer = cast(Any, None)

    try:
        save_population_pickle(pop, path)
    finally:
        pop.para_initializer = initializer_backup


def resume_from_checkpoint(
    run_name: str = "default",
    fitness_fn: Optional[FitnessFunction] = None,
) -> Pop:
    """
    Load a previously saved population from a checkpoint file and optionally re-assign
    the fitness function.

    Args:
        run_name (str): Checkpoint name (stored as 'checkpoints/{run_name}.pkl').
        fitness_fn (Optional[FitnessFunction]): Fitness function to be reattached
            (required if it was not serializable).

    Returns:
        Pop: Restored population ready to be resumed.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """

    path = _checkpoint_path(run_name)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint '{path}' not found.")

    pop = load_population_pickle(path)

    if fitness_fn:
        pop.set_fitness_function(fitness_fn)

    return pop


def save_best_indiv(pop: Pop, *, run_name: str = "default") -> None:
    """
    Save the best individual from a population to a separate file.

    The file will be stored as 'checkpoints/{run_name}_best.pkl'.

    Args:
        pop (Pop): Population from which to extract and save the best individual.
        run_name (str): Optional name to identify the saved individual.
    """

    best = pop.best()
    path = _best_indiv_path(run_name)
    save_indiv(best, path)


def load_best_indiv(run_name: str = "default") -> Indiv:
    """
    Load the best individual saved via `save_best_indiv()`.

    Args:
        run_name (str): Name used during saving (without '_best.pkl' suffix).

    Returns:
        Indiv: Deserialized best individual.

    Raises:
        FileNotFoundError: If the corresponding file does not exist.
    """

    path = _best_indiv_path(run_name)
    if not path.exists():
        raise FileNotFoundError(f"Best-indiv file '{path}' not found.")

    indiv = load_indiv(path)

    return indiv


def save_population_pickle(pop: Pop, path: str | Path) -> None:
    """Save a Pop instance to the specified file using pickle."""
    write_pickle(pop, path)


def load_population_pickle(path: str | Path) -> Pop:
    """Load a Pop instance from a pickle file."""
    return read_pickle(path)


def save_indiv(indiv: Indiv, path: str | Path) -> None:
    """Save an Indiv instance to the specified file using pickle."""
    write_pickle(indiv, Path(path))


def load_indiv(path: str | Path) -> Indiv:
    """Load an Indiv instance from a pickle file."""
    return read_pickle(path)


def write_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(path: str | Path) -> Any:
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)
