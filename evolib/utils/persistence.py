# SPDX-License-Identifier: MIT

import pickle
from pathlib import Path
from typing import Any, Optional

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
    Save the current population to a standardized checkpoint path.

    Args:
    pop (Pop): Population to save.
    run_name (str): Identifier for the checkpoint file.
    """

    path = _checkpoint_path(run_name)
    save_population_pickle(pop, path)


def resume_from_checkpoint(
    run_name: str = "default",
    fitness_fn: Optional[FitnessFunction] = None,
) -> Pop:
    """
    Resume a previously saved evolutionary run.

    Args:
    run_name (str): Identifier of the checkpoint file.
    fitness_fn (callable, optional): Fitness function to assign.

    Returns:
    Pop: The resumed population.
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
    Save the best individual of the population.

    Args:
    pop (Pop): Population from which to extract the best.
    run_name (str): Identifier for the output file.
    """
    best = pop.best()
    path = _best_indiv_path(run_name)
    save_indiv(best, path)


def load_best_indiv(run_name: str = "default") -> Indiv:
    """
    Load the best individual saved separately from a checkpoint run.

    Args:
    run_name (str): Identifier used during saving.


    Returns:
    Indiv: The deserialized best individual.
    """
    path = _best_indiv_path(run_name)
    if not path.exists():
        raise FileNotFoundError(f"Best-indiv file '{path}' not found.")

    indiv = load_indiv(path)

    return indiv


def save_population_pickle(pop: Pop, path: str | Path) -> None:
    write_pickle(pop, path)


def load_population_pickle(path: str | Path) -> Pop:
    return read_pickle(path)


def save_indiv(indiv: Indiv, path: str | Path) -> None:
    write_pickle(indiv, Path(path))


def load_indiv(path: str | Path) -> Indiv:
    return read_pickle(path)


def write_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(path: str | Path) -> Any:
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)
