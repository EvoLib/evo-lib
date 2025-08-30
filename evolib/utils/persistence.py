# SPDX-License-Identifier: MIT

import pickle

from evolib.core.individual import Indiv
from evolib.core.population import Pop


def save_population_pickle(pop: Pop, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(pop, f)


def load_population_pickle(path: str) -> Pop:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_indiv(indiv: Indiv, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(indiv, f)


def load_indiv(path: str) -> Indiv:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_best_indiv(pop: Pop, path: str) -> None:
    save_indiv(pop.best(), path)
