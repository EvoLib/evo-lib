# SPDX-License-Identifier: MIT
"""Initializers for ParaVector representations."""

from typing import Callable

import numpy as np

from evolib.core.population import Pop
from evolib.initializers.registry import register_initializer
from evolib.representation.vector import ParaVector


def random_initializer(cfg: dict) -> Callable[[Pop], ParaVector]:
    dim = int(cfg["dim"])
    bounds = tuple(cfg.get("bounds", (-1.0, 1.0)))

    def init_fn(_: Pop) -> ParaVector:
        vector = np.random.uniform(bounds[0], bounds[1], size=dim)
        return ParaVector(vector=vector)

    return init_fn


def zero_initializer(cfg: dict) -> Callable[[Pop], ParaVector]:
    dim = int(cfg["dim"])

    def init_fn(_: Pop) -> ParaVector:
        return ParaVector(vector=np.zeros(dim))

    return init_fn


def fixed_initializer(cfg: dict) -> Callable[[Pop], ParaVector]:
    values = np.array(cfg["values"])
    tau = float(cfg.get("tau", 0.0))

    def init_fn(_: Pop) -> ParaVector:
        return ParaVector(vector=values.copy(), tau=tau)

    return init_fn


def normal_initializer(cfg: dict) -> Callable[[Pop], ParaVector]:
    dim = int(cfg["dim"])
    mean = float(cfg.get("mean", 0.0))
    std = float(cfg.get("std", 1.0))
    tau = float(cfg.get("tau", 0.0))

    def init_fn(_: Pop) -> ParaVector:
        vector = np.random.normal(loc=mean, scale=std, size=dim)
        return ParaVector(vector=vector, tau=tau)

    return init_fn


def vector_adaptive_initializer(cfg: dict) -> Callable[[Pop], ParaVector]:
    dim = int(cfg["dim"])
    bounds = tuple(cfg.get("bounds", (-1.0, 1.0)))
    init_strength = float(cfg.get("init_strength", 0.05))
    tau = float(cfg.get("tau", 0.1))

    def init_fn(_: Pop) -> ParaVector:
        vector = np.random.uniform(bounds[0], bounds[1], size=dim)
        strengths = np.full(dim, init_strength)
        return ParaVector(vector=vector, para_mutation_strengths=strengths, tau=tau)

    return init_fn


# Registration
register_initializer("random_initializer", random_initializer)
register_initializer("zero_initializer", zero_initializer)
register_initializer("fixed_initializer", fixed_initializer)
register_initializer("normal_initializer", normal_initializer)
register_initializer("vector_adaptive", vector_adaptive_initializer)
