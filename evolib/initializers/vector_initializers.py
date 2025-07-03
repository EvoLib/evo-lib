# SPDX-License-Identifier: MIT
"""Initializers for ParaVector representations."""

from typing import Callable

import numpy as np

from evolib.core.population import Pop
from evolib.initializers.registry import register_initializer
from evolib.representation.vector import ParaVector
from evolib.interfaces.enums import CrossoverStrategy, MutationStrategy


def apply_config(cfg: dict, pv: ParaVector) -> None:
    # Representation
    representation_cfg = cfg.get("representation", {})
    pv.representation = representation_cfg["type"]

    pv.dim = representation_cfg["dim"]
    pv.tau = representation_cfg.get("tau", 0.0)
    pv.bounds = representation_cfg["bounds"]

    # Mutation
    mutation_cfg = cfg.get("mutation", {})
    pv.mutation_strategy = MutationStrategy(mutation_cfg.get("strategy", "constant"))

    if pv.mutation_strategy == MutationStrategy.CONSTANT:
        pv.mutation_probability = mutation_cfg["probability"]
        pv.mutation_strength = mutation_cfg["strength"]

    if pv.mutation_strategy == MutationStrategy.EXPONENTIAL_DECAY:
        pv.min_mutation_probability = mutation_cfg["min_probability"]
        pv.max_mutation_probability = mutation_cfg["max_probability"]

        pv.min_mutation_strength = mutation_cfg["min_strength"]
        pv.max_mutation_strength = mutation_cfg["max_strength"]

    if pv.mutation_strategy == MutationStrategy.ADAPTIVE_GLOBAL:
        pv.mutation_probability = mutation_cfg["init_probability"]
        pv.min_mutation_probability = mutation_cfg["min_probability"]
        pv.max_mutation_probability = mutation_cfg["max_probability"]

        pv.mutation_strength = mutation_cfg["init_strength"]
        pv.min_mutation_strength = mutation_cfg["min_strength"]
        pv.max_mutation_strength = mutation_cfg["max_strength"]

        pv.min_diversity_threshold = mutation_cfg["min_diversity_threshold"]
        pv.max_diversity_threshold = mutation_cfg["max_diversity_threshold"]

        pv.mutation_inc_factor = mutation_cfg["increase_factor"]
        pv.mutation_dec_factor = mutation_cfg["decrease_factor"]

    if pv.mutation_strategy == MutationStrategy.ADAPTIVE_INDIVIDUAL:
        pv.mutation_probability = None
        pv.mutation_strength = None

        pv.min_mutation_probability = mutation_cfg["min_probability"]
        pv.max_mutation_probability = mutation_cfg["max_probability"]

        pv.min_mutation_strength = mutation_cfg["min_strength"]
        pv.max_mutation_strength = mutation_cfg["max_strength"]

    if pv.mutation_strategy == MutationStrategy.ADAPTIVE_PER_PARAMETER:
        pv.mutation_probability = None
        pv.mutation_strength = None

        pv.min_mutation_probability = None
        pv.max_mutation_probability = None

        pv.min_mutation_strength = mutation_cfg["min_strength"]
        pv.max_mutation_strength = mutation_cfg["max_strength"]

    # Crossover
    crossover_cfg = cfg.get("crossover", None)
    if crossover_cfg is None:
        pv.crossover_strategy = CrossoverStrategy.NONE
        pv.crossover_probability = None
    else:
        pv.crossover_strategy = CrossoverStrategy(cfg["crossover"]["strategy"])
        if pv.crossover_strategy == CrossoverStrategy.CONSTANT:
            pv.crossover_probability = cfg["crossover"]["probability"]

        if pv.crossover_strategy == CrossoverStrategy.EXPONENTIAL_DECAY:
            pv.min_crossover_probability = cfg["crossover"]["min_probability"]
            pv.max_crossover_probability = cfg["crossover"]["max_probability"]

        if pv.crossover_strategy == CrossoverStrategy.ADAPTIVE_GLOBAL:
            pv.crossover_probability = cfg["crossover"]["init_probability"]
            pv.min_crossover_probability = cfg["crossover"]["min_probability"]
            pv.max_crossover_probability = cfg["crossover"]["max_probability"]

            pv.crossover_inc_factor = cfg["crossover"]["increase_factor"]
            pv.crossover_dec_factor = cfg["crossover"]["decrease_factor"]


def random_initializer(cfg: dict) -> Callable[[Pop], ParaVector]:

    def init_fn(_: Pop) -> ParaVector:
        pv = ParaVector()
        apply_config(cfg, pv)
        pv.vector = np.random.uniform(pv.bounds[0], pv.bounds[1], size=pv.dim)
        return pv

    return init_fn


def zero_initializer(cfg: dict) -> Callable[[Pop], ParaVector]:

    def init_fn(_: Pop) -> ParaVector:
        pv = ParaVector()
        apply_config(cfg, pv)
        pv.vector = np.zeros(pv.dim)
        return pv

    return init_fn


def fixed_initializer(cfg: dict) -> Callable[[Pop], ParaVector]:
    values = np.array(cfg["representation"]["values"])

    def init_fn(_: Pop) -> ParaVector:
        pv = ParaVector()
        apply_config(cfg, pv)
        pv.vector = values.copy()
        return pv

    return init_fn


def normal_initializer(cfg: dict) -> Callable[[Pop], ParaVector]:
    mean = float(cfg.get("mean", 0.0))
    std = float(cfg.get("std", 1.0))

    def init_fn(_: Pop) -> ParaVector:
        pv = ParaVector()
        apply_config(cfg, pv)
        pv.vector = np.random.normal(loc=mean, scale=std, size=pv.dim)
        return pv

    return init_fn


def vector_adaptive_initializer(cfg: dict) -> Callable[[Pop], ParaVector]:

    def init_fn(_: Pop) -> ParaVector:
        pv = ParaVector()
        apply_config(cfg, pv)
        pv.vector = np.random.uniform(pv.bounds[0], pv.bounds[1], size=pv.dim)
        pv.para_mutation_strengths = np.full(pv.dim, pv.mutation_strength)
        return pv

    return init_fn


# Registration
register_initializer("random_initializer", random_initializer)
register_initializer("zero_initializer", zero_initializer)
register_initializer("fixed_initializer", fixed_initializer)
register_initializer("normal_initializer", normal_initializer)
register_initializer("vector_adaptive", vector_adaptive_initializer)
