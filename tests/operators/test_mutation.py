import numpy as np

from evolib.operators.mutation import (
    adapt_mutation_probability_by_diversity,
    adapt_mutation_strength,
    adapt_mutation_strength_by_diversity,
    adapt_mutation_strengths,
    adapted_tau,
    exponential_mutation_probability,
    exponential_mutation_strength,
)
from evolib.representation.evo_params import EvoControlParams

# --- BASIC UTILS ---


def test_adapted_tau() -> None:
    assert adapted_tau(16) == 0.25
    assert adapted_tau(0) == 0.0


# --- EXPONENTIAL STRATEGIES ---


def test_exponential_mutation_strength_decay_respects_param_bounds() -> None:
    params = EvoControlParams()
    params.min_mutation_strength = 0.1
    params.max_mutation_strength = 0.9

    val = exponential_mutation_strength(params, gen=20, max_gen=100)
    assert isinstance(val, float)
    assert 0.1 <= val <= 0.9


def test_exponential_mutation_probability_decay_respects_param_bounds() -> None:
    params = EvoControlParams()
    params.min_mutation_probability = 0.1
    params.max_mutation_probability = 0.9

    val = exponential_mutation_probability(params, gen=20, max_gen=100)
    assert isinstance(val, float)
    assert 0.1 <= val <= 0.9


# --- LOG-NORMAL SELF-ADAPTATION (CLIPPED BY BOUNDS ARGUMENT) ---


def test_adapt_mutation_strength_respects_bounds() -> None:
    np.random.seed(42)
    params = EvoControlParams()
    params.mutation_strength = 0.3
    params.tau = 0.5
    bounds = (0.1, 0.5)

    val = adapt_mutation_strength(params, bounds=bounds)
    assert isinstance(val, float)
    assert bounds[0] <= val <= bounds[1]


def test_adapt_mutation_strengths_vector_clipped_by_bounds() -> None:
    np.random.seed(42)
    params = EvoControlParams()
    params.tau = 0.5
    params.mutation_strengths = np.full(4, 0.3)
    bounds = (0.1, 0.5)

    vec = adapt_mutation_strengths(params, bounds=bounds)
    assert vec.shape == (4,)
    assert np.all(vec >= bounds[0])
    assert np.all(vec <= bounds[1])


# --- DIVERSITY ADAPTATION (CLIPPED BY PARAM-GRENZEN) ---


def test_adapt_mutation_strength_by_diversity_increase() -> None:
    params = EvoControlParams()
    params.min_diversity_threshold = 0.1
    params.max_diversity_threshold = 0.5
    params.mutation_inc_factor = 2.0
    params.mutation_dec_factor = 0.5
    params.min_mutation_strength = 0.1
    params.max_mutation_strength = 0.9

    # low diversity -> increase (but clipped at max)
    val = adapt_mutation_strength_by_diversity(0.6, 0.05, params)
    assert val == min(0.6 * 2.0, 0.9)


def test_adapt_mutation_strength_by_diversity_decrease() -> None:
    params = EvoControlParams()
    params.min_diversity_threshold = 0.1
    params.max_diversity_threshold = 0.5
    params.mutation_inc_factor = 2.0
    params.mutation_dec_factor = 0.5
    params.min_mutation_strength = 0.1
    params.max_mutation_strength = 0.9

    # high diversity -> decrease (but clipped at min)
    val = adapt_mutation_strength_by_diversity(0.2, 0.6, params)
    assert val == max(0.2 * 0.5, 0.1)


def test_adapt_mutation_probability_by_diversity_increase() -> None:
    params = EvoControlParams()
    params.min_diversity_threshold = 0.1
    params.max_diversity_threshold = 0.5
    params.mutation_inc_factor = 2.0
    params.mutation_dec_factor = 0.5
    params.min_mutation_probability = 0.1
    params.max_mutation_probability = 0.9

    val = adapt_mutation_probability_by_diversity(0.4, 0.05, params)
    assert val == min(0.4 * 2.0, 0.9)


def test_adapt_mutation_probability_by_diversity_decrease() -> None:
    params = EvoControlParams()
    params.min_diversity_threshold = 0.1
    params.max_diversity_threshold = 0.5
    params.mutation_inc_factor = 2.0
    params.mutation_dec_factor = 0.5
    params.min_mutation_probability = 0.1
    params.max_mutation_probability = 0.9

    val = adapt_mutation_probability_by_diversity(0.4, 0.6, params)
    assert val == max(0.4 * 0.5, 0.1)
