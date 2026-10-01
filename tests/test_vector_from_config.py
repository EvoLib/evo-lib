import numpy as np

from evolib.config.base_component_config import MutationConfig
from evolib.config.vector_component_config import VectorComponentConfig
from evolib.interfaces.enums import MutationStrategy
from evolib.representation.vector import Vector


def _constant_mutation() -> MutationConfig:
    return MutationConfig(
        strategy=MutationStrategy.CONSTANT,
        strength=0.1,
        probability=1.0,
    )


def test_vector_from_config_fixed() -> None:
    cfg = VectorComponentConfig(
        dim=3,
        initializer="fixed",
        values=[-0.5, 0.0, 0.5],
        bounds=(-1.0, 1.0),
        mutation=_constant_mutation(),
    )

    para = Vector.from_config(cfg)

    assert np.array_equal(para.vector, np.array([-0.5, 0.0, 0.5]))
    assert para.dim == 3
    assert para.bounds == (-1.0, 1.0)


def test_vector_from_config_uniform_uses_init_bounds() -> None:
    np.random.seed(1)
    cfg = VectorComponentConfig(
        dim=16,
        initializer="uniform",
        bounds=(-1.0, 1.0),
        init_bounds=(-0.25, 0.25),
        mutation=_constant_mutation(),
    )

    para = Vector.from_config(cfg)

    assert para.vector.shape == (16,)
    assert np.all(para.vector >= -0.25)
    assert np.all(para.vector <= 0.25)


def test_vector_from_config_normal_clips_to_init_bounds() -> None:
    np.random.seed(1)
    cfg = VectorComponentConfig(
        dim=16,
        initializer="normal",
        bounds=(-1.0, 1.0),
        init_bounds=(-0.1, 0.1),
        mean=0.0,
        std=1.0,
        mutation=_constant_mutation(),
    )

    para = Vector.from_config(cfg)

    assert para.vector.shape == (16,)
    assert np.all(para.vector >= -0.1)
    assert np.all(para.vector <= 0.1)


def test_vector_from_config_zero() -> None:
    cfg = VectorComponentConfig(
        dim=4,
        initializer="zero",
        bounds=(-1.0, 1.0),
        mutation=_constant_mutation(),
    )

    para = Vector.from_config(cfg)

    assert np.array_equal(para.vector, np.zeros(4))


def test_vector_from_config_adaptive_initializes_sigmas() -> None:
    np.random.seed(1)
    cfg = VectorComponentConfig(
        dim=8,
        initializer="adaptive",
        bounds=(-1.0, 1.0),
        init_bounds=(-0.5, 0.5),
        randomize_mutation_strengths=True,
        mutation=MutationConfig(
            strategy=MutationStrategy.ADAPTIVE_PER_PARAMETER,
            probability=1.0,
            min_strength=0.01,
            max_strength=0.1,
        ),
    )

    para = Vector.from_config(cfg)
    strengths = para.evo_params.mutation_strengths

    assert para.vector.shape == (8,)
    assert np.all(para.vector >= -0.5)
    assert np.all(para.vector <= 0.5)
    assert strengths is not None
    assert strengths.shape == (8,)
    assert np.all(strengths >= 0.01)
    assert np.all(strengths <= 0.1)


def test_vector_from_config_does_not_modify_config() -> None:
    cfg = VectorComponentConfig(
        dim=6,
        initializer="normal",
        bounds=(-1.0, 1.0),
        mutation=_constant_mutation(),
    )
    before = cfg.model_dump()

    Vector.from_config(cfg)

    assert cfg.model_dump() == before


def test_vector_from_config_normal_respects_zero_std() -> None:
    cfg = VectorComponentConfig(
        dim=4,
        initializer="normal",
        mean=0.25,
        std=0.0,
        bounds=(-1.0, 1.0),
        mutation=_constant_mutation(),
    )

    para = Vector.from_config(cfg)

    assert np.array_equal(para.vector, np.full(4, 0.25))


def test_vector_config_accepts_explicit_null_init_bounds() -> None:
    cfg = VectorComponentConfig(
        dim=4,
        initializer="uniform",
        bounds=(-0.5, 0.5),
        init_bounds=None,
        mutation=_constant_mutation(),
    )

    para = Vector.from_config(cfg)

    assert para.init_bounds == (-0.5, 0.5)


def test_adaptive_individual_uses_mutation_strength_bounds() -> None:
    np.random.seed(1)
    cfg = VectorComponentConfig(
        dim=4,
        initializer="zero",
        bounds=(10.0, 20.0),
        mutation=MutationConfig(
            strategy=MutationStrategy.ADAPTIVE_INDIVIDUAL,
            probability=1.0,
            min_strength=0.01,
            max_strength=0.05,
        ),
    )
    para = Vector.from_config(cfg)

    para.update_mutation_parameters(generation=1, max_generations=10)

    assert para.evo_params.mutation_strength is not None
    assert 0.01 <= para.evo_params.mutation_strength <= 0.05


def test_adaptive_per_parameter_uses_mutation_strength_bounds() -> None:
    np.random.seed(1)
    cfg = VectorComponentConfig(
        dim=4,
        initializer="zero",
        bounds=(10.0, 20.0),
        mutation=MutationConfig(
            strategy=MutationStrategy.ADAPTIVE_PER_PARAMETER,
            probability=1.0,
            min_strength=0.01,
            max_strength=0.05,
        ),
    )
    para = Vector.from_config(cfg)

    para.update_mutation_parameters(generation=1, max_generations=10)

    strengths = para.evo_params.mutation_strengths
    assert strengths is not None
    assert np.all(strengths >= 0.01)
    assert np.all(strengths <= 0.05)


def test_adaptive_per_parameter_respects_zero_probability() -> None:
    np.random.seed(1)
    cfg = VectorComponentConfig(
        dim=4,
        initializer="adaptive",
        bounds=(-1.0, 1.0),
        init_bounds=(-0.5, 0.5),
        randomize_mutation_strengths=True,
        mutation=MutationConfig(
            strategy=MutationStrategy.ADAPTIVE_PER_PARAMETER,
            probability=0.0,
            min_strength=0.01,
            max_strength=0.05,
        ),
    )
    para = Vector.from_config(cfg)
    before = para.vector.copy()

    para.mutate()

    assert np.array_equal(para.vector, before)
