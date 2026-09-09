import numpy as np

from evolib.config.base_component_config import MutationConfig
from evolib.config.vector_component_config import VectorComponentConfig
from evolib.interfaces.enums import MutationStrategy
from evolib.representation.netvector import NetVector
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


def test_vector_from_config_supports_net_structure() -> None:
    layer_dims = [2, 3, 1]
    expected = NetVector(dim=layer_dims, activation="linear")

    cfg = VectorComponentConfig(
        structure="net",
        dim=layer_dims,
        activation="linear",
        initializer="normal",
        bounds=(-1.0, 1.0),
        mean=0.0,
        std=0.5,
        mutation=_constant_mutation(),
    )

    para = Vector.from_config(cfg)

    assert para.dim == expected.n_parameters
    assert para.vector.shape == (expected.n_parameters,)


def test_vector_from_config_does_not_modify_config() -> None:
    cfg = VectorComponentConfig(
        structure="tensor",
        dim=[2, 3],
        initializer="zero",
        bounds=(-1.0, 1.0),
        mutation=_constant_mutation(),
    )
    before = cfg.model_dump()

    para = Vector.from_config(cfg)

    assert para.dim == 6
    assert para.shape == (2, 3)
    assert cfg.model_dump() == before
