from typing import Callable

import numpy as np

ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(0, x),
    "identity": lambda x: x,
}


class ParaNet:
    """Simple feedforward neural network with 1 hidden layer, controlled by a flat
    parameter vector."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "tanh",
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation_fn = ACTIVATIONS[activation]

        self.weight_shapes = [
            (hidden_dim, input_dim),  # input → hidden
            (output_dim, hidden_dim),  # hidden → output
        ]
        self.bias_shapes = [
            (hidden_dim,),  # bias for hidden layer
            (output_dim,),  # bias for output layer
        ]
        self.n_parameters = sum(
            np.prod(s) for s in self.weight_shapes + self.bias_shapes
        )

    def forward(self, x: np.ndarray, para_vector: np.ndarray) -> np.ndarray:
        """Computes y = f(x) based on a flat parameter vector."""
        w1, b1, w2, b2 = self._unpack_parameters(para_vector)
        h = self.activation_fn(w1 @ x + b1)
        y = self.activation_fn(w2 @ h + b2)
        return y

    def _unpack_parameters(self, para: np.ndarray) -> tuple[np.ndarray, ...]:
        i = 0

        def next_chunk(shape: tuple[int, ...]) -> np.ndarray:
            nonlocal i
            size = int(np.prod(shape))
            chunk = para[i : i + size].reshape(shape)
            i += size
            return chunk

        w1 = next_chunk(self.weight_shapes[0])
        b1 = next_chunk(self.bias_shapes[0])
        w2 = next_chunk(self.weight_shapes[1])
        b2 = next_chunk(self.bias_shapes[1])
        return w1, b1, w2, b2
