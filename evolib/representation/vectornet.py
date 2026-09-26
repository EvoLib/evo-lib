# SPDX-License-Identifier: MIT
"""Fixed-topology neural network backed by an evolvable parameter vector."""

from collections.abc import Callable

import numpy as np

from evolib.representation.vector import Vector

ActivationFunction = Callable[[np.ndarray], np.ndarray]

_ACTIVATIONS: dict[str, ActivationFunction] = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(0, x),
    "linear": lambda x: x,
}


class VectorNet(Vector):
    """
    Fixed-topology feedforward network represented by a flat parameter vector.

    The network owns its weights and biases through ``Vector.vector`` and reuses
    ``Vector`` mutation and crossover behavior. Network topology is fixed
    """

    def __init__(self, layer_dims: list[int], activation: str = "tanh") -> None:
        super().__init__()

        if len(layer_dims) < 2 or not all(
            isinstance(dim, int) and dim > 0 for dim in layer_dims
        ):
            raise ValueError("layer_dims must contain at least two positive integers")

        if activation not in _ACTIVATIONS:
            allowed = ", ".join(sorted(_ACTIVATIONS))
            raise ValueError(
                f"Unknown activation {activation!r}. Allowed activations: {allowed}."
            )

        self.layer_dims = list(layer_dims)
        self.n_layers = len(self.layer_dims) - 1
        self.activation = activation
        self.activation_fn = _ACTIVATIONS[activation]

        self.weight_shapes = [
            (self.layer_dims[i + 1], self.layer_dims[i]) for i in range(self.n_layers)
        ]
        self.bias_shapes = [(self.layer_dims[i + 1],) for i in range(self.n_layers)]
        self.n_parameters = int(
            sum(np.prod(shape) for shape in self.weight_shapes + self.bias_shapes)
        )

        self.dim = self.n_parameters
        self.vector = np.zeros(self.n_parameters, dtype=float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the network using its current parameter vector.

        ``x`` may be a single input vector with shape ``(input_dim,)`` or a batch with
        shape ``(batch_size, input_dim)``. The configured activation is applied after
        each hidden layer; the output layer remains linear.
        """
        inputs = np.asarray(x, dtype=float)
        if inputs.ndim not in (1, 2) or inputs.shape[-1] != self.layer_dims[0]:
            raise ValueError(
                f"Expected input shape ({self.layer_dims[0]},) or "
                f"(batch, {self.layer_dims[0]}), got {inputs.shape}."
            )

        weights, biases = self._unpack_parameters()
        values = inputs

        for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
            values = values @ weight.T + bias
            if layer_idx < self.n_layers - 1:
                values = self.activation_fn(values)

        return values

    def _unpack_parameters(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return the flat parameter vector as per-layer weights and biases."""
        if self.vector.size != self.n_parameters:
            raise ValueError(
                f"Expected {self.n_parameters} parameters, got {self.vector.size}."
            )

        offset = 0
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []

        for weight_shape, bias_shape in zip(self.weight_shapes, self.bias_shapes):
            weight_size = int(np.prod(weight_shape))
            bias_size = int(np.prod(bias_shape))

            weights.append(
                self.vector[offset : offset + weight_size].reshape(weight_shape)
            )
            offset += weight_size

            biases.append(self.vector[offset : offset + bias_size].reshape(bias_shape))
            offset += bias_size

        return weights, biases
