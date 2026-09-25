# SPDX-License-Identifier: MIT
"""
NetVector – Interpreter for flat vectors as feedforward neural networks.

NetVector interprets a flat parameter vector as the weights and biases of a fully
connected feedforward network. It does not contain trainable parameters or evolutionary
logic itself.

Network-shaped Vector modules can be configured with ``structure: net`` and used with
NetVector for forward evaluation.
"""

from typing import Callable

import numpy as np

from evolib.config.schema import FullConfig

ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(0, x),
    "linear": lambda x: x,
}


class NetVector:
    def __init__(self, dim: list[int], activation: str = "tanh") -> None:
        if not isinstance(dim, list) or not all(isinstance(d, int) for d in dim):
            raise ValueError(f"dim must be list[int], but got: {dim} ({type(dim)})")
        if len(dim) < 2:
            raise ValueError("dim must include at least input and output layers")

        self.dim = dim
        self.n_layers = len(dim) - 1
        self.activation_fn = ACTIVATIONS[activation]

        self.weight_shapes = [(dim[i + 1], dim[i]) for i in range(self.n_layers)]
        self.bias_shapes = [(dim[i + 1],) for i in range(self.n_layers)]
        self.n_parameters = sum(
            np.prod(s) for s in self.weight_shapes + self.bias_shapes
        )

    def forward(self, x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Evaluates the network on input x using the flat parameter vector.

        Activation is applied after all but the last layer.

        Args:
            x (np.ndarray): Input vector (shape: [input_dim] or [batch, input_dim])
            vector (np.ndarray): Flat parameter vector with correct dimension

        Returns:
            np.ndarray: Output of the network
        """
        weights, biases = self._unpack_parameters(vector)
        h = x
        for i in range(self.n_layers):
            h = weights[i] @ h + biases[i]
            if i < self.n_layers - 1:
                h = self.activation_fn(h)
        return h

    def _unpack_parameters(
        self, vector: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Decomposes the flat vector into per-layer weights and biases.

        Args:
            vector (np.ndarray): Flat parameter vector

        Returns:
            Tuple of two lists: weights and biases
        """
        i = 0
        weights, biases = [], []
        for w_shape, b_shape in zip(self.weight_shapes, self.bias_shapes):
            w_size = int(np.prod(w_shape))
            b_size = int(np.prod(b_shape))
            weights.append(vector[i : i + w_size].reshape(w_shape))
            i += w_size
            biases.append(vector[i : i + b_size].reshape(b_shape))
            i += b_size
        return weights, biases

    @classmethod
    def from_config(cls, cfg: FullConfig, module: str) -> "NetVector":
        mod = cfg.modules[module]
        if not isinstance(mod.dim, list):
            raise TypeError(f"Module '{module}': expected dim as list[int]")
        return cls(dim=mod.dim, activation=mod.activation or "tanh")
