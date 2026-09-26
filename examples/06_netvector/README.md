# 06_netvector – Neural Networks as Vectors

These examples use `VectorNet`, a fixed feedforward network whose weights and biases are
stored and evolved as one flat parameter vector. Unlike `EvoNet`, its topology does not
change during evolution.

## Learning Goals

- Configure a fixed feedforward network with `type: vectornet`.
- Evaluate the evolved network directly with `VectorNet.forward(...)`.
- Combine a `VectorNet` with small `Vector` modules in `ParaComposite`.

## Prerequisites

Knowledge from `01_basic_usage` (population setup and fitness functions).

### `01_netvector_sine_approximation.py`

- **Config:** `configs/01_netvector_sine_approximation.yaml`
- **Goal:** Approximate `y = sin(x)` on `[0, 2π]`.
- **Architecture:** Defined by `dim: [input, hidden..., output]` and `activation`.
- **Representation:** `nnet` is a `VectorNet` that owns its parameter vector.
- **Fitness:** MSE between predictions and the sine curve.

The fitness function obtains the evolved `VectorNet` from the individual and calls
`forward(x)` directly.

### `02_netvector_modulated_output.py`

- **Goal:** Learn a gain that scales the network output: `ŷ = gain · net(x)`.
- **Representation:** `controller` is a 1D `Vector`; `nnet` is a `VectorNet`.
- **Fitness:** MSE against `sin(x)`.

This example shows how a small scalar module can be evolved together with a network.

### `03_netvector_gain_and_bias.py`

- **Goal:** Learn gain and bias in addition to the network:
  `ŷ = gain · net(x) + bias`.
- **Target:** `f(x) = 0.8 · sin(x) + 0.2`.
- **Representation:** `controller` contains gain and bias; `nnet` is a `VectorNet`.
- **Fitness:** MSE against the scaled and shifted sine curve.

### Minimal YAML pattern

```yaml
modules:
  nnet:
    type: vectornet
    dim: [1, 8, 1]
    activation: tanh
    initializer: normal
    bounds: [-1.0, 1.0]
    mutation:
      strategy: constant
      probability: 0.3
      strength: 0.05
```
