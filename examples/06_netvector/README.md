# 06_netvector – Neural Networks as Vectors

This folder contains examples that approximate functions using a fixed **feedforward neural network** encoded as a flat parameter vector.  
The network is stored in a `Vector` with `structure: net` and **interpreted** by `NetVector` (weights + biases) at evaluation time.  

---

## Learning Goals

* Configure a fixed feedforward network in YAML.  
* Initialize it as a vector and evaluate via `NetVector.forward(...)`.  
* Combine modules with `ParaComposite` (e.g. small scalar controllers + network).  

---

## Prerequisites

* Knowledge from `01_basic_usage` (population setup, fitness definition).  

---

### `01_netvector_sine_approximation.py`

* **Config:** `configs/01_netvector_sine_approximation.yaml`  
* **Goal:** Approximate `y = sin(x)` on `[0, 2π]`.  
* **Architecture:** Defined in YAML via `dim: [input, hidden..., output]` and an activation (e.g. `tanh`).  
* **Representation:** Parameters live in `Vector` (module name: `nnet`), interpreted by `NetVector` in the fitness function.  
* **Fitness:** MSE between predicted values and the analytic sine curve.  

**Key idea:** keep the evolved object simple (just one vector) and call `NetVector.forward(x, vector)` for evaluation.

---

### `02_netvector_modulated_output.py`

* **Config:** `configs/02_netvector_modulated_output.yaml`  
* **Goal:** Learn a **gain** that scales the network output: `ŷ = gain · net(x)`  
* **Representation:** `ParaComposite` with:  
  * `controller`: 1D `Vector` holding the gain  
  * `nnet`: network vector interpreted by `NetVector`  
* **Fitness:** MSE vs. `sin(x)`.  

This demonstrates **composition**: evolving a small scalar jointly with the network.

---

### `03_netvector_gain_and_bias.py`

* **Config:** `configs/03_netvector_gain_and_bias.yaml`  
* **Goal:** Learn **gain** and **bias** in addition to the network: `ŷ = gain · net(x) + bias`  
* **Target:** `f(x) = 0.8 · sin(x) + 0.2`  
* **Representation:** `ParaComposite` with:  
  * `controller`: 2 scalars (**gain**, **bias**)  
  * `nnet`: network vector interpreted by `NetVector`  
* **Fitness:** MSE to the scaled & shifted sine.  

Useful for cases where global scaling/offset improves fitting without making the network larger.


### Minimal YAML pattern

A compact example of how a NetVector-based module is declared (names match the examples):

```yaml
modules:
  nnet:
    type: vector
    structure: net
    dim: [1, 8, 1]        # [input, hidden, output]
    activation: tanh
    initializer: normal_net
    bounds: [-1.0, 1.0]
    mutation:
      strategy: constant
      probability: 0.3
      strength: 0.05
```

> `initializer: normal_net` creates a vector with the correct size for the network
> and fills it from a normal distribution, then clips to `bounds`.

