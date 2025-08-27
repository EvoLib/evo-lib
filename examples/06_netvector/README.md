## NetVector Experiments

NetVector encodes a **feedforward neural network** as a flat parameter vector using `Vector` with `structure: net`.  At evaluation time, the vector is **interpreted** by `NetVector` (weights + biases) to perform forward passes.  This gives you a clean, didactic baseline before moving on to EvoNet’s explicit topology evolution.

---

### What you’ll learn here

* How to configure a fixed architecture in YAML and initialize it as a vector
* How to evaluate a `Vector` via `NetVector.forward(...)`
* How to combine modules with `ParaComposite` (e.g., a small controller vector + network)

> All examples use **MSE** as fitness. Plots are shown with Matplotlib.

---

### 01 – Sine Approximation with NetVector

* **File:** `01_netvector_sine_approximation.py`
* **Config:** `configs/01_netvector_sine_approximation.yaml`
* **Goal:** Approximate `y = sin(x)` on `[0, 2π]`.
* **Architecture:** Defined in YAML via `dim: [input, hidden..., output]` and an activation (e.g., `tanh`).
* **Representation:** Parameters live in `Vector` (module name: `nnet`), interpreted by `NetVector` in the fitness function.
* **Fitness:** MSE between predicted values and the analytic sine curve.
* **Visualization:** Line plot of target vs. best individual (no files saved).

**Key idea:** keep the evolved object simple (just one vector), and use `NetVector.forward(x, vector)` when evaluating.

---

### 02 – Modulated Output (Gain)

* **File:** `02_netvector_modulated_output.py`
* **Config:** `configs/02_netvector_modulated_output.yaml`
* **Goal:** Learn a **gain** that scales the network output:
  `ŷ = gain · net(x)`
* **Representation:** `ParaComposite` with two modules:

  * `controller`: a 1D `Vector` holding **gain**
  * `nnet`: a `Vector` interpreted by `NetVector`
* **Fitness:** MSE vs. `sin(x)`.
* **Visualization:** Plot comparing target `sin(x)` and the modulated prediction.

This showcases **composition**: small scalar controls can be evolved jointly with a network.

---

### 03 – Gain and Bias Modulation

* **File:** `03_netvector_gain_and_bias.py`
* **Config:** `configs/03_netvector_gain_and_bias.yaml`
* **Goal:** Learn **gain** and **bias** in addition to the network:
  `ŷ = gain · net(x) + bias`
* **Target:** `f(x) = 0.8 · sin(x) + 0.2`
* **Representation:** `ParaComposite` with:

  * `controller`: 2 scalars (**gain**, **bias**)
  * `nnet`: network vector interpreted by `NetVector`
* **Fitness:** MSE to the scaled & shifted sine.
* **Visualization:** Plot of target vs. best individual with learned gain/bias shown in legend.

This pattern is handy for cases where global scaling/offset improves fitting without making the network larger.

---

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

---


All scripts print the generation status to stdout and show a Matplotlib plot at the end.

---

### When to use NetVector vs. EvoNet

* **NetVector (this folder):** fixed feedforward topology, concise and fast; great for teaching and baselines.
* **EvoNet (see `../07_evonet`):** explicit graph with potential **structural evolution** (add/remove neurons/edges, activation changes, etc.).

Start with NetVector to validate problem framing and mutation settings, then switch to EvoNet when you need topological flexibility.

