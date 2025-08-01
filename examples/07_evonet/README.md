## 🧠 EvoNet Experiments

This section introduces EvoNet — an evolvable neural network representation with explicit topology. It supports evolving both parameters and structure, enabling more flexible and interpretable controllers compared to fixed architectures.

---

### 📘 01 – Sine Approximation with EvoNet

* **File:** `01_evonet_sine_approximation.py`
* **Goal:** Approximate the function `y = sin(x)` in the domain `[0, 2π]`
* **Input Normalization:** Input is scaled to `[-1.0, 1.0]` for better network performance
* **Architecture:** Fixed feedforward EvoNet (`dim: [1, 16, 16, 1]`)
* **Mutation:** Weight and bias mutation (Gaussian)
* **Fitness:** Mean Squared Error (MSE) between predicted and true sine values
* **Visualization:** 
  * Network structure using `net.print_graph(...)`
  * Combined into animated frames for export
* **Output:** `01_frames/gen_XXXX.png` (combined network and fit per generation)

---

<p align="center">
  <img src="./01_frames/01_evonet_sine_approximation.gif" alt="Sample EvoNet Frame" width="512"/>
</p>

---
