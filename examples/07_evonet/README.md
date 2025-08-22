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
  <img src="./01_frames/01_sine_approximation.gif" alt="Sample EvoNet Frame" width="512"/>
</p>

---

### 📗 02 – Tiny Image Approximation with EvoNet

* **File:** `02_image_approximation.py`
* **Goal:** Learn a tiny grayscale image by mapping 2D coordinates `(x, y)` -> intensity `[0, 1]`. Fitness is the **MSE** between predicted and target image.
* **Input Normalization:** Coordinates are normalized to `[-1, 1]`.
* **Target:** Synthetic pattern (ring + horizontal bar) to keep the task interpretable.
* **Architecture:** EvoNet (`2 -> … -> 1`), defined in the YAML config `configs/02_evonet_image_approximation.yaml`.
* **Mutation:** Gaussian mutation on weights & biases; strategy and parameters come from the config.
* **Fitness:** Mean Squared Error (MSE) between predicted and target image.
* **Visualization:** Saves side-by-side target vs. prediction frames whenever the best fitness improves.
* **Output:** `02_frames/gen_XXXX.png` (progress frames).

---

<p align="center">
  <img src="./02_frames/02_image_approximation.gif" alt="Sample EvoNet Frame" width="512"/>
</p>


---

### 📘 03 – Structural Mutation on XOR

* **File:** `03_structural_xor.py`
* **Goal:** Solve the XOR function `f(x1, x2) = x1 ⊕ x2` using a minimal EvoNet that grows via structural mutation
* **Input:** 4 binary combinations `[0,0], [0,1], [1,0], [1,1]`
* **Architecture:** Starts with only `[2, 1]` (input → output), then expands via:

  * `add_neuron`, `add_connection`, `split_connection`
* **Mutation:**

  * Weights: Gaussian mutation (constant)
  * Structure: Evolves topology (connections and neurons)
* **Fitness:** Mean Squared Error (MSE) on all 4 XOR samples
* **Visualization:**

  * Combined output: network structure + predicted XOR outputs
  * Saves frame on improvement, plus final best frame
* **Output:** `03_frames/gen_XXXX.png` (per-gen frame), `final_best.png` (best net)

---

<p align="center">
  <img src="./03_frames/final_best.png" alt="Structural XOR Final" width="512"/>
</p>

---
