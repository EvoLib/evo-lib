## 🧠 Advanced Topics in Evolutionary Optimization

This section explores advanced scenarios where evolutionary strategies are applied to more challenging or realistic problems. These include multi-objective trade-offs, fitness landscape visualization, and vector-based control tasks.

---

### 📘 01 – Multi-Objective Optimization: Fit vs. Smoothness

* **File:** `04_multiobjective_tradeoff.py`
* **Goal:** Approximate a sine function while balancing accuracy and smoothness
* **Representation:** Support points + linear interpolation
* **Fitness:** Weighted sum:
  $\text{MSE} + \lambda \cdot \text{Smoothness Penalty}$
* **Extras:** Logs both metrics (`extra_metrics`) for Pareto analysis
* **Output:** `04_frames_multiobjective/`

---

<p align="center">
  <img src="./04_frames_multiobjective/04_multiobjective.gif" alt="Sample Plott" width="512"/>
</p>

---

### 📘 05 – Fitness Landscape Exploration

* **File:** `05_fitness_landscape_exploration.py`
* **Goal:** Visualize and analyze the fitness surface of a benchmark function
* **Function:** Ackley (2D)
* **Plot:** Contour map with current best point
* **Use:** To understand optimization dynamics
* **Output:** `05_frames_landscape/`

---

<p align="center">
  <img src="./05_frames_landscape/05_landscape.gif" alt="Sample Plott" width="512"/>
</p>

---

### 📘 06 – Rosenbrock Surface with Optimization Path

* **File:** `06_rosenbrock_surface_path.py`
* **Goal:** Show how an evolutionary strategy navigates the narrow valley of the Rosenbrock function
* **Visualization:** 3D surface with real-time optimization path
* **Output:** `06_frames_rosenbrock/`

---

<p align="center">
  <img src="./06_frames_rosenbrock/06_rosenbrock.gif" alt="Sample Plott" width="512"/>
</p>

---


### 📘 07 – Vector-Based Control (No Neural Net)

* **File:** `07_vector_control.py`
* **Task:** Reach a target using a sequence of velocity vectors
* **Representation:** Flat vector with 2×N dimensions (x/y velocity at each time step)
* **Fitness:** Final distance to goal
* **Output:** `07_frames_vector_control/`

---

<p align="center">
  <img src="./07_frames_vector_control/07_vector_control.gif" alt="Sample Plott" width="512"/>
</p>

---

### 📘 08 – Vector-Based Control with Obstacles

* **File:** `08_vector_control_with_obstacles.py`
* **Goal:** Reach target while avoiding circular obstacles
* **Encoding:** Same as 07
* **Fitness:** Final distance + penalty for obstacle collisions
* **Penalty:** Soft quadratic penalty per contact
* **Output:** `08_frames_vector_obstacles/`

---

<p align="center">
  <img src="./08_frames_vector_obstacles/08_vector_control_obstacles.gif" alt="Sample Plott" width="512"/>
</p>

---

## ▶️ Running Examples

Each script can be executed directly:

```bash
python 01_constrained_optimization.py
python 04_multiobjective_tradeoff.py
python 08_vector_control_with_obstacles.py
...
```

### ▶️ Animations

To generate videos from the saved frames:

```bash
ffmpeg -framerate 10 -i 08_frames_vector_obstacles/gen_%03d.png -c:v libx264 -pix_fmt yuv420p obstacles_run.mp4
```

