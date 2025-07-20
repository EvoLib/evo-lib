# Comparison Studies – Evolutionary Strategy Components

This folder contains a set of controlled experiments that illustrate how different components of an evolutionary algorithm influence optimization behavior. Each script focuses on a single variation, making it ideal for teaching and method evaluation.

---

## 📘 01 – Logging & History Inspection

- **File:** `01_history.py`
- **Goal:** Demonstrates how to log and inspect fitness statistics across generations.
- **Features:** Prints best, mean, and std per generation using `history_logger`.

---

## 📘 02 – Plotting

- **File:** `02_plotting.py`
- **Goal:** Introduces how to visualize evolutionary progress using built-in plotting utilities.
- **Plot Output:** `./figures/02_plotting.png`

---

## 📘 03 – Compare Mutation Strengths

- **File:** `03_compare_runs.py`
- **Goal:** Runs two configurations with different mutation strengths to compare their effects.
- **Focus:** Demonstrates convergence speed and stability.
- **Plot Output:** `./figures/02_Compare_Runs.png`

---

## 📘 04 – Mutation Rate Decay vs. Static

- **File:** `04_exponential_decay_vs_static.py`
- **Goal:** Compares constant vs. exponentially decaying mutation rates.
- **Benchmark:** 4D Rosenbrock.
- **Plot Output:** `./figures/04_exponential_decay.png`

---

## 📘 05 – Adaptive Global Mutation vs. Static

- **File:** `05_adaptive_global_vs_static.py`
- **Goal:** Global mutation strength that adapts over time vs. fixed value.
- **Mechanism:** Based on population-wide feedback.
- **Plot Output:** `./figures/05_adaptive_global.png`

---

## 📘 06 – Adaptive Individual Mutation vs. Static

- **File:** `06_adaptive_individual_vs_static.py`
- **Goal:** Per-individual mutation rates (`σ_i`) vs. static global σ.
- **Mechanism:** Self-adaptive, using τ.
- **Plot Output:** `./figures/06_adaptive_individual_vs_static.png`

---

## 📘 07 – Comparison of Selection Strategies

- **File:** `07_selection_comparison.py`
- **Goal:** Compare multiple parent selection methods.
- **Strategies:** Tournament, rank-linear, rank-exp, roulette, SUS, truncation, Boltzmann, random.
- **Benchmark:** Rastrigin.
- **Plot Output:** `./figures/07_selection_comparison.png`

---

## 📘 08 – Selection Pressure via num_parents

- **File:** `08_selection_pressure.py`
- **Goal:** Varies `num_parents` to show impact on convergence.
- **Selection Method:** Fixed (e.g. rank-linear).
- **Plot Output:** `./figures/08_selection_pressure.png`

---

## 📘 09 – Selection vs. Mutation Pressure

- **File:** `09_selection_vs_mutation_pressure.py`
- **Goal:** Explores how selection pressure and mutation strength interact.
- **Variants:** 4 combinations of parent count and mutation rate.
- **Plot Output:** `./figures/09_selection_vs_mutation.png`

---

## 📘 10 – Stochastic vs. Deterministic Selection

- **File:** `10_selection_stochastic_vs_deterministic.py`
- **Goal:** Compares roulette (stochastic), tournament (semi), and truncation (deterministic).
- **Focus:** Trade-off between exploration and exploitation.
- **Plot Output:** `./figures/10_selection_stochastic_vs_deterministic.png`

---

## 📘 11 – Comparison of Crossover Operators

- **File:** `11_crossover_comparison.py`
- **Goal:** Shows the effect of different crossover methods under mutation-free conditions.
- **Operators:** BLX, arithmetic, SBX, intermediate, heuristic, differential.
- **Plot Output:** `./figures/11_crossover_comparison.png`

---

## 🧪 Usage

Each script can be run directly. YAML configuration files must exist in the appropriate subfolder (e.g., `./11_configs/`).

```bash
python 07_selection_comparison.py
python 11_crossover_comparison.py
```

The output will include printed logs and saved plots for direct comparison.

---

## 🧭 Didactic Purpose

| Category     | Focus                          |
|--------------|--------------------------------|
| Mutation     | Strength, adaptation, decay    |
| Selection    | Strategy type, pressure        |
| Crossover    | Operator comparison            |
| Infrastructure | Logging, plotting, history   |

Each example isolates one core mechanism of evolutionary search, enabling focused learning and analysis.
