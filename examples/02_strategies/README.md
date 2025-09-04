## 02_strategies - Evolution Strategies

This folder demonstrates how different **evolution strategies** are applied in EvoLib.  
The focus here is on showing the mechanics of `(μ + λ)`, `(μ , λ)`, and flexible
composition of operators.

⚠️ These examples are for **illustration purposes**.:  
In real experiments you usually configure the strategy in YAML and run via
`pop.run()`. Direct calls to operators (e.g. `evolve_mu_plus_lambda`) are only used here
to reveal the internal steps.

## Learning Goals
- Understand the difference between **(μ + λ)** and **(μ , λ)** strategies.
- Observe how strategies are applied step by step and over multiple generations.
- Learn how to configure **flexible composition** of selection, mutation, crossover,
  and replacement.
- See the transition from detailed scripts to more compact, realistic workflows.

## Prerequisites
- Knowledge from `01_basic_usage` (population setup, mutation, fitness).
- A basic YAML config (e.g. `population.yaml`) defining parent/offspring pools
  and a vector module.

## Files & Expected Output
Running these examples will print either detailed individual values
(parameters + fitness) or population summaries.  
Because initialization and operators are stochastic, numbers will vary between runs.
What matters is the **pattern**:
- `(μ + λ)` retains parents in the next generation.
- `(μ , λ)` replaces parents entirely with offspring.
- With `flexible`, you can see how the operator pipeline evolves the population
according to your chosen components.

#### `01_step_by_step_evolution.py`:

    Manually constructs a single generation loop.  
    ⚠️ For illustration only; in practice, use `pop.run()`.
    ```
    $ python 01_step_by_step_evolution.py
    0) Evaluate parents :
      Indiv 0: [...]
      Indiv 1: [...]

    1) Update parameters:
      Indiv 0: [...]
      Indiv 1: [...]

    2) Reproduction (clone parents -> offspring):
      Indiv 0: [...]
      Indiv 1: [...]

    3) Crossover:
      Indiv 0: [...]
      Indiv 1: [...]

    4) Mutation:
      Indiv 0: [...]
      Indiv 1: [...]

    5) Evaluate offspring:
      Indiv 0: [...]
      Indiv 1: [...]

    6) Replacement:
      Indiv 0: [...]
      Indiv 1: [...]

    Population: Gen:   1 Fit: 0.00026955
    Best Indiv age: 0
    Max Generation: 10
    Number of Indivs: 2
    Number of Elites: 0
    Population fitness: 0.002
    Worst Indiv: 0.004
    ```
      
#### `02_mu_lambda_step.py`:  

    Demonstrates `(μ + λ)` and `(μ , λ)` strategies once, printing parents and offspring.  
    ⚠️ Operators are called directly for clarity.
    ```
    $ python 02_mu_lambda_step.py
    Initial Parents
      Indiv 0: Parameter = [-0.09826022], Fitness = 0.000093
      Indiv 1: Parameter = [0.91075599], Fitness = 0.688031

    After Mu Plus Lambda
      Indiv 0: Parameter = [-0.06251137], Fitness = 0.000015
      Indiv 1: Parameter = [-0.0858429], Fitness = 0.000054

    After Mu Comma Lambda
      Indiv 0: Parameter = [-0.05951417], Fitness = 0.000013
      Indiv 1: Parameter = [-0.06225795], Fitness = 0.000015
    ```

#### `03_mu_lambda.py`:

    Applies `(μ + λ)` repeatedly over several steps. Shows how the population
    evolves across generations.
    ```
    $ python 03_mu_lambda.py
    Initial Parents
      Indiv 0: [...]
      Indiv 1: [...]

    After Mu Plus Lambda - Step 1
      Indiv 0: [...]
      Indiv 1: [...]

    After Mu Plus Lambda - Step 2
      [...]

    After Mu Plus Lambda - Step 3
      [...]
    ```

#### `04_flexible.py`:  
    Shows how to configure `evolution.strategy: flexible` and explicitly combine
    selection, mutation, crossover, and replacement.  
    Highlights EvoLib’s modular design.  
    ⚠️ Loop with `evolve_flexible` is for teaching; in normal runs use `pop.run()`.
    ```
    $ python 04_flexible.py
    Initial Parents
      Indiv 0: [...]
      Indiv 1: [...]
      Indiv 2: [...]

    Population: Gen:   1 Fit: 0.123456
    Population: Gen:   2 Fit: 0.101234
    [...]

    Final Population
      Indiv 0: [...]
      Indiv 1: [...]
      Indiv 2: [...]
    ```

### See Also
- [`../03_comparisons/`](../03_comparisons) — side-by-side comparisons of strategies
  (e.g. exponential decay vs. static mutation, adaptive global vs. static).
- [`../01_basic_usage/`](../01_basic_usage) — introductory examples.


### Notes
These examples are **minimal and for illustration purposes**. For practical experiments, always prefer the higher-level API (`resume_or_create`, `pop.run()`).

