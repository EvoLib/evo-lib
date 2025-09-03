## 01_basic_usage/ — Getting Started

This folder provides introductory examples to help you understand the basic usage of EvoLib.  
They cover the essential building blocks: creating populations, mutating individuals, and defining fitness functions.

## Learning Goals
- Load a population from a YAML config.
- Inspect individuals and population status.
- Define and use a custom fitness function in a run
- Understand manual mutation vs. automated runs.

## Prerequisites
None. This is the ideal entry point for new users.

## Expected Output

Running these scripts will print the population status to the console:

> **Note:** Due to random initialization and stochastic operators (mutation, crossover, selection),
> the exact numerical values in your output will vary between runs.
> What matters is the overall pattern: population statistics are printed generation by generation,
> and fitness values should gradually improve as evolution progresses.

- `01_getting_started.py`: shows the basic structure of a population loaded from a YAML config.
    ```
    $ python 01_getting_started.py
    Population: Gen:   0 Fit: 0.00000000
    Best Indiv age: 0
    Max Generation: 0
    Number of Indivs: 10
    Number of Elites: 0
    Population fitness: 0.000
    Worst Indiv: 0.000
    ```

- `02_mutation.py`: shows parameter values of a single individual before and after manual mutation.
    ```
    $ python 02_mutation.py 
    Before mutation: {'test-vector': 'Vector=[-0.419, 0.37, 0.944, 0.284] | Global mutation_strength=0.0100'}
    After mutation:  {'test-vector': 'Vector=[-0.405, 0.368, 0.932, 0.284] | Global mutation_strength=0.0100'}
    ```

- `03_population_mutation.py`: shows how the parameters of all individuals in the population change after mutation.
    ```
    $ python 03_population_mutation.py
    Before mutation:
      Indiv 0: {'test-vector': 'Vector=[-0.495, 0.138, 0.069, -0.931] | Global mutation_strength=0.0100'}
      Indiv 1: {'test-vector': 'Vector=[0.962, -0.135, -0.303, -0.281] | Global mutation_strength=0.0100'}
    [...]
    After mutation:
      Indiv 0: {'test-vector': 'Vector=[-0.482, 0.147, 0.08, -0.938] | Global mutation_strength=0.0100'}
      Indiv 1: {'test-vector': 'Vector=[0.977, -0.121, -0.315, -0.286] | Global mutation_strength=0.0100'}
    [...]
    ```

- `04_fitness.py`: prints generation-by-generation status updates, including best and average fitness, as the population evolves through the optimization loop (`resume_or_create` + `pop.run()`). Over time, fitness values should improve as evolution progresses.
    ```
    $ python 04_fitness.py 
    start: strategy=EvolutionStrategy.MU_PLUS_LAMBDA, mu=2, lambda=4, max_gen=10
    Population: Gen:   1 Fit: 0.01941376
    Population: Gen:   2 Fit: 0.01743742
    [...]
    Population: Gen:  10 Fit: 0.00638989
    ```

## See Also
- [`../02_strategies/01_step_by_step_evolution.py`](../02_strategies/01_step_by_step_evolution.py) — reveals the internal mechanics of one generation.
- [`../03_comparisons/01_history.py`](../03_comparisons/01_history.py) — demonstrates logging and exporting history as a DataFrame.

## Files
- `01_getting_started.py`:  
  Minimal example - load a population from a YAML config and print its status.  

- `02_mutation.py`:
  Shows how to create a single individual, apply mutation manually, and inspect parameter changes.<br>
  ⚠️ **Didactic only**: in real experiments, mutation is applied automatically inside the evolution loop.

- `03_population_mutation.py`:
  Applies mutation across all individuals of the initialized population.<br>
  ⚠️ **Illustration only**: in practice, this is handled automatically by `pop.run()` or `run_one_generation()`.

- `04_fitness.py`:  
  Demonstrates how to define and register a **custom fitness function**.  
  Uses `resume_or_create(...)` to create or resume a run and executes the evolution loop with `pop.run()`.  
  This example shows the canonical workflow for actual experiments.


### Notes
These examples are **minimal and didactic**.
They are meant to illustrate concepts step by step:
- How EvoLib represents populations and individuals
- How mutation affects parameters
- How to attach a fitness function and run evolution
For practical experiments, always prefer the higher-level API (`resume_or_create`, `pop.run()`).
