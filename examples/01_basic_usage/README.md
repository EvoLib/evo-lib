## 01_basic_usage/ — Getting Started

This folder provides introductory examples to help you understand the basic usage of EvoLib.  
They cover the essential building blocks: creating populations, mutating individuals, and defining fitness functions.

## Learning Goals
- Load a population from a YAML config.
- Inspect individuals and population status.
- Attach a custom fitness function.
- Understand manual mutation vs. automated runs.

## Prerequisites
None. This is the ideal entry point for new users.

## Expected Output

Console output showing population status.
In 04_fitness.py you will see a full run driven by resume_or_create.

## See Also

- [`01_step_by_step_evolution.py - inner mechanics.`](../02_strategies/01_step_by_step_evolution.py)

- [`03_comparisons/01_history.py - logging and data frames.`](../03_comparisons/01_history.py)


### Files
- `01_getting_started.py`:  
  Minimal example - load a population from a YAML config and print its status.  

- `02_mutation.py`:  
  Shows how to create a single individual, apply mutation manually, and inspect parameter changes.  
  This is for demonstration - in real runs, mutation is handled automatically inside the evolution loop.

- `03_population_mutation.py`:  
  Applies mutation across all individuals of the initialized population.  
  Again, this manual approach is only for illustration. Normally, mutation is performed automatically.

- `04_fitness.py`:  
  Demonstrates how to define and register a **custom fitness function**.  
  Uses `resume_or_create(...)` to create or resume a run and executes the evolution loop with `pop.run()`.  
  This example shows the canonical workflow for actual experiments.

### Notes
- These examples are **minimal**. The goal is didactic clarity: understanding how EvoLib organizes populations, mutation, and fitness evaluation.
