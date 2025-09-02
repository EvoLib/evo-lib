## 01_basic_usage/ — Getting Started

This folder provides introductory examples to help you understand the basic usage of EvoLib.  
They cover the essential building blocks: creating populations, mutating individuals, and defining fitness functions.

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
