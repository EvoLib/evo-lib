"""
Example 01-01 - Getting Started with EvoLib

This example demonstrates:
- How to load configuration files
- How to create and add an individual
- How to display configuration details
"""

from evolib import Pop

# Create a population using the configutation file
pop = Pop(config_path="population.yaml")

# Create and add an individual to the population
indiv = pop.create_indiv()
pop.add_indiv(indiv)

# print population information
pop.print_status(verbosity=10)

# print indiv information
indiv.print_status()
