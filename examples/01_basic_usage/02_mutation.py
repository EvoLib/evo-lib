"""
Example 01-02 — Single-Individual Mutation.

This example demonstrates:
- How to create a population using configuration files.
- How to apply mutation and inspect parameter changes
"""

from evolib import Population  # full name; alias Pop is also available


def main() -> None:
    # Load example configuration for the population
    # Uses the mutation strategy defined in population.yaml
    pop = Population(config_path="population.yaml")

    # Create a single individual
    indiv = pop.create_indiv()

    # Show parameter before mutation
    print(f"Before mutation: {indiv.para.get_status()}")

    # Apply mutation
    indiv.mutate()

    # Show parameter after mutation
    print(f"After mutation:  {indiv.para.get_status()}")


if __name__ == "__main__":
    main()
