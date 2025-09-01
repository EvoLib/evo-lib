"""
Example 01-03 — Population Mutation.

This example demonstrates:
- How to access the individuals already initialized by Pop
- How to apply mutation to all individuals
- How to compare their state before and after mutation

Note:
    Here mutation is applied manually for demonstration purposes.
    In normal use cases, mutation (along with crossover, replacement, etc.)
    is handled automatically when calling `pop.run()` or `pop.run_one_generation()`.
"""

from evolib import Population  # alias Pop is also available


def main() -> None:
    pop = Population(config_path="population.yaml")

    print("Before mutation:")
    for i, indiv in enumerate(pop.indivs):
        print(f"  Indiv {i}: {indiv.para.get_status()}")

    for indiv in pop.indivs:
        indiv.mutate()

    print("\nAfter mutation:")
    for i, indiv in enumerate(pop.indivs):
        print(f"  Indiv {i}: {indiv.para.get_status()}")


if __name__ == "__main__":
    main()
