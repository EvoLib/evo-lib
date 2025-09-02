"""
Example 01-01 — Getting Started with EvoLib.

This example demonstrates:
- How to load a population from a YAML config
- How to inspect basic population status
"""

from evolib import Population  # alias Pop is also available


def main() -> None:
    pop = Population(config_path="population.yaml")
    pop.print_status(verbosity=2)


if __name__ == "__main__":
    main()
