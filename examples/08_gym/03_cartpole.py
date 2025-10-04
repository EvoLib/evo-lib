"""
Example: Evolve a controller for the CartPole-v1 environment.

This example demonstrates how an evolutionary algorithm can evolve
a neural controller that balances the pole on a cart.
Each individual is evaluated in the CartPole-v1 environment and
the best individual of each generation is visualized as a GIF.
"""

from evolib import GymEnv, Individual, Population, resume_or_create

CONFIG_FILE = "./configs/03_cartpole.yaml"
FRAME_FOLDER = "03_frames"
MAX_STEPS = 500

# init environment once (can be reused for all individuals)
cartpole_env = GymEnv("CartPole-v1", max_steps=MAX_STEPS)


def eval_fitness(indiv: Individual) -> None:
    """Evaluate one individual by running CartPole and assign fitness."""
    fitness = cartpole_env.evaluate(indiv, module="brain")
    indiv.fitness = -fitness


def on_generation_end(pop: Population) -> None:
    """Save an animated GIF of the current best individual each generation."""
    best = pop.best()
    gif = cartpole_env.visualize(
        best,
        pop.generation_num,
        filename=f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.gif",
    )
    print(f"Saved: {gif}")


if __name__ == "__main__":
    pop = resume_or_create(
        CONFIG_FILE,
        fitness_function=eval_fitness,
        run_name="03_cartpole",
    )

    pop.run(verbosity=1, on_generation_end=on_generation_end)
