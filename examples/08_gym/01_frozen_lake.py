"""
Example: Training on the FrozenLake-v1 environment.

FrozenLake is a task where the agent must reach the goal while
avoiding holes on a slippery surface.
This example demonstrates how stochastic transitions make
evolutionary training noisy and challenging.
"""

from evolib import GymEnv, Individual, Population, resume_or_create

CONFIG_FILE = "./configs/01_frozen_lake.yaml"
FRAME_FOLDER = "01_frames"
MAX_STEPS = 150

# init environment once (can be reused for all individuals)
gym_env = GymEnv("FrozenLake-v1", max_steps=MAX_STEPS, is_slippery=True, map_name="8x8")


def eval_fitness(indiv: Individual) -> None:
    """
    Assign fitness to an individual by running several FrozenLake episodes.

    The agent controls discrete moves (left, down, right, up) on an 8x8 grid. Fitness is
    defined as the negative average reward across 5 episodes, encouraging robust
    policies that consistently reach the goal while avoiding holes.
    """

    fitness = gym_env.evaluate(indiv, module="brain", episodes=5)
    indiv.fitness = -fitness


def on_generation_end(pop: Population) -> None:
    """Visualize the best individual of the current generation as a GIF."""

    best = pop.best()
    gif = gym_env.visualize(
        best,
        pop.generation_num,
        filename=f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.gif",
    )
    print(f"Saved: {gif}")


if __name__ == "__main__":
    pop = resume_or_create(
        CONFIG_FILE,
        fitness_function=eval_fitness,
        run_name="FrozenLake-v1",
    )

    pop.run(verbosity=1, on_generation_end=on_generation_end)
