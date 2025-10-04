"""
Example: Training on the CliffWalking-v1 environment.

This setup illustrates why CliffWalking is considered an
"anti-evolution" benchmark: progress appears rarely and
suboptimal strategies may dominate due to the reward structure.
"""

from evolib import GymEnv, Individual, Population, resume_or_create

CONFIG_FILE = "./configs/02_cliff_walking.yaml"
FRAME_FOLDER = "02_frames"
MAX_STEPS = 100

# Initialize the environment once (can be reused for all individuals)
gym_env = GymEnv("CliffWalking-v1", max_steps=MAX_STEPS)


def eval_fitness(indiv: Individual) -> None:
    """
    Evaluate the fitness of an individual in the CliffWalking-v1 environment.

    The environment returns cumulative rewards (≤ 0).
    To align with EvoLib's convention of minimizing fitness,
    the reward is negated: higher rewards (closer to 0) become
    smaller fitness values and are therefore preferred.
    """
    reward = gym_env.evaluate(indiv, module="brain")
    indiv.fitness = -reward


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
        run_name="CliffWalking-v1",
    )

    pop.run(verbosity=1, on_generation_end=on_generation_end)
