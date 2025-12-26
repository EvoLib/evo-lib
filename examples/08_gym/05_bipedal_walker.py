"""
Example: BipedalWalker

This script demonstrates how to evolve a neural network controller
for the continuous-control **BipedalWalker-v3** Gymnasium environment.
The walker must learn to coordinate its legs and joints to move
forward without falling.
"""

from evolib import GymEnv, Individual, Population, resume_or_create, save_checkpoint

CONFIG_FILE = "./configs/05_bipedal_walker.yaml"
FRAME_FOLDER = "05_frames"
RUN_NAME = "05_BipedalWalker"
MAX_STEPS = 1600


# init environment once (can be reused for all individuals)
gym_env = GymEnv("BipedalWalker-v3", max_steps=MAX_STEPS)


def eval_walker_fitness(indiv: Individual) -> None:
    reward = gym_env.evaluate(indiv, module="brain", episodes=3)
    indiv.fitness = -reward


def checkpoint(pop: Population) -> None:
    save_checkpoint(pop, run_name=RUN_NAME)
    print("Checkpoint saved.")


def on_improvement(pop: Population) -> None:
    best = pop.best()
    gif = gym_env.visualize(
        best,
        pop.generation_num,
        filename=f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.gif",
    )
    print(f"Saved: {gif}")


def on_generation_end(pop: Population) -> None:
    checkpoint(pop)


if __name__ == "__main__":
    pop = resume_or_create(
        CONFIG_FILE,
        fitness_function=eval_walker_fitness,
        run_name=RUN_NAME,
    )

    pop.run(
        verbosity=1, on_generation_end=on_generation_end, on_improvement=on_improvement
    )
