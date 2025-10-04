"""
Example: LunarLander with EvoLib

This script demonstrates how to evolve a neural network controller
for the classic **LunarLander-v3** Gymnasium environment.
Each individual’s network is evaluated by running a full episode
and using the negative cumulative reward as fitness (minimization).
At the end of selected generations, the best agent is visualized
and exported as an animated GIF for inspection.
"""

from evolib import GymEnv, Individual, Population, resume_or_create

CONFIG_FILE = "./configs/04_lunarlander.yaml"
FRAME_FOLDER = "04_frames"
MAX_STEPS = 500

# init environment once (can be reused for all individuals)
gym_env = GymEnv("LunarLander-v3", max_steps=MAX_STEPS)


def eval_lunar_fitness(indiv: Individual) -> None:
    """Assign fitness to an individual by running one LunarLander episode."""
    fitness = gym_env.evaluate(indiv, module="brain")
    indiv.fitness = -fitness


def on_generation_end(pop: Population) -> None:
    """Visualize the best individual every 20 generations (and at the end)."""
    if pop.generation_num % 20 == 0 or pop.generation_num == pop.max_generations:
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
        fitness_function=eval_lunar_fitness,
        run_name="04_lunarlander",
    )
    pop.run(verbosity=1, on_generation_end=on_generation_end)
