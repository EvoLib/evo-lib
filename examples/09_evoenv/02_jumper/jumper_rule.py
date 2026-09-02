# SPDX-License-Identifier: MIT
"""Run a simple sensor-based rule controller on JumperEnv."""

from evoenv.core.controller import CallbackController
from evoenv.core.env import Action, Observation
from evoenv.envs.jumper_defaults import DEFAULT_FPS
from evoenv.envs.jumper_task import JumperTask
from evoenv.renderers.pygame_common import PygameWindow
from evoenv.renderers.pygame_jumper import draw_env

TASK_CONFIG_PATH = "task.yaml"
FPS = DEFAULT_FPS


def jumper_rule(observation: Observation) -> Action:
    """Jump when the sensor reports a close obstacle."""
    sensor_value = observation[0]
    normalized_obstacle_height = observation[1]

    should_jump = sensor_value >= 0.75
    jump_strength = 0.75 + 0.35 * normalized_obstacle_height

    if should_jump:
        return [1.0, jump_strength]

    return [0.0, 0.0]


def main() -> None:
    """Run the rule-based Jumper demo."""
    task = JumperTask.from_yaml(TASK_CONFIG_PATH)
    env = task.make_env()
    controller = CallbackController(jumper_rule)

    observation = env.reset()
    total_reward = 0.0

    window = PygameWindow(
        (env.width, env.height),
        caption="EvoEnv - Jumper Rule",
        fps=FPS,
    )

    while window.running:
        if window.process_events():
            observation = env.reset()
            total_reward = 0.0

        if not window.running:
            break

        action = controller.act(observation)
        observation, reward, done, _info = env.step(action)
        total_reward += reward

        if done:
            print(f"Reward: {total_reward:.2f}")
            observation = env.reset()
            total_reward = 0.0

        draw_env(
            window.screen,
            env,
            total_reward,
            window.font,
            title="Sensor-rule Jumper",
        )
        window.update()

    window.close()


if __name__ == "__main__":
    main()
