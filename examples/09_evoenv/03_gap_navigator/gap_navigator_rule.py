# SPDX-License-Identifier: MIT
"""Run a simple sensor-based rule controller on GapNavigatorEnv."""

from evoenv.cli import parse_env_args
from evoenv.core.controller import CallbackController
from evoenv.core.difficulty import difficulty_task_path
from evoenv.core.env import Action, Observation
from evoenv.envs.gap_navigator_defaults import DEFAULT_FPS
from evoenv.envs.gap_navigator_task import GapNavigatorTask
from evoenv.renderers.pygame_common import PygameWindow
from evoenv.renderers.pygame_gap_navigator import draw_env

args = parse_env_args(description="Run a GapNavigator rule agent.")
task = GapNavigatorTask.from_yaml(difficulty_task_path(args.difficulty))


def gap_navigator_rule(observation: Observation) -> Action:
    """
    Steer away from the side with stronger obstacle sensor activation.

    The rule deliberately uses only sensor values and the agent's own velocity. It does
    not receive the gap center.
    """
    sensor_values = observation[:-2]
    velocity_x = observation[-1]

    if not sensor_values:
        return [0.0]

    midpoint = len(sensor_values) // 2
    left_pressure = sum(sensor_values[:midpoint])
    right_pressure = sum(sensor_values[midpoint:])
    steering = left_pressure - right_pressure

    steering -= velocity_x * 0.35

    return [max(-1.0, min(1.0, steering))]


def main() -> None:
    """Run the sensor-based GapNavigator demo."""
    env = task.make_env()
    controller = CallbackController(gap_navigator_rule)

    window = PygameWindow(
        (env.width, env.height),
        caption="EvoLib Env - GapNavigator Rule",
        fps=DEFAULT_FPS,
    )

    observation = env.reset()
    total_reward = 0.0

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
            title="Sensor-rule GapNavigator",
        )
        window.update()

    window.close()


if __name__ == "__main__":
    main()
