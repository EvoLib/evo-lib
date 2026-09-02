# SPDX-License-Identifier: MIT
"""Play the Jumper environment manually."""

import pygame
from evoenv.core.env import Action, Observation
from evoenv.envs.jumper_defaults import DEFAULT_FPS
from evoenv.envs.jumper_task import JumperTask
from evoenv.renderers.pygame_common import PygameWindow
from evoenv.renderers.pygame_jumper import draw_env

TASK_CONFIG_PATH = "task.yaml"
FPS = DEFAULT_FPS


class ManualJumperController:
    """
    Manual jump controller.

    Controls:
    - SPACE: jump
    """

    def __init__(self) -> None:
        self.jump = 0.0

    def update(self) -> None:
        """Read keyboard state and update jump value."""
        keys = pygame.key.get_pressed()
        self.jump = 1.0 if keys[pygame.K_SPACE] else 0.0

    def act(self, _observation: Observation) -> Action:
        """Return the current jump action."""
        return [self.jump, 1.0]


def main() -> None:
    """Run the manual Jumper demo."""
    task = JumperTask.from_yaml(TASK_CONFIG_PATH)
    env = task.make_env()
    controller = ManualJumperController()

    window = PygameWindow(
        (env.width, env.height),
        caption="Jumper - Manual",
        fps=FPS,
    )

    observation = env.reset()
    total_reward = 0.0

    while window.running:
        if window.process_events():
            observation = env.reset()
            total_reward = 0.0

        if not window.running:
            break

        controller.update()
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
            title="Manual Jumper",
        )
        window.update()

    window.close()


if __name__ == "__main__":
    main()
