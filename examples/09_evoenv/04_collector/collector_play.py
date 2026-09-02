# SPDX-License-Identifier: MIT
"""Play the Collector environment manually."""

import pygame
from evoenv.core.env import Action, Observation
from evoenv.envs.collector_defaults import DEFAULT_FPS
from evoenv.envs.collector_task import CollectorTask
from evoenv.renderers.pygame_collector import draw_env
from evoenv.renderers.pygame_common import PygameWindow

TASK_CONFIG_PATH = "task.yaml"
FPS = DEFAULT_FPS


class ManualCollectorController:
    """
    Manual movement controller.

    Controls:
    - LEFT/RIGHT: turn
    - UP: move forward
    """

    def __init__(self) -> None:
        self.turn = 0.0
        self.throttle = 0.0

    def update(self) -> None:
        """Read keyboard state and update movement values."""
        keys = pygame.key.get_pressed()

        self.turn = 0.0
        if keys[pygame.K_LEFT]:
            self.turn -= 0.35
        if keys[pygame.K_RIGHT]:
            self.turn += 0.35

        self.throttle = 1.0 if keys[pygame.K_UP] else 0.0

    def act(self, _observation: Observation) -> Action:
        """Return the current movement action."""
        return [self.turn, self.throttle]


def main() -> None:
    """Run the manual Collector demo."""
    task = CollectorTask.from_yaml(TASK_CONFIG_PATH)
    env = task.make_env()
    controller = ManualCollectorController()

    window = PygameWindow(
        (env.width, env.height),
        caption="Collector - Manual",
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
            title="Manual Collector",
        )
        window.update()

    window.close()


if __name__ == "__main__":
    main()
