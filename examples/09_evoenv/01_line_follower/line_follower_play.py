# SPDX-License-Identifier: MIT
"""Play the LineFollower environment manually."""

import pygame
from evoenv.cli import parse_env_args
from evoenv.core.difficulty import difficulty_task_path
from evoenv.core.env import Action, Observation
from evoenv.envs.line_follower_defaults import DEFAULT_FPS
from evoenv.envs.line_follower_task import LineFollowerTask
from evoenv.renderers.pygame_common import PygameWindow
from evoenv.renderers.pygame_line_follower import draw_env

FPS = DEFAULT_FPS

args = parse_env_args(description="Play a Line Follower agent.")
task = LineFollowerTask.from_yaml(
    difficulty_task_path(args.difficulty),
    difficulty=args.difficulty,
)


class ManualController:
    """
    Manual steering controller.

    Controls:
    - LEFT: steer left
    - RIGHT: steer right
    """

    def __init__(self, turn_strength: float = 1.0) -> None:
        self.turn_strength = turn_strength
        self.turn = 0.0

    def update(self) -> None:
        """Read keyboard state and update steering value."""
        keys = pygame.key.get_pressed()
        self.turn = 0.0

        if keys[pygame.K_LEFT]:
            self.turn -= self.turn_strength

        if keys[pygame.K_RIGHT]:
            self.turn += self.turn_strength

        self.turn = max(-1.0, min(1.0, self.turn))

    def act(self, _observation: Observation) -> Action:
        """Return the current steering action."""
        return [self.turn]


def main() -> None:
    """Run the manual LineFollower demo."""
    env = task.make_env()
    controller = ManualController()

    window = PygameWindow(
        (env.width, env.height),
        caption="LineFollower - Manual",
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

        observation, reward, done, _ = env.step(action)
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
            title="Manual LineFollower",
        )
        window.update()

    window.close()


if __name__ == "__main__":
    main()
