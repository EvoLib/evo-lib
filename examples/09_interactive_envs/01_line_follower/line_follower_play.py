import pygame
from evoenv.cli import parse_env_args
from evoenv.core.env import Action, Observation
from evoenv.envs.line_follower import LineFollowerEnv
from evoenv.envs.line_follower_defaults import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_WIDTH,
)
from evoenv.renderers.pygame_line_follower import draw_env

SCREEN_WIDTH = DEFAULT_WIDTH
SCREEN_HEIGHT = DEFAULT_HEIGHT
MAX_STEPS = DEFAULT_MAX_STEPS
FPS = DEFAULT_FPS


args = parse_env_args(description="Play a Line Follower agent.")
difficulty = args.difficulty


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

    env = LineFollowerEnv(
        width=SCREEN_WIDTH,
        height=SCREEN_HEIGHT,
        max_steps=MAX_STEPS,
        difficulty=difficulty,
    )
    controller = ManualController()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("LineFollower - Manual")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    observation = env.reset()
    total_reward = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_r:
                    observation = env.reset()
                    total_reward = 0.0

        controller.update()
        action = controller.act(observation)

        observation, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            print(f"Reward: {total_reward:.2f}")
            observation = env.reset()
            total_reward = 0.0

        draw_env(
            screen,
            env,
            total_reward,
            font,
            title="Manual LineFollower",
        )

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
