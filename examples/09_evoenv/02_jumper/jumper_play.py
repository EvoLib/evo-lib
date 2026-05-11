import pygame
from evoenv.cli import parse_env_args
from evoenv.core.env import Action, Observation
from evoenv.envs.jumper import JumperEnv
from evoenv.envs.jumper_defaults import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_WIDTH,
)
from evoenv.renderers.pygame_jumper import draw_env

SCREEN_WIDTH = DEFAULT_WIDTH
SCREEN_HEIGHT = DEFAULT_HEIGHT
MAX_STEPS = DEFAULT_MAX_STEPS
FPS = DEFAULT_FPS

args = parse_env_args(description="Play a Jumper agent.")
difficulty = args.difficulty


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
        return [self.jump, 0.75]


def main() -> None:
    """Run the manual Jumper demo."""
    env = JumperEnv(
        width=SCREEN_WIDTH,
        height=SCREEN_HEIGHT,
        max_steps=MAX_STEPS,
        difficulty=difficulty,
    )

    controller = ManualJumperController()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Jumper - Manual")
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

        draw_env(screen, env, total_reward, font, title="Manual Jumper")
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
