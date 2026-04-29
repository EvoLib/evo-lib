# SPDX-License-Identifier: MIT
"""User-facing helper for evaluating and visualizing LineFollower agents."""

from typing import Any

from evolib import Indiv
from evolib.evolib_envs.core.env import Action, Observation
from evolib.evolib_envs.core.evaluator import evaluate_episode
from evolib.evolib_envs.envs.line_follower import LineFollowerEnv
from evolib.evolib_envs.envs.line_follower_defaults import (
    DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    DEFAULT_DEBUG_MAX_STEPS,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_WIDTH,
)
from evolib.evolib_envs.renderers.pygame_line_follower import run_debug_episode


class LineFollowerController:
    """Map an EvoLib individual to LineFollower steering actions."""

    def __init__(self, indiv: Indiv, *, module: str = "brain") -> None:
        self.net: Any = indiv.para[module]

    def act(self, observation: Observation) -> Action:
        """Return a clipped steering action in [-1, 1]."""

        output = self.net.calc(observation)
        turn = float(output[0])
        turn = max(-1.0, min(1.0, turn))
        return [turn]


class LineFollowerTask:
    """
    Evaluate and visualize individuals on the LineFollower environment.

    This class is intentionally specific to LineFollower. It keeps example code close to
    the GymEnv style while hiding controller wiring and rendering details.
    """

    def __init__(
        self,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        max_steps: int = DEFAULT_MAX_STEPS,
        seed: int | None = None,
        module: str = "brain",
    ) -> None:
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.seed = seed
        self.module = module

    def make_env(self) -> LineFollowerEnv:
        """Create a fresh LineFollower environment instance."""

        return LineFollowerEnv(
            width=self.width,
            height=self.height,
            max_steps=self.max_steps,
        )

    def make_controller(self, indiv: Indiv) -> LineFollowerController:
        """Create the default LineFollower controller for one individual."""

        return LineFollowerController(indiv, module=self.module)

    def evaluate(self, indiv: Indiv) -> float:
        """Evaluate one individual and return the accumulated reward."""

        env = self.make_env()
        controller = self.make_controller(indiv)

        return evaluate_episode(
            env,
            controller,
            seed=self.seed,
            max_steps=self.max_steps,
        )

    def visualize(
        self,
        indiv: Indiv,
        *,
        generation: int,
        every: int = DEFAULT_DEBUG_EVERY_N_GENERATIONS,
        steps: int = DEFAULT_DEBUG_MAX_STEPS,
        title: str | None = None,
    ) -> None:
        """Render one debug episode for an individual."""

        display_title = title or f"Training Debug - Gen {generation}"

        run_debug_episode(
            self.make_env(),
            self.make_controller(indiv),
            enabled=True,
            generation=generation,
            every=every,
            steps=steps,
            seed=self.seed,
            title=display_title,
        )
