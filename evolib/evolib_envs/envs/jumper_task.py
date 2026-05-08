# SPDX-License-Identifier: MIT
"""Helper for evaluating and visualizing Jumper agents."""

from pathlib import Path
from typing import Any

from evolib import Indiv
from evolib.evolib_envs.core.checkpoint import EnvCheckpoint
from evolib.evolib_envs.core.env import Action, Observation
from evolib.evolib_envs.core.task import BaseTask
from evolib.evolib_envs.core.task_registry import register_task_loader
from evolib.evolib_envs.envs.jumper import JumperEnv
from evolib.evolib_envs.envs.jumper_defaults import (
    DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    DEFAULT_DEBUG_MAX_STEPS,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_WIDTH,
)
from evolib.evolib_envs.envs.jumper_settings import JumperDifficulty
from evolib.evolib_envs.renderers.pygame_jumper import run_debug_episode


class JumperController:
    """Map an EvoLib individual to Jumper jump actions."""

    def __init__(self, indiv: Indiv, *, module: str = "brain") -> None:
        self.net: Any = indiv.para[module]

    def act(self, observation: Observation) -> Action:
        """Return a clipped jump action in [0, 1]."""

        output = self.net.calc(observation)
        jump_signal = float(output[0])
        jump_signal = max(0.0, min(1.0, jump_signal))
        jump_force = float(output[1])
        jump_force = max(0.0, min(1.0, jump_force))
        return [jump_signal, jump_force]


class JumperTask(BaseTask[JumperEnv, JumperController]):
    """Evaluate and visualize individuals on the Jumper environment."""

    def __init__(
        self,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        max_steps: int = DEFAULT_MAX_STEPS,
        seed: int | None = None,
        module: str = "brain",
        difficulty: str | JumperDifficulty = JumperDifficulty.MEDIUM,
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            seed=seed,
            module=module,
            difficulty=difficulty,
        )

    def make_env(self) -> JumperEnv:
        """Create a fresh Jumper environment instance."""

        return JumperEnv(
            width=self.width,
            height=self.height,
            max_steps=self.max_steps,
            difficulty=self.difficulty,
        )

    def make_controller(self, indiv: Indiv) -> JumperController:
        """Create the default Jumper controller for one individual."""

        return JumperController(indiv, module=self.module)

    def visualize(
        self,
        indiv: Indiv,
        *,
        generation: int,
        every: int = DEFAULT_DEBUG_EVERY_N_GENERATIONS,
        steps: int = DEFAULT_DEBUG_MAX_STEPS,
        title: str | None = None,
        filename: str | Path | None = None,
        gif_fps: int = DEFAULT_FPS,
        frame_skip: int = 1,
    ) -> Path | None:
        """Render one debug episode for an individual."""

        display_title = title or f"Jumper Training Debug - Gen {generation}"

        return run_debug_episode(
            self.make_env(),
            self.make_controller(indiv),
            enabled=True,
            generation=generation,
            every=every,
            steps=steps,
            seed=self.seed,
            title=display_title,
            filename=filename,
            gif_fps=gif_fps,
            frame_skip=frame_skip,
        )


def load_jumper_task(checkpoint: EnvCheckpoint) -> JumperTask:
    """Create a Jumper task from checkpoint metadata."""

    env = checkpoint.env

    return JumperTask(
        seed=checkpoint.seed,
        difficulty=env.difficulty or JumperDifficulty.MEDIUM,
        **env.params,
    )


def register_jumper_task() -> None:
    """Register the Jumper task loader."""

    register_task_loader("jumper", load_jumper_task)
