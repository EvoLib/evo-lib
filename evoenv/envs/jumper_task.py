# SPDX-License-Identifier: MIT
"""Helper for evaluating and visualizing Jumper agents."""

from pathlib import Path
from typing import Any

from evoenv.core.checkpoint import EnvCheckpoint
from evoenv.core.env import Action, Observation
from evoenv.core.sensors import RaySensor
from evoenv.core.task import BaseTask
from evoenv.core.task_registry import register_task_loader
from evoenv.envs.jumper import JumperEnv
from evoenv.envs.jumper_defaults import (
    DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    DEFAULT_DEBUG_MAX_STEPS,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_WIDTH,
)
from evoenv.envs.jumper_settings import DEFAULT_JUMPER_SETTINGS, JumperSettings
from evoenv.renderers.pygame_jumper import run_debug_episode

from evolib import Indiv


class JumperController:
    """Map an EvoLib individual to Jumper jump actions."""

    def __init__(self, indiv: Indiv, *, module: str = "brain") -> None:
        self.net: Any = indiv.para[module]

    def act(self, observation: Observation) -> Action:
        """Return clipped jump action values in [0, 1]."""
        output = self.net.calc(observation)

        jump_signal = max(0.0, min(1.0, float(output[0])))
        jump_strength = max(0.0, min(1.0, float(output[1])))

        return [jump_signal, jump_strength]


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
        settings: JumperSettings = DEFAULT_JUMPER_SETTINGS,
        sensor: RaySensor | None = None,
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            seed=seed,
            module=module,
            difficulty="standard",
        )
        self.settings = settings
        self.sensor = sensor

    def make_env(self) -> JumperEnv:
        """Create a fresh Jumper environment instance."""
        return JumperEnv(
            width=self.width,
            height=self.height,
            max_steps=self.max_steps,
            settings=self.settings,
            sensor=self.sensor,
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
    return JumperTask(
        seed=checkpoint.seed,
        **checkpoint.env.params,
    )


def register_jumper_task() -> None:
    """Register the Jumper task loader."""
    register_task_loader("jumper", load_jumper_task)
