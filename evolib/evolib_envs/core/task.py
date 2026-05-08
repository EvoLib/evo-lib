# SPDX-License-Identifier: MIT
"""Base classes and protocols for environment tasks."""

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from evolib import Indiv
from evolib.evolib_envs.core.controller import Controller
from evolib.evolib_envs.core.env import Env
from evolib.evolib_envs.core.evaluator import evaluate_episode

EnvT = TypeVar("EnvT", bound=Env)
ControllerT = TypeVar("ControllerT", bound=Controller)


class Task(Protocol):
    """Minimal protocol for tasks loaded from checkpoints."""

    def visualize(self, *args: Any, **kwargs: Any) -> Path | None:
        """Visualize an evaluated individual."""
        ...


class BaseTask(ABC, Generic[EnvT, ControllerT]):
    """
    Shared base class for episodic EvoLib environment tasks.

    The base class centralizes evaluation logic and common task metadata. Concrete tasks
    still define environment creation, controller creation, and visualization.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        max_steps: int,
        seed: int | None = None,
        module: str = "brain",
        difficulty: str | StrEnum = "medium",
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.max_steps = int(max_steps)
        self.seed = seed
        self.module = module
        self.difficulty = difficulty

    @abstractmethod
    def make_env(self) -> EnvT:
        """Create a fresh environment instance."""
        ...

    @abstractmethod
    def make_controller(self, indiv: Indiv) -> ControllerT:
        """Create a controller for one individual."""
        ...

    def evaluate(self, indiv: Indiv) -> float:
        """Run one full episode and return the accumulated reward."""

        env = self.make_env()
        controller = self.make_controller(indiv)

        return evaluate_episode(
            env=env,
            controller=controller,
            seed=self.seed,
            max_steps=self.max_steps,
        )

    @abstractmethod
    def visualize(
        self,
        indiv: Indiv,
        *,
        generation: int,
        every: int = 5,
        steps: int,
        title: str | None = None,
        filename: str | Path | None = None,
        gif_fps: int = 60,
        frame_skip: int = 1,
    ) -> Path | None:
        """Visualize one episode and optionally return the written artifact path."""
        ...
