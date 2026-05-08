# SPDX-License-Identifier: MIT
"""Registry for recreating environment tasks from checkpoints."""

from collections.abc import Callable

from evolib.evolib_envs.core.checkpoint import EnvCheckpoint
from evolib.evolib_envs.core.task import Task

TaskLoader = Callable[[EnvCheckpoint], Task]

_TASK_LOADERS: dict[str, TaskLoader] = {}


def register_task_loader(env_name: str, loader: TaskLoader) -> None:
    """Register a task loader for an environment name."""

    if env_name in _TASK_LOADERS:
        return

    _TASK_LOADERS[env_name] = loader


def load_task(checkpoint: EnvCheckpoint) -> Task:
    """Create the matching task for a checkpoint."""

    env_name = checkpoint.env.name

    try:
        loader = _TASK_LOADERS[env_name]
    except KeyError as exc:
        registered_names = ", ".join(sorted(_TASK_LOADERS)) or "none"
        raise ValueError(
            f"No task loader registered for environment {env_name!r}. "
            f"Registered environments: {registered_names}."
        ) from exc

    return loader(checkpoint)


def registered_task_names() -> list[str]:
    """Return all registered task names."""

    return sorted(_TASK_LOADERS)
