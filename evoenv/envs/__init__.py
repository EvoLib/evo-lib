# SPDX-License-Identifier: MIT

from evoenv.envs.jumper_task import register_jumper_task
from evoenv.envs.line_follower_task import register_line_follower_task


def register_builtin_tasks() -> None:
    """Register all built-in environment tasks."""

    register_line_follower_task()
    register_jumper_task()
