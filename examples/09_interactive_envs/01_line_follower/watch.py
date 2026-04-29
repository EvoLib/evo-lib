# SPDX-License-Identifier: MIT
"""Watch a trained LineFollower individual with Pygame visualization."""

from evolib import load_best_indiv
from evolib.evolib_envs.envs.line_follower_task import LineFollowerTask

RUN_NAME = "line_follower"


indiv = load_best_indiv(run_name=RUN_NAME)
if indiv is None:
    raise RuntimeError(f"No best individual loaded for run_name={RUN_NAME!r}")

line_task = LineFollowerTask()

line_task.visualize(
    indiv,
    generation=1,
    every=1,
    title="Evolved LineFollower",
)
