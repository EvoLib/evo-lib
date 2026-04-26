# SPDX-License-Identifier: MIT
import math
import random

from evolib.evolib_envs.core.env import Action, Env, Observation, StepResult


class LineFollowerEnv(Env):
    """
    Minimal line follower environment without rendering.

    The agent controls a simple differential drive robot trying to stay on a line.
    """

    observation_size = 2
    action_size = 2

    def __init__(self) -> None:
        self.x = 0.0
        self.angle = 0.0
        self.step_count = 0

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            random.seed(seed)

        self.x = random.uniform(-0.5, 0.5)
        self.angle = random.uniform(-0.2, 0.2)
        self.step_count = 0

        return self._observe()

    def step(self, action: Action) -> StepResult:
        left, right = action

        # simple differential steering
        turn = (right - left) * 0.1
        self.angle += turn

        # forward movement
        self.x += math.sin(self.angle) * 0.05

        self.step_count += 1

        obs = self._observe()
        reward = -abs(self.x)

        done = abs(self.x) > 2.0 or self.step_count > 500

        return obs, reward, done, {}

    def _observe(self) -> Observation:
        # two simple sensors left/right of center
        left_sensor = max(0.0, 1.0 - abs(self.x - 0.2))
        right_sensor = max(0.0, 1.0 - abs(self.x + 0.2))
        return [left_sensor, right_sensor]
