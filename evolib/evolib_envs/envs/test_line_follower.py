# SPDX-License-Identifier: MIT
"""
Minimal test for LineFollowerEnv.

Runs a few steps with constant action and prints observations and rewards.
"""

from evolib.evolib_envs.envs.line_follower import LineFollowerEnv


def run_basic_test() -> None:
    env = LineFollowerEnv()

    obs = env.reset(seed=42)
    print(f"Initial observation: {obs}")

    for step in range(10):
        action = [0.5, 0.5]  # go straight

        obs, reward, done, _ = env.step(action)

        print(f"step={step:02d} " f"obs={obs} " f"reward={reward:.3f} " f"done={done}")

        if done:
            print("Episode finished early.")
            break


if __name__ == "__main__":
    run_basic_test()
