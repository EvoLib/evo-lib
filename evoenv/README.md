# EvoEnv

EvoEnv provides small, configurable Pygame-based environments for evolutionary
experiments with EvoLib.

The environments use compact observations and actions, support headless
training, and can be visualized with Pygame when behavior needs to be
inspected.
EvoEnv is intended for experiments where the complete task can be inspected directly
from the code and configuration.

## Scope and Intended Use

EvoEnv is intended for small, controlled evolutionary experiments where the
relationship between observations, actions, rewards, and behavior should remain
easy to inspect.

Typical tasks use compact numerical observations and a small action space. They
may involve:

- steering from local sensor input
- action timing
- obstacle avoidance
- target following
- reward-shaping experiments
- recurrent or memory-dependent behavior
- small changes to controller or sensor structure

EvoEnv is intended for environments created for a specific experimental question
rather than selected as a general benchmark.

The current implementation supports single-agent environments. General batch
execution and shared-world multi-agent environments are not yet implemented.

Raw-pixel observations, large CNNs, large recurrent policies, and
high-performance game simulation are outside the intended scope.

## Built-in environments

| Environment | Main focus |
|---|---|
| Line Follower | Local point sensors and continuous steering |
| Jumper | Obstacle sensing and jump timing |
| Gap Navigator | Directional ray sensors and horizontal navigation |
| Collector | Target following, exploration, and obstacle avoidance |

### Line Follower

A small agent follows a generated path using local point sensors and one
continuous steering action.

The example shows how a compact sensor layout can control continuous movement
and how difficulty can change without changing the controller interface.

### Jumper

A side-scrolling agent must trigger jumps at suitable times to avoid obstacles.

The example focuses on action timing, proximity sensing, and the balance between
survival rewards, jump costs, and collision penalties.

### Gap Navigator

An agent moves horizontally through successive obstacle rows and must align with
open gaps.

The example uses directional ray sensors and demonstrates spatial navigation,
movement penalties, collision handling, and passage rewards.

### Collector

An agent moves through a two-dimensional arena, follows food targets, and avoids
walls and rectangular obstacles.

The example combines continuous steering and throttle with target-relative
observations, obstacle rays, exploration rewards, and distance-based reward
shaping.

Runnable examples and preview GIFs are available in the
[EvoEnv example overview](https://github.com/EvoLib/evo-lib/tree/main/examples/09_evoenv).

## How EvoEnv is structured

EvoEnv separates simulation, control, EvoLib integration, and visualization.

| Component | Responsibility |
|---|---|
| `Env` | State, observations, rewards, and episode termination |
| `Controller` | Mapping observations to actions |
| `Task` | Connecting EvoLib individuals with an environment |
| `Renderer` | Visualizing environment state with Pygame |
| `Checkpoint` | Storing an individual together with task metadata |
| `Task registry` | Reconstructing a task from checkpoint metadata |


### Task

The task configuration controls the environment layout, reward terms, sensor
geometry, and environment-specific options.

```yaml
env:
  width: 600
  height: 600
  max_steps: 1500
  player_y_offset: 55

  row_speed: 4.0
  row_interval: 62
  obstacle_height: 28

  min_gap_width: 135
  max_gap_width: 195
  edge_margin: 35

  player_speed: 5.6
  terminate_on_collision: false

reward:
  pass_reward: 0.0  # pass_reward is only used when terminate_on_collision is enabled.

  gap_alignment_reward: 0.040
  movement_penalty: 0.014
  collision_penalty: 6.5
  near_wall_penalty: 0.040

fitness:
  sensor_count_penalty: 0.0
  sensor_length_penalty: 1.0
  sensor_length_scale: 500.0

sensors:
  max_sensors: 6

  max_length: 500.0
  min_active_length: 0.0

  min_angle: -1.57079632679
  max_angle: 1.57079632679
```


### Renderer

A renderer displays the current environment state. Rendering is separate from
simulation, so the same environment can be used for headless training and
for interactive inspection.

## Example scripts

Each EvoEnv example provides the same set of scripts:

```text
play -> rule -> train -> watch
```

| Script | Purpose |
|---|---|
| `*_play.py` | Inspect the environment through manual control |
| `*_rule.py` | Run a simple hand-written baseline |
| `*_train.py` | Evolve a controller without continuous rendering |
| `*_watch.py` | Load and visualize a saved checkpoint |

A useful sequence when exploring an environment is:

1. Run `play` to understand the controls and task.
2. Run `rule` to verify that the task is solvable with simple behavior.
3. Run `train` to evolve a controller.
4. Run `watch` to inspect the saved result.

For example:

```bash
cd examples/09_evoenv/04_collector

python collector_play.py
python collector_rule.py
python collector_train.py
python collector_watch.py collector.pkl
```

Some training scripts provide a `--debug` option that periodically renders the
current best individual:

```bash
python collector_train.py --debug
```

## Configuration

EvoEnv examples normally use two kinds of YAML configuration.

### EvoLib configuration

The EvoLib configuration describes the evolutionary process, including the
population, parameter representation, selection, mutation, and stopping
conditions.

A typical example file is named `config.yaml` or uses difficulty-specific names
such as `config_easy.yaml`.

### Task configuration

The task configuration describes the environment itself. It commonly separates
simulation, reward, sensor, and optional task-specific settings:

```yaml
env:
  # arena, movement, object, and episode settings

reward:
  # rewards and penalties

sensor:
  # sensor geometry and ranges

exploration:
  # optional environment-specific settings
```

The configuration is validated when it is loaded. Invalid combinations should
produce an explicit error instead of silently changing the task.

## Headless training and visualization

Training and evaluation do not require an active display and can therefore run
on remote servers and other headless systems.

EvoEnv may use Pygame for geometry and simulation helpers, but a Pygame window
is only required for interactive play and visualization. 
Headless evaluation and rendered runs use the same environment logic and
therefore produce the same rewards and simulation behavior.

## Checkpoints

A checkpoint stores the evolved individual together with the information needed
to reconstruct the task.

This normally includes:

- the environment name
- the complete task configuration
- the controller module name
- the random seed

The corresponding `watch` script loads the checkpoint, reconstructs the task,
and visualizes the stored individual.

## Reproducibility

Calling `reset(seed=...)` with the same seed should reproduce the complete
initial episode state, including object placement and task-specific random
choices.

Training tasks use the configured EvoLib random seed where applicable. The seed
and task configuration are stored in checkpoints so that trained behavior can
be inspected under the same conditions.

## Building a custom environment

A custom environment normally consists of three required parts and an optional
renderer:

1. A headless environment implementing `reset()` and `step()`.
2. A controller that maps observations to actions.
3. A task that connects the environment to EvoLib individuals.
4. An optional Pygame renderer for inspection.

A custom local environment does not need the full example structure used by the
built-in environments.

