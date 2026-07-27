# 04_collector – Target Following, Obstacle Avoidance, and Reward Shaping

Collector is a small 2D food-collection environment for evolved controllers.

The agent moves freely through an arena, steers toward the nearest food item,
avoids walls and rectangular obstacles, and collects as much food as possible
within one episode.

The example combines four concepts:

- continuous steering and throttle control
- compact target-direction observations
- local obstacle sensing
- sparse task rewards with additional reward shaping

For a general overview of the Pygame-based EvoEnv examples, see the main README
in `examples/09_evoenv/`.

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/09_evoenv/04_collector/frames/collector.gif" alt="Collector sample" width="512"/>
</p>

---

## Goal

The actual task objective is simple:

> Collect as many food items as possible within one episode.

A useful controller should therefore:

- turn toward the nearest food item
- keep moving instead of remaining stationary
- avoid walls and rectangular obstacles
- collect food repeatedly across the episode
- avoid wasting time in small loops

Collector is not a blind exploration task. The observation contains the
direction and distance to the nearest food item. The controller must combine
this target cue with local obstacle information and produce suitable movement
actions.

---

## Example Workflow

The example follows the standard EvoEnv workflow:

```text
play -> rule -> train -> watch
```

| File | Purpose |
|---|---|
| `collector_play.py` | Explore the environment with manual controls. |
| `collector_rule.py` | Run a simple hand-written baseline controller. |
| `collector_train.py` | Evolve an EvoNet controller. |
| `collector_watch.py` | Load and visualize a saved checkpoint. |

---

## Configuration

Collector uses two configuration files:

| File | Purpose |
|---|---|
| `task.yaml` | Environment, reward, sensor, and exploration settings. |
| `config.yaml` | EvoLib population, evolution, and EvoNet settings. |

Collector intentionally provides one standard configuration instead of
`easy`, `medium`, and `hard` presets.

---

## Observation Space

The environment returns six values.

Collector intentionally uses exactly three obstacle rays in left, center, and
right order. The center ray detects obstacles directly ahead, while the side
rays help the controller choose an avoidance direction.

| Index | Value | Range | Meaning |
|---:|---|---|---|
| 0 | `target_angle_sin` | `[-1.0, 1.0]` | Sine of the relative angle to the nearest food item. |
| 1 | `target_angle_cos` | `[-1.0, 1.0]` | Cosine of the relative angle to the nearest food item. |
| 2 | `target_distance` | `[0.0, 1.0]` | Distance to the nearest food item, normalized by the arena diagonal. |
| 3 | `left_sensor` | `[0.0, 1.0]` | Proximity value of the left obstacle ray. |
| 4 | `center_sensor` | `[0.0, 1.0]` | Proximity value of the forward obstacle ray. |
| 5 | `right_sensor` | `[0.0, 1.0]` | Proximity value of the right obstacle ray. |

Obstacle values close to `0.0` mean that no obstacle was detected nearby.
Values close to `1.0` indicate a close wall or obstacle.

The default EvoNet input dimension is therefore `6`.

---

## Action Space

The controller returns two continuous values.

| Index | Value | Range | Meaning |
|---:|---|---|---|
| 0 | `turn` | `[-1.0, 1.0]` | Requested heading change. |
| 1 | `throttle` | `[0.0, 1.0]` | Requested forward speed. |

Examples:

```python
# Turn right while moving forward.
[0.5, 1.0]

# Remain stationary.
[0.0, 0.0]
```

The environment clips both action values to their valid ranges.

---

## Controller Network

The default training configuration uses a small feed-forward EvoNet:

```yaml
modules:
  brain:
    type: evonet
    dim: [6, 6, 2]
    activation: [linear, tanh, tanh]
```

| Layer | Size | Meaning |
|---|---:|---|
| input | 6 | Target cue and three obstacle sensors. |
| hidden | 6 | Small nonlinear controller layer. |
| output | 2 | Turn and throttle. |

The output layer uses `tanh`. `CollectorController` uses the first output as the
turn action and maps the second output from `[-1.0, 1.0]` to throttle in
`[0.0, 1.0]`.

---

## Reward Design

### Sparse task reward

The direct task reward is the food reward:

```text
food reward = collected food items × food_reward
```

With only this reward, early controllers often receive identical results.
A controller that remains stationary, a controller that moves in the wrong
direction, and a controller that almost reaches food all receive no food reward
as long as none of them actually collects an item.

This produces a sparse fitness signal. Evolution has little information for
distinguishing partially useful behavior before reliable collection emerges.

### Reward shaping

Collector adds intermediate feedback so that partial progress becomes visible
to selection.

The reward for one simulation step is:

```text
step reward =
    food reward
    + normalized distance progress
    + exploration bonus
    - collision penalty
    - step penalty
    - turn penalty
```

The components have different roles:

| Component | Role | Meaning |
|---|---|---|
| `food_reward` | Task objective | Reward for actually collecting food. |
| `distance_reward` | Shaping signal | Rewards movement toward the nearest food and penalizes movement away from it. |
| `exploration_reward` | Shaping signal | Rewards entering a previously unvisited grid cell. |
| `collision_penalty` | Behavioral preference | Discourages contact with walls and obstacles. |
| `step_penalty` | Behavioral preference | Favors faster solutions and discourages inactivity. |
| `turn_penalty` | Behavioral preference | Discourages unnecessary strong steering. |

The distinction is important: `food_reward` represents what the agent should
ultimately achieve. The remaining terms provide intermediate feedback or define
which kinds of solutions are preferred.

### Distance progress

The distance term is based on the change in distance to the nearest food item:

```python
progress = (previous_distance - current_distance) / base_speed
reward += progress * distance_reward
```

Dividing by `base_speed` makes the value approximately independent of the
configured movement speed.

At full speed, one step directly toward the target produces approximately:

```text
progress        = +1.0
distance reward = +1.0 × 0.2 = +0.200
```

One full-speed step directly away from the target produces approximately:

```text
progress        = -1.0
distance reward = -1.0 × 0.2 = -0.200
```

This allows evolution to distinguish useful target approach before the
controller reliably reaches and collects food.

The default distance term is substantial rather than merely cosmetic. For an
unchanged target initially 200 pixels away and collected at a distance of about
12 pixels, the cumulative contribution from direct approach is approximately:

```text
(200 - 12) / 2.5 × 0.2 = 15.04
```

This can exceed the `food_reward` of `10.0`. The configured distance reward is
therefore an important part of the optimized objective, not just a small tie
breaker between otherwise equal controllers.

### Exploration bonus

The arena is divided into grid cells. Entering a cell for the first time during
an episode adds `exploration_reward`.

The bonus discourages stationary behavior, tight circles, and repeated movement
through the same small area. It is intentionally much smaller than the food and
distance rewards so that exploration alone is not the main objective.

### Penalties

The remaining terms add behavioral preferences:

- `collision_penalty` makes repeated contact with walls or obstacles expensive.
- `step_penalty` favors collecting food in fewer simulation steps.
- `turn_penalty` slightly favors smooth and direct movement over permanent
  maximum steering.

These penalties are not required to define food collection itself. They shape
which successful behavior evolution prefers.

### Reward shaping changes the optimized objective

Reward shaping is not neutral. Evolution optimizes the complete accumulated
reward, not only the number of collected food items.

With the default configuration, the controller is effectively optimized to:

> Collect food, approach the nearest target, visit new areas, avoid collisions,
> finish efficiently, and avoid unnecessary steering.

Shaping values must therefore be interpreted as part of the task definition.
Values that are too large can produce unintended behavior:

| Excessive term | Possible result |
|---|---|
| `distance_reward` | Approaching targets matters more than completing collection. |
| `exploration_reward` | The controller wanders instead of pursuing food. |
| `collision_penalty` | The controller becomes overly cautious or nearly stationary. |
| `turn_penalty` | The controller avoids necessary sharp turns. |

The default values are a practical starting point, not a neutral or universally
optimal reward design.

---

## Default Reward Settings

The current `task.yaml` uses:

| Setting | Value | Meaning |
|---|---:|---|
| `food_reward` | `10.0` | Reward for each collected food item. |
| `distance_reward` | `0.2` | Multiplier for normalized target progress. |
| `exploration_reward` | `0.01` | Reward for entering a new grid cell. |
| `collision_penalty` | `2.0` | Penalty for wall or obstacle contact. |
| `step_penalty` | `0.005` | Cost per simulation step. |
| `turn_penalty` | `0.02` | Cost for strong steering actions. |

Example for one full-speed step directly toward food with `turn = 0.25`, no
collision, and no newly visited cell:

| Component | Reward |
|---|---:|
| distance progress | `+0.200` |
| step penalty | `-0.005` |
| turn penalty | `-0.005` |
| **total** | **`+0.190`** |

Collecting food during the same step additionally contributes `+10.0`.

EvoLib uses minimization, so accumulated episode reward is converted to fitness
with:

```python
indiv.fitness = -reward
```

A larger accumulated reward therefore becomes a smaller, better fitness value.

---

## Default Environment Settings

The current standard `task.yaml` uses:

| Setting | Value |
|---|---:|
| `width` | `800` |
| `height` | `450` |
| `max_steps` | `1500` |
| `agent_radius` | `8` |
| `base_speed` | `2.5` |
| `turn_strength` | `0.18` |
| `food_count` | `20` |
| `food_radius` | `5` |
| `collect_radius` | `12.0` |
| `obstacle_count` | `7` |
| `obstacle_min_size` | `40` |
| `obstacle_max_size` | `90` |
| `spawn_margin` | `30` |
| `terminate_on_collision` | `false` |
| `ray_length` | `60.0` |
| `ray_angles` | `[-0.6, 0.0, 0.6]` |
| `cell_size` | `32` |

---

## Rule-Based Baseline

The rule controller directly interprets the target cue and the three obstacle
rays:

```python
target_angle_sin = observation[0]
left_sensor = observation[3]
center_sensor = observation[4]
right_sensor = observation[5]
```

Its basic behavior is:

```text
turn toward the target
if an obstacle is ahead: slow down and turn toward the clearer side
if an obstacle is detected on one side: turn away from it
```

The baseline is intentionally simple. It demonstrates the observation-action
relationship and verifies that the task is solvable without providing an
optimized solution.

---

## Expected Behavior

Early evolved controllers often:

- spin in place
- remain nearly stationary
- drive repeatedly into walls
- ignore the target cue
- approach food without completing collection
- collect one nearby item by chance and then stagnate

Useful evolved controllers should eventually:

- move consistently through the arena
- steer toward the nearest food item
- react to obstacles before colliding
- choose a viable avoidance direction
- collect several food items per episode
- avoid relying only on exploration reward

---

## Suggested Reward Experiments

Collector can be used to compare how different reward definitions affect
evolution.

| Variant | Expected effect |
|---|---|
| `food_reward` only | Very sparse signal; many early controllers remain indistinguishable. |
| `food_reward + distance_reward` | Target approach becomes selectable before reliable collection emerges. |
| `food_reward + exploration_reward` | Broader movement, but potentially weak target-directed behavior. |
| no `turn_penalty` | Allows more aggressive steering and tests whether smoothness regularization is useful. |
| all default terms | Practical shaped baseline for the complete task. |

For a controlled comparison, change one reward component at a time and keep the
population settings, random seeds, and environment configuration unchanged.
Useful metrics include:

- total episode reward
- collected food count
- collision count
- visited cell count
- generations until the first successful collection

---

## Files

Package-side support files:

| File | Purpose |
|---|---|
| `evoenv/envs/collector.py` | Headless simulation, observations, rewards, and spawning. |
| `evoenv/envs/collector_objects.py` | Agent, food, obstacle, and geometry helpers. |
| `evoenv/envs/collector_config.py` | Pydantic task configuration models and validation. |
| `evoenv/envs/collector_defaults.py` | Shared runtime defaults. |
| `evoenv/envs/collector_task.py` | EvoLib controller and task integration. |
| `evoenv/renderers/pygame_collector.py` | Pygame visualization and GIF export. |

---

## Run

Manual control:

```bash
python collector_play.py
```

Rule-based controller:

```bash
python collector_rule.py
```

Train an evolved controller:

```bash
python collector_train.py
```

Train with debug visualization:

```bash
python collector_train.py --debug
```

Watch a saved checkpoint:

```bash
python collector_watch.py collector.pkl
```

---

