# EvoEnv

EvoEnv provides small, controllable Pygame environments for focused
evolutionary experiments with EvoLib.

The project is intended for tasks where observations, actions, rewards, and
controller behavior should remain visible and easy to inspect. It is not a
general reinforcement-learning benchmark suite and does not target large or
high-performance game environments.

The main goals are:

- simple environments with a clear experimental purpose
- compact observation and action spaces
- reproducible headless evaluation
- optional Pygame visualization
- short iteration cycles
- consistent structure across environments
- straightforward integration with EvoLib controllers

For runnable examples, see
[`examples/09_evoenv/`](../examples/09_evoenv/).

---

## Scope

EvoEnv is designed for small controller tasks such as:

- steering from local sensor input
- action timing
- obstacle avoidance
- target following
- reward-shaping experiments
- recurrent or memory-dependent behavior
- small multi-agent experiments

The environments should remain small enough that their complete behavior can be
understood from the implementation and observed directly in the renderer.

---

## Architecture

EvoEnv separates simulation, control, integration, and visualization.

| Component | Responsibility |
|---|---|
| `Env` | Simulation state, observations, rewards, and episode termination |
| `Controller` | Mapping observations to actions |
| `Task` | Connecting EvoLib individuals with environments |
| `Renderer` | Pygame drawing, overlays, and debug visualization |
| `Checkpoint` | Persisting trained individuals and environment metadata |

### Environment

An environment contains the complete headless simulation.

```python
class Env:
    def reset(self, seed: int | None = None): ...
    def step(self, action): ...
```

An environment is responsible for:

- initializing episode state
- applying actions
- advancing the simulation
- producing observations
- calculating rewards
- reporting termination and diagnostic information

Environment code must not depend on an active Pygame window. This allows
training and evaluation to run without rendering.

### Controller

A controller maps one observation to one action.

```python
class Controller:
    def act(self, observation): ...
```

Controllers may be:

- manually operated
- rule based
- backed by an EvoLib individual
- implemented specifically for testing

Keeping the controller interface small makes rule-based and evolved behavior
directly comparable.

### Task

A task is the integration layer between EvoEnv and EvoLib.

A task typically:

- loads and stores the task configuration
- creates fresh environment instances
- creates controllers for EvoLib individuals
- evaluates individuals over complete episodes
- exposes optional visualization helpers
- reconstructs environments from checkpoint metadata

Example scripts should use the task instead of reconstructing environment
parameters independently.

### Renderer

A renderer visualizes environment state without owning simulation logic.

Renderer responsibilities include:

- drawing agents and objects
- drawing sensors
- displaying episode statistics
- handling debug visualization
- optionally exporting GIFs

The renderer reads state from the environment. It does not calculate rewards or
change environment behavior.

### Checkpoints

A checkpoint stores the trained individual together with the information needed
to reconstruct its task.

Typical checkpoint metadata includes:

- environment name
- task configuration
- controller module name
- random seed

This allows training and visualization to remain separate.

---

## Package Layout

```text
evoenv/
├── core/
│   ├── checkpoint.py
│   ├── config.py
│   ├── controller.py
│   ├── env.py
│   ├── sensors.py
│   ├── task.py
│   └── task_registry.py
├── envs/
│   ├── <name>.py
│   ├── <name>_config.py
│   ├── <name>_defaults.py
│   ├── <name>_objects.py
│   └── <name>_task.py
└── renderers/
    ├── pygame_common.py
    └── pygame_<name>.py
```

The standard environment layout is intentionally complete and consistent.
Individual environments may not require every file technically, but keeping the
same structure makes examples easier to navigate, compare, and extend.

### Environment files

| File | Purpose |
|---|---|
| `<name>.py` | Headless simulation and reward logic |
| `<name>_objects.py` | Environment-specific state objects and geometry helpers |
| `<name>_config.py` | Typed YAML configuration models |
| `<name>_defaults.py` | Shared runtime defaults used by package and examples |
| `<name>_task.py` | EvoLib integration and checkpoint task loader |
| `pygame_<name>.py` | Pygame rendering and debug episodes |

---

## Configuration

Environment parameters should be represented by typed configuration models and
loaded from YAML.

Configuration belongs in dedicated config models rather than being distributed
across environment, task, and example scripts.

A task configuration commonly separates:

```yaml
env:
  # simulation parameters

reward:
  # rewards and penalties

sensor:
  # sensor geometry

exploration:
  # optional task-specific settings
```

Cross-field constraints should be validated when the configuration is loaded.
Invalid configurations should fail explicitly instead of silently changing the
task.

---

## Reproducibility

`reset(seed=...)` should reproduce the complete initial episode state,
including:

- agent state
- object placement
- obstacle placement
- task-specific random choices

Training tasks should use the configured EvoLib random seed where applicable.
The seed and complete task configuration should be stored in checkpoints.

---

## Example Workflow

Runnable examples use the same four-step workflow:

```text
play -> rule -> train -> watch
```

| Script | Purpose |
|---|---|
| `*_play.py` | Inspect the environment through manual control |
| `*_rule.py` | Verify solvability with a simple rule-based controller |
| `*_train.py` | Evolve a controller headlessly, with optional debug rendering |
| `*_watch.py` | Load and visualize a saved checkpoint |

The common workflow is documented in the
[example overview](../examples/09_evoenv/).

---

## Adding an Environment

A new environment should normally be added in this order:

1. Define the task and its compact observation and action spaces.
2. Implement the headless environment and environment-specific objects.
3. Add strict typed configuration models.
4. Add a task class that creates environments and controllers.
5. Add a renderer that only reads environment state.
6. Add `play`, `rule`, `train`, and `watch` example scripts.
7. Add deterministic tests for reset, stepping, rewards, collisions, and
   checkpoint reconstruction.
8. Add a focused example README and preview GIF.

Prefer a provisional but documented implementation over premature abstraction.
Shared code should be extracted only after the same concept appears in multiple
environments or clearly belongs to the core API.

---

## Design Constraints

New environments should generally preserve these constraints:

- small observation and action spaces
- transparent reward calculation
- no rendering dependency during evaluation
- explicit termination conditions
- deterministic seeded resets
- typed configuration
- flat and understandable user-facing APIs
- no environment-specific logic in generic render loops
- no training logic inside the environment

Large visual policies, raw-pixel inputs, large CNNs, and large recurrent
controllers are outside the intended scope.

---

## Documentation

Each environment should provide:

- a concise description of its purpose
- observation and action tables
- reward calculation
- default configuration
- expected rule-based and evolved behavior
- commands for `play`, `rule`, `train`, and `watch`

Public APIs added to `evoenv.core` should also be included in the Sphinx API
documentation.
