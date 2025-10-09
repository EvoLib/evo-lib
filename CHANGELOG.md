## EvoLib 0.2.0b3dev6 [unreleased]

### Added
- **HELI (Hierarchical Evolution with Lineage Incubation):**
  New operator enabling short local micro-evolutions for structure-mutated
  individuals. Each mutated topology spawns a temporary subpopulation that
  evolves for a few generations before reintegration into the main pool.

  **Highlights**
  - Configurable via `evolution.heli` in YAML configs  
  - Parameters: `generations`, `offspring_per_seed`, `max_fraction`, `reduce_sigma_factor`  
  - Uses fixed `(μ + λ)` strategy per subpopulation  
  - Compatible with `EvoNet` and structural mutation logic  
  - Verbosity control via `pop.heli_verbosity`

  Example YAML snippet:
  ```yaml
  evolution:
    strategy: mu_plus_lambda
    heli:
      generations: 5
      offspring_per_seed: 10
      max_fraction: 0.1
      reduce_sigma_factor: 0.5
   ```

- Added new examples: LunarLander, CartPole, CliffWalking, FrozenLake
- `GymEnv` now accepts additional `**env_kwargs` to forward environment-specific
  parameters directly to `gym.make`.
- `evaluate()` in `GymEnv` supports an optional `episodes` argument to average
  fitness over multiple runs for more stable evaluation in stochastic environments.
- Added `GymEnvWrapper` in `evolib.envs` to evaluate Individuals in Gymnasium environments.
  Includes `.evaluate()` for fitness calculation and `.visualize()` to render episodes as GIFs.

### Changed

### Fixed


## EvoLib 0.2.0b2 (2025-10-03)

### Added
- Basic Ray support for parallel fitness evaluation.
  - New `parallel:` block in YAML configs (`backend: none|ray`, `num_cpus`, `address`).
  - Population transparently uses Ray for fitness evaluation when configured.
  - Default behavior remains sequential if no parallel section is provided.
- New example: `05_recurrent_timeseries`, `06_recurrent_trading.py` 
- New EvoNet initializer `identity_evonet`: sets self-recurrent connections to 0.8 and initializes other weights near zero. Encourages internal memory dynamics from the start.
- Add new example recurrent_bit_prediction for EvoNet bit sequence forecasting
- Add lfsr_sequence, xor_sequence, random_fixed_sequence to benchmarks
- Added normal_evonet, random_evonet, and zero_evonet.
- Each individual now has a unique `indiv.id` (UUID) for tracking and debugging.
- `Indiv.copy()` extended with `reset_*` flags (`reset_id`, `reset_fitness`,
  `reset_age`, `reset_evaluation`, `reset_origin`) for flexible cloning.
  By default, `reset_id=True` ensures every copy gets a new unique ID.
- Added: `recurrent` option in EvoNet module config.
  Allows selecting initial recurrent connections via presets (`none`, `direct`, `local`, `all`).
- Added: `activations_allowed` option for EvoNet modules.
  This restricts the pool of activation functions when using `activation: random`.
- Added `plot_bit_prediction` in plotting.py: specialized visualization for bit sequence prediction, combining raster (Input/Target/Pred) and line/scatter view.
- Added `Population.step()` as a shorthand for `run_one_generation()`.
- Added pred_lw, true_lw, pred_ls, and true_ls parameters to plot_approximation() for customizing line width and style.
- Added automatic aging of individuals: age_indivs() is now called at the beginning of each generation in all strategies.
- `mutate_activations`: optional `layers` parameter for finer control.

### Changed
- Config validation tightened:
  - All config models now use `extra="forbid"` (typos/unexpected fields raise errors).
- rename print_graph() -> plot()
- Changed Pop.best() to sort the population by default (sort=True) to ensure consistent retrieval of the best individual.
- Add DummyPara as placeholder for uninitialized individuals
- `crossover_offspring`: now delegates crossover to `para.crossover_with`
- Unified individual copy handling: all strategies now use `Indiv.copy()`
  instead of `deepcopy` or `copy_indiv`.
- Added safeguard in remove_old_indivs(): if all individuals exceed max_indiv_age, the best individual is retained to prevent population collapse.
- Evolution strategies now respect max_indiv_age: individuals exceeding the configured age are removed after replacement (not applicable for (mu, lambda)).
- Internal refactor of activation mutation logic (no change to defaults).

### Fixed
- Fixed resume_or_create() to reload random_seed from config when resuming a population, ensuring deterministic behavior across restarts.
- `update_mutation_parameters`: Parents now receive adaptive parameters only once during population initialization.
  In all subsequent generations, only offspring parameters are updated.
- Fixed `selection_random` removal logic (now removes original parent).
- `selection_rank_based` now respects the `exp_base` parameter.
- Fixed adaptive update of mutation strength to consistently use min/max bounds instead of weight/bias value ranges.
