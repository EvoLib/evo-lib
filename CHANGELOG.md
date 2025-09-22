## [0.2.0b2] - unreleased

### Added
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
- Unified individual copy handling: all strategies now use `Indiv.copy()`
  instead of `deepcopy` or `copy_indiv`.
- Added safeguard in remove_old_indivs(): if all individuals exceed max_indiv_age, the best individual is retained to prevent population collapse.
- Evolution strategies now respect max_indiv_age: individuals exceeding the configured age are removed after replacement (not applicable for (mu, lambda)).
- Internal refactor of activation mutation logic (no change to defaults).

### Fixed
- `update_mutation_parameters`: Parents now receive adaptive parameters only once during population initialization.
  In all subsequent generations, only offspring parameters are updated.
- Fixed `selection_random` removal logic (now removes original parent).
- `selection_rank_based` now respects the `exp_base` parameter.
- Fixed adaptive update of mutation strength to consistently use min/max bounds instead of weight/bias value ranges.
