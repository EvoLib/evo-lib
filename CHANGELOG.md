## [0.2.0b2] - unreleased

### Added
- Added pred_lw, true_lw, pred_ls, and true_ls parameters to plot_approximation() for customizing line width and style.
- Added automatic aging of individuals: age_indivs() is now called at the beginning of each generation in all strategies.
- `mutate_activations`: optional `layers` parameter for finer control.

### Changed
- Internal refactor of activation mutation logic (no change to defaults).

### Fixed
