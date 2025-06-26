## 03_analysis/ — Logging, Plotting, and Comparison

This folder introduces EvoLib's tools for analysis and visualization:

- Tracking fitness metrics over generations
- Plotting fitness history
- Comparing multiple evolutionary runs visually

### Files
- `01_history.py`: Logs per-generation fitness metrics
- `02_plotting.py`: Uses `plot_fitness()` for visualization
- `03_compare_runs.py`: Compares static mutation rates visually
- `04_exponential_decay_vs_static.py`: Compares decay vs. constant mutation
- `05_adaptive_global_vs_static.py`: Compares adaptive vs. constant mutation

### Output
- Visualizations are saved as `.png` under `figures/`

### Usage
These examples are intended to build on earlier examples by providing visual tools for analyzing performance across strategies
