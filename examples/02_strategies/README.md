## 02_strategies/ — Evolution Strategies and Adaptive Mutation

This folder demonstrates core evolutionary strategies:

- (μ, λ) and (μ + λ) Evolution Strategies
- Step-by-step evolution to show internal structure
- Adaptive global mutation and mutation decay

### Files
- `01_step_by_step_evolution.py`: Manually constructs evolution loop
- `02_mu_lambda_step.py`: Applies (μ + λ) strategy using fixed parameters
- `03_mu_lambda.py`: Full (μ + λ) execution with default config
- `04_exponential_decay.py`: Uses exponentially decaying mutation strength
- `05_adaptive_global.py`: Uses global adaptive mutation rate (no plots)

### Notes
- Results are printed to console
- Visual comparison for both `04` and `05` is located in `03_analysis/`
