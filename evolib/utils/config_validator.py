# SPDX-License-Identifier: MIT
def validate_mutation_config(cfg: dict, strategy: str) -> None:
    required_fields = {
        "constant": ["probability", "strength"],
        "exponential_decay": [
            "min_probability",
            "max_probability",
            "min_strength",
            "max_strength",
        ],
        "adaptive_global": [
            "init_probability",
            "min_probability",
            "max_probability",
            "init_strength",
            "min_strength",
            "max_strength",
            "min_diversity_threshold",
            "max_diversity_threshold",
            "increase_factor",
            "decrease_factor",
        ],
        "adaptive_individual": [
            "min_probability",
            "max_probability",
            "min_strength",
            "max_strength",
        ],
        "adaptive_per_parameter": ["min_strength", "max_strength"],
    }

    if strategy not in required_fields:
        raise ValueError(f"Unknown mutation strategy: '{strategy}'")

    missing = [k for k in required_fields[strategy] if k not in cfg]
    if missing:
        raise ValueError(f"Missing mutation config keys for '{strategy}': {missing}")


def validate_crossover_config(cfg: dict, strategy: str) -> None:
    required_fields = {
        "constant": [],
        "exponential_decay": ["min_probability", "max_probability"],
        "adaptive_global": ["init_probability", "min_probability", "max_probability"],
        "adaptive_individual": ["min_probability", "max_probability"],
    }

    if strategy not in required_fields:
        raise ValueError(f"Unknown crossover strategy: '{strategy}'")

    missing = [k for k in required_fields[strategy] if k not in cfg]
    if missing:
        raise ValueError(f"Missing crossover config keys for '{strategy}': {missing}")


def validate_selection_config(cfg: dict) -> None:
    method = cfg.get("method")
    if method is None:
        raise ValueError("Selection method not specified.")

    if method == "tournament":
        if "tournament_size" not in cfg:
            raise ValueError("Missing 'tournament_size' for tournament selection.")
    elif method in ("roulette", "sus", "truncation", "boltzmann"):
        pass  # No required parameters
    else:
        raise ValueError(f"Unknown selection method: '{method}'")


def validate_replacement_config(cfg: dict) -> None:
    method = cfg.get("method")
    if method is None:
        raise ValueError("Replacement method not specified.")

    valid_methods = ("elitism", "truncation", "generational")
    if method not in valid_methods:
        raise ValueError(f"Unknown replacement method: '{method}'")


def validate_full_config(config: dict) -> None:
    """
    Validates the full configuration dictionary for consistency and completeness.

    Raises:
        ValueError if any required fields are missing or invalid.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary.")

    # Validate population_size
    pop_size = config.get("parent_pool_size")
    if not isinstance(pop_size, int) or pop_size <= 0:
        raise ValueError("Invalid or missing 'population_size'.")

    # Mutation
    mutation_cfg = config.get("mutation")
    if not isinstance(mutation_cfg, dict):
        raise ValueError("Missing or invalid 'mutation' section.")

    strategy = mutation_cfg.get("strategy")
    if not isinstance(strategy, str):
        raise ValueError("Missing or invalid mutation strategy.")

    validate_mutation_config(mutation_cfg, strategy)

    # Optional: Crossover
    crossover_cfg = config.get("crossover", None)
    if crossover_cfg is not None:
        strategy = crossover_cfg.get("strategy", "constant")
        validate_crossover_config(crossover_cfg, strategy)


#    # Selection
#    selection_cfg = config.get("selection")
#    if not isinstance(selection_cfg, dict):
#        raise ValueError("Missing or invalid 'selection' section.")
#
#    validate_selection_config(selection_cfg)
#
#    # Replacement
#    replacement_cfg = config.get("replacement")
#    if not isinstance(replacement_cfg, dict):
#        raise ValueError("Missing or invalid 'replacement' section.")
#
#    validate_replacement_config(replacement_cfg)
#
#    # Optional: Warn if unused sections are present
#    for section in config.keys():
#        if section not in {"population_size", "mutation", "selection", "replacement"}:
#            warn(f"Unknown section '{section}' in config – ignored.")
