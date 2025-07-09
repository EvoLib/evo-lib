# SPDX-License-Identifier: MIT
from evolib.config.schema import FullConfig


def validate_full_config(cfg: dict) -> FullConfig:
    """
    Validates and parses a configuration dictionary using pydantic.

    Returns:
        FullConfig: A validated and typed config object.
    Raises:
        pydantic.ValidationError: If the config is invalid.
    """
    return FullConfig(**cfg)
