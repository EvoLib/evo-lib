# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from evolib.config.schema import FullConfig

# EvoNet presets
from evolib.initializers.evonet_initializers import (
    initializer_default_evonet,
    initializer_identity_evonet,
    initializer_unconnected_evonet,
)

# Vector initializers
from evolib.initializers.vector_initializers import (
    initializer_adaptive_vector,
    initializer_fixed_vector,
    initializer_normal_vector,
    initializer_random_vector,
    initializer_zero_vector,
)
from evolib.initializers.vectornet_initializers import initializer_normal_vectornet
from evolib.interfaces.enums import RepresentationType
from evolib.interfaces.types import ModuleConfig
from evolib.representation.base import ParaBase
from evolib.representation.composite import ParaComposite

InitializerFunction = Callable[[FullConfig, str], ParaBase]


def _resolve_vector_initializer(name: str) -> InitializerFunction:
    """Resolve Vector initializer by name."""
    name = str(name)

    match name:
        case "normal":
            return initializer_normal_vector
        case "uniform":
            return initializer_random_vector
        case "zero":
            return initializer_zero_vector
        case "fixed":
            return initializer_fixed_vector
        case "adaptive":
            return initializer_adaptive_vector
        case _:
            raise ValueError(
                f"Unknown vector initializer '{name}'. "
                "Allowed: normal, uniform, zero, fixed, adaptive."
            )


def _resolve_vectornet_initializer(name: str) -> InitializerFunction:
    """Resolve VectorNet initializer by name."""
    if name == "normal":
        return initializer_normal_vectornet
    raise ValueError(f"Unknown vectornet initializer '{name}'. Allowed: normal.")


def _resolve_evonet_initializer(name: str) -> InitializerFunction:
    """Resolve EvoNet topology presets via initializer name."""
    name = str(name)

    match name:
        case "default":
            return initializer_default_evonet
        case "unconnected":
            return initializer_unconnected_evonet
        case "identity":
            return initializer_identity_evonet
        case _:
            raise ValueError(
                f"Unknown evonet initializer '{name}'. "
                "Allowed: default, unconnected, identity."
            )


def resolve_initializer_fn(cfg: ModuleConfig) -> InitializerFunction:
    """Resolve initializer function based on module type."""
    mod_type = getattr(cfg, "type", None)
    init_name = getattr(cfg, "initializer", None)
    if init_name is None:
        raise ValueError("module.initializer must be set")

    match mod_type:
        case RepresentationType.VECTOR:
            return _resolve_vector_initializer(str(init_name))
        case RepresentationType.VECTORNET:
            return _resolve_vectornet_initializer(str(init_name))
        case RepresentationType.EVONET:
            return _resolve_evonet_initializer(str(init_name))
        case _:
            raise ValueError(f"Unsupported module type: {mod_type!r}")


def build_composite_initializer(config: FullConfig) -> Callable[[Any], ParaBase]:
    """Build a composite initializer used by Pop to create Para instances for each
    module."""

    def _init(_: Any) -> ParaBase:
        modules: dict[str, ParaBase] = {}
        for module_name, module_cfg in config.modules.items():
            fn = resolve_initializer_fn(module_cfg)
            modules[module_name] = fn(config, module_name)
        return ParaComposite(modules)

    return _init
