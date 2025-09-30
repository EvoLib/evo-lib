# SPDX-License-Identifier: MIT
"""
Parallelization helpers for EvoLib.

Currently supports optional Ray integration for fitness evaluation. Falls back to
sequential evaluation if Ray is not available or not requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from evolib.core.individual import Indiv

if TYPE_CHECKING:
    import ray
else:
    try:
        import ray
    except ImportError:
        ray = None  # type: ignore[assignment]

# Global flag to avoid repeated init/shutdown
_ray_initialized = False


def _ensure_ray_initialized(
    num_cpus: Optional[int] = None, address: Optional[str] = None
) -> None:
    """Initialize Ray once with given resources."""
    global _ray_initialized

    if ray is None:
        raise RuntimeError("Ray is not installed. Install with `pip install ray`.")

    if not _ray_initialized:
        init_kwargs: dict[str, Any] = {}
        if num_cpus is not None:
            init_kwargs["num_cpus"] = num_cpus
        if address is not None:
            init_kwargs["address"] = address
        ray.init(**init_kwargs)
        _ray_initialized = True


def shutdown_ray() -> None:
    """Shutdown Ray if it was started by EvoLib."""
    global _ray_initialized
    if ray is not None and _ray_initialized:
        ray.shutdown()
        _ray_initialized = False


def map_fitness(
    indivs: list[Indiv],
    fitness_fn: Callable[[Indiv], float | None],
    *,
    backend: str = "none",
    num_cpus: Optional[int] = None,
    address: Optional[str] = None,
) -> None:
    """
    Evaluate a list of individuals using the chosen backend.

    Args:
        indivs: List of individuals to evaluate (modified in-place).
        fitness_fn: Fitness function that assigns indiv.fitness.
        backend: "none" (sequential) or "ray".
        num_cpus: Number of CPUs for Ray (optional).
        address: Ray cluster address (optional).
    """

    if backend != "ray":
        # Sequential fallback
        for indiv in indivs:
            result = fitness_fn(indiv)
            if result is not None:
                indiv.fitness = float(result)
            indiv.is_evaluated = True
        return

    # Ray backend
    _ensure_ray_initialized(num_cpus=num_cpus, address=address)

    @ray.remote
    def _eval_remote(indiv: Indiv) -> Indiv:
        result = fitness_fn(indiv)
        if result is not None:
            indiv.fitness = float(result)
        indiv.is_evaluated = True
        return indiv

    futures = [_eval_remote.remote(ind) for ind in indivs]
    results = ray.get(futures)

    # Update original objects in-place
    for orig, res in zip(indivs, results):
        orig.fitness = res.fitness
        orig.extra_metrics = getattr(res, "extra_metrics", {})
        orig.is_evaluated = True
