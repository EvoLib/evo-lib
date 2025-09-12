Public API
==========

EvoLib provides a flat import surface through ``evolib/api.py``.
Instead of navigating deep into subpackages, you can import the most relevant classes, functions, and utilities directly:

.. code-block:: python

    from evolib import Pop, Indiv, Vector, EvoNet, NetVector
    from evolib import plot_fitness, rastrigin, mse_loss
    from evolib import save_checkpoint, resume_from_checkpoint

Core Classes
------------

- ``Pop`` (``evolib.core.population.Population``)
- ``Indiv`` (``evolib.core.individual.Indiv``)
- ``Vector`` (``evolib.representation.vector.Vector``)
- ``EvoNet`` (``evolib.representation.evonet.EvoNet``)
- ``NetVector`` (``evolib.representation.netvector.NetVector``)
- ``HistoryLogger`` (``evolib.utils.history_logger.HistoryLogger``)

Fitness Functions
-----------------

- Type alias: ``evolib.interfaces.types.FitnessFunction``

Benchmark Functions:
- ``sphere`` (``sphere_2d``, ``sphere_3d``)
- ``rastrigin`` (``rastrigin_2d``, ``rastrigin_3d``)
- ``ackley`` (``ackley_2d``, ``ackley_3d``)
- ``rosenbrock`` (``rosenbrock_2d``, ``rosenbrock_3d``)
- ``griewank`` (``griewank_2d``, ``griewank_3d``)
- ``schwefel`` (``schwefel_2d``, ``schwefel_3d``)
- ``simple_quadratic``

Loss Functions
--------------

- ``mse_loss`` • ``mae_loss`` • ``huber_loss`` • ``bce_loss`` • ``cce_loss``

Plotting Utilities
------------------

- ``plot_fitness`` • ``plot_approximation`` • ``plot_history`` • ``plot_diversity``
- ``plot_mutation_trends`` • ``plot_fitness_comparison`` • ``save_combined_net_plot``

Checkpointing
-------------

- ``save_checkpoint`` • ``resume_from_checkpoint`` • ``resume_or_create``
- ``save_best_indiv`` • ``load_best_indiv``

Notes
-----

- This page mirrors what is re-exported in ``evolib/api.py`` (``__all__``).
- For detailed docs, see the dedicated pages (e.g., ``api_core_individual``).
- Importing via the public API is the recommended way for end users.
