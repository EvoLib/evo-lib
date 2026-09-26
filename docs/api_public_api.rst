Public API
==========

EvoLib provides a flat public API for the classes, functions, and utilities
commonly used by applications and examples.

.. code-block:: python

    from evolib import Population, Individual, Vector, VectorNet, EvoNet
    from evolib import plot_fitness, rastrigin, mse_loss
    from evolib import save_checkpoint, resume_from_checkpoint

``Pop`` and ``Indiv`` are available as aliases for ``Population`` and
``Individual``.

Core Classes
------------

- ``Population``
- ``Individual``
- ``Vector``
- ``EvoNet``
- ``VectorNet``
- ``HistoryLogger``

Environment Integration
-----------------------

- ``GymEnv``

Fitness Functions
-----------------

- ``FitnessFunction``

Benchmark functions:

- ``sphere`` (``sphere_2d``, ``sphere_3d``)
- ``rastrigin`` (``rastrigin_2d``, ``rastrigin_3d``)
- ``ackley`` (``ackley_2d``, ``ackley_3d``)
- ``rosenbrock`` (``rosenbrock_2d``, ``rosenbrock_3d``)
- ``griewank`` (``griewank_2d``, ``griewank_3d``)
- ``schwefel`` (``schwefel_2d``, ``schwefel_3d``)
- ``simple_quadratic``
- ``lfsr_sequence``
- ``generate_timeseries``

Loss Functions
--------------

- ``mse_loss``
- ``mae_loss``
- ``huber_loss``
- ``bce_loss``
- ``cce_loss``

Plotting Utilities
------------------

- ``plot_fitness``
- ``plot_approximation``
- ``plot_bit_prediction``
- ``plot_history``
- ``plot_diversity``
- ``plot_mutation_trends``
- ``plot_fitness_comparison``
- ``save_combined_net_plot``

Checkpointing
-------------

- ``save_checkpoint``
- ``resume_from_checkpoint``
- ``resume_or_create``
- ``save_best_indiv``
- ``load_best_indiv``

Notes
-----

The names listed above are re-exported through the public API in
``evolib/api.py``. ``Pop`` and ``Indiv`` are additional aliases for
``Population`` and ``Individual``.

Importing from ``evolib`` is the recommended interface for end users.
