# SPDX-License-Identifier: MIT

from evolib.core.individual import Indiv


def copy_indiv(indiv: Indiv) -> Indiv:
    """
    Universal copy function for individuals.

    Delegates to the individual's own `copy()` method, which can be
    customized per subclass. This provides a consistent interface for
    copying individuals across the library.

    Args:
        indiv (Indiv): The individual to be copied.

    Returns:
        Indiv: A deep or custom copy of the individual.
    """
    return indiv.copy()
