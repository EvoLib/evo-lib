# SPDX-License-Identifier: MIT

import numpy as np

from evolib.core.individual import Indiv


def default_update_tau(indiv: Indiv) -> None:
    """Standard tau update: 1 / sqrt(len(para))"""
    if indiv.para is not None and hasattr(indiv.para, "__len__"):
        n = len(indiv.para)
        indiv.tau = 1.0 / np.sqrt(n) if n > 0 else 0.0
    else:
        indiv.tau = 0.0


# def tau_for_neural_net(indiv: Indiv):
#    weights = extract_weights_from_net(indiv.para)
#    n = len(weights)
#    indiv.tau = 1.0 / np.sqrt(n)
