import numpy as np

from evolib.core.individual import Indiv
from evolib.operators.mutation import mutate_gauss


def test_mutate_gauss_changes_values() -> None:
    indiv = Indiv(para=np.zeros(3))
    before = indiv.para.copy()
    mutated = mutate_gauss(indiv.para, mutation_strength=0.1, bounds=(-1, 1))
    assert not np.array_equal(mutated, before)
