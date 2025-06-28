import numpy as np

from evolib.interface.structs import MutationParams
from evolib.operators.mutation import mutate_gauss
from evolib.representation.base import ParaBase


class ParaVector(ParaBase):
    def __init__(self, vector: np.ndarray, tau: float = 0.0):
        self.vector = vector
        self.tau = tau  # falls benötigt

    def mutate(self, params: MutationParams) -> None:
        self.vector = mutate_gauss(self.vector, params.strength, params.bounds)

    def copy(self) -> "ParaVector":
        return ParaVector(self.vector.copy(), self.tau)
