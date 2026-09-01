# SPDX-License-Identifier: MIT
"""
EvoNetComponentConfig defines the EvoLib-specific extension of EvoNet configuration.

Network structure and initialization are provided by ``evonet.EvoNetConfig``. This class
adds only the evolutionary operators required by EvoLib.
"""

from typing import Optional

from evonet import EvoNetConfig
from pydantic import Field

from evolib.config.base_component_config import (
    CrossoverConfig,
    EvoNetMutationConfig,
    StructuralMutationConfig,
)
from evolib.interfaces.enums import RepresentationType


class EvoNetComponentConfig(EvoNetConfig):
    """
    EvoLib configuration for an evolvable EvoNet module.

    Network structure and initialization fields are inherited from ``EvoNetConfig``.
    EvoLib adds the module discriminator and evolutionary operator configuration.
    """

    # Module type is fixed to "evonet"
    type: RepresentationType = Field(
        default=RepresentationType.EVONET,
        description='Fixed module discriminator; must be "evonet" for this schema.',
    )

    # Evolutionary operators
    mutation: Optional[EvoNetMutationConfig] = None
    crossover: Optional[CrossoverConfig] = None
    structural: Optional[StructuralMutationConfig] = None
