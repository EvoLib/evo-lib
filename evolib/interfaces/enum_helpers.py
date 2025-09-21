# SPDX-License-Identifier: MIT

from typing import Optional, Set

from evonet.enums import RecurrentKind


def resolve_recurrent_kinds(preset: Optional[str]) -> Set[RecurrentKind]:
    """
    Map a preset string to the corresponding set of recurrent connection types.

    Parameters
    ----------
    preset : Optional[str]
        One of: "none", "direct", "local", "all", or None.

    Returns
    -------
    Set[RecurrentKind]
        A set of recurrent kinds to use.
    """
    if preset is None or preset == "none":
        return set()
    elif preset == "direct":
        return {RecurrentKind.DIRECT}
    elif preset == "local":
        return {RecurrentKind.DIRECT, RecurrentKind.LATERAL}
    elif preset == "all":
        return {RecurrentKind.DIRECT, RecurrentKind.LATERAL, RecurrentKind.INDIRECT}
    else:
        raise ValueError(f"Unknown recurrent preset: {preset}")
