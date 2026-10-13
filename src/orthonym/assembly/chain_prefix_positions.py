"""The substituent positions that the prefixes of a chain parent cite.

The locant of a lone substituent on ethane is omitted (b), the Blue Book,
'ethanol'). Citation of locants (:2869) cites all locants of a unit when any is
essential, so a second prefix on the same chain makes the locant of the first essential. The
prefixes of a chain come from more than one writer (the branch namer, the functional-group
prefix loop, the ring-prefix generator), and the branch namer sees only its own.
"""
from typing import Any


def chain_prefix_positions(features: Any, branch_namer_positions: int) -> int:
    """The number of substituent branches on the principal chain of ``features``, at least the
    ``branch_namer_positions`` the branch namer itself cites. ``features.substituents`` holds
    every branch of the chain by its locant, whoever writes its prefix."""
    branches = sum(len(groups) for groups in (features.substituents or {}).values())
    return max(branch_namer_positions, branches)
