"""Reading the name of a capped molecule that is turned into a substituent prefix.

A fragment with a free valence is named as a molecule after its free valence is capped with
hydrogen. ``parent_to_prefix`` then edits that name into a prefix, and an edit of a name can
keep only what the name itself proves. The functions here decide what it proves.
"""

from typing import Dict, Optional, Tuple

#: chain stem -> carbon count, the whole range of ``get_chain_prefix`` (1-9999),
#: built on first use (``chain_stem_ending``).
_CHAIN_STEM_CARBONS: Optional[Dict[str, int]] = None


def chain_stem_ending(stem: str) -> Optional[Tuple[str, int]]:
    """The longest ending of ``stem`` that is a chain stem of
    (``get_chain_prefix``), as ``(text before it, carbon count)``; None when no
    ending is one.

    Longest first, so a numerical term belongs to its stem: 'undec' -> ('', 11),
    'dodec' -> ('', 12) 'undeca', 'dodeca'), and 'methoxymeth' ->
    ('methoxy', 1) rather than 'methoxym' on 'eth'; '2-methylprop' -> ('2-methyl', 3).
    """
    global _CHAIN_STEM_CARBONS
    if _CHAIN_STEM_CARBONS is None:
        from ..data.chain_names import get_chain_prefix
        _CHAIN_STEM_CARBONS = {get_chain_prefix(n): n for n in range(1, 10000)}
    for cut in range(len(stem)):
        n = _CHAIN_STEM_CARBONS.get(stem[cut:])
        if n is not None:
            return stem[:cut], n
    return None


def carries_capped_locants(name: str) -> bool:
    """True when the capped molecule's name cites a locant of its own chain: a digit outside
    every pair of enclosing marks. A locant inside marks belongs to a nested prefix that is
    numbered, and complete, on its own ('(4-methylphenyl)diazene').

     Substituent groups derived from parent hydrides with 'ene' or 'yne' endings,
    (the Blue Book): "for acyclic parent structures, all locants for the free valences,
    including '1', must be cited"; The principal substituent chain, (:22718): the
    free valence takes the lowest locant of the SUBSTITUENT's own numbering. The name of the
    capped molecule numbers the chain for its own principal characteristic group or its own
    multiple bonds and never numbers the free valence, so a name that cites a locant of its chain
    cannot be turned into a prefix by editing its ending: the free-valence locant is absent, and
    the locants that are present may run from the other end of the chain."""
    depth = 0
    for ch in name:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth = max(0, depth - 1)
        elif depth == 0 and ch.isdigit():
            return True
    return False


def terminal_yl_prefix(capped_name: str, candidate: str) -> Optional[str]:
    """``candidate``, the capped molecule's name ended in 'yl', or None when the name does not
    prove the places of the prefix.

    The capped name cites no locant of its chain, and for a two-carbon chain that is the only
    chain length at which a substituent prefix on 'ethene' or 'ethyne' needs none. A substituent
    X and the free valence stand on the same carbon or on the two carbons of 'ethene' (CH2=C(X)-
    and XCH=CH-, two isomers:, the Blue Book, '2-chloroethen-1-yl (preferred
    prefix)'), so the free-valence locant and the locant of X are cited and the capped name has
    no way to say them. 'ethyne' has one hydrogen on each carbon, so X and the free valence are
    on different carbons and no isomer exists,:2953; 'oxoethenyl (preferred
    prefix)',:2983): the prefix cites no locant and, being a compound prefix, takes enclosing
    marks,:7232). A longer chain with a substituent in front of its stem is declined
    for the same reason as 'ethene'.

    ``candidate`` is returned unchanged for a name that is not a substituted chain."""
    if carries_capped_locants(capped_name):
        return None
    if not capped_name.endswith(('ene', 'yne')):
        return candidate
    found = chain_stem_ending(capped_name[:-3])
    if found is None or not found[0]:
        return candidate
    if found[1] == 2 and capped_name.endswith('yne'):
        return candidate
    return None
